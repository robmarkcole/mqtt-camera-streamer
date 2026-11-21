"""Benchmark the real local inference pipeline without MQTT or rate-limit sleeps."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from onnx_camera import (
    PredictionProcessor,
    create_predictor,
    load_config,
    prediction_stream,
)
from plates import PlateProcessor


def summarize(samples):
    """Summarize elapsed milliseconds without extrapolating from model-only time."""
    if not samples:
        return None
    return {
        "samples": len(samples),
        "mean_ms": round(float(np.mean(samples)), 3),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def source_description(source):
    """Do not copy camera URLs or embedded credentials into shareable reports."""
    if isinstance(source, str) and "://" in source:
        return source.split(":", 1)[0] + "://<redacted>"
    return source


def pi_health():
    """Read optional Linux telemetry; record unavailable values as null."""
    result = {"temperature_c": None, "throttle_hex": None}
    try:
        result["temperature_c"] = (
            float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        )
    except (OSError, ValueError):
        pass
    try:
        command = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if command.returncode == 0 and command.stdout.strip().startswith(
            "throttled=0x"
        ):
            result["throttle_hex"] = command.stdout.strip().split("=", 1)[1]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def benchmark(config, frames=100, warmup=5, lpr_mode="configured"):
    """Measure capture+detector, tracking, optional ALPR, and whole frame latency."""
    if lpr_mode == "off":
        config = config.model_copy(
            update={"lpr": config.lpr.model_copy(update={"enabled": False})}
        )
    elif lpr_mode == "every-frame":
        config = config.model_copy(
            update={"lpr": config.lpr.model_copy(update={"enabled": True})}
        )
    if frames < 1 or warmup < 0:
        raise ValueError("frames must be positive and warmup nonnegative")
    cpu_started = time.process_time()
    started = time.perf_counter()
    predictor = create_predictor(config)
    detector_load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    plates = (
        PlateProcessor(config.lpr, stale_seconds=config.stale_seconds)
        if config.lpr.enabled
        else None
    )
    plate_load_ms = (time.perf_counter() - started) * 1000 if plates else None
    processor = PredictionProcessor(config)
    # Measure inference/tracking/OCR without media allocation or output work.
    config = config.model_copy(
        update={
            "images": config.images.model_copy(update={"transport": "off"}),
            "storage": config.storage.model_copy(update={"enabled": False}),
        }
    )
    stream = prediction_stream(predictor, config)
    samples = {
        key: []
        for key in ("source_and_detection", "tracking", "lpr_scan", "frame_total")
    }
    measured_started = None
    warmup_completed = 0
    exhausted = False
    health_before = pi_health()
    try:
        for index in range(warmup + frames):
            if index == warmup:
                measured_started = time.perf_counter()
                cpu_started = time.process_time()
            frame_started = time.perf_counter()
            try:
                detections = next(stream)
            except StopIteration:
                exhausted = True
                break
            detection_ms = (time.perf_counter() - frame_started) * 1000
            stage_started = time.perf_counter()
            processor.process(detections, index + 1, detection_ms)
            tracking_ms = (time.perf_counter() - stage_started) * 1000
            plate_ms = None
            if plates is not None:
                previous_scan = plates.last_scan_at
                stage_started = time.perf_counter()
                plates.process(
                    detections.metadata["source_image"], force=lpr_mode == "every-frame"
                )
                if previous_scan != plates.last_scan_at:
                    plate_ms = (time.perf_counter() - stage_started) * 1000
            total_ms = (time.perf_counter() - frame_started) * 1000
            if index < warmup:
                warmup_completed += 1
                continue
            samples["source_and_detection"].append(detection_ms)
            samples["tracking"].append(tracking_ms)
            samples["frame_total"].append(total_ms)
            if plate_ms is not None:
                samples["lpr_scan"].append(plate_ms)
        elapsed = (
            time.perf_counter() - measured_started
            if measured_started is not None
            else 0
        )
        cpu_seconds = (
            time.process_time() - cpu_started if measured_started is not None else 0
        )
    finally:
        stream.close()
    count = len(samples["frame_total"])
    if not count:
        raise ValueError(
            "Source ended before any measured frames; lower warmup or use a longer recording"
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. This is the whole process high-water mark.
    peak_rss_mib = rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    versions = {}
    for package in (
        "supervision",
        "trackers",
        "onnxruntime",
        "fast-alpr",
        "numpy",
        "opencv-python-headless",
        "av",
    ):
        versions[package] = importlib.metadata.version(package)
    with config.model_path.open("rb") as artifact:
        model_sha256 = hashlib.file_digest(artifact, "sha256").hexdigest()
    session = getattr(predictor, "session", None)
    quantization = None
    if session is not None:
        props = session.get_modelmeta().custom_metadata_map
        raw = props.get(f"{getattr(predictor, 'family', 'rfdetr')}_quantization")
        quantization = json.loads(raw) if raw else None
    return {
        "schema_version": 2,
        "quantization": quantization,
        "model_bytes": config.model_path.stat().st_size,
        "detector_backend": "onnxruntime-direct",
        "helpers_source": getattr(predictor, "helpers_source", "rfdetr-1.12.1"),
        "detector_family": getattr(predictor, "family", "rfdetr"),
        "measured_at": datetime.now(UTC).isoformat(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "versions": versions,
        "source": source_description(config.source),
        "model": config.model_path.name,
        "model_sha256": model_sha256,
        "onnx_threads": config.onnx_threads,
        "configured_max_fps": config.max_fps,
        "lpr_mode": lpr_mode,
        "lpr_enabled": config.lpr.enabled,
        "lpr_models": {
            "detector": config.lpr.detector_model,
            "ocr": config.lpr.ocr_model,
        }
        if plates
        else None,
        "warmup_frames": warmup_completed,
        "measured_frames": count,
        "source_exhausted": exhausted,
        "detector_load_ms": round(detector_load_ms, 3),
        "plate_load_ms": round(plate_load_ms, 3) if plate_load_ms is not None else None,
        "elapsed_seconds": round(elapsed, 3),
        "pipeline_fps": round(count / elapsed, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "cpu_core_equivalents": round(cpu_seconds / elapsed, 3),
        "peak_process_rss_mib": round(peak_rss_mib, 1),
        "health_before": health_before,
        "health_after": pi_health(),
        "stages": {key: summarize(values) for key, values in samples.items()},
        "scope": "capture/decode + detector preprocessing/inference/postprocessing + tracking + optional plate scans; no MQTT, JPEG encoding, recording, RTSP output, annotations, or application rate limit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/onnx.yml")
    parser.add_argument(
        "--source",
        help="Override source with a recording, stream, or numeric camera index",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Override ONNX model path, relative to the working directory",
    )
    parser.add_argument(
        "--onnx-threads", type=int, help="Override ONNX Runtime CPU threads"
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--lpr", choices=("configured", "off", "every-frame"), default="configured"
    )
    parser.add_argument(
        "--output", type=Path, help="Save the JSON report as well as printing it"
    )
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup nonnegative")
    if args.onnx_threads is not None and args.onnx_threads < 1:
        parser.error("--onnx-threads must be positive")
    config = load_config(args.config)
    if args.model is not None:
        if args.model.suffix.lower() != ".onnx":
            parser.error("--model must point to an ONNX artifact")
        config.model_path = args.model.resolve()
    if args.onnx_threads is not None:
        config.onnx_threads = args.onnx_threads
    if args.source is not None:
        config.source = (
            int(args.source)
            if args.source.isascii() and args.source.isdigit()
            else args.source
        )
    report = benchmark(config, args.frames, args.warmup, args.lpr)
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
