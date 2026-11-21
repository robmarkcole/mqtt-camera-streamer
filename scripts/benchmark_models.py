"""Compare ONNX models serially in fresh processes, reversing alternate repeats."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(
    config, models, source, output, threads=4, frames=20, warmup=3, repeats=2, lpr="off"
):
    models = [Path(path).resolve() for path in models]
    if len(set(models)) != len(models) or not models:
        raise ValueError("Pass distinct ONNX models")
    if min(threads, frames, repeats) < 1 or warmup < 0:
        raise ValueError(
            "threads, frames and repeats must be positive; warmup nonnegative"
        )
    if any(not path.is_file() or path.suffix.lower() != ".onnx" for path in models):
        raise ValueError("Every model must be an existing ONNX artifact")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve prior reports")
    source_path = Path(str(source))
    source_hash = file_hash(source_path) if source_path.is_file() else None
    config_hash = file_hash(Path(config))
    grouped = {path: [] for path in models}
    for repeat in range(repeats):
        order = models if repeat % 2 == 0 else list(reversed(models))
        for model in order:
            prefix = f"{repeat + 1:02d}-{models.index(model):02d}-{model.stem}"
            report_path = output / f"{prefix}.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("benchmark.py")),
                "--config",
                str(Path(config).resolve()),
                "--model",
                str(model),
                "--source",
                str(source),
                "--onnx-threads",
                str(threads),
                "--frames",
                str(frames),
                "--warmup",
                str(warmup),
                "--lpr",
                lpr,
                "--output",
                str(report_path.resolve()),
            ]
            print(f"Benchmark {repeat + 1}/{repeats}: {model.name} ({lpr})", flush=True)
            with (output / f"{prefix}.log").open("w") as log:
                try:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError:
                    raise RuntimeError(
                        f"Benchmark failed for {model.name}; inspect its run log"
                    ) from None
            report = json.loads(report_path.read_text())
            if report["measured_frames"] != frames:
                raise ValueError(
                    "Source ended early; models must process the same frame count"
                )
            grouped[model].append(report)
    rows = []
    for model, reports in grouped.items():
        hashes = {report["model_sha256"] for report in reports}
        if len(hashes) != 1:
            raise ValueError("Model changed during repeated benchmarking")
        rows.append(
            {
                "model": model.name,
                "model_sha256": reports[0]["model_sha256"],
                "model_bytes": reports[0]["model_bytes"],
                "quantization": reports[0]["quantization"],
                "median_fps": statistics.median(
                    report["pipeline_fps"] for report in reports
                ),
                "median_frame_p95_ms": statistics.median(
                    report["stages"]["frame_total"]["p95_ms"] for report in reports
                ),
                "max_peak_rss_mib": max(
                    report["peak_process_rss_mib"] for report in reports
                ),
                "runs": reports,
            }
        )
    if file_hash(Path(config)) != config_hash or (
        source_hash is not None and file_hash(source_path) != source_hash
    ):
        raise ValueError("Configuration or recording changed during comparison")
    result = {
        "schema_version": 1,
        "onnx_threads": threads,
        "lpr": lpr,
        "frames_per_run": frames,
        "warmup_per_run": warmup,
        "repeats": repeats,
        "order": "forward, reverse on alternate repeats",
        "config_sha256": config_hash,
        "source": str(source)
        if "://" not in str(source)
        else str(source).split(":", 1)[0] + "://<redacted>",
        "source_sha256": source_hash,
        "models": rows,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/onnx.yml")
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--onnx-threads", default=4, type=int)
    parser.add_argument("--frames", default=20, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--repeats", default=2, type=int)
    parser.add_argument(
        "--lpr", choices=("off", "configured", "every-frame"), default="off"
    )
    args = parser.parse_args()
    result = compare(
        args.config,
        args.models,
        args.source,
        args.output_dir,
        args.onnx_threads,
        args.frames,
        args.warmup,
        args.repeats,
        args.lpr,
    )
    for model in result["models"]:
        print(
            f"{model['model']}: {model['median_fps']:.3f} FPS, {model['max_peak_rss_mib']:.1f} MiB peak RSS"
        )


if __name__ == "__main__":
    main()
