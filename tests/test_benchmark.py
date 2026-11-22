"""Validate benchmark boundaries and reports without downloading model weights."""

import hashlib
from pathlib import Path

import benchmark as module
import numpy as np
import pytest
import supervision as sv
from onnx_camera import resolve_camera_source
from plates import PlateProcessor


class Stream:
    def __init__(self, count):
        self.remaining = count
        self.closed = False

    def __next__(self):
        if self.remaining == 0:
            raise StopIteration
        self.remaining -= 1
        result = sv.Detections.empty()
        result.metadata["source_image"] = np.zeros((20, 30, 3), dtype=np.uint8)
        return result

    def close(self):
        self.closed = True


@pytest.fixture
def setup(config, tmp_path, monkeypatch):
    config.model_path = tmp_path / "test.onnx"
    config.model_path.write_bytes(b"test model")
    monkeypatch.setattr(module, "create_predictor", lambda config: object())
    monkeypatch.setattr(
        module, "pi_health", lambda: {"temperature_c": None, "throttle_hex": None}
    )
    return config


def test_report_excludes_warmup_and_closes_finite_source(setup, monkeypatch):
    stream = Stream(5)
    monkeypatch.setattr(module, "prediction_stream", lambda *args: stream)
    report = module.benchmark(setup, frames=10, warmup=2, lpr_mode="off")
    assert report["warmup_frames"] == 2
    assert report["measured_frames"] == 3
    assert report["source_exhausted"]
    assert report["stages"]["frame_total"]["samples"] == 3
    assert report["stages"]["lpr_scan"] is None
    assert report["model_sha256"] == hashlib.sha256(b"test model").hexdigest()
    assert report["pipeline_fps"] > 0
    assert stream.closed


def test_empty_measurement_fails_and_releases_source(setup, monkeypatch):
    stream = Stream(1)
    monkeypatch.setattr(module, "prediction_stream", lambda *args: stream)
    with pytest.raises(ValueError, match="Source ended"):
        module.benchmark(setup, frames=2, warmup=1)
    assert stream.closed


def test_every_frame_plate_mode_measures_all_scans(setup, monkeypatch):
    class Engine:
        def predict(self, frame):
            return []

    stream = Stream(4)
    monkeypatch.setattr(module, "prediction_stream", lambda *args: stream)
    monkeypatch.setattr(
        module,
        "PlateProcessor",
        lambda config, **kwargs: PlateProcessor(config, engine=Engine(), **kwargs),
    )
    report = module.benchmark(setup, frames=3, warmup=1, lpr_mode="every-frame")
    assert report["lpr_enabled"]
    assert report["stages"]["lpr_scan"]["samples"] == 3
    assert not setup.lpr.enabled


def test_url_redaction_and_statistics():
    assert (
        module.source_description("rtsp://user:secret@camera/stream?token=hidden")
        == "rtsp://<redacted>"
    )
    assert module.source_description(0) == 0
    assert module.summarize([]) is None
    stats = module.summarize([1, 2, 3, 4, 5])
    assert stats["p50_ms"] == 3
    assert stats["p95_ms"] == 4.8


def test_stable_v4l2_source_resolution(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path("/dev/video4"))
    monkeypatch.setattr(Path, "is_char_device", lambda self: True)
    assert resolve_camera_source("/dev/v4l/by-id/usb-webcam-video-index0") == 4
    assert resolve_camera_source(1) == 1
    assert resolve_camera_source("rtsp://camera/stream") == "rtsp://camera/stream"


def test_non_camera_device_is_rejected(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path("/dev/null"))
    with pytest.raises(ValueError, match="V4L2"):
        resolve_camera_source("/dev/null")


@pytest.mark.parametrize("family", ["rfdetr", "yolox"])
def test_quantization_report_reads_detector_family(setup, monkeypatch, family):
    import json
    from types import SimpleNamespace

    details = {"mode": "static", "precision": "mixed-fp32-int8"}
    predictor = SimpleNamespace(
        family=family,
        session=SimpleNamespace(
            get_modelmeta=lambda: SimpleNamespace(
                custom_metadata_map={family + "_quantization": json.dumps(details)}
            )
        ),
    )
    monkeypatch.setattr(module, "create_predictor", lambda config: predictor)
    monkeypatch.setattr(module, "prediction_stream", lambda *args: Stream(2))
    report = module.benchmark(setup, frames=1, warmup=1, lpr_mode="off")
    assert report["quantization"] == details
    assert report["detector_family"] == family
