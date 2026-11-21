"""Check the single-source boundary around the reused Roboflow capture worker."""

import numpy as np
import pytest
import supervision as sv
from onnx_camera import prediction_stream
from vendor.rfdetr.streams import iter_live_frames


class FiniteCapture:
    def __init__(self, *args):
        self.frames = [
            np.full((4, 6, 3), [value, 2, 9], dtype=np.uint8) for value in range(5)
        ]
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return 5

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


def test_finite_capture_preserves_every_frame_rgb_and_releases(monkeypatch):
    capture = FiniteCapture()
    monkeypatch.setattr("cv2.VideoCapture", lambda *args: capture)
    iterator = iter_live_frames(["clip.mp4"])
    frames = [batch[0] for batch in iterator]
    assert len(frames) == 5 and capture.released
    assert [frame[0, 0].tolist() for frame in frames] == [
        [9, 2, value] for value in range(5)
    ]


def test_prediction_stream_closes_capture_if_detector_fails(config, monkeypatch):
    capture = FiniteCapture()
    monkeypatch.setattr("cv2.VideoCapture", lambda *args: capture)

    class Detector:
        def predict_frame(self, *args, **kwargs):
            raise ValueError("Inference failed")

    with pytest.raises(ValueError, match="Inference failed"):
        next(prediction_stream(Detector(), config))
    assert capture.released


def test_partial_prediction_stream_close_stops_capture(config, monkeypatch):
    capture = FiniteCapture()
    monkeypatch.setattr("cv2.VideoCapture", lambda *args: capture)
    seen = []

    class Detector:
        def predict_frame(self, rgb, **kwargs):
            seen.append(kwargs)
            return sv.Detections.empty()

    stream = prediction_stream(Detector(), config)
    next(stream)
    stream.close()
    assert capture.released
    assert seen == [
        {"threshold": config.tracking.recovery_threshold, "include_source_image": True}
    ]
