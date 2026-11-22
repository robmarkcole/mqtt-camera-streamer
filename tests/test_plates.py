"""Exercise OCR consensus, real Roboflow tracking and retained MQTT plate state."""

import json

import numpy as np
import pytest
from fast_alpr import ALPRResult
from fast_alpr.base import DetectionResult, OcrResult
from onnx_camera import HomeAssistantPublisher
from open_image_models.detection.core.base import BoundingBox
from plates import PlateConfig, PlateProcessor, normalize_reading
from test_onnx_camera import Observer

FRAME = np.zeros((40, 100, 3), dtype=np.uint8)


def reading(text="AB12CDE", confidence=0.95, detection_confidence=0.9):
    return ALPRResult(
        detection=DetectionResult(
            label="License Plate",
            confidence=detection_confidence,
            bounding_box=BoundingBox(x1=10, y1=10, x2=90, y2=30),
        ),
        ocr=OcrResult(text=text, confidence=confidence) if text is not None else None,
    )


class Engine:
    def __init__(self):
        self.results = [reading()]
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame)
        return self.results


@pytest.fixture
def processor():
    return PlateProcessor(PlateConfig(enabled=True), engine=Engine())


@pytest.mark.parametrize(
    "text,scores,expected",
    [
        ("ab 12-cde", [0.95, 0.99], "AB12CDE"),
        ("AB12CDE", [0.99, 0.2], ""),
        ("AB?12CDE", 0.99, ""),
        ("AB", 0.99, ""),
        ("AB12CDE", float("nan"), ""),
        ("AB12CDE", [], ""),
    ],
)
def test_reading_validation(text, scores, expected):
    assert (
        normalize_reading(OcrResult(text=text, confidence=scores), PlateConfig())[0]
        == expected
    )


def test_consensus_requires_repeated_readings_and_preserves_color_order(processor):
    rgb = FRAME.copy()
    rgb[:, :, 0] = 10
    rgb[:, :, 2] = 30
    assert processor.process(rgb, now=0) == []
    assert processor.process(rgb, now=1) == []
    events = processor.process(rgb, now=2)
    assert len(events) == 1
    event = events[0]
    assert event["plate"] == "AB12CDE"
    assert event["matching_readings"] == 2
    assert event["tracker_id"] >= 0
    assert event["recognized_at"].endswith("+00:00")
    assert len(event["event_id"]) == 32
    assert processor.engine.frames[0][0, 0].tolist() == [30, 0, 10]
    # Once recognized, a parked plate never retriggers even after cooldown expiry.
    for now in range(3, 40):
        assert processor.process(FRAME, now=now) == []


def test_conflicting_ocr_needs_majority(processor):
    events = []
    for now, text in enumerate(["AB12CDE", "AB12CDE", "AB12CDF", "AB12CDE", "AB12CDE"]):
        processor.engine.results = [reading(text)]
        current = processor.process(FRAME, now=now)
        if now < 4:
            assert not current
        events.extend(current)
    assert len(events) == 1 and events[0]["plate"] == "AB12CDE"
    assert events[0]["matching_readings"] == 3


def test_cooldown_survives_fragmented_track_and_allows_later_pass(processor):
    for now in range(3):
        first = processor.process(FRAME, now=now)
    assert first
    processor.engine.results = []
    for now in range(3, 9):
        assert not processor.process(FRAME, now=now)
    processor.engine.results = [reading()]
    for now in range(9, 40):
        assert not processor.process(FRAME, now=now)
    processor.engine.results = []
    for now in range(40, 47):
        processor.process(FRAME, now=now)
    processor.engine.results = [reading()]
    later = []
    for now in range(47, 50):
        later.extend(processor.process(FRAME, now=now))
    assert len(later) == 1
    assert later[0]["event_id"] != first[0]["event_id"]


def test_invalid_ocr_and_expired_readings_cannot_confirm(processor):
    processor.config.confirmation_window_seconds = 2
    processor.process(FRAME, now=0)
    processor.process(FRAME, now=1)
    processor.engine.results = [reading(text=None)]
    for now in range(2, 5):
        assert not processor.process(FRAME, now=now)
    processor.engine.results = [reading()]
    assert not processor.process(FRAME, now=5)
    assert processor.process(FRAME, now=6)


def test_sampling_and_stale_consensus_reset(processor):
    processor.process(FRAME, now=0)
    processor.process(FRAME, now=0.2)
    processor.process(FRAME, now=0.9)
    assert len(processor.engine.frames) == 1
    processor.process(FRAME, now=1)
    assert not processor.process(FRAME, now=32)
    assert not processor.process(FRAME, now=33)
    assert processor.process(FRAME, now=34)


@pytest.mark.parametrize(
    "values",
    [
        {"minimum_readings": 0},
        {"minimum_readings": 33},
        {"plate_pattern": "["},
        {"interval_seconds": 0},
        {"ocr_threshold": float("nan")},
        {"lost_track_seconds": 1},
        {"agreement": 0.5},
        {"cooldown_seconds": -1},
        {"minimum_readings": 10, "confirmation_window_seconds": 1},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        PlateConfig.model_validate(values)


def test_plate_mqtt_state_retained_but_events_not_replayed(config, broker_port):
    config.mqtt.port = broker_port
    config.lpr.enabled = True
    config.stale_seconds = 30
    processor = PlateProcessor(config.lpr, engine=Engine())
    for now in range(3):
        events = processor.process(FRAME, now=now)
    event = events[0]
    observer = Observer(broker_port)
    base = "mqtt_camera/test_camera"
    discovery = "homeassistant/sensor/test_camera_last_plate/config"
    try:
        with HomeAssistantPublisher(config) as publisher:
            payload = json.loads(observer.wait(discovery)[1])
            assert payload["state_topic"] == f"{base}/plate"
            assert "expire_after" not in payload
            publisher.publish_plate(event)
            published = observer.wait(f"{base}/plate/events")
            assert json.loads(published[1])["plate"] == "AB12CDE"
            assert not published[2]
            late = Observer(broker_port)
            try:
                retained = late.wait(f"{base}/plate")
                assert retained[2]
                assert json.loads(retained[1])["camera_id"] == "test_camera"
                assert not any(
                    topic == f"{base}/plate/events" for topic, _, _ in late.messages
                )
                after = len(observer.messages)
                observer.client.publish(
                    "homeassistant/status", "online", qos=1
                ).wait_for_publish(2)
                observer.wait(discovery, after=after)
                assert not any(
                    topic == f"{base}/plate/events"
                    for topic, _, _ in observer.messages[after:]
                )
            finally:
                late.close()
    finally:
        observer.close()
