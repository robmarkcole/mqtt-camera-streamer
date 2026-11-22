"""Check real track lifecycles, exact watches, event delivery and plate geometry."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import supervision as sv
from media import annotated_image
from onnx_camera import HomeAssistantPublisher, PredictionProcessor
from plates import PlateConfig, PlateProcessor
from test_onnx_camera import Observer
from test_plates import FRAME, Engine
from vehicle_events import enrich_plate, event_payload


def car():
    return sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([1]),
        data={"class_name": np.array(["car"])},
    )


def test_confirmed_vehicle_emits_once_recovers_then_loses(config):
    config.stale_seconds = 30
    p = PredictionProcessor(config)
    assert p.process(car(), 1, 0, now=0)["events"] == []
    detected = p.process(car(), 2, 0, now=0.5)["events"]
    assert len(detected) == 1
    assert detected[0]["event_type"] == "vehicle_detected"
    for seq in range(3, 8):
        assert p.process(car(), seq, 0, now=seq * 0.5)["events"] == []
    assert p.process(sv.Detections.empty(), 8, 0, now=4)["events"] == []
    assert p.process(car(), 9, 0, now=4.5)["events"] == []
    lost = p.process(sv.Detections.empty(), 10, 0, now=6.5)["events"]
    assert len(lost) == 1 and lost[0]["event_type"] == "vehicle_lost"
    assert lost[0]["vehicle_tracker_id"] == detected[0]["vehicle_tracker_id"]
    assert p.process(sv.Detections.empty(), 11, 0, now=7)["events"] == []


def test_stale_reset_and_disabled_tracking_do_not_invent_departures(config):
    p = PredictionProcessor(config)
    p.process(car(), 1, 0, now=0)
    p.process(car(), 2, 0, now=0.5)
    assert p.process(sv.Detections.empty(), 3, 0, now=20)["events"] == []
    config.tracking.enabled = False
    p = PredictionProcessor(config)
    assert p.process(car(), 1, 0, now=0)["events"] == []


def test_plate_link_requires_confirmed_containing_vehicle():
    event = {
        "plate": "ABC123",
        "box_xyxy": [10, 20, 30, 40],
        "recognized_at": "2026-11-22T00:00:00+00:00",
        "ocr_confidence": 0.95,
    }
    payload = {
        "detections": [
            {
                "label": "car",
                "tracker_id": 2,
                "box_xyxy": [0, 0, 100, 80],
                "confidence": 0.9,
            }
        ]
    }
    r = enrich_plate(event, payload, (100, 200, 3), ["car"])
    assert r["vehicle_type"] == "car" and r["vehicle_tracker_id"] == 2
    assert r["box_centre_normalized"] == [0.1, 0.3] and r["box_x_centre"] == 20
    payload["detections"][0]["box_xyxy"] = [100, 50, 150, 90]
    assert enrich_plate(event, payload, (100, 200, 3), ["car"])["vehicle_type"] is None


def test_exact_normalized_watch_preserves_consensus():
    config = PlateConfig(enabled=True, watched_plates={"ab 12-cde": "Family"})
    p = PlateProcessor(config, engine=Engine())
    p.process(FRAME, now=0)
    p.process(FRAME, now=1)
    e = p.process(FRAME, now=2)[0]
    assert e["watched"] and e["plate_alias"] == "Family"
    for now in range(3, 35):
        assert not p.process(FRAME, now=now)
    for invalid in [
        {"AB12CDE": "one", "ab12-cde": "two"},
        {"??": "wrong"},
        {"ABC123": ""},
    ]:
        with pytest.raises(ValueError):
            PlateConfig(watched_plates=invalid)


def test_real_mqtt_event_discovery_no_replay_and_watch(config, broker_port):
    config.mqtt.port = broker_port
    config.demo = True
    config.lpr.enabled = True
    observer = Observer(broker_port)
    try:
        with HomeAssistantPublisher(config) as publisher:
            discovered = json.loads(
                observer.wait("homeassistant/event/test_camera_vehicle_events/config")[
                    1
                ]
            )
            assert "watched_plate_detected" in discovered["event_types"]
            event = event_payload(
                "vehicle_detected", vehicle_type="car", vehicle_tracker_id=2
            )
            publisher.publish_event(event)
            received = json.loads(observer.wait("mqtt_camera/test_camera/events")[1])
            assert received["demo"] and received["camera_id"] == "test_camera"
            plate = {
                "schema_version": 1,
                "event_id": "plate-id",
                "plate": "ABC123",
                "recognized_at": "2026-11-22T00:00:00+00:00",
                "watched": True,
                "plate_alias": "Family",
            }
            publisher.publish_plate(plate)
            observer.wait("mqtt_camera/test_camera/plate/events")
            import time

            deadline = time.monotonic() + 5
            while not any(
                t.endswith("/events") and b"watched_plate_detected" in p
                for t, p, r in observer.messages
            ):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            publisher.connected.clear()
            with patch.object(publisher, "_publish") as send:
                publisher.publish_event(event)
                send.assert_not_called()
            publisher.connected.set()
        late = Observer(broker_port)
        try:
            late.wait("mqtt_camera/test_camera/plate")
            assert not [m for m in late.messages if m[0].endswith("/events")]
        finally:
            late.close()
    finally:
        observer.close()


def test_annotation_keeps_original_rgb_and_uses_real_boxes():
    rgb = np.zeros((120, 200, 3), dtype=np.uint8)
    original = rgb.copy()
    p = {
        "detections": [
            {
                "label": "car",
                "tracker_id": 2,
                "class_id": 1,
                "confidence": 0.9,
                "box_xyxy": [40, 40, 180, 100],
            }
        ]
    }
    result = annotated_image(rgb, p, [])
    np.testing.assert_array_equal(rgb, original)
    assert result.shape == rgb.shape and np.any(result != original)
