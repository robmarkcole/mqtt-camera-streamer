"""Exercise real Supervision detections and MQTT discovery on a local broker."""

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from paho.mqtt import client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from onnx_camera import (
    HomeAssistantPublisher,
    load_config,
    prediction_payload,
)


def detections():
    return sv.Detections(
        xyxy=np.array([[1, 2, 30, 40], [10, 20, 50, 60], [0, 0, 10, 10]], dtype=float),
        confidence=np.array([0.9, 0.8, 0.7]),
        class_id=np.array([1, 3, 17]),
        data={"class_name": np.array(["person", "car", "cat"])},
    )


def test_prediction_filtering_and_empty_frame(config):
    payload = prediction_payload(detections(), config, 42, 12.34)
    assert payload["counts"] == {"person": 1, "vehicle": 1}
    assert [d["label"] for d in payload["detections"]] == ["person", "car"]
    assert payload["detections"][0]["box_xyxy"] == [1, 2, 30, 40]
    assert payload["sequence"] == 42
    assert payload["predicted_at"].endswith("+00:00")
    assert json.loads(json.dumps(payload))["processing_ms"] == 12.3
    empty = prediction_payload(sv.Detections.empty(), config, 43, 5)
    assert empty["counts"] == {"person": 0, "vehicle": 0}
    assert empty["detections"] == []


def test_config_paths_and_invalid_values(tmp_path):
    path = tmp_path / "camera.yml"
    path.write_text("""
camera_id: front_door
name: Front door
model_path: ../models/nano.onnx
classes: {person: [person]}
mqtt: {broker: localhost}
""")
    config = load_config(path)
    assert config.model_path == (tmp_path.parent / "models/nano.onnx")
    for original, replacement in [
        ("front_door", "bad/id"),
        ("[person]", "[]"),
        ("broker: localhost", 'broker: localhost, topic_prefix: "bad/#"'),
    ]:
        text = path.read_text()
        path.write_text(text.replace(original, replacement))
        with pytest.raises(ValueError):
            load_config(path)
        path.write_text(text)
    path.write_text(path.read_text() + "max_fps: 0\n")
    with pytest.raises(ValueError):
        load_config(path)


class Observer:
    def __init__(self, port):
        self.messages = []
        self.ready = threading.Event()
        self.condition = threading.Condition()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = lambda client, *args: client.subscribe("#", qos=1)
        self.client.on_subscribe = lambda *args: self.ready.set()
        self.client.on_message = self.receive
        self.client.connect("127.0.0.1", port)
        self.client.loop_start()
        assert self.ready.wait(5)

    def receive(self, client, userdata, message):
        with self.condition:
            self.messages.append((message.topic, message.payload, message.retain))
            self.condition.notify_all()

    def wait(self, topic, payload=None, after=0):
        def find():
            return next(
                (
                    m
                    for m in self.messages[after:]
                    if m[0] == topic and (payload is None or m[1] == payload)
                ),
                None,
            )

        with self.condition:
            assert self.condition.wait_for(find, timeout=5), (
                f"missing MQTT message: {topic} {payload}"
            )
            return find()

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


def test_real_mqtt_discovery_state_restart_staleness_and_shutdown(config, broker_port):
    config.mqtt.port = broker_port
    observer = Observer(broker_port)
    base = "mqtt_camera/test_camera"
    discovery = "homeassistant/binary_sensor/test_camera_person_binary_sensor/config"
    try:
        with HomeAssistantPublisher(config) as publisher:
            discovered = json.loads(observer.wait(discovery)[1])
            assert discovered["availability_topic"] == f"{base}/availability"
            assert discovered["expire_after"] == 2
            assert "value_json.presence" in discovered["value_template"]
            assert discovered["device"]["identifiers"] == ["mqtt_camera_test_camera"]
            observer.wait(f"{base}/availability", b"offline")
            publisher.publish(prediction_payload(detections(), config, 1, 10))
            prediction = observer.wait(f"{base}/predictions")
            assert not prediction[2]
            assert json.loads(prediction[1])["counts"]["person"] == 1
            state = json.loads(observer.wait(f"{base}/state")[1])
            assert "detections" not in state
            observer.wait(f"{base}/availability", b"online")
            # A new subscriber receives retained discovery, but no historical predictions/state.
            late = Observer(broker_port)
            try:
                assert late.wait(discovery)[2]
                late.wait(f"{base}/availability", b"online")
                assert not any(
                    m[0] in (f"{base}/predictions", f"{base}/state")
                    for m in late.messages
                )
            finally:
                late.close()
            after = len(observer.messages)
            observer.client.publish(
                "homeassistant/status", "online", qos=1
            ).wait_for_publish(2)
            observer.wait(discovery, after=after)
            observer.wait(f"{base}/state", after=after)
            after = len(observer.messages)
            observer.wait(f"{base}/availability", b"offline", after=after)
            after = len(observer.messages)
            publisher.publish(prediction_payload(sv.Detections.empty(), config, 2, 10))
            zero_state = json.loads(observer.wait(f"{base}/state", after=after)[1])
            assert zero_state["counts"] == {"person": 0, "vehicle": 0}
            observer.wait(f"{base}/availability", b"online", after=after)
            after = len(observer.messages)
        observer.wait(f"{base}/availability", b"offline", after=after)
    finally:
        observer.close()


def test_no_stale_predictions_queued_when_disconnected(config, broker_port):
    config.mqtt.port = broker_port
    publisher = HomeAssistantPublisher(config)
    publisher.publish(prediction_payload(detections(), config, 1, 10))
    observer = Observer(broker_port)
    base = "mqtt_camera/test_camera"
    try:
        with publisher:
            observer.wait(f"{base}/availability", b"offline")
            assert not any(
                topic in (f"{base}/state", f"{base}/predictions")
                for topic, _, _ in observer.messages
            )
            publisher.publish(prediction_payload(sv.Detections.empty(), config, 2, 10))
            state = json.loads(observer.wait(f"{base}/state")[1])
            assert state["sequence"] == 2
            assert state["counts"]["person"] == 0
    finally:
        observer.close()
