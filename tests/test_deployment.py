"""Check deployment configuration and legacy MQTT compatibility under uv."""

import time
from pathlib import Path

import mqtt as legacy_mqtt
import pytest
from onnx_camera import load_config
from test_onnx_camera import Observer


def test_pi_example_is_valid_and_uses_separate_secret_environment():
    config = load_config(Path(__file__).resolve().parents[1] / "deploy/pi.yml")
    assert config.source.startswith("/dev/v4l/by-id/")
    assert config.model_path.is_absolute()
    assert config.mqtt.username_env == "MQTT_USERNAME"
    assert config.mqtt.password_env == "MQTT_PASSWORD"
    assert config.torch_threads == 2
    assert not config.lpr.enabled


def test_legacy_mqtt_callbacks_work_with_paho_v2(broker_port, monkeypatch):
    monkeypatch.setenv("MQTT_USERNAME", "dev_test")
    monkeypatch.setenv("MQTT_PASSWORD", "dev_test_password")
    observer = Observer(broker_port)
    client = legacy_mqtt.get_mqtt_client()
    try:
        client.connect("127.0.0.1", broker_port)
        client.loop_start()
        deadline = time.monotonic() + 5
        while not client.connected_flag and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.connected_flag
        client.publish("legacy/test", b"frame", qos=1).wait_for_publish(2)
        assert observer.wait("legacy/test")[1] == b"frame"
    finally:
        client.disconnect()
        client.loop_stop()
        observer.close()


def test_legacy_mqtt_rejects_password_without_username(monkeypatch):
    monkeypatch.delenv("MQTT_USERNAME", raising=False)
    monkeypatch.setenv("MQTT_PASSWORD", "dev_test_password")
    with pytest.raises(ValueError, match="without a username"):
        legacy_mqtt.get_mqtt_client()
