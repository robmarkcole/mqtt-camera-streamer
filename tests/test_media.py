"""Exercise media rate limits, file retention and real MQTT image discovery."""

import io
import json
import sqlite3
from unittest.mock import Mock

import numpy as np
from media import ImageConfig, MediaOutputs, SnapshotStore, StorageConfig, jpeg_image
from onnx_camera import HomeAssistantPublisher
from PIL import Image
from test_onnx_camera import Observer


def test_file_storage_retains_paths_and_preserves_old_blobs(tmp_path):
    old = sqlite3.connect(tmp_path / "records.db")
    old.execute("CREATE TABLE mqtt_camera (name TEXT, data BLOB)")
    old.execute("INSERT INTO mqtt_camera VALUES ('old', ?)", (b"legacy",))
    old.commit()
    old.close()
    store = SnapshotStore(StorageConfig(directory=tmp_path, max_files=2))
    payload = {"predicted_at": "2026-01-01T00:00:00+00:00", "sequence": 1}
    jpeg = jpeg_image(np.zeros((20, 40, 3), dtype=np.uint8), ImageConfig())
    paths = [
        store.save(jpeg, "camera", payload, [{"plate": "ABC123"}]) for _ in range(3)
    ]
    assert not (tmp_path / paths[0]).exists()
    assert all((tmp_path / path).read_bytes() == jpeg for path in paths[1:])
    rows = store.db.execute(
        "SELECT image_path, predictions_json, plates_json FROM frames"
    ).fetchall()
    assert len(rows) == 2
    assert json.loads(rows[0][1]) == payload
    assert json.loads(rows[0][2])[0]["plate"] == "ABC123"
    assert store.db.execute("SELECT data FROM mqtt_camera").fetchone()[0] == b"legacy"
    store.db.execute("UPDATE frames SET created_unix = 0")
    store.db.commit()
    store.prune()
    assert store.db.execute("SELECT count(*) FROM frames").fetchone()[0] == 0
    assert not any(tmp_path.rglob("*.jpg"))
    store.close()


def test_failed_database_insert_removes_file(tmp_path):
    store = SnapshotStore(StorageConfig(directory=tmp_path))
    store.db.execute("DROP TABLE frames")
    import pytest

    with pytest.raises(sqlite3.OperationalError):
        store.save(b"jpeg", "camera", {"predicted_at": "now"})
    assert not any(tmp_path.rglob("*.jpg"))
    assert not any(tmp_path.rglob("*.tmp"))
    store.close()


def test_sparse_images_and_plate_event_forces_snapshot(config, tmp_path, monkeypatch):
    config.storage = StorageConfig(enabled=True, directory=tmp_path)
    config.images.max_width = 100
    publisher = Mock()
    clock = [10.0]
    monkeypatch.setattr("media.time.monotonic", lambda: clock[0])
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    payload = {"predicted_at": "now"}
    with MediaOutputs(config, publisher) as output:
        output.process(frame, payload)
        output.process(frame, payload)
        assert publisher.publish_snapshot.call_count == 1
        clock[0] += 2
        output.process(frame, payload)
        assert publisher.publish_snapshot.call_count == 2
        assert output.store.db.execute("SELECT count(*) FROM frames").fetchone()[0] == 1
        output.process(frame, payload, [{"plate": "ABC123"}])
        assert publisher.publish_snapshot.call_count == 3
        assert output.store.db.execute("SELECT count(*) FROM frames").fetchone()[0] == 2
    with Image.open(io.BytesIO(publisher.publish_snapshot.call_args.args[0])) as image:
        assert image.format == "JPEG" and image.size == (100, 50)


def test_rtsp_mode_sends_no_mqtt_images_and_can_record(config, tmp_path, monkeypatch):
    config.images.transport = "rtsp"
    config.storage = StorageConfig(enabled=True, directory=tmp_path)
    rtsp = Mock()
    monkeypatch.setattr("media.RTSPOutput", Mock(return_value=rtsp))
    publisher = Mock()
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    with MediaOutputs(config, publisher) as output:
        output.process(frame, {"predicted_at": "now"})
    rtsp.submit.assert_called_once_with(frame)
    rtsp.close.assert_called_once()
    publisher.publish_snapshot.assert_not_called()
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_mqtt_snapshot_retention_and_discovery_cleanup(config, broker_port):
    config.mqtt.port = broker_port
    jpeg = jpeg_image(np.zeros((20, 40, 3), dtype=np.uint8), config.images)
    observer = Observer(broker_port)
    discovery = "homeassistant/camera/test_camera_snapshot/config"
    topic = "mqtt_camera/test_camera/snapshot"
    try:
        with HomeAssistantPublisher(config) as publisher:
            camera = json.loads(observer.wait(discovery)[1])
            assert camera["topic"] == topic
            publisher.publish_snapshot(jpeg)
            assert observer.wait(topic)[1] == jpeg
            late = Observer(broker_port)
            try:
                assert late.wait(topic)[2]
            finally:
                late.close()
        after = len(observer.messages)
        config.images.transport = "off"
        with HomeAssistantPublisher(config):
            observer.wait(discovery, b"", after=after)
    finally:
        observer.close()


def test_rtsp_latest_frame_slot_is_bounded_and_shutdown_is_clean(monkeypatch):
    import threading

    from media import RTSPOutput

    monkeypatch.setenv("RTSP_PUBLISH_URL", "rtsp://localhost:8554/camera")
    ready = threading.Event()
    release = threading.Event()

    def blocked_network(self):
        ready.set()
        release.wait(5)
        self.stop.wait(5)

    monkeypatch.setattr(RTSPOutput, "_run", blocked_network)
    output = RTSPOutput(ImageConfig(transport="rtsp", max_width=40), 3)
    try:
        assert ready.wait(2)
        for value in range(10):
            output.submit(np.full((21, 41, 3), value, dtype=np.uint8))
        latest, _ = output.latest
        assert latest.shape == (20, 40, 3)
        assert np.all(latest == 9)
    finally:
        release.set()
        output.close()
    assert not output.thread.is_alive()
