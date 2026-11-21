"""RF-DETR ONNX webcam predictions and Home Assistant MQTT discovery."""

import argparse
import json
import logging
import math
import os
import re
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from media import ImageConfig, MediaOutputs, StorageConfig
from onnx_detector import ONNXDetector
from paho.mqtt import client as mqtt
from plates import PlateConfig, PlateProcessor
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
)
from trackers import ByteTrackTracker
from vendor.rfdetr.streams import iter_live_frames

LOG = logging.getLogger(__name__)


class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    broker: str = Field(min_length=1)
    port: int = Field(default=1883, ge=1, le=65535)
    topic_prefix: str = "mqtt_camera"
    discovery_prefix: str = "homeassistant"
    ha_status_topic: str = "homeassistant/status"
    username_env: str = "MQTT_USERNAME"
    password_env: str = "MQTT_PASSWORD"
    tls: bool = False

    @field_validator("topic_prefix", "discovery_prefix", "ha_status_topic")
    @classmethod
    def valid_topic(cls, value):
        if not value.strip("/") or any(c in value for c in "+#\x00"):
            raise ValueError(
                "topics must be nonempty and cannot contain MQTT wildcards"
            )
        return value.strip("/")


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    minimum_consecutive_frames: int = Field(default=2, ge=1)
    lost_track_seconds: float = Field(default=2, gt=0)
    recovery_threshold: float = Field(default=0.1, ge=0, le=1)


class PresenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hold_seconds: NonNegativeFloat = 5
    hold_seconds_by_class: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera_id: str = Field(pattern=r"^[a-zA-Z0-9_]+$")
    name: str = Field(min_length=1)
    source: int | str = 0
    model_path: Path
    threshold: float = Field(default=0.5, ge=0, le=1)
    max_fps: float = Field(default=2, gt=0)
    onnx_threads: int = Field(
        default=2, ge=1, validation_alias=AliasChoices("onnx_threads", "torch_threads")
    )
    stale_seconds: int = Field(default=30, ge=1)
    classes: dict[str, list[str]]
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)
    lpr: PlateConfig = Field(default_factory=PlateConfig)
    images: ImageConfig = Field(default_factory=ImageConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    mqtt: BrokerConfig

    @model_validator(mode="after")
    def valid_processing(self):
        if self.tracking.enabled and self.tracking.recovery_threshold > self.threshold:
            raise ValueError("tracking.recovery_threshold must not exceed threshold")
        unknown = self.presence.hold_seconds_by_class.keys() - self.classes.keys()
        if unknown:
            raise ValueError(f"Unknown presence hold groups: {sorted(unknown)}")
        if self.lpr.enabled:
            period = max(self.lpr.interval_seconds, 1 / self.max_fps)
            if self.lpr.lost_track_seconds <= period:
                raise ValueError(
                    "lpr.lost_track_seconds must exceed the scan/frame interval"
                )
            if self.lpr.interval_seconds >= self.stale_seconds:
                raise ValueError("lpr.interval_seconds must be below stale_seconds")
            if (
                self.lpr.confirmation_window_seconds
                < (self.lpr.minimum_readings - 1) * period
            ):
                raise ValueError("LPR confirmation window is too short for max_fps")
        return self

    @field_validator("source")
    @classmethod
    def valid_source(cls, value):
        if isinstance(value, int) and value < 0 or isinstance(value, str) and not value:
            raise ValueError("source must be a camera index or nonempty path/URL")
        return value

    @field_validator("classes")
    @classmethod
    def valid_classes(cls, value):
        import re

        if not value or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", key)
            or not labels
            or any(not label.strip() for label in labels)
            for key, labels in value.items()
        ):
            raise ValueError("classes need safe entity keys and nonempty label lists")
        return value


def load_config(path):
    """Load a standalone inference config; resolve model paths beside it."""
    path = Path(path).resolve()
    config = CameraConfig.model_validate(yaml.safe_load(path.read_text()))
    config.model_path = (path.parent / config.model_path).resolve()
    config.storage.directory = (path.parent / config.storage.directory).resolve()
    if config.model_path.suffix.lower() != ".onnx":
        raise ValueError("model_path must point to an RF-DETR ONNX export")
    if config.stale_seconds <= 1 / config.max_fps:
        raise ValueError("stale_seconds must exceed the interval between predictions")
    return config


def select_classes(detections: sv.Detections, config):
    """Filter model output with Supervision before association or publication."""
    names = np.asarray(detections.data.get("class_name", []), dtype=str)
    if len(names) != len(detections):
        raise ValueError("RF-DETR predictions must include class_name metadata")
    wanted = {label for labels in config.classes.values() for label in labels}
    return detections[np.isin(names, list(wanted))]


def prediction_payload(detections: sv.Detections, config, sequence, processing_ms):
    """Serialize current detections; the processor applies held presence."""
    selected = select_classes(detections, config)
    names = np.asarray(selected.data.get("class_name", []), dtype=str)
    counts = {
        key: int(np.isin(names, labels).sum()) for key, labels in config.classes.items()
    }
    objects = [
        {
            "label": str(label),
            "class_id": int(class_id),
            "confidence": float(confidence),
            "box_xyxy": box.tolist(),
            "tracker_id": int(selected.tracker_id[index])
            if selected.tracker_id is not None
            else None,
        }
        for index, (label, class_id, confidence, box) in enumerate(
            zip(names, selected.class_id, selected.confidence, selected.xyxy)
        )
    ]
    return {
        "schema_version": 1,
        "camera_id": config.camera_id,
        "sequence": sequence,
        "predicted_at": datetime.now(UTC).isoformat(),
        "model": config.model_path.name,
        "processing_ms": round(processing_ms, 1),
        "counts": counts,
        "presence": {key: count > 0 for key, count in counts.items()},
        "detections": objects,
    }


class PresenceHold:
    """Hold presence in elapsed seconds while keeping instantaneous counts."""

    def __init__(self, config):
        self.holds = {
            key: config.presence.hold_seconds_by_class.get(
                key, config.presence.hold_seconds
            )
            for key in config.classes
        }
        self.last_seen = {}

    def update(self, counts, now):
        for key, count in counts.items():
            if count > 0:
                self.last_seen[key] = now
        return {
            key: count > 0 or now - self.last_seen.get(key, -math.inf) < self.holds[key]
            for key, count in counts.items()
        }

    def reset(self):
        self.last_seen.clear()


class PredictionProcessor:
    """Reuse Roboflow ByteTrack and gate presence on confirmed, observed tracks."""

    def __init__(self, config):
        self.config = config
        self.presence = PresenceHold(config)
        self.last_frame_at = None
        self.tracker = (
            ByteTrackTracker(
                lost_track_buffer=math.ceil(config.tracking.lost_track_seconds * 30),
                frame_rate=config.max_fps,
                track_activation_threshold=config.threshold,
                high_conf_det_threshold=config.threshold,
                minimum_consecutive_frames=config.tracking.minimum_consecutive_frames,
            )
            if config.tracking.enabled
            else None
        )

    def process(self, detections, sequence, processing_ms, *, now=None):
        now = time.monotonic() if now is None else now
        if (
            self.last_frame_at is not None
            and now - self.last_frame_at >= self.config.stale_seconds
        ):
            self.presence.reset()
            if self.tracker is not None:
                self.tracker.reset()
        self.last_frame_at = now
        selected = select_classes(detections, self.config)
        if self.tracker is not None:
            selected = self.tracker.update(selected, timestamp=now)
            selected = selected[selected.tracker_id >= 0]
        payload = prediction_payload(selected, self.config, sequence, processing_ms)
        payload["presence"] = self.presence.update(payload["counts"], now)
        return payload


class HomeAssistantPublisher:
    """Publish discovery and fresh state; keep inference out of MQTT callbacks."""

    def __init__(self, config):
        self.config = config
        self.base = f"{config.mqtt.topic_prefix}/{config.camera_id}"
        self.connected = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.last_result = None
        self.last_plate = None
        self.last_snapshot = None
        self.last_received = -math.inf
        self.online = False
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"mqtt_camera_{config.camera_id}",
        )
        self.client.max_queued_messages_set(64)
        self.client.reconnect_delay_set(1, 30)
        self.client.will_set(f"{self.base}/availability", "offline", qos=1, retain=True)
        username = os.getenv(config.mqtt.username_env)
        password = os.getenv(config.mqtt.password_env)
        if password and not username:
            raise ValueError("MQTT password is configured without a username")
        if username:
            self.client.username_pw_set(username, password)
        if config.mqtt.tls:
            self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.watchdog = threading.Thread(target=self._watch, daemon=True)

    def _publish(self, suffix, payload, *, retain=False, qos=0):
        info = self.client.publish(
            f"{self.base}/{suffix}", payload, qos=qos, retain=retain
        )
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            LOG.warning(
                "MQTT publish to %s failed: %s", suffix, mqtt.error_string(info.rc)
            )
        return info

    def discovery(self):
        """Yield per-entity discovery, grouped under one stable camera device."""
        common = {
            "state_topic": f"{self.base}/state",
            "availability_topic": f"{self.base}/availability",
            "expire_after": self.config.stale_seconds,
            "device": {
                "identifiers": [f"mqtt_camera_{self.config.camera_id}"],
                "name": self.config.name,
                "manufacturer": "mqtt-camera-streamer",
                "model": "RF-DETR ONNX",
            },
        }
        for key in self.config.classes:
            for platform in ("sensor", "binary_sensor"):
                entity = f"{self.config.camera_id}_{key}_{platform}"
                payload = dict(common, unique_id=f"mqtt_camera_{entity}")
                if platform == "sensor":
                    payload.update(
                        name=f"{key.replace('_', ' ').title()} count",
                        value_template="{{ value_json.counts['" + key + "'] }}",
                        state_class="measurement",
                    )
                else:
                    payload.update(
                        name=f"{key.replace('_', ' ').title()} detected",
                        device_class="occupancy",
                        value_template=(
                            "{{ 'ON' if value_json.presence['"
                            + key
                            + "'] else 'OFF' }}"
                        ),
                    )
                topic = (
                    f"{self.config.mqtt.discovery_prefix}/{platform}/{entity}/config"
                )
                yield topic, payload

        camera_topic = f"{self.config.mqtt.discovery_prefix}/camera/{self.config.camera_id}_snapshot/config"
        camera_payload = None
        if self.config.images.transport == "mqtt":
            camera_payload = {
                "name": "Snapshot",
                "unique_id": f"mqtt_camera_{self.config.camera_id}_snapshot",
                "topic": f"{self.base}/snapshot",
                "availability_topic": f"{self.base}/availability",
                "device": common["device"],
            }
        # Clear our previous discovery when switching away from MQTT images.
        yield camera_topic, camera_payload

        if self.config.lpr.enabled:
            plate_common = dict(common, state_topic=f"{self.base}/plate")
            # These are historical observations, not live occupancy states.
            plate_common.pop("expire_after")
            for key, name, template, extra in (
                ("last_plate", "Last plate", "{{ value_json.plate }}", {}),
                (
                    "plate_recognized_at",
                    "Plate recognized at",
                    "{{ value_json.recognized_at }}",
                    {"device_class": "timestamp"},
                ),
                (
                    "plate_confidence",
                    "Plate confidence",
                    "{{ (value_json.ocr_confidence * 100) | round(1) }}",
                    {"unit_of_measurement": "%", "entity_category": "diagnostic"},
                ),
            ):
                entity = f"{self.config.camera_id}_{key}"
                yield (
                    f"{self.config.mqtt.discovery_prefix}/sensor/{entity}/config",
                    dict(
                        plate_common,
                        unique_id=f"mqtt_camera_{entity}",
                        name=name,
                        value_template=template,
                        **extra,
                    ),
                )

    def _announce(self):
        for topic, payload in self.discovery():
            info = self.client.publish(
                topic,
                json.dumps(payload) if payload is not None else b"",
                qos=1,
                retain=True,
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                LOG.warning(
                    "MQTT discovery publish failed: %s", mqtt.error_string(info.rc)
                )

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            LOG.error("MQTT connection refused: %s", reason_code)
            return
        with self.lock:
            self._announce()
            client.subscribe(self.config.mqtt.ha_status_topic, qos=1)
            self._publish("availability", "offline", retain=True, qos=1)
            if self.last_plate is not None:
                self._publish("plate", json.dumps(self.last_plate), retain=True, qos=1)
            if (
                self.last_snapshot is not None
                and time.monotonic() - self.last_received < self.config.stale_seconds
            ):
                self._publish("snapshot", self.last_snapshot, retain=True)
            self.online = False
            self.connected.set()
        LOG.info("Connected to MQTT broker")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        self.online = False

    def _on_message(self, client, userdata, message):
        if message.payload == b"online":
            with self.lock:
                self._announce()
                if time.monotonic() - self.last_received < self.config.stale_seconds:
                    self._publish("state", json.dumps(self.last_result))

    def _watch(self):
        while not self.stop.wait(0.5):
            with self.lock:
                if (
                    self.online
                    and time.monotonic() - self.last_received
                    >= self.config.stale_seconds
                ):
                    self._publish("availability", "offline", retain=True, qos=1)
                    self.online = False

    def publish(self, payload):
        with self.lock:
            self.last_result = {k: v for k, v in payload.items() if k != "detections"}
            self.last_received = time.monotonic()
            if not self.connected.is_set():
                return
            # Transient predictions and state use QoS 0 so reconnects cannot replay a backlog.
            self._publish("predictions", json.dumps(payload, allow_nan=False))
            state = self._publish(
                "state", json.dumps(self.last_result, allow_nan=False)
            )
            if state.rc == mqtt.MQTT_ERR_SUCCESS and not self.online:
                self._publish("availability", "online", retain=True, qos=1)
                self.online = True

    def publish_snapshot(self, jpeg):
        """Keep only the newest JPEG; never queue disconnected camera frames."""
        with self.lock:
            self.last_snapshot = jpeg
            if self.connected.is_set():
                self._publish("snapshot", jpeg, retain=True)

    def publish_plate(self, event):
        """Retain the last confirmed observation; never replay transient plate events."""
        with self.lock:
            self.last_plate = dict(event, camera_id=self.config.camera_id)
            if not self.connected.is_set():
                return
            payload = json.dumps(self.last_plate, allow_nan=False)
            self._publish("plate", payload, retain=True, qos=1)
            self._publish("plate/events", payload)

    def __enter__(self):
        self.client.connect_async(
            self.config.mqtt.broker, self.config.mqtt.port, keepalive=15
        )
        self.client.loop_start()
        if not self.connected.wait(10):
            self.client.disconnect()
            self.client.loop_stop()
            raise ConnectionError(
                "MQTT connection timed out; check broker and credentials"
            )
        self.watchdog.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.watchdog.join(timeout=2)
        try:
            if self.connected.is_set():
                self._publish(
                    "availability", "offline", retain=True, qos=1
                ).wait_for_publish(timeout=2)
        finally:
            self.client.disconnect()
            self.client.loop_stop()


def create_predictor(config):
    """Load the same CPU predictor for the service and performance benchmark."""
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Export the model first: {config.model_path}")
    predictor = ONNXDetector(config.model_path, threads=config.onnx_threads)
    supported = set(predictor.class_names)
    requested = {label for labels in config.classes.values() for label in labels}
    if missing := requested - supported:
        raise ValueError(f"Configured classes absent from model: {sorted(missing)}")
    return predictor


def resolve_camera_source(source):
    """Resolve stable Linux V4L2 symlinks to indexes accepted by RF-DETR."""
    if not isinstance(source, str) or not source.startswith("/dev/"):
        return source
    device = Path(source).resolve(strict=True)
    match = re.fullmatch(r"/dev/video(\d+)", str(device))
    if match is None or not device.is_char_device():
        raise ValueError(
            "Camera device must resolve to a V4L2 /dev/videoN character device"
        )
    return int(match[1])


def prediction_stream(predictor, config):
    """Reuse Roboflow's bounded latest-frame capture with our ONNX-only detector."""
    frames = iter_live_frames(
        [resolve_camera_source(config.source)], stream_buffer=False
    )
    try:
        for batch in frames:
            yield predictor.predict_frame(
                batch[0],
                threshold=(
                    config.tracking.recovery_threshold
                    if config.tracking.enabled
                    else config.threshold
                ),
                include_source_image=(
                    config.lpr.enabled
                    or config.images.transport != "off"
                    or config.storage.enabled
                ),
            )
    finally:
        frames.close()


def run(config, max_frames=None):
    """Run camera inference and publish results without blocking MQTT callbacks."""
    predictor = create_predictor(config)
    processor = PredictionProcessor(config)
    plates = (
        PlateProcessor(config.lpr, stale_seconds=config.stale_seconds)
        if config.lpr.enabled
        else None
    )
    results = prediction_stream(predictor, config)
    try:
        with (
            HomeAssistantPublisher(config) as publisher,
            MediaOutputs(config, publisher) as media,
        ):
            sequence = 0
            while max_frames is None or sequence < max_frames:
                started = time.monotonic()
                try:
                    detections = next(results)
                except StopIteration:
                    break
                processing_ms = (time.monotonic() - started) * 1000
                sequence += 1
                payload = processor.process(detections, sequence, processing_ms)
                publisher.publish(payload)
                frame = detections.metadata.get("source_image")
                events = []
                if plates is not None:
                    if frame is None:
                        raise ValueError(
                            "RF-DETR result is missing its RGB source_image"
                        )
                    events = [
                        dict(event, frame_sequence=sequence)
                        for event in plates.process(frame)
                    ]
                    for event in events:
                        publisher.publish_plate(event)
                media.process(frame, payload, events)
                LOG.info(
                    "Frame %s: %s (%.0f ms)", sequence, payload["counts"], processing_ms
                )
                time.sleep(max(0, 1 / config.max_fps - (time.monotonic() - started)))
    finally:
        results.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.getenv("MQTT_CAMERA_ONNX_CONFIG", "config/onnx.yml")
    )
    parser.add_argument(
        "--max-frames", type=int, help="Stop after N predictions (smoke test)"
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        run(load_config(args.config), args.max_frames)
    except KeyboardInterrupt:
        LOG.info("Stopped")


if __name__ == "__main__":
    main()
