"""Sparse MQTT JPEGs, optional PyAV RTSP output and file-backed SQLite records."""

import io
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

LOG = logging.getLogger(__name__)


class ImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transport: Literal["mqtt", "rtsp", "off"] = "mqtt"
    interval_seconds: float = Field(default=1, gt=0)
    jpeg_quality: int = Field(default=80, ge=1, le=95)
    max_width: int = Field(default=960, ge=2)
    rtsp_url_env: str = "RTSP_PUBLISH_URL"
    rtsp_fps: float = Field(default=2, gt=0, le=30)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    directory: Path = Path("../captures")
    interval_seconds: float = Field(default=5, gt=0)
    max_files: int = Field(default=10000, ge=1)
    max_age_days: float = Field(default=7, gt=0)


def jpeg_image(rgb, config):
    image = Image.fromarray(rgb)
    if image.width > config.max_width:
        image = image.resize(
            (
                config.max_width,
                max(1, round(image.height * config.max_width / image.width)),
            )
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=config.jpeg_quality)
    return output.getvalue()


class SnapshotStore:
    """Own a single-threaded connection; existing legacy BLOB tables are untouched."""

    def __init__(self, config):
        self.config = config
        self.root = config.directory.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "records.db")
        self.db.execute("""CREATE TABLE IF NOT EXISTS frames (
            id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, captured_at TEXT NOT NULL,
            created_unix REAL NOT NULL, image_path TEXT NOT NULL,
            predictions_json TEXT NOT NULL, plates_json TEXT NOT NULL)""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS frames_created ON frames(created_unix)"
        )
        self.db.commit()
        self.prune()

    def save(self, jpeg, camera_id, payload, plates=()):
        identifier = uuid.uuid4().hex
        # A generated filename keeps camera IDs and MQTT topics out of filesystem paths.
        relative = Path(time.strftime("%Y-%m-%d", time.gmtime())) / f"{identifier}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_bytes(jpeg)
            os.replace(temporary, destination)
            with self.db:
                self.db.execute(
                    "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        camera_id,
                        payload["predicted_at"],
                        time.time(),
                        relative.as_posix(),
                        json.dumps(payload, allow_nan=False),
                        json.dumps(plates, allow_nan=False),
                    ),
                )
        except BaseException:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        self.prune()
        return relative

    def prune(self):
        cutoff = time.time() - self.config.max_age_days * 86400
        rows = self.db.execute(
            """SELECT id, image_path FROM frames
            WHERE created_unix < ? OR id IN (
                SELECT id FROM frames ORDER BY created_unix DESC, id DESC LIMIT -1 OFFSET ?)
            """,
            (cutoff, self.config.max_files),
        ).fetchall()
        for identifier, relative in rows:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                LOG.warning("Ignoring an image path outside the storage directory")
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOG.warning("Could not remove an expired snapshot")
                continue
            with self.db:
                self.db.execute("DELETE FROM frames WHERE id = ?", (identifier,))

    def close(self):
        self.db.close()


class RTSPOutput:
    """Publish with the PyAV/FFmpeg libraries already used by Supervision.

    One latest-frame slot prevents a slow encoder/network from queuing frames.
    The worker exclusively owns the container, including close operations.
    """

    def __init__(self, config, stale_seconds):
        import av

        self.config = config
        self.stale_seconds = stale_seconds
        self.url = os.getenv(config.rtsp_url_env, "")
        if not self.url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"Set {config.rtsp_url_env} to an RTSP publishing URL")
        av.codec.Codec(
            "libx264", "w"
        )  # Fail at startup if the wheel lacks H.264 encoding.
        self.lock = threading.Lock()
        self.latest = None
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="rtsp-output"
        )
        self.thread.start()

    def submit(self, rgb):
        image = Image.fromarray(rgb)
        width = max(2, min(image.width, self.config.max_width) // 2 * 2)
        height = max(2, round(image.height * width / image.width) // 2 * 2)
        frame = (np.asarray(image.resize((width, height))), time.monotonic())
        with self.lock:
            self.latest = frame

    def _run(self):
        import av

        container = None
        dimensions = None
        sequence = 0
        retry_after = 0

        def close_container():
            nonlocal container
            if container is not None:
                try:
                    container.close()
                except (av.error.FFmpegError, OSError):
                    pass
                container = None

        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    latest = self.latest
                if latest is None or started - latest[1] >= self.stale_seconds:
                    close_container()
                    self.stop.wait(0.1)
                    continue
                rgb, _ = latest
                height, width = rgb.shape[:2]
                if dimensions != (width, height):
                    close_container()
                if started < retry_after:
                    self.stop.wait(0.1)
                    continue
                try:
                    if container is None:
                        container = av.open(
                            self.url,
                            mode="w",
                            format="rtsp",
                            options={"rtsp_transport": "tcp"},
                            timeout=5.0,
                        )
                        stream = container.add_stream(
                            "libx264", rate=Fraction(str(self.config.rtsp_fps))
                        )
                        stream.width, stream.height = width, height
                        stream.pix_fmt = "yuv420p"
                        stream.codec_context.thread_count = 2
                        stream.codec_context.gop_size = max(
                            1, round(self.config.rtsp_fps * 2)
                        )
                        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
                        dimensions = (width, height)
                        sequence = 0
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts = sequence
                    sequence += 1
                    for packet in stream.encode(frame):
                        container.mux(packet)
                except (av.error.FFmpegError, OSError, ValueError):
                    if not self.stop.is_set():
                        LOG.warning(
                            "RTSP publisher disconnected; retrying in 10 seconds"
                        )
                    close_container()
                    retry_after = time.monotonic() + 10
                self.stop.wait(
                    max(0, 1 / self.config.rtsp_fps - (time.monotonic() - started))
                )
        finally:
            close_container()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            LOG.warning(
                "RTSP network operation has not stopped; worker will exit with the service"
            )


class MediaOutputs:
    def __init__(self, config, publisher):
        self.config = config
        self.publisher = publisher
        self.snapshot_at = self.saved_at = -float("inf")
        self.store = None
        self.rtsp = None

    def __enter__(self):
        try:
            if self.config.storage.enabled:
                self.store = SnapshotStore(self.config.storage)
            if self.config.images.transport == "rtsp":
                self.rtsp = RTSPOutput(self.config.images, self.config.stale_seconds)
            return self
        except BaseException:
            self.__exit__()
            raise

    def process(self, rgb, payload, plates=()):
        if rgb is None:
            if self.config.images.transport != "off" or self.store:
                raise ValueError("RF-DETR result is missing its RGB source_image")
            return
        now = time.monotonic()
        publish = self.config.images.transport == "mqtt" and (
            bool(plates)
            or now - self.snapshot_at >= self.config.images.interval_seconds
        )
        save = self.store is not None and (
            bool(plates) or now - self.saved_at >= self.config.storage.interval_seconds
        )
        if self.rtsp:
            self.rtsp.submit(rgb)
        if publish or save:
            jpeg = jpeg_image(rgb, self.config.images)
            if publish:
                self.publisher.publish_snapshot(jpeg)
                self.snapshot_at = now
            if save:
                self.store.save(jpeg, self.config.camera_id, payload, plates)
                self.saved_at = now

    def __exit__(self, *exc):
        if self.rtsp:
            self.rtsp.close()
        if self.store:
            self.store.close()
