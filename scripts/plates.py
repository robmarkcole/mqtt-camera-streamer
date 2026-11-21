"""Local pretrained ONNX plate recognition with Roboflow tracking and consensus."""

import math
import re
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import supervision as sv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from trackers import ByteTrackTracker


class PlateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    detector_model: str = Field(
        default="yolo-v9-t-384-license-plate-end2end", min_length=1
    )
    ocr_model: str = Field(default="cct-xs-v2-global-model", min_length=1)
    detection_threshold: float = Field(default=0.5, ge=0, le=1)
    ocr_threshold: float = Field(default=0.8, ge=0, le=1)
    interval_seconds: float = Field(default=1, gt=0)
    lost_track_seconds: float = Field(default=5, gt=0)
    minimum_readings: int = Field(default=2, ge=1, le=32)
    agreement: float = Field(default=0.75, gt=0.5, le=1)
    confirmation_window_seconds: float = Field(default=5, gt=0)
    cooldown_seconds: float = Field(default=30, ge=0)
    plate_pattern: str = Field(default=r"[A-Z0-9]{3,10}", min_length=1)
    num_threads: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def valid_timing(self):
        if self.lost_track_seconds <= self.interval_seconds:
            raise ValueError("lpr.lost_track_seconds must exceed interval_seconds")
        if (
            self.confirmation_window_seconds
            < (self.minimum_readings - 1) * self.interval_seconds
        ):
            raise ValueError(
                "lpr.confirmation_window_seconds is too short for minimum_readings"
            )
        return self

    @field_validator("plate_pattern")
    @classmethod
    def valid_pattern(cls, value):
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError("Invalid plate_pattern regular expression") from error
        return value


def create_alpr(config):
    """Reuse FastALPR's detector, original-resolution crops and OCR on CPU."""
    import onnxruntime as ort
    from fast_alpr import ALPR

    options = ort.SessionOptions()
    options.intra_op_num_threads = config.num_threads
    options.inter_op_num_threads = 1
    return ALPR(
        detector_model=config.detector_model,
        detector_conf_thresh=config.detection_threshold,
        detector_providers=["CPUExecutionProvider"],
        detector_sess_options=options,
        ocr_model=config.ocr_model,
        ocr_device="cpu",
        ocr_providers=["CPUExecutionProvider"],
        ocr_sess_options=options,
    )


def normalize_reading(ocr, config):
    """Validate text and the least-confident character; never guess O/0 or I/1."""
    if ocr is None:
        return "", 0.0
    text = re.sub(r"[\s-]+", "", ocr.text.upper())
    scores = ocr.confidence if isinstance(ocr.confidence, list) else [ocr.confidence]
    if not scores or any(
        not math.isfinite(score) or not 0 <= score <= 1 for score in scores
    ):
        return "", 0.0
    confidence = min(scores)
    if confidence < config.ocr_threshold or not re.fullmatch(
        config.plate_pattern, text
    ):
        return "", 0.0
    return text, confidence


@dataclass
class PlateHistory:
    last_seen: float
    readings: deque = field(default_factory=lambda: deque(maxlen=32))
    emitted: bool = False


class PlateProcessor:
    """Confirm text per plate track and emit once per track plus a text cooldown."""

    def __init__(self, config, *, stale_seconds=30, engine=None):
        self.config = config
        self.stale_seconds = stale_seconds
        self.engine = create_alpr(config) if engine is None else engine
        self.tracker = ByteTrackTracker(
            lost_track_buffer=math.ceil(config.lost_track_seconds * 30),
            frame_rate=1 / config.interval_seconds,
            minimum_consecutive_frames=2,
            high_conf_det_threshold=config.detection_threshold,
            track_activation_threshold=config.detection_threshold,
        )
        self.last_scan_at = None
        self.histories = {}
        self.recent_plates = {}

    def _detections(self, results):
        boxes, scores, texts, ocr_scores = [], [], [], []
        for result in results:
            detection = result.detection
            box = detection.bounding_box
            xyxy = [box.x1, box.y1, box.x2, box.y2]
            if (
                not np.isfinite(xyxy).all()
                or xyxy[2] <= xyxy[0]
                or xyxy[3] <= xyxy[1]
                or not math.isfinite(detection.confidence)
                or not self.config.detection_threshold <= detection.confidence <= 1
            ):
                continue
            text, confidence = normalize_reading(result.ocr, self.config)
            boxes.append(xyxy)
            scores.append(detection.confidence)
            texts.append(text)
            ocr_scores.append(confidence)
        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            confidence=np.asarray(scores, dtype=float),
            class_id=np.zeros(len(boxes), dtype=int),
            data={
                "plate": np.asarray(texts, dtype=str),
                "ocr_confidence": np.asarray(ocr_scores, dtype=float),
            },
        )

    def process(self, rgb_frame, *, now=None, force=False):
        now = time.monotonic() if now is None else now
        if self.last_scan_at is not None:
            if now - self.last_scan_at >= self.stale_seconds:
                self.tracker.reset()
                self.histories.clear()
            elif not force and now - self.last_scan_at < self.config.interval_seconds:
                return []
        self.last_scan_at = now
        # RF-DETR source_image is RGB; FastALPR expects OpenCV-style BGR input.
        results = self.engine.predict(np.ascontiguousarray(rgb_frame[:, :, ::-1]))
        tracked = self.tracker.update(self._detections(results), timestamp=now)
        self.histories = {
            key: history
            for key, history in self.histories.items()
            if now - history.last_seen < self.config.lost_track_seconds
        }
        self.recent_plates = {
            text: emitted_at
            for text, emitted_at in self.recent_plates.items()
            if now - emitted_at < self.config.cooldown_seconds
        }
        events = []
        for index, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id < 0:
                continue
            tracker_id = int(tracker_id)
            history = self.histories.setdefault(tracker_id, PlateHistory(last_seen=now))
            history.last_seen = now
            while (
                history.readings
                and now - history.readings[0][0]
                > self.config.confirmation_window_seconds
            ):
                history.readings.popleft()
            text = str(tracked.data["plate"][index])
            if not text or history.emitted:
                continue
            confidence = float(tracked.data["ocr_confidence"][index])
            history.readings.append((now, text, confidence))
            winner, count = Counter(
                reading[1] for reading in history.readings
            ).most_common(1)[0]
            if (
                count < self.config.minimum_readings
                or count / len(history.readings) < self.config.agreement
            ):
                continue
            # A stationary track must not re-emit when the cross-track cooldown expires.
            history.emitted = True
            if winner in self.recent_plates:
                continue
            self.recent_plates[winner] = now
            events.append(
                {
                    "schema_version": 1,
                    "event_id": uuid.uuid4().hex,
                    "plate": winner,
                    "recognized_at": datetime.now(UTC).isoformat(),
                    "tracker_id": tracker_id,
                    "ocr_confidence": max(
                        reading[2]
                        for reading in history.readings
                        if reading[1] == winner
                    ),
                    "detection_confidence": float(tracked.confidence[index]),
                    "box_xyxy": tracked.xyxy[index].tolist(),
                    "matching_readings": count,
                    "detector_model": self.config.detector_model,
                    "ocr_model": self.config.ocr_model,
                }
            )
        # Keep consensus state bounded even on very busy or noisy input.
        while len(self.histories) > 256:
            del self.histories[
                min(self.histories, key=lambda key: self.histories[key].last_seen)
            ]
        while len(self.recent_plates) > 1024:
            del self.recent_plates[min(self.recent_plates, key=self.recent_plates.get)]
        return events
