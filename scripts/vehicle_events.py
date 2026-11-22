"""Small, local vehicle lifecycle events and conservative plate association."""

import uuid
from datetime import UTC, datetime

EVENT_TYPES = [
    "vehicle_detected",
    "vehicle_lost",
    "plate_recognized",
    "watched_plate_detected",
]


def event_payload(kind, **data):
    return {
        **data,
        "schema_version": 1,
        "event_id": uuid.uuid4().hex,
        "event_type": kind,
        "occurred_at": datetime.now(UTC).isoformat(),
    }


class VehicleEvents:
    def __init__(self, config):
        self.labels = set(config.classes.get("vehicle", []))
        self.timeout = config.tracking.lost_track_seconds
        self.tracks = {}

    def reset(self):
        self.tracks.clear()

    def update(self, detections, now):
        events = []
        current = {
            d["tracker_id"]: d
            for d in detections
            if d["label"] in self.labels and d.get("tracker_id") is not None
        }
        for track_id, detection in current.items():
            vehicle = {
                "vehicle_type": detection["label"],
                "vehicle_tracker_id": track_id,
                "vehicle_confidence": detection["confidence"],
                "vehicle_box_xyxy": detection["box_xyxy"],
            }
            if track_id not in self.tracks:
                events.append(event_payload("vehicle_detected", **vehicle))
            self.tracks[track_id] = (now, vehicle)
        for track_id, (last_seen, vehicle) in list(self.tracks.items()):
            if track_id not in current and now - last_seen >= self.timeout:
                events.append(event_payload("vehicle_lost", **vehicle))
                del self.tracks[track_id]
        # Bound memory on busy scenes; eviction is not evidence of a departure.
        while len(self.tracks) > 2048:
            del self.tracks[min(self.tracks, key=lambda key: self.tracks[key][0])]
        return events


def enrich_plate(event, payload, shape, labels):
    """Associate only when the plate centre is inside a confirmed vehicle box."""
    height, width = shape[:2]
    x1, y1, x2, y2 = event["box_xyxy"]
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = []
    for detection in payload["detections"]:
        left, top, right, bottom = detection["box_xyxy"]
        if (
            detection["label"] in labels
            and detection.get("tracker_id") is not None
            and left <= x <= right
            and top <= y <= bottom
        ):
            candidates.append(((right - left) * (bottom - top), detection))
    vehicle = min(candidates, key=lambda item: item[0])[1] if candidates else None
    return dict(
        event,
        event_type="plate_recognized",
        occurred_at=event["recognized_at"],
        confidence=event["ocr_confidence"],
        box_x_centre=x,
        box_y_centre=y,
        box_centre_normalized=[x / width, y / height],
        vehicle_type=vehicle["label"] if vehicle else None,
        vehicle_tracker_id=vehicle["tracker_id"] if vehicle else None,
        vehicle_confidence=vehicle["confidence"] if vehicle else None,
        vehicle_box_xyxy=vehicle["box_xyxy"] if vehicle else None,
    )
