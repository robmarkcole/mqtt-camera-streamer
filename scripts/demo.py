"""Run real ONNX inference on stock photos under separate, explicitly demo MQTT IDs."""

import argparse
import hashlib
import json
import math
import re
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from media import annotated_image, jpeg_image
from onnx_camera import (
    HomeAssistantPublisher,
    PredictionProcessor,
    create_predictor,
    load_config,
)
from PIL import Image
from plates import PlateProcessor, create_alpr
from vehicle_events import enrich_plate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/onnx.yml")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=[
            Path("examples/images/plate-sample.png"),
            Path("examples/images/rome-parking.jpg"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("work/stock-demo"))
    parser.add_argument(
        "--hold",
        type=float,
        default=60,
        help="Seconds to serve paused demo results for dashboard viewing",
    )
    args = parser.parse_args()
    if not math.isfinite(args.hold) or args.hold < 0:
        parser.error("--hold must be nonnegative")
    base = load_config(args.config)
    base.demo = True
    base.onnx_threads = min(base.onnx_threads, 2)
    base.lpr.enabled = True
    base.lpr.num_threads = min(base.lpr.num_threads, 2)
    base.tracking.enabled = True
    base.images.transport = "mqtt"
    base.storage.enabled = False
    base.stale_seconds = max(300, base.stale_seconds, math.ceil(args.hold) + 60)
    base.lpr.watched_plates.setdefault("5AU5341", "Demo sample")
    args.output.mkdir(parents=True, exist_ok=True)
    detector = create_predictor(base)
    engine = create_alpr(base.lpr)
    records = []
    publishers = []
    with ExitStack() as stack:
        for index, path in enumerate(args.images):
            config = base.model_copy(deep=True)
            stem = re.sub("[^a-z0-9_]", "_", path.stem.lower())[:40]
            config.camera_id = f"demo_{index}_{stem}"
            config.name = (
                "Stock plate demo" if index == 0 else f"Stock street demo {index}"
            )
            processor = PredictionProcessor(config)
            plates = PlateProcessor(config.lpr, engine=engine)
            publisher = stack.enter_context(HomeAssistantPublisher(config))
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            recognized = []
            vehicle_events = []
            best_jpeg = None
            for sequence in range(1, 7):
                started = time.monotonic()
                detections = detector.predict_frame(
                    rgb,
                    threshold=config.tracking.recovery_threshold,
                    include_source_image=True,
                )
                payload = processor.process(
                    detections, sequence, (time.monotonic() - started) * 1000
                )
                publisher.publish(payload)
                vehicle_events.extend(payload["events"])
                found = [
                    enrich_plate(
                        event, payload, rgb.shape, config.classes.get("vehicle", [])
                    )
                    for event in plates.process(rgb, force=True)
                ]
                for event in found:
                    publisher.publish_plate(event)
                if found or not recognized:
                    best_jpeg = jpeg_image(
                        annotated_image(rgb, payload, found), config.images
                    )
                    publisher.publish_snapshot(best_jpeg)
                recognized.extend(found)
                time.sleep(max(0, 0.75 - (time.monotonic() - started)))
            (args.output / f"{stem}.jpg").write_bytes(best_jpeg)
            record = {
                "demo": True,
                "source": path.name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "camera_id": config.camera_id,
                "name": config.name,
                "preset": config.preset,
                "model": config.model_path.name,
                "prediction": payload,
                "vehicle_events": vehicle_events,
                "plate_events": recognized,
                "note": "Real inference on six repeated copies of a stock image, not a live vehicle arrival or movement",
            }
            records.append(record)
            publishers.append((publisher, payload, processor, rgb))
            print(
                json.dumps(
                    {
                        "source": path.name,
                        "counts": payload["counts"],
                        "plate_events": recognized,
                        "vehicle_event_count": len(vehicle_events),
                    }
                ),
                flush=True,
            )
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        until = time.monotonic() + args.hold
        while time.monotonic() < until:
            # Keep a paused, labelled still-image demo available; do not invent new predictions/events.
            for publisher, payload, processor, rgb in publishers:
                publisher.publish(dict(payload, events=[]))
            time.sleep(min(5, max(0, until - time.monotonic())))
        for publisher, payload, processor, rgb in publishers:
            empty = detector.predict_frame(
                np.zeros_like(rgb), threshold=base.tracking.recovery_threshold
            )
            time.sleep(base.tracking.lost_track_seconds + 0.1)
            publisher.publish(processor.process(empty, payload["sequence"] + 1, 0))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
