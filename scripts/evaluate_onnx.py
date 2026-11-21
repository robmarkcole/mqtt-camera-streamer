"""Evaluate ONNX detection exports on labelled YOLO images using Supervision mAP."""

import argparse
import json
from pathlib import Path

import numpy as np
import supervision as sv
from onnx_detector import ONNXDetector
from PIL import Image
from quantize_onnx import file_hash, image_list
from supervision.metrics import MeanAveragePrecision


def yolo_targets(path, width, height, class_count):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file (use an empty file for negatives): {path}"
        )
    boxes, classes = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Expected YOLO detection labels: class cx cy width height")
        label, *box = map(float, fields)
        if (
            not label.is_integer()
            or not 0 <= label < class_count
            or not np.isfinite(box).all()
            or any(not 0 <= value <= 1 for value in box)
        ):
            raise ValueError(f"Invalid class or normalized box in {path}")
        cx, cy, bw, bh = box
        if bw <= 0 or bh <= 0:
            raise ValueError(f"YOLO boxes must have positive area: {path}")
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
        classes.append(int(label))
    return sv.Detections(
        xyxy=np.clip(
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            0,
            [width, height, width, height],
        ),
        class_id=np.array(classes, dtype=int),
    )


def evaluate(
    model,
    images_path,
    labels_dir,
    class_names,
    calibration=None,
    threshold=0.001,
    threads=2,
):
    images = image_list(images_path)
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name for name in class_names)
    ):
        raise ValueError("Class names must be a nonempty JSON list of strings")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Dataset class names must be unique and in YOLO ordinal order")
    hashes = {file_hash(path) for path in images}
    if calibration and hashes.intersection(
        file_hash(path) for path in image_list(calibration)
    ):
        raise ValueError("Evaluation and calibration images overlap")
    manifest_path = Path(model).with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("output_sha256") != file_hash(model):
            raise ValueError("Quantization manifest does not match model hash")
        if hashes.intersection(
            item["sha256"] for item in manifest.get("calibration_images", [])
        ):
            raise ValueError("Evaluation images overlap this model's calibration set")
    detector = ONNXDetector(model, threads=threads)
    ordinal = {name: index for index, name in enumerate(class_names)}
    if set(detector.class_names) - ordinal.keys():
        raise ValueError("Dataset class names do not cover the exported labels")
    all_metric, relevant_metric = MeanAveragePrecision(), MeanAveragePrecision()
    relevant_ids = [
        ordinal[name]
        for name in ("person", "car", "motorcycle", "bus", "truck")
        if name in ordinal
    ]
    counts = []
    for path in images:
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        prediction = detector.predict_frame(rgb, threshold=threshold)
        valid = np.array(
            [name in ordinal for name in prediction.data["class_name"]], dtype=bool
        )
        prediction = prediction[valid]
        prediction.class_id = np.array(
            [ordinal[name] for name in prediction.data["class_name"]], dtype=int
        )
        target = yolo_targets(
            Path(labels_dir) / (path.stem + ".txt"),
            rgb.shape[1],
            rgb.shape[0],
            len(class_names),
        )
        all_metric.update(prediction, target)
        relevant_metric.update(
            prediction[np.isin(prediction.class_id, relevant_ids)],
            target[np.isin(target.class_id, relevant_ids)],
        )
        counts.append(
            {
                "image": path.name,
                "sha256": file_hash(path),
                "targets": len(target),
                "predictions_at_0_5": int(np.sum(prediction.confidence > 0.5)),
            }
        )
    overall, relevant = all_metric.compute(), relevant_metric.compute()
    return {
        "schema_version": 1,
        "model": Path(model).name,
        "model_sha256": file_hash(model),
        "images": len(images),
        "minimum_confidence": threshold,
        "mAP50": float(overall.map50),
        "mAP50_95": float(overall.map50_95),
        "person_vehicle_mAP50": float(relevant.map50),
        "person_vehicle_mAP50_95": float(relevant.map50_95),
        "per_image": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--images", type=Path, required=True, help="Text list of evaluation images"
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--class-names",
        type=Path,
        required=True,
        help="JSON list of class names in dataset ordinal order",
    )
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--dataset-note",
        default="Small labelled-image diagnostic; not a generalization guarantee",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.threads < 1:
        parser.error("threshold must be in [0,1] and threads positive")
    report = evaluate(
        args.model,
        args.images,
        args.labels,
        json.loads(args.class_names.read_text()),
        args.calibration_list,
        args.threshold,
        args.threads,
    )
    report["dataset_note"] = args.dataset_note
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_image"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
