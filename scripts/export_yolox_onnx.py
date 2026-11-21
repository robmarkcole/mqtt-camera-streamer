"""Export official YOLOX Nano/Tiny on a development host; no Torch on the Pi."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from vendor.yolox import UPSTREAM_COMMIT
from vendor.yolox.coco_classes import COCO_CLASSES
from yolox_detector import YOLOXDetector, YOLOXMetadata, preprocess_yolox


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_revision(path):
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Use the pinned YOLOX source revision {UPSTREAM_COMMIT}")
    subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--", "yolox", "exps"],
        check=True,
    )
    return revision


def verify_parity(
    model, detector, rgb, reference_preproc, reference_decode, reference_nms
):
    import torch

    tensor, ratio = preprocess_yolox(rgb, detector.metadata)
    reference_tensor, reference_ratio = reference_preproc(
        np.ascontiguousarray(rgb[:, :, ::-1]), detector.metadata.input_shape[2:]
    )
    np.testing.assert_array_equal(tensor[0], reference_tensor)
    assert ratio == reference_ratio
    with torch.no_grad():
        expected = model(torch.from_numpy(tensor)).cpu().numpy()
    actual = detector.session.run(
        [detector.metadata.output_name], {detector.metadata.input_name: tensor}
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    predictions = reference_decode(expected.copy(), detector.metadata.input_shape[2:])[
        0
    ]
    boxes = (
        np.concatenate(
            (
                predictions[:, :2] - predictions[:, 2:4] / 2,
                predictions[:, :2] + predictions[:, 2:4] / 2,
            ),
            axis=1,
        )
        / ratio
    )
    selected = reference_nms(
        boxes,
        predictions[:, 4:5] * predictions[:, 5:],
        nms_thr=0.45,
        score_thr=0.1,
        class_agnostic=True,
    )
    result = detector.predict_frame(rgb, threshold=0.1)
    if selected is None:
        selected = np.empty((0, 6))
    height, width = rgb.shape[:2]
    expected_boxes = np.clip(selected[:, :4], 0, [width, height, width, height])
    valid = (expected_boxes[:, 2] > expected_boxes[:, 0]) & (
        expected_boxes[:, 3] > expected_boxes[:, 1]
    )
    selected, expected_boxes = selected[valid], expected_boxes[valid]
    np.testing.assert_array_equal(result.class_id, selected[:, 5].astype(int))
    np.testing.assert_allclose(result.confidence, selected[:, 4], atol=5e-4, rtol=1e-3)
    np.testing.assert_allclose(result.xyxy, expected_boxes, atol=0.25, rtol=1e-3)
    max_box_error = (
        float(np.max(np.abs(result.xyxy - expected_boxes))) if len(result) else 0
    )
    return {
        "detections": len(result),
        "max_raw_absolute_error": float(np.max(np.abs(actual - expected))),
        "max_box_error_pixels": max_box_error,
    }


def export(
    source_tree, checkpoint, output, variant, verify_images=(), class_names=None
):
    import onnx
    import torch

    source_tree, checkpoint, output = (
        Path(source_tree).resolve(),
        Path(checkpoint).resolve(),
        Path(output).resolve(),
    )
    if output.suffix.lower() != ".onnx":
        raise ValueError("Output must have an .onnx suffix")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    revision = source_revision(source_tree)
    sys.path.insert(0, str(source_tree))
    import yolox
    from yolox.exp import get_exp
    from yolox.utils import demo_postprocess, multiclass_nms

    if not Path(yolox.__file__).resolve().is_relative_to(source_tree):
        raise ValueError(
            "Loaded YOLOX package does not belong to the pinned source tree"
        )
    torch.set_num_threads(2)
    classes = list(COCO_CLASSES) if class_names is None else class_names
    exp = get_exp(exp_name=f"yolox-{variant}")
    exp.num_classes = len(classes)
    model = exp.get_model().cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.head.decode_in_inference = False
    shape = (1, 3, *exp.test_size)
    metadata = YOLOXMetadata(
        schema_version=1,
        family="yolox",
        variant=f"yolox-{variant}",
        source_commit=revision,
        checkpoint_sha256=file_hash(checkpoint),
        input_shape=shape,
        input_name="images",
        output_name="output",
        input_dtype="float32",
        input_layout="NCHW",
        preprocessing="bgr-0-255-top-left-letterbox-114-linear",
        decode_in_graph=False,
        strides=(8, 16, 32),
        class_names=classes,
        nms_iou_threshold=0.45,
        nms_class_agnostic=True,
    )
    spec = importlib.util.spec_from_file_location(
        "yolox_reference_preproc", source_tree / "yolox/data/data_augment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".export-") as directory:
        staged = Path(directory) / output.name
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.zeros(shape),
                str(staged),
                input_names=[metadata.input_name],
                output_names=[metadata.output_name],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
        graph = onnx.load(staged)
        onnx.helper.set_model_props(
            graph, {"yolox_inference": metadata.model_dump_json()}
        )
        onnx.checker.check_model(graph)
        onnx.save(graph, staged)
        detector = YOLOXDetector(staged, threads=2)
        parity = [
            dict(
                image="zero-input-smoke",
                **verify_parity(
                    model,
                    detector,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    module.preproc,
                    demo_postprocess,
                    multiclass_nms,
                ),
            )
        ]
        for image_path in verify_images:
            with Image.open(image_path) as image:
                rgb = np.array(image.convert("RGB"))
            parity.append(
                dict(
                    image=Path(image_path).name,
                    sha256=file_hash(image_path),
                    **verify_parity(
                        model,
                        detector,
                        rgb,
                        module.preproc,
                        demo_postprocess,
                        multiclass_nms,
                    ),
                )
            )
        report = {
            "schema_version": 1,
            "family": "yolox",
            "variant": metadata.variant,
            "upstream_commit": revision,
            "checkpoint_sha256": metadata.checkpoint_sha256,
            "output_sha256": file_hash(staged),
            "model_bytes": staged.stat().st_size,
            "input_shape": shape,
            "precision": "fp32",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "onnx", "onnxruntime"]
            },
            "parity": parity,
        }
        os.replace(staged, output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("nano", "tiny"), required=True)
    parser.add_argument("--verify-images", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--class-names", type=Path, help="Ordered JSON labels for a custom checkpoint"
    )
    args = parser.parse_args()
    names = json.loads(args.class_names.read_text()) if args.class_names else None
    report = export(
        args.source_tree,
        args.checkpoint,
        args.output,
        args.variant,
        args.verify_images,
        names,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
