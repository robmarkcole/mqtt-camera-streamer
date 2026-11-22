"""Validate calibration isolation, matching preprocessing and labelled evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
from evaluate_onnx import evaluate, yolo_targets
from onnx_detector import ONNXDetector, preprocess
from PIL import Image
from quantize_onnx import ImageCalibrationReader, file_hash, image_list

MODEL = Path(__file__).parent / "fixtures/rfdetr-test.onnx"


def test_image_list_is_relative_distinct_and_nonempty(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (7, 9)).save(image)
    listing = tmp_path / "images.txt"
    listing.write_text("frame.png\n\n")
    assert image_list(listing) == [image]
    listing.write_text("frame.png\nframe.png\n")
    with pytest.raises(ValueError, match="distinct"):
        image_list(listing)
    listing.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        image_list(listing)


def test_calibration_uses_runtime_preprocessing_and_rewinds(tmp_path):
    image = tmp_path / "frame.png"
    rgb = np.full((8, 7, 3), [42, 125, 255], dtype=np.uint8)
    Image.fromarray(rgb).save(image)
    contract = ONNXDetector(MODEL).metadata
    reader = ImageCalibrationReader([image], contract)
    np.testing.assert_array_equal(reader.get_next()["input"], preprocess(rgb, contract))
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next()["input"].shape == (1, 3, 4, 4)


def test_supervision_evaluation_with_real_cpu_predictions(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (100, 50)).save(image)
    (tmp_path / "frame.txt").write_text("0 .5 .5 1 1\n1 .5 .5 1 1\n1 .25 .25 .5 .5\n")
    listing = tmp_path / "images.txt"
    listing.write_text("frame.png\n")
    report = evaluate(MODEL, listing, tmp_path, ["person", "car"])
    assert report["images"] == 1
    assert report["mAP50"] == pytest.approx(1)
    assert report["mAP50_95"] == pytest.approx(1)
    assert report["person_vehicle_mAP50"] == pytest.approx(1)


def test_evaluation_refuses_calibration_content_under_another_name(tmp_path):
    first, second = tmp_path / "one.png", tmp_path / "two.png"
    Image.new("RGB", (10, 20)).save(first)
    second.write_bytes(first.read_bytes())
    calibration, evaluation = tmp_path / "calibration.txt", tmp_path / "evaluation.txt"
    calibration.write_text("one.png\n")
    evaluation.write_text("two.png\n")
    with pytest.raises(ValueError, match="overlap"):
        evaluate(MODEL, evaluation, tmp_path, ["person", "car"], calibration)


def test_evaluation_checks_quantization_manifest_hash(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(MODEL.read_bytes())
    model.with_suffix(".json").write_text(json.dumps({"output_sha256": "wrong"}))
    image = tmp_path / "one.png"
    Image.new("RGB", (10, 20)).save(image)
    listing = tmp_path / "evaluation.txt"
    listing.write_text("one.png\n")
    with pytest.raises(ValueError, match="model hash"):
        evaluate(model, listing, tmp_path, ["person", "car"])
    assert file_hash(model) == file_hash(MODEL)


def test_missing_and_invalid_yolo_labels_fail(tmp_path):
    path = tmp_path / "image.txt"
    with pytest.raises(FileNotFoundError):
        yolo_targets(path, 100, 50, 2)
    for line in ["0 .5 .5 2 1", "2 .5 .5 1 1", "0 nan .5 1 1", "0 .5 .5"]:
        path.write_text(line)
        with pytest.raises(ValueError):
            yolo_targets(path, 100, 50, 2)


def test_benchmark_comparison_uses_fresh_serial_processes_and_reverse_order(
    tmp_path, monkeypatch
):
    from benchmark_models import compare

    config = tmp_path / "config.yml"
    config.write_text("test: true")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    models = [tmp_path / "fp32.onnx", tmp_path / "int8.onnx"]
    for model in models:
        model.write_bytes(MODEL.read_bytes())
    calls = []

    def run(command, **kwargs):
        model = Path(command[command.index("--model") + 1])
        calls.append(model.name)
        report = {
            "measured_frames": 2,
            "model_sha256": file_hash(model),
            "model_bytes": model.stat().st_size,
            "quantization": None,
            "pipeline_fps": 1,
            "stages": {"frame_total": {"p95_ms": 1000}},
            "peak_process_rss_mib": 50,
        }
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report))

    monkeypatch.setattr("benchmark_models.subprocess.run", run)
    report = compare(
        config, models, str(source), tmp_path / "results", frames=2, repeats=2
    )
    assert calls == ["fp32.onnx", "int8.onnx", "int8.onnx", "fp32.onnx"]
    assert len(report["models"][0]["runs"]) == 2
    assert report["source_sha256"] == file_hash(source)
    assert (tmp_path / "results/summary.json").is_file()


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_real_quantization_preserves_contract_and_original(tmp_path, mode):
    onnx = pytest.importorskip("onnx", reason="Run with the optional quantize extra")
    from onnx import helper, numpy_helper
    from quantize_onnx import quantize

    original = onnx.load(MODEL)
    rng = np.random.default_rng(42)
    weights = [
        numpy_helper.from_array(
            (rng.normal(size=shape) * 0.01).astype(np.float32), name
        )
        for name, shape in [
            ("body_w", (48, 8)),
            ("boxes_w", (8, 8)),
            ("classes_w", (8, 6)),
        ]
    ]
    weights += [
        numpy_helper.from_array(np.array(shape, dtype=np.int64), name)
        for name, shape in [("boxes_shape", (1, 2, 4)), ("classes_shape", (1, 2, 3))]
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "body_w"], ["hidden"], name="body"),
            helper.make_node("Relu", ["hidden"], ["activated"]),
            helper.make_node(
                "MatMul", ["activated", "boxes_w"], ["boxes"], name="boxes_head"
            ),
            helper.make_node(
                "MatMul", ["activated", "classes_w"], ["classes"], name="classes_head"
            ),
            helper.make_node("Reshape", ["boxes", "boxes_shape"], ["dets"]),
            helper.make_node("Reshape", ["classes", "classes_shape"], ["labels"]),
        ],
        "tiny-rfdetr-quantization",
        list(original.graph.input),
        list(original.graph.output),
        weights,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    model.metadata_props.extend(original.metadata_props)
    source, output = tmp_path / "source.onnx", tmp_path / "int8.onnx"
    onnx.save(model, source)
    before = file_hash(source)
    paths = []
    for index, color in enumerate([(10, 20, 30), (255, 230, 180)]):
        image = tmp_path / f"calib-{index}.png"
        Image.new("RGB", (8, 8), color).save(image)
        paths.append(image)
    listing = tmp_path / "calibration.txt"
    listing.write_text("\n".join(str(path) for path in paths))
    manifest = quantize(source, output, mode, listing if mode == "static" else None)
    assert file_hash(source) == before
    assert manifest["output_sha256"] == file_hash(output)
    assert manifest["weights_per_channel"] is False
    assert set(manifest["excluded_nodes"]) == {"boxes_head", "classes_head"}
    assert ONNXDetector(output).metadata == ONNXDetector(source).metadata
    assert output.with_suffix(".json").is_file()
    assert len(manifest["calibration_images"]) == (2 if mode == "static" else 0)

    with pytest.raises(ValueError, match="overwrite"):
        quantize(source, source, mode, listing if mode == "static" else None)
    with pytest.raises(FileExistsError):
        quantize(source, output, mode, listing if mode == "static" else None)
    with pytest.raises(ValueError, match="original FP32"):
        quantize(
            output, tmp_path / "twice.onnx", mode, listing if mode == "static" else None
        )

    with pytest.raises(ValueError, match="ONNX artifacts"):
        quantize(
            source,
            tmp_path / "invalid.bin",
            mode,
            listing if mode == "static" else None,
        )
