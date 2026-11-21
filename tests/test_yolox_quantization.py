"""Exercise calibrated Conv quantization, original RGB contract and manifests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from quantize_onnx import ImageCalibrationReader, file_hash
from quantize_yolox import calibration_tensor
from yolox_detector import YOLOXDetector, preprocess_yolox

FIXTURE = Path(__file__).parent / "fixtures/yolox-test.onnx"


def test_calibration_matches_bgr_runtime_and_rewinds(tmp_path):
    rgb = np.full((16, 32, 3), [7, 31, 197], dtype=np.uint8)
    path = tmp_path / "calibration.png"
    Image.fromarray(rgb).save(path)
    metadata = YOLOXDetector(FIXTURE).metadata
    reader = ImageCalibrationReader([path], metadata, calibration_tensor)
    np.testing.assert_array_equal(
        reader.get_next()["images"], preprocess_yolox(rgb, metadata)[0]
    )
    assert reader.get_next() is None
    reader.rewind()
    np.testing.assert_array_equal(reader.get_next()["images"][0, :, 0, 0], [197, 31, 7])


@pytest.mark.parametrize("keep_depthwise", [False, True])
def test_real_conv_quantization_keeps_float_heads_and_contract(
    tmp_path, keep_depthwise
):
    onnx = pytest.importorskip("onnx", reason="Run with the optional quantize extra")
    from evaluate_onnx import evaluate
    from onnx import helper, numpy_helper
    from quantize_yolox import quantize

    original = onnx.load(FIXTURE)
    rng = np.random.default_rng(42)
    nodes = [
        helper.make_node(
            "Conv",
            ["images", "depthwise_w"],
            ["depthwise"],
            name="depthwise",
            kernel_shape=[1, 1],
            group=3,
        ),
        helper.make_node(
            "Conv",
            ["depthwise", "body_w"],
            ["hidden"],
            name="body",
            kernel_shape=[1, 1],
        ),
        helper.make_node("Relu", ["hidden"], ["features"]),
    ]
    weights = [
        numpy_helper.from_array(
            np.full((3, 1, 1, 1), 0.1, dtype=np.float32), "depthwise_w"
        ),
        numpy_helper.from_array(
            (rng.normal(size=(7, 3, 1, 1)) * 0.01).astype(np.float32), "body_w"
        ),
    ]
    heads = []
    for kind in ["reg", "obj", "cls"]:
        for level in range(3):
            name = f"/head/{kind}_preds.{level}/Conv"
            out = f"{kind}{level}"
            heads.append(out)
            weights.append(
                numpy_helper.from_array(
                    np.ones((7, 7, 1, 1), dtype=np.float32) * 0.001, out + "_w"
                )
            )
            nodes.append(
                helper.make_node(
                    "Conv",
                    ["features", out + "_w"],
                    [out],
                    name=name,
                    kernel_shape=[1, 1],
                )
            )
    weights.extend(
        [
            numpy_helper.from_array(np.array([1, 1, 7], dtype=np.int64), "shape"),
            numpy_helper.from_array(np.array([1, 21, 1], dtype=np.int64), "repeats"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Sum", heads, ["sum"]),
            helper.make_node("GlobalAveragePool", ["sum"], ["pooled"]),
            helper.make_node("Reshape", ["pooled", "shape"], ["row"]),
            helper.make_node("Tile", ["row", "repeats"], ["raw"]),
            helper.make_node("Sigmoid", ["raw"], ["output"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "tiny-yolox-convs",
        list(original.graph.input),
        list(original.graph.output),
        weights,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    model.metadata_props.extend(original.metadata_props)
    source, output = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    onnx.save(model, source)
    before = file_hash(source)
    images = []
    for index, color in enumerate([(7, 31, 197), (122, 5, 42)]):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (32, 16), color).save(path)
        images.append(path)
    listing = tmp_path / "calibration.txt"
    listing.write_text("\n".join(str(p) for p in images))
    manifest = quantize(
        source,
        output,
        listing,
        method="percentile" if keep_depthwise else "minmax",
        keep_depthwise_float=keep_depthwise,
    )
    assert manifest["calibration_method"] == (
        "Percentile" if keep_depthwise else "MinMax"
    )
    assert file_hash(source) == before
    assert file_hash(output) == manifest["output_sha256"]
    assert manifest["int8_weight_tensors"] == (1 if keep_depthwise else 2)
    assert manifest["weights_per_channel"] is True
    assert len(manifest["excluded_nodes"]) == (10 if keep_depthwise else 9)
    assert ("depthwise" in manifest["excluded_nodes"]) == keep_depthwise
    assert len(manifest["calibration_images"]) == 2
    actual = YOLOXDetector(output)
    assert actual.metadata == YOLOXDetector(source).metadata
    rgb = np.full((16, 32, 3), 50, dtype=np.uint8)
    result = actual.predict_frame(rgb, include_source_image=True)
    assert result.metadata["source_image"] is rgb
    props = actual.session.get_modelmeta().custom_metadata_map
    assert json.loads(props["yolox_quantization"])["source_sha256"] == before
    exported = onnx.load(output)
    for w in exported.graph.initializer:
        if w.name in {name + "_w" for name in heads}:
            assert w.data_type == onnx.TensorProto.FLOAT
    with pytest.raises(ValueError, match="original FP32"):
        quantize(output, tmp_path / "twice.onnx", listing)
    with pytest.raises(ValueError, match="overwrite"):
        quantize(source, source, listing)
    with pytest.raises(FileExistsError):
        quantize(source, output, listing)
    with pytest.raises(ValueError, match="positive"):
        quantize(source, tmp_path / "invalid.onnx", listing, limit=0)

    with pytest.raises(ValueError, match="overlap"):
        evaluate(output, listing, tmp_path, ["person", "car"])
    with pytest.raises(ValueError, match="nine named"):
        quantize(FIXTURE, tmp_path / "unsupported.onnx", listing)
