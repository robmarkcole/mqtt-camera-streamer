"""Check real ONNX execution, schema/labels, decode and Torch-free imports."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
from onnx_detector import DetectionMetadata, ONNXDetector, preprocess
from vendor.rfdetr.decode import decode_detections

MODEL = Path(__file__).parent / "fixtures/rfdetr-test.onnx"


@pytest.fixture
def metadata():
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return json.loads(session.get_modelmeta().custom_metadata_map["rfdetr_inference"])


def test_real_cpu_session_multiclass_decode_and_source_metadata():
    detector = ONNXDetector(MODEL)
    rgb = np.zeros((50, 100, 3), dtype=np.uint8)
    result = detector.predict_frame(rgb, include_source_image=True)
    np.testing.assert_array_equal(result.class_id, [0, 1, 1])
    np.testing.assert_array_equal(result.data["class_name"], ["person", "car", "car"])
    np.testing.assert_allclose(
        result.xyxy, [[0, 0, 100, 50], [0, 0, 100, 50], [0, 0, 50, 25]]
    )
    assert result.metadata["source_image"] is rgb
    empty = detector.predict_frame(rgb, threshold=1, include_source_image=True)
    assert len(empty) == 0 and empty.xyxy.shape == (0, 4)
    assert empty.metadata["source_image"] is rgb
    assert "source_image" not in detector.predict_frame(rgb).metadata
    assert detector.session.get_providers() == ["CPUExecutionProvider"]


def test_preprocess_color_normalization_and_half_pixel_resize(metadata):
    contract = DetectionMetadata.model_validate(metadata)
    rgb = np.full((9, 7, 3), [255, 0, 128], dtype=np.uint8)
    tensor = preprocess(rgb, contract)
    assert (
        tensor.shape == (1, 3, 4, 4)
        and tensor.dtype == np.float32
        and tensor.flags.c_contiguous
    )
    expected = (np.array([1, 0, 128 / 255]) - np.array(contract.means)) / np.array(
        contract.stds
    )
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6)
    # Downsample a checkerboard: non-antialiased half-pixel 8 -> 4 samples average 2x2.
    checker = ((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[:, :, None], 3, axis=2)
    np.testing.assert_allclose(
        preprocess(image, contract)[0, 0], (0.5 - 0.485) / 0.229, atol=1e-6
    )
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess(rgb.astype(float), contract)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"task": "segment"},
        {"resize_antialias": True},
        {"input_shape": [2, 3, 4, 4]},
        {"input_layout": "NHWC"},
        {"stds": [0, 1, 1]},
        {"means": [float("nan"), 0, 0]},
        {"pixel_scale": 1},
        {"class_id_to_name": {"9": "person"}},
        {"outputs": {"pred_boxes": "dets", "pred_logits": "dets"}},
    ],
)
def test_unsupported_metadata_fails_explicitly(metadata, update):
    with pytest.raises(ValueError):
        DetectionMetadata.model_validate(dict(metadata, **update))


def test_sparse_coco_and_contiguous_custom_labels(metadata):
    assert DetectionMetadata.model_validate(metadata).class_id_to_name == {
        0: "person",
        1: "car",
    }
    metadata.update(num_classes=90, class_id_to_name={"1": "person", "90": "car"})
    assert DetectionMetadata.model_validate(metadata).class_id_to_name == {
        1: "person",
        90: "car",
    }
    metadata.update(num_classes=2, class_id_to_name={})
    assert DetectionMetadata.model_validate(metadata).class_id_to_name == {
        0: "person",
        1: "car",
    }


def test_topk_ties_strict_threshold_and_no_last_class_loss():
    boxes = np.array([[0.5, 0.5, 2, 2], [0.25, 0.25, 0.2, 0.2]], dtype=np.float32)
    logits = np.array([[0, 2, 2], [2, -5, -5]], dtype=np.float32)
    result = decode_detections(boxes, logits, (100, 50), threshold=0.5, num_select=3)
    np.testing.assert_array_equal(result.class_id, [1, 2, 0])
    np.testing.assert_allclose(result.xyxy[:2], [[0, 0, 100, 50]] * 2)


def test_graph_interface_mismatch_is_rejected(monkeypatch, metadata):
    metadata["input_name"] = "wrong"
    real = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    fake = SimpleNamespace(
        get_modelmeta=lambda: SimpleNamespace(
            custom_metadata_map={"rfdetr_inference": json.dumps(metadata)}
        ),
        get_inputs=real.get_inputs,
        get_outputs=real.get_outputs,
    )
    monkeypatch.setattr("onnx_detector.ort.InferenceSession", lambda *a, **kw: fake)
    with pytest.raises(ValueError, match="input names"):
        ONNXDetector(MODEL)


def test_missing_metadata_is_rejected(monkeypatch):
    fake = SimpleNamespace(
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={})
    )
    monkeypatch.setattr("onnx_detector.ort.InferenceSession", lambda *a, **kw: fake)
    with pytest.raises(ValueError, match="Missing rfdetr_inference"):
        ONNXDetector(MODEL)


def test_runtime_imports_and_inference_do_not_import_export_stack():
    code = """
import sys, importlib.abc
class BlockExportStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'rfdetr', 'transformers', 'onnx'}:
            raise AssertionError('Runtime attempted export dependency: ' + fullname)
sys.meta_path.insert(0, BlockExportStack())
import onnx_camera, benchmark, plates, fast_alpr, numpy as np
from onnx_detector import ONNXDetector
assert len(ONNXDetector(sys.argv[1]).predict_frame(np.zeros((10, 10, 3), dtype=np.uint8))) == 3
"""
    subprocess.run(
        [sys.executable, "-c", code, str(MODEL.resolve())],
        cwd=Path(__file__).resolve().parents[1] / "scripts",
        check=True,
        capture_output=True,
        text=True,
    )


def test_sparse_class_90_is_a_real_class_not_background(monkeypatch, metadata):
    metadata.update(num_classes=90, class_id_to_name={"1": "person", "90": "car"})
    real = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    boxes = np.full((1, 2, 4), 0.5, dtype=np.float32)
    logits = np.full((1, 2, 91), -20, dtype=np.float32)
    logits[0, 0, 90], logits[0, 1, 1] = 5, 4
    fake = SimpleNamespace(
        get_modelmeta=lambda: SimpleNamespace(
            custom_metadata_map={"rfdetr_inference": json.dumps(metadata)}
        ),
        get_inputs=real.get_inputs,
        get_outputs=lambda: [
            SimpleNamespace(name="dets", shape=[1, 2, 4], type="tensor(float)"),
            SimpleNamespace(name="labels", shape=[1, 2, 91], type="tensor(float)"),
        ],
        run=lambda *args: [boxes, logits],
    )
    monkeypatch.setattr("onnx_detector.ort.InferenceSession", lambda *a, **kw: fake)
    result = ONNXDetector(MODEL).predict_frame(np.zeros((20, 40, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result.class_id, [90, 1])
    np.testing.assert_array_equal(result.data["class_name"], ["car", "person"])
