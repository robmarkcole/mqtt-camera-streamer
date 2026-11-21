"""Check the real YOLOX contract, letterbox geometry, NMS and Torch-free path."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx_camera import create_predictor
from onnx_detector import load_detector
from yolox_detector import YOLOXDetector, YOLOXMetadata, preprocess_yolox

MODEL = Path(__file__).parent / "fixtures/yolox-test.onnx"


@pytest.fixture
def metadata():
    return YOLOXMetadata.model_validate_json(
        ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        .get_modelmeta()
        .custom_metadata_map["yolox_inference"]
    )


def test_real_cpu_decode_objectness_class_nms_and_original_rgb():
    detector = load_detector(MODEL)
    assert detector.family == "yolox"
    rgb = np.full((32, 64, 3), [12, 24, 48], dtype=np.uint8)
    result = detector.predict_frame(rgb, include_source_image=True)
    np.testing.assert_array_equal(result.class_id, [0, 1])
    np.testing.assert_array_equal(result.data["class_name"], ["person", "car"])
    np.testing.assert_allclose(result.confidence, [0.81, 0.64], atol=1e-6)
    np.testing.assert_allclose(
        result.xyxy, [[0, 0, 32, 32], [40, 8, 56, 24]], atol=1e-5
    )
    assert result.metadata["source_image"] is rgb
    np.testing.assert_array_equal(detector.predict_frame(rgb).xyxy, result.xyxy)
    np.testing.assert_array_equal(rgb[0, 0], [12, 24, 48])
    empty = detector.predict_frame(rgb, threshold=1, include_source_image=True)
    assert len(empty) == 0 and empty.data["class_name"].shape == (0,)
    assert empty.metadata["source_image"] is rgb


def test_bgr_raw_pixels_top_left_letterbox_and_no_normalization(metadata):
    rgb = np.full((16, 32, 3), [12, 24, 48], dtype=np.uint8)
    tensor, ratio = preprocess_yolox(rgb, metadata)
    assert ratio == 1 and tensor.shape == (1, 3, 32, 32) and tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [48, 24, 12])
    np.testing.assert_array_equal(tensor[0, :, 16:, 0], np.full((3, 16), 114))
    assert tensor.flags.c_contiguous
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess_yolox(rgb.astype(float), metadata)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"decode_in_graph": True},
        {"strides": [8, 16, 64]},
        {"input_shape": [1, 3, 33, 32]},
        {"input_dtype": "float16"},
        {"preprocessing": "rgb-imagenet"},
        {"class_names": ["car", "car"]},
        {"class_names": [""]},
        {"nms_class_agnostic": False},
    ],
)
def test_unsupported_contracts_fail(metadata, update):
    with pytest.raises(ValueError):
        YOLOXMetadata.model_validate(dict(metadata.model_dump(), **update))


def test_factory_creates_one_session_and_preserves_rfdetr(monkeypatch):
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    calls = []

    def create(*args):
        calls.append(args)
        return session

    monkeypatch.setattr("onnx_detector.create_cpu_session", create)
    detector = load_detector(MODEL, threads=4)
    assert detector.session is session and len(calls) == 1
    monkeypatch.undo()
    assert (
        load_detector(Path(__file__).parent / "fixtures/rfdetr-test.onnx").family
        == "rfdetr"
    )


def test_config_needs_only_model_path_change(config):
    config.model_path = MODEL
    config.classes["vehicle"] = ["car"]
    assert create_predictor(config).family == "yolox"


def test_missing_and_ambiguous_metadata_are_rejected(monkeypatch):
    from types import SimpleNamespace

    for props in [{}, {"rfdetr_inference": "{}", "yolox_inference": "{}"}]:
        session = SimpleNamespace(
            get_modelmeta=lambda props=props: SimpleNamespace(custom_metadata_map=props)
        )
        monkeypatch.setattr(
            "onnx_detector.create_cpu_session", lambda *a, session=session: session
        )
        with pytest.raises(ValueError, match="exactly one"):
            load_detector(MODEL)


def test_export_stack_not_imported_by_yolox_runtime():
    code = """
import sys,importlib.abc
class BlockTraining(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','torchvision','yolox','rfdetr','onnx','transformers'}:
   raise AssertionError('Training import: '+fullname)
sys.meta_path.insert(0,BlockTraining())
import onnx_camera,numpy as np
from onnx_detector import load_detector
model=load_detector(sys.argv[1])
assert model.family=='yolox'
assert len(model.predict_frame(np.zeros((32,64,3),dtype=np.uint8)))==2
"""
    subprocess.run(
        [sys.executable, "-c", code, str(MODEL.resolve())],
        cwd=Path(__file__).resolve().parents[1] / "scripts",
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("part", ["input", "output"])
def test_graph_metadata_mismatch_is_rejected(part):
    from types import SimpleNamespace

    real = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = (
        real.get_inputs()
        if part != "input"
        else [
            SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 64, 64])
        ]
    )
    outputs = (
        real.get_outputs()
        if part != "output"
        else [SimpleNamespace(name="output", type="tensor(float)", shape=[1, 21, 85])]
    )
    fake = SimpleNamespace(
        get_modelmeta=real.get_modelmeta,
        get_inputs=lambda: inputs,
        get_outputs=lambda: outputs,
    )
    with pytest.raises(ValueError, match="graph " + part):
        YOLOXDetector(MODEL, session=fake)


@pytest.mark.parametrize("fault", ["probability", "nonfinite", "shape"])
def test_invalid_raw_outputs_fail_instead_of_silent_decoding(fault):
    from types import SimpleNamespace

    real = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    raw = np.zeros((1, 21, 7), dtype=np.float32)
    if fault == "probability":
        raw[0, 0, 4] = -1
    elif fault == "nonfinite":
        raw[0, 0, 0] = np.nan
    else:
        raw = raw[:, :20]
    fake = SimpleNamespace(
        get_modelmeta=real.get_modelmeta,
        get_inputs=real.get_inputs,
        get_outputs=real.get_outputs,
        run=lambda *a: [raw],
    )
    with pytest.raises(ValueError):
        YOLOXDetector(MODEL, session=fake).predict_frame(
            np.zeros((32, 64, 3), dtype=np.uint8)
        )
