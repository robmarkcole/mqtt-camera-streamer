"""Cover preset resolution, metadata mismatches and explicit experiment overrides."""

import json
import sys
from pathlib import Path

import benchmark
import numpy as np
import pytest
from onnx_camera import create_predictor, load_config, prediction_payload
from onnx_detector import MODEL_DIRECTORY, PRESETS, validate_preset

YAML = """camera_id: test_camera
name: Test camera
classes: {person: [person], vehicle: [car]}
mqtt: {broker: localhost}
"""
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("preset", ["speed", "accuracy"])
def test_presets_resolve_under_installation_not_external_config(tmp_path, preset):
    path = tmp_path / "camera.yml"
    path.write_text(YAML + f"preset: {preset}\n")
    config = load_config(path)
    assert config.preset == preset
    assert config.model_path == MODEL_DIRECTORY / PRESETS[preset]["path"]
    assert config.tracking.enabled and config.images.transport == "mqtt"
    assert config.presence.hold_seconds == 5


def test_speed_default_and_legacy_relative_model_path(tmp_path):
    path = tmp_path / "camera.yml"
    path.write_text(YAML)
    assert load_config(path).preset == "speed"
    path.write_text(YAML + "model_path: custom.onnx\n")
    config = load_config(path)
    assert config.preset is None
    assert config.model_path == tmp_path / "custom.onnx"
    path.write_text(YAML + "preset: turbo\n")
    with pytest.raises(ValueError, match="preset"):
        load_config(path)


def test_real_speed_pipeline_and_wrong_preset_model(tmp_path):
    path = tmp_path / "camera.yml"
    path.write_text(
        YAML + f"preset: speed\nmodel_path: {FIXTURES / 'yolox-test.onnx'}\n"
    )
    config = load_config(path)
    detector = create_predictor(config)
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    prediction = detector.predict_frame(rgb, include_source_image=True)
    assert prediction.metadata["source_image"] is rgb
    payload = prediction_payload(prediction, config, 1, 0)
    assert payload["preset"] == "speed" and payload["counts"]["vehicle"] == 1
    config.model_path = FIXTURES / "rfdetr-test.onnx"
    with pytest.raises(ValueError, match="requires yolox-nano FP32"):
        create_predictor(config)


@pytest.mark.parametrize("preset", ["speed", "accuracy"])
def test_preset_rejects_wrong_family_variant_or_precision(preset):
    family = PRESETS[preset]["family"]
    metadata = {"variant": PRESETS[preset]["variant"], "input_dtype": "float32"}
    props = {family + "_inference": json.dumps(metadata)}
    if preset == "accuracy":
        props["rfdetr_quantization"] = json.dumps(
            {"mode": "static", "precision": "mixed-fp32-int8"}
        )
    validate_preset(preset, props)
    for change in [{"variant": "other"}, {"input_dtype": "float16"}]:
        bad = dict(props)
        bad[family + "_inference"] = json.dumps(dict(metadata, **change))
        with pytest.raises(ValueError, match="requires"):
            validate_preset(preset, bad)
    bad = dict(props)
    if preset == "speed":
        bad["yolox_quantization"] = '{"mode":"static"}'
    else:
        bad.pop("rfdetr_quantization")
    with pytest.raises(ValueError, match="requires"):
        validate_preset(preset, bad)
    if preset == "accuracy":
        bad = dict(props)
        bad["rfdetr_quantization"] = '{"mode":"dynamic","precision":"mixed-fp32-int8"}'
        with pytest.raises(ValueError, match="requires"):
            validate_preset(preset, bad)


def test_benchmark_model_override_clears_preset(tmp_path, monkeypatch, capsys):
    path = tmp_path / "camera.yml"
    path.write_text(YAML + "preset: accuracy\n")
    model = FIXTURES / "yolox-test.onnx"

    def run(config, *args):
        assert config.preset is None and config.model_path == model
        return {"preset": config.preset}

    monkeypatch.setattr(benchmark, "benchmark", run)
    monkeypatch.setattr(
        sys, "argv", ["benchmark.py", "--config", str(path), "--model", str(model)]
    )
    benchmark.main()
    assert json.loads(capsys.readouterr().out)["preset"] is None
