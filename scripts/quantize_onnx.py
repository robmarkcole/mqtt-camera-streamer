"""Build an experimental mixed-precision INT8 RF-DETR copy on a development host."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from onnx_detector import DetectionMetadata, ONNXDetector, preprocess
from PIL import Image
from vendor.rfdetr.quantization import nodes_to_exclude

LOG = logging.getLogger(__name__)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_list(path, limit=None):
    path = Path(path).resolve()
    paths = [
        (path.parent / line.strip()).resolve()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if limit is not None:
        paths = paths[:limit]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Calibration list must contain nonempty, distinct image paths")
    for image in paths:
        if not image.is_file():
            raise FileNotFoundError(image)
    return paths


class ImageCalibrationReader:
    """Stream one image at a time using exactly the deployed NumPy preprocessing."""

    def __init__(self, paths, metadata, preprocess_fn=preprocess):
        self.paths = paths
        self.metadata = metadata
        self.preprocess = preprocess_fn
        self.rewind()

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        return {self.metadata.input_name: self.preprocess(rgb, self.metadata)}

    def rewind(self):
        self.iterator = iter(self.paths)


def quantize(model_path, output, mode="static", calibration=None, limit=100):
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source, output = Path(model_path).resolve(), Path(output).resolve()
    if source.suffix.lower() != ".onnx" or output.suffix.lower() != ".onnx":
        raise ValueError("Source and output must be ONNX artifacts")
    if source == output:
        raise ValueError("Quantization must not overwrite the FP32 source")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Output already exists: {output}")
    if mode not in ("static", "dynamic") or limit < 1:
        raise ValueError("mode must be static/dynamic and limit must be positive")
    graph = onnx.load(source)
    props = {item.key: item.value for item in graph.metadata_props}
    if "rfdetr_quantization" in props:
        raise ValueError(
            "Use the original FP32 model, not an already quantized artifact"
        )
    metadata = DetectionMetadata.model_validate_json(props["rfdetr_inference"])
    if metadata.input_dtype != "float32":
        raise ValueError("Quantization requires the original FP32 export")
    paths = image_list(calibration, limit) if calibration is not None else []
    if mode == "static" and not paths:
        raise ValueError("Static INT8 requires representative calibration images")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(source)
    with tempfile.TemporaryDirectory(
        prefix=".quantize-", dir=output.parent
    ) as directory:
        prepared = Path(directory) / "prepared.onnx"
        staged = Path(directory) / "quantized.onnx"
        LOG.info("Preparing ONNX graph for %s INT8", mode)
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
        prepared_graph = onnx.load(prepared)
        excluded = nodes_to_exclude(prepared_graph.graph)
        common = {
            "op_types_to_quantize": ["MatMul", "Gemm"],
            "nodes_to_exclude": excluded,
            "weight_type": QuantType.QInt8,
        }
        LOG.info(
            "Quantizing linear layers; keeping %d attention/head nodes in float",
            len(excluded),
        )
        if mode == "static":
            # Match Roboflow's RF-DETR 1.12.1 recipe: per-tensor S8S8, float outputs.
            quantize_static(
                str(prepared),
                str(staged),
                ImageCalibrationReader(paths, metadata),
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                per_channel=False,
                activation_type=QuantType.QInt8,
                extra_options={
                    "ActivationSymmetric": False,
                    "WeightSymmetric": True,
                    "OpTypesToExcludeOutputQuantization": ["MatMul", "Gemm"],
                },
                **common,
            )
        else:
            quantize_dynamic(
                str(prepared),
                str(staged),
                per_channel=False,
                extra_options={"MatMulConstBOnly": True},
                **common,
            )
        quantized = onnx.load(staged)
        counts = Counter(node.op_type for node in quantized.graph.node)
        if not (counts["QuantizeLinear"] or counts["DynamicQuantizeLinear"]):
            raise ValueError("No quantized operators produced; refusing an INT8 label")
        manifest = {
            "schema_version": 1,
            "mode": mode,
            "precision": "mixed-fp32-int8",
            "source_sha256": source_hash,
            "recipe": "rfdetr-1.12.1-linear-only",
            "weights_per_channel": False,
            "calibration_method": "MinMax" if mode == "static" else None,
            "calibration_images": [
                {"name": p.name, "sha256": file_hash(p)} for p in paths
            ]
            if mode == "static"
            else [],
            "excluded_nodes": excluded,
            "operator_counts": dict(counts),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("onnx", "onnxruntime", "numpy")
            },
        }
        # Preserve the original label/preprocessing contract and record precision separately.
        props["rfdetr_quantization"] = json.dumps(
            {
                key: manifest[key]
                for key in ("mode", "precision", "source_sha256", "recipe")
            }
        )
        onnx.helper.set_model_props(quantized, props)
        onnx.checker.check_model(quantized)
        onnx.save(quantized, staged)
        candidate = ONNXDetector(staged, threads=2)
        if paths:
            with Image.open(paths[0]) as image:
                rgb = np.array(image.convert("RGB"))
        else:
            rgb = np.zeros((*metadata.input_shape[2:], 3), dtype=np.uint8)
        candidate.predict_frame(
            rgb
        )  # Check executable CPU kernels, not just graph syntax.
        manifest.update(
            output_sha256=file_hash(staged),
            source_bytes=source.stat().st_size,
            output_bytes=staged.stat().st_size,
        )
        os.replace(staged, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.mode == "static" and args.calibration_list is None:
        parser.error("static mode requires --calibration-list")
    report = quantize(
        args.model, args.output, args.mode, args.calibration_list, args.limit
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
