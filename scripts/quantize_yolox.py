"""Create static QDQ INT8 YOLOX copies on a development host, without Torch."""

import argparse
import importlib.metadata
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from quantize_onnx import ImageCalibrationReader, file_hash, image_list
from yolox_detector import YOLOXDetector, YOLOXMetadata, preprocess_yolox

LOG = logging.getLogger(__name__)


def calibration_tensor(rgb, metadata):
    return preprocess_yolox(rgb, metadata)[0]


def quantize(
    model_path,
    output,
    calibration,
    limit=96,
    method="minmax",
    keep_depthwise_float=False,
):
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source, output = Path(model_path).resolve(), Path(output).resolve()
    if source.suffix.lower() != ".onnx" or output.suffix.lower() != ".onnx":
        raise ValueError("Source and output must be ONNX artifacts")
    if source == output:
        raise ValueError("Quantization must not overwrite the FP32 source")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if method not in ("minmax", "percentile"):
        raise ValueError("Calibration method must be minmax or percentile")
    if limit < 1:
        raise ValueError("Calibration limit must be positive")
    graph = onnx.load(source)
    props = {p.key: p.value for p in graph.metadata_props}
    if any(key.endswith("_quantization") for key in props) or any(
        n.op_type
        in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "ConvInteger"}
        for n in graph.graph.node
    ):
        raise ValueError("Use the original FP32 export, not an already quantized model")
    if "yolox_inference" not in props or "rfdetr_inference" in props:
        raise ValueError("Requires an unambiguous yolox_inference export")
    metadata = YOLOXMetadata.model_validate_json(props["yolox_inference"])
    # Validate executable IO before spending time on calibration.
    YOLOXDetector(source, threads=2)
    paths = image_list(calibration, limit)
    # Keep the nine regression/objectness/classification projection layers float.
    # Their low-cost outputs feed the raw decoder and calibrated score thresholds.
    heads = [
        n.name
        for n in graph.graph.node
        if n.op_type == "Conv"
        and any(f"/head/{kind}_preds." in n.name for kind in ("reg", "obj", "cls"))
    ]
    if len(heads) != 9 or len(set(heads)) != 9:
        raise ValueError("Expected nine named YOLOX output-head convolutions")
    depthwise = (
        [
            node.name
            for node in graph.graph.node
            if node.op_type == "Conv"
            and any(a.name == "group" and a.i > 1 for a in node.attribute)
        ]
        if keep_depthwise_float
        else []
    )
    excluded = heads + depthwise
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(source)
    with tempfile.TemporaryDirectory(
        prefix=".quantize-", dir=output.parent
    ) as directory:
        prepared, staged = (
            Path(directory) / "prepared.onnx",
            Path(directory) / "int8.onnx",
        )
        # The export already folds BN. Keep named Conv/SiLU nodes separate so the
        # quantizer can see Conv rather than optimized floating FusedConv kernels.
        quant_pre_process(
            source, prepared, skip_optimization=True, skip_symbolic_shape=True
        )
        LOG.info(
            "Calibrating %s with %d images; keeping %d output heads float",
            metadata.variant,
            len(paths),
            len(heads),
        )
        quantize_static(
            prepared,
            staged,
            ImageCalibrationReader(paths, metadata, calibration_tensor),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=excluded,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax
            if method == "minmax"
            else CalibrationMethod.Percentile,
            calibration_providers=["CPUExecutionProvider"],
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "CalibPercentile": 99.99,
            },
        )
        quantized = onnx.load(staged)
        counts = Counter(n.op_type for n in quantized.graph.node)
        int8_weights = [
            w
            for w in quantized.graph.initializer
            if w.data_type == onnx.TensorProto.INT8 and len(w.dims) >= 2
        ]
        if not counts["QuantizeLinear"] or not int8_weights:
            raise ValueError("No executable INT8 quantization produced")
        manifest = {
            "schema_version": 1,
            "family": "yolox",
            "variant": metadata.variant,
            "mode": "static",
            "precision": "mixed-fp32-int8",
            "source_sha256": source_hash,
            "recipe": "yolox-conv-qdq-s8s8-float-output-heads-v1"
            + ("-float-depthwise" if keep_depthwise_float else ""),
            "weights_per_channel": True,
            "calibration_method": "MinMax" if method == "minmax" else "Percentile",
            "calibration_percentile": 99.99 if method == "percentile" else None,
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "format": "QDQ",
            "calibration_images": [
                {"name": p.name, "sha256": file_hash(p)} for p in paths
            ],
            "excluded_nodes": excluded,
            "depthwise_kept_float": keep_depthwise_float,
            "operator_counts": dict(counts),
            "int8_weight_tensors": len(int8_weights),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("onnx", "onnxruntime", "numpy")
            },
        }
        props["yolox_quantization"] = json.dumps(
            {k: manifest[k] for k in ("mode", "precision", "source_sha256", "recipe")}
        )
        onnx.helper.set_model_props(quantized, props)
        onnx.checker.check_model(quantized)
        onnx.save(quantized, staged)
        candidate = YOLOXDetector(staged, threads=2)
        with Image.open(paths[0]) as image:
            candidate.predict_frame(np.array(image.convert("RGB")))
        if file_hash(source) != source_hash:
            raise ValueError("FP32 source changed during quantization")
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
    parser.add_argument("--calibration-list", required=True, type=Path)
    parser.add_argument("--limit", default=96, type=int)
    parser.add_argument(
        "--calibration-method", choices=("minmax", "percentile"), default="minmax"
    )
    parser.add_argument("--keep-depthwise-float", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(
        json.dumps(
            quantize(
                args.model,
                args.output,
                args.calibration_list,
                args.limit,
                args.calibration_method,
                args.keep_depthwise_float,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
