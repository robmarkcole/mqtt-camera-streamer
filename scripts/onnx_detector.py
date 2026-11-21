"""Torch-free RF-DETR detection with embedded metadata and ONNX Runtime CPU.

Preprocessing follows RF-DETR 1.12.1's NumPy fallback; upstream attribution and
Apache license are retained in vendor/rfdetr.
"""

import json
from typing import Literal

import numpy as np
import onnxruntime as ort
import supervision as sv
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from vendor.rfdetr._resize import _bilinear_resize_half_pixel
from vendor.rfdetr.decode import decode_detections


class DetectionMetadata(BaseModel):
    """The supported subset of RF-DETR schema 1; never guess label/layout rules."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    schema_version: Literal[1]
    format: Literal["onnx"]
    task: Literal["detect"]
    input_layout: Literal["NCHW"]
    input_dtype: Literal["float32", "float16"]
    input_shape: tuple[int, int, int, int]
    input_name: str
    outputs: dict[str, str]
    means: list[float]
    stds: list[float]
    pixel_scale: float
    channel_order: Literal["RGB"]
    resize_interpolation: Literal["bilinear"]
    resize_antialias: Literal[False]
    selection_policy: Literal["all_logits"]
    box_format: Literal["cxcywh_normalized"]
    class_names: list[str] = Field(min_length=1)
    class_id_to_name: dict[int, str] = Field(default_factory=dict)
    num_classes: int = Field(ge=1)
    num_select: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_contract(self):
        batch, channels, height, width = self.input_shape
        if batch not in (-1, 1) or channels not in (1, 3) or min(height, width) <= 0:
            raise ValueError(
                "Only batch-one RGB/grayscale detection exports are supported"
            )
        if (
            len(self.means) != channels
            or len(self.stds) != channels
            or any(v <= 0 for v in self.stds)
        ):
            raise ValueError("Normalization means/stds must match the input channels")
        if self.pixel_scale != 1 / 255:
            raise ValueError("Expected RF-DETR 1/255 pixel scaling")
        if (
            set(self.outputs) != {"pred_boxes", "pred_logits"}
            or len(set(self.outputs.values())) != 2
        ):
            raise ValueError(
                "Metadata must map distinct pred_boxes and pred_logits outputs"
            )
        if not self.class_id_to_name and len(self.class_names) == self.num_classes:
            self.class_id_to_name = dict(enumerate(self.class_names))
        if not self.class_id_to_name or any(
            key < 0 or key > self.num_classes or not value
            for key, value in self.class_id_to_name.items()
        ):
            raise ValueError("A valid exported class_id_to_name mapping is required")
        if set(self.class_names) != set(self.class_id_to_name.values()):
            raise ValueError("Class names and exported class mapping disagree")
        return self


def preprocess(rgb, metadata):
    """Use upstream half-pixel bilinear, without Pillow's downsampling antialias."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 HWC source image")
    _, channels, height, width = metadata.input_shape
    pixels = (
        np.asarray(
            Image.fromarray(rgb).convert("L" if channels == 1 else "RGB"),
            dtype=np.float32,
        )
        / 255.0
    )
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    chw = _bilinear_resize_half_pixel(pixels.transpose(2, 0, 1), height, width)
    mean = np.asarray(metadata.means, dtype=np.float32)[:, None, None]
    std = np.asarray(metadata.stds, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((chw - mean) / std)[None], dtype=metadata.input_dtype)


class ONNXDetector:
    def __init__(self, model_path, threads=2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        raw = self.session.get_modelmeta().custom_metadata_map.get("rfdetr_inference")
        if raw is None:
            raise ValueError(
                "Missing rfdetr_inference metadata; re-export using the export extra"
            )
        self.metadata = DetectionMetadata.model_validate(json.loads(raw))
        self.class_names = self.metadata.class_names
        self._check_graph()
        self.output_names = [
            self.metadata.outputs[key] for key in ("pred_boxes", "pred_logits")
        ]

    def _check_graph(self):
        inputs = self.session.get_inputs()
        m = self.metadata
        if len(inputs) != 1 or inputs[0].name != m.input_name:
            raise ValueError("ONNX input names disagree with RF-DETR metadata")
        expected_type = (
            "tensor(float)" if m.input_dtype == "float32" else "tensor(float16)"
        )
        if inputs[0].type != expected_type or len(inputs[0].shape) != 4:
            raise ValueError("ONNX input dtype/rank disagrees with metadata")
        expected = (1, *m.input_shape[1:])
        if any(
            isinstance(actual, int) and actual != size
            for actual, size in zip(inputs[0].shape, expected)
        ):
            raise ValueError("ONNX input shape disagrees with metadata")
        outputs = {item.name: item for item in self.session.get_outputs()}
        for role, tail in (("pred_boxes", 4), ("pred_logits", m.num_classes + 1)):
            output = outputs.get(m.outputs[role])
            if (
                output is None
                or len(output.shape) != 3
                or output.type not in ("tensor(float)", "tensor(float16)")
            ):
                raise ValueError(f"Invalid ONNX {role} output")
            if isinstance(output.shape[-1], int) and output.shape[-1] != tail:
                raise ValueError(f"ONNX {role} shape disagrees with metadata")

    def predict_frame(self, rgb, threshold=0.5, include_source_image=False):
        boxes, logits = self.session.run(
            self.output_names,
            {self.metadata.input_name: preprocess(rgb, self.metadata)},
        )
        if (
            boxes.ndim != 3
            or logits.ndim != 3
            or boxes.shape[0] != 1
            or logits.shape[0] != 1
            or boxes.shape[1] != logits.shape[1]
            or boxes.shape[2] != 4
            or logits.shape[2] != self.metadata.num_classes + 1
        ):
            raise ValueError("ONNX returned incompatible detection tensor shapes")
        decoded = decode_detections(
            boxes[0],
            logits[0],
            (rgb.shape[1], rgb.shape[0]),
            threshold=threshold,
            num_select=self.metadata.num_select,
        )
        # all_logits selection preserves sparse COCO IDs and custom model slot layout.
        names = np.array(
            [
                self.metadata.class_id_to_name.get(
                    int(cid),
                    "__background__" if cid == self.metadata.num_classes else "",
                )
                for cid in decoded.class_id
            ],
            dtype=object,
        )
        result = sv.Detections(
            xyxy=decoded.xyxy,
            confidence=decoded.confidence,
            class_id=decoded.class_id.astype(int),
            data={"class_name": names},
        )
        if include_source_image:
            result.metadata["source_image"] = rgb
        return result
