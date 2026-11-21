"""Torch-free YOLOX ONNX adapter using the upstream letterbox/decode/NMS rules."""

import json
from typing import Literal

import numpy as np
import supervision as sv
from onnx_detector import create_cpu_session
from pydantic import BaseModel, ConfigDict, Field, model_validator
from vendor.yolox import UPSTREAM_COMMIT
from vendor.yolox.postprocess import demo_postprocess, multiclass_nms
from vendor.yolox.preprocess import preproc


class YOLOXMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1]
    family: Literal["yolox"]
    variant: Literal["yolox-nano", "yolox-tiny"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_shape: tuple[int, int, int, int]
    input_name: str
    output_name: str
    input_dtype: Literal["float32"]
    input_layout: Literal["NCHW"]
    preprocessing: Literal["bgr-0-255-top-left-letterbox-114-linear"]
    decode_in_graph: Literal[False]
    strides: tuple[int, int, int]
    class_names: list[str] = Field(min_length=1)
    nms_iou_threshold: float = Field(default=0.45, gt=0, le=1)
    nms_class_agnostic: Literal[True]

    @model_validator(mode="after")
    def valid_contract(self):
        batch, channels, height, width = self.input_shape
        if (
            batch != 1
            or channels != 3
            or min(height, width) < 32
            or height % 32
            or width % 32
        ):
            raise ValueError(
                "YOLOX requires fixed batch-one RGB capture with model dimensions divisible by 32"
            )
        if self.strides != (8, 16, 32):
            raise ValueError("Only the three-level YOLOX Nano/Tiny head is supported")
        if len(set(self.class_names)) != len(self.class_names) or any(
            not name.strip() for name in self.class_names
        ):
            raise ValueError("YOLOX needs distinct nonempty class labels")
        return self

    @property
    def query_count(self):
        _, _, height, width = self.input_shape
        return sum((height // stride) * (width // stride) for stride in self.strides)


def preprocess_yolox(rgb, metadata):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 HWC source image")
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    tensor, ratio = preproc(bgr, metadata.input_shape[2:])
    return tensor[None], ratio


class YOLOXDetector:
    family = "yolox"
    helpers_source = f"yolox-{UPSTREAM_COMMIT}"

    def __init__(self, model_path, threads=2, *, session=None):
        self.session = (
            session if session is not None else create_cpu_session(model_path, threads)
        )
        raw = self.session.get_modelmeta().custom_metadata_map.get("yolox_inference")
        if raw is None:
            raise ValueError(
                "Missing yolox_inference metadata; use export_yolox_onnx.py"
            )
        self.metadata = YOLOXMetadata.model_validate(json.loads(raw))
        self.class_names = self.metadata.class_names
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != self.metadata.input_name
            or inputs[0].type != "tensor(float)"
            or tuple(inputs[0].shape) != self.metadata.input_shape
        ):
            raise ValueError("YOLOX graph input disagrees with embedded metadata")
        expected = (1, self.metadata.query_count, len(self.class_names) + 5)
        if (
            len(outputs) != 1
            or outputs[0].name != self.metadata.output_name
            or outputs[0].type != "tensor(float)"
            or tuple(outputs[0].shape) != expected
        ):
            raise ValueError("YOLOX graph output disagrees with raw-grid metadata")

    def predict_frame(self, rgb, threshold=0.5, include_source_image=False):
        tensor, ratio = preprocess_yolox(rgb, self.metadata)
        raw = self.session.run(
            [self.metadata.output_name], {self.metadata.input_name: tensor}
        )[0]
        expected = (1, self.metadata.query_count, len(self.class_names) + 5)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError("YOLOX returned malformed or nonfinite predictions")
        if np.any(raw[:, :, 4:] < 0) or np.any(raw[:, :, 4:] > 1):
            raise ValueError(
                "YOLOX objectness/class outputs must already be sigmoid probabilities"
            )
        decoded = demo_postprocess(raw.copy(), self.metadata.input_shape[2:])[0]
        if not np.isfinite(decoded).all():
            raise ValueError("YOLOX decoded nonfinite boxes")
        centers, sizes = decoded[:, :2], decoded[:, 2:4]
        boxes = (
            np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
        )
        scores = decoded[:, 4:5] * decoded[:, 5:]
        selected = multiclass_nms(
            boxes,
            scores,
            nms_thr=self.metadata.nms_iou_threshold,
            score_thr=threshold,
            class_agnostic=True,
        )
        if selected is None:
            result = sv.Detections.empty()
            result.data["class_name"] = np.empty(0, dtype=object)
        else:
            height, width = rgb.shape[:2]
            clipped = np.clip(
                selected[:, :4], 0, [width, height, width, height]
            ).astype(np.float32)
            valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
            clipped, selected = clipped[valid], selected[valid]
            class_id = selected[:, 5].astype(int)
            result = sv.Detections(
                xyxy=clipped,
                confidence=selected[:, 4].astype(np.float32),
                class_id=class_id,
                data={
                    "class_name": np.array(
                        [self.class_names[index] for index in class_id], dtype=object
                    )
                },
            )
        if include_source_image:
            result.metadata["source_image"] = rgb
        return result
