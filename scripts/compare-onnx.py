"""Compare the lean adapter with RF-DETR's official ONNX predictor (export extra)."""

import argparse
import json

import numpy as np
from onnx_detector import ONNXDetector, preprocess
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--onnx-threads", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.onnx_threads < 1:
        parser.error("threshold must be in [0, 1] and onnx-threads must be positive")
    from rfdetr import RFDETRInference
    from rfdetr.export._runtime.preprocess import preprocess_to_nchw

    lean = ONNXDetector(args.model, threads=args.onnx_threads)
    reference = RFDETRInference(args.model, device="cpu")
    report = []
    for path in args.images:
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        expected_input = preprocess_to_nchw(
            Image.fromarray(rgb),
            *lean.metadata.input_shape[2:],
            lean.metadata.input_shape[1],
        )
        actual_input = preprocess(rgb, lean.metadata)
        np.testing.assert_allclose(actual_input, expected_input, atol=3e-5, rtol=3e-5)
        expected = reference.predict(rgb, threshold=args.threshold)
        actual = lean.predict_frame(rgb, threshold=args.threshold)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)
        np.testing.assert_array_equal(
            actual.data["class_name"], expected.data["class_name"]
        )
        np.testing.assert_allclose(
            actual.confidence, expected.confidence, atol=2e-5, rtol=2e-4
        )
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=0.05, rtol=1e-4)
        report.append(
            {
                "image": path,
                "detections": len(actual),
                "max_input_absolute_error": float(
                    np.max(np.abs(actual_input - expected_input))
                ),
                "max_score_absolute_error": float(
                    np.max(np.abs(actual.confidence - expected.confidence))
                )
                if len(actual)
                else 0,
                "max_box_absolute_error_pixels": float(
                    np.max(np.abs(actual.xyxy - expected.xyxy))
                )
                if len(actual)
                else 0,
            }
        )
    print(
        json.dumps(
            {"matched": True, "threshold": args.threshold, "results": report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
