# YOLOX inference helpers

Extracted from the Apache-2.0 Megvii YOLOX repository at commit
`6ddff4824372906469a7fae2dc3206c7aa4bbaee`. See `LICENSE`, `NOTICE` and
`PROVENANCE.json` for attribution, source paths and hashes.

`preprocess.py` extracts `preproc` (BGR 0–255, linear resize, top-left letterbox,
114 padding). `postprocess.py` extracts raw grid decoding and NumPy NMS functions.
The selected algorithms are unchanged; training/visualization functions and
package-level imports are omitted. `coco_classes.py` is copied unchanged.

The adapter converts captured RGB to BGR for inference, retains original RGB for
OCR/media, and clips final boxes to the source image. Zero-area clipped boxes are
removed before tracking. These wrapper policies are tested separately.

When updating, preserve upstream notices, update provenance hashes and the pinned
revision, compare ONNX and PyTorch preprocessing/raw outputs/final detections, run
contract/import tests, and repeat representative labelled and Pi benchmarks.
The upstream training package belongs only on the development/export computer.
