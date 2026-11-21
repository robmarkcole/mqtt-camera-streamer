# RF-DETR helpers

These small helpers come from the PyPI `rfdetr==1.12.1` distribution published by
[Roboflow](https://github.com/roboflow/rf-detr). They retain their Apache 2.0 license
and copyright; see `LICENSE`, `NOTICE` and `PROVENANCE.json`.

`_resize.py`, `_topk.py` and `_class_layout.py` are unchanged. `decode.py` uses local
imports and the standard logger. `streams.py` uses the standard logger and reduces finite-source prefetch from
30 to two frames to limit Pi memory use. Live capture still keeps only the latest
frame. Apart from that queue bound and removing package-level imports, no
preprocessing, decoding or reconnect policy is changed.

The service implements its own small metadata/session adapter around these helpers.
Input normalization follows the upstream NumPy fallback in
`src/rfdetr/export/_runtime/preprocess.py`. The rest of RF-DETR is installed only
through the optional export extra.

When updating, compare these files with the selected distribution, preserve its
license/copyright, update the provenance hashes and modified-file notices, run the
capture/decoding tests, and run `scripts/compare-onnx.py` against the official
predictor on representative frames at both tracking and detection thresholds.
Do not update the helper copy implicitly when changing the optional exporter.
