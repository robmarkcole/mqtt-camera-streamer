# Dependencies and inference performance

## Changes implemented

- **PyTorch, TorchVision, Transformers, RF-DETR and the ONNX Python package are
  absent from the default runtime.** ONNX Runtime reads embedded model metadata
  through its session API. The official training/export stack is confined to
  `--extra export`, with CPU-only Torch wheels selected for Linux export hosts.
  Keep that extra off the Pi. `uv.lock` includes optional dependencies; their
  presence there does not mean they are installed in the runtime.
- The default environment uses **opencv-python-headless**, with no GUI OpenCV,
  Qt/X11 or Streamlit dependency. The pinned Trackers release requests GUI OpenCV
  in its package metadata; a version-specific uv metadata correction substitutes
  headless OpenCV and preserves its other published requirements. Recheck that
  correction when upgrading Trackers. Installing both distributions would let
  them overwrite the shared `cv2` module; use the locked uv workflow.
- The service reuses RF-DETR 1.12.1's small NumPy resize/top-k/decode helpers and
  bounded OpenCV capture implementation under their Apache license. Their source
  paths, hashes and changes are recorded in `scripts/vendor/rfdetr/PROVENANCE.json`.
  This avoids importing the training package or maintaining a different decoding
  rule. Finite recordings prefetch at most two frames (upstream used 30); live
  capture retains one latest frame. This also bounds benchmark decode memory.
  Screen capture and YouTube download packages remain omitted.
- Streamlit and its viewer script have been removed; imutils is available through
  `--extra legacy` and Datasette through `--extra database`. Development dependencies
  are excluded by `--no-dev`. The lockfile records all extras, but the default
  environment does not install all packages mentioned in the lockfile.
- RTSP output reuses PyAV, already in the Supervision dependency graph. PyAV 19
  provides bounded output-operation timeouts; no separate FFmpeg executable is
  needed. An RTSP server is required only for this optional mode.
- Home Assistant already provides the dashboard. No web server/viewer is needed
  on the Pi to publish detections, presence and plate events.

## Where further reductions are worthwhile

| Component | Practical assessment | Next step |
| --- | --- | --- |
| PyTorch / TorchVision / Transformers | Removed from the runtime; retained only in the optional export environment. | Compare new model exports against the official predictor before deployment. |
| OpenCV | Headless OpenCV removes GUI baggage. RF-DETR capture, FastALPR/Open Image Models and parts of Supervision/Trackers still use `cv2`. | Keep headless OpenCV for this reused stack. Replacing just the webcam reader will not remove the dependency. |
| PyAV / FFmpeg | PyAV is already pulled in by Supervision and can be a useful capture/decode API. It includes native codec dependencies. | Consider a PyAV source adapter if capture is the measured bottleneck; test disconnects, latest-frame buffering, pixel format and timestamps. Do not assume it is smaller/faster on every platform. |
| Direct V4L2 / MJPEG + Pillow | Can suit a fixed USB camera and avoid some general video plumbing. It adds device, buffer, format and reconnect logic. | Evaluate only for a camera-specific build; retaining the current downstream libraries still retains OpenCV. |
| Viewer | Streamlit has been removed completely. | Use HA dashboards. Use the optional PyAV RTSP output and a laptop player over SSH for live preview; no Python viewer is installed. |
| Plate models | The pretrained ONNX detector/OCR weights are small relative to the RF-DETR Python stack, and are loaded only with LPR enabled. | Adjust scan cadence and CPU threads; profile the real frame/crop before changing models. |

## Validating the lean adapter

On a development computer with representative images:

```sh
uv run --extra export scripts/compare-onnx.py --model models/rfdetr-nano/rfdetr-nano.onnx --images /path/to/scene1.jpg /path/to/scene2.jpg --threshold 0.1
uv run --extra export scripts/compare-onnx.py --model models/rfdetr-nano/rfdetr-nano.onnx --images /path/to/scene1.jpg /path/to/scene2.jpg --threshold 0.5
uv sync --locked --no-dev
```

The comparison checks normalized inputs, class IDs/names, confidence and boxes
against the pinned official RF-DETR ONNX predictor. It allows only small floating
point differences from NumPy versus Torch interpolation. Tests also check sparse
COCO IDs (including 90), contiguous custom labels, multiple classes per query,
tie order, strict thresholding, empty frames, malformed contracts and imports with
the export stack blocked. The runtime deliberately supports detection only.
A clean Linux ARM64 runtime installation of this lockfile contained 42 packages
(previously 70), occupying about 520 MiB in the virtual environment, excluding
Python, model files and model caches. Those are container measurements, not Pi
hardware results; see the [Pi guide](raspberry-pi.md#measured-raspberry-pi-4-smoke-test)
for the subsequent physical Pi 4 checks. The two-thread detection run used about 551 MiB peak process
RSS with a finite recording; live capture buffers fewer frames. ONNX Runtime and
model workspaces still consume memory even after the training stack is removed.

Lighter dependencies do not guarantee higher FPS: on that ARM64 VM, detection
measured about 0.76 FPS with two ORT threads and 1.38 FPS with four. The previous
wrapper used an automatic ORT thread pool; `torch_threads` only limited Torch
preprocessing. Tune `onnx_threads` on the actual board, with the full service and
thermal monitoring, rather than interpreting package removal as a speed claim.

Repeat parity/accuracy checks on representative data when changing the exported
artifact or vendored helpers; this is not a full COCO accuracy evaluation.

## Optional INT8 tools

The `quantize` extra adds ONNX and SymPy only for development-time quantization.
It does not add Torch. Copy the resulting model to the Pi and keep the normal
runtime installation without that extra. [Measured INT8 results and commands](int8.md)
show the speed, file-size and quality tradeoff; the application still uses the
same Supervision, tracking and ALPR path.

## Ideas from Roboflow's blog

These are deployment ideas, not Pi throughput promises. Cross-check code snippets
against the pinned libraries; older posts may use different APIs.

1. **Profile the whole path and reduce inference cadence first.**
   [Advanced Techniques for Optimizing AI Inference Costs](https://blog.roboflow.com/advanced-techniques-for-optimizing-ai-inference-costs/)
   discusses decoding, preprocessing, models, tracking and output costs. The
   benchmark measures the inference path, with media/MQTT output costs excluded; the service already has a maximum inference
   rate, independent plate cadence and latest-frame capture. Tracking currently
   associates processed detections; it does not invent trustworthy measurements
   for every skipped camera frame.
2. **Preserve the existing streaming abstractions.**
   [Realtime Video Stream Analysis](https://blog.roboflow.com/video-stream-analysis/)
   demonstrates input streams plus prediction sinks. Our single-camera service
   reuses the attributed RF-DETR capture helper and sends MQTT results through a separate
   adapter. A broader InferencePipeline/server becomes more relevant if managing
   many cameras/models or remote inference outweighs its added dependencies.
3. **Use a smaller model, resolution or quantization only after measuring accuracy.**
   [What to Think About When Choosing Vision Model Sizes](https://blog.roboflow.com/computer-vision-model-tradeoff/)
   explains the accuracy/latency tradeoff. Nano is the current baseline. Export a
   separately validated ONNX artifact for each experiment; do not override the
   fixed model input dimensions during inference. Test calibrated INT8 against
   FP32 on representative scenes and plate crops. CPU results vary by operator
   support, so quantization is not assumed to improve speed.
4. **Avoid unnecessary image transfer and encoding.**
   [Drone-Based Security with Computer Vision](https://blog.roboflow.com/drone-based-security-reconnaissance-system/)
   samples frames on a cadence and reduces transfer size. Our local inference path
   sends predictions and optional sparse JPEGs over MQTT; it does not round-trip
   frames through the broker for inference. Keep
   original-resolution images for plate crops; shrinking those indiscriminately
   can erase the characters OCR needs.

Next experiments, in order: compare `onnx_threads` and plate `num_threads` on the
Pi; compare configured plate cadence with every-frame ALPR; evaluate representative
input resolution/model artifacts; then consider ROI/motion gating. Add an accuracy
set alongside speed results so reduced compute does not
silently turn missed objects/plates into apparently better throughput.

Relevant references: [RF-DETR ONNX adapter](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_onnx/inference.py),
[RF-DETR CPU cookbook](https://rfdetr.roboflow.com/latest/cookbooks/export-cpu/),
[uv PyTorch source configuration](https://docs.astral.sh/uv/guides/integration/pytorch/),
[OpenCV wheel variants](https://pypi.org/project/opencv-python-headless/),
[PyAV](https://pyav.org/docs/stable/).

YOLOX Nano/Tiny support adds no base runtime packages. Its attributed NumPy/OpenCV
letterbox, decode and NMS helpers live in `scripts/vendor/yolox`; the optional
`yolox-export` extra stays on the development computer. See [the YOLOX guide](yolox.md)
for source provenance, export parity and Pi speed/quality measurements.
