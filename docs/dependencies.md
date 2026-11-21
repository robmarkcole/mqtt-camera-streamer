# Dependencies and inference performance

## Changes implemented

- Linux explicitly uses the official **CPU-only PyTorch/TorchVision index**.
  This removes the CUDA dependency chain from the Pi installation. macOS uses
  its normal PyPI wheels. The versions/builds can differ between platforms, so
  compare model outputs and benchmark on the target. GPU-enabled Linux would
  need an intentional source/profile change.
- The default environment uses **opencv-python-headless**, with no GUI OpenCV,
  Qt/X11 or Streamlit dependency. The pinned Trackers release requests GUI OpenCV
  in its package metadata; a version-specific uv metadata correction substitutes
  headless OpenCV and preserves its other published requirements. Recheck that
  correction when upgrading Trackers. Installing both distributions would let
  them overwrite the shared `cv2` module; use the locked uv workflow.
- RF-DETR uses the `onnx` extra rather than its broader `stream` extra. USB,
  recorded video and RTSP capture still use the reused OpenCV source API. Screen
  capture (`mss`) and YouTube download (`yt-dlp`) packages are omitted.
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
| PyTorch / TorchVision / Transformers | The official RF-DETR predictor imports them even for ONNX. This is the largest remaining stack. | Prototype a lean ONNX Runtime adapter only if install size/RSS is a problem; compare preprocessing, class IDs, boxes and scores against the official API before switching. |
| OpenCV | Headless OpenCV removes GUI baggage. RF-DETR capture, FastALPR/Open Image Models and parts of Supervision/Trackers still use `cv2`. | Keep headless OpenCV for this reused stack. Replacing just the webcam reader will not remove the dependency. |
| PyAV / FFmpeg | PyAV is already pulled in by Supervision and can be a useful capture/decode API. It includes native codec dependencies. | Consider a PyAV source adapter if capture is the measured bottleneck; test disconnects, latest-frame buffering, pixel format and timestamps. Do not assume it is smaller/faster on every platform. |
| Direct V4L2 / MJPEG + Pillow | Can suit a fixed USB camera and avoid some general video plumbing. It adds device, buffer, format and reconnect logic. | Evaluate only for a camera-specific build; retaining the current downstream libraries still retains OpenCV. |
| Viewer | Streamlit has been removed completely. | Use HA dashboards. Use the optional PyAV RTSP output and a laptop player over SSH for live preview; no Python viewer is installed. |
| Plate models | The pretrained ONNX detector/OCR weights are small relative to the RF-DETR Python stack, and are loaded only with LPR enabled. | Adjust scan cadence and CPU threads; profile the real frame/crop before changing models. |

The current adapter also reads embedded ONNX metadata using the `onnx` package,
so removing the ONNX export extra would break model loading. Uninstalling PyTorch
or forcing no-deps installs would break the supported RF-DETR path. The benchmark
is the baseline for deciding whether a separate lean runtime is worth maintaining.

## Ideas from Roboflow's blog

These are deployment ideas, not Pi throughput promises. Cross-check code snippets
against the pinned libraries; older posts may use different APIs.

1. **Profile the whole path and reduce inference cadence first.**
   [Advanced Techniques for Optimizing AI Inference Costs](https://blog.roboflow.com/advanced-techniques-for-optimizing-ai-inference-costs/)
   discusses decoding, preprocessing, models, tracking and output costs. The new
   benchmark separates those stages; the service already has a maximum inference
   rate, independent plate cadence and latest-frame capture. Tracking currently
   associates processed detections; it does not invent trustworthy measurements
   for every skipped camera frame.
2. **Preserve the existing streaming abstractions.**
   [Realtime Video Stream Analysis](https://blog.roboflow.com/video-stream-analysis/)
   demonstrates input streams plus prediction sinks. Our single-camera service
   already reuses RF-DETR's capture loop and sends MQTT results through a separate
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
   sends predictions and optional sparse JPEGs over MQTT; it does not round-trip frames through the broker for inference. Keep
   original-resolution images for plate crops; shrinking those indiscriminately
   can erase the characters OCR needs.

Next experiments, in order: compare `torch_threads` and plate `num_threads` on the
Pi; compare configured plate cadence with every-frame ALPR; evaluate representative
input resolution/model artifacts; then consider ROI/motion gating or a smaller
runtime. Add an accuracy set alongside speed results so reduced compute does not
silently turn missed objects/plates into apparently better throughput.

Relevant references: [RF-DETR ONNX adapter](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_onnx/inference.py),
[RF-DETR CPU cookbook](https://rfdetr.roboflow.com/latest/cookbooks/export-cpu/),
[uv PyTorch source configuration](https://docs.astral.sh/uv/guides/integration/pytorch/),
[OpenCV wheel variants](https://pypi.org/project/opencv-python-headless/),
[PyAV](https://pyav.org/docs/stable/).
