# YOLOX Nano and Tiny ONNX

The camera service accepts metadata-bearing **YOLOX Nano/Tiny FP32 ONNX** models
alongside RF-DETR FP32/INT8. Model selection is automatic from embedded metadata;
no detector-family flag or new runtime dependency is required. The Pi remains on
ONNX Runtime CPU with Supervision, Roboflow ByteTrack, presence holds, FastALPR/OCR
and Home Assistant MQTT. PyTorch/TorchVision stay on the export computer.

## Models and contracts

These exports use the official
[YOLOX checkpoints](https://github.com/Megvii-BaseDetection/YOLOX#benchmark) from
release `0.1.1rc0`, with source commit
`6ddff4824372906469a7fae2dc3206c7aa4bbaee`. The upstream code/helpers use Apache 2.0;
license and notices are retained in `scripts/vendor/yolox`.

- Nano: 416×416 input, approximately 0.91M parameters, 3.5 MiB ONNX.
- Tiny: 416×416 input, approximately 5.06M parameters, 19 MiB ONNX.
- RF-DETR INT8 reference: 384×384 input, approximately 32 MiB ONNX.

The adapter reuses upstream BGR/raw-pixel preprocessing, top-left letterbox with
114 padding, grid/stride decoding, objectness × class scoring and class-agnostic
NMS at IoU 0.45. It does not apply RF-DETR's RGB/ImageNet normalization or a second
sigmoid. Final boxes are clipped to the original image and zero-area boxes removed.
Original RGB images remain available for OCR/snapshots/RTSP, including empty
object-detection frames. The bounded Roboflow capture helper is unchanged.

Exported ONNX files contain `yolox_inference` metadata specifying input shape,
class labels, raw head format, preprocessing, source revision and checkpoint hash.
Decoded-in-graph, legacy-normalized, missing/ambiguous metadata and inconsistent
graph shapes are rejected. External files should be exported through the helper;
the adapter does not guess conventions from filenames or an 85-column output.

`class_id` is model-specific: YOLOX COCO labels are contiguous 0–79, whereas the
RF-DETR COCO export uses sparse IDs. Use the emitted `label` and `model` fields
when consuming predictions. Presence/count groups continue to match label names.

## Export on a development computer

The optional `yolox-export` extra installs Torch/TorchVision/ONNX and the small
utilities needed by the pinned upstream source. It is not installed on the Pi.
The default inference environment remains unchanged when using a separate env:

```sh
git clone https://github.com/Megvii-BaseDetection/YOLOX.git work/YOLOX
git -C work/YOLOX checkout --detach 6ddff4824372906469a7fae2dc3206c7aa4bbaee
mkdir -p models/yolox-nano models/yolox-tiny
curl -L --fail https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -o models/yolox-nano/yolox_nano.pth
curl -L --fail https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth -o models/yolox-tiny/yolox_tiny.pth
UV_PROJECT_ENVIRONMENT=.venv-yolox-export uv run --locked --no-dev --extra yolox-export scripts/export_yolox_onnx.py \
  --source-tree work/YOLOX --variant nano --checkpoint models/yolox-nano/yolox_nano.pth \
  --output models/yolox-nano/yolox-nano.onnx --verify-images /path/to/scene.jpg
UV_PROJECT_ENVIRONMENT=.venv-yolox-export uv run --locked --no-dev --extra yolox-export scripts/export_yolox_onnx.py \
  --source-tree work/YOLOX --variant tiny --checkpoint models/yolox-tiny/yolox_tiny.pth \
  --output models/yolox-tiny/yolox-tiny.onnx --verify-images /path/to/scene.jpg
```

The helper checks the upstream revision and tracked source changes, uses
`torch.load(weights_only=True)` and strict state loading, exports raw batch-one
FP32 heads at opset 17, and checks executable ONNX output against upstream
PyTorch and postprocessing before publishing. The legacy TorchScript exporter is
selected explicitly to match this upstream architecture. No ONNX simplifier or
full upstream YOLOX package installation is required. Output overwrite is refused.

Each ONNX file has a same-stem JSON provenance/parity manifest. Checkpoint hashes
for the tested official downloads were:

- nano: `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b`
- tiny: `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0`

A custom checkpoint can use `--class-names classes.json` containing an ordered
JSON list matching its classifier. Model architecture/weights must match the
chosen Nano/Tiny experiment. Fine-tuning remains a development-machine task.

## Configure the Pi

Copy the ONNX file and JSON manifest beside the existing models. Keep the Pi on
`uv sync --locked --no-dev`, without either export extra. Copy your normal config
and change its `model_path`, for example:

```yaml
model_path: ../models/yolox-nano/yolox-nano.onnx
onnx_threads: 4
```

Run the same `scripts/onnx_camera.py --config ...` command with the usual MQTT
credentials. Model-specific preprocessing is selected automatically; source,
tracking, holds, classes, LPR and image-transport settings continue to apply.
The existing `max_fps` cap is still honored. Raise it only after measuring the
full workload; a faster benchmark alone does not change the configured cap.

No changes to the plate detector/OCR models are made. The existing INT8 generator
is RF-DETR-specific; the YOLOX models compared here are FP32.

## Physical Raspberry Pi comparison

Measured on the same 4 GB Raspberry Pi 4, 64-bit Debian 13.6, with its desktop and
RetroArch still running, using four ORT CPU threads. Models ran serially in fresh
processes, with reversed order on the second repeat. All processed identical
recordings using their normal model input resolutions; the application rate cap
and MQTT/media output costs were excluded.

| Model | Detection + tracking FPS | With every-frame ALPR FPS | Detection peak RSS | ONNX size |
| --- | ---: | ---: | ---: | ---: |
| RF-DETR Nano static INT8 | 0.661 | 0.575 | 433.5 MiB | 32.35 MiB |
| YOLOX Nano FP32 | 4.313 | 1.744 | 267.3 MiB | 3.54 MiB |
| YOLOX Tiny FP32 | 1.870 | 1.164 | 295.6 MiB | 19.32 MiB |

Detection/tracking used 20 measured frames after three warmups per run, repeated
twice. Every-frame ALPR used 16 measured frames after three warmups, repeated twice,
on the public plate sample recording. FPS values are medians across runs; peak
RSS is the maximum for the fixed detection recording. These are short shared-load
checks, not sustained headless throughput guarantees.

All three models passed native Pi tracking and plate checks: confirmed vehicle
tracks remained stable across the repeated sample and each emitted exactly one
plate event. Both YOLOX variants also passed the full MQTT/Home Assistant path:
nine prediction messages, JPEG camera snapshots, online/offline availability and
one recognised-plate event with a fresh Home Assistant timestamp. The plate
`5AU5341` came from the public FastALPR test image, not a live vehicle observation.
Existing presence hold and repeat-suppression tests passed unchanged. Short USB webcam smoke checks also completed (10 measured frames each): Nano
5.14 FPS, Tiny 1.91 FPS. These live frames differ between runs and are not the
matched-recording comparison above. The webcam format was restored afterward.
No thermal
throttling was reported during the comparison (`get_throttled=0x0`).

## Quality tradeoff

The same 32 labelled COCO128 train2017 images from the INT8 diagnostic were used,
with a 0.001 evaluation score floor and Supervision box mAP. These images are not
held out from the detectors' original training, so this is a small comparative
diagnostic rather than COCO val2017 or deployment-camera validation.

| Model | All-class mAP50–95 | Person/vehicle mAP50–95 |
| --- | ---: | ---: |
| RF-DETR INT8 | 72.21 | 74.24 |
| YOLOX Nano | 46.91 | 58.89 |
| YOLOX Tiny | 56.37 | 59.64 |

Nano provides the largest speed/size gain; Tiny scored higher overall in this
sample. RF-DETR INT8 retained the highest measured quality. Neither YOLOX variant
is automatically selected as the default. Validate person/vehicle recall on real
camera footage, including small/distant objects and lighting changes, before
choosing a deployment model. Tracking/holds cannot recover objects a detector misses.

Use the same tools to reproduce the comparison:

```sh
uv run --locked --no-dev scripts/benchmark_models.py --config config/pi.local.yml \
  --models models/rfdetr-nano/rfdetr-nano-int8-static.onnx models/yolox-nano/yolox-nano.onnx models/yolox-tiny/yolox-tiny.onnx \
  --source /path/to/recording.mp4 --onnx-threads 4 --frames 20 --warmup 3 --repeats 2 \
  --lpr off --output-dir work/yolox-detection
# Repeat with a representative plate recording and --lpr every-frame for equal ALPR work.
```

`scripts/evaluate_onnx.py` now selects either adapter from metadata, while retaining
its explicit dataset label mapping and calibration-overlap checks. See the
[INT8 guide](int8.md) for evaluation inputs, manifests and benchmark scope.
