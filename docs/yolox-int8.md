# YOLOX INT8 experiments on Raspberry Pi

YOLOX Nano/Tiny can be statically quantized on a development computer with the
existing optional `quantize` extra. The Pi only needs the normal locked runtime:
ONNX Runtime CPU, Supervision, Roboflow ByteTrack, FastALPR and Home Assistant
MQTT. Torch, TorchVision and ONNX Python remain excluded from that environment.
The default model and application FPS cap are unchanged.

## Recipe and reproduction

The helper uses ONNX Runtime static QDQ, signed INT8 activations/weights and
per-output-channel Conv weights. It keeps the nine box, class and objectness
projection convolutions in FP32; sigmoid, SiLU and decoding also remain float.
This is mixed FP32/INT8, not an integer-only model. BatchNorm was folded during
export; graph preprocessing keeps Conv and SiLU separate for quantization.
See the official [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
for QDQ, calibration and accuracy debugging.

Calibration uses the exact deployed RGB-to-BGR, raw 0–255, top-left letterbox
preprocessing. No RF-DETR normalization is applied. Original RGB frames still
feed plate recognition and media output. Quantization preserves the complete
`yolox_inference` contract and adds `yolox_quantization` provenance. Same-stem JSON
manifests record calibration image hashes, excluded nodes, model hashes, settings
and tool versions. Overwriting originals, already quantized sources and existing
outputs is refused; the result must execute through the CPU adapter before saving.

First export the metadata-bearing FP32 models using [the YOLOX guide](yolox.md).
Use a separate development environment for quantization:

```sh
# Tiny: MinMax retained most measured accuracy.
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --locked --no-dev --extra quantize scripts/quantize_yolox.py \
  --model models/yolox-tiny/yolox-tiny.onnx \
  --output models/yolox-tiny/yolox-tiny-int8-static.onnx \
  --calibration-list /path/to/calibration.txt --limit 96 --calibration-method minmax
# Nano: percentile 99.99 was less damaging, but still lost substantial accuracy.
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --locked --no-dev --extra quantize scripts/quantize_yolox.py \
  --model models/yolox-nano/yolox-nano.onnx \
  --output models/yolox-nano/yolox-nano-int8-static.onnx \
  --calibration-list /path/to/calibration.txt --limit 96 --calibration-method percentile
```

Each list contains one image path per line, relative to the list or absolute.
Use representative deployment images for a real installation. The experiments
reused 96 COCO128 images for calibration and the same separate 32 images for the
accuracy diagnostic. The evaluator rejects calibration/evaluation overlap by
image content, including overlap recorded in the quantization manifest.

## Accuracy comparison

| Model | All-class mAP50–95 | Person/vehicle mAP50–95 |
| --- | ---: | ---: |
| RF-DETR Nano INT8 reference | 72.21 | 74.24 |
| YOLOX Nano FP32 | 46.91 | 58.89 |
| YOLOX Nano INT8, percentile | 27.38 | 48.97 |
| YOLOX Tiny FP32 | 56.37 | 59.64 |
| YOLOX Tiny INT8, MinMax | 55.73 | 58.70 |

Scores are percentages from Supervision box mAP at a 0.001 score floor, evaluated
through ONNX Runtime CPU on the development computer. These 32
COCO128 train2017 images are separate from calibration, but are not held out from
the models' original training. Calibration variants were compared on this same
small diagnostic: it is not an independent final validation set or full COCO
val2017. Check recall on representative camera footage before deployment.

Nano MinMax scored 19.33 all-class / 41.90 person/vehicle mAP50–95. Percentile
99.99 recovered 27.38 / 48.97. An additional percentile candidate with depthwise
convolutions kept float scored 28.46 / 49.30: insufficient recovery to recommend
Nano INT8. Reproduce that diagnostic using `--keep-depthwise-float` and a distinct
output filename. It was not selected for the five-model hardware comparison.
Tiny MinMax lost only 0.64 percentage points of all-class mAP versus its FP32
source in this sample. Leave Nano FP32 available as the safer small-model choice.

## Physical Pi benchmarks

| Model | Detection + tracking FPS | Every-frame ALPR FPS | Detection peak RSS | ONNX size |
| --- | ---: | ---: | ---: | ---: |
| RF-DETR Nano INT8 | 0.655 | 0.573 | 433.5 MiB | 32.35 MiB |
| YOLOX Nano FP32 | 4.329 | 1.754 | 267.4 MiB | 3.54 MiB |
| YOLOX Nano INT8 (percentile) | 3.865 | 1.667 | 278.1 MiB | 1.38 MiB |
| YOLOX Tiny FP32 | 1.872 | 1.114 | 295.5 MiB | 19.32 MiB |
| YOLOX Tiny INT8 (MinMax) | 2.512 | 1.353 | 290.3 MiB | 5.26 MiB |

All five models were remeasured serially on the same Raspberry Pi 4 Model B
Rev 1.2, 4 GB, 64-bit Debian 13.6, four ORT threads, with the existing desktop and
RetroArch left running. Fresh processes ran in forward then reverse order.
Detection/tracking used the same people recording, 20 measured frames after three
warmups, twice per model. Every-frame ALPR used the same public plate recording,
16 measured frames after three warmups, twice per model. FPS is the median of two
runs. Peak RSS is the maximum on the detection recording; source resolution and
content differ in the ALPR recording, so memory must be compared within a workload.
Application rate limits, MQTT, JPEG/storage and RTSP output costs are excluded.
These short shared-load runs do not establish sustained headless throughput.

Both INT8 models passed native Pi tracking and plate-consensus checks: confirmed
vehicle tracks stayed present across the repeated public sample and each emitted
one plate event. The full MQTT path passed for both: nine fresh predictions,
JPEG camera snapshots, online/offline availability, one plate event and a fresh
Home Assistant last-plate timestamp. Existing presence-hold and suppression tests
also passed. The recognised `5AU5341` is from the public FastALPR image, not a live
vehicle observation. No thermal throttling was reported during the comparison.

Short live USB smoke checks completed with 10 measured frames after three warmups:
Nano INT8 4.33 FPS and Tiny INT8 2.82 FPS. These live scenes differ and
are not the matched-recording speed comparison. Camera format was restored to
1280×720 MJPG at 30 FPS. No camera service was enabled at boot.

Tiny INT8 improved detection/tracking throughput by about 34%, and every-frame
ALPR throughput by about 21%, compared with Tiny FP32. Its model is about 73%
smaller. Nano INT8 was about 11% slower for detection/tracking and lost substantial
measured accuracy. Keep Nano FP32 with these recipes. Smaller weight files did
not consistently reduce process RSS. Neither candidate replaces the default.

Copy the model and JSON manifest onto the Pi, select its `model_path` in a copied
configuration and keep `uv sync --locked --no-dev`. Tracking, presence holds,
plate consensus and MQTT discovery need no model-specific changes. Class IDs
remain YOLOX contiguous COCO ordinals; use labels for cross-family comparisons.

Reproduce the speed comparison with the existing benchmark tool:

```sh
uv run --locked --no-dev scripts/benchmark_models.py --config config/pi.local.yml \
  --models models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  models/yolox-nano/yolox-nano.onnx models/yolox-nano/yolox-nano-int8-static.onnx \
  models/yolox-tiny/yolox-tiny.onnx models/yolox-tiny/yolox-tiny-int8-static.onnx \
  --source /path/to/people.mp4 --onnx-threads 4 --frames 20 --warmup 3 --repeats 2 \
  --lpr off --output-dir work/yolox-int8-detection
# Repeat on a plate recording with --lpr every-frame --frames 16 in a new output directory.
```

Use `scripts/evaluate_onnx.py` with the same evaluation list, YOLO labels and COCO
class-name mapping for each model; pass `--calibration-list` to enforce separation.
See [the INT8 evaluation guide](int8.md) for dataset preparation and evaluator
commands. No additional runtime packages or changes to plate models are required.
