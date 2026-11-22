# RF-DETR Nano INT8 experiments on Raspberry Pi

> Historical experiment record. For the supported presets, see [model setup](../models.md).
> Results are unchanged; commands use the current tool locations.

INT8 is available as an experimental model choice; **FP32 remains the default**.
The inference application, Supervision detections, Roboflow ByteTrack, presence
holds, MQTT/HA, snapshots/RTSP and FastALPR/OCR use the same runtime path. No Torch,
TorchVision, RF-DETR Python package or quantization toolkit is needed on the Pi.

## What was quantized

The static recipe follows RF-DETR **1.12.1**: quantize selected `MatMul`/`Gemm`
inputs/weights, retaining attention-score multiplies, final detection heads,
normalization, surrounding operations and linear outputs in floating point.
Weights are per-tensor INT8, with calibrated S8 activations, QDQ format and MinMax
ranges. This is a **mixed FP32/INT8 graph**, not a fully integer model.

The graph-selection helpers are reused under their Apache license in
`scripts/vendor/rfdetr/quantization.py`. See
[Roboflow's CPU/INT8 cookbook](https://rfdetr.roboflow.com/latest/cookbooks/export-cpu/)
and [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

The comparison also tried dynamic activation quantization with per-tensor INT8
weights. A per-channel dynamic variant failed an executable CPU-kernel check
(`MatmulInteger` B zero-point shape); it was rejected rather than deployed.
Static and dynamic candidates both use the upstream attention/head exclusions.
Convolutions remain floating point. Plate detector and OCR models are unchanged.

## Physical Pi results

Tested 2026-11-21 on the same Raspberry Pi 4 Model B (4 GB, 64-bit Debian 13.6,
approximately 1.5 GHz) used in the [deployment smoke test](../raspberry-pi.md).
The desktop/RetroArch workload remained running. Four ORT threads were used for
all variants, in serial fresh processes, with reversed model order on the second
repeat. A fixed recording ensured every model processed the same frames.
The ONNX Runtime version was 1.32.0. Absolute numbers are short shared-machine
measurements, not a sustained headless throughput guarantee.

| Model | File size | Detection + tracking FPS | Frame p95 | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 102.8 MiB | 0.454 | 2.24 s | 495 MiB |
| Static INT8 | 32.4 MiB | 0.657 | 1.55 s | 431 MiB |
| Dynamic INT8 | 32.2 MiB | 0.661 | 1.55 s | 429 MiB |

Detection used `people-walking.mp4`, with 12 measured frames after three warmup
frames per run, repeated twice. FPS/p95 are medians across runs; RSS is the maximum
process high-water mark. MQTT, JPEG encoding, recording, RTSP and rate-limit sleep
are excluded. These recording results include decode/prefetch memory and should
not be compared directly with the earlier live 640×480 webcam RSS figures.

| Model | Detection + tracking + every-frame ALPR FPS | Frame p95 | Peak RSS |
| --- | ---: | ---: | ---: |
| FP32 | 0.421 | 2.39 s | 406 MiB |
| Static INT8 | 0.577 | 1.76 s | 347 MiB |
| Dynamic INT8 | 0.583 | 1.74 s | 344 MiB |

The ALPR comparison used the fixed public plate sample recording, eight measured
frames after two warmups per run, repeated twice. Each model ran a plate scan on
every measured frame. Static INT8 improved detection throughput about 45% and the
full detector/tracker/ALPR path about 37%, with roughly 69% smaller model files.
This is still about 1.7 seconds per processed frame with ALPR, not real-time video.


Benchmark health samples reached 52.6°C, with `get_throttled=0x0` throughout
the recorded checks. The Pi runtime remained at 42 packages without the export or
quantization extras. Functional checks confirmed tracks and the same single plate
event from the public sample with FP32, static INT8 and dynamic INT8. Static INT8
also passed the Pi-to-MQTT-to-Home-Assistant sample run, including the snapshot
camera and exactly one plate event with the updated recognition timestamp. A
separate eight-frame live-webcam check (two warmups, four threads) measured
0.695 FPS and 318 MiB peak RSS for static INT8. That live check was not a paired
comparison; the fixed-recording rows above are the controlled speed comparison.


## Detection quality check

Calibration used 96 images; evaluation used 32 different images from
[COCO128](https://docs.ultralytics.com/datasets/detect/coco128), split with random
seed 42. The evaluation images were excluded from quantization calibration by
content hash. Both lists came from COCO **train2017**: they are not held out from
the pretrained detector's original training. This is a small regression diagnostic,
not COCO val2017 accuracy or evidence of performance on your camera/night scenes.

Supervision computed box mAP at IoU 0.50:0.95 using predictions above 0.001 and
explicit dataset-class-name mapping to exported labels. The person/vehicle subset
contains person, car, motorcycle, bus and truck, evaluated as individual classes.

| Model | All-class mAP50–95 | Person/vehicle mAP50–95 |
| --- | ---: | ---: |
| FP32 | 73.49 | 71.23 |
| Static INT8 | 72.21 | 74.24 |
| Dynamic INT8 | 71.78 | 72.72 |

Static INT8 lost 1.28 points overall, dynamic INT8 lost 1.71. The person/vehicle
scores increased on this small sample; that is not a general accuracy improvement
claim. Static INT8 is the preferred experimental candidate: similar speed to
dynamic with the smaller overall loss. Validate representative labelled camera
footage before choosing it for deployment. Even INT8 remains below video-rate
inference on this Pi 4 and does not establish fast-moving plate reliability.

## Reproduce quantization on a development computer

Use an existing metadata-bearing FP32 export. The optional `quantize` extra adds
ONNX/SymPy tools, without installing the training stack. A separate environment
keeps the application's normal environment unchanged:

```sh
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --locked --no-dev --extra quantize scripts/quantize_onnx.py \
  --model models/rfdetr-nano/rfdetr-nano.onnx \
  --output models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  --calibration-list work/calibration.txt --limit 100
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --locked --no-dev --extra quantize scripts/quantize_onnx.py \
  --model models/rfdetr-nano/rfdetr-nano.onnx \
  --output models/rfdetr-nano/rfdetr-nano-int8-dynamic.onnx --mode dynamic
```

The calibration text file contains one representative image path per line,
relative to that list file or absolute. Use real deployment-like images rather
than noise/blank frames. The tool preserves the source, writes a separate model
and JSON manifest, checks CPU execution before publishing the artifact, and refuses
to overwrite existing outputs. The manifest records source/output hashes,
calibration-image hashes, op counts, exclusions and tool versions. The original
label/preprocessing metadata is preserved; input/output tensors remain float.

For labelled evaluation, supply a disjoint image list, a directory of YOLO box
labels (empty label files for negatives), and a JSON list of class names in the
dataset's ordinal order. No Ultralytics runtime is required:

```sh
uv run scripts/evaluate_onnx.py --model models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  --images work/evaluation.txt --labels /path/to/labels --class-names work/class-names.json \
  --output work/eval-int8.json
```

The evaluator refuses overlap with the model manifest's calibration hashes.
Use the same evaluation set for FP32 and every candidate. Model files and datasets
remain ignored by Git; do not mistake a small successful calibration experiment
for an accuracy validation on unseen deployment data.

## Reproduce comparison on the Pi

Copy the INT8 ONNX file and its same-stem JSON manifest beside the FP32 model.
Keep `uv sync --locked --no-dev` without the `quantize` or `export` extras on the Pi.

```sh
uv run --locked --no-dev scripts/benchmark_models.py --config config/pi.local.yml \
  --models models/rfdetr-nano/rfdetr-nano.onnx models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  --source /path/to/representative-recording.mp4 --onnx-threads 4 \
  --frames 20 --warmup 3 --repeats 2 --lpr off --output-dir work/compare-detection
uv run --locked --no-dev scripts/benchmark_models.py --config config/pi.local.yml \
  --models models/rfdetr-nano/rfdetr-nano.onnx models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  --source /path/to/plate-recording.mp4 --onnx-threads 4 \
  --frames 20 --warmup 3 --repeats 2 --lpr every-frame --output-dir work/compare-plates
```

Use a new/empty output directory for each comparison. The tool saves raw per-run
reports/logs and a summary, and rejects incomplete runs. `--lpr every-frame`
measures the plate detector/OCR on every processed frame for an equal-work
comparison; configured cadence can scan different numbers of frames as inference
speed changes. Prepared plate model caches are required for offline tests.

The individual benchmark also accepts `--model` and `--onnx-threads` overrides.
To try INT8 in the normal camera service, change only `model_path` in a separate
local config and keep the original FP32 config for rollback. Existing thresholds,
tracking and hold settings still apply, but should be checked against changed
confidence scores. Do not enable a faster model solely on its file-size result.

For Conv-based Nano/Tiny models, use the separate [YOLOX INT8 guide](yolox-int8.md).
The RF-DETR linear-layer recipe described here does not apply to YOLOX.
