# Speed and Accuracy presets

Choose one preset in the camera configuration. Both use ONNX Runtime CPU,
Supervision, Roboflow ByteTrack, configurable presence holds, optional pretrained
plate detection/OCR and the same Home Assistant MQTT entities. No preset adds
runtime dependencies or requires PyTorch on the Pi.

FPS measured on a **Raspberry Pi 4 Model B (4 GB)**; mAP evaluated separately on
the development computer.

| Preset | Model | Detection + tracking FPS | Every-frame OCR FPS | mAP50 (%) |
| --- | --- | ---: | ---: | ---: |
| `speed` (new-config default) | YOLOX Nano FP32 | 4.33 | 1.75 | 64.92 |
| `accuracy` | RF-DETR Nano static INT8 | 0.65 | 0.57 | 88.55 |

These are short, matched-recording tests on a shared 4 GB Pi 4 with four ORT
threads, excluding MQTT/media output costs. Accuracy uses a 32-image COCO128
train2017 diagnostic, separate from calibration but not the models' original
training. They are not deployment guarantees. The [experiment record](experiments/README.md)
keeps full protocols, raw reports and all candidates, including Tiny and YOLOX INT8.
The [Pi 3 follow-up](experiments/pi3.md) records setup and power-limited results
on a 1 GB board; the table above remains the Pi 4 comparison.

## Select and run

```yaml
preset: speed # or accuracy
onnx_threads: 4
max_fps: 1 # publication/inference cap; benchmark before raising it
```

No model path is required. Files are resolved under the installed repository,
independent of the YAML location or working directory:

| Preset | Required file under the repository |
| --- | --- |
| Speed | `models/yolox-nano/yolox-nano.onnx` |
| Accuracy | `models/rfdetr-nano/rfdetr-nano-int8-static.onnx` |

Copy each same-stem JSON provenance manifest alongside the ONNX file. The service
checks embedded family, Nano variant and precision; it rejects Tiny, YOLOX INT8,
RF-DETR FP32 and RF-DETR dynamic INT8 when used with these presets. Preset selection never
silently falls back to another model or downloads detector weights at runtime.

```sh
uv sync --locked --no-dev
cp config/onnx.yml config/onnx.local.yml
# Edit preset, source, camera ID and MQTT settings; optionally enable lpr.enabled.
uv run --locked --no-dev scripts/onnx_camera.py --config config/onnx.local.yml
```

Changing the preset requires restarting the service. It does not change MQTT
entity IDs, plate models, thresholds, hold times or the configured FPS cap.
Tracker IDs reset on restart. Check thresholds and recall on actual camera footage.

## Prepare Speed on a development computer

Use the pinned official Apache-2.0 YOLOX source and Nano checkpoint. The optional
export stack stays in its own uv environment:

```sh
git clone https://github.com/Megvii-BaseDetection/YOLOX.git work/YOLOX
git -C work/YOLOX checkout --detach 6ddff4824372906469a7fae2dc3206c7aa4bbaee
mkdir -p models/yolox-nano
curl -L --fail https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -o models/yolox-nano/yolox_nano.pth
UV_PROJECT_ENVIRONMENT=.venv-yolox-export uv run --locked --no-dev --extra yolox-export scripts/export_yolox_onnx.py \
  --source-tree work/YOLOX --variant nano --checkpoint models/yolox-nano/yolox_nano.pth \
  --output models/yolox-nano/yolox-nano.onnx --verify-images /path/to/scene.jpg
```

The exporter embeds preprocessing/label metadata and verifies raw predictions and
final detections against upstream PyTorch. Source licenses, helper provenance and
checkpoint hashes are retained in the [original experiment](experiments/yolox-fp32.md).

## Prepare Accuracy on a development computer

Export RF-DETR Nano, then statically calibrate its linear layers with representative
images. These commands use separate export and quantization environments:

```sh
UV_PROJECT_ENVIRONMENT=.venv-rfdetr-export uv run --locked --no-dev --extra export scripts/export-rfdetr-onnx.py
UV_PROJECT_ENVIRONMENT=.venv-quantize uv run --locked --no-dev --extra quantize scripts/quantize_onnx.py \
  --model models/rfdetr-nano/rfdetr-nano.onnx \
  --output models/rfdetr-nano/rfdetr-nano-int8-static.onnx \
  --mode static --calibration-list /path/to/calibration.txt --limit 96
```

The list contains one image path per line, absolute or relative to the list.
This is the attributed RF-DETR linear-layer recipe, retaining sensitive attention
and output layers in float: Accuracy is mixed FP32/INT8, not integer-only.
The quantizer preserves metadata, refuses overwrites and verifies executable CPU
inference. See [calibration and accuracy evaluation](experiments/rfdetr-int8.md)
for the original data split and reproduction commands. Use an independent
labelled set for validating a new calibration or fine-tuned model.

Copy only the chosen ONNX file and JSON manifest to the matching Pi model directory.
Never copy a desktop virtual environment or install export/quantization extras on
the Pi. See [headless deployment](raspberry-pi.md) for uv/systemd and MQTT setup.

## Existing configurations and custom artifacts

Existing configurations containing only `model_path` keep their metadata-selected
behavior. An explicit `model_path` without `preset` is also available for custom
models and experimental tooling; relative paths resolve beside the YAML file.
For a relocated or fine-tuned artifact of a supported preset, set both `preset`
and `model_path`: the same family/variant/precision checks still apply. Adjust
`classes` to the exported labels when fine-tuning; plate recognition remains its
own independent detector/OCR pipeline.

`benchmark.py --model ...` deliberately clears preset selection for explicit
artifact comparisons. The application and benchmark JSON include the selected
`preset` (null for custom/legacy paths) alongside the actual model filename.
