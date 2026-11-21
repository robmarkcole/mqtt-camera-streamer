# mqtt-camera-streamer

Run USB webcam or RTSP frames through RF-DETR or YOLOX ONNX, track detections with
Roboflow ByteTrack/Supervision, and publish predictions to Home Assistant over
MQTT. Optional pretrained license plate detection and OCR run locally on CPU.

The runtime uses ONNX Runtime directly, with no PyTorch, TorchVision, Transformers
or detector training package. Supervision, Roboflow ByteTrack and plate recognition
remain available. Model export uses the optional `export` (RF-DETR) or
`yolox-export` extra on a development computer.

The inference service is headless. Home Assistant provides the dashboard and
automations; a viewer is not required on the camera computer.

## Quick start with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root. Python 3.12 is pinned in `.python-version`.

```sh
uv sync --locked
uv run --extra export scripts/export-rfdetr-onnx.py
cp config/onnx.yml config/onnx.local.yml
```

Edit the local file's camera source, broker host/port and camera ID. Verify the
external webcam index rather than assuming it is `0`:

```sh
uv run scripts/check-opencv-cameras.py
```

Set `MQTT_USERNAME` and `MQTT_PASSWORD` in the environment when authentication is
required. These are broker credentials, not Home Assistant API tokens. The
[Home Assistant Mosquitto app](https://github.com/home-assistant/app-mosquitto)
requires credentials. Configure Home Assistant's MQTT integration on the same
broker with discovery enabled.

```sh
uv run scripts/onnx_camera.py --config config/onnx.local.yml
# Stop after ten predictions for a smoke test:
uv run scripts/onnx_camera.py --config config/onnx.local.yml --max-frames 10
```

One device per camera appears in Home Assistant with person/vehicle presence,
count entities and a snapshot camera. Sparse JPEGs over MQTT are the default;
RTSP output and local file-backed recording are optional. Two-frame track confirmation and configurable presence holds
reduce flickering; counts remain instantaneous. Configuration and model files
with local settings/weights are ignored by Git.

## Guides

| Task | Guide |
| --- | --- |
| Headless Raspberry Pi, uv and systemd | [Raspberry Pi deployment](docs/raspberry-pi.md) |
| Configuration, tracking and MQTT discovery | [ONNX and Home Assistant](docs/onnx-home-assistant.md) |
| MQTT snapshots, RTSP, SSH preview and file-backed SQLite | [Images and RTSP](docs/images-and-rtsp.md) |
| Pretrained plate detection and OCR | [License plates](docs/license-plates.md) |
| YOLOX Nano/Tiny, export and Pi comparisons | [YOLOX models](docs/yolox.md) |
| INT8 quantization and Pi comparisons | [RF-DETR INT8](docs/int8.md), [YOLOX INT8](docs/yolox-int8.md) |
| Dependency reduction and deployment ideas | [Dependencies and performance](docs/dependencies.md) |

## License plates

Cache the pretrained ONNX models, enable `lpr.enabled` in the local configuration,
and run the same service:

```sh
uv run scripts/download-plate-models.py --config config/onnx.local.yml
uv run scripts/onnx_camera.py --config config/onnx.local.yml
```

Confirmed readings update Last plate, timestamp and confidence sensors. A
non-retained `mqtt_camera/<camera_id>/plate/events` message drives automations.
Repeated matching OCR readings and a cooldown suppress duplicate events.

## Benchmark

Use a representative recording to compare machines without opening a webcam or
requiring an MQTT broker:

```sh
uv run scripts/benchmark.py --config config/onnx.local.yml \
  --source /path/to/recording.mp4 --warmup 5 --frames 100 \
  --lpr off --output work/benchmark-detection.json
uv run scripts/benchmark.py --config config/onnx.local.yml \
  --source /path/to/recording.mp4 --warmup 5 --frames 100 \
  --lpr every-frame --output work/benchmark-plates.json
```

Reports include warm steady-state p50/p95 stage latency, observed pipeline FPS,
model load time, model hash, versions, CPU time and peak process RSS. Pi telemetry
is included when available. This measures capture/decode plus inference and
tracking, with optional plate scans; MQTT, JPEG encoding, recording, RTSP output
and the service's rate-limit sleep are excluded. These results are not model-only FPS or a promise of Pi performance.

## Legacy frame streaming examples

The original scripts publish complete PNG frames at low frame rates, and can
rotate or save frames received through MQTT. Their configuration is
`config/config.yml`, separate from the inference service's configuration.
`MQTT_CAMERA_CONFIG` overrides the legacy configuration path.

```sh
# Check the legacy YAML can be loaded (does not validate its values):
uv run scripts/validate-config.py
# Capture and publish complete camera frames:
uv run --extra legacy scripts/opencv-camera-publish.py
# Rotate received images onto another topic:
uv run scripts/processing.py
# Save received frames to disk or SQLite:
uv run scripts/save-captures.py
uv run scripts/db-recorder.py
```

Create the configured capture directory before saving images. Set the legacy
broker host/port in `config/config.yml`; these scripts also use `MQTT_USERNAME`
and `MQTT_PASSWORD`. A raw frame topic such as `homie/mac_webcam/capture` can be
shown using Home Assistant's [MQTT camera](https://www.home-assistant.io/integrations/camera.mqtt/).

The SQLite browser is an optional workstation tool:

```sh
# Browse recorded metadata and image paths:
uv run --extra database datasette captures/records.db
```

Stop these foreground commands with Ctrl-C. A CSI camera requires its own modern
Picamera2/libcamera capture path or a compatible network stream; this project
does not directly implement CSI capture. The Pi guide covers USB V4L2 webcams.

## Development

```sh
uv sync --locked
uv run pytest -q
uv run ruff check scripts/onnx_camera.py scripts/plates.py scripts/media.py scripts/benchmark.py tests
```

Dependencies live in `pyproject.toml` and `uv.lock`. Use `uv add`, `uv remove` or
`uv lock` to change them and commit the lockfile. `uv sync --locked --no-dev`
prepares a runtime environment; repeat `--no-dev` with `uv run` on that host.
Streamlit has been removed. Imutils and Datasette are opt-in extras. Screen
capture and YouTube download dependencies are not part of this deployment profile.
Install model export extras only when exporting or comparing models; use
`uv sync --locked --no-dev` on the Pi to exclude them and development tools.
RTSP output reuses PyAV (already used by Supervision), with no FFmpeg executable
required; it needs an RTSP server such as the optional standalone MediaMTX binary.

## License

See [LICENSE](LICENSE). Pretrained model provenance and upstream links are in the
plate and dependency guides.
