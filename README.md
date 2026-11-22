# mqtt-camera-streamer

Run a USB webcam or RTSP source through one of two ONNX presets:

| Preset | Model | Purpose |
| --- | --- | --- |
| **Speed** (default for new configs) | YOLOX Nano FP32 | Faster person/vehicle detection on the Pi |
| **Accuracy** | RF-DETR Nano static INT8 | Higher measured detection accuracy |

Both share Supervision, Roboflow ByteTrack, configurable presence holds, optional
pretrained plate recognition and Home Assistant MQTT. The headless Pi runtime
has no PyTorch, TorchVision or detector training package. Sparse MQTT JPEGs are
the default; RTSP and disk-backed recording are optional.

## Quick start with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root. Python 3.12 is pinned in `.python-version`. Prepare or
copy the chosen ONNX model using [model setup](docs/models.md).

```sh
uv sync --locked --no-dev
cp config/onnx.yml config/onnx.local.yml
```

Edit `preset: speed` or `preset: accuracy`, camera source, broker host/port and
camera ID in the local file. Verify the
external webcam index rather than assuming it is `0`:

```sh
uv run --locked --no-dev scripts/check-opencv-cameras.py
```

Set `MQTT_USERNAME` and `MQTT_PASSWORD` in the environment when authentication is
required. These are broker credentials, not Home Assistant API tokens. The
[Home Assistant Mosquitto app](https://github.com/home-assistant/app-mosquitto)
requires credentials. Configure Home Assistant's MQTT integration on the same
broker with discovery enabled.

```sh
uv run --locked --no-dev scripts/onnx_camera.py --config config/onnx.local.yml
# Stop after ten predictions for a smoke test:
uv run --locked --no-dev scripts/onnx_camera.py --config config/onnx.local.yml --max-frames 10
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
| Speed/Accuracy selection and model preparation | [Model presets](docs/models.md) |
| All model and quantization experiments | [Experiment record](docs/experiments/README.md) |
| Dependency reduction and deployment ideas | [Dependencies and performance](docs/dependencies.md) |
| Original MQTT frame-streaming scripts | [Legacy examples](docs/legacy.md) |

## License plates

Cache the pretrained ONNX models, enable `lpr.enabled` in the local configuration,
and run the same service:

```sh
uv run --locked --no-dev scripts/download-plate-models.py --config config/onnx.local.yml
uv run --locked --no-dev scripts/onnx_camera.py --config config/onnx.local.yml
```

Confirmed readings update Last plate, timestamp and confidence sensors. A
non-retained `mqtt_camera/<camera_id>/plate/events` message drives automations.
Repeated matching OCR readings and a cooldown suppress duplicate events.

## Benchmark

Use a representative recording to compare machines without opening a webcam or
requiring an MQTT broker:

```sh
uv run --locked --no-dev scripts/benchmark.py --config config/onnx.local.yml \
  --source /path/to/recording.mp4 --warmup 5 --frames 100 \
  --lpr off --output work/benchmark-detection.json
uv run --locked --no-dev scripts/benchmark.py --config config/onnx.local.yml \
  --source /path/to/recording.mp4 --warmup 5 --frames 100 \
  --lpr every-frame --output work/benchmark-plates.json
```

Reports include warm steady-state p50/p95 stage latency, observed pipeline FPS,
model load time, model hash, versions, CPU time and peak process RSS. Pi telemetry
is included when available. This measures capture/decode plus inference and
tracking, with optional plate scans; MQTT, JPEG encoding, recording, RTSP output
and the service's rate-limit sleep are excluded. These results are not model-only FPS or a promise of Pi performance.

## Development

```sh
uv sync --locked
uv run pytest -q
uv run ruff check scripts/onnx_camera.py scripts/onnx_detector.py scripts/yolox_detector.py scripts/plates.py scripts/media.py scripts/benchmark.py tests
```

Dependencies live in `pyproject.toml` and `uv.lock`. Use `uv add`, `uv remove` or
`uv lock` to change them and commit the lockfile. `uv sync --locked --no-dev`
prepares a runtime environment; repeat `--no-dev` with `uv run` on that host.
Imutils and Datasette remain opt-in legacy/workstation extras. No web viewer,
screen capture or YouTube download package is required by the runtime.
Install model export extras only when exporting or comparing models; use
`uv sync --locked --no-dev` on the Pi to exclude them and development tools.
RTSP output reuses PyAV (already used by Supervision), with no FFmpeg executable
required; it needs an RTSP server such as the optional standalone MediaMTX binary.

## License

See [LICENSE](LICENSE). Pretrained model provenance and upstream links are in the
plate and dependency guides.
