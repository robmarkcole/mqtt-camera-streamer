# RF-DETR ONNX webcam detection with Home Assistant

The detector reuses RF-DETR's ONNX prediction and live camera APIs and passes its
Supervision `Detections` through class filtering. It publishes predictions and
Home Assistant MQTT discovery without needing a Roboflow account or API key.
Model download/export needs internet access; inference uses the local ONNX file.

## Install and export

Run these commands from the repository root with Python 3.12:

```sh
python3.12 -m venv .venv-onnx
source .venv-onnx/bin/activate
python -m pip install -r requirements-onnx.txt
python scripts/export-rfdetr-onnx.py
```

The export script downloads pretrained RF-DETR Nano and writes
`models/rfdetr-nano/rfdetr-nano.onnx`, including the metadata used by the official
`RFDETRInference` API. Model files and virtual environments are ignored by Git.
The upstream RF-DETR package currently also installs PyTorch, even though this
service runs its exported model through ONNX Runtime on CPU.

## Configure and run

Copy `config/onnx.yml` to `config/onnx.local.yml` and edit:

- `source`: your external webcam index. Do not assume USB is always camera 0.
  On macOS, grant the launching application camera access in System Settings.
  RF-DETR also accepts video paths and RTSP URLs. Paths/URLs containing credentials
  should stay in the ignored local configuration.
- `camera_id`: a stable, unique ID for this camera; use a different ID per service.
- `mqtt.broker` and `mqtt.port`: the broker configured in Home Assistant's MQTT
  integration. The service does not connect directly to Home Assistant's HTTP API.
- `classes`: entity keys mapped to model labels. Defaults provide person and
  vehicle groups. Unknown labels fail at startup rather than silently reporting 0.
- `threshold`, `max_fps`, and `stale_seconds`: confidence, maximum publication /
  inference rate, and the time without results before entities become unavailable.

Model paths are relative to the YAML file. Run from the repository root for
relative video sources. If authentication is required, set `MQTT_USERNAME` and
`MQTT_PASSWORD` in your environment; do not put secrets in committed YAML. Enable
`mqtt.tls` for TLS using the system trust store, and set the broker's TLS port.

```sh
python scripts/onnx_camera.py --config config/onnx.local.yml
# Bounded smoke test:
python scripts/onnx_camera.py --config config/onnx.local.yml --max-frames 10
```

`MQTT_CAMERA_ONNX_CONFIG` can set the default config path. Ctrl-C or SIGTERM closes
the camera stream, publishes offline availability, and disconnects MQTT. Camera
read recovery uses RF-DETR's stream handling; an exhausted retry or inference
error exits the service. A process supervisor can restart it.

The RF-DETR source generator keeps the latest live frame (`stream_buffer=False`)
instead of queuing frames while inference runs. Lower `max_fps` reduces inference
and MQTT load; it does not set the webcam's hardware capture rate. Raspberry Pi
throughput is not established: use 64-bit Linux, start at a low rate, and measure
on the target board. Tracking and OCR are outside this first milestone.

## Home Assistant and MQTT

Enable Home Assistant's MQTT integration on the same broker, with discovery
enabled. One device named `USB camera` appears with person/vehicle occupancy
binary sensors and count sensors. Counts describe objects visible in the current
frame, not unique visitors. A zero-detection frame clears previous counts.

Under `mqtt_camera/<camera_id>/` the service publishes:

| Topic | Payload | Retained |
| --- | --- | --- |
| `predictions` | JSON detections, labels, scores and pixel `box_xyxy` coordinates | No |
| `state` | JSON counts, sequence, model filename, prediction time, processing duration | No |
| `availability` | `online` or `offline` | Yes |

`predicted_at` is UTC prediction completion time, not camera exposure time.
`processing_ms` includes obtaining the next result and inference; it excludes
the rate-limit sleep. Bounding boxes refer to original source-image pixels.
The schema carries `schema_version: 1`; frame sequence resets on service restart.
Transient predictions use QoS 0 to avoid replaying obsolete observations after
broker outages. Discovery and availability use QoS 1. State is periodically
refreshed even when unchanged. Per-frame boxes stay out of entity attributes and
Home Assistant's recorder.

Discovery configurations are retained under `homeassistant/.../config`, have
stable unique IDs, and are republished on MQTT reconnect and HA's `online` birth
message. The service stays offline until it has a result, marks stalled inference
offline, and uses MQTT last will for unexpected disconnects. Entity state also
expires after `stale_seconds`. On restart HA waits for fresh state rather than
restoring an old detection. If you rename a camera/group, remove the old retained
discovery topics or device entries; automatic entity migration is not implemented.

Use the discovered person binary sensor's actual entity ID in an automation:

```yaml
alias: Webcam person detected
triggers:
  - trigger: state
    entity_id: binary_sensor.usb_camera_person_detected
    from: "off"
    to: "on"
actions:
  - action: persistent_notification.create
    data:
      title: Webcam
      message: A person was detected.
```

Adjust the entity ID in Home Assistant; its generated name can differ. This
milestone uses frame-level presence, so confidence fluctuations may toggle it.
Tracking and configurable hold times can be added next.

## Verify changes

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_onnx_camera.py
ruff check scripts/onnx_camera.py scripts/export-rfdetr-onnx.py tests/test_onnx_camera.py
```

Tests use real Supervision objects and an ephemeral MQTT broker bound to loopback.
They verify discovery, zero-detection state, non-retained predictions, HA birth
handling, stale availability, and orderly shutdown without needing a camera or HA.

Upstream APIs: [RF-DETR ONNX](https://rfdetr.roboflow.com/latest/exports/onnx/),
[live streams](https://rfdetr.roboflow.com/latest/learn/run/videos-and-streams/),
[Supervision](https://supervision.roboflow.com/latest/), and
[Home Assistant MQTT discovery](https://www.home-assistant.io/integrations/mqtt/#mqtt-discovery).
