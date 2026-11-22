# ONNX webcam detection with Home Assistant

Choose `preset: speed` (YOLOX Nano FP32) or `preset: accuracy` (RF-DETR Nano
static INT8). [Model setup](models.md) covers artifact preparation and the measured
tradeoff. Both use ONNX Runtime CPU and emit Supervision detections into the same
class filtering, Roboflow ByteTrack, presence-hold, plate and MQTT pipeline.

The runtime excludes PyTorch, TorchVision, Transformers and detector training
packages. Small attributed preprocessing/decoding and bounded capture helpers
are retained under `scripts/vendor`. No Roboflow account or API key is required.

## Configure and run

Copy `config/onnx.yml` to `config/onnx.local.yml` and edit:

- `preset`: `speed` (default for new configs) or `accuracy`. Default model paths
  are relative to the installed repository; see [model setup](models.md).
- `source`: your external webcam index. Do not assume USB is always camera 0.
  On macOS, grant the launching application camera access in System Settings.
  Video paths and RTSP URLs are also accepted. Paths/URLs containing credentials
  should stay in the ignored local configuration.
- `camera_id`: a stable, unique ID for this camera; use a different ID per service.
- `mqtt.broker` and `mqtt.port`: the broker configured in Home Assistant's MQTT
  integration. The service does not connect directly to Home Assistant's HTTP API.
- `classes`: entity keys mapped to model labels. Defaults provide person and
  vehicle groups. Unknown labels fail at startup rather than silently reporting 0.
- `threshold`, `max_fps`, and `stale_seconds`: confidence, maximum publication /
  inference rate, and the time without results before entities become unavailable.
- `onnx_threads`: ONNX Runtime intra-op CPU inference threads (default 2).
  The old `torch_threads` key is accepted as a migration alias, but now controls
  ORT rather than Torch preprocessing; rebenchmark migrated configurations.
- `tracking`: track confirmation, recovery time and low-confidence association.
- `presence`: the default hold duration and optional overrides per entity group.

Existing explicit `model_path` configurations still work; those paths are relative
to the YAML file. A preset plus an override path validates the selected model.
Run from the repository root for relative video sources. If authentication is required, set `MQTT_USERNAME` and
`MQTT_PASSWORD` in your environment; do not put secrets in committed YAML. Enable
`mqtt.tls` for TLS using the system trust store, and set the broker's TLS port.

```sh
uv run scripts/onnx_camera.py --config config/onnx.local.yml
# Bounded smoke test:
uv run scripts/onnx_camera.py --config config/onnx.local.yml --max-frames 10
```

`MQTT_CAMERA_ONNX_CONFIG` can set the default config path. Ctrl-C or SIGTERM closes
the camera stream, publishes offline availability, and disconnects MQTT. Camera
read recovery uses RF-DETR's stream handling; an exhausted retry or inference
error exits the service. A process supervisor can restart it.

The RF-DETR source generator keeps the latest live frame (`stream_buffer=False`)
instead of queuing frames while inference runs. Lower `max_fps` reduces inference
and MQTT load; it does not set the webcam's hardware capture rate. Raspberry Pi
performance depends on the workload: use 64-bit Linux, start at a low rate, and
measure on the target board. [Recorded Pi results](models.md) compare both presets.
See [license plate detection and OCR](license-plates.md) to enable the optional pretrained plate pipeline and Home Assistant plate sensors.

## Tracking and presence hold times

The service uses `ByteTrackTracker` from Roboflow's `trackers` package. The default
requires two consecutive observations before a track contributes to counts or
presence. An isolated one-frame detection does not activate a binary sensor.
The detector's `threshold` is the minimum confidence for starting a new track;
`tracking.recovery_threshold` admits weaker detections only for association with
an existing track. Unconfirmed or unmatched tracker IDs (`-1`) are filtered out.

```yaml
tracking:
  enabled: true
  minimum_consecutive_frames: 2
  lost_track_seconds: 2
  recovery_threshold: 0.1
presence:
  hold_seconds: 5
  hold_seconds_by_class:
    person: 10
    vehicle: 15
```

`lost_track_seconds` controls how long ByteTrack can recover an object's ID after
missed observations. The service supplies monotonic timestamps so recovery uses
elapsed time even when inference slows down. IDs are local to a camera process;
they are not permanent identities and reset after a stale-stream outage.

Presence stays on until the configured hold has elapsed since the last confirmed
observation in that group. A reappearing object, including one with a new tracker
ID, extends the same presence episode without another off-to-on transition.
Overrides use keys from `classes` (such as `vehicle`), not individual model labels.
Unknown keys, negative durations and non-finite settings fail validation. Set
`hold_seconds: 0` for immediate clearing; set `tracking.enabled: false` to bypass
track confirmation while retaining presence holds.

Holds are evaluated on each new prediction, so expiry is rounded to the next
processed frame. No background state refresh hides camera failures: stale streams
still become unavailable, and resuming after an outage clears old tracks and holds.
Ordinary MQTT reconnects preserve the ongoing tracking/presence episode.

## Home Assistant and MQTT

Enable Home Assistant's MQTT integration on the same broker, with discovery
enabled. One device named `USB camera` appears with person/vehicle occupancy
binary sensors and count sensors. Counts describe confirmed objects visible in
the current frame, not unique visitors. A zero-detection frame clears counts;
presence may remain on until its hold expires.

Under `mqtt_camera/<camera_id>/` the service publishes:

| Topic | Payload | Retained |
| --- | --- | --- |
| `predictions` | JSON confirmed detections, tracker IDs, labels, scores and pixel `box_xyxy` coordinates | No |
| `state` | JSON instantaneous counts, held presence, sequence, model filename, prediction time, processing duration | No |
| `availability` | `online` or `offline` | Yes |

`predicted_at` is UTC prediction completion time, not camera exposure time.
`processing_ms` includes obtaining the next result and detector inference; it
excludes the rate-limit sleep and optional plate stage. Bounding boxes refer to
original source-image pixels. `preset` identifies the selection (`speed`, `accuracy`,
or null for custom/legacy paths); `model` records the actual ONNX filename.
The schema carries `schema_version: 1`; `presence` and per-detection `tracker_id`
are additive fields. A tracker ID is `null` when tracking is disabled. Frame
sequence resets on service restart. Binary sensor discovery reads `presence`;
count sensors continue to read `counts`. Existing entity IDs stay the same.
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

Adjust the entity ID in Home Assistant; its generated name can differ. This fires
once per presence episode. Brief detection gaps within the hold do not create new
transitions; a confirmed detection after the hold has expired begins a new episode.

## Verify changes

```sh
uv sync --locked
uv run pytest -q
uv run ruff check scripts/onnx_camera.py scripts/export-rfdetr-onnx.py tests
```

Tests use real Supervision objects and an ephemeral MQTT broker bound to loopback.
They verify discovery, zero-detection state, non-retained predictions, HA birth
handling, stale availability, and orderly shutdown without needing a camera or HA.
Tracking tests use the real Roboflow tracker and simulated timestamps to verify
confirmation, weak-detection recovery, per-group holds, outage reset and one stable
presence episode across missing frames.

Use `uv sync --locked --no-dev` and `uv run --no-dev ...` on a runtime-only host.
Legacy tools are optional uv extras described in the README. Update dependencies
with `uv lock` and commit the resulting `uv.lock` together with any
`pyproject.toml` changes. See the [Pi deployment guide](raspberry-pi.md) for the
systemd environment and inference benchmark.

Upstream APIs: [RF-DETR ONNX](https://rfdetr.roboflow.com/latest/exports/onnx/),
[live streams](https://rfdetr.roboflow.com/latest/learn/run/videos-and-streams/),
[Supervision](https://supervision.roboflow.com/latest/),
[Roboflow tracking](https://supervision.roboflow.com/latest/how_to/track_objects/), and
[Home Assistant MQTT discovery](https://www.home-assistant.io/integrations/mqtt/#mqtt-discovery).

## Images and recordings

Sparse MQTT JPEGs now discover a Snapshot camera by default. Use `images.transport: rtsp`
for optional PyAV RTSP output, or `off` for metadata only. Optional `storage` writes
JPEGs plus relative paths/predictions in SQLite. See [images and RTSP](images-and-rtsp.md)
for configuration, retention, SSH preview and the additional RTSP server.
