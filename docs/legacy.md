# Legacy MQTT frame streaming

Run these examples from the repository root. For the current inference service,
see [the README](../README.md).

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
