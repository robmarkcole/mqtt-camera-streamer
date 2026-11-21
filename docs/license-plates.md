# Local license plate detection and OCR

The optional plate stage uses pretrained ONNX models on CPU:

- **Detection:** `yolo-v9-t-384-license-plate-end2end`, through Open Image Models.
- **OCR:** `cct-xs-v2-global-model`, through FastPlateOCR.
- **Pipeline:** FastALPR handles detector preprocessing, cropping and OCR.
- **Tracking:** Roboflow ByteTrack and Supervision maintain plate tracks and data.

RF-DETR continues to provide person/vehicle detection. The plate stage scans the
same original-resolution RGB source frame, converts it to FastALPR's BGR input,
and does not depend on a vehicle being detected first. Plate tracker IDs are
separate from person/vehicle tracker IDs; no vehicle-to-plate association is
claimed. Images and crops are not uploaded or saved by this stage.

## Enable

From the repository root:

```sh
uv sync --locked
uv run scripts/download-plate-models.py --config config/onnx.local.yml
```

The preparation command downloads/caches both models using the upstream model
hubs. Inference works locally after the files are cached. No Roboflow API key is
required. The model names can be changed to other supported upstream hub models;
their input/output handling remains in the libraries.

Add or update the following section in your local configuration:

```yaml
lpr:
  enabled: true
  detector_model: yolo-v9-t-384-license-plate-end2end
  ocr_model: cct-xs-v2-global-model
  detection_threshold: 0.5
  ocr_threshold: 0.8
  interval_seconds: 1
  lost_track_seconds: 5
  minimum_readings: 2
  agreement: 0.75
  confirmation_window_seconds: 5
  cooldown_seconds: 30
  plate_pattern: '[A-Z0-9]{3,10}'
  num_threads: 2
```

Then run the normal camera service:

```sh
uv run scripts/onnx_camera.py --config config/onnx.local.yml
```

`interval_seconds` limits how often plate detection/OCR runs; it does not change
ordinary person/vehicle inference. Actual scan frequency is also limited by
`max_fps` and processing speed. Keep the track recovery budget and confirmation
window comfortably above the scan interval. Two CPU threads per plate model are
used by default. Raspberry Pi performance and real-world plate accuracy still
need to be measured on the intended hardware and camera scene.

Invalid settings or missing models fail explicitly. A model inference error exits
the service, and the shared camera availability becomes offline. Increasing the
scan interval reduces work without publishing synthetic healthy updates.

## Confirmation and duplicate suppression

Plate boxes first need two observations to receive a confirmed ByteTrack ID.
Valid OCR readings are collected for that track. The default requires at least
two matching readings with at least 75% agreement within five seconds. Starting
from a new track, this normally takes three scans: one to start the track and two
readings from confirmed observations.

Text is uppercased and spaces/hyphens removed. The regular expression must match
the whole result. The default accepts three to ten ASCII letters/digits; adjust
it for your plate formats. `ocr_threshold` applies to the least-confident
character, not the average. Ambiguous characters such as O/0 are not guessed or
silently replaced. The event's OCR confidence is the strongest matching reading's
minimum character confidence, not a calibrated probability of correctness.

Each confirmed track emits once. A parked vehicle does not periodically retrigger
when the cooldown expires. The text cooldown also suppresses duplicate events
if tracking fragments into a new ID. A new track after that cooldown can emit a
new event for the same plate. Outages clear pending consensus; cooldowns remain
in memory to avoid immediate duplicate events on recovery. Histories are bounded.
These states reset on process restart, so cross-process deduplication is not
provided.

## Home Assistant and MQTT

When enabled, discovery adds three sensors to the existing camera device:

- **Last plate:** the last confirmed plate text.
- **Plate recognized at:** UTC confirmation timestamp.
- **Plate confidence:** confidence percentage (diagnostic entity).

The entities are initially unknown until a confirmed reading exists. These are
historical observations: an empty frame does not clear the last plate. They use
camera availability but do not have the occupancy sensors' state expiry.

| Topic under `mqtt_camera/<camera_id>/` | Contents | Retained / QoS |
| --- | --- | --- |
| `plate` | Latest confirmed observation | Yes / 1 |
| `plate/events` | New confirmed observation for automations | No / 0 |

Both payloads include `schema_version`, `event_id`, `camera_id`, `plate`,
`recognized_at`, `tracker_id`, `box_xyxy`, `ocr_confidence`,
`detection_confidence`, `matching_readings`, and model names. The camera loop adds
`frame_sequence` to connect an event with its source prediction.

Latest state survives Home Assistant restarts through retention. The running
publisher also restores its cached last plate after a broker reconnect. Events
are not replayed after reconnect or HA birth messages. QoS 0 events can be lost
during connection failures; this is not a durable event log. Runtime deduplication
continues while MQTT is disconnected. A confirmation during a broker outage may
restore the latest sensor state on reconnect without producing an automation.

Use the event topic for repeated visits, since a sensor showing the same plate
text may not change between visits:

```yaml
alias: Plate recognized
triggers:
  - trigger: mqtt
    topic: mqtt_camera/usb_camera/plate/events
actions:
  - action: persistent_notification.create
    data:
      title: License plate recognized
      message: "{{ trigger.payload_json.plate }}"
```

Disabling LPR stops scans and events. Remove its retained discovery topics or
device entities if you also want to remove the historical plate sensors.

## Sources and model provenance

- [FastALPR usage and models](https://github.com/ankandrew/fast-alpr)
- [Pretrained detector hub](https://github.com/ankandrew/open-image-models)
- [Pretrained OCR models](https://github.com/ankandrew/fast-plate-ocr)
- [Roboflow tracking API](https://supervision.roboflow.com/latest/how_to/track_objects/)

The upstream projects describe their model training, available alternatives and
licenses. Model files are cached rather than committed to this repository.
