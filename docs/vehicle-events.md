# Vehicle events and Home Assistant

The local pipeline uses Roboflow tracking, ONNX detection and FastALPR OCR. It
publishes useful vehicle/plate observations through the existing MQTT broker;
Home Assistant discovers a native MQTT event entity. No cloud token, custom HA
integration or additional Pi package is required.

## What was reused from the earlier integration

Reviewed [HASS-plate-recognizer](https://github.com/robmarkcole/HASS-plate-recognizer)
and its [event implementation](https://github.com/robmarkcole/HASS-plate-recognizer/blob/main/custom_components/platerecognizer/image_processing.py)
on 2026-11-22. It fires `platerecognizer.vehicle_detected` for each returned vehicle
on each scan, carrying plate text, confidence, region code, vehicle type, plate-box
centre and entity ID. It also exposes watched plates and annotated images;
orientation is obtained from its optional external API response.

This project reuses the useful event/box/watchlist ideas with local models:

| Capability | Local implementation |
| --- | --- |
| Plate and confidence | Consensus-confirmed FastALPR result; confidence remains 0–1 |
| Vehicle type and track | General detector label and confirmed Roboflow track ID |
| Plate-to-vehicle link | Smallest confirmed vehicle box containing the plate centre; null when no match |
| Box centre | `box_x_centre`, `box_y_centre` in source pixels; `box_centre_normalized` in source-relative coordinates |
| Watched plates | Exact normalized matches with optional friendly aliases |
| Annotated images | Supervision boxes and labels on a copy of RGB output |
| Region, orientation, make/model/colour | Not inferred by these models; omitted |

A vehicle label is a prediction, not a make/model identity. In the stock example,
Nano labels the foreground car as `truck`; that actual result is shown unchanged.
Detection/OCR confidence is not an accuracy guarantee. Neither image processing
nor the event fields establish physical direction, speed, ownership or identity.

## Events and topics

Under `mqtt_camera/<camera_id>/`:

| Event | When it fires | Topic |
| --- | --- | --- |
| `vehicle_detected` | Once when a confirmed vehicle track is first observed | `events` and `vehicle/events` |
| `vehicle_lost` | A previously confirmed track is no longer observed for `tracking.lost_track_seconds` | `events` and `vehicle/events` |
| `plate_recognized` | Repeated matching OCR confirms a plate, subject to per-track suppression and cooldown | `events` and existing `plate/events` |
| `watched_plate_detected` | A newly confirmed plate exactly matches the configured watchlist | `events` |

Events carry `event_id`, `event_type`, `occurred_at`, `camera_id` and `demo`.
New event publication also identifies the selected preset and actual model.
A plate event retains its original `recognized_at`, OCR/plate confidence, plate
box and plate tracker ID; vehicle association uses a separate `vehicle_tracker_id`.
The watched event has its own ID and links back through `plate_event_id`.

`vehicle_lost` means the track was lost, not proof that a vehicle physically left
a property. Presence can remain held after it, according to `presence.hold_seconds`.
Short gaps recover the same track without a new detection event. A stale stream
resets tracking without inventing lost/exit events. Track IDs are local to a
process and reset on restart. Vehicle lifecycle events require tracking enabled; `classes.vehicle` selects the
labels that produce these events.

Transient events use QoS 0, are never retained and have no application reconnect
backlog. They are best-effort signals, not guaranteed exactly-once delivery.
The `plate` topic retains the latest historical plate observation; snapshots and
MQTT discovery also retain their latest data. Historical event/plate entities
remain readable when the camera is offline. A reconnect republishes state and
discovery, not old automation events.

## Configure watched plates and images

```yaml
tracking:
  enabled: true
  minimum_consecutive_frames: 2
  lost_track_seconds: 3
presence:
  hold_seconds: 10
lpr:
  enabled: true
  interval_seconds: 2
  minimum_readings: 2
  watched_plates:
    AB12CDE: Family car
images:
  transport: mqtt
  annotate: true
```

Plate keys are uppercased with spaces/hyphens removed; duplicate normalized keys
are rejected. Matching is exact after normalization. Fuzzy substitutions are not
used to trigger a watched-plate event. Repeated OCR, confidence thresholds,
confirmation windows and the existing cooldown still apply. The watchlist is
optional; no plates are preconfigured in the live example.

Annotations affect MQTT snapshots, RTSP output and stored images. General
inference and OCR retain the original RGB pixels. A plate confirmation forces a
snapshot with its plate box; normal snapshots show the current object tracks.
Set `images.annotate: false` to send plain images.

## Dashboard and automations

Home Assistant discovers count/presence sensors, a snapshot camera, a detection
sensor with live object attributes, a historical vehicle event entity, and plate,
alias/type/confidence/timestamp sensors. The native MQTT event entity stores the
latest event timestamp/type and its payload attributes. It is distinct from the
older integration's custom Home Assistant event-bus name.

A ready-to-edit [dashboard example](../examples/home-assistant-dashboard.yml)
contains Live and Stock demo views. Adjust entity IDs for your camera name; use
built-in cards only. The stock view uses recorded results, labelled separately
from live data. See [the demonstration guide](stock-demo.md) for its images.

An MQTT automation can consume events directly without installing another
integration. For example, log recognised watched plates from the live camera:

```yaml
alias: Log watched vehicle
triggers:
  - trigger: mqtt
    topic: mqtt_camera/pi_camera/events
conditions:
  - condition: template
    value_template: >-
      {{ trigger.payload_json.event_type == 'watched_plate_detected'
         and not trigger.payload_json.get('demo', false) }}
actions:
  - action: logbook.log
    data:
      name: Watched vehicle
      message: >-
        {{ trigger.payload_json.plate_alias }}
        ({{ trigger.payload_json.plate }}) recognised
```

If existing automations require a Home Assistant event-bus event, add an HA MQTT
triggered automation that fires your chosen `mqtt_camera.*` event and copies the
payload fields. The Pi itself needs only broker credentials, not HA API access.
