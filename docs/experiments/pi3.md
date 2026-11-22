# Raspberry Pi 3 setup and benchmarks

Tested on 2026-11-22 with a Raspberry Pi 3 Model B Rev 1.2 (1 GB),
64-bit Debian 13.7, and an Alcor USB webcam. The installed application is the
merged default branch, `master`, commit
`4629998dcdf4f2acc889f175a148ed9cc7912216` (PR #46). It includes Speed/Accuracy,
Supervision, Roboflow tracking, presence holds, plate recognition, MQTT discovery
and events, annotated snapshots, and the MIT project license.

## Installation

The target already had the required system libraries and camera/video-group
access. Git was absent and sudo required a password, so the reviewed default
branch was transferred using `git archive` and extracted under the user's
`~/mqtt-camera-streamer`. `DEPLOYED_REVISION` records the source commit; this
installation is a source snapshot, not a Git checkout. Install Git before using
`git pull`, or transfer another reviewed snapshot to update it.

uv was installed under `~/.local/bin`, and `uv sync --locked --no-dev` installed
42 runtime packages with managed Python 3.12.16. No Torch, TorchVision, RF-DETR
Python package, ONNX export/calibration package or development tools were installed.
Both preset ONNX artifacts/manifests and the pretrained FastALPR models were
prepared before offline inference. This adds no application dependencies.

The user systemd service uses the existing unit template and private configuration
under `~/.config/mqtt-camera`. The camera uses its stable `/dev/v4l/by-id` path,
with 640×480 YUYV capture. MQTT connects to the existing development broker on the
Mac over LAN; the Mac and Docker must stay available. No SSH broker tunnel or Home
Assistant API token is needed by the Pi application.

## Protocol and power limitation

The matched preset comparison uses four ONNX Runtime detector threads and two
plate-model threads. Each model runs in a fresh process, twice, in forward then
reverse order. Detection/tracking uses 20 measured frames after three warmups;
every-frame plate recognition uses 16 after three. Recording and model SHA-256
hashes match the [Pi 4 comparison](yolox-int8.md#physical-pi-benchmarks).
Application rate limits, MQTT, annotations, JPEG encoding, storage and RTSP output
are excluded. This is a pipeline benchmark, not model-only FPS or sustained live
camera throughput. Accuracy/mAP is not re-evaluated on the Pi 3.

**These measurements are power-limited.** The Pi initially reported historical
undervoltage/throttling (`0x50000`) and reported active undervoltage/throttling
(`0x50005`) during inference. The user chose to continue with the current supply.
Temperature and flags are retained per run. The Pi 4 reference reported `0x0`.
These results describe this setup, not the full capability of a correctly powered
Pi 3. Memory, power conditions, OS and background workload differ between boards.

## Matched preset comparison

| Preset | Pi 3 detection + tracking FPS | Pi 4 reference FPS | Pi 3 every-frame OCR FPS | Pi 4 reference OCR FPS | Pi 3 detection peak RSS | Pi 4 detection peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Speed — YOLOX Nano FP32 | 1.49 | 4.33 | 0.56 | 1.75 | 269.2 MiB | 267.4 MiB |
| Accuracy — RF-DETR Nano static INT8 | 0.25 | 0.65 | 0.23 | 0.57 | 432.7 MiB | 433.5 MiB |

These are medians of two runs, rounded to two decimals. Peak RSS is the maximum
of the detection runs; the smaller plate recording has a different memory
footprint and should not be used to infer OCR's incremental RAM cost. The Pi 4
numbers are historical results from earlier code and a different background
workload, not a controlled comparison isolating the CPU. No swap was used in the
observed Pi 3 checks. Per-run endpoint temperatures stayed below 59°C during the
four-thread comparison, while active undervoltage/throttling was observed.

## Thread probe and live settings

Two additional Speed runs on the same people recording used two detector threads,
20 measured frames and three warmups each. Median detection/tracking throughput
was **1.89 FPS**, versus **1.49 FPS** at four threads; peak RSS was 268.7 MiB.
The two-thread endpoint samples reported only historical flags, but active
undervoltage was seen again in live use. This probe does not show that a thread
change fixes the power supply, and its different run order/temperature is another
reason not to generalise the gain.

The live service therefore uses **Speed, two detector threads, two plate-model
threads, a 1 FPS cap, plate scans every five seconds and a ten-second presence
hold**. Every-frame OCR numbers above remain the matched four-thread comparison;
they are not a measurement of this duty-cycled configuration. The live watchlist
is empty. Annotated 640px-wide MQTT snapshots are enabled; RTSP and storage are off.

## End-to-end validation

An isolated public-sample run published ten predictions, four confirmed vehicle
tracks, one `5AU5341` plate recognition and one watched-plate match. Native Home
Assistant plate state was checked. The sample used a separate camera ID with
`demo: true`; its discovery entries were removed afterward, and its watchlist was
never applied to the live camera. This repeats a stock still, not real movement.

The actual webcam then produced at least eight fresh live predictions and 640×480
JPEGs. Home Assistant's camera proxy returned a valid JPEG, and the native camera,
detection-details and person/vehicle presence entities were available. The service
was active/enabled with zero restarts and `Linger=yes`; no reboot was performed,
so startup after a reboot has not yet been independently tested on this board.
The previous Pi 4 service remains stopped and disabled.

The live camera entity is `camera.window_pi_camera_snapshot`. Select it from the
Home Assistant camera/device UI or add it to a dashboard. Check the service over
SSH with `systemctl --user status mqtt-camera.service`, replacing `status` with
`stop` or `start` to manage it. Use
`systemctl --user disable --now mqtt-camera.service` to stop it and disable boot
startup. Configuration and broker credentials remain private on the Pi.

## Home Assistant dashboard follow-up

The `mqtt-camera` dashboard's **Live** view now uses `window_pi_camera` entities
throughout, replacing the stopped Pi 4 camera. It shows the annotated MQTT image,
current object labels/confidence/track IDs, counts, held presence, vehicle events,
plate number/confidence/time/watchlist alias and vehicle-type prediction. The
exported [dashboard example](../../examples/home-assistant-dashboard.yml) matches
the saved configuration. Its Stock demo view remains explicitly separate.

All twelve referenced native MQTT entities were present, the latest prediction
was fresh and the HA camera proxy returned a 640×480 JPEG. The UI was checked
after reloading its cached dashboard configuration. With no confirmed live plate
yet, the plate status explains that OCR is enabled and awaiting repeated matching
readings. The old Pi 4's historical public-sample plate is no longer displayed in
the Live view. Live camera screenshots remain local rather than being committed;
the README's public stock-image screenshots remain reproducible examples.

See the [dashboard validation record](results/2026-11-22-pi3-dashboard.json).

See the [raw validation/benchmark report](results/2026-11-22-pi3.json) for all runs,
model/recording hashes, package versions, power flags and test scope. The setup
and tests required no application code change or new runtime dependencies.
