# Model and runtime experiment record

Recorded on 2026-11-22. The experiments below ran on 2026-11-21; they are retained
as evidence, not additional supported presets. Current setup is [Speed and
Accuracy](../models.md). Historical recommendations in individual reports describe
the decision at that time.

## Current decision

- **Speed: YOLOX Nano FP32.** Fastest measured detector/tracker on the Pi.
- **Accuracy: RF-DETR Nano static INT8.** Highest measured quality of the candidates
  in the latest comparison, with lower model size and better speed than RF-DETR FP32.
- Tiny INT8 was a useful middle ground, but was not selected for the two-preset
  product surface. Tiny needs the same YOLOX adapter; no third runtime is required.
- YOLOX Nano INT8 was smaller but slower and substantially less accurate in the
  tested recipes. It is not selected for Speed.

## Latest matched comparison

FPS measured on a **Raspberry Pi 4 Model B (4 GB)**; mAP evaluated separately on
the development computer.

| Candidate | Detection + tracking FPS | Every-frame OCR FPS | mAP50 (%) | mAP50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| YOLOX Nano FP32 — Speed | 4.33 | 1.75 | 64.92 | 46.91 |
| YOLOX Nano INT8, percentile | 3.87 | 1.67 | 39.09 | 27.38 |
| YOLOX Tiny INT8, MinMax | 2.51 | 1.35 | 76.29 | 55.73 |
| YOLOX Tiny FP32 | 1.87 | 1.11 | 75.96 | 56.37 |
| RF-DETR Nano static INT8 — Accuracy | 0.65 | 0.57 | 88.55 | 72.21 |

Pi 4 Model B Rev 1.2, 4 GB, 64-bit Debian 13.6, four ORT CPU threads; desktop and
RetroArch left running. Runs were serial, in fresh processes, forward/reverse
order across two repeats. Detection used 20 measured frames plus three warmups;
every-frame ALPR used 16 plus three. Identical recordings were used within each
workload. Rate caps, MQTT and media output costs were excluded. These are short
shared-load checks, not sustained headless guarantees.

Accuracy was measured on the development computer using CPU ONNX Runtime and
Supervision, at a 0.001 score floor on 32 COCO128 train2017 images. The 96-image
calibration set was content-disjoint. These images are not held out from original
model training; calibration variants were also compared on this diagnostic.
This is not full COCO validation or an independent deployment acceptance set.
Tiny INT8's slightly higher mAP50 does not establish a general quality improvement;
its mAP50–95 was slightly lower than FP32.

## Records and reproduction

| Experiment | Detailed notes | Raw report |
| --- | --- | --- |
| Initial USB, tracking, plate, MQTT, RTSP and storage checks | [Pi smoke tests](pi-smoke.md) | Notes include the tested code revision |
| RF-DETR FP32, static INT8 and rejected dynamic INT8 | [RF-DETR quantization](rfdetr-int8.md) | [RF-DETR report](results/2026-11-21-rfdetr-int8.json) |
| YOLOX Nano/Tiny export parity, native Pi and accuracy checks | [YOLOX FP32](yolox-fp32.md) | [YOLOX report](results/2026-11-21-yolox-fp32.json) |
| YOLOX INT8 calibration, speed, quality and end-to-end checks | [YOLOX INT8](yolox-int8.md) | [YOLOX INT8 report](results/2026-11-21-yolox-int8.json) |
| Merged default branch on a 1 GB Pi 3, power-limited preset comparison and live MQTT checks | [Pi 3 setup and benchmarks](pi3.md) | [Pi 3 report](results/2026-11-22-pi3.json) |

Raw reports retain model/checkpoint/calibration hashes, software versions,
individual runs, temperature/throttling, memory, diagnostic scores, export parity
and MQTT/plate checks where applicable. Absolute checkout/home paths are redacted.
`5AU5341` in plate tests is the public FastALPR sample, not a live vehicle observation.
No model weights, private credentials or captured webcam images are committed.

## Preset implementation validation

On 2026-11-22 both presets passed native Pi USB checks and nine-frame MQTT/Home
Assistant checks, with confirmed tracks, JPEG snapshots, one plate event and a
fresh plate timestamp. Selected model hashes matched the comparison above. The
full optional calibration suite passed 117 tests; the default suite passed 113
with four optional tests skipped. The Pi retained 42 packages without training or
export dependencies. See the [preset validation report](results/2026-11-22-presets.json).
Camera format was restored; no boot service was enabled.

## Tooling boundaries

The normal service uses two adapters and one shared pipeline. Export, calibration,
labelled evaluation and repeated benchmark tools run separately from that service.
The YOLOX quantizer is archived at `scripts/experiments/quantize_yolox.py`; its
calibration and executable-graph tests remain in the suite. It still uses the
optional `quantize` uv extra on a development computer, not on the Pi.

Do not remove metadata validation, original RGB preservation, upstream license
notices or model-specific preprocessing to reduce line count. Both presets need
those contracts. Fine-tuning and model updates require new labelled quality checks
and representative Pi measurements; the historical numbers are not promises for
new artifacts.

## Speed deployment and reboot validation

On 2026-11-22 Speed was deployed as a boot-persistent systemd user service on
`mypi`, with lingering, tracking, plate recognition and authenticated LAN MQTT.
An isolated nine-frame public-sample test confirmed four vehicle tracks and one
plate event through the actual service; its test Home Assistant device was removed.

After an administrator-authorised reboot, the boot ID changed and the service
started automatically about 20 seconds after boot. It remained enabled/running
with zero restarts. Fresh Speed predictions and 640×480 JPEGs reached Home
Assistant, and the current-boot journal confirmed both plate models loaded. The
runtime stayed at 42 packages without Torch/training/export dependencies. See
[the deployment report](results/2026-11-22-systemd.json) and
[user-service instructions](../raspberry-pi.md#alternative-boot-persistent-user-service).

The current broker and Home Assistant instance run on the Mac, which must remain
available with Docker running. The Pi connects directly over LAN, without an SSH
tunnel. The 20-second metric is service activation, not a sustained throughput or
full boot-to-OCR latency guarantee.

## Local vehicle events and stock-image demonstration

On 2026-11-22 the Pi emitted all four vehicle/plate/watch event types through
native Home Assistant MQTT discovery, with no new runtime dependency. Six
repeated public stock frames produced four confirmed vehicle tracks and one
5AU5341 plate/watch match in the FastALPR sample, and three vehicle tracks in the
Rome image. Native HA history recorded 4 detected / 4 lost / 1 plate / 1 watch
events for the first device and 3 detected / 3 lost for the second. Event IDs were
unique; stock events used separate camera IDs and `demo: true`.

The temporary inference demo finished; the live Speed systemd service remained
enabled/running, with its watchlist unchanged and 42 Torch-free runtime packages.
The two screenshots and public image licenses are included in the updated README.
The foreground car's `truck` label is an actual Nano classification error, not a
manually corrected result. See [demo reproduction](../stock-demo.md),
[event semantics](../vehicle-events.md) and
[the validation report](results/2026-11-22-vehicle-events.json).
