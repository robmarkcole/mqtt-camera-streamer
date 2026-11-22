# Initial Raspberry Pi smoke tests

Historical results; see [current setup](../raspberry-pi.md).

## Measured Raspberry Pi 4 smoke test

Tested on 2026-11-21 with code commit `21d7914`: Raspberry Pi 4 Model B Rev 1.2,
4 GB RAM, 64-bit Debian 13.6, CPU around 1.5 GHz, and the exported RF-DETR Nano
384 model. A desktop and RetroArch were running, so these are short shared-machine
measurements, not dedicated headless throughput limits. Each row measured ten
live webcam frames after two warmup frames, without MQTT/media output overhead.

| Pipeline | ORT threads | Measured FPS | Frame p95 | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| Detection + tracking | 2 | 0.35 | 3.35 s | 377 MiB |
| Detection + tracking | 4 | 0.45 | 2.36 s | 381 MiB |
| Detection + tracking + every-frame plate scans | 4 | 0.38 | 2.89 s | 401 MiB |

Temperature reached about 51°C during these checks; `get_throttled` remained
`0x0`. The installed runtime contained 42 packages, with no Torch, TorchVision,
RF-DETR, Transformers or ONNX Python/export package. OCR scans averaged about
363 ms per frame; general RF-DETR inference was the dominant cost.

The Alcor USB webcam advertises 1280×720 MJPEG, but the current OpenCV capture
path opened it at **640×480 YUYV, 30 FPS**. Check actual capture settings rather
than assuming the advertised format is selected. The initial frame was black;
warmed-up frames were normal. Camera resolution/FPS/codec controls are not yet
exposed in the application configuration.

Hardware acceptance checks also passed: live webcam JPEGs and predictions reached
the local Home Assistant test instance through an authenticated MQTT SSH tunnel;
JPEG files and relative SQLite paths were verified; a public pretrained plate
sample produced one confirmed event and updated HA's plate/timestamp sensors;
and a laptop decoded H.264 from the Pi through a TCP-only RTSP SSH tunnel.
RTSP mode continued MQTT predictions and sent no MQTT images.

At these rates this is an occasional detection/snapshot pipeline, not reliable
fast-moving plate recognition. Benchmark representative footage before relying
on it. Next performance experiments should measure explicit camera capture
settings and a calibrated/accuracy-checked quantized or smaller model.

Tests used a user-owned installation without sudo. No system services were enabled;
boot behavior, long uptime and production-broker integration remain deployment
checks. The systemd units were separately verified in a Linux ARM64 container.

## INT8 follow-up

The [INT8 experiment guide](rfdetr-int8.md) records a subsequent comparison on this Pi.
Static INT8 improved detection/tracking from about 0.45 to 0.66 FPS and the
pipeline with every-frame plate scans from about 0.42 to 0.58 FPS. Its model file
was about 32 MiB versus 103 MiB FP32, with a 1.28-point mAP50–95 loss on the small
calibration-disjoint diagnostic. FP32 remains the default pending evaluation on
representative deployment footage; the guide includes quantization and comparison
commands that keep all development tools off the Pi.

