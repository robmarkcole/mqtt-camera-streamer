# Headless Raspberry Pi deployment

Target a **64-bit Raspberry Pi OS Lite** installation with a USB UVC/V4L2
webcam. Start with RF-DETR Nano at a low inference rate, then benchmark on the
actual board. The official RF-DETR package still includes PyTorch/TorchVision;
allow space for the Python environment and model files. No desktop, Streamlit,
Docker or Home Assistant installation is required on the Pi.

The deployment files assume `/opt/mqtt-camera-streamer`, a dedicated
`mqtt-camera` service user, and uv at `/usr/local/bin/uv`. The service's Python
and environment live under `/var/lib/mqtt-camera`, avoiding an administrator's
home directory and making the same environment usable during boot and over SSH.

## 1. Prepare headless access

In Raspberry Pi Imager choose the 64-bit Lite OS. Configure a hostname, your own
administrator username, network/Wi-Fi settings, and SSH public-key access in OS
customisation before writing the card. Boot the Pi, then connect using SSH.
See [Raspberry Pi's setup guide](https://www.raspberrypi.com/documentation/computers/getting-started.html).

On the Pi:

```sh
uname -m # expected: aarch64
sudo apt update
sudo apt install -y git curl ca-certificates v4l-utils libglib2.0-0 libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
/usr/local/bin/uv --version
```

Use adequate power and cooling for sustained inference. Capture temperature and
throttling before/after a long benchmark; a short cold run may overstate capacity.
The image selection and uv installer are described in the linked official docs.

## 2. Install the reviewed application and environment

The implementation currently lives on `feat/onnx-home-assistant`; use a reviewed
release or commit when available. The commands below select that feature branch:

```sh
sudo git clone --branch feat/onnx-home-assistant https://github.com/robmarkcole/mqtt-camera-streamer.git /opt/mqtt-camera-streamer
sudo useradd --system --user-group --create-home --home-dir /var/lib/mqtt-camera --shell /usr/sbin/nologin mqtt-camera
sudo usermod -aG video mqtt-camera
sudo install -d -o mqtt-camera -g mqtt-camera -m 0750 /var/cache/mqtt-camera
sudo -Hu mqtt-camera /opt/mqtt-camera-streamer/deploy/uv.sh sync --locked --no-dev
```

`deploy/uv.sh` consistently sets the project environment, Python installation and
cache paths. It accepts ordinary uv arguments. Build the environment on the Pi;
an environment copied from macOS or another architecture is not portable.
This locked runtime installation excludes development tools and optional viewers.
Headless OpenCV is used, so Qt/X11 and `libGL` are not needed by this profile.
The lockfile selects CPU-only PyTorch/TorchVision wheels on Linux to avoid
installing CUDA libraries on the Pi.

## 3. Copy the exported model

Export RF-DETR on the development computer rather than doing training on the Pi:

```sh
uv run scripts/export-rfdetr-onnx.py
scp models/rfdetr-nano/rfdetr-nano.onnx YOUR_ADMIN@YOUR_PI:/tmp/rfdetr-nano.onnx
```

Then, on the Pi:

```sh
sudo install -d -m 0755 /opt/mqtt-camera-streamer/models/rfdetr-nano
sudo install -m 0644 /tmp/rfdetr-nano.onnx /opt/mqtt-camera-streamer/models/rfdetr-nano/rfdetr-nano.onnx
```

The official export embeds inference metadata. The benchmark records a SHA-256
digest so reports from different machines can be matched to the same artifact.

## 4. Configure the USB camera and MQTT

```sh
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
v4l2-ctl --list-formats-ext --device /dev/video0
sudo install -d -m 0755 /etc/mqtt-camera
sudo install -m 0644 /opt/mqtt-camera-streamer/deploy/pi.yml /etc/mqtt-camera/config.yml
sudo install -o root -g root -m 0600 /opt/mqtt-camera-streamer/deploy/mqtt.env.example /etc/mqtt-camera/mqtt.env
sudoedit /etc/mqtt-camera/config.yml
sudoedit /etc/mqtt-camera/mqtt.env
```

Replace the example source with the webcam's actual `/dev/v4l/by-id/...-video-index0`
path. A stable link is resolved to its current `/dev/videoN` index at startup;
this avoids assuming the USB webcam is always index 0. Some webcams expose extra
metadata/video nodes, so check which node produces images. After a camera moves
to a new index during unplug/replug, a service restart resolves the link again.

Set a unique `camera_id`, model path, and the broker's **LAN IP or DNS name**.
For the Home Assistant Mosquitto app, use port `1883` and a dedicated MQTT user's
username/password in `mqtt.env`. Usernames `homeassistant` and `addons` are
reserved by the app. These credentials are not a Home Assistant API token.
Use port `8883` and `mqtt.tls: true` when the broker has a TLS certificate trusted
by the Pi. The app and configuration are documented
[here](https://github.com/home-assistant/app-mosquitto/blob/main/mosquitto/DOCS.md).

The supplied example starts at 1 inference FPS with a 60-second stale timeout.
`torch_threads` limits Torch/TorchVision preprocessing; the upstream RF-DETR
adapter controls its own ONNX Runtime thread pool. LPR is off in the Pi example
until it has been benchmarked; the local development configuration can differ.

Enable MQTT discovery in Home Assistant. No code runs inside Home Assistant and
the Pi does not need access to Home Assistant's HTTP API. The environment file is
read by systemd; do not commit a filled-in copy to the repository.

## 5. Cache optional plate models

If plate recognition will be enabled, prepare its cache as the same service user
with network access available:

```sh
sudo -Hu mqtt-camera /opt/mqtt-camera-streamer/deploy/uv.sh run --no-sync --no-dev scripts/download-plate-models.py --config /etc/mqtt-camera/config.yml
```

Then enable `lpr.enabled` after measuring its overhead. Global YOLOv9/CCT defaults
are inherited if their model names are omitted. See [plate settings](license-plates.md).
uv's `--offline` controls dependency resolution; it does not disable model-hub
HTTP downloads. Preparing models first makes normal inference work from cache.

## 6. Benchmark before enabling the service

Place a representative recording where `mqtt-camera` can read it, for example
`/var/lib/mqtt-camera/sample.mp4`. Run from the same environment used by systemd:

```sh
sudo -Hu mqtt-camera /opt/mqtt-camera-streamer/deploy/uv.sh run --offline --no-sync --no-dev scripts/benchmark.py --config /etc/mqtt-camera/config.yml --source /var/lib/mqtt-camera/sample.mp4 --warmup 5 --frames 100 --lpr off --output /var/lib/mqtt-camera/bench/detection.json
sudo -Hu mqtt-camera /opt/mqtt-camera-streamer/deploy/uv.sh run --offline --no-sync --no-dev scripts/benchmark.py --config /etc/mqtt-camera/config.yml --source /var/lib/mqtt-camera/sample.mp4 --warmup 5 --frames 100 --lpr every-frame --output /var/lib/mqtt-camera/bench/plates.json
```

These commands do not connect to MQTT. Use a long enough recording for warmup and
measurement. `--lpr every-frame` enables ALPR for the benchmark and measures every
frame; `--lpr configured` uses the configuration's enable flag and scan interval.
Configured mode can yield only a few plate samples during an unthrottled run, so
check sample counts before interpreting its p95. `--source 0` benchmarks a live
camera instead; stop the service first so the camera has a single owner.

The JSON report separates model load time from warm measured frames. It includes
p50/p95 capture+RF-DETR latency, tracking time, actual plate scan time, whole-frame
latency, observed FPS, CPU core equivalents, and peak process RSS. Capture/decode,
pre/postprocessing and source blocking are included; MQTT and application rate
limiting, JPEG encoding, recording and RTSP publishing are excluded. It is not a raw neural-network kernel benchmark. Peak RSS
includes loading/warmup and is a process high-water mark. Temperature and
`vcgencmd get_throttled` are recorded when available, otherwise null.

Repeat runs with the same recording/model hash, then test sustained operation
with the real webcam. Leave headroom below observed capacity for MQTT, decoding,
thermal variation and other services. Raising `max_fps` cannot make slow inference
faster. Fast-moving plates also need sufficient image detail and short exposure;
held presence and tracking do not recover unreadable characters or missed passages.

## 7. Install and start systemd

```sh
sudo install -m 0644 /opt/mqtt-camera-streamer/deploy/mqtt-camera.service /etc/systemd/system/mqtt-camera.service
sudo systemd-analyze verify /etc/systemd/system/mqtt-camera.service
sudo systemctl daemon-reload
sudo systemctl enable --now mqtt-camera
sudo systemctl status mqtt-camera
sudo journalctl -u mqtt-camera -f
```

The unit uses `uv run --offline --no-sync --no-dev` with the prepared environment.
It waits for network-online, restarts after failure or an unexpected end of the
source, grants video-group access, and stores caches/state in writable service
directories. Code under `/opt` is read-only in the service sandbox. `KillMode=mixed`
lets uv forward SIGTERM to Python before systemd's eventual timeout; a normal stop
publishes offline availability and releases the camera. No boot-time dependency
sync is performed. Optional image recordings belong under
`/var/lib/mqtt-camera/captures`, which is writable in the sandbox.

```sh
sudo systemctl stop mqtt-camera
sudo systemctl restart mqtt-camera
sudo journalctl -u mqtt-camera --since "10 minutes ago"
```

If startup fails, check the log, model path, broker reachability/credentials and
the service user's camera permissions. `network-online` does not guarantee the
broker is reachable; the application times out and systemd retries. Ensure the
Pi has no second process holding the webcam. Tune journald retention for the
storage capacity of the device.

## Optional RTSP and SSH preview

MQTT snapshots are enabled by default and appear in Home Assistant without extra
services. To publish RTSP instead, reuse the installed PyAV library and optionally
install the standalone MediaMTX server. There is no FFmpeg executable or GUI
requirement on the Pi. See [images, RTSP, recording and SSH preview](images-and-rtsp.md)
for configuration and a laptop-to-Pi TCP tunnel. RTSP adds software H.264 encoding;
measure sustained CPU/temperature with the full service running.

## Updates

Stop the service, update to a reviewed revision, run the locked sync as the
service user, prepare any changed models, then restart:

```sh
sudo systemctl stop mqtt-camera
# Update /opt/mqtt-camera-streamer to the reviewed revision here.
sudo -Hu mqtt-camera /opt/mqtt-camera-streamer/deploy/uv.sh sync --locked --no-dev
sudo systemctl restart mqtt-camera
```

If the unit file changed, install it again and run `systemctl daemon-reload`.
Keep the prior revision/lockfile and model artifact for rollback. Preparation and
ARM64-container validation are not a claim that the service has been tested on a
physical Raspberry Pi; benchmark and camera/boot testing on the target remain
required. [systemd reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).
