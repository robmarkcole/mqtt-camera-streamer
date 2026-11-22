# Headless Raspberry Pi deployment

Target a **64-bit Raspberry Pi OS Lite** installation with a USB UVC/V4L2
webcam. Choose `preset: speed` (YOLOX Nano FP32) or `preset: accuracy`
(RF-DETR Nano static INT8), then benchmark on the actual board. The runtime uses ONNX Runtime with NumPy preprocessing; PyTorch,
TorchVision, Transformers and the RF-DETR Python package are excluded. No desktop,
Streamlit, Docker or Home Assistant installation is required on the Pi.

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
Do not enable the `export` extra on the Pi; it belongs on the model development
computer. Torch-related entries remain in the lockfile for that optional extra,
but are absent from this runtime installation.

## 3. Copy the selected preset model

Prepare the model on the development computer using [model setup](models.md).
Keep export and quantization dependencies off the Pi. For Speed:

```sh
scp models/yolox-nano/yolox-nano.onnx models/yolox-nano/yolox-nano.json YOUR_ADMIN@YOUR_PI:/tmp/
```

Then, on the Pi:

```sh
sudo install -d -m 0755 /opt/mqtt-camera-streamer/models/yolox-nano
sudo install -m 0644 /tmp/yolox-nano.onnx /tmp/yolox-nano.json /opt/mqtt-camera-streamer/models/yolox-nano/
```

For Accuracy, copy `rfdetr-nano-int8-static.onnx` and its JSON manifest into
`/opt/mqtt-camera-streamer/models/rfdetr-nano/` and set `preset: accuracy`.
Presets resolve models under the installed repository, even when the configuration
is in `/etc/mqtt-camera`. No model path is required in the normal configuration.

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

Set a unique `camera_id`, choose `preset: speed` or `preset: accuracy`, and set
the broker's **LAN IP or DNS name**.
For the Home Assistant Mosquitto app, use port `1883` and a dedicated MQTT user's
username/password in `mqtt.env`. Usernames `homeassistant` and `addons` are
reserved by the app. These credentials are not a Home Assistant API token.
Use port `8883` and `mqtt.tls: true` when the broker has a TLS certificate trusted
by the Pi. The app and configuration are documented
[here](https://github.com/home-assistant/app-mosquitto/blob/main/mosquitto/DOCS.md).

The supplied example starts at 1 inference FPS with a 60-second stale timeout.
`onnx_threads` limits ONNX Runtime CPU inference threads; spinning is
disabled to reduce idle CPU contention. Start by comparing 2 and 4 threads on
a four-core Pi; a lower thread budget can reduce inference FPS. This setting is
different from the previous Torch preprocessing limit. LPR is off in the example
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
p50/p95 capture+detector latency, tracking time, actual plate scan time, whole-frame
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

## Alternative: boot-persistent user service

For a home-directory installation at `~/mqtt-camera-streamer`, use the supplied
user unit instead of the `/opt` system unit. It expects uv at `~/.local/bin/uv`,
uses the same locked environment and runs as the ordinary camera/video-group
user. Only one service should own the USB webcam.

On the Pi, as that user:

```sh
cd ~/mqtt-camera-streamer
~/.local/bin/uv sync --locked --no-dev
install -d -m 0700 ~/.config/mqtt-camera
install -d ~/.config/systemd/user
install -m 0600 deploy/pi.yml ~/.config/mqtt-camera/config.yml
install -m 0600 deploy/mqtt.env.example ~/.config/mqtt-camera/mqtt.env
# Edit config and broker credentials before starting:
nano ~/.config/mqtt-camera/config.yml
nano ~/.config/mqtt-camera/mqtt.env
```

Select `preset: speed`, the webcam's stable device path, `tracking.enabled: true`
and `lpr.enabled: true`. Point `mqtt.broker` directly at a reachable LAN broker;
loopback refers to the Pi itself. For storage, use a writable path such as
`/home/YOUR_USER/.local/state/mqtt-camera/captures`, rather than `/var/lib`.
Keep `max_fps: 1` initially and cache plate weights under the same user:

```sh
~/.local/bin/uv run --locked --no-dev scripts/download-plate-models.py --config ~/.config/mqtt-camera/config.yml
install -m 0644 deploy/mqtt-camera-user.service ~/.config/systemd/user/mqtt-camera.service
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/mqtt-camera.service
systemctl --user enable --now mqtt-camera.service
systemctl --user status mqtt-camera.service
```

Lingering starts the user manager at boot and keeps it running after logout.
`loginctl show-user "$USER" -p Linger` must report `yes`; enabling it may require
administrator authentication on other installations. The unit uses offline/no-sync
uv execution, private environment-file permissions and automatic restart after
errors. The restart policy also handles a network or USB device not ready at boot.
It neither installs dependencies nor requires an SSH tunnel during startup.

Use `systemctl --user restart mqtt-camera.service` after editing the config,
`systemctl --user stop mqtt-camera.service` to release the camera, and
`systemctl --user disable --now mqtt-camera.service` to remove boot startup.
Read current-boot logs with:

```sh
journalctl -b _SYSTEMD_USER_UNIT=mqtt-camera.service _UID="$(id -u)" -f
```

To check boot recovery, reboot with administrator authorization, reconnect and
check `systemctl --user is-enabled mqtt-camera.service`,
`systemctl --user is-active mqtt-camera.service`, the current-boot journal and
fresh Home Assistant predictions/snapshots. A running service before a reboot
alone does not establish reboot recovery.

The current development deployment uses Mosquitto/Home Assistant on the Mac over
LAN. That Mac and Docker must remain running; it is not a Pi-hosted broker.
For standalone production use, configure the permanent Home Assistant/Mosquitto
host and port instead. The Home Assistant Mosquitto app normally uses port 1883.

## Optional RTSP and SSH preview

MQTT snapshots are enabled by default and appear in Home Assistant without extra
services. To publish RTSP instead, reuse the installed PyAV library and optionally
install the standalone MediaMTX server. There is no FFmpeg executable or GUI
requirement on the Pi. See [images, RTSP, recording and SSH preview](images-and-rtsp.md)
for configuration and a laptop-to-Pi TCP tunnel. RTSP adds software H.264 encoding;
measure sustained CPU/temperature with the full service running.

## Measured performance

The [two-preset guide](models.md) compares Speed and Accuracy on the physical Pi.
The full [experiment record](experiments/README.md) retains the original smoke
checks, FP32/INT8 comparisons and unsuccessful candidates. Short checks do not
replace sustained workload, boot and recovery tests on your installation.

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
the short Pi measurements do not replace sustained workload, camera/boot
and recovery testing on the intended installation. [systemd reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).

Both presets use the same service, MQTT entities, tracking and plate pipeline.
No export extra belongs in the Pi runtime.

For the home-directory user service, use `systemctl --user stop mqtt-camera.service`,
update the reviewed checkout, run `~/.local/bin/uv sync --locked --no-dev` there,
then `systemctl --user start mqtt-camera.service`. If its unit template changes,
copy it back to `~/.config/systemd/user/mqtt-camera.service` and run
`systemctl --user daemon-reload` before starting.
