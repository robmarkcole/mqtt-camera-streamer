# Images, RTSP and recording

## Default: MQTT snapshots

Keep the original one-broker setup. Predictions and availability remain small
JSON/text messages; a separate `mqtt_camera/<camera_id>/snapshot` topic carries
raw JPEG bytes. Home Assistant automatically discovers a Snapshot camera on the
same device as the presence/count/plate sensors. No Streamlit or HTTP server is
needed. This uses Home Assistant's [MQTT camera](https://www.home-assistant.io/integrations/camera.mqtt/).

```yaml
images:
  transport: mqtt       # mqtt (default), rtsp, both, or off
  interval_seconds: 1   # maximum snapshot cadence; also limited by inference FPS
  max_width: 960
  jpeg_quality: 80
```

Only the latest JPEG is retained, with QoS 0 and no disconnected frame queue.
A newly confirmed plate also requests a snapshot. The camera shares the inference
availability watchdog: an old retained picture is historical when unavailable.
Switching to `rtsp` or `off` removes this application's MQTT camera discovery.
To delete a historical retained JPEG too, publish an empty retained message to
its snapshot topic using your broker credentials.

Sparse images are useful for simple dashboards and occasional evidence.
Continuous image messages consume broker bandwidth, retained storage and client
processing without video compression. Keep image cadence/size bounded; use RTSP
for a video client. Removing MQTT images altogether would lose the simple setup
that this project is intended to preserve.

## Optional RTSP using existing PyAV libraries

RTSP publishing reuses **PyAV**, already required by Supervision. No separate
FFmpeg executable, OpenCV VideoWriter, GStreamer, desktop or additional Python
streaming framework is needed. PyAV's wheels include FFmpeg libraries; this is
not a codec-free implementation. An **RTSP server** is still required: this
project publishes to a server, rather than implementing the server protocol.
See [PyAV's encoding example](https://pyav.org/docs/stable/cookbook/numpy.html)
and [MediaMTX](https://mediamtx.org/docs/kickoff/install).

```yaml
images:
  transport: rtsp
  max_width: 960
  rtsp_fps: 2
  rtsp_url_env: RTSP_PUBLISH_URL
```

Set `RTSP_PUBLISH_URL` in the process environment (the systemd `mqtt.env` file on
the Pi), for example `rtsp://publisher:YOUR_PASSWORD@127.0.0.1:8554/camera`.
URL-encode special characters in URL credentials. MQTT still carries predictions,
presence and plate events; `rtsp` mode sends **no MQTT images**. Use
`images.transport: both` to keep the discovered MQTT Snapshot camera while also
publishing the same annotated frames over RTSP. Both outputs share the existing
capture/inference pipeline; this does not open the webcam a second time or add
Python dependencies. MQTT snapshot cadence and RTSP output FPS remain separate.
Configure Home
Assistant's [Generic Camera](https://www.home-assistant.io/integrations/generic/)
with the RTSP read URL if desired. An RTSP camera is not created by MQTT discovery.

The worker encodes H.264 in software, uses TCP transport and keeps only one latest
frame. It repeats that frame between inference results, disconnects after
`stale_seconds` without new results, and retries failed publishing after ten
seconds. Picture changes are limited by inference FPS; this is a low-frame-rate
preview, not native 30 FPS capture or guaranteed hardware encoding. Benchmark
CPU and temperature with RTSP enabled on the actual Pi. The inference benchmark
excludes media output costs.

### Optional MediaMTX on the Pi

Install the Linux ARM64 standalone MediaMTX binary from a reviewed official
release into `/usr/local/bin/mediamtx`; match the configuration to that release.
The supplied example was checked against **1.21.4**. Docker is not required.
Follow the upstream [installation instructions](https://mediamtx.org/docs/kickoff/install)
and check the release's published integrity information before installation.
Then:

```sh
sudo install -o root -g mqtt-camera -m 0640 /opt/mqtt-camera-streamer/deploy/mediamtx.yml /etc/mqtt-camera/mediamtx.yml
sudoedit /etc/mqtt-camera/mediamtx.yml
sudo install -m 0644 /opt/mqtt-camera-streamer/deploy/mediamtx.service /etc/systemd/system/mediamtx.service
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
```

Replace **both** example passwords. The publisher account is restricted to
loopback and the `camera` path; the viewer can read that path. Set the matching
publish URL in `/etc/mqtt-camera/mqtt.env`, select `images.transport: rtsp`, then
restart `mqtt-camera`. The example binds to loopback for SSH preview; other
protocols are disabled. To allow LAN readers such as Home Assistant, change
`rtspAddress` to `:8554`; their read URL is then
`rtsp://viewer:YOUR_VIEW_PASSWORD@YOUR_PI:8554/camera`. RTSP authentication is separate
from MQTT authentication. Plain RTSP is intended for a trusted LAN or SSH tunnel;
use an appropriate encrypted deployment for other networks.

For the home-directory/user-service deployment, install the same ARM64 binary at
`~/.local/bin/mediamtx`, save its private configuration at
`~/.config/mqtt-camera/mediamtx.yml`, and install the supplied user unit:

```sh
install -m 0644 deploy/mediamtx-user.service ~/.config/systemd/user/mediamtx.service
systemctl --user daemon-reload
systemctl --user enable --now mediamtx.service
```

Keep the publisher restricted to loopback, use a separate authenticated reader
for Home Assistant, and store URL credentials in private environment/configuration
files. The same lingering setup described in the Pi guide applies to this unit.
Choose `both` in the camera configuration to retain MQTT snapshots alongside RTSP.

### Preview through SSH

On your **laptop**, keep this forwarding session open:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 18554:127.0.0.1:8554 YOUR_ADMIN@YOUR_PI
```

Open `rtsp://viewer@127.0.0.1:18554/camera` in VLC on the laptop and enter the
viewer password when prompted. Enable **Use RTP over RTSP (TCP)** in VLC's
RTSP input preferences, or start VLC with `--rtsp-tcp`. On macOS, for example:

```sh
/Applications/VLC.app/Contents/MacOS/VLC --rtsp-tcp 'rtsp://viewer@127.0.0.1:18554/camera'
```

TCP carries both control and video through this one SSH port; UDP video ports
would not pass through that forwarding command. No X11 forwarding or player on
the Pi is needed. VLC's [RTSP input implementation](https://github.com/videolan/vlc/blob/master/modules/access/live555.cpp)
supports this transport option. The default MQTT mode can instead be previewed
using Home Assistant's Snapshot camera, without installing MediaMTX.

## Optional file-backed recording, for any image transport

```yaml
storage:
  enabled: true
  directory: /var/lib/mqtt-camera/captures
  interval_seconds: 5
  max_files: 10000
  max_age_days: 7
```

Recording is opt-in to avoid silently filling an SD card. **When enabled, paths
are the default storage format for both MQTT and RTSP.** A new plate event also
saves a snapshot. JPEGs are written locally, independently of transport; the
service does not download frames back from RTSP. Images use the configured
snapshot width/quality, while OCR receives the original source resolution.

`records.db` contains a `frames` table: `id`, `camera_id`, `captured_at` (prediction
completion time, not a hardware exposure timestamp), `created_unix`, `image_path`,
`predictions_json`, and `plates_json`. Paths are relative to the capture directory,
so move/backup the directory and database together. Image writes use temporary
files plus atomic rename, and an insert failure removes the new file. Expired
records/files are pruned on startup and after saves, bounded by age and count.
Retention does not migrate or delete older `mqtt_camera` BLOB-table rows or
unindexed files; back those up and manage them explicitly. A hard crash between
file rename and DB commit can leave an unindexed image.

The legacy `scripts/db-recorder.py` also writes this file/path format. Existing
BLOB tables are preserved. Datasette can inspect metadata but filesystem paths
are not image URLs; a remote browser needs separate access to the image files.
There is no image-serving web dependency on the Pi.

Snapshots and RTSP frames use Supervision annotations by default. Set `images.annotate: false` for plain output; inference/OCR always use the original RGB frame. See the [public stock demonstration](stock-demo.md).
