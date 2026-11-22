# Pi 3 simultaneous MQTT snapshots and RTSP

On 2026-11-22 the Pi 3 production deployment added an authenticated RTSP preview
while keeping the existing MQTT snapshot camera, counts, tracked presence and
vehicle events. Plate recognition was disabled at the user's request. Development
Home Assistant/Mosquitto containers remain stopped with restart policy `no`;
production no longer depends on the development computer.

## Implementation and deployment

`images.transport: both` reuses the existing capture, inference, annotation and
PyAV encoder. It retains MQTT camera discovery and rate-limited JPEG publication
while submitting the same display frames to the bounded RTSP worker. The new
mode adds no Python dependency; the Pi retains 42 runtime packages without Torch.
MQTT-only, RTSP-only and off remain available.

The standalone ARM64 MediaMTX 1.21.4 archive was verified against its published
SHA-256 checksum before installation. The binary runs under an enabled user
systemd service, using a private configuration. Publishing is restricted to
loopback with its own account; Home Assistant uses a separate authenticated reader
on the camera path. Anonymous reads were rejected. Other streaming protocols and
the management API are not enabled. No internet/router port forwarding was added.

The camera user service starts after MediaMTX and wants that unit without making
MQTT depend on the RTSP server remaining up. Publishing reconnects use the
existing worker retry logic. Both units were active/enabled with zero restarts;
lingering remains enabled, but no reboot was performed during this follow-up.

## Verification

- PyAV decoded eight H.264 frames at 640×480 with increasing timestamps spanning
  3.5 seconds, consistent with the configured two-FPS encoded preview.
- Home Assistant's built-in Generic Camera accepted the stream, and its HLS
  master/media playlists and a video segment were retrieved successfully.
- The production dashboard played a 640×480 video element (`paused: false`,
  `readyState: 4`). Its MQTT snapshot fallback remains available below the RTSP
  card; all thirteen existing household cards were preserved.
- Plate recognition remains disabled, and the runtime package count stayed 42.
- All 124 tests passed, including a regression for simultaneous outputs and
  retained MQTT camera discovery; targeted Ruff and systemd unit verification passed.

The encoder repeats the latest inference image at two FPS; new picture content
is limited by the one-FPS detection cap. This is an annotated preview rather than
native webcam-rate video. The earlier [Pi 3/Pi 4 benchmark](pi3.md) excludes media
output, so those scores do not measure the overhead of simultaneous RTSP and
MQTT. Process/temperature/power samples in the raw report are observations, not a
sustained performance benchmark or evidence that the earlier power issue is fixed.
No swap was used in the final sample.

Credentials, production URLs, rollback dashboard backups and live screenshots
remain private. See [RTSP and SSH setup](../images-and-rtsp.md) and the
[validation record](results/2026-11-22-pi3-rtsp.json).
