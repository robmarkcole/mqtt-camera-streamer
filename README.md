# mqtt-camera-streamer
**TLDR:** Publish frames from a connected camera to an MQTT topic, and view the feed in a browser with [Streamlit](https://github.com/streamlit/streamlit). Configuration is via `config.yml`. USB webcam & MJPEG/RTSP stream support is via openCV, whilst the raspberry pi camera is supported by [picamera](https://picamera.readthedocs.io/en/release-1.13/).

**Long introduction:** A typical task in IOT/science is that you have a camera connected to one computer and you want to view the camera feed on a different computer, and maybe preprocess the images before saving them to disk. I have always found this to be may more work than expected. In particular working with camera streams can get quite complicated, and may lead you to experiment with tools like Gstreamer and ffmpeg that have a steep learning curve. In contrast, working with [MQTT](http://mqtt.org/) is very straightforward and is also probably familiar to anyone with an interest in IOT. Whilst MQTT is rarely used for sending files, I have not encountered any issues doing this.

`mqtt-camera-streamer` uses MQTT to send frames from a camera connected to a computer over a network at low frames-per-second (FPS). A viewer is provided for viewing the camera stream on any computer on the network. Frames can be saved to disk for further processing. Also it is possible to setup an image processing pipeline by linking MQTT topics together, using an `on_message(topic)` to do some processing and send the processed image downstream on another topic.

**Note** that this is not a high FPS solution, and in practice I achieve around 1 FPS which is practical for tasks such as preprocessing (cropping, rotating) images prior to viewing them. This code is written for simplicity and ease of use, not high performance.

## OpenCV or Picamera
On Mac/linux/windows OpenCV is used to read the images from a connected camera or MJPEG/RTSP stream. On a raspberry pi installing openCV can be troublesome. For RPi users you are recommended to use an official rpi camera (the ones with the ribbon) and use the dedicated RPi camera script. This means you aren't required to install openCV on the RPi.

## Installation
Use a venv to isolate your environment, and install the required dependencies:
```
$ (base) python3 -m venv venv
$ (base) source venv/bin/activate
$ (venv) pip3 install -r requirements.txt
```

## Listing cameras
The `check-cameras.py` script assists in discovering which cameras are on your computer (does not work with picamera). If your laptop has a built-in webcam this will generally be listed as `VIDEO_SOURCE = 0`. If you plug in an external USB webcam this takes precedence over the built-in webcam, with the external camera becoming `VIDEO_SOURCE = 0` and the built-in webcam becoming `VIDEO_SOURCE = 1`.

To check which cameras are detected run:
```
$ (venv) python3 scripts/check-cameras.py
```
You then configure the desired camera as e.g. `video_source: 0`. Alternatively you can configure the video source as an MJPEG or RTSP stream. For example in `config.yml` you would configure `video_source: "rtsp://admin:password@192.168.1.94:554/11"`

## Camera usage
Use the `config.yml` file in `config` directory to configure your system (mqtt broker IP etc) and validate the config can be loaded by running:
```
$ (venv) python3 scripts/validate-config.py
```
**Note** that this script does not check the accuracy of any of the values in `config.yml`, just that the file path is correct and the file structure is OK.

By default `scripts/camera.py` will look for the config file at `./config/config.yml` but an alternative path can be specified using the environment variable `MQTT_CAMERA_CONFIG`

To publish camera frames over MQTT:
```
$ (venv) python3 scripts/camera.py
```

## Camera display
To view the camera stream with Streamlit:
```
$ (venv) streamlit run scripts/viewer.py
```

<p align="center">
<img src="https://github.com/robmarkcole/mqtt-camera-streamer/blob/master/docs/images/viewer_usage.png" width="500">
</p>

**Note:** if Streamlit becomes unresponsive, `ctrl-z` to pause Streamlit then `kill -9 %%`. Also note that the viewer can be run on any machine on your network.

## Save frames
To save frames to disk:
```
$ (venv) python3 scripts/save-captures.py
```

## Image processing pipeline
To process a camera stream (the example rotates the image):
```
$ (venv) python3 scripts/processing.py
```

## Home Assistant
You can view the camera feed using [Home Assistant](https://www.home-assistant.io/) and configuring an [MQTT camera](https://www.home-assistant.io/components/camera.mqtt/). Add to your `configuration.yaml`:
```yaml
camera:
  - platform: mqtt
    topic: homie/mac_webcam/capture
    name: mqtt_camera
  - platform: mqtt
    topic: homie/mac_webcam/capture/rotated
    name: mqtt_camera_rotated
```

<p align="center">
<img src="https://github.com/robmarkcole/mqtt-camera-streamer/blob/master/docs/images/ha_usage.png" width="500">
</p>

## MQTT
Need an MQTT broker? If you have Docker installed I recommend [eclipse-mosquitto](https://hub.docker.com/_/eclipse-mosquitto). A basic broker can be run with:
```
docker run -p 1883:1883 -d eclipse-mosquitto
```
Note that I have structured the MQTT topics following the homie MQTT convention, linked in the references. This is not necessary but is best practice IMO.

### References
* [imageZMQ](https://github.com/jeffbass/imagezmq) -> inspired this project, but uses ZMQ
* [homie MQTT convention](https://homieiot.github.io/) -> convention for structuring MQTT topics
* [yolocam_mqtt](https://github.com/LarsAC/yolocam_mqtt/blob/master/yolo_mqtt_server.py) -> another source of ideas
* [In-depth review and comparison of the Raspberry Pi High Quality Camera](https://medium.com/@alexellisuk/in-depth-review-and-comparison-of-the-raspberry-pi-high-quality-camera-806490c4aeb7)