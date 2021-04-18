# mqtt-camera-streamer
**Summary:** Publish frames from a connected camera or MJPEG/RTSP stream to an MQTT topic, and view the feed in a browser on another computer with [Streamlit](https://github.com/streamlit/streamlit).

**Long introduction:** A typical task in IOT/science is that you have a camera connected to one computer and you want to view the camera feed on a second computer, and maybe preprocess the images before saving them to disk. I have always found this to be way more effort than expected. In particular, working with camera streams can get quite complicated and may lead you to experiment with tools like Gstreamer and ffmpeg that have a steep learning curve. In contrast, working with [MQTT](http://mqtt.org/) is very straightforward and is often familiar to anyone with an interest in IOT. This repo, `mqtt-camera-streamer` uses MQTT to send frames from a camera over a network at low frames-per-second (FPS). A viewer is provided for viewing the camera stream on any computer on the network. Frames can be saved to disk for further processing. Also it is possible to setup an image processing pipeline by linking MQTT topics together, using an `on_message(topic)` to do some processing and send the processed image downstream on another topic.

**Note** that this is not a high FPS solution, and in practice I achieve around 1 FPS which is practical for IOT experiments and tasks such as preprocessing (cropping, rotating) images prior to viewing them. This code is written for simplicity and ease of use, not high performance.

## Installation
Install system wide on an RPi, or on other OS use a venv to isolate your environment, and install the required dependencies:
```
$ (base) python3 -m venv venv
$ (base) source venv/bin/activate
$ (venv) pip3 install -r requirements.txt
```

## Listing cameras with OpenCV
The `check-opencv-cameras.py` script assists in discovering which cameras OpenCV can connect to on your computer (does not work with RPi camera). If your laptop has a built-in webcam this will generally be listed as `VIDEO_SOURCE = 0`. If you plug in an external USB webcam this takes precedence over the built-in webcam, with the external camera becoming `VIDEO_SOURCE = 0` and the built-in webcam becoming `VIDEO_SOURCE = 1`.

To check which OpenCV cameras are detected run:
```
$ (venv) python3 scripts/check-opencv-cameras.py
```

## Configuration using `config.yml`
Use the `config.yml` file in the `config` directory to configure your system. If your desired camera is listed as source 0 you will configure `video_source: 0`. Alternatively you can configure the video source as an MJPEG or RTSP stream. For example in `config.yml` you may configure something like `video_source: "rtsp://admin:password@192.168.1.94:554/11"` for a commercial RTSP camera. To configure a RPi camera running the `web_streaming.py` example you configure `video_source: http://pi_ip:8000/stream.mjpg`

Validate the config can be loaded by running:
```
$ (venv) python3 scripts/validate-config.py
```

**Note** that this script does not check the accuracy of any of the values in `config.yml`, just that the file path is correct and the file structure is OK.

By default `scripts/opencv-camera.py` will look for the config file at `./config/config.yml` but an alternative path can be specified using the environment variable `MQTT_CAMERA_CONFIG`. You can set this using `export MQTT_CAMERA_CONFIG=/home/pi/github/mqtt-camera-streamer/config/config.yml`

## Publish camera frames
To publish camera frames with OpenCV over MQTT:
```
$ (venv) python3 scripts/opencv-camera-publish.py
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

## Save frames to db
As `save-captures.py` but in addition saving the frame thumbnail to a sqlite db:
```
$ (venv) python3 scripts/db-recorder.py
```

The images can be viewed using [sqlite browser](https://sqlitebrowser.org/)

<p align="center">
<img src="https://github.com/robmarkcole/mqtt-camera-streamer/blob/master/docs/images/sqlite-browser.jpg" width="800">
</p>

If you wish to run a server with UI for browsing the images then [datasette](https://datasette.io/) with the [datasette-render-images](https://datasette.io/plugins/datasette-render-images) plugin can be used.

```
$ (venv) pip install datasette
$ (venv) pip install datasette-render-images
$ (venv) datasette captures/records.db
```

<p align="center">
<img src="https://github.com/robmarkcole/mqtt-camera-streamer/blob/master/docs/images/datasette.jpg" width="800">
</p>


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
  - platform: mjpeg # the raw mjpeg feed if using picamera
    name: picamera
    mjpeg_url: http://192.168.1.134:8000/stream.mjpg
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

## OpenCV & streamlit on RPi
OpenCV is used to read the images from a connected camera or MJPEG/RTSP stream. On a Raspberry pi (RPi) installing OpenCV can be troublesome, and I found it necessary to first `sudo apt-get install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test libilmbase-dev libopenexr-dev libgstreamer1.0-dev libavcodec58 libavformat58 libswscale5` before installing opencv using the instructions below. Likewise Streamlit can be challenging to install on an RPi, and if you dont need it then remove it from `requirements.txt`. If you do wish to install Streamlit on the RPi see [this thread](https://discuss.streamlit.io/t/raspberry-pi-streamlit/2900) for latest guidance. On 24/3/2021 I was able to install `opencv-python==4.5.1.48` but not streamlit on an RPi4 32bit.

## RPi camera
Use an official RPi camera and ensure [picamera](https://picamera.readthedocs.io/en/release-1.13/) is installed with `pip3 install picamera`. If you use the RPi in desktop mode you can check the camera feed using `raspistill -o image.jpg`. Use the official [web_streaming](https://github.com/waveform80/picamera/blob/master/docs/examples/web_streaming.py) example which creates an mjpeg stream on `http://pi_ip:8000/stream.mjpg`. This mjpeg stream can be configured as a source with `mqtt-camera-streamer` to translate the mjepg stream to an mqtt stream.

## RPi service
You can run any of the scripts as a [service](https://www.raspberrypi.org/documentation/linux/usage/systemd.md), which means they will automatically start on RPi boot, and can be easily started & stopped. Create the service file in the appropriate location on the RPi using:

```sudo nano /etc/systemd/system/my_script.service```

Entering the following (adapted for your `script.py` file location and args, assumes you are using system python3):
```
[Unit]
Description=Service for mqtt-camera-publish
After=network.target

[Service]
ExecStart=/usr/bin/python3 -u opencv-camera-publish.py
WorkingDirectory=/home/pi/github/mqtt-camera-streamer/scripts
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Once this file has been created you can to start the service using:
```sudo systemctl start my_script.service```

View the status and logs with:
```sudo systemctl status my_script.service```

Stop the service with:
```sudo systemctl stop my_script.service```

Restart the service with:
```sudo systemctl restart my_script.service```

You can have the service auto-start on rpi boot by using:
```sudo systemctl enable my_script.service```

You can disable auto-start using:
```sudo systemctl disable my_script.service```

### References
* [imageZMQ](https://github.com/jeffbass/imagezmq) -> inspired this project, but uses ZMQ. [Discussion on ZMQ vs MQTT here](https://github.com/jeffbass/imagezmq/issues/5)
* [homie MQTT convention](https://homieiot.github.io/) -> convention for structuring MQTT topics
* [yolocam_mqtt](https://github.com/LarsAC/yolocam_mqtt/blob/master/yolo_mqtt_server.py) -> another source of ideas
* [In-depth review and comparison of the Raspberry Pi High Quality Camera](https://medium.com/@alexellisuk/in-depth-review-and-comparison-of-the-raspberry-pi-high-quality-camera-806490c4aeb7)