# mqtt-camera-streamer
Working with camera streams can get quite complicated, and may lead you to experiment with complicated tools like Gstreamer and ffmpeg. In contrast working with MQTT is very straightforward, and is also very familiar to anyone with an interest in IOT. It is possible to setup a processing pipeline just by linking MQTT topics together, using an `on_message(topic)` to do some processing and send the processed data downstream on another topic. The aim of this code is to publish frames from a camera feed to an MQTT topic, and demonstrate how this can be used to create an image processing pipeline that can be used in an IOT project. 

### References
* [homie MQTT convention](https://homieiot.github.io/)
* [yolocam_mqtt](https://github.com/LarsAC/yolocam_mqtt/blob/master/yolo_mqtt_server.py)
* [cusca](https://github.com/dgomes/cusca)