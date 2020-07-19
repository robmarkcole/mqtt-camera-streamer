"""
Capture frames from a camera using OpenCV and publish on an MQTT topic.
"""
import os
import time

from helpers import get_config, get_now_string, pil_image_to_byte_array
from imutils import opencv2matplotlib
from imutils.video import VideoStream
from mqtt import get_mqtt_client
from PIL import Image

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC_CAMERA = CONFIG["camera"]["mqtt_topic"]
VIDEO_SOURCE = CONFIG["camera"]["video_source"]
FPS = CONFIG["camera"]["fps"]


def main():
    client = get_mqtt_client()
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_start()

    # Open camera
    camera = VideoStream(src=VIDEO_SOURCE, framerate=FPS).start()
    time.sleep(2)  # Webcam light should come on if using one

    while True:
        frame = camera.read()
        np_array_RGB = opencv2matplotlib(frame)  # Convert to RGB

        image = Image.fromarray(np_array_RGB)  #  PIL image
        byte_array = pil_image_to_byte_array(image)
        client.publish(MQTT_TOPIC_CAMERA, byte_array, qos=MQTT_QOS)
        now = get_now_string()
        print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")
        time.sleep(1 / FPS)


if __name__ == "__main__":
    main()
