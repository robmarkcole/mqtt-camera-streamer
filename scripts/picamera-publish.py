"""
Capture frames from an RPi camera using picamera and publish on an MQTT topic.
https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-to-a-pil-image
"""
import os
import time
from io import BytesIO

from helpers import get_config, get_now_string, pil_image_to_byte_array
from mqtt import get_mqtt_client
from picamera import PiCamera
from PIL import Image

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC_CAMERA = CONFIG["camera"]["mqtt_topic"]
# VIDEO_SOURCE = CONFIG["camera"]["video_source"]
FPS = CONFIG["camera"]["fps"]


def main():
    client = get_mqtt_client()
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_start()

    # Open camera
    stream = BytesIO()
    camera = PiCamera()
    camera.resolution = (1024, 768)
    camera.start_preview()
    time.sleep(2)  # Webcam light should come on if using one

    while True:
        camera.capture(stream, format="jpeg")
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image = Image.open(stream)
        byte_array = pil_image_to_byte_array(image)
        client.publish(MQTT_TOPIC_CAMERA, byte_array, qos=MQTT_QOS)
        now = get_now_string()
        print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")
        time.sleep(1 / FPS)
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()


if __name__ == "__main__":
    main()
