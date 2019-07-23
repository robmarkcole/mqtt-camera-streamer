"""
Capture frames from a camera using openCV and publish on an MQTT topic.
"""
import time

from mqtt import get_mqtt_client
from helpers import pil_image_to_byte_array, get_now_string
from imutils.video import WebcamVideoStream
from imutils import opencv2matplotlib

from PIL import Image

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC_CAMERA = "homie/mac_webcam/capture"
MQTT_QOS = 1

VIDEO_SOURCE = 0 # "rtsp://admin:password@192.168.1.94:554/11" # Int or string path
FPS = 2 # Limit to prevent CPU overheating!


def main():
    client = get_mqtt_client()
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_start()

    # Open camera
    camera = WebcamVideoStream(src=VIDEO_SOURCE).start()
    time.sleep(2)  # Webcam light should come on if using one

    while True:
        frame = camera.read()
        np_array_RGB = opencv2matplotlib(frame)  # Convert to RGB

        image = Image.fromarray(np_array_RGB)  #  PIL image
        byte_array = pil_image_to_byte_array(image)
        client.publish(MQTT_TOPIC_CAMERA, byte_array, qos=MQTT_QOS)
        now = get_now_string()
        print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")
        time.sleep(1/FPS)


if __name__ == "__main__":
    main()
