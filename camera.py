"""
Capture frames from a camera using openCV and publish on an MQTT topic.
"""
import time
import sys

from mqtt import get_mqtt_client
from helpers import pil_image_to_byte_array, get_now_string

from PIL import Image
import cv2

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC_CAMERA = "homie/mac_webcam/capture"
MQTT_QOS = 1

DATETIME_STR_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
VIDEO_SOURCE = 0


def main():
    client = get_mqtt_client()
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_start()

    # Open camera and capture frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    time.sleep(2)  # Webcam light should come on if using one

    try:
        while True:
            _, np_array = cap.read()  # capture a frame
            np_array_RGB = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(np_array_RGB)  #  PIL image
            # image.show() # for debugging only

            byte_array = pil_image_to_byte_array(image)
            client.publish(MQTT_TOPIC_CAMERA, byte_array, qos=MQTT_QOS)
            now = get_now_string()
            print(f"published frame on topic: {MQTT_TOPIC_CAMERA} at {now}")

    except KeyboardInterrupt:
        print("Closing camera")
        cap.release()
        sys.exit()


if __name__ == "__main__":
    main()
