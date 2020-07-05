"""
Subscribes to the feed, does processing on the image, and forwards as new feed.
"""
import os
import time

from helpers import (
    byte_array_to_pil_image,
    get_config,
    get_now_string,
    pil_image_to_byte_array,
)
from mqtt import get_mqtt_client

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_SUBSCRIBE_TOPIC = CONFIG["processing"]["subscribe_topic"]
MQTT_PUBLISH_TOPIC = CONFIG["processing"]["publish_topic"]

ROTATE_ANGLE = 45  # Angle of rotation in degrees to apply

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    now = get_now_string()
    print("message on " + str(msg.topic) + f" at {now}")
    try:
        image = byte_array_to_pil_image(msg.payload)  # PIL image
        image = image.rotate(ROTATE_ANGLE)  # Apply rotation
        byte_array = pil_image_to_byte_array(image)

        client.publish(MQTT_PUBLISH_TOPIC, byte_array, qos=MQTT_QOS)
        print(f"published processed frame on topic: {MQTT_PUBLISH_TOPIC} at {now}")

    except Exception as exc:
        print(exc)


def main():
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_SUBSCRIBE_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
