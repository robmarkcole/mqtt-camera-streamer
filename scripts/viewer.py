import os
import time

import numpy as np
import streamlit as st
from helpers import byte_array_to_pil_image, get_config, get_now_string
from mqtt import get_mqtt_client
from paho.mqtt import client as mqtt
from PIL import Image

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC = CONFIG["save-captures"]["mqtt_topic"]

VIEWER_WIDTH = 600


def get_random_numpy():
    """Return a dummy frame."""
    return np.random.randint(0, 100, size=(32, 32))


title = st.title(MQTT_TOPIC)
viewer = st.image(get_random_numpy(), width=VIEWER_WIDTH)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    st.write(f"Connected with result code {str(rc)} to MQTT broker on {MQTT_BROKER}")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if msg.topic != MQTT_TOPIC:
        return
    image = byte_array_to_pil_image(msg.payload)
    image = image.convert("RGB")
    viewer.image(image, width=VIEWER_WIDTH)


def main():
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
