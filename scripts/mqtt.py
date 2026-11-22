"""
Some boilerplate code to handle MQTT.
"""

import os

from paho.mqtt import client as mqtt


# Reqired callbacks
def on_connect(client, userdata, flags, rc, properties):
    # print(f"CONNACK received with code {rc}")
    if rc == 0:
        print("connected to MQTT broker")
        client.connected_flag = True  # set flag
    else:
        print("Bad connection to MQTT broker, returned code=", rc)


def on_publish(client, userdata, mid, reason_code, properties):
    print("mid: " + str(mid))


def get_mqtt_client():
    """Return the MQTT client object."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    username, password = os.getenv("MQTT_USERNAME"), os.getenv("MQTT_PASSWORD")
    if password and not username:
        raise ValueError("MQTT password is configured without a username")
    if username:
        client.username_pw_set(username, password)
    client.connected_flag = False  # set flag
    client.on_connect = on_connect
    client.on_publish = on_publish
    return client
