"""
Some boilerplate code to handle MQTT.
"""
from paho.mqtt import client as mqtt


# Reqired callbacks
def on_connect(client, userdata, flags, rc):
    # print(f"CONNACK received with code {rc}")
    if rc == 0:
        print("connected to MQTT broker")
        client.connected_flag = True  # set flag
    else:
        print("Bad connection to MQTT broker, returned code=", rc)


def on_publish(client, userdata, mid):
    print("mid: " + str(mid))


def get_mqtt_client():
    """Return the MQTT client object."""
    client = mqtt.Client()
    client.connected_flag = False  # set flag
    client.on_connect = on_connect
    client.on_publish = on_publish
    return client
