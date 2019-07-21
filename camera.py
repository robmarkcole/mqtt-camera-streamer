"""
Capture frames from a camera using openCV and publish on an MQTT topic.
"""

from PIL import Image
import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "homie/mac_webcam/capture"

# Reqired callbacks
def on_connect(client, userdata, flags, rc):
    print(f"CONNACK received with code {rc}")
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_publish(client, userdata, mid):
    print("mid: " + str(mid))


def get_mqtt_client():
    """Return the MQTT client object."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish

    # client.username_pw_set(username, password)
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.loop_start()
    return client


def main():
    client = get_mqtt_client()
    image_file = "tests/images/test_jpg_1.jpg"
    with open(image_file, "rb") as file:
        filecontent = file.read()
        byteArr = bytearray(filecontent)
        client.publish(MQTT_TOPIC, byteArr, qos=1)


if __name__ == "__main__":
    main()
