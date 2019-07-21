"""
Subscribe and save camera images with timestamp.
"""
import time
import mqtt as mqtt

#  from PIL import Image

CAPTURES_DIRECTORY = "tests/captures/"

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC = "homie/mac_webcam/capture"

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print("on_message: " + str(msg.topic))


def main():
    client = mqtt.get_mqtt_client()
    # client.username_pw_set(username, password)
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
