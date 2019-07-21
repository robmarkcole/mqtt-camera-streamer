"""
Subscribe and save camera images with timestamp.
"""
import time

from helpers import byte_array_to_pil_image, get_now_string
from mqtt import get_mqtt_client

CAPTURES_DIRECTORY = "tests/captures/"
DATETIME_STR_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC = "homie/mac_webcam/capture"

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    now = get_now_string()
    print("message on " + str(msg.topic) + f" at {now}")

    try:
        image = byte_array_to_pil_image(msg.payload)
        image = image.convert("RGB")

        save_file_path = CAPTURES_DIRECTORY + f"capture_{now}.jpg"
        image.save(save_file_path)
        print(f"Saved {save_file_path}")

    except Exception as exc:
        print(exc)


def main():
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
