"""
Subscribes to the feed, does processing on the image, and forwards as new feed.
"""
import datetime
import time
import io
import mqtt as mqtt

from PIL import Image

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_SUBSCRIBE_TOPIC = "homie/mac_webcam/capture"
MQTT_PUBLISH_TOPIC = "homie/mac_webcam/capture/rotated"
MQTT_QOS = 1

DATETIME_STR_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
ROTATE_ANGLE = 45  # Angle of rotation in degrees to apply

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    date_string = datetime.datetime.now().strftime(DATETIME_STR_FORMAT)
    print("message on " + str(msg.topic) + f" at {date_string}")
    try:
        image = Image.open(io.BytesIO(msg.payload))  # PIL image
        image = image.rotate(ROTATE_ANGLE) # Apply rotation

        ## Get the bytearray
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, "PNG")
        imgByteArr = imgByteArr.getvalue()

        # Publish
        client.publish(MQTT_PUBLISH_TOPIC, imgByteArr, qos=MQTT_QOS)
        date_string = datetime.datetime.now().strftime(DATETIME_STR_FORMAT)
        print(f"published frame on topic: {MQTT_PUBLISH_TOPIC} at {date_string}")

    except Exception as exc:
        print(exc)


def main():
    client = mqtt.get_mqtt_client()
    # client.username_pw_set(username, password)
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_SUBSCRIBE_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
