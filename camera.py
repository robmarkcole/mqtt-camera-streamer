"""
Capture frames from a camera using openCV and publish on an MQTT topic.
"""
import time
import io
import mqtt as mqtt

from PIL import Image
import cv2

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC = "homie/mac_webcam/capture"
MQTT_QOS = 1

VIDEO_SOURCE = 0


def main():
    client = mqtt.get_mqtt_client()
    # client.username_pw_set(username, password)
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_start()

    # Open camera and capture frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    time.sleep(3)  # Webcam light should come on
    _, np_array = cap.read()  # capture a frame
    print("Frame captured!")
    imgRGB = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = Image.fromarray(imgRGB)  #  PIL image
    # image.show() # for debugging only

    ## Get the bytearray
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, "PNG")
    imgByteArr = imgByteArr.getvalue()

    # Publish
    client.publish(MQTT_TOPIC, imgByteArr, qos=MQTT_QOS)
    print(f"published frame on topic: {MQTT_TOPIC}")
    print("completed")
    cap.release()


if __name__ == "__main__":
    main()
