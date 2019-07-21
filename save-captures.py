"""
Subscribe and save camera images with timestamp.
"""

import mqtt as mqtt
# from PIL import Image

MQTT_BROKER = "192.168.1.164"
MQTT_PORT = 1883
MQTT_TOPIC = "homie/mac_webcam/capture"

def main():
    client = mqtt.get_mqtt_client()
    # client.username_pw_set(username, password)
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4) # Wait for connection setup to complete
    client.loop_start()

    image_file = "tests/images/test_jpg_1.jpg"
    with open(image_file, "rb") as file:
        filecontent = file.read()
        byteArr = bytearray(filecontent)
        client.publish(MQTT_TOPIC, byteArr, qos=1)
        print(f"published frame on topic: {MQTT_TOPIC}")
    print("completed")


if __name__ == "__main__":
    main()