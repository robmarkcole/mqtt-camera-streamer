"""
Modified version of save-captires.py that additionally saves image thumbnail to sqlite db.

Ref https://towardsdev.com/storing-digital-files-in-remote-sql-databases-in-python-73494f09d39b
"""
import os
import time
import sqlite3

from helpers import (
    byte_array_to_pil_image,
    get_config,
    get_now_string,
    sqlite_connect,
    convert_into_binary,
    pil_image_to_byte_array,
)
from mqtt import get_mqtt_client

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

SAVE_TOPIC = CONFIG["save-captures"]["mqtt_topic"]
CAPTURES_DIRECTORY = CONFIG["save-captures"]["captures_directory"]

DB = CAPTURES_DIRECTORY + "/records.db"
DB_TABLE = "mqtt_camera"

CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    name TEXT NOT NULL, data BLOB
);
"""
THUMBNAIL_SIZE = (500, 200)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    now = get_now_string()
    print("message on " + str(msg.topic) + f" at {now}")

    try:
        image = byte_array_to_pil_image(msg.payload)
        image = image.convert("RGB")

        save_file_path = CAPTURES_DIRECTORY + f"capture_{now}.jpg"
        image.save(save_file_path)

        # Additional code to save thumbnail to db
        db_conn = sqlite_connect(DB)
        cursor = db_conn.cursor()

        db_insert_blob = """
        INSERT INTO {table_name} (name, data) VALUES (?, ?)
        """.format(
            table_name=DB_TABLE
        )

        image.thumbnail(size=THUMBNAIL_SIZE)
        binary_data = pil_image_to_byte_array(image)
        data_tuple = (save_file_path, binary_data)

        # Execute the query
        cursor.execute(db_insert_blob, data_tuple)
        db_conn.commit()
        print(f"Saved {save_file_path} and inserted to db")
        cursor.close()

    except Exception as exc:
        print(exc)


def main():
    db_conn = sqlite_connect(DB)
    cursor = db_conn.cursor()
    cursor.execute(CREATE_TABLE_DDL.format(table_name=DB_TABLE))
    db_conn.commit()
    db_conn.close()

    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(SAVE_TOPIC)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
