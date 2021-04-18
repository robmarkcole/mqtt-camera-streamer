"""
Helper functions.

Source -> https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
"""
import datetime
import io

import yaml
from PIL import Image
import sqlite3

DATETIME_STR_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"


def pil_image_to_byte_array(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, "PNG")
    return imgByteArr.getvalue()


def byte_array_to_pil_image(byte_array):
    return Image.open(io.BytesIO(byte_array))


def get_now_string() -> str:
    return datetime.datetime.now().strftime(DATETIME_STR_FORMAT)


def get_config(config_filepath: str) -> dict:
    with open(config_filepath) as f:
        config = yaml.safe_load(f)
    return config


# Create a function to connect to a database with SQLite
def sqlite_connect(db_name: str) -> sqlite3.Connection:
    """Connect to a database if exists. Create an instance if otherwise.
    Args:
        db_name: The name of the database to connect
    Returns:
        an sqlite3.connection object
    """
    try:
        # Create a connection
        conn = sqlite3.connect(db_name)
    except sqlite3.Error:
        print(f"Error connecting to the database '{db_name}'")
    finally:
        return conn


def convert_into_binary(file_path: str):
    with open(file_path, "rb") as file:
        binary = file.read()
    return binary
