"""
Helper functions.

Source -> https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
"""
import datetime
import io
from PIL import Image
import yaml

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
