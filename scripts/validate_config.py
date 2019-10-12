import os
from helpers import get_config
import pprint

CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")

try:
    CONFIG = get_config(CONFIG_FILE_PATH)
except Exception as exc:
    print(f"Invalid path to config file: {CONFIG_FILE_PATH}")
    print(exc)

try:
    pprint.pprint(CONFIG)
    print(f"Valid config in {CONFIG_FILE_PATH}, congratulations!")
except Exception as exc:
    print(f"Invalid config in {CONFIG_FILE_PATH}")
    print(exc)
