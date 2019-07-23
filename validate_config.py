from helpers import get_config
import pprint

CONFIG_FILE = "config.yml"

try:
    CONFIG = get_config(CONFIG_FILE)
    pprint.pprint(CONFIG)
except Exception as exc:
    print(f"Invalid config in {CONFIG_FILE}")
    print(exc)
