"""Record legacy MQTT images as JPEG files and relative paths in SQLite."""

import io
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from helpers import get_config
from media import SnapshotStore, StorageConfig
from mqtt import get_mqtt_client
from PIL import Image

LOG = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    config = get_config(os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml"))
    capture = config["save-captures"]
    storage = StorageConfig(enabled=True, directory=Path(capture["captures_directory"]))
    store = SnapshotStore(storage)
    client = get_mqtt_client()

    def on_connect(client, userdata, flags, reason, properties):
        if not reason.is_failure:
            client.subscribe(capture["mqtt_topic"], qos=config["mqtt"]["QOS"])

    def on_message(client, userdata, message):
        try:
            with Image.open(io.BytesIO(message.payload)) as image:
                output = io.BytesIO()
                image.convert("RGB").save(output, format="JPEG", quality=80)
            store.save(
                output.getvalue(),
                message.topic,
                {
                    "predicted_at": datetime.now(UTC).isoformat(),
                    "mqtt_topic": message.topic,
                },
            )
        except Exception:
            LOG.exception("Could not record MQTT image")

    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(config["mqtt"]["broker"], config["mqtt"]["port"])
        client.loop_forever()
    finally:
        client.disconnect()
        store.close()


if __name__ == "__main__":
    main()
