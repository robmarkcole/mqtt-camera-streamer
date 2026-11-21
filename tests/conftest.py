"""Shared camera configuration and loopback MQTT broker fixtures."""

import asyncio
import socket
import sys
import threading
from pathlib import Path

import pytest
from amqtt.broker import Broker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from onnx_camera import CameraConfig


@pytest.fixture
def config():
    return CameraConfig(
        camera_id="test_camera",
        name="Test camera",
        model_path=Path("nano.onnx"),
        stale_seconds=2,
        max_fps=2,
        classes={"person": ["person"], "vehicle": ["car", "truck"]},
        mqtt={"broker": "127.0.0.1"},
    )


@pytest.fixture
def broker_port():
    ready = threading.Event()
    control = {}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def serve():
        broker = Broker(
            {
                "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}},
                "plugins": {
                    "amqtt.plugins.authentication.AnonymousAuthPlugin": {
                        "allow_anonymous": True
                    }
                },
            }
        )
        await broker.start()
        control.update(loop=asyncio.get_running_loop(), stop=asyncio.Event())
        ready.set()
        await control["stop"].wait()
        await broker.shutdown()

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    assert ready.wait(10), "test broker failed to start"
    try:
        yield port
    finally:
        control["loop"].call_soon_threadsafe(control["stop"].set)
        thread.join(10)
