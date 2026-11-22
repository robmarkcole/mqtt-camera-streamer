"""Exercise the real Roboflow tracker with missed frames and elapsed-time holds."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from onnx_camera import CameraConfig, HomeAssistantPublisher, PredictionProcessor
from test_onnx_camera import Observer, detections


@pytest.fixture
def processing_config(config):
    config.stale_seconds = 30
    return config


def person(confidence=0.9):
    return sv.Detections(
        xyxy=np.array([[10, 10, 100, 100]], dtype=float),
        confidence=np.array([confidence]),
        class_id=np.array([1]),
        data={"class_name": np.array(["person"])},
    )


def test_confirmation_gap_recovery_and_low_confidence_matching(processing_config):
    processor = PredictionProcessor(processing_config)
    # Weak detections cannot start a track, and isolated strong detections cannot confirm it.
    assert not processor.process(person(0.3), 1, 0, now=0)["presence"]["person"]
    assert not processor.process(person(), 2, 0, now=0.5)["presence"]["person"]
    confirmed = processor.process(person(), 3, 0, now=1)
    track_id = confirmed["detections"][0]["tracker_id"]
    assert confirmed["counts"]["person"] == 1
    missed = processor.process(sv.Detections.empty(), 4, 0, now=1.5)
    assert missed["counts"]["person"] == 0
    assert missed["presence"]["person"]
    recovered = processor.process(person(0.3), 5, 0, now=2)
    assert recovered["detections"][0]["tracker_id"] == track_id
    assert recovered["presence"]["person"]


def test_single_frame_spikes_do_not_create_presence(processing_config):
    processor = PredictionProcessor(processing_config)
    for sequence in range(8):
        output = processor.process(
            person() if sequence % 2 == 0 else sv.Detections.empty(),
            sequence,
            0,
            now=sequence * 0.5,
        )
        assert not output["presence"]["person"]
        assert output["counts"]["person"] == 0


def test_independent_hold_durations_and_instantaneous_counts(processing_config):
    processing_config.presence.hold_seconds_by_class = {"person": 2, "vehicle": 4}
    processor = PredictionProcessor(processing_config)
    processor.process(detections(), 1, 0, now=0)
    both = processor.process(detections(), 2, 0, now=0.5)
    assert both["presence"] == {"person": True, "vehicle": True}
    gap = processor.process(sv.Detections.empty(), 3, 0, now=1)
    assert gap["counts"] == {"person": 0, "vehicle": 0}
    assert gap["presence"] == {"person": True, "vehicle": True}
    person_expired = processor.process(sv.Detections.empty(), 4, 0, now=2.5)
    assert person_expired["presence"] == {"person": False, "vehicle": True}
    all_expired = processor.process(sv.Detections.empty(), 5, 0, now=4.5)
    assert all_expired["presence"] == {"person": False, "vehicle": False}


def test_new_track_inside_hold_does_not_start_another_presence_episode(
    processing_config,
):
    processor = PredictionProcessor(processing_config)
    processor.process(person(), 1, 0, now=0)
    initial = processor.process(person(), 2, 0, now=0.5)
    processor.process(sv.Detections.empty(), 3, 0, now=1)
    # Beyond the tracker recovery budget, but within the group's presence hold.
    replacement = processor.process(person(), 4, 0, now=3)
    assert replacement["counts"]["person"] == 0
    assert replacement["presence"]["person"]
    confirmed = processor.process(person(), 5, 0, now=3.5)
    assert (
        confirmed["detections"][0]["tracker_id"]
        != initial["detections"][0]["tracker_id"]
    )
    assert confirmed["presence"]["person"]
    assert (
        processor.process(sv.Detections.empty(), 6, 0, now=8.5)["presence"]["person"]
        is False
    )


def test_outage_resets_tracks_and_hold_even_with_long_hold(config):
    config.presence.hold_seconds = 100
    processor = PredictionProcessor(config)
    processor.process(person(), 1, 0, now=0)
    assert processor.process(person(), 2, 0, now=0.5)["presence"]["person"]
    resumed = processor.process(person(), 3, 0, now=3)
    assert resumed["presence"]["person"] is False
    assert resumed["counts"]["person"] == 0


def test_tracking_can_be_disabled_and_hold_can_be_zero(processing_config):
    processing_config.tracking.enabled = False
    processing_config.presence.hold_seconds = 0
    processor = PredictionProcessor(processing_config)
    visible = processor.process(person(), 1, 0, now=0)
    assert visible["presence"]["person"]
    assert visible["detections"][0]["tracker_id"] is None
    assert not processor.process(sv.Detections.empty(), 2, 0, now=0.5)["presence"][
        "person"
    ]


@pytest.mark.parametrize(
    "section,changes",
    [
        ("tracking", {"minimum_consecutive_frames": 0}),
        ("tracking", {"lost_track_seconds": 0}),
        ("tracking", {"recovery_threshold": 0.9}),
        ("presence", {"hold_seconds": -1}),
        ("presence", {"hold_seconds": float("nan")}),
        ("presence", {"hold_seconds_by_class": {"person": float("inf")}}),
        ("presence", {"hold_seconds_by_class": {"typo": 5}}),
    ],
)
def test_invalid_processing_settings_rejected(config, section, changes):
    values = config.model_dump()
    values[section].update(changes)
    with pytest.raises(ValueError):
        CameraConfig.model_validate(values)


def test_mqtt_keeps_presence_on_across_missing_frames(processing_config, broker_port):
    processing_config.mqtt.port = broker_port
    processing_config.presence.hold_seconds = 2
    processor = PredictionProcessor(processing_config)
    observer = Observer(broker_port)
    states = []
    inputs = [
        (0, person()),
        (0.5, person()),
        (1, sv.Detections.empty()),
        (1.5, person()),
        (2, sv.Detections.empty()),
        (3.5, sv.Detections.empty()),
    ]
    try:
        with HomeAssistantPublisher(processing_config) as publisher:
            for sequence, (now, frame) in enumerate(inputs):
                after = len(observer.messages)
                publisher.publish(processor.process(frame, sequence, 0, now=now))
                state = json.loads(
                    observer.wait("mqtt_camera/test_camera/state", after=after)[1]
                )
                states.append(state)
        assert [s["presence"]["person"] for s in states] == [
            False,
            True,
            True,
            True,
            True,
            False,
        ]
        assert [s["counts"]["person"] for s in states] == [0, 1, 0, 1, 0, 0]
    finally:
        observer.close()
