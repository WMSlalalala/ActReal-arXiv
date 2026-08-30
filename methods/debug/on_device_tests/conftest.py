"""Deterministic, license-clean fixtures for the on-device unit tests.

The publication artifact must not redistribute trajectories or IMU waveforms
derived from HMOG.  These bundles are therefore constructed in a temporary
directory from simple analytic signals.  They exercise serialization,
planning, injection, and accounting only; they are not evaluation samples and
must never be used for paper metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from actreal.bundle import ActionBundle
from actreal.duration_law import build_timing
from actreal.mapping import ScreenMapping
from actreal.touch_track import (
    DOWN,
    MOVE,
    POINTER_DOWN,
    POINTER_UP,
    UP,
    TouchPoint,
    TouchTrack,
)


DEVICE_W = 1080.0
DEVICE_H = 2424.0
PERIOD_MS = 10.0
TOUCH_OFFSET_MS = 100.0


def _synthetic_imu(action_index: int, frames: int = 96) -> list[list[float]]:
    """A deterministic six-axis analytic signal with a gravity component."""

    rows: list[list[float]] = []
    phase = 0.31 * float(action_index + 1)
    for frame in range(frames):
        t = frame * PERIOD_MS / 1000.0
        rows.append(
            [
                0.08 * math.sin(2.0 * math.pi * 1.3 * t + phase),
                0.06 * math.cos(2.0 * math.pi * 0.9 * t + phase),
                9.81 + 0.04 * math.sin(2.0 * math.pi * 1.7 * t),
                0.015 * math.sin(2.0 * math.pi * 1.1 * t + phase),
                0.012 * math.cos(2.0 * math.pi * 1.5 * t),
                0.010 * math.sin(2.0 * math.pi * 0.7 * t - phase),
            ]
        )
    return rows


def _single_touch(action: str, duration_ms: float, variant: int = 0) -> TouchTrack:
    x0 = 430.0 + 18.0 * variant
    y0 = 1180.0 + 12.0 * variant
    if action == "tap":
        points = [
            TouchPoint(0.0, x0, y0, 0.72, 0.055, 0, DOWN),
            TouchPoint(duration_ms / 2.0, x0 + 2.0, y0 + 1.0, 0.76, 0.056, 0, MOVE),
            TouchPoint(duration_ms, x0 + 3.0, y0 + 2.0, 0.68, 0.054, 0, UP),
        ]
    elif action == "scroll":
        points = [
            TouchPoint(0.0, 540.0, 1680.0, 0.70, 0.056, 0, DOWN),
            TouchPoint(duration_ms * 0.33, 544.0, 1370.0, 0.74, 0.057, 0, MOVE),
            TouchPoint(duration_ms * 0.67, 548.0, 980.0, 0.71, 0.056, 0, MOVE),
            TouchPoint(duration_ms, 552.0, 650.0, 0.65, 0.054, 0, UP),
        ]
    elif action == "swipe":
        points = [
            TouchPoint(0.0, 260.0, 1210.0, 0.69, 0.055, 0, DOWN),
            TouchPoint(duration_ms * 0.5, 540.0, 1204.0, 0.73, 0.056, 0, MOVE),
            TouchPoint(duration_ms, 830.0, 1198.0, 0.66, 0.054, 0, UP),
        ]
    else:  # keystroke is represented as one synthetic contact for unit tests.
        points = [
            TouchPoint(0.0, 540.0, 1900.0, 0.74, 0.055, 0, DOWN),
            TouchPoint(duration_ms, 542.0, 1901.0, 0.67, 0.054, 0, UP),
        ]
    return TouchTrack(
        action=action,
        points=points,
        orientation_id=0,
        screen_w=DEVICE_W,
        screen_h=DEVICE_H,
        source="deterministic_synthetic_pytest_fixture",
    )


def _pinch(duration_ms: float) -> TouchTrack:
    return TouchTrack(
        action="pinch",
        points=[
            TouchPoint(0.0, 500.0, 1212.0, 0.72, 0.055, 0, DOWN),
            TouchPoint(0.0, 580.0, 1212.0, 0.72, 0.055, 1, POINTER_DOWN),
            TouchPoint(duration_ms / 2.0, 450.0, 1212.0, 0.74, 0.056, 0, MOVE),
            TouchPoint(duration_ms / 2.0, 630.0, 1212.0, 0.74, 0.056, 1, MOVE),
            TouchPoint(duration_ms, 400.0, 1212.0, 0.68, 0.054, 1, POINTER_UP),
            TouchPoint(duration_ms, 680.0, 1212.0, 0.68, 0.054, 0, UP),
        ],
        orientation_id=0,
        screen_w=DEVICE_W,
        screen_h=DEVICE_H,
        source="deterministic_synthetic_pytest_fixture",
    )


@pytest.fixture(scope="session")
def synthetic_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small bundle library without any participant-derived data."""

    directory = tmp_path_factory.mktemp("synthetic_actreal_bundles")
    mapping = ScreenMapping.isotropic(device_w=DEVICE_W, device_h=DEVICE_H)
    specifications = [
        ("tap", 100.0, 0),
        ("tap", 100.0, 1),
        ("scroll", 360.0, 0),
        ("swipe", 240.0, 0),
        ("pinch", 260.0, 0),
        ("keystroke", 420.0, 0),
    ]
    for index, (action, duration_ms, variant) in enumerate(specifications):
        touch = _pinch(duration_ms) if action == "pinch" else _single_touch(
            action, duration_ms, variant
        )
        bundle = ActionBundle(
            bundle_id=f"synthetic-{action}-{variant}",
            action=action,
            touch=touch,
            imu=_synthetic_imu(index),
            imu_period_ms=PERIOD_MS,
            touch_offset_ms=TOUCH_OFFSET_MS,
            mapping=mapping,
            provenance={
                "orientation_id": 0,
                "touch_source": "deterministic_synthetic_pytest_fixture",
                "imu_source": "analytic_sine_cosine_pytest_fixture",
                "imu_active_frames": max(1, int(round(duration_ms / PERIOD_MS))),
                "synthetic_fixture": True,
            },
        )
        destination = directory / f"synthetic_{action}_{variant}.json"
        destination.write_text(bundle.to_json() + "\n", encoding="utf-8")

    durations = {
        "tap": [70.0, 80.0, 90.0, 100.0, 110.0],
        "scroll": [280.0, 320.0, 360.0, 400.0, 440.0],
        "swipe": [180.0, 210.0, 240.0, 270.0, 300.0],
        "pinch": [200.0, 230.0, 260.0, 290.0, 320.0],
        "keystroke": [320.0, 370.0, 420.0, 470.0, 520.0],
    }
    actions = {
        action: build_timing(
            values,
            action=action,
            victim="synthetic-pytest-subject",
            source_event_ids=[f"synthetic-{action}-{i}" for i in range(len(values))],
        ).as_dict()
        for action, values in durations.items()
    }
    profile = {
        "schema_version": "actreal_timing_profile_v1",
        "split": "synthetic_fixture",
        "source": "deterministic_pytest_fixture",
        "actions": actions,
    }
    (directory / "timing_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory
