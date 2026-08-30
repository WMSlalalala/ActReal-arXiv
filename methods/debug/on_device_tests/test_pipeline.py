"""Mapping, bundling, and the round trip through the app protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal.bundle import ActionBundle, compile_bundle, from_json_dict, load_bundle
from actreal.control import ControlClient, ControlError
from actreal.imu_source import IMUWindow
from actreal.mapping import ScreenMapping
from actreal.session import play_bundle, verify
from actreal.simulator import AppSimulator
from actreal.touch_track import DOWN, UP, TouchPoint, TouchTrack

PIXEL_W, PIXEL_H = 1080, 2424


def _mapping(w: int = PIXEL_W, h: int = PIXEL_H) -> ScreenMapping:
    return ScreenMapping.isotropic(device_w=w, device_h=h)


def _track(action: str = "tap") -> TouchTrack:
    return TouchTrack(
        action=action,
        points=[
            TouchPoint(0.0, 400.0, 800.0, 0.6, 0.1, 0, DOWN),
            TouchPoint(40.0, 404.0, 806.0, 0.7, 0.1, 0, UP),
        ],
        orientation_id=0,
        screen_w=1080.0,
        screen_h=1920.0,
        source="unit",
    )


def _window(action: str = "tap", frames: int = 12, active=(3, 8), period_ms: float = 10.0):
    rng = np.random.default_rng(5)
    return IMUWindow(
        action=action,
        samples=rng.normal(size=(frames, 6)).astype(np.float32),
        active_start=active[0],
        active_end=active[1],
        orientation_id=0,
        requested_duration_ms=(active[1] - active[0]) * period_ms,
        period_ms=period_ms,
        source="unit",
    )


# -- mapping ------------------------------------------------------------------


def test_isotropic_mapping_uses_one_factor_for_both_axes():
    m = _mapping()
    assert m.scale == pytest.approx(min(PIXEL_W / 1080, PIXEL_H / 1920))
    # Same source distance in x and y must map to the same device distance,
    # or the kinematics the detectors were fitted on would change.
    dx = m.to_device((100.0, 0.0))[0] - m.to_device((0.0, 0.0))[0]
    dy = m.to_device((0.0, 100.0))[1] - m.to_device((0.0, 0.0))[1]
    assert dx == pytest.approx(dy)


def test_mapping_round_trip_returns_the_original_point():
    m = _mapping(1440, 3120)
    for xy in [(0.0, 0.0), (540.0, 960.0), (1079.0, 1919.0)]:
        assert m.to_source(m.to_device(xy)) == pytest.approx(xy)


def test_usable_rect_is_the_part_of_the_screen_with_a_source_coordinate():
    m = _mapping()
    left, top, right, bottom = m.usable_rect
    assert m.to_source((left, top)) == pytest.approx((0.0, 0.0))
    assert m.to_source((right, bottom)) == pytest.approx((1080.0, 1920.0))
    assert m.contains_device_point((540.0, 1200.0))
    assert not m.contains_device_point((540.0, 10.0))


def test_clamping_reports_how_far_a_target_had_to_move():
    m = _mapping()
    (x, y), moved = m.clamp_device_point((540.0, 10.0))
    assert y == pytest.approx(m.usable_rect[1])
    assert moved > 0
    _, unmoved = m.clamp_device_point((540.0, 1200.0))
    assert unmoved == 0.0


def test_track_survives_a_trip_through_device_space():
    m = _mapping(1440, 3120)
    track = _track()
    back = m.track_to_source(m.track_to_device(track))
    for a, b in zip(track.points, back.points):
        assert (a.x, a.y) == pytest.approx((b.x, b.y))


# -- bundling -----------------------------------------------------------------


def test_bundle_places_the_touch_on_the_windows_first_active_frame():
    bundle = compile_bundle(
        action="tap", touch=_track(), imu=_window(), mapping=_mapping()
    )
    # Active span starts at frame 3, so the gesture begins 30 ms in and the
    # leading frames are inertia with no contact.
    assert bundle.touch_offset_ms == pytest.approx(30.0)
    assert bundle.touch_start_ms == pytest.approx(30.0)
    assert bundle.touch_end_ms == pytest.approx(70.0)


def test_bundle_refuses_a_gesture_longer_than_its_own_inertia():
    long_track = TouchTrack(
        action="tap",
        points=[
            TouchPoint(0.0, 400.0, 800.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(5000.0, 400.0, 800.0, 1.0, 0.0, 0, UP),
        ],
        orientation_id=0,
        screen_w=1080.0,
        screen_h=1920.0,
    )
    with pytest.raises(ValueError, match="of IMU remains"):
        compile_bundle(
            action="tap", touch=long_track, imu=_window(), mapping=_mapping()
        )


def test_bundle_refuses_mismatched_halves():
    with pytest.raises(ValueError, match="but IMU is"):
        compile_bundle(
            action="tap", touch=_track("tap"), imu=_window("scroll"), mapping=_mapping()
        )


def test_bundle_carries_a_non_nominal_period_rather_than_assuming_100hz():
    bundle = compile_bundle(
        action="keystroke",
        touch=_track("keystroke"),
        imu=_window("keystroke", frames=512, active=(0, 512), period_ms=59.84),
        mapping=_mapping(),
    )
    assert bundle.imu_period_ms == pytest.approx(59.84)
    assert bundle.imu_duration_ms == pytest.approx(512 * 59.84)


def test_bundle_json_round_trip_preserves_everything_that_gets_played():
    bundle = compile_bundle(
        action="tap", touch=_track(), imu=_window(), mapping=_mapping()
    )
    back = from_json_dict(json.loads(bundle.to_json()))
    assert back.bundle_id == bundle.bundle_id
    assert back.touch_offset_ms == pytest.approx(bundle.touch_offset_ms)
    assert back.imu_period_ms == pytest.approx(bundle.imu_period_ms)
    assert back.touch_points_on_bundle_clock() == bundle.touch_points_on_bundle_clock()
    assert np.allclose(back.imu_rows(), bundle.imu_rows(), atol=1e-5)


def test_bundle_rejects_a_ragged_imu_window():
    with pytest.raises(ValueError, match="6 channels"):
        ActionBundle(
            bundle_id="x",
            action="tap",
            touch=_track(),
            imu=[[0.0] * 6, [0.0] * 5],
            imu_period_ms=10.0,
            touch_offset_ms=0.0,
            mapping=_mapping(),
        )


# -- protocol round trip ------------------------------------------------------


def _play(action: str, start, end, bundle_dir: Path) -> tuple:
    sim = AppSimulator(display_w=PIXEL_W, display_h=PIXEL_H)
    client = ControlClient(sim)
    candidates = sorted(bundle_dir.glob(f"synthetic_{action}_*.json"))
    if not candidates:
        raise AssertionError(f"synthetic fixture has no {action} bundle")
    bundle = load_bundle(candidates[0]).reanchored(start, end)
    client.clear()
    receipt = play_bundle(client, bundle)
    return client, bundle, receipt


@pytest.mark.parametrize(
    "action,start,end",
    [
        ("tap", (540.0, 1200.0), None),
        ("scroll", (540.0, 1500.0), (540.0, 700.0)),
        ("swipe", (250.0, 1100.0), (830.0, 1100.0)),
    ],
)
def test_every_planned_sample_reaches_the_app(
    action, start, end, synthetic_bundle_dir
):
    client, bundle, receipt = _play(action, start, end, synthetic_bundle_dir)
    report = verify(client.dump(), bundle, receipt)
    assert report.ok, report.as_dict()
    assert report.touch.missing == 0
    assert report.imu.missing == 0
    assert report.alignment_error_ms == pytest.approx(0.0, abs=1e-6)


def test_scheduling_imu_while_the_bus_is_in_real_mode_is_refused():
    sim = AppSimulator()
    client = ControlClient(sim)
    client.set_imu_mode("real")
    with pytest.raises(ControlError, match="real mode"):
        client.schedule_imu([[0.0] * 6], start_in_ms=0.0)


def test_a_dropped_touch_point_is_reported_as_missing_not_as_drift(
    synthetic_bundle_dir,
):
    client, bundle, receipt = _play(
        "scroll", (540.0, 1500.0), (540.0, 700.0), synthetic_bundle_dir
    )
    dump = client.dump()
    # Simulate the input pipeline losing one event in the middle.
    del dump["touch"][len(dump["touch"]) // 2]
    report = verify(dump, bundle, receipt)
    assert report.touch.missing == 1
    # The surviving points must still match on value, not cascade into errors.
    assert report.touch.max_value_error == pytest.approx(0.0)


def test_background_frames_do_not_count_as_the_bundles_own_imu(
    synthetic_bundle_dir,
):
    client, bundle, receipt = _play(
        "tap", (540.0, 1200.0), None, synthetic_bundle_dir
    )
    dump = client.dump()
    dump["imu"].append([999, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, "injected", "background", -1])
    report = verify(dump, bundle, receipt)
    assert report.imu.extra == 0
    assert report.imu.missing == 0


def test_clock_offset_is_applied_rather_than_assumed_zero(synthetic_bundle_dir):
    # The simulator's offset is deliberately large; a host that ignored it
    # would report an alignment error of about that many milliseconds.
    client, bundle, receipt = _play(
        "tap", (540.0, 1200.0), None, synthetic_bundle_dir
    )
    report = verify(client.dump(), bundle, receipt)
    assert abs(report.alignment_error_ms) < 1e-6
