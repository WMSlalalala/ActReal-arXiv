"""The trajectory half: de-hold, re-hold, and geometry conditioning."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal import config
from actreal.touch_track import (
    DOWN,
    UP,
    TouchPoint,
    TouchTrack,
    contact_runs,
    detector_grid_ms,
    detector_grid_span_ms,
    from_observation,
    to_observation,
)

ACTIONS = ("tap", "scroll", "swipe", "keystroke")


def _release_grid_span_ms(samples: int) -> float:
    """The release observation implementation packaged with the artifact."""

    root = str(config.GENERATION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from pipeline.android_touch_observation import (  # type: ignore
        detector_grid_span_ms as upstream,
    )

    return upstream(samples)


@pytest.mark.parametrize("samples", [2, 3, 11, 35, 179, 256, 512, 1000])
def test_grid_span_matches_release_bit_for_bit(samples):
    assert detector_grid_span_ms(samples) == _release_grid_span_ms(samples)


def test_grid_span_rejects_degenerate_counts():
    with pytest.raises(ValueError):
        detector_grid_span_ms(1)


def test_contact_runs_splits_a_keystroke_into_key_presses():
    flags = [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    assert contact_runs(flags) == [(1, 2), (5, 7), (9, 9)]


def test_contact_runs_handles_no_contact():
    assert contact_runs([0, 0, 0]) == []


def _make_track(points):
    return TouchTrack(
        action="tap",
        points=points,
        orientation_id=0,
        screen_w=1080.0,
        screen_h=1920.0,
    )


def test_track_rejects_a_gesture_that_does_not_start_with_down():
    with pytest.raises(ValueError):
        _make_track([TouchPoint(0.0, 1.0, 1.0, 1.0, 0.0, 0, UP)])


def test_track_rejects_time_going_backwards():
    with pytest.raises(ValueError):
        _make_track(
            [
                TouchPoint(10.0, 1.0, 1.0, 1.0, 0.0, 0, DOWN),
                TouchPoint(0.0, 1.0, 1.0, 1.0, 0.0, 0, UP),
            ]
        )


def test_reanchor_translation_puts_down_exactly_on_target_and_keeps_shape():
    track = _make_track(
        [
            TouchPoint(0.0, 100.0, 200.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(50.0, 108.0, 214.0, 1.0, 0.0, 0, UP),
        ]
    )
    moved = track.reanchor((540.0, 1200.0))
    assert moved.down_xy == pytest.approx((540.0, 1200.0))
    # The lift drift is carried over untouched, not re-derived.
    assert moved.travel_px() == pytest.approx(track.travel_px())


def test_reanchor_chord_hits_both_requested_endpoints():
    track = _make_track(
        [
            TouchPoint(0.0, 100.0, 200.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(25.0, 140.0, 260.0, 1.0, 0.0, 0, "MOVE"),
            TouchPoint(50.0, 300.0, 200.0, 1.0, 0.0, 0, UP),
        ]
    )
    moved = track.reanchor((540.0, 1600.0), (540.0, 400.0))
    assert moved.down_xy == pytest.approx((540.0, 1600.0), abs=1e-6)
    assert moved.up_xy == pytest.approx((540.0, 400.0), abs=1e-6)


def test_reanchor_chord_falls_back_to_translation_for_a_donor_without_travel():
    track = _make_track(
        [
            TouchPoint(0.0, 100.0, 200.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(50.0, 100.0, 200.0, 1.0, 0.0, 0, UP),
        ]
    )
    moved = track.reanchor((540.0, 1200.0), (540.0, 400.0))
    assert moved.down_xy == pytest.approx((540.0, 1200.0))


def test_retime_scales_the_gesture_and_keeps_the_sample_pattern():
    track = _make_track(
        [
            TouchPoint(0.0, 100.0, 200.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(30.0, 120.0, 200.0, 1.0, 0.0, 0, "MOVE"),
            TouchPoint(60.0, 140.0, 200.0, 1.0, 0.0, 0, UP),
        ]
    )
    slower = track.retime(120.0)
    assert slower.duration_ms == pytest.approx(120.0)
    assert [p.t_ms for p in slower.points] == pytest.approx([0.0, 60.0, 120.0])


def test_single_frame_contact_still_produces_a_valid_lifecycle():
    traj = np.zeros((3, 9), dtype=np.float32)
    traj[1] = (1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.01, 1.0)
    traj[:, 7] = np.array([0.0, 0.01, 0.02], dtype=np.float32)
    track = from_observation(traj, action="tap")
    assert [p.action for p in track.points] == [DOWN, UP]
    assert track.duration_ms == pytest.approx(config.GRID_PERIOD_MS)


def _synthetic_observation(action: str, frames: int) -> np.ndarray:
    """A deterministic held trajectory with no participant-derived samples."""

    trajectory = np.zeros((frames, 9), dtype=np.float32)
    trajectory[:, 7] = np.arange(frames, dtype=np.float32) * np.float32(0.01)
    trajectory[:, 8] = 1.0
    first, last = 2, frames - 3
    for index in range(first, last + 1):
        # Repeat each semantic position twice so de-holding has rows to remove.
        progress = (index - first) // 2
        denominator = max(1, (last - first) // 2)
        fraction = float(progress) / float(denominator)
        if action == "tap":
            x, y = 0.42 + 0.002 * fraction, 0.57 + 0.001 * fraction
        elif action == "scroll":
            x, y = 0.50 + 0.003 * fraction, 0.80 - 0.55 * fraction
        else:
            x, y = 0.20 + 0.62 * fraction, 0.51 + 0.004 * fraction
        trajectory[index, :5] = (1.0, x, y, 0.72, 1.0)
    return trajectory


def _synthetic_keystroke_observation() -> np.ndarray:
    trajectory = np.zeros((18, 9), dtype=np.float32)
    trajectory[:, 7] = np.arange(18, dtype=np.float32) * np.float32(0.01)
    trajectory[:, 8] = 1.0
    for run_index, (first, last) in enumerate(((1, 3), (6, 8), (12, 15))):
        for index in range(first, last + 1):
            trajectory[index, :5] = (
                1.0,
                0.30 + 0.12 * run_index,
                0.82 - 0.03 * run_index,
                0.70 + 0.02 * run_index,
                1.0,
            )
    return trajectory


@pytest.mark.parametrize(
    "action,frames", [("tap", 35), ("scroll", 179), ("swipe", 167)]
)
def test_synthetic_events_survive_de_hold_and_re_hold_exactly(action, frames):
    trajectory = _synthetic_observation(action, frames)
    track = from_observation(trajectory, action=action)
    span_ms = float(trajectory[-1, 7]) * 1000.0
    rebuilt = to_observation(track, frames, span_ms=span_ms)

    contact_ref = trajectory[:, 0] > 0.5
    contact_new = rebuilt[:, 0] > 0.5
    assert (contact_ref == contact_new).all(), f"{action}: contact pattern changed"

    # Columns 1..3 are x, y and pressure -- what actually gets injected.
    delta = np.abs(rebuilt[contact_ref][:, 1:4] - trajectory[contact_ref][:, 1:4])
    assert delta.max() < 1e-5, f"{action}: {delta.max()} coordinate drift"


def test_de_hold_drops_the_repeated_rows_the_hold_invented():
    trajectory = _synthetic_observation("tap", 35)
    track = from_observation(trajectory, action="tap")
    # A tap holds still for most of its window; the delivered events are far
    # fewer than the grid rows, which is the whole point of undoing the hold.
    assert len(track.points) < len(trajectory)


def test_keystroke_is_realised_as_one_contact_not_key_by_key():
    """An agent types through Android's text interface, not by touching keys.

    The recording has one contact run per key, but no agent produces that:
    the text goes in through the input command or the clipboard and no per-key
    MotionEvent is ever dispatched.  Delivering the recorded key-by-key
    structure would put a touch pattern on screen that the agent's own typing
    never makes.
    """

    track = from_observation(
        _synthetic_keystroke_observation(), action="keystroke"
    )
    downs = sum(1 for p in track.points if p.action == DOWN)
    ups = sum(1 for p in track.points if p.action == UP)
    assert (downs, ups) == (1, 1), "a keystroke is one pointer lifecycle"
    assert track.points[0].action == DOWN
    assert track.points[-1].action == UP


def test_merging_a_keystroke_never_emits_a_gap_frames_zeroes():
    """Gap rows are zeros, not coordinates, and must not become touch points."""

    track = from_observation(
        _synthetic_keystroke_observation(), action="keystroke"
    )
    assert all(p.x > 0.0 and p.y > 0.0 for p in track.points)
    assert all(p.pressure > 0.0 for p in track.points)


def test_segmentation_can_still_be_asked_for_explicitly():
    traj = np.zeros((7, 9), dtype=np.float32)
    for i in (0, 1, 4, 5):
        traj[i] = (1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    traj[:, 7] = np.arange(7, dtype=np.float32) * 0.01

    merged = from_observation(traj, action="keystroke", single_segment=True)
    split = from_observation(traj, action="keystroke", single_segment=False)
    assert sum(1 for p in merged.points if p.action == DOWN) == 1
    assert sum(1 for p in split.points if p.action == DOWN) == 2
