from __future__ import annotations

"""Canonical Android-event observation for genuine and replayed touch data.

The detector grid is a regular 100 Hz clock, but Android touch delivery is
not.  Coordinates therefore use zero-order hold (the most recent MotionEvent)
rather than interpolation between future events.  This distinction is
important: interpolation invents observations that Android never delivered
and leaves a very strong synthetic-data fingerprint in ``dx``/``dy``.

The functions in this module are deliberately label blind.  They accept raw
Android rows and apply exactly one observation operator to genuine data,
direct replays, and deformed replays.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ACTION_MASK = 0xFF
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
ACTION_CANCEL = 3
ACTION_POINTER_DOWN = 5
ACTION_POINTER_UP = 6

PERIOD_MS = 10.0
TRAJECTORY_CHANNELS = (
    "contact_mask",
    "x_rel",
    "y_rel",
    "pressure",
    "pointer_count",
    "dx_rel",
    "dy_rel",
    "elapsed_seconds",
    "availability",
)


class TouchObservationError(ValueError):
    pass


@dataclass(frozen=True)
class TouchObservation:
    """One canonical detector-grid observation.

    ``touch`` has seven columns: contact, x, y, pressure, pointer count,
    dx, and dy.  ``trajectory`` appends elapsed seconds and an event-level
    availability bit.
    """

    touch: np.ndarray
    trajectory: np.ndarray
    touch_observed: bool
    source_updates: int


def screen_dimensions_for_orientation(orientation_id: int) -> tuple[float, float]:
    """Return HMOG Galaxy S4 width/height in physical pixels."""

    if int(orientation_id) in (1, 3):
        return 1920.0, 1080.0
    if int(orientation_id) in (-1, 0, 2):
        return 1080.0, 1920.0
    raise TouchObservationError(f"unknown orientation_id {orientation_id!r}")


def _as_vector(name: str, value: Iterable[object], dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise TouchObservationError(f"{name} must be one-dimensional")
    return result


def _frame_slices(
    count: int,
    *,
    frame_end: np.ndarray | None,
    frame_index: np.ndarray | None,
) -> list[slice]:
    if count < 1:
        return []
    if frame_end is not None:
        ends = np.flatnonzero(frame_end.astype(bool)) + 1
        if not len(ends) or int(ends[-1]) != count:
            raise TouchObservationError("frame_end does not terminate every row")
        starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
        return [slice(int(left), int(right)) for left, right in zip(starts, ends)]
    if frame_index is None:
        return [slice(index, index + 1) for index in range(count)]
    changes = np.flatnonzero(frame_index[1:] != frame_index[:-1]) + 1
    bounds = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            changes,
            np.asarray([count], dtype=np.int64),
        )
    )
    return [
        slice(int(left), int(right))
        for left, right in zip(bounds[:-1], bounds[1:])
    ]


def _validate_physical_xy(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    width_px: float,
    height_px: float,
) -> None:
    if not np.isfinite(x_px).all() or not np.isfinite(y_px).all():
        raise TouchObservationError("touch coordinates contain non-finite values")
    if (
        np.any(x_px < 0.0)
        or np.any(x_px > width_px)
        or np.any(y_px < 0.0)
        or np.any(y_px > height_px)
    ):
        raise TouchObservationError("touch coordinates leave the physical screen")


def _scaled_times(
    times_ms: np.ndarray,
    *,
    source_duration_ms: float | None,
    target_samples: int,
    target_duration_ms: float,
) -> np.ndarray:
    if target_samples < 2:
        raise TouchObservationError("target_samples must be at least two")
    if not np.isfinite(times_ms).all() or np.any(np.diff(times_ms) < 0.0):
        raise TouchObservationError("Android event times must be finite and ordered")
    start = float(times_ms[0])
    relative = times_ms - start
    observed_end = float(np.max(relative))
    source_duration = (
        observed_end
        if source_duration_ms is None
        else float(source_duration_ms)
    )
    if not np.isfinite(source_duration) or source_duration <= 0.0:
        raise TouchObservationError("source_duration_ms must be positive")
    if source_duration + 1.0e-6 < observed_end:
        raise TouchObservationError("source_duration_ms ends before the last row")
    if not np.isfinite(target_duration_ms) or target_duration_ms <= 0.0:
        raise TouchObservationError("target_duration_ms must be positive")
    # Normalize first so an observed row exactly at source_duration maps to
    # target_duration exactly.  Multiplying by the precomputed ratio can put
    # that endpoint a few ulps after the detector's final grid time, causing
    # searchsorted to miss ACTION_UP and hold the penultimate row instead.
    return (relative / source_duration) * float(target_duration_ms)


def _state_arrays(
    updates: list[tuple[float, tuple[tuple[float, float, float], ...]]],
) -> tuple[np.ndarray, np.ndarray]:
    if not updates:
        return np.empty(0, dtype=np.float64), np.empty((0, 5), dtype=np.float64)
    times: list[float] = []
    states: list[list[float]] = []
    for time_ms, pointers in updates:
        if pointers:
            values = np.asarray(pointers, dtype=np.float64)
            state = [
                1.0,
                float(np.mean(values[:, 0])),
                float(np.mean(values[:, 1])),
                float(np.mean(values[:, 2])),
                float(len(values)),
            ]
        else:
            state = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Multiple Android transitions may share a timestamp.  The state after
        # the final transition is what a 100 Hz observer sees at that instant.
        if times and abs(time_ms - times[-1]) <= 1.0e-9:
            states[-1] = state
        else:
            times.append(float(time_ms))
            states.append(state)
    return np.asarray(times, dtype=np.float64), np.asarray(states, dtype=np.float64)


def _gesture_updates(
    *,
    action: str,
    frames: list[slice],
    times_ms: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    pressure: np.ndarray,
    pointer_id: np.ndarray,
    android_action: np.ndarray,
) -> list[tuple[float, tuple[tuple[float, float, float], ...]]]:
    updates: list[tuple[float, tuple[tuple[float, float, float], ...]]] = []
    for frame in frames:
        positions = np.arange(frame.start, frame.stop, dtype=np.int64)
        if action == "pinch":
            # Pinch features are defined only by simultaneous two-pointer
            # observations.  A trailing primary ACTION_UP must never be mixed
            # with the previous two-pointer centroid.
            chosen: list[int] = []
            for pointer in np.unique(pointer_id[positions]):
                candidates = positions[pointer_id[positions] == pointer]
                base = android_action[candidates] & ACTION_MASK
                live = candidates[
                    ~np.isin(base, (ACTION_UP, ACTION_CANCEL))
                ]
                if len(live):
                    chosen.append(int(live[-1]))
            if len(chosen) < 2:
                continue
            chosen = sorted(chosen, key=lambda index: int(pointer_id[index]))[:2]
        else:
            # Single-pointer strokes keep the last coordinate carried by every
            # MotionEvent, including ACTION_UP.  Event windows in the genuine
            # HMOG release cover the gesture itself, so the observation remains
            # available throughout that window.
            candidates = positions[pointer_id[positions] == pointer_id[positions][0]]
            chosen = [int(candidates[-1])]
        pointers = tuple(
            (
                float(x_rel[index]),
                float(y_rel[index]),
                float(pressure[index]),
            )
            for index in chosen
        )
        updates.append((float(times_ms[chosen[-1]]), pointers))
    if not updates:
        raise TouchObservationError(f"{action} has no usable Android frames")
    # Stretch the usable gesture state across the whole detector event.  This
    # matches the genuine event slicing and, for pinch, avoids reintroducing a
    # one-pointer tail after selecting the simultaneous interval.
    if updates[0][0] > 0.0:
        updates.insert(0, (0.0, updates[0][1]))
    return updates


def _keystroke_updates(
    *,
    frames: list[slice],
    times_ms: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    pressure: np.ndarray,
    pointer_id: np.ndarray,
    android_action: np.ndarray,
    key_index: np.ndarray,
) -> list[tuple[float, tuple[tuple[float, float, float], ...]]]:
    active: dict[tuple[int, int], tuple[float, float, float]] = {}
    updates: list[tuple[float, tuple[tuple[float, float, float], ...]]] = []
    for frame in frames:
        positions = np.arange(frame.start, frame.stop, dtype=np.int64)
        removals: list[tuple[int, int]] = []
        for index in positions:
            identity = (int(key_index[index]), int(pointer_id[index]))
            base = int(android_action[index]) & ACTION_MASK
            value = (
                float(x_rel[index]),
                float(y_rel[index]),
                float(pressure[index]),
            )
            if base in (ACTION_DOWN, ACTION_POINTER_DOWN):
                active[identity] = value
            elif base == ACTION_MOVE:
                if identity in active:
                    active[identity] = value
            elif base in (ACTION_UP, ACTION_POINTER_UP, ACTION_CANCEL):
                removals.append(identity)
        for identity in removals:
            active.pop(identity, None)
        snapshot = tuple(active[key] for key in sorted(active))
        updates.append((float(times_ms[positions[-1]]), snapshot))
    return updates


def observe_android_rows(
    *,
    action: str,
    target_samples: int,
    orientation_id: int,
    t_ms: Iterable[object],
    x_px: Iterable[object],
    y_px: Iterable[object],
    pressure: Iterable[object],
    pointer_id: Iterable[object],
    android_action: Iterable[object],
    frame_end: Iterable[object] | None = None,
    frame_index: Iterable[object] | None = None,
    key_index: Iterable[object] | None = None,
    source_duration_ms: float | None = None,
    period_ms: float = PERIOD_MS,
    target_duration_ms: float | None = None,
) -> TouchObservation:
    """Observe irregular Android rows on a regular detector clock.

    No coordinate, pressure, or movement value is linearly interpolated.  The
    only time transformation is a global event-duration warp performed before
    zero-order hold.
    """

    if action not in {"tap", "scroll", "swipe", "pinch", "keystroke"}:
        raise TouchObservationError(f"unsupported action {action!r}")
    times = _as_vector("t_ms", t_ms, np.float64)
    x = _as_vector("x_px", x_px, np.float64)
    y = _as_vector("y_px", y_px, np.float64)
    force = _as_vector("pressure", pressure, np.float64)
    pointers = _as_vector("pointer_id", pointer_id, np.int64)
    actions = _as_vector("android_action", android_action, np.int64)
    arrays = (times, x, y, force, pointers, actions)
    if not len(times) or any(len(value) != len(times) for value in arrays):
        raise TouchObservationError("Android row arrays must have equal non-zero length")
    if not np.isfinite(force).all() or np.any(force < 0.0) or np.any(force > 1.0):
        raise TouchObservationError("pressure must be finite and in [0, 1]")
    width_px, height_px = screen_dimensions_for_orientation(orientation_id)
    _validate_physical_xy(x, y, width_px=width_px, height_px=height_px)
    # Physical bounds were validated above.  Normalize by division only: even
    # a sub-pixel out-of-bounds value must be rejected instead of silently
    # projected back onto the screen edge.
    x_rel = x / width_px
    y_rel = y / height_px
    output_duration_ms = (
        float((target_samples - 1) * period_ms)
        if target_duration_ms is None
        else float(target_duration_ms)
    )
    scaled = _scaled_times(
        times,
        source_duration_ms=source_duration_ms,
        target_samples=target_samples,
        target_duration_ms=output_duration_ms,
    )
    ends = (
        None
        if frame_end is None
        else _as_vector("frame_end", frame_end, np.uint8)
    )
    indices = (
        None
        if frame_index is None
        else _as_vector("frame_index", frame_index, np.int64)
    )
    if ends is not None and len(ends) != len(times):
        raise TouchObservationError("frame_end length mismatch")
    if indices is not None and len(indices) != len(times):
        raise TouchObservationError("frame_index length mismatch")
    frames = _frame_slices(len(times), frame_end=ends, frame_index=indices)
    if action == "keystroke":
        keys = (
            np.full(len(times), -1, dtype=np.int64)
            if key_index is None
            else _as_vector("key_index", key_index, np.int64)
        )
        if len(keys) != len(times):
            raise TouchObservationError("key_index length mismatch")
        updates = _keystroke_updates(
            frames=frames,
            times_ms=scaled,
            x_rel=x_rel,
            y_rel=y_rel,
            pressure=force,
            pointer_id=pointers,
            android_action=actions,
            key_index=keys,
        )
    else:
        updates = _gesture_updates(
            action=action,
            frames=frames,
            times_ms=scaled,
            x_rel=x_rel,
            y_rel=y_rel,
            pressure=force,
            pointer_id=pointers,
            android_action=actions,
        )
    update_times, states = _state_arrays(updates)
    grid = np.linspace(
        0.0,
        output_duration_ms,
        target_samples,
        dtype=np.float64,
    )
    touch = np.zeros((target_samples, 7), dtype=np.float32)
    if len(update_times):
        selected = np.searchsorted(update_times, grid, side="right") - 1
        valid = selected >= 0
        touch[valid, :5] = states[selected[valid]].astype(np.float32)
    consecutive = (touch[1:, 0] > 0.0) & (touch[:-1, 0] > 0.0)
    differences = np.diff(touch[:, 1:3], axis=0)
    touch[1:, 5][consecutive] = differences[consecutive, 0]
    touch[1:, 6][consecutive] = differences[consecutive, 1]
    observed = bool(len(update_times))
    trajectory = trajectory_from_touch(
        touch,
        touch_observed=observed,
        period_ms=period_ms,
        duration_ms=output_duration_ms,
    )
    return TouchObservation(
        touch=touch,
        trajectory=trajectory,
        touch_observed=observed,
        source_updates=len(update_times),
    )


def detector_grid_clock(samples: int, duration_ms: float) -> np.ndarray:
    """Return the elapsed column exactly as the observer writes it.

    The detector clock belongs to the observer, not to whatever produced the
    contact: it is a regular grid spanning the observed event.  Every producer
    has to build it with these operations in this order, because a float32
    column keeps the rounding signature of the arithmetic that filled it.  A
    path that reaches the same nominal seconds another way -- rescaling a
    donor's own clock, or dividing before taking the linear span instead of
    after -- lands an ulp away on some samples, and a detector reads that
    difference without learning anything about the gesture.
    """

    count = int(samples)
    span_ms = float(duration_ms)
    if count < 2:
        raise TouchObservationError("detector grid needs at least two samples")
    if not np.isfinite(span_ms) or span_ms <= 0.0:
        raise TouchObservationError("detector grid duration must be positive")
    return np.linspace(0.0, span_ms / 1000.0, count, dtype=np.float32)


def detector_grid_span_ms(samples: int, period_ms: float = PERIOD_MS) -> float:
    """Report what a row count spans, as the observer's own clock carries it.

    Every duration in this pipeline has already been through a float32 elapsed
    column before it is handed back to the clock: a genuine event's comes off
    ``trajectory[-1, 7]``, and so does a carrier's.  A duration computed straight
    from a row count has not, and the exact decimal it produces fills a grid that
    is bit-identical to the ideal ramp, where every column that went the normal
    way carries a rounding residual near 3e-8.  Measured over 4,000 events each:
    5.12% of genuine scroll columns match that ideal exactly and 6.70% of
    carrier-timed fake ones, against 100.00% of the ones built from a clean
    decimal.  An axis-aligned tree splits that in one cut and learns nothing
    about the gesture -- ``elapsed_s.skew`` came out the top feature of
    ``hmog_style_rf`` at 5.6x the next, having not reached the top twenty before
    -- while an SVM or a sequence model cannot use it at all.  That is why one
    build improved every non-tree scroll cell and destroyed both tree ones.

    So a row count is converted here, through the same float32 seconds the rest
    of the pipeline hands over, and never by multiplying the period out.
    """

    count = int(samples)
    if count < 2:
        raise TouchObservationError("detector grid needs at least two samples")
    # Widen before scaling, exactly as `_detector_window_duration_ms` does when
    # it reads a duration back off a float32 elapsed column.  Scaling while
    # still narrow rounds the product back to the clean decimal and undoes the
    # whole point: float32(0.1) * 1000 is 100.0, float64(float32(0.1)) * 1000 is
    # 100.00000149011612, and only the second fills the grid a real column fills.
    span_s = float(np.float32(((count - 1) * float(period_ms)) / 1000.0))
    return span_s * 1000.0


def trajectory_from_touch(
    touch: np.ndarray,
    *,
    touch_observed: bool,
    period_ms: float = PERIOD_MS,
    duration_ms: float | None = None,
) -> np.ndarray:
    values = np.asarray(touch, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise TouchObservationError("touch must have shape [samples, 7]")
    if not np.isfinite(values).all():
        raise TouchObservationError("touch contains non-finite values")
    trajectory = np.zeros((len(values), 9), dtype=np.float32)
    trajectory[:, :7] = values
    output_duration_ms = (
        float((len(values) - 1) * period_ms)
        if duration_ms is None
        else float(duration_ms)
    )
    if not np.isfinite(output_duration_ms) or output_duration_ms <= 0.0:
        raise TouchObservationError("duration_ms must be positive")
    trajectory[:, 7] = detector_grid_clock(len(values), output_duration_ms)
    trajectory[:, 8] = float(bool(touch_observed))
    return trajectory


def active_xy_repeat_rate(trajectory: np.ndarray) -> float:
    """Return the fraction of consecutive active samples with unchanged XY."""

    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9:
        raise TouchObservationError("trajectory must have shape [samples, 9]")
    active = (values[1:, 0] > 0.0) & (values[:-1, 0] > 0.0)
    if not np.any(active):
        return 0.0
    repeated = np.all(values[1:, 1:3] == values[:-1, 1:3], axis=1)
    return float(np.mean(repeated[active]))


__all__ = [
    "PERIOD_MS",
    "TRAJECTORY_CHANNELS",
    "TouchObservation",
    "TouchObservationError",
    "active_xy_repeat_rate",
    "detector_grid_clock",
    "detector_grid_span_ms",
    "observe_android_rows",
    "screen_dimensions_for_orientation",
    "trajectory_from_touch",
]
