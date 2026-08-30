"""How long a bound gesture takes, read off the victim's own five recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np

__all__ = [
    "FiveShotGestureTimingError",
    "GestureDurationLaw",
    "MIN_TRAVEL_PX",
    "carrier_window_imu",
    "contact_travel_px",
    "duration_law_from_pairs",
    "law_from_material",
    "window_slice_bounds",
]

# Travel below this reads as a stationary contact, where a duration drawn from
# the travel says nothing.  It is the floor the law is evaluated at, not a
# filter: the gesture still keeps whatever endpoints it was asked for.
MIN_TRAVEL_PX = 1.0


class FiveShotGestureTimingError(RuntimeError):
    """Raised when the victim's material cannot support a duration."""


@dataclass(frozen=True)
class GestureDurationLaw:
    """The victim's own travel-to-duration curve, five points wide."""

    log_travel_px: np.ndarray
    log_duration_ms: np.ndarray
    source_event_ids: tuple[str, ...]
    log_residuals: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    log_duration_support: tuple[float, float] = (-math.inf, math.inf)

    def __post_init__(self) -> None:
        if self.log_travel_px.shape != self.log_duration_ms.shape:
            raise FiveShotGestureTimingError("law knots are not paired")
        if len(self.log_travel_px) < 2:
            raise FiveShotGestureTimingError(
                "a travel-to-duration law needs at least two distinct travels"
            )
        if np.any(np.diff(self.log_travel_px) <= 0.0):
            raise FiveShotGestureTimingError("law knots are not strictly sorted")
        if self.log_residuals.ndim != 1:
            raise FiveShotGestureTimingError("law residuals must be one row")

    def duration_ms(self, travel_px: float, *, log_offset: float = 0.0) -> float:
        """Report how long the victim takes to cover this travel."""

        travel = float(travel_px)
        if not math.isfinite(travel) or not math.isfinite(log_offset):
            raise FiveShotGestureTimingError("travel and offset must be finite")
        return float(np.exp(self._reading(travel) + float(log_offset)))

    def residual_index(self, draw: int) -> int:
        """Report which of the victim's departures a draw lands on."""

        if not len(self.log_residuals):
            return -1
        return int(draw) % len(self.log_residuals)

    def _reading(self, travel_px: float) -> float:
        return float(
            np.interp(
                math.log(max(float(travel_px), MIN_TRAVEL_PX)),
                self.log_travel_px,
                self.log_duration_ms,
            )
        )

    def residual(self, draw: int) -> float:
        """Pick one of the victim's own departures from their own curve."""

        if not len(self.log_residuals):
            return 0.0
        return float(self.log_residuals[int(draw) % len(self.log_residuals)])

    @property
    def knots(self) -> int:
        return int(len(self.log_travel_px))

    @property
    def residual_spread(self) -> float:
        """Report the spread the victim's material shows around its own curve."""

        if len(self.log_residuals) < 2:
            return 0.0
        return float(np.std(self.log_residuals, ddof=1))


def duration_law_from_pairs(
    pairs: Iterable[tuple[float, float]],
    *,
    source_event_ids: Sequence[str],
) -> GestureDurationLaw:
    """Read the victim's five (travel, duration) pairs as an interpolable curve."""

    travels: list[float] = []
    durations: list[float] = []
    for travel, duration in pairs:
        travel = float(travel)
        duration = float(duration)
        if not math.isfinite(travel) or not math.isfinite(duration):
            continue
        if travel <= 0.0 or duration <= 0.0:
            continue
        travels.append(math.log(max(travel, MIN_TRAVEL_PX)))
        durations.append(math.log(duration))
    if len(travels) < 2:
        raise FiveShotGestureTimingError(
            "the victim's material carries fewer than two usable gestures"
        )
    x = np.asarray(travels, dtype=np.float64)
    y = np.asarray(durations, dtype=np.float64)
    unique = np.unique(x)
    averaged = np.asarray(
        [float(y[x == value].mean()) for value in unique], dtype=np.float64
    )
    if len(unique) < 2:
        raise FiveShotGestureTimingError(
            "the victim's material covers a single travel, so travel says nothing"
        )
    return GestureDurationLaw(
        log_travel_px=unique,
        log_duration_ms=averaged,
        source_event_ids=tuple(str(value) for value in source_event_ids),
        log_residuals=_leave_one_out_residuals(x, y),
        log_duration_support=(float(y.min()), float(y.max())),
    )


def _leave_one_out_residuals(
    log_travel: np.ndarray, log_duration: np.ndarray
) -> np.ndarray:
    """Measure how far each recording lands from the curve the others draw."""

    residuals: list[float] = []
    for index in range(len(log_travel)):
        keep = np.ones(len(log_travel), dtype=bool)
        keep[index] = False
        rest_x = log_travel[keep]
        rest_y = log_duration[keep]
        unique = np.unique(rest_x)
        if len(unique) < 2:
            continue
        averaged = np.asarray(
            [float(rest_y[rest_x == value].mean()) for value in unique],
            dtype=np.float64,
        )
        predicted = float(np.interp(log_travel[index], unique, averaged))
        residuals.append(float(log_duration[index]) - predicted)
    if len(residuals) < 2:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(residuals, dtype=np.float64)


def contact_travel_px(
    trajectory: np.ndarray, *, width_px: float, height_px: float
) -> float:
    """Measure how far a recorded contact actually moved, in pixels."""

    rows = np.asarray(trajectory)
    if rows.ndim != 2 or rows.shape[1] < 3 or not len(rows):
        raise FiveShotGestureTimingError("trajectory rows are malformed")
    contact = np.flatnonzero(rows[:, 0] > 0.5)
    if len(contact) < 2:
        return 0.0
    first = contact[0]
    last = contact[-1]
    dx = (float(rows[last, 1]) - float(rows[first, 1])) * float(width_px)
    dy = (float(rows[last, 2]) - float(rows[first, 2])) * float(height_px)
    return float(math.hypot(dx, dy))


def window_slice_bounds(
    *,
    active_start: int,
    active_stop: int,
    window_samples: int,
    samples: int,
) -> tuple[int, int]:
    """Choose which stretch of the padded window the gesture is cut from."""

    window_samples = int(window_samples)
    samples = int(samples)
    active_start = int(active_start)
    active_stop = int(active_stop)
    if window_samples < 2 or samples < 2:
        raise FiveShotGestureTimingError("a window slice needs at least two frames")
    if samples > window_samples:
        raise FiveShotGestureTimingError(
            f"requested {samples} frames from a {window_samples}-frame window"
        )
    if not 0 <= active_start < active_stop <= window_samples:
        raise FiveShotGestureTimingError("the active span is outside the window")
    active_samples = active_stop - active_start
    if samples <= active_samples:
        start = active_start + (active_samples - samples) // 2
    else:
        start = active_start - (samples - active_samples) // 2
    start = max(0, min(start, window_samples - samples))
    return start, start + samples


def carrier_window_imu(
    *,
    window: np.ndarray,
    mask: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cut `samples` frames of the carrier's own IMU around its action."""

    rows = np.asarray(window)
    flags = np.asarray(mask)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise FiveShotGestureTimingError("carrier window must hold six IMU channels")
    if flags.shape != (len(rows),):
        raise FiveShotGestureTimingError("carrier mask does not match its window")
    active = np.flatnonzero(flags.astype(bool))
    if not len(active):
        raise FiveShotGestureTimingError("carrier window marks no action")
    requested = int(samples)
    capped = min(requested, len(rows))
    start, stop = window_slice_bounds(
        active_start=int(active[0]),
        active_stop=int(active[-1]) + 1,
        window_samples=len(rows),
        samples=capped,
    )
    audit = {
        "carrier_window_samples": int(len(rows)),
        "carrier_active_samples": int(active[-1] + 1 - active[0]),
        "carrier_active_span": [int(active[0]), int(active[-1] + 1)],
        "requested_samples": requested,
        "cut_span": [int(start), int(stop)],
        "capped_to_window": bool(capped != requested),
    }
    return np.ascontiguousarray(rows[start:stop], dtype=np.float32), audit


def law_from_material(
    shots: Sequence[Any],
    *,
    width_px: float,
    height_px: float,
    duration_key: str = "raw_duration_ms",
) -> GestureDurationLaw:
    """Build the law from a victim's frozen material rows."""

    pairs: list[tuple[float, float]] = []
    ids: list[str] = []
    for shot in shots:
        row: Mapping[str, Any] = shot.row
        duration = row.get(duration_key, shot.duration_ms)
        travel = contact_travel_px(
            shot.trajectory, width_px=width_px, height_px=height_px
        )
        pairs.append((travel, float(duration)))
        ids.append(str(shot.event_id))
    return duration_law_from_pairs(pairs, source_event_ids=ids)
