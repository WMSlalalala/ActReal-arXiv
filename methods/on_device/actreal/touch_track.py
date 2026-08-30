"""The touch half of an action: the Android events, not the observation grid.

The release stores trajectories as detector-grid observations: a regular 100 Hz
sampling where each row repeats the most recent MotionEvent (zero-order hold).
That is what a detector sees, but it is not what can be injected -- replaying
one row per 10 ms would deliver ten times the events Android ever dispatched,
and the invented rows would be the strongest synthetic fingerprint in the
stream.

:func:`from_observation` therefore inverts the hold: a row that repeats the
previous one carried no new MotionEvent and is dropped.  What survives is the
event sequence Android actually delivered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

from . import config


def _numpy():
    """Import numpy only where array work actually happens.

    The packaged local build loads bundles, plays them and checks what came
    back -- none of which touches an array.  Keeping the import out of module
    scope is what lets that package run on a laptop with nothing installed.
    """

    import numpy as np

    return np

DOWN = "DOWN"
MOVE = "MOVE"
UP = "UP"
POINTER_DOWN = "POINTER_DOWN"
POINTER_UP = "POINTER_UP"

# A tap keeps its DOWN anchor exactly and may drift on lift within the device's
# own touch slop; inside this radius Android still dispatches the contact as a
# single-finger tap rather than a scroll.
TAP_DRIFT_LIMIT_PX = 24.0

# Actions whose recorded gaps do not become separate pointer lifecycles.
#
# Typing is the case.  The recording has one contact run per key, but an agent
# does not type that way: it hands the text to Android through the input
# command or the clipboard, and no per-key MotionEvent is ever dispatched.
# Reproducing the recorded key-by-key structure would deliver a touch pattern
# the agent's own typing never produces, so a keystroke is realised as one
# contact spanning the event.
SINGLE_SEGMENT_ACTIONS = frozenset({"keystroke"})


@dataclass(frozen=True)
class TouchPoint:
    t_ms: float
    x: float
    y: float
    pressure: float
    size: float
    pointer_id: int
    action: str


@dataclass
class TouchTrack:
    """One gesture as a list of MotionEvents on a relative millisecond clock."""

    action: str
    points: list[TouchPoint]
    orientation_id: int
    screen_w: float
    screen_h: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a touch track needs at least one point")
        if self.points[0].action != DOWN:
            raise ValueError(f"first point must be {DOWN}, got {self.points[0].action}")
        if self.points[-1].action not in (UP, POINTER_UP):
            raise ValueError(f"last point must be an UP, got {self.points[-1].action}")
        times = [p.t_ms for p in self.points]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("touch points must be non-decreasing in time")

    @property
    def duration_ms(self) -> float:
        return float(self.points[-1].t_ms - self.points[0].t_ms)

    @property
    def down_xy(self) -> tuple[float, float]:
        return (self.points[0].x, self.points[0].y)

    @property
    def up_xy(self) -> tuple[float, float]:
        return (self.points[-1].x, self.points[-1].y)

    @property
    def pointer_count(self) -> int:
        return len({p.pointer_id for p in self.points})

    def travel_px(self) -> float:
        (x0, y0), (x1, y1) = self.down_xy, self.up_xy
        return float(math.hypot(x1 - x0, y1 - y0))

    # -- geometry conditioning ------------------------------------------------

    def reanchor(
        self,
        start: tuple[float, float],
        end: Optional[tuple[float, float]] = None,
    ) -> "TouchTrack":
        """Move this donor gesture onto the geometry the agent asked for.

        With ``end`` omitted the track is translated so DOWN lands on ``start``;
        the shape, and therefore any lift drift, is carried over untouched.
        That is the tap case: Android hit-tests the DOWN coordinate only.

        With ``end`` given the chord is mapped onto the requested chord -- a
        rotation and a uniform scale, so both endpoints land exactly while the
        residual shape around the chord is preserved.  Travel is the outcome of
        a scroll or a swipe, so both of its ends have to be honoured.
        """

        sx, sy = float(start[0]), float(start[1])
        dx0, dy0 = self.down_xy

        if end is None:
            shifted = [replace(p, x=p.x - dx0 + sx, y=p.y - dy0 + sy) for p in self.points]
            return replace(self, points=shifted, source=self.source + "+translate")

        ex, ey = float(end[0]), float(end[1])
        ux, uy = self.up_xy
        src_vx, src_vy = ux - dx0, uy - dy0
        src_len = math.hypot(src_vx, src_vy)
        dst_vx, dst_vy = ex - sx, ey - sy
        dst_len = math.hypot(dst_vx, dst_vy)
        if src_len < 1e-6:
            # A donor with no travel cannot define a chord frame; fall back to
            # translation so the caller still gets the requested DOWN.
            return self.reanchor(start)
        if dst_len < 1e-6:
            return self.reanchor(start)

        scale = dst_len / src_len
        src_ang = math.atan2(src_vy, src_vx)
        dst_ang = math.atan2(dst_vy, dst_vx)
        rot = dst_ang - src_ang
        cos_r, sin_r = math.cos(rot) * scale, math.sin(rot) * scale

        mapped = []
        for p in self.points:
            rx, ry = p.x - dx0, p.y - dy0
            mapped.append(
                replace(p, x=sx + cos_r * rx - sin_r * ry, y=sy + sin_r * rx + cos_r * ry)
            )
        return replace(self, points=mapped, source=self.source + "+chord")

    def retime(self, duration_ms: float) -> "TouchTrack":
        """Scale the gesture onto a new duration, keeping the sample pattern."""

        current = self.duration_ms
        if current <= 0 or duration_ms <= 0:
            return self
        factor = float(duration_ms) / current
        t0 = self.points[0].t_ms
        return replace(
            self,
            points=[replace(p, t_ms=t0 + (p.t_ms - t0) * factor) for p in self.points],
            source=self.source + "+retime",
        )

    def shifted(self, offset_ms: float) -> "TouchTrack":
        return replace(self, points=[replace(p, t_ms=p.t_ms + offset_ms) for p in self.points])

    def clamped(self, width: float, height: float) -> "TouchTrack":
        """Keep every point on screen; report how far anything had to move."""

        out = []
        for p in self.points:
            out.append(
                replace(
                    p,
                    x=min(max(p.x, 0.0), width - 1.0),
                    y=min(max(p.y, 0.0), height - 1.0),
                )
            )
        return replace(self, points=out)

    def max_shift_from(self, other: "TouchTrack") -> float:
        return max(
            math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(self.points, other.points)
        )


def screen_for_orientation(orientation_id: int) -> tuple[float, float]:
    """HMOG Galaxy S4 physical pixels, the space every trajectory lives in."""

    if int(orientation_id) in (1, 3):
        return 1920.0, 1080.0
    if int(orientation_id) in (-1, 0, 2):
        return 1080.0, 1920.0
    raise ValueError(f"unknown orientation_id {orientation_id!r}")


def from_observation(
    trajectory: "object",
    *,
    action: str,
    orientation_id: int = 0,
    source: str = "released_event",
    move_epsilon_px: float = 1e-3,
    single_segment: Optional[bool] = None,
) -> TouchTrack:
    """Undo the zero-order hold and recover the delivered MotionEvents.

    ``trajectory`` is (T, 9): contact, x_rel, y_rel, pressure, pointer_count,
    dx_rel, dy_rel, elapsed_seconds, availability.

    ``single_segment`` collapses the recording's separate contact runs into one
    pointer lifecycle; it defaults to true for the actions in
    :data:`SINGLE_SEGMENT_ACTIONS` and false otherwise.
    """

    np = _numpy()
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 9:
        raise ValueError(f"trajectory must be (T,9), got {traj.shape}")

    width, height = screen_for_orientation(orientation_id)
    contact = traj[:, 0] > 0.5
    if not contact.any():
        raise ValueError("trajectory has no contact frames")

    if single_segment is None:
        single_segment = action in SINGLE_SEGMENT_ACTIONS
    runs = contact_runs(contact)
    if single_segment and runs:
        runs = [(runs[0][0], runs[-1][1])]

    points: list[TouchPoint] = []
    for first, last in runs:
        # Each run of contact frames is one pointer lifecycle -- except where
        # the action is realised as a single contact (see
        # SINGLE_SEGMENT_ACTIONS), in which case the runs were merged above and
        # the frames between them are carried as MOVEs.
        prev: Optional[tuple[float, float, float]] = None
        run: list[TouchPoint] = []
        for i in range(first, last + 1):
            if not contact[i]:
                # A merged span crosses frames the recording had no contact on.
                # Those rows are zeros, not coordinates; the pointer simply
                # stays where the last real frame left it.
                continue
            x = float(traj[i, 1] * width)
            y = float(traj[i, 2] * height)
            pressure = float(traj[i, 3])
            # The event's own clock is the float32 elapsed column, widened.
            # Its step is not always the nominal 10 ms -- a long keystroke is
            # spread across a capped window -- so the column is read, never
            # reconstructed from the frame index.
            t_ms = float(traj[i, 7]) * 1000.0
            current = (x, y, pressure)
            is_first = i == first
            is_last = i == last
            changed = prev is None or any(
                abs(a - b) > move_epsilon_px for a, b in zip(current, prev)
            )
            if not (is_first or is_last or changed):
                # A repeated row is the hold, not a new event Android delivered.
                continue
            run.append(
                TouchPoint(
                    t_ms=t_ms,
                    x=x,
                    y=y,
                    pressure=pressure,
                    size=0.0,
                    pointer_id=0,
                    action=DOWN if is_first else (UP if is_last else MOVE),
                )
            )
            prev = current
        if len(run) == 1:
            # DOWN and UP at the same instant is not a dispatchable gesture;
            # give the lift the grid period so the lifecycle stays valid.
            only = run[0]
            run.append(replace(only, t_ms=only.t_ms + config.GRID_PERIOD_MS, action=UP))
        points.extend(run)

    return TouchTrack(
        action=action,
        points=points,
        orientation_id=int(orientation_id),
        screen_w=width,
        screen_h=height,
        source=source,
    )


def contact_runs(contact: Sequence[bool]) -> list[tuple[int, int]]:
    """Inclusive [first, last] index pairs for each run of contact frames."""

    np = _numpy()
    flags = np.asarray(contact, dtype=bool)
    if not flags.any():
        return []
    edges = np.diff(flags.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if flags[0]:
        starts.insert(0, 0)
    if flags[-1]:
        ends.append(len(flags) - 1)
    return list(zip(starts, ends))


def detector_grid_span_ms(samples: int, period_ms: float = config.GRID_PERIOD_MS) -> float:
    """What a row count spans, carried the way the observer's clock carries it.

    Mirrors ``security_exp.android_touch_observation.detector_grid_span_ms``
    bit for bit, and ``tests/test_grid_matches_release.py`` checks it against
    that implementation.  The widening is not cosmetic: every duration in the
    pipeline has been through a float32 elapsed column, and a span computed
    straight from a row count has not.  The clean decimal fills a grid that is
    bit-identical to the ideal ramp while a real column carries a rounding
    residual near 3e-8 -- an axis-aligned tree splits that in one cut and
    learns nothing about the gesture.
    """

    count = int(samples)
    if count < 2:
        raise ValueError("detector grid needs at least two samples")
    np = _numpy()
    span_s = float(np.float32(((count - 1) * float(period_ms)) / 1000.0))
    return span_s * 1000.0


def detector_grid_ms(samples: int, span_ms: Optional[float] = None):
    """The event clock, in milliseconds, as float64 widened from float32."""

    count = int(samples)
    if span_ms is None:
        span_ms = detector_grid_span_ms(count)
    np = _numpy()
    grid_s = np.linspace(0.0, float(span_ms) / 1000.0, count, dtype=np.float32)
    return grid_s.astype(np.float64) * 1000.0


def to_observation(
    track: TouchTrack,
    frames: int,
    *,
    span_ms: Optional[float] = None,
):
    """Re-apply the hold, so an injected track can be scored like a recorded one.

    This is the inverse of :func:`from_observation` and is what feeds the frozen
    detectors: zero-order hold onto the event grid, never interpolation.
    """

    np = _numpy()
    out = np.zeros((frames, 9), dtype=np.float32)
    grid = detector_grid_ms(frames, span_ms)
    idx = 0
    last: Optional[TouchPoint] = None
    prev_xy: Optional[tuple[float, float]] = None
    for frame in range(frames):
        t = grid[frame]
        # Both clocks originate in float32 elapsed-time columns and are widened
        # separately.  A nominal 270 ms point can therefore compare as
        # 270.000011 against 269.999981 ms.  Treat sub-microsecond widening
        # residuals as the same grid instant; otherwise a real update is held
        # back by one full 10 ms detector frame.
        while idx < len(track.points) and track.points[idx].t_ms <= t + 1e-3:
            last = track.points[idx]
            idx += 1
        lifted = last is not None and last.action in (UP, POINTER_UP) and last.t_ms < t
        if last is None or lifted:
            out[frame] = (0, 0, 0, 0, 0, 0, 0, grid[frame] / 1000.0, 1)
            prev_xy = None
            continue
        x_rel = last.x / track.screen_w
        y_rel = last.y / track.screen_h
        dx = dy = 0.0
        if prev_xy is not None:
            dx, dy = x_rel - prev_xy[0], y_rel - prev_xy[1]
        out[frame] = (
            1.0,
            x_rel,
            y_rel,
            last.pressure,
            1.0,
            dx,
            dy,
            grid[frame] / 1000.0,
            1.0,
        )
        prev_xy = (x_rel, y_rel)
    return out


def summarise(track: TouchTrack) -> dict:
    return {
        "action": track.action,
        "points": len(track.points),
        "duration_ms": round(track.duration_ms, 3),
        "down_xy": [round(v, 2) for v in track.down_xy],
        "up_xy": [round(v, 2) for v in track.up_xy],
        "travel_px": round(track.travel_px(), 2),
        "pressure_min": round(min(p.pressure for p in track.points), 4),
        "pressure_max": round(max(p.pressure for p in track.points), 4),
        "source": track.source,
    }
