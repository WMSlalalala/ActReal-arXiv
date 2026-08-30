from __future__ import annotations

"""Compose a fake typing event's touch from the victim's own typing rhythm.

What a keystroke event looks like on the detector grid decides this module's
whole shape.  A genuine event is a run of single-pointer contacts, one per key,
and within a contact the coordinate does not move: HMOG's keyboard reports a
DOWN and an UP for each key and nothing between, so the zero-order-hold
observation holds one fixed point for the whole press.  Pressure is the constant
the S4 digitizer returns for every contact.  So the only quantities that vary
between people are *where* the key is and *how long* the finger stays, and only
the second of those belongs to the person.

That is why this module needs no per-keycode source contact.  The old chord bank
matched a request for key K to a recorded contact of key K because a contact
carried shape; here it carries none, and a hold measured while the victim typed
"e" is the same observable as one measured while they typed "t".  Hold duration
is strongly a property of the typist (32% of its variance is explained by who is
typing, against 1% by which key), which is exactly the signal a five-shot attack
is entitled to reuse.

The key positions come from the bound target, so the composed contacts land on
the keyboard the request names rather than where the victim once typed.  That
also keeps the output from degenerating into a copy of the victim's own event.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

from .keystroke_replay import (
    KeystrokeReplayError,
    TimingBounds,
    allocate_hold_flight_ms,
)


# A press that travels further than the device's touch slop stops being a press,
# so the victim's own excursions are carried only up to it.  The carrier archives
# record that slop as 24 px, and it is the same bound the tap route uses.
KEYSTROKE_PRESS_SLOP_PX = 24.0


class FiveShotKeystrokeTouchError(ValueError):
    """Raised when a requested typing event cannot be composed."""


def _draw(seed: int, index: int, count: int) -> int:
    """Pick one recorded press deterministically for this key of this event."""

    if count < 1:
        return 0
    return int(
        int.from_bytes(
            hashlib.sha256(
                f"keystroke-press|{int(seed)}|{int(index)}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        % count
    )


def _press_excursion(
    rhythm: KeystrokeRhythm,
    samples: int,
    width_px: float,
    height_px: float,
    choice: int,
) -> np.ndarray:
    """Return this press's own travel away from where it landed."""

    count = int(samples)
    zero = np.zeros((max(count, 0), 2), dtype=np.float64)
    if count < 2 or not rhythm.press_paths:
        return zero
    path = np.asarray(rhythm.press_paths[choice % len(rhythm.press_paths)],
                      dtype=np.float64)
    if len(path) < 2:
        return zero
    held = np.clip(
        np.floor(np.linspace(0.0, 1.0, count) * (len(path) - 1) + 0.5).astype(
            np.int64
        ),
        0,
        len(path) - 1,
    )
    excursion = path[held]
    excursion = excursion - excursion[0]
    dimensions = np.asarray((float(width_px), float(height_px)))
    reach = float(np.max(np.linalg.norm(excursion * dimensions, axis=1)))
    if reach > KEYSTROKE_PRESS_SLOP_PX:
        excursion = excursion * (KEYSTROKE_PRESS_SLOP_PX / reach)
    return excursion


@dataclass(frozen=True)
class KeystrokeRhythm:
    """One victim's own hold and flight durations, in milliseconds."""

    holds_ms: tuple[int, ...]
    flights_ms: tuple[int, ...]
    source_event_ids: tuple[str, ...]
    # Each recorded press's own path away from where it landed, in screen
    # fractions, one row per observed sample.  A finger on a key is not a fixed
    # point: measured over 121,182 genuine contact segments, 6.10% travel, and
    # 18.9% of genuine typing events contain at least one that does against 0.66%
    # of the composed ones.  Keeping the victim's own excursions is what removes
    # that.
    press_paths: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if not self.holds_ms:
            raise FiveShotKeystrokeTouchError("a rhythm needs at least one hold")
        if any(int(value) <= 0 for value in self.holds_ms):
            raise FiveShotKeystrokeTouchError("holds must be positive")
        if any(int(value) < 0 for value in self.flights_ms):
            raise FiveShotKeystrokeTouchError("flights must be non-negative")

    def draw(self, key_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Take the holds and flights one request needs, cyclically."""

        keys = int(key_count)
        if keys < 1:
            raise FiveShotKeystrokeTouchError("a typing event needs one key")
        holds = np.asarray(
            [self.holds_ms[index % len(self.holds_ms)] for index in range(keys)],
            dtype=np.int64,
        )
        if keys == 1:
            return holds, np.zeros(0, dtype=np.int64)
        if not self.flights_ms:
            raise FiveShotKeystrokeTouchError(
                "a multi-key request needs at least one recorded flight"
            )
        flights = np.asarray(
            [
                self.flights_ms[index % len(self.flights_ms)]
                for index in range(keys - 1)
            ],
            dtype=np.int64,
        )
        return holds, flights


@dataclass(frozen=True)
class ComposedKeystrokeTouch:
    """The composed detector trajectory plus what is needed to audit it."""

    trajectory: np.ndarray
    holds_ms: np.ndarray
    flights_ms: np.ndarray
    contact_samples: int
    contact_segments: int
    anchors_px: tuple[tuple[float, float], ...]
    # The instant each key goes down, on the event's own millisecond clock.
    # The IMU adapter is driven from this same schedule so a composed event's
    # impacts land where its contacts do.
    key_down_ms: tuple[float, ...]


def compose_keystroke_touch(
    *,
    anchors_px: Sequence[Sequence[float]],
    rhythm: KeystrokeRhythm,
    target_samples: int,
    target_duration_ms: float,
    screen_width_px: float,
    screen_height_px: float,
    pressure: float = 1.0,
    bounds: TimingBounds = TimingBounds(),
    seed: int = 0,
) -> ComposedKeystrokeTouch:
    """Place one contact per requested key and hold it for the victim's dwell."""

    anchors = np.asarray(anchors_px, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or not len(anchors):
        raise FiveShotKeystrokeTouchError("key anchors must be a non-empty Nx2 array")
    if not np.isfinite(anchors).all():
        raise FiveShotKeystrokeTouchError("key anchors must be finite")
    samples = int(target_samples)
    duration = float(target_duration_ms)
    if samples < 2 or not np.isfinite(duration) or duration <= 0.0:
        raise FiveShotKeystrokeTouchError("carrier window is not usable")
    if (
        not np.isfinite(screen_width_px)
        or not np.isfinite(screen_height_px)
        or screen_width_px <= 0.0
        or screen_height_px <= 0.0
    ):
        raise FiveShotKeystrokeTouchError("screen dimensions are not usable")
    if (
        np.any(anchors[:, 0] < 0.0)
        or np.any(anchors[:, 0] > screen_width_px)
        or np.any(anchors[:, 1] < 0.0)
        or np.any(anchors[:, 1] > screen_height_px)
    ):
        raise FiveShotKeystrokeTouchError("a requested key anchor is off screen")

    source_holds, source_flights = rhythm.draw(len(anchors))
    # The schedule sums to exactly the total it is given, so the last key is
    # released at that instant.  Rounding the carrier's fractional window up put
    planned_duration_ms = int(math.floor(duration))
    if planned_duration_ms <= 0:
        raise FiveShotKeystrokeTouchError("carrier window is shorter than a millisecond")
    try:
        holds, flights = allocate_hold_flight_ms(
            source_holds,
            source_flights,
            total_duration_ms=planned_duration_ms,
            bounds=bounds,
        )
    except KeystrokeReplayError as error:
        raise FiveShotKeystrokeTouchError(
            f"the victim's rhythm cannot fill this carrier window: {error}"
        ) from error

    grid = np.linspace(0.0, duration, samples, dtype=np.float64)
    trajectory = np.zeros((samples, 9), dtype=np.float32)
    trajectory[:, 7] = (grid / 1000.0).astype(np.float32)
    trajectory[:, 8] = 1.0

    start = 0.0
    covered = 0
    key_down_ms: list[float] = []
    for index in range(len(anchors)):
        if index:
            start += float(flights[index - 1])
        key_down_ms.append(float(start))
        stop = start + float(holds[index])
        inside = np.flatnonzero((grid >= start) & (grid < stop))
        if not len(inside):
            # A hold shorter than the grid period would vanish; keep the press
            # observable by holding the single nearest sample instead of
            # silently dropping a key the request asked for.
            inside = np.asarray(
                [int(np.argmin(np.abs(grid - start)))], dtype=np.int64
            )
        excursion = _press_excursion(
            rhythm, len(inside), screen_width_px, screen_height_px,
            _draw(seed, index, len(rhythm.press_paths)),
        )
        trajectory[inside, 0] = 1.0
        trajectory[inside, 1] = (
            anchors[index, 0] / screen_width_px + excursion[:, 0]
        ).astype(np.float32)
        trajectory[inside, 2] = (
            anchors[index, 1] / screen_height_px + excursion[:, 1]
        ).astype(np.float32)
        trajectory[inside, 3] = np.float32(pressure)
        trajectory[inside, 4] = 1.0
        covered += len(inside)
        start = stop

    contact = trajectory[:, 0] > 0.5
    deltas = np.zeros((samples, 2), dtype=np.float32)
    consecutive = contact[1:] & contact[:-1]
    differences = np.diff(trajectory[:, 1:3], axis=0)
    deltas[1:][consecutive] = differences[consecutive]
    trajectory[:, 5:7] = deltas

    segments = int(
        np.count_nonzero(contact[1:] & ~contact[:-1]) + (1 if contact[0] else 0)
    )
    return ComposedKeystrokeTouch(
        trajectory=trajectory,
        holds_ms=holds,
        flights_ms=flights,
        contact_samples=int(covered),
        contact_segments=segments,
        anchors_px=tuple((float(a), float(b)) for a, b in anchors),
        key_down_ms=tuple(key_down_ms),
    )


def rhythm_from_material(shots: Sequence[object]) -> KeystrokeRhythm:
    """Read one victim's typing rhythm out of their frozen material."""

    holds: list[int] = []
    flights: list[int] = []
    identities: list[str] = []
    paths: list[np.ndarray] = []
    for shot in shots:
        row = getattr(shot, "row", None)
        if row is None:
            raise FiveShotKeystrokeTouchError("material shot carries no row")
        try:
            down = [int(value) for value in row["key_down_ms"]]
            up = [int(value) for value in row["key_up_ms"]]
        except (KeyError, TypeError, ValueError) as error:
            raise FiveShotKeystrokeTouchError(
                "keystroke material lacks its per-key millisecond schedule"
            ) from error
        if len(down) != len(up) or not down:
            raise FiveShotKeystrokeTouchError(
                "keystroke material down/up schedules disagree"
            )
        identities.append(str(getattr(shot, "event_id", "")))
        trajectory = np.asarray(
            getattr(shot, "trajectory", np.zeros((0, 9))), dtype=np.float64
        )
        if trajectory.ndim == 2 and trajectory.shape[1] == 9 and len(trajectory):
            contact = trajectory[:, 0] > 0.5
            index = 0
            while index < len(contact):
                if not contact[index]:
                    index += 1
                    continue
                stop = index
                while stop < len(contact) and contact[stop]:
                    stop += 1
                if stop - index >= 2:
                    paths.append(
                        trajectory[index:stop, 1:3] - trajectory[index, 1:3]
                    )
                index = stop
        for start, stop in zip(down, up):
            holds.append(max(1, int(stop) - int(start)))
        for index in range(len(down) - 1):
            flights.append(max(0, int(down[index + 1]) - int(up[index])))
    if not holds:
        raise FiveShotKeystrokeTouchError("keystroke material yielded no holds")
    return KeystrokeRhythm(
        holds_ms=tuple(holds),
        flights_ms=tuple(flights),
        source_event_ids=tuple(identities),
        press_paths=tuple(paths),
    )

