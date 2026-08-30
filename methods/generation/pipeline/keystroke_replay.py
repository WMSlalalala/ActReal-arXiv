from __future__ import annotations

"""Train-only replay composition for HMOG soft-keyboard trajectories.

The HMOG keystroke archive stores a typing event as independent Android
DOWN/MOVE/UP contacts separated by periods in which no finger is on screen.
This module keeps those contacts as the atomic unit.  It never draws a line
between keys and never invents MOVE samples.  A composed event changes only
the contact timestamps (when necessary to meet the requested duration) and,
when a target key anchor is supplied, applies one common in-screen XY
translation to the complete chord.  Pressure, size, pointer metadata, Android
actions, and within-chord shape are copied from training-user contacts.

The returned :class:`AndroidKeystrokeRows` can be passed directly to
``android_touch_observation.observe_android_rows`` via
``rows.observation_kwargs()``.
"""

from dataclasses import dataclass
import heapq
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .android_touch_observation import (
    ACTION_DOWN,
    ACTION_MOVE,
    ACTION_UP,
    screen_dimensions_for_orientation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HMOG_KEYSTROKE_CORPUS = (
    REPO_ROOT
    / "licensed_input"
    / "hmog"
    / "processed_trajectories"
    / "hmog_trajectory_keystroke.npz"
)
DEFAULT_USER_SPLIT = REPO_ROOT / "data" / "splits" / "users_seed42.json"

ACTION_MASK = 0xFF
OUTPUT_SPLITS = ("train", "development", "test")


class KeystrokeReplayError(ValueError):
    pass


@dataclass(frozen=True)
class TimingBounds:
    """Hard, integer-millisecond support for the timing projection."""

    hold_min_ms: int = 53
    hold_max_ms: int = 140
    flight_min_ms: int = 66
    flight_max_ms: int = 1130

    def __post_init__(self) -> None:
        values = (
            self.hold_min_ms,
            self.hold_max_ms,
            self.flight_min_ms,
            self.flight_max_ms,
        )
        if any(int(value) != value or value < 0 for value in values):
            raise KeystrokeReplayError("timing bounds must be non-negative integers")
        if self.hold_min_ms < 1 or self.hold_min_ms > self.hold_max_ms:
            raise KeystrokeReplayError("invalid hold timing bounds")
        if self.flight_min_ms > self.flight_max_ms:
            raise KeystrokeReplayError("invalid flight timing bounds")


@dataclass(frozen=True)
class ChordPrimitive:
    """One genuine single-key Android contact copied from the raw corpus."""

    source_event_id: int
    source_event_index: int
    source_user_id: int
    source_key_index: int
    orientation_id: int
    keycode: int
    source_hold_ms: int
    source_flight_from_previous_ms: int
    t_rel_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray
    size: np.ndarray
    pointer_count: np.ndarray
    pointer_id: np.ndarray
    android_action: np.ndarray
    frame_index: np.ndarray
    frame_end: np.ndarray


@dataclass(frozen=True)
class AndroidKeystrokeRows:
    """Raw Android rows and an exact DOWN/UP schedule for one replay."""

    duration_ms: int
    orientation_id: int
    keycodes: np.ndarray
    hold_ms: np.ndarray
    flight_ms: np.ndarray
    key_down_ms: np.ndarray
    key_up_ms: np.ndarray
    source_event_ids: np.ndarray
    source_key_indices: np.ndarray
    source_keycodes: np.ndarray
    source_use_ordinals: np.ndarray
    translations_px: np.ndarray
    t_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray
    size: np.ndarray
    pointer_count: np.ndarray
    pointer_id: np.ndarray
    android_action: np.ndarray
    key_index: np.ndarray
    keycode: np.ndarray
    frame_index: np.ndarray
    frame_end: np.ndarray

    def observation_kwargs(
        self,
        *,
        detector_duration_ms: float | None = None,
    ) -> dict[str, object]:
        """Arguments shared with ``observe_android_rows``."""

        output_duration_ms = (
            float(self.duration_ms)
            if detector_duration_ms is None
            else float(detector_duration_ms)
        )
        if not np.isfinite(output_duration_ms) or output_duration_ms <= 0.0:
            raise KeystrokeReplayError(
                "detector_duration_ms must be finite and positive"
            )

        return {
            "action": "keystroke",
            "orientation_id": int(self.orientation_id),
            "t_ms": self.t_ms,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "pressure": self.pressure,
            "pointer_id": self.pointer_id,
            "android_action": self.android_action,
            "frame_end": self.frame_end,
            "key_index": self.key_index,
            "source_duration_ms": float(self.duration_ms),
            "target_duration_ms": output_duration_ms,
        }


def load_train_user_ids(split_path: str | Path = DEFAULT_USER_SPLIT) -> tuple[int, ...]:
    source = Path(split_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw = payload.get("train_users")
    if not isinstance(raw, list) or not raw:
        raise KeystrokeReplayError(f"{source}: train_users is missing or empty")
    users = tuple(int(value) for value in raw)
    if len(users) != len(set(users)):
        raise KeystrokeReplayError(f"{source}: train_users contains duplicates")
    return users


def build_genuine_event_id_lookup(
    corpus_path: str | Path = DEFAULT_HMOG_KEYSTROKE_CORPUS,
    *,
    allowed_user_ids: Iterable[int] | None = None,
) -> dict[int, int]:
    """Return ``genuine event_id -> raw archive event index``."""

    source = Path(corpus_path).resolve()
    with np.load(source, allow_pickle=False) as archive:
        if "event_id" not in archive or "user_id" not in archive:
            raise KeystrokeReplayError(f"{source}: event_id/user_id is missing")
        event_ids = np.asarray(archive["event_id"], dtype=np.int64)
        user_ids = np.asarray(archive["user_id"], dtype=np.int64)
    if event_ids.ndim != 1 or user_ids.shape != event_ids.shape:
        raise KeystrokeReplayError(f"{source}: malformed event_id/user_id arrays")
    selected = np.ones(len(event_ids), dtype=bool)
    if allowed_user_ids is not None:
        allowed = np.asarray(sorted({int(value) for value in allowed_user_ids}))
        selected = np.isin(user_ids, allowed)
    indices = np.flatnonzero(selected)
    chosen = event_ids[indices]
    if len(np.unique(chosen)) != len(chosen):
        raise KeystrokeReplayError(f"{source}: genuine event_id is not unique")
    return {int(event_ids[index]): int(index) for index in indices}


def donor_output_split(source_event_id: int, *, seed: int = 42) -> str:
    """Assign one donor event family to exactly one detector output split."""

    digest = hashlib.sha256(
        f"keystroke-donor|{int(seed)}|{int(source_event_id)}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") % 100
    if value < 70:
        return "train"
    if value < 80:
        return "development"
    return "test"


def _bounded_integer_projection(
    base: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: int,
) -> np.ndarray:
    """Project onto an integer box-constrained simplex.

    The continuous solution is ``clip(base + lambda, lower, upper)``.  A
    deterministic largest-remainder pass then obtains an exact integer sum.
    """

    values = np.asarray(base, dtype=np.float64)
    minimum = np.asarray(lower, dtype=np.int64)
    maximum = np.asarray(upper, dtype=np.int64)
    if values.ndim != 1 or minimum.shape != values.shape or maximum.shape != values.shape:
        raise KeystrokeReplayError("projection arrays must be equal one-dimensional shapes")
    if not len(values):
        if int(target_sum) != 0:
            raise KeystrokeReplayError("non-zero target for an empty timing vector")
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(values).all() or np.any(minimum > maximum):
        raise KeystrokeReplayError("invalid timing projection input")
    target = int(target_sum)
    if target < int(minimum.sum()) or target > int(maximum.sum()):
        raise KeystrokeReplayError(
            f"timing target {target} is outside "
            f"[{int(minimum.sum())}, {int(maximum.sum())}]"
        )
    values = np.clip(values, minimum, maximum)
    low = float(np.min(minimum - values)) - 1.0
    high = float(np.max(maximum - values)) + 1.0
    for _ in range(96):
        middle = (low + high) / 2.0
        total = float(np.clip(values + middle, minimum, maximum).sum())
        if total < target:
            low = middle
        else:
            high = middle
    continuous = np.clip(values + (low + high) / 2.0, minimum, maximum)
    result = np.floor(continuous + 1.0e-10).astype(np.int64)
    result = np.clip(result, minimum, maximum)
    delta = target - int(result.sum())
    if delta > 0:
        fractions = continuous - np.floor(continuous)
        order = np.lexsort((np.arange(len(result)), -fractions))
        while delta:
            changed = False
            for index in order:
                if result[index] < maximum[index]:
                    result[index] += 1
                    delta -= 1
                    changed = True
                    if not delta:
                        break
            if not changed:
                raise KeystrokeReplayError("could not close positive timing remainder")
    elif delta < 0:
        fractions = continuous - np.floor(continuous)
        order = np.lexsort((np.arange(len(result)), fractions))
        while delta:
            changed = False
            for index in order:
                if result[index] > minimum[index]:
                    result[index] -= 1
                    delta += 1
                    changed = True
                    if not delta:
                        break
            if not changed:
                raise KeystrokeReplayError("could not close negative timing remainder")
    if int(result.sum()) != target:
        raise KeystrokeReplayError("integer timing projection did not close exactly")
    return result


def allocate_hold_flight_ms(
    source_hold_ms: Sequence[int],
    source_flight_ms: Sequence[int],
    *,
    total_duration_ms: int,
    bounds: TimingBounds = TimingBounds(),
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate exact integer holds and flights without leaving support."""

    holds_base = np.asarray(source_hold_ms, dtype=np.int64)
    flights_base = np.asarray(source_flight_ms, dtype=np.int64)
    if holds_base.ndim != 1 or not len(holds_base):
        raise KeystrokeReplayError("at least one source hold is required")
    if flights_base.shape != (len(holds_base) - 1,):
        raise KeystrokeReplayError("flight count must equal key count minus one")
    if np.any(holds_base <= 0) or np.any(flights_base < 0):
        raise KeystrokeReplayError("source timing must be non-negative")
    duration = int(total_duration_ms)
    if duration != total_duration_ms or duration <= 0:
        raise KeystrokeReplayError("total_duration_ms must be a positive integer")

    hold_lower = np.full(len(holds_base), bounds.hold_min_ms, dtype=np.int64)
    hold_upper = np.full(len(holds_base), bounds.hold_max_ms, dtype=np.int64)
    flight_lower = np.full(len(flights_base), bounds.flight_min_ms, dtype=np.int64)
    flight_upper = np.full(len(flights_base), bounds.flight_max_ms, dtype=np.int64)
    minimum_total = int(hold_lower.sum() + flight_lower.sum())
    maximum_total = int(hold_upper.sum() + flight_upper.sum())
    if duration < minimum_total or duration > maximum_total:
        raise KeystrokeReplayError(
            f"duration {duration} ms is outside {len(holds_base)}-key support "
            f"[{minimum_total}, {maximum_total}] ms"
        )

    holds = np.clip(holds_base, hold_lower, hold_upper)
    flight_budget = duration - int(holds.sum())
    flight_minimum = int(flight_lower.sum())
    flight_maximum = int(flight_upper.sum())
    if flight_minimum <= flight_budget <= flight_maximum:
        flights = _bounded_integer_projection(
            flights_base, flight_lower, flight_upper, flight_budget
        )
        return holds, flights
    if flight_budget < flight_minimum:
        flights = flight_lower.copy()
    else:
        flights = flight_upper.copy()
    hold_budget = duration - int(flights.sum())
    holds = _bounded_integer_projection(
        holds_base, hold_lower, hold_upper, hold_budget
    )
    return holds, flights


def _integer_chord_time_warp(
    source_times_ms: Sequence[int] | np.ndarray,
    *,
    target_hold_ms: int,
) -> np.ndarray:
    """Warp a chord without collapsing distinct MotionEvent timestamps."""

    source = np.asarray(source_times_ms, dtype=np.float64)
    target = int(target_hold_ms)
    if (
        source.ndim != 1
        or len(source) < 2
        or not np.isfinite(source).all()
        or np.any(np.diff(source) < 0.0)
        or target != target_hold_ms
        or target <= 0
    ):
        raise KeystrokeReplayError("chord time warp input is invalid")
    relative = source - float(source[0])
    unique, inverse = np.unique(relative, return_inverse=True)
    if len(unique) < 2 or float(unique[-1]) <= 0.0:
        raise KeystrokeReplayError("chord has no positive time extent")
    interval_count = len(unique) - 1
    if target < interval_count:
        raise KeystrokeReplayError(
            f"target hold {target} ms cannot preserve {len(unique)} unique "
            "MotionEvent timestamps"
        )
    intervals = np.diff(unique)
    distributable = target - interval_count
    exact_extra = distributable * intervals / float(np.sum(intervals))
    extra = np.floor(exact_extra).astype(np.int64)
    remainder = distributable - int(np.sum(extra))
    if remainder:
        order = np.lexsort(
            (
                np.arange(interval_count, dtype=np.int64),
                -(exact_extra - extra),
            )
        )
        extra[order[:remainder]] += 1
    warped_unique = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(1 + extra, dtype=np.int64),
        )
    )
    if int(warped_unique[-1]) != target:
        raise AssertionError("integer chord time warp missed the requested endpoint")
    return warped_unique[inverse]


def _random_cyclic_positions(
    count: int,
    rng: np.random.Generator,
) -> Iterable[int]:
    """Visit every position once without allocating a full permutation."""

    size = int(count)
    if size <= 0:
        return
    start = int(rng.integers(0, size))
    if size == 1:
        yield start
        return
    step = int(rng.integers(1, size))
    while math.gcd(step, size) != 1:
        step += 1
        if step == size:
            step = 1
    for offset in range(size):
        yield (start + offset * step) % size


def _normalise_target_anchors(
    target_xy_px: Sequence[Sequence[float] | None] | np.ndarray | None,
    *,
    key_count: int,
    orientation_id: int,
) -> tuple[tuple[float, float] | None, ...]:
    """Validate optional per-key DOWN anchors without silently clipping them."""

    if target_xy_px is None:
        return (None,) * key_count
    if len(target_xy_px) != key_count:
        raise KeystrokeReplayError(
            "target_xy_px count must equal the target keycode count"
        )
    width, height = screen_dimensions_for_orientation(orientation_id)
    anchors: list[tuple[float, float] | None] = []
    for index, raw_anchor in enumerate(target_xy_px):
        if raw_anchor is None:
            anchors.append(None)
            continue
        anchor = np.asarray(raw_anchor, dtype=np.float64)
        if anchor.shape != (2,) or not np.isfinite(anchor).all():
            raise KeystrokeReplayError(
                f"target_xy_px[{index}] must be a finite XY pair or None"
            )
        x, y = float(anchor[0]), float(anchor[1])
        if x < 0.0 or x > width or y < 0.0 or y > height:
            raise KeystrokeReplayError(
                f"target_xy_px[{index}] leaves the physical screen"
            )
        anchors.append((x, y))
    return tuple(anchors)


def _translate_chord_to_anchor(
    primitive: ChordPrimitive,
    target_anchor: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Apply the closest legal common translation to a complete chord."""

    if target_anchor is None:
        return primitive.x_px, primitive.y_px, (0.0, 0.0)
    width, height = screen_dimensions_for_orientation(primitive.orientation_id)
    x64 = primitive.x_px.astype(np.float64)
    y64 = primitive.y_px.astype(np.float64)
    desired_x = float(target_anchor[0]) - float(x64[0])
    desired_y = float(target_anchor[1]) - float(y64[0])
    minimum_x = -float(np.min(x64))
    maximum_x = width - float(np.max(x64))
    minimum_y = -float(np.min(y64))
    maximum_y = height - float(np.max(y64))
    if minimum_x > maximum_x or minimum_y > maximum_y:
        raise KeystrokeReplayError("source chord has no legal common translation")
    translation_x = float(np.clip(desired_x, minimum_x, maximum_x))
    translation_y = float(np.clip(desired_y, minimum_y, maximum_y))
    translated_x = (x64 + translation_x).astype(np.float32)
    translated_y = (y64 + translation_y).astype(np.float32)
    if (
        not np.isfinite(translated_x).all()
        or not np.isfinite(translated_y).all()
        or np.any(translated_x < 0.0)
        or np.any(translated_x > width)
        or np.any(translated_y < 0.0)
        or np.any(translated_y > height)
    ):
        raise KeystrokeReplayError(
            "common chord translation unexpectedly left the physical screen"
        )
    return translated_x, translated_y, (translation_x, translation_y)


def _validate_composed_rows(rows: AndroidKeystrokeRows) -> None:
    """Fail closed on schedule, lifecycle, geometry, and key-count errors."""

    key_count = len(rows.keycodes)
    if key_count < 1:
        raise KeystrokeReplayError("composed keystroke has no keys")
    key_vectors = (
        rows.hold_ms,
        rows.key_down_ms,
        rows.key_up_ms,
        rows.source_event_ids,
        rows.source_key_indices,
        rows.source_keycodes,
        rows.source_use_ordinals,
    )
    if any(np.asarray(vector).shape != (key_count,) for vector in key_vectors):
        raise KeystrokeReplayError("composed per-key metadata has inconsistent shape")
    if np.asarray(rows.flight_ms).shape != (key_count - 1,):
        raise KeystrokeReplayError("composed flight count is inconsistent")
    if np.any(rows.source_use_ordinals < 1):
        raise KeystrokeReplayError("source contact use ordinals must be positive")
    translations = np.asarray(rows.translations_px, dtype=np.float64)
    if translations.shape != (key_count, 2) or not np.isfinite(translations).all():
        raise KeystrokeReplayError("composed translations are malformed")
    if int(rows.duration_ms) != rows.duration_ms or int(rows.duration_ms) <= 0:
        raise KeystrokeReplayError("composed duration is invalid")
    duration = int(rows.duration_ms)
    if np.any(rows.hold_ms <= 0) or np.any(rows.flight_ms < 0):
        raise KeystrokeReplayError("composed hold/flight timing is invalid")
    if int(np.sum(rows.hold_ms) + np.sum(rows.flight_ms)) != duration:
        raise KeystrokeReplayError("composed timing does not sum to the duration")
    if (
        int(rows.key_down_ms[0]) != 0
        or np.any(rows.key_up_ms != rows.key_down_ms + rows.hold_ms)
        or int(rows.key_up_ms[-1]) != duration
    ):
        raise KeystrokeReplayError("composed DOWN/UP schedule is inconsistent")
    if key_count > 1 and np.any(
        rows.key_down_ms[1:] != rows.key_up_ms[:-1] + rows.flight_ms
    ):
        raise KeystrokeReplayError("composed flight schedule is inconsistent")

    row_count = len(rows.t_ms)
    row_vectors = (
        rows.x_px,
        rows.y_px,
        rows.pressure,
        rows.size,
        rows.pointer_count,
        rows.pointer_id,
        rows.android_action,
        rows.key_index,
        rows.keycode,
        rows.frame_index,
        rows.frame_end,
    )
    if row_count < 2 or any(
        np.asarray(vector).shape != (row_count,) for vector in row_vectors
    ):
        raise KeystrokeReplayError("composed Android rows have inconsistent shape")
    if (
        int(rows.t_ms[0]) != 0
        or int(rows.t_ms[-1]) != duration
        or np.any(np.diff(rows.t_ms) < 0)
    ):
        raise KeystrokeReplayError("composed Android timestamps are inconsistent")
    if (
        not np.isfinite(rows.pressure).all()
        or np.any(rows.pressure < 0.0)
        or np.any(rows.pressure > 1.0)
    ):
        raise KeystrokeReplayError("composed pressure is invalid")
    if not np.isfinite(rows.size).all() or np.any(rows.size < 0.0):
        raise KeystrokeReplayError("composed touch size is invalid")
    width, height = screen_dimensions_for_orientation(rows.orientation_id)
    if (
        not np.isfinite(rows.x_px).all()
        or not np.isfinite(rows.y_px).all()
        or np.any(rows.x_px < 0.0)
        or np.any(rows.x_px > width)
        or np.any(rows.y_px < 0.0)
        or np.any(rows.y_px > height)
    ):
        raise KeystrokeReplayError("composed chord leaves the physical screen")
    if np.any(rows.pointer_count != 1):
        raise KeystrokeReplayError("keystroke chord must remain single-pointer")
    if np.any(np.diff(rows.frame_index) < 0) or not bool(rows.frame_end[-1]):
        raise KeystrokeReplayError("composed frame lifecycle is inconsistent")

    for index in range(key_count):
        selected = np.flatnonzero(rows.key_index == index)
        if len(selected) < 2 or np.any(np.diff(selected) != 1):
            raise KeystrokeReplayError(
                f"target key {index} is not one contiguous Android chord"
            )
        actions = rows.android_action[selected] & ACTION_MASK
        if (
            int(actions[0]) != ACTION_DOWN
            or int(actions[-1]) != ACTION_UP
            or np.any(actions[1:-1] != ACTION_MOVE)
        ):
            raise KeystrokeReplayError(
                f"target key {index} has an invalid DOWN/MOVE/UP lifecycle"
            )
        if np.any(rows.pointer_id[selected] != rows.pointer_id[selected[0]]):
            raise KeystrokeReplayError(
                f"target key {index} changes pointer identity"
            )
        if np.any(rows.keycode[selected] != rows.keycodes[index]):
            raise KeystrokeReplayError(
                f"target key {index} row keycode does not match its target"
            )
        if (
            int(rows.t_ms[selected[0]]) != int(rows.key_down_ms[index])
            or int(rows.t_ms[selected[-1]]) != int(rows.key_up_ms[index])
        ):
            raise KeystrokeReplayError(
                f"target key {index} row times do not match its schedule"
            )
    if not np.array_equal(np.unique(rows.key_index), np.arange(key_count)):
        raise KeystrokeReplayError("composed key count does not match the target")


class HmogKeystrokeChordBank:
    """Lazy per-keycode index over genuine train-user HMOG contacts."""

    _REQUIRED_ARRAYS = (
        "event_id",
        "user_id",
        "event_key_offsets",
        "keycode",
        "key_orientation_id",
        "key_flight_from_previous_ms",
        "key_touch_found",
        "key_touch_offsets",
        "flat_t_rel_ms",
        "flat_x",
        "flat_y",
        "flat_pressure",
        "flat_size",
        "flat_pointer_count",
        "flat_pointer_id",
        "flat_action_code",
        "flat_frame_index",
    )

    def __init__(
        self,
        *,
        source: Path,
        train_user_ids: tuple[int, ...],
        arrays: Mapping[str, np.ndarray],
        candidate_key_indices: np.ndarray,
        maximum_movement_px: float,
    ) -> None:
        self.source = source
        self.train_user_ids = train_user_ids
        self._train_user_set = frozenset(train_user_ids)
        self._arrays = dict(arrays)
        self.maximum_movement_px = float(maximum_movement_px)
        if not np.isfinite(self.maximum_movement_px) or self.maximum_movement_px <= 0.0:
            raise KeystrokeReplayError("maximum_movement_px must be positive")
        event_offsets = self._arrays["event_key_offsets"]
        all_key_indices = np.arange(len(self._arrays["keycode"]), dtype=np.int64)
        self._key_event_index = np.searchsorted(
            event_offsets[1:], all_key_indices, side="right"
        ).astype(np.int64)
        grouped: dict[tuple[int, int], list[int]] = {}
        grouped_by_orientation: dict[int, list[int]] = {}
        for key_index in np.asarray(candidate_key_indices, dtype=np.int64):
            orientation = int(self._arrays["key_orientation_id"][key_index])
            keycode = int(self._arrays["keycode"][key_index])
            grouped.setdefault((orientation, keycode), []).append(int(key_index))
            grouped_by_orientation.setdefault(orientation, []).append(int(key_index))
        self._by_orientation_keycode = {
            key: np.asarray(values, dtype=np.int64) for key, values in grouped.items()
        }
        self._by_orientation = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in grouped_by_orientation.items()
        }
        self._split_pool_cache: dict[
            tuple[str, int],
            tuple[
                dict[int, tuple[int, ...]],
                dict[tuple[int, int], tuple[int, ...]],
                dict[int, int],
            ],
        ] = {}
        self._invalid_source_key_indices: set[int] = set()

    @classmethod
    def from_hmog(
        cls,
        corpus_path: str | Path = DEFAULT_HMOG_KEYSTROKE_CORPUS,
        *,
        split_path: str | Path = DEFAULT_USER_SPLIT,
        source_registry_path: str | Path | None = None,
        maximum_movement_px: float = 24.0,
        allowed_source_event_ids: Iterable[int] | None = None,
        excluded_source_event_ids: Iterable[int] = (),
    ) -> "HmogKeystrokeChordBank":
        source = Path(corpus_path).resolve()
        train_users = load_train_user_ids(split_path)
        with np.load(source, allow_pickle=False) as archive:
            missing = sorted(set(cls._REQUIRED_ARRAYS) - set(archive.files))
            if missing:
                raise KeystrokeReplayError(f"{source}: missing arrays {missing}")
            arrays = {
                name: np.asarray(archive[name]).copy() for name in cls._REQUIRED_ARRAYS
            }
        event_ids = arrays["event_id"]
        event_users = arrays["user_id"]
        event_offsets = arrays["event_key_offsets"]
        key_count = len(arrays["keycode"])
        touch_offsets = arrays["key_touch_offsets"]
        if (
            event_ids.ndim != 1
            or event_users.shape != event_ids.shape
            or event_offsets.shape != (len(event_ids) + 1,)
            or int(event_offsets[0]) != 0
            or int(event_offsets[-1]) != key_count
            or touch_offsets.shape != (key_count + 1,)
            or int(touch_offsets[0]) != 0
            or int(touch_offsets[-1]) != len(arrays["flat_t_rel_ms"])
        ):
            raise KeystrokeReplayError(f"{source}: malformed ragged offsets")
        for name in (
            "key_orientation_id",
            "key_flight_from_previous_ms",
            "key_touch_found",
        ):
            if arrays[name].shape != (key_count,):
                raise KeystrokeReplayError(f"{source}: malformed {name}")
        flat_count = len(arrays["flat_t_rel_ms"])
        for name in (
            "flat_x",
            "flat_y",
            "flat_pressure",
            "flat_size",
            "flat_pointer_count",
            "flat_pointer_id",
            "flat_action_code",
            "flat_frame_index",
        ):
            if arrays[name].shape != (flat_count,):
                raise KeystrokeReplayError(f"{source}: malformed {name}")

        all_key_indices = np.arange(key_count, dtype=np.int64)
        key_event_index = np.searchsorted(
            event_offsets[1:], all_key_indices, side="right"
        )
        train_mask = np.isin(event_users[key_event_index], np.asarray(train_users))
        allowed_events = (
            None
            if allowed_source_event_ids is None
            else np.asarray(
                sorted({int(value) for value in allowed_source_event_ids}),
                dtype=np.int64,
            )
        )
        if allowed_events is not None and not len(allowed_events):
            raise KeystrokeReplayError("allowed_source_event_ids cannot be empty")
        allowed_mask = (
            np.ones(key_count, dtype=bool)
            if allowed_events is None
            else np.isin(event_ids[key_event_index], allowed_events)
        )
        excluded_events = np.asarray(
            sorted({int(value) for value in excluded_source_event_ids}),
            dtype=np.int64,
        )
        exclusion_mask = (
            np.ones(key_count, dtype=bool)
            if not len(excluded_events)
            else ~np.isin(event_ids[key_event_index], excluded_events)
        )
        found_mask = arrays["key_touch_found"].astype(bool)
        candidates = all_key_indices[
            train_mask & allowed_mask & found_mask & exclusion_mask
        ]
        if source_registry_path is not None:
            registry = Path(source_registry_path).resolve()
            with np.load(registry, allow_pickle=False) as source_registry:
                if "geometry_source_key_index" not in source_registry:
                    raise KeystrokeReplayError(
                        f"{registry}: geometry_source_key_index is missing"
                    )
                registered = np.asarray(
                    source_registry["geometry_source_key_index"], dtype=np.int64
                )
            if np.any(registered < 0) or np.any(registered >= key_count):
                raise KeystrokeReplayError(f"{registry}: key index is out of range")
            candidates = np.intersect1d(candidates, registered, assume_unique=False)
        candidates = np.unique(candidates)
        if not len(candidates):
            raise KeystrokeReplayError(f"{source}: no train-user key contacts")
        return cls(
            source=source,
            train_user_ids=train_users,
            arrays=arrays,
            candidate_key_indices=candidates,
            maximum_movement_px=maximum_movement_px,
        )

    @property
    def source_key_count(self) -> int:
        return int(sum(len(values) for values in self._by_orientation_keycode.values()))

    @property
    def supported_orientation_keycodes(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._by_orientation_keycode))

    def _split_candidate_pools(
        self,
        *,
        output_split: str,
        split_seed: int,
    ) -> tuple[
        dict[int, tuple[int, ...]],
        dict[tuple[int, int], tuple[int, ...]],
        dict[int, int],
    ]:
        """Return split-eligible contacts in one stable hashed tie-break order."""

        if output_split not in OUTPUT_SPLITS:
            raise KeystrokeReplayError(f"unknown output split {output_split!r}")
        cache_key = (output_split, int(split_seed))
        cached = self._split_pool_cache.get(cache_key)
        if cached is not None:
            return cached

        eligible: list[int] = []
        ranks: dict[int, int] = {}
        for orientation_indices in self._by_orientation.values():
            for raw_index in orientation_indices:
                source_index = int(raw_index)
                event_index = int(self._key_event_index[source_index])
                source_event_id = int(self._arrays["event_id"][event_index])
                if donor_output_split(source_event_id, seed=split_seed) != output_split:
                    continue
                digest = hashlib.sha256(
                    (
                        f"keystroke-contact|{int(split_seed)}|"
                        f"{source_event_id}|{source_index}"
                    ).encode("utf-8")
                ).digest()
                ranks[source_index] = int.from_bytes(digest[:8], "big")
                eligible.append(source_index)
        eligible.sort(key=lambda index: (ranks[index], index))
        by_orientation: dict[int, list[int]] = {}
        by_exact: dict[tuple[int, int], list[int]] = {}
        for source_index in eligible:
            orientation = int(self._arrays["key_orientation_id"][source_index])
            keycode = int(self._arrays["keycode"][source_index])
            by_orientation.setdefault(orientation, []).append(source_index)
            by_exact.setdefault((orientation, keycode), []).append(source_index)
        result = (
            {key: tuple(values) for key, values in by_orientation.items()},
            {key: tuple(values) for key, values in by_exact.items()},
            ranks,
        )
        self._split_pool_cache[cache_key] = result
        return result

    def _primitive(self, source_key_index: int) -> ChordPrimitive:
        arrays = self._arrays
        key_index = int(source_key_index)
        if key_index < 0 or key_index >= len(arrays["keycode"]):
            raise KeystrokeReplayError("source key index is out of range")
        event_index = int(self._key_event_index[key_index])
        user_id = int(arrays["user_id"][event_index])
        if user_id not in self._train_user_set:
            raise KeystrokeReplayError("non-training contact entered the chord bank")
        left = int(arrays["key_touch_offsets"][key_index])
        right = int(arrays["key_touch_offsets"][key_index + 1])
        if right - left < 2:
            raise KeystrokeReplayError("key contact has fewer than DOWN/UP rows")
        times = np.asarray(arrays["flat_t_rel_ms"][left:right], dtype=np.int64)
        times = times - int(times[0])
        if np.any(np.diff(times) < 0) or int(times[-1]) <= 0:
            raise KeystrokeReplayError("key contact has invalid timestamps")
        actions = np.asarray(arrays["flat_action_code"][left:right], dtype=np.int16)
        masked_actions = actions & ACTION_MASK
        if int(masked_actions[0]) != ACTION_DOWN or int(masked_actions[-1]) != ACTION_UP:
            raise KeystrokeReplayError("key contact is not a complete DOWN-to-UP chord")
        if np.any(masked_actions[1:-1] != ACTION_MOVE):
            raise KeystrokeReplayError(
                "key contact is not one strict DOWN/MOVE/UP lifecycle"
            )
        x = np.asarray(arrays["flat_x"][left:right], dtype=np.float32)
        y = np.asarray(arrays["flat_y"][left:right], dtype=np.float32)
        pressure = np.asarray(arrays["flat_pressure"][left:right], dtype=np.float32)
        size = np.asarray(arrays["flat_size"][left:right], dtype=np.float32)
        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(pressure).all()
            or not np.isfinite(size).all()
            or np.any(pressure < 0.0)
            or np.any(pressure > 1.0)
            or np.any(size < 0.0)
        ):
            raise KeystrokeReplayError("key contact contains invalid physical values")
        movement = np.hypot(x - x[0], y - y[0])
        if float(np.max(movement)) > self.maximum_movement_px + 1.0e-6:
            raise KeystrokeReplayError("key contact exceeds the movement safety bound")
        orientation = int(arrays["key_orientation_id"][key_index])
        width, height = screen_dimensions_for_orientation(orientation)
        if np.any(x < 0.0) or np.any(x > width) or np.any(y < 0.0) or np.any(y > height):
            raise KeystrokeReplayError("key contact leaves the physical screen")
        source_frames = np.asarray(
            arrays["flat_frame_index"][left:right], dtype=np.int64
        )
        frame_index = source_frames - int(source_frames[0])
        if np.any(np.diff(frame_index) < 0):
            raise KeystrokeReplayError("key contact frame indices are not monotonic")
        pointer_count = np.asarray(
            arrays["flat_pointer_count"][left:right], dtype=np.int8
        )
        pointer_id = np.asarray(
            arrays["flat_pointer_id"][left:right], dtype=np.int16
        )
        if np.any(pointer_count != 1) or np.any(pointer_id != pointer_id[0]):
            raise KeystrokeReplayError(
                "key contact is not one stable single-pointer lifecycle"
            )
        frame_end = np.ones(len(frame_index), dtype=np.uint8)
        if len(frame_index) > 1:
            frame_end[:-1] = (frame_index[1:] != frame_index[:-1]).astype(np.uint8)
        return ChordPrimitive(
            source_event_id=int(arrays["event_id"][event_index]),
            source_event_index=event_index,
            source_user_id=user_id,
            source_key_index=key_index,
            orientation_id=orientation,
            keycode=int(arrays["keycode"][key_index]),
            source_hold_ms=int(times[-1]),
            source_flight_from_previous_ms=max(
                0, int(arrays["key_flight_from_previous_ms"][key_index])
            ),
            t_rel_ms=times,
            x_px=x,
            y_px=y,
            pressure=pressure,
            size=size,
            pointer_count=pointer_count,
            pointer_id=pointer_id,
            android_action=actions,
            frame_index=frame_index,
            frame_end=frame_end,
        )

    def sample_primitive(
        self,
        *,
        keycode: int,
        orientation_id: int,
        rng: np.random.Generator,
        excluded_source_key_indices: set[int] | None = None,
        output_split: str | None = None,
        split_seed: int = 42,
    ) -> ChordPrimitive:
        orientation = int(orientation_id)
        target_keycode = int(keycode)
        orientation_candidates = self._by_orientation.get(orientation)
        if orientation_candidates is None or not len(orientation_candidates):
            raise KeystrokeReplayError(
                f"no train chord for orientation={orientation_id}"
            )
        excluded = excluded_source_key_indices or set()
        if output_split is not None and output_split not in OUTPUT_SPLITS:
            raise KeystrokeReplayError(f"unknown output split {output_split!r}")
        exact_candidates = self._by_orientation_keycode.get(
            (orientation, target_keycode)
        )
        candidate_groups = []
        if exact_candidates is not None and len(exact_candidates):
            candidate_groups.append(exact_candidates)
        candidate_groups.append(orientation_candidates)
        errors: list[str] = []
        tried: set[int] = set()
        for candidates in candidate_groups:
            for position in _random_cyclic_positions(len(candidates), rng):
                source_index = int(candidates[position])
                if source_index in tried:
                    continue
                tried.add(source_index)
                if source_index in excluded:
                    continue
                event_index = int(self._key_event_index[source_index])
                source_event_id = int(self._arrays["event_id"][event_index])
                if (
                    output_split is not None
                    and donor_output_split(source_event_id, seed=split_seed)
                    != output_split
                ):
                    continue
                try:
                    return self._primitive(source_index)
                except KeystrokeReplayError as exc:
                    errors.append(str(exc))
        detail = errors[-1] if errors else "all candidates were excluded"
        raise KeystrokeReplayError(
            f"no valid train chord for orientation={orientation_id}, "
            f"target_keycode={keycode}, including same-orientation fallback: {detail}"
        )

    def compose(
        self,
        *,
        keycodes: Sequence[int],
        total_duration_ms: int,
        orientation_id: int,
        seed: int,
        bounds: TimingBounds = TimingBounds(),
        output_split: str | None = None,
        split_seed: int = 42,
        excluded_source_key_indices: set[int] | None = None,
        target_xy_px: Sequence[Sequence[float] | None] | np.ndarray | None = None,
        _usage_state: _KeystrokeContactUsage | None = None,
    ) -> AndroidKeystrokeRows:
        target_keycodes = np.asarray(keycodes, dtype=np.int32)
        if target_keycodes.ndim != 1 or not len(target_keycodes):
            raise KeystrokeReplayError("keycodes must contain at least one key")
        orientation = int(orientation_id)
        anchors = _normalise_target_anchors(
            target_xy_px,
            key_count=len(target_keycodes),
            orientation_id=orientation,
        )
        rng = np.random.default_rng(int(seed))
        primitives: list[ChordPrimitive] = []
        source_use_ordinals: list[int] = []
        if _usage_state is not None and excluded_source_key_indices is not None:
            raise KeystrokeReplayError(
                "usage state and excluded_source_key_indices are mutually exclusive"
            )
        globally_used = excluded_source_key_indices or set()
        used = set(globally_used)
        selected_for_event: set[int] = set()
        for target_keycode in target_keycodes:
            if _usage_state is None:
                primitive = self.sample_primitive(
                    keycode=int(target_keycode),
                    orientation_id=orientation,
                    rng=rng,
                    excluded_source_key_indices=used,
                    output_split=output_split,
                    split_seed=split_seed,
                )
                source_use_ordinal = 1
            else:
                primitive, source_use_ordinal = _usage_state.sample_primitive(
                    keycode=int(target_keycode),
                    orientation_id=orientation,
                    selected_for_event=selected_for_event,
                )
            primitives.append(primitive)
            source_use_ordinals.append(source_use_ordinal)
            used.add(primitive.source_key_index)
            selected_for_event.add(primitive.source_key_index)
        source_holds = [primitive.source_hold_ms for primitive in primitives]
        source_flights = [
            primitive.source_flight_from_previous_ms for primitive in primitives[1:]
        ]
        holds, flights = allocate_hold_flight_ms(
            source_holds,
            source_flights,
            total_duration_ms=total_duration_ms,
            bounds=bounds,
        )

        key_down = np.zeros(len(primitives), dtype=np.int64)
        key_up = np.zeros(len(primitives), dtype=np.int64)
        row_parts: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "t_ms",
                "x_px",
                "y_px",
                "pressure",
                "size",
                "pointer_count",
                "pointer_id",
                "android_action",
                "key_index",
                "keycode",
                "frame_index",
                "frame_end",
            )
        }
        translations: list[tuple[float, float]] = []
        cursor = 0
        next_frame_index = 0
        for index, (primitive, hold) in enumerate(zip(primitives, holds)):
            key_down[index] = cursor
            source_times = primitive.t_rel_ms.astype(np.float64)
            warped = _integer_chord_time_warp(
                source_times,
                target_hold_ms=int(hold),
            )
            if np.any(np.diff(warped) < 0):
                raise KeystrokeReplayError("time warp broke chord ordering")
            row_parts["t_ms"].append(warped + cursor)
            translated_x, translated_y, translation = _translate_chord_to_anchor(
                primitive,
                anchors[index],
            )
            translations.append(translation)
            row_parts["x_px"].append(translated_x)
            row_parts["y_px"].append(translated_y)
            row_parts["pressure"].append(primitive.pressure)
            row_parts["size"].append(primitive.size)
            row_parts["pointer_count"].append(primitive.pointer_count)
            row_parts["pointer_id"].append(primitive.pointer_id)
            row_parts["android_action"].append(primitive.android_action)
            row_parts["key_index"].append(
                np.full(len(warped), index, dtype=np.int16)
            )
            row_parts["keycode"].append(
                np.full(len(warped), int(target_keycodes[index]), dtype=np.int32)
            )
            local_frames = primitive.frame_index
            row_parts["frame_index"].append(local_frames + next_frame_index)
            row_parts["frame_end"].append(primitive.frame_end)
            next_frame_index += int(local_frames[-1]) + 1
            cursor += int(hold)
            key_up[index] = cursor
            if index < len(flights):
                cursor += int(flights[index])
        duration = int(total_duration_ms)
        if cursor != duration or int(key_up[-1]) != duration:
            raise KeystrokeReplayError("composed schedule did not end exactly")

        dtypes: dict[str, np.dtype] = {
            "t_ms": np.dtype(np.int64),
            "x_px": np.dtype(np.float32),
            "y_px": np.dtype(np.float32),
            "pressure": np.dtype(np.float32),
            "size": np.dtype(np.float32),
            "pointer_count": np.dtype(np.int8),
            "pointer_id": np.dtype(np.int16),
            "android_action": np.dtype(np.int16),
            "key_index": np.dtype(np.int16),
            "keycode": np.dtype(np.int32),
            "frame_index": np.dtype(np.int64),
            "frame_end": np.dtype(np.uint8),
        }
        rows = {
            name: np.concatenate(parts).astype(dtypes[name], copy=False)
            for name, parts in row_parts.items()
        }
        result = AndroidKeystrokeRows(
            duration_ms=duration,
            orientation_id=orientation,
            keycodes=target_keycodes,
            hold_ms=holds,
            flight_ms=flights,
            key_down_ms=key_down,
            key_up_ms=key_up,
            source_event_ids=np.asarray(
                [primitive.source_event_id for primitive in primitives],
                dtype=np.int64,
            ),
            source_key_indices=np.asarray(
                [primitive.source_key_index for primitive in primitives],
                dtype=np.int64,
            ),
            source_keycodes=np.asarray(
                [primitive.keycode for primitive in primitives],
                dtype=np.int32,
            ),
            source_use_ordinals=np.asarray(source_use_ordinals, dtype=np.int32),
            translations_px=np.asarray(translations, dtype=np.float64),
            t_ms=rows["t_ms"],
            x_px=rows["x_px"],
            y_px=rows["y_px"],
            pressure=rows["pressure"],
            size=rows["size"],
            pointer_count=rows["pointer_count"],
            pointer_id=rows["pointer_id"],
            android_action=rows["android_action"],
            key_index=rows["key_index"],
            keycode=rows["keycode"],
            frame_index=rows["frame_index"],
            frame_end=rows["frame_end"],
        )
        _validate_composed_rows(result)
        # Commit only after selection, timing projection, and row construction
        # all succeed.  A failed composition must not consume donor contacts.
        if _usage_state is not None:
            _usage_state.commit(selected_for_event)
        elif excluded_source_key_indices is not None:
            excluded_source_key_indices.update(selected_for_event)
        return result


class _KeystrokeContactUsage:
    """Balanced, split-confined contact scheduler used by one allocator."""

    def __init__(
        self,
        bank: HmogKeystrokeChordBank,
        *,
        output_split: str,
        split_seed: int,
        maximum_uses_per_contact: int,
    ) -> None:
        maximum = int(maximum_uses_per_contact)
        if maximum != maximum_uses_per_contact or maximum < 1:
            raise KeystrokeReplayError(
                "maximum_uses_per_contact must be a positive integer"
            )
        self.bank = bank
        self.output_split = output_split
        self.split_seed = int(split_seed)
        self.maximum_uses_per_contact = maximum
        orientation_pools, exact_pools, ranks = bank._split_candidate_pools(
            output_split=output_split,
            split_seed=split_seed,
        )
        self._ranks = ranks
        self._orientation_heaps: dict[int, list[tuple[int, int, int]]] = {
            key: [(0, ranks[index], index) for index in pool]
            for key, pool in orientation_pools.items()
        }
        self._exact_heaps: dict[
            tuple[int, int], list[tuple[int, int, int]]
        ] = {
            key: [(0, ranks[index], index) for index in pool]
            for key, pool in exact_pools.items()
        }
        for heap in self._orientation_heaps.values():
            heapq.heapify(heap)
        for heap in self._exact_heaps.values():
            heapq.heapify(heap)
        self._use_counts: dict[int, int] = {}

    @property
    def used_source_key_indices(self) -> frozenset[int]:
        return frozenset(self._use_counts)

    @property
    def source_key_use_counts(self) -> dict[int, int]:
        return dict(self._use_counts)

    def _sample_heap(
        self,
        heap: list[tuple[int, int, int]] | None,
        *,
        selected_for_event: set[int],
    ) -> tuple[ChordPrimitive, int] | None:
        if heap is None:
            return None
        temporarily_removed: list[tuple[int, int, int]] = []
        try:
            while heap:
                recorded_use, rank, source_index = heapq.heappop(heap)
                current_use = self._use_counts.get(source_index, 0)
                if (
                    source_index in self.bank._invalid_source_key_indices
                    or current_use >= self.maximum_uses_per_contact
                ):
                    continue
                if recorded_use != current_use:
                    heapq.heappush(heap, (current_use, rank, source_index))
                    continue
                if source_index in selected_for_event:
                    temporarily_removed.append(
                        (recorded_use, rank, source_index)
                    )
                    continue
                try:
                    primitive = self.bank._primitive(source_index)
                except KeystrokeReplayError:
                    self.bank._invalid_source_key_indices.add(source_index)
                    continue
                # Selection is provisional.  Keep the current entry available
                # until the complete event passes validation and commit updates
                # the authoritative use count.
                heapq.heappush(heap, (recorded_use, rank, source_index))
                return primitive, current_use + 1
            return None
        finally:
            for entry in temporarily_removed:
                heapq.heappush(heap, entry)

    def sample_primitive(
        self,
        *,
        keycode: int,
        orientation_id: int,
        selected_for_event: set[int],
    ) -> tuple[ChordPrimitive, int]:
        orientation = int(orientation_id)
        target_keycode = int(keycode)
        exact = self._sample_heap(
            self._exact_heaps.get((orientation, target_keycode)),
            selected_for_event=selected_for_event,
        )
        fallback = self._sample_heap(
            self._orientation_heaps.get(orientation),
            selected_for_event=selected_for_event,
        )
        # Balance use counts across the complete orientation pool.  Exact
        # keycode is a preference only among candidates at the same minimum
        # use ordinal; an unused fallback must precede a second use of an exact
        # chord.
        if exact is not None and (
            fallback is None or exact[1] <= fallback[1]
        ):
            return exact
        if fallback is not None:
            return fallback
        raise KeystrokeReplayError(
            f"no split-confined chord below use cap "
            f"{self.maximum_uses_per_contact} for orientation={orientation}, "
            f"target_keycode={target_keycode}"
        )

    def commit(self, source_key_indices: Iterable[int]) -> None:
        chosen = tuple(int(index) for index in source_key_indices)
        if len(chosen) != len(set(chosen)):
            raise KeystrokeReplayError(
                "one source contact cannot appear twice in one composed event"
            )
        pending: dict[int, int] = {}
        for source_index in chosen:
            current = self._use_counts.get(source_index, 0)
            if current >= self.maximum_uses_per_contact:
                raise KeystrokeReplayError(
                    f"source contact {source_index} reached its use cap"
                )
            pending[source_index] = current + 1
        # Commit atomically only after every selected contact was checked.
        self._use_counts.update(pending)


class KeystrokeReplayAllocator:
    """Stateful primitive-level allocator for one detector output split."""

    def __init__(
        self,
        bank: HmogKeystrokeChordBank,
        *,
        output_split: str,
        split_seed: int = 42,
        maximum_uses_per_contact: int = 1,
    ) -> None:
        if output_split not in OUTPUT_SPLITS:
            raise KeystrokeReplayError(f"unknown output split {output_split!r}")
        self.bank = bank
        self.output_split = output_split
        self.split_seed = int(split_seed)
        self.maximum_uses_per_contact = int(maximum_uses_per_contact)
        self._usage_state = _KeystrokeContactUsage(
            bank,
            output_split=output_split,
            split_seed=split_seed,
            maximum_uses_per_contact=maximum_uses_per_contact,
        )

    @property
    def used_source_key_indices(self) -> frozenset[int]:
        return self._usage_state.used_source_key_indices

    @property
    def source_key_use_counts(self) -> dict[int, int]:
        return self._usage_state.source_key_use_counts

    def compose(
        self,
        *,
        keycodes: Sequence[int],
        total_duration_ms: int,
        orientation_id: int,
        seed: int,
        bounds: TimingBounds = TimingBounds(),
        target_xy_px: Sequence[Sequence[float] | None] | np.ndarray | None = None,
    ) -> AndroidKeystrokeRows:
        return self.bank.compose(
            keycodes=keycodes,
            total_duration_ms=total_duration_ms,
            orientation_id=orientation_id,
            seed=seed,
            bounds=bounds,
            target_xy_px=target_xy_px,
            output_split=self.output_split,
            split_seed=self.split_seed,
            _usage_state=self._usage_state,
        )


__all__ = [
    "AndroidKeystrokeRows",
    "ChordPrimitive",
    "DEFAULT_HMOG_KEYSTROKE_CORPUS",
    "DEFAULT_USER_SPLIT",
    "HmogKeystrokeChordBank",
    "KeystrokeReplayAllocator",
    "KeystrokeReplayError",
    "TimingBounds",
    "allocate_hold_flight_ms",
    "build_genuine_event_id_lookup",
    "donor_output_split",
    "load_train_user_ids",
]
