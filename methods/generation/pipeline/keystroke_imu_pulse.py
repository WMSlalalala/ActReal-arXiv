"""Impulse-response adapter for keystroke IMU."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .audit import sha256_array, sha256_file

SCHEMA_VERSION = "hmog-keystroke-imu-pulse-adapter-v1"
ACCEPTED_MATERIAL_SCHEMAS = (
    "hmog_fiveshot_attack_material_v2",
    "hmog_fiveshot_attack_material_v3",
)
IMU_CHANNELS = 6
PERIOD_MS = 10.0
DEFAULT_PRE_MS = 50.0
DEFAULT_POST_MS = 250.0
DEFAULT_BASELINE_CUTOFF_HZ = 3.0
DEFAULT_RESIDUAL_BLOCK_MS = 250.0


class KeystrokeImuPulseError(ValueError):
    """Raised for any malformed source event or generation request."""


@dataclass(frozen=True)
class PulseSourceEvent:
    """One real keystroke event used as adapter material."""

    event_id: str
    imu: np.ndarray
    key_down_ms: np.ndarray
    is_letter: np.ndarray
    # When present, this is the real source-sample timeline on the same clock
    # as key_down_ms.  Native HMOG material uses absolute Android elapsed-time
    # milliseconds.  None preserves the regular 100 Hz fallback contract.
    imu_t_ms: np.ndarray | None = None


@dataclass(frozen=True)
class KeystrokeImuPulseModel:
    """Per-user impact templates, posture baselines and a residual pool."""

    schema_version: str
    user_id: str
    pre_samples: int
    post_samples: int
    letter_template: np.ndarray
    other_template: np.ndarray
    letter_observations: int
    other_observations: int
    baselines: tuple[np.ndarray, ...]
    residual_blocks: tuple[np.ndarray, ...]
    source_event_ids: tuple[str, ...]
    baseline_window_samples: int
    explicit_timeline_sources: int

    @property
    def template_samples(self) -> int:
        return self.pre_samples + self.post_samples


@dataclass(frozen=True)
class GeneratedKeystrokeImu:
    """A complete synthesised event plus the accounting needed to audit it."""

    imu: np.ndarray
    provenance: dict[str, object]


def _as_imu(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != IMU_CHANNELS:
        raise KeystrokeImuPulseError(f"{name} must have shape [samples, 6]")
    if len(array) < 2:
        raise KeystrokeImuPulseError(f"{name} needs at least two samples")
    if not np.isfinite(array).all():
        raise KeystrokeImuPulseError(f"{name} is not finite")
    return array


def _as_schedule(
    key_down_ms: Iterable[object],
    is_letter: Iterable[object],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    downs = np.asarray(list(key_down_ms), dtype=np.float64).reshape(-1)
    letters = np.asarray(list(is_letter), dtype=np.bool_).reshape(-1)
    if downs.shape != letters.shape:
        raise KeystrokeImuPulseError(f"{name} key times and letter flags differ")
    if not len(downs):
        raise KeystrokeImuPulseError(f"{name} has no key press")
    if not np.isfinite(downs).all() or np.any(downs < 0.0):
        raise KeystrokeImuPulseError(f"{name} key times must be finite and >= 0")
    if np.any(np.diff(downs) < 0.0):
        raise KeystrokeImuPulseError(f"{name} key times must be sorted")
    return downs, letters


def _as_source_timeline(
    value: object | None, *, samples: int, name: str
) -> np.ndarray:
    if value is None:
        return np.arange(samples, dtype=np.float64) * PERIOD_MS
    timeline = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(timeline) != samples:
        raise KeystrokeImuPulseError(
            f"{name} timeline length differs from its IMU"
        )
    if not np.isfinite(timeline).all() or np.any(np.diff(timeline) <= 0.0):
        raise KeystrokeImuPulseError(
            f"{name} timeline must be finite and strictly increasing"
        )
    return timeline


def _nearest_timeline_index(timeline: np.ndarray, time_ms: float) -> int:
    """Return the nearest real sample without assuming 10 ms per index."""

    right = int(np.searchsorted(timeline, float(time_ms), side="left"))
    if right <= 0:
        return 0
    if right >= len(timeline):
        return len(timeline) - 1
    left = right - 1
    return (
        left
        if abs(float(timeline[left]) - float(time_ms))
        <= abs(float(timeline[right]) - float(time_ms))
        else right
    )


def _samples_for_duration(duration_ms: float) -> int:
    duration = float(duration_ms)
    if not np.isfinite(duration) or duration <= 0.0:
        raise KeystrokeImuPulseError("duration must be finite and positive")
    samples = int(round(duration / PERIOD_MS))
    if samples < 2:
        raise KeystrokeImuPulseError("duration must span at least two samples")
    return samples


def _baseline(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded, applied twice for a flat response."""

    if window < 3:
        return signal.copy()
    smoothed = signal
    for _ in range(2):
        pad = window // 2
        padded = np.pad(smoothed, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.stack(
            [
                np.convolve(padded[:, channel], kernel, mode="valid")[
                    : len(signal)
                ]
                for channel in range(signal.shape[1])
            ],
            axis=1,
        )
    return smoothed


def _time_scale(signal: np.ndarray, target: int) -> np.ndarray:
    """Resample a slow signal onto a different length by linear interpolation."""

    if len(signal) == target:
        return signal.copy()
    source = np.linspace(0.0, 1.0, len(signal))
    destination = np.linspace(0.0, 1.0, target)
    return np.stack(
        [
            np.interp(destination, source, signal[:, channel])
            for channel in range(signal.shape[1])
        ],
        axis=1,
    )


def _place(
    canvas: np.ndarray, template: np.ndarray, start: int
) -> tuple[int, int]:
    """Add a template at ``start`` and report the clipped sample span."""

    end = start + len(template)
    lo = max(0, start)
    hi = min(len(canvas), end)
    if hi <= lo:
        return 0, 0
    canvas[lo:hi] += template[lo - start : hi - start]
    return lo, hi


def _baseline_window(
    baselines: Sequence[np.ndarray], samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, str]:
    """Take the victim's posture drift as recorded, not stretched to fit."""

    lengths = [len(np.asarray(baseline)) for baseline in baselines]
    usable = [index for index, length in enumerate(lengths) if length >= samples]
    if usable:
        # The shortest recording that still covers the window, so the least of
        # the victim's material is thrown away.
        choice = min(usable, key=lambda index: (lengths[index], index))
        band = np.asarray(baselines[choice], dtype=np.float64)
        start = int(rng.integers(len(band) - samples + 1))
        return band[start : start + samples], choice, "recorded_window"
    choice = int(np.argmax(lengths))
    return (
        _time_scale(np.asarray(baselines[choice], dtype=np.float64), samples),
        choice,
        "time_scaled_fallback",
    )


def _quiet_reservoir(runs: Sequence[np.ndarray]) -> np.ndarray:
    """Chain the victim's quiet stretches into one band, smoothest join first."""

    pieces = [np.asarray(run, dtype=np.float64) for run in runs]
    if len(pieces) == 1:
        return pieces[0]
    order = [0]
    remaining = set(range(1, len(pieces)))
    while remaining:
        tail = pieces[order[-1]][-1]
        nearest = min(
            remaining,
            key=lambda index: (
                float(np.linalg.norm(pieces[index][0] - tail)), index
            ),
        )
        order.append(nearest)
        remaining.discard(nearest)
    return np.concatenate([pieces[index] for index in order], axis=0)


def _joined_residual(
    runs: Sequence[np.ndarray], samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Read the quiet band contiguously from a seeded phase, wrapping if short."""

    band = _quiet_reservoir(runs)
    if len(band) < 2:
        return np.zeros((samples, IMU_CHANNELS), dtype=np.float64)
    start = int(rng.integers(len(band)))
    repeats = int(np.ceil((start + samples) / len(band)))
    return np.tile(band, (repeats, 1))[start : start + samples]


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def pulse_source_from_material(
    row: Mapping[str, object], *, verify_hashes: bool = True
) -> PulseSourceEvent:
    """Load one five-shot keystroke from its full native 100 Hz slice."""

    if (
        str(row.get("schema_version", ""))
        not in ACCEPTED_MATERIAL_SCHEMAS
        or str(row.get("action", "")) != "keystroke"
        or str(row.get("imu_source_kind", ""))
        != "native_continuous_100hz_slice"
    ):
        raise KeystrokeImuPulseError(
            "material row is not an aligned five-shot keystroke"
        )
    source = Path(str(row.get("native_stream", ""))).resolve()
    if not source.is_file():
        raise KeystrokeImuPulseError("native material stream is missing")
    if verify_hashes and sha256_file(source) != str(
        row.get("native_stream_sha256", "")
    ):
        raise KeystrokeImuPulseError("native material stream hash changed")
    try:
        left = int(row["native_start_sample"])
        right = int(row["native_end_sample"])
    except (KeyError, TypeError, ValueError) as exc:
        raise KeystrokeImuPulseError("invalid native material interval") from exc
    with np.load(source, allow_pickle=False) as archive:
        required = {"imu", "timestamp_ns", "sample_rate_hz"}
        if required - set(archive.files):
            raise KeystrokeImuPulseError("native material stream is incomplete")
        if int(np.asarray(archive["sample_rate_hz"]).item()) != 100:
            raise KeystrokeImuPulseError("native material stream is not 100 Hz")
        imu = np.ascontiguousarray(
            np.asarray(archive["imu"][left:right], dtype=np.float32)
        )
        timestamp_ns = np.ascontiguousarray(
            np.asarray(archive["timestamp_ns"][left:right], dtype=np.int64)
        )
    if len(imu) < 2 or len(imu) != len(timestamp_ns):
        raise KeystrokeImuPulseError("native material slice is empty or incomplete")
    if np.any(np.diff(timestamp_ns) != 10_000_000):
        raise KeystrokeImuPulseError("native material timeline is not regular 100 Hz")
    if verify_hashes:
        if sha256_array(imu) != str(row.get("imu_sha256", "")):
            raise KeystrokeImuPulseError("native material IMU hash changed")
        if sha256_array(timestamp_ns) != str(
            row.get("imu_timestamp_ns_sha256", "")
        ):
            raise KeystrokeImuPulseError("native material timeline hash changed")
    downs = np.asarray(row.get("key_down_ms", ()), dtype=np.float64)
    letters = np.asarray(row.get("key_is_letter", ()), dtype=np.bool_)
    _as_schedule(downs, letters, name="material")
    return PulseSourceEvent(
        event_id=str(row.get("event_id", "")),
        imu=imu,
        key_down_ms=downs,
        is_letter=letters,
        imu_t_ms=timestamp_ns.astype(np.float64) / 1_000_000.0,
    )


def load_user_pulse_sources(
    material_manifest: str | Path,
    *,
    user_id: str,
    verify_hashes: bool = True,
) -> tuple[PulseSourceEvent, ...]:
    """Load the exact five aligned real keystrokes frozen for one user."""

    path = Path(material_manifest).resolve()
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                str(row.get("user_id", "")) == str(user_id)
                and str(row.get("action", "")) == "keystroke"
            ):
                rows.append(row)
    rows.sort(key=lambda value: int(value["shot_ordinal"]))
    if len(rows) != 5 or [int(row["shot_ordinal"]) for row in rows] != list(
        range(5)
    ):
        raise KeystrokeImuPulseError(
            f"{user_id} does not have exactly five aligned keystroke sources"
        )
    return tuple(
        pulse_source_from_material(row, verify_hashes=verify_hashes) for row in rows
    )


def fit_keystroke_imu_pulse_model(
    sources: Sequence[PulseSourceEvent],
    *,
    user_id: str,
    pre_ms: float = DEFAULT_PRE_MS,
    post_ms: float = DEFAULT_POST_MS,
    baseline_cutoff_hz: float = DEFAULT_BASELINE_CUTOFF_HZ,
    residual_block_ms: float = DEFAULT_RESIDUAL_BLOCK_MS,
) -> KeystrokeImuPulseModel:
    """Fit posture baselines, impact templates and a residual pool for one user."""

    if not sources:
        raise KeystrokeImuPulseError("pulse model needs at least one source event")
    pre_samples = int(round(float(pre_ms) / PERIOD_MS))
    post_samples = int(round(float(post_ms) / PERIOD_MS))
    if pre_samples < 0 or post_samples < 1:
        raise KeystrokeImuPulseError("template window is empty")
    if not np.isfinite(baseline_cutoff_hz) or baseline_cutoff_hz <= 0.0:
        raise KeystrokeImuPulseError("baseline cutoff must be positive")
    window = int(round(1000.0 / PERIOD_MS / float(baseline_cutoff_hz)))
    block = max(2, int(round(float(residual_block_ms) / PERIOD_MS)))

    baselines: list[np.ndarray] = []
    letter_stack: list[np.ndarray] = []
    other_stack: list[np.ndarray] = []
    detrended: list[np.ndarray] = []
    impact_masks: list[np.ndarray] = []
    identities: list[str] = []
    explicit_timeline_sources = 0
    for source in sources:
        imu = _as_imu(source.imu, name=f"source {source.event_id}")
        downs, letters = _as_schedule(
            source.key_down_ms, source.is_letter, name=f"source {source.event_id}"
        )
        timeline = _as_source_timeline(
            source.imu_t_ms,
            samples=len(imu),
            name=f"source {source.event_id}",
        )
        explicit_timeline_sources += int(source.imu_t_ms is not None)
        baseline = _baseline(imu, window)
        residual = imu - baseline
        baselines.append(baseline)
        detrended.append(residual)
        identities.append(str(source.event_id))
        struck = np.zeros(len(residual), dtype=bool)
        for down in downs:
            hit = _nearest_timeline_index(timeline, float(down))
            struck[
                max(0, hit - pre_samples) : min(len(residual), hit + post_samples)
            ] = True
        impact_masks.append(struck)
        for down, letter in zip(downs, letters):
            # Detector shards keep a long event's full time span but may
            # compress it to 512 rows.  Index*10ms is therefore not a valid
            # source clock.  Explicit native timestamps align each impact to
            # the actual KeyPress time without stretching it.
            center = _nearest_timeline_index(timeline, float(down))
            start = center - pre_samples
            end = start + pre_samples + post_samples
            if start < 0 or end > len(residual):
                continue
            (letter_stack if bool(letter) else other_stack).append(
                residual[start:end]
            )

    if not letter_stack and not other_stack:
        raise KeystrokeImuPulseError(
            "no key press had a complete impact window in the source events"
        )
    zero = np.zeros((pre_samples + post_samples, IMU_CHANNELS), dtype=np.float64)
    letter_template = (
        np.mean(np.stack(letter_stack, axis=0), axis=0) if letter_stack else zero
    )
    other_template = (
        np.mean(np.stack(other_stack, axis=0), axis=0) if other_stack else zero
    )
    if not letter_stack:
        letter_template = other_template.copy()
    if not other_stack:
        other_template = letter_template.copy()

    # The residual pool is pasted at arbitrary positions in the synthesised
    # event, so any impact left inside it lands where no key was pressed.  Tiling
    residual_blocks: list[np.ndarray] = []
    for residual, struck in zip(detrended, impact_masks):
        quiet = ~struck
        edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
        bounds = np.concatenate(([0], edges + 1, [len(quiet)]))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if quiet[lo] and hi - lo >= block:
                residual_blocks.append(residual[lo:hi])
    if not residual_blocks:
        # Dense enough typing can leave no quiet run at all.  Falling back to the
        # whole source is worse than this pool but better than refusing to build.
        residual_blocks = [np.asarray(residual) for residual in detrended]
    if not residual_blocks:
        residual_blocks = [np.zeros((block, IMU_CHANNELS), dtype=np.float64)]

    return KeystrokeImuPulseModel(
        schema_version=SCHEMA_VERSION,
        user_id=str(user_id),
        pre_samples=pre_samples,
        post_samples=post_samples,
        letter_template=letter_template,
        other_template=other_template,
        letter_observations=len(letter_stack),
        other_observations=len(other_stack),
        baselines=tuple(baselines),
        residual_blocks=tuple(residual_blocks),
        source_event_ids=tuple(identities),
        baseline_window_samples=window,
        explicit_timeline_sources=explicit_timeline_sources,
    )


def generate_keystroke_imu(
    model: KeystrokeImuPulseModel,
    *,
    key_down_ms: Iterable[object],
    is_letter: Iterable[object],
    duration_ms: float,
    seed: int,
    event_id: str = "",
) -> GeneratedKeystrokeImu:
    """Synthesise one complete event for an exact requested key schedule."""

    if model.schema_version != SCHEMA_VERSION:
        raise KeystrokeImuPulseError("pulse model schema mismatch")
    downs, letters = _as_schedule(key_down_ms, is_letter, name="request")
    samples = _samples_for_duration(duration_ms)
    if float(downs[-1]) >= float(duration_ms):
        raise KeystrokeImuPulseError("a requested key press starts after the event")

    rng = np.random.default_rng(
        _stable_seed(SCHEMA_VERSION, model.user_id, event_id, int(seed))
    )
    canvas, baseline_choice, baseline_policy = _baseline_window(
        model.baselines, samples, rng
    )
    canvas = np.array(canvas, dtype=np.float64, copy=True)

    placed = 0
    clipped = 0
    for ordinal, (down, letter) in enumerate(zip(downs, letters)):
        template = (
            model.letter_template if bool(letter) else model.other_template
        )
        start = int(round(float(down) / PERIOD_MS)) - model.pre_samples
        lo, hi = _place(canvas, template, start)
        if hi <= lo:
            continue
        placed += 1
        if (hi - lo) != len(template):
            clipped += 1

    # Butting independent blocks together leaves a step at every join, which is
    # the very artefact this module's opening note says a detector finds with one
    residual = _joined_residual(model.residual_blocks, samples, rng)
    canvas = canvas + residual

    if not np.isfinite(canvas).all():
        raise KeystrokeImuPulseError("synthesised IMU is not finite")
    output = np.ascontiguousarray(canvas, dtype=np.float32)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "role": "keystroke_imu_pulse_adapter_event",
        "human_replay": False,
        "model_used": False,
        "diffusion_used": False,
        "window_join_used": False,
        "source_user_id": model.user_id,
        "source_event_ids": list(model.source_event_ids),
        "baseline_source_index": baseline_choice,
        "baseline_policy": baseline_policy,
        "baseline_time_scaled": True,
        "impulse_time_scaled": False,
        "requested_keys": int(len(downs)),
        "requested_letters": int(np.count_nonzero(letters)),
        "impulses_placed": int(placed),
        "impulses_clipped_at_edges": int(clipped),
        "letter_observations": int(model.letter_observations),
        "other_observations": int(model.other_observations),
        "template_samples": int(model.template_samples),
        "pre_samples": int(model.pre_samples),
        "post_samples": int(model.post_samples),
        "baseline_window_samples": int(model.baseline_window_samples),
        "explicit_timeline_sources": int(model.explicit_timeline_sources),
        "residual_blocks": int(len(model.residual_blocks)),
        "samples": int(samples),
        "duration_ms": float(duration_ms),
        "seed": int(seed),
    }
    return GeneratedKeystrokeImu(imu=output, provenance=provenance)


__all__ = [
    "GeneratedKeystrokeImu",
    "KeystrokeImuPulseError",
    "KeystrokeImuPulseModel",
    "PulseSourceEvent",
    "SCHEMA_VERSION",
    "fit_keystroke_imu_pulse_model",
    "generate_keystroke_imu",
    "load_user_pulse_sources",
    "pulse_source_from_material",
]
