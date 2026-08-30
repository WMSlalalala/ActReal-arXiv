from __future__ import annotations

"""Compact conditional generator for single-pointer touch trajectories.

The fitted artifact never stores raw events and generation never performs a
donor lookup.  Equal-endpoint taps use the stationary coordinate support
actually observed in training, while unequal-endpoint taps, scrolls, and
swipes use train-fitted row-increment models.  Those models preserve exact
pauses and local high-frequency motion before a global bridge enforces the
requested endpoint.  All three actions treat requested start and end
coordinates as hard constraints rather than search criteria.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .android_touch_observation import (
    ACTION_DOWN,
    ACTION_MASK,
    ACTION_MOVE,
    ACTION_UP,
    PERIOD_MS,
    screen_dimensions_for_orientation,
)


SCHEMA_VERSION = (
    "conditional-touch-generator-v4-exact-endpoint-smooth-tap-swipe"
)
SHAPE_SUPPORT_QUANTILE = 0.90
MAX_SUPPORT_RESAMPLE_ATTEMPTS = 1024
SUPPORTED_ACTIONS = ("tap", "scroll", "swipe")
DIRECTION8 = (
    "right",
    "down_right",
    "down",
    "down_left",
    "left",
    "up_left",
    "up",
    "up_right",
)
STATIONARY = "stationary"
_REQUIRED_ROW_FIELDS = (
    "event_id",
    "action",
    "orientation_id",
    "t_ms",
    "x_px",
    "y_px",
    "pressure",
    "android_action",
)


class ConditionalTouchGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedTouch:
    """One generated raw single-pointer contact."""

    action: str
    orientation_id: int
    requested_direction: str | None
    realized_direction: str
    t_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray
    pointer_id: np.ndarray
    android_action: np.ndarray
    frame_index: np.ndarray
    frame_end: np.ndarray
    residual_scale: float
    tap_stationary_branch: bool


@dataclass(frozen=True)
class _FunctionalModel:
    action: str
    orientation_id: int
    grid_size: int
    feature_center: np.ndarray
    feature_scale: np.ndarray
    channel_scale: np.ndarray
    beta: np.ndarray
    noise_loadings: np.ndarray
    stationary_probability: float
    training_event_count: int
    continuous_event_count: int
    median_update_rate_hz: float
    increment_model: "_IncrementModel | None"
    pressure_model: "_PressureModel"
    shape_model: "_ShapeModel | None"


@dataclass(frozen=True)
class _IncrementModel:
    """Aggregate row-increment statistics; never contains event rows/IDs."""

    phase_bin_count: int
    amplitude_feature_center: np.ndarray
    amplitude_feature_scale: np.ndarray
    amplitude_beta: np.ndarray
    amplitude_residual_quantiles: np.ndarray
    initial_move_probability: np.ndarray
    transition_probability: np.ndarray
    unconditional_move_probability: np.ndarray
    normalized_increment_mean: np.ndarray
    innovation_transform: np.ndarray
    autoregression: np.ndarray
    innovation_radius_quantiles: np.ndarray
    phase_active_counts: np.ndarray
    phase_interval_counts: np.ndarray
    transition_counts: np.ndarray
    training_event_count: int
    training_interval_count: int
    active_interval_count: int
    amplitude_event_count: int


@dataclass(frozen=True)
class _PressureModel:
    exact_zero_probability: float
    exact_one_probability: float
    interior_mean_quantiles: np.ndarray
    training_event_count: int
    exact_zero_event_count: int
    exact_one_event_count: int
    interior_event_count: int


@dataclass(frozen=True)
class _ShapeModel:
    path_chord_quantiles: np.ndarray
    lateral_excursion_quantiles: np.ndarray
    lateral_rms_quantiles: np.ndarray
    longitudinal_excursion_quantiles: np.ndarray
    longitudinal_reversal_quantiles: np.ndarray
    half_pixel_fraction_quantiles: np.ndarray
    metric_copula_transform: np.ndarray
    support_quantile: float
    path_chord_hard_max: float
    lateral_excursion_hard_max: float
    lateral_rms_hard_max: float
    longitudinal_excursion_hard_max: float
    longitudinal_reversal_hard_max: float
    coordinate_lattice_quantum_px: float
    coordinate_lattice_probability: float
    coordinate_lattice_count: int
    coordinate_value_count: int
    coordinate_fraction_modes: np.ndarray
    coordinate_fraction_probabilities: np.ndarray
    coordinate_fraction_counts: np.ndarray
    training_event_count: int


@dataclass(frozen=True)
class _TrainingCurve:
    action: str
    orientation_id: int
    duration_ms: float
    chord_length_px: float
    update_rate_hz: float
    curve: np.ndarray
    tap_stationary: bool
    tap_endpoint_equal: bool
    increment_phase: np.ndarray
    increment_value: np.ndarray
    increment_active: np.ndarray
    pressure_mean: float
    path_chord_ratio: float
    lateral_excursion_ratio: float
    lateral_rms_ratio: float
    longitudinal_excursion_ratio: float
    longitudinal_reversal_fraction: float
    half_pixel_gridpoint_fraction: float
    coordinate_values: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Bound once when this module is imported.  A long-lived process that imported
# older bytes cannot relabel its artifact with a hash read from newer hot-edited
# source at save time.
IMPORT_GENERATOR_SOURCE_SHA256 = _sha256_file(Path(__file__).resolve())
IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256 = _sha256_file(
    Path(__file__).resolve().with_name("android_touch_observation.py")
)


def _source_fingerprint_sha256(
    *,
    generator_source_sha256: str,
    android_touch_observation_source_sha256: str,
) -> str:
    """Bind the generator to every local source that changes its semantics."""

    canonical_sources = {
        "android_touch_observation.py": android_touch_observation_source_sha256,
        "conditional_touch_generator.py": generator_source_sha256,
    }
    payload = json.dumps(
        canonical_sources, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


IMPORT_SOURCE_FINGERPRINT_SHA256 = _source_fingerprint_sha256(
    generator_source_sha256=IMPORT_GENERATOR_SOURCE_SHA256,
    android_touch_observation_source_sha256=(
        IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
    ),
)


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _direction8(dx: float, dy: float) -> str:
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ConditionalTouchGeneratorError("endpoint vector must be finite")
    if float(np.hypot(dx, dy)) <= 0.0:
        return STATIONARY
    angle = float(np.arctan2(dy, dx))
    index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
    return DIRECTION8[index]


def _as_xy(name: str, value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ConditionalTouchGeneratorError(
            f"{name} must contain two finite coordinates"
        )
    return result


def _validate_timeline(value: Iterable[float]) -> np.ndarray:
    t_ms = np.asarray(value, dtype=np.float64)
    if t_ms.ndim != 1 or len(t_ms) < 2:
        raise ConditionalTouchGeneratorError("t_ms must contain at least two rows")
    if not np.isfinite(t_ms).all():
        raise ConditionalTouchGeneratorError("t_ms contains non-finite values")
    if np.any(np.diff(t_ms) < 0.0):
        raise ConditionalTouchGeneratorError("t_ms must be nondecreasing")
    if float(t_ms[-1] - t_ms[0]) <= 0.0:
        raise ConditionalTouchGeneratorError("touch duration must be positive")
    return t_ms.copy()


def _frame_lifecycle(t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(
        (np.asarray([True]), np.not_equal(t_ms[1:], t_ms[:-1]))
    )
    frame_index = np.cumsum(starts, dtype=np.int64) - 1
    frame_end = np.concatenate(
        (np.not_equal(frame_index[1:], frame_index[:-1]), np.asarray([True]))
    )
    return frame_index, frame_end


def _interpolate_last_value(
    t_ms: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a fitting curve after ZOH-style duplicate-time collapse."""

    unique_times, first_indices = np.unique(t_ms, return_index=True)
    last_indices = np.concatenate((first_indices[1:] - 1, [len(t_ms) - 1]))
    selected = last_indices.astype(np.int64)
    u = (unique_times - unique_times[0]) / (unique_times[-1] - unique_times[0])
    return np.interp(grid, u, values[selected])


def _empirical_quantiles(values: Sequence[float] | np.ndarray, count: int) -> np.ndarray:
    """Return a compact inverse empirical CDF, including observed endpoints."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.asarray([0.0], dtype=np.float64)
    if len(array) == 1:
        return np.asarray([float(array[0])], dtype=np.float64)
    probabilities = np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float64)
    return np.asarray(np.quantile(array, probabilities), dtype=np.float64)


def _sample_empirical_quantiles(
    quantiles: np.ndarray,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(quantiles, dtype=np.float64)
    if len(values) == 1:
        return float(values[0])
    position = float(rng.random()) * float(len(values) - 1)
    lower = min(int(np.floor(position)), len(values) - 2)
    fraction = position - lower
    return float(values[lower] + fraction * (values[lower + 1] - values[lower]))


def _evaluate_empirical_quantiles(
    quantiles: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(quantiles, dtype=np.float64)
    if len(values) == 1:
        return float(values[0])
    position = float(np.clip(probability, 0.0, 1.0)) * float(len(values) - 1)
    lower = min(int(np.floor(position)), len(values) - 2)
    fraction = position - lower
    return float(values[lower] + fraction * (values[lower + 1] - values[lower]))


def _detector_zoh_indices(
    u: np.ndarray,
    duration_ms: float,
    target_samples: int | None = None,
) -> np.ndarray:
    target_samples = (
        max(2, int(round(duration_ms / PERIOD_MS)) + 1)
        if target_samples is None
        else int(target_samples)
    )
    detector_u = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    selected = np.searchsorted(u, detector_u, side="right") - 1
    return np.clip(selected, 0, len(u) - 1).astype(np.int64)


def _detector_shape_metrics(
    points: np.ndarray,
    u: np.ndarray,
    duration_ms: float,
    target_samples: int | None = None,
) -> np.ndarray:
    selected = _detector_zoh_indices(u, duration_ms, target_samples)
    observed = np.asarray(points, dtype=np.float64)[selected]
    chord = observed[-1] - observed[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 0.0:
        return np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
    tangent = chord / chord_length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    local = np.column_stack(
        (
            (observed - observed[0]) @ tangent / chord_length,
            (observed - observed[0]) @ normal / chord_length,
        )
    )
    increments = np.diff(local, axis=0)
    step = np.linalg.norm(increments, axis=1)
    nonzero = step > 1.0e-12
    reverse = (
        float(np.mean(increments[nonzero, 0] < -1.0e-12))
        if np.any(nonzero)
        else 0.0
    )
    detector_u = np.linspace(0.0, 1.0, len(observed), dtype=np.float64)
    half_pixel_distance = np.abs(observed * 2.0 - np.rint(observed * 2.0)) / 2.0
    return np.asarray(
        (
            float(np.sum(step)),
            float(np.max(np.abs(local[:, 1]))),
            float(np.sqrt(np.mean(local[:, 1] ** 2))),
            float(np.max(np.abs(local[:, 0] - detector_u))),
            reverse,
            float(np.mean(half_pixel_distance <= 2.5e-4)),
        ),
        dtype=np.float64,
    )


def _event_curve(
    *,
    action: str,
    orientation_id: int,
    t_ms: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    pressure: np.ndarray,
    android_action: np.ndarray,
    grid: np.ndarray,
    tap_stationary_tolerance_px: float,
) -> _TrainingCurve:
    masks = np.bitwise_and(android_action.astype(np.int64), ACTION_MASK)
    down_rows = np.flatnonzero(masks == ACTION_DOWN)
    if len(down_rows) != 1:
        raise ConditionalTouchGeneratorError(
            "event must have exactly one ACTION_DOWN"
        )
    down = int(down_rows[0])
    up_rows = np.flatnonzero((np.arange(len(masks)) > down) & (masks == ACTION_UP))
    if len(up_rows) != 1 or np.count_nonzero(masks == ACTION_UP) != 1:
        raise ConditionalTouchGeneratorError(
            "event must have exactly one ACTION_UP after ACTION_DOWN"
        )
    up = int(up_rows[0])
    event_slice = slice(down, up + 1)
    t_event = _validate_timeline(t_ms[event_slice])
    x_event = x_px[event_slice]
    y_event = y_px[event_slice]
    pressure_event = pressure[event_slice]
    if not (
        np.isfinite(x_event).all()
        and np.isfinite(y_event).all()
        and np.isfinite(pressure_event).all()
    ):
        raise ConditionalTouchGeneratorError("event values must be finite")
    width_px, height_px = screen_dimensions_for_orientation(orientation_id)
    if (
        np.any(x_event < 0.0)
        or np.any(x_event > width_px)
        or np.any(y_event < 0.0)
        or np.any(y_event > height_px)
    ):
        raise ConditionalTouchGeneratorError("training coordinates leave the screen")

    duration_ms = float(t_event[-1] - t_event[0])
    update_rate_hz = float((len(t_event) - 1) * 1000.0 / duration_ms)
    x_grid = _interpolate_last_value(t_event, x_event, grid)
    y_grid = _interpolate_last_value(t_event, y_event, grid)
    pressure_grid = _interpolate_last_value(t_event, pressure_event, grid)
    pressure_grid = np.clip(pressure_grid, 1.0e-5, 1.0 - 1.0e-5)
    pressure_logit = np.log(pressure_grid / (1.0 - pressure_grid))

    event_points = np.column_stack((x_event, y_event))
    start = np.asarray((x_event[0], y_event[0]), dtype=np.float64)
    end = np.asarray((x_event[-1], y_event[-1]), dtype=np.float64)
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    points = np.column_stack((x_grid, y_grid))
    event_u = (t_event - t_event[0]) / duration_ms
    increment_phase = 0.5 * (event_u[:-1] + event_u[1:])
    raw_increment = np.diff(event_points, axis=0)
    pressure_mean = float(np.mean(pressure_event))
    if action == "tap":
        stationary = bool(
            np.max(
                np.linalg.norm(
                    np.column_stack((x_event, y_event)) - start[None, :],
                    axis=1,
                )
            )
            <= tap_stationary_tolerance_px
        )
        endpoint_equal = bool(np.array_equal(start, end))
        linear = start[None, :] + grid[:, None] * chord[None, :]
        if not endpoint_equal:
            if chord_length <= 1.0e-9:
                raise ConditionalTouchGeneratorError(
                    "unequal tap endpoint displacement is numerically zero"
                )
            tangent = chord / chord_length
            normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
            residual = points - linear
            coordinate_curve = np.column_stack(
                (
                    residual @ tangent / chord_length,
                    residual @ normal / chord_length,
                )
            )
            local_increment = np.column_stack(
                (raw_increment @ tangent, raw_increment @ normal)
            ) / chord_length
            increment_value = local_increment * float(len(raw_increment))
            increment_active = (
                np.linalg.norm(raw_increment, axis=1)
                > tap_stationary_tolerance_px
            )
            detector_metrics = _detector_shape_metrics(
                event_points,
                event_u,
                duration_ms,
            )
            path_chord_ratio = float(detector_metrics[0])
            lateral_excursion_ratio = float(detector_metrics[1])
            lateral_rms_ratio = float(detector_metrics[2])
            longitudinal_excursion_ratio = float(detector_metrics[3])
            longitudinal_reversal_fraction = float(detector_metrics[4])
            half_pixel_gridpoint_fraction = float(detector_metrics[5])
        else:
            # Equal-endpoint moved loops are retained in the support audit but
            # deliberately excluded from the unequal-endpoint increment fit.
            # The real training split currently contains no such event.
            chord_length = 0.0
            coordinate_curve = points - linear
            increment_value = raw_increment
            increment_active = (
                np.linalg.norm(raw_increment, axis=1)
                > tap_stationary_tolerance_px
            )
            path_chord_ratio = 1.0
            lateral_excursion_ratio = 0.0
            lateral_rms_ratio = 0.0
            longitudinal_excursion_ratio = 0.0
            longitudinal_reversal_fraction = 0.0
            half_pixel_gridpoint_fraction = float(
                np.mean(
                    np.abs(event_points * 2.0 - np.rint(event_points * 2.0))
                    / 2.0
                    <= 1.0e-7
                )
            )
    else:
        if chord_length <= 1.0e-9:
            raise ConditionalTouchGeneratorError(
                "scroll/swipe training endpoint displacement is zero"
            )
        tangent = chord / chord_length
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
        linear = start[None, :] + grid[:, None] * chord[None, :]
        residual = points - linear
        coordinate_curve = np.column_stack(
            (
                residual @ tangent / chord_length,
                residual @ normal / chord_length,
            )
        )
        stationary = False
        endpoint_equal = False
        local_increment = np.column_stack(
            (raw_increment @ tangent, raw_increment @ normal)
        ) / chord_length
        increment_value = local_increment * float(len(raw_increment))
        increment_active = np.linalg.norm(raw_increment, axis=1) > 0.0
        detector_metrics = _detector_shape_metrics(
            event_points,
            event_u,
            duration_ms,
        )
        path_chord_ratio = float(detector_metrics[0])
        lateral_excursion_ratio = float(detector_metrics[1])
        lateral_rms_ratio = float(detector_metrics[2])
        longitudinal_excursion_ratio = float(detector_metrics[3])
        longitudinal_reversal_fraction = float(detector_metrics[4])
        half_pixel_gridpoint_fraction = float(detector_metrics[5])
    coordinate_curve[0] = 0.0
    coordinate_curve[-1] = 0.0
    curve = np.column_stack((coordinate_curve, pressure_logit))
    return _TrainingCurve(
        action=action,
        orientation_id=orientation_id,
        duration_ms=duration_ms,
        chord_length_px=chord_length,
        update_rate_hz=update_rate_hz,
        curve=curve,
        tap_stationary=stationary,
        tap_endpoint_equal=endpoint_equal,
        increment_phase=np.asarray(increment_phase, dtype=np.float64),
        increment_value=np.asarray(increment_value, dtype=np.float64),
        increment_active=np.asarray(increment_active, dtype=np.bool_),
        pressure_mean=pressure_mean,
        path_chord_ratio=path_chord_ratio,
        lateral_excursion_ratio=lateral_excursion_ratio,
        lateral_rms_ratio=lateral_rms_ratio,
        longitudinal_excursion_ratio=longitudinal_excursion_ratio,
        longitudinal_reversal_fraction=longitudinal_reversal_fraction,
        half_pixel_gridpoint_fraction=half_pixel_gridpoint_fraction,
        coordinate_values=np.asarray(event_points, dtype=np.float64),
    )


def _probability_from_counts(
    active: float,
    total: float,
    fallback: float,
) -> float:
    if total <= 0.0:
        return float(fallback)
    return float(active / total)


def _covariance_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted covariance square-root and inverse, with numeric PSD cleanup."""

    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        zeros = np.zeros((2, 2), dtype=np.float64)
        return zeros, zeros
    centered = array - np.mean(array, axis=0, keepdims=True)
    covariance = centered.T @ centered / float(len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    numeric_floor = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    positive = eigenvalues > numeric_floor
    square_root = np.zeros((2, 2), dtype=np.float64)
    inverse = np.zeros((2, 2), dtype=np.float64)
    if np.any(positive):
        basis = eigenvectors[:, positive]
        roots = np.sqrt(eigenvalues[positive])
        square_root = (basis * roots[None, :]) @ basis.T
        inverse = (basis / roots[None, :]) @ basis.T
    return square_root, inverse


def _fit_increment_model(
    curves: Sequence[_TrainingCurve],
    *,
    grid_size: int,
    ridge: float,
) -> _IncrementModel:
    """Fit aggregate pause, amplitude, phase, covariance and AR statistics."""

    phase_bin_count = max(4, int(grid_size) - 1)
    modeling_curves = list(curves)
    if not modeling_curves or any(
        curve.chord_length_px <= 1.0e-9 for curve in modeling_curves
    ):
        raise ConditionalTouchGeneratorError(
            "increment fitting requires positive endpoint displacement"
        )
    amplitude_curves: list[_TrainingCurve] = []
    amplitude_scales: list[float] = []
    for curve in modeling_curves:
        active_values = curve.increment_value[curve.increment_active]
        if len(active_values):
            scale = float(np.sqrt(np.mean(np.sum(active_values**2, axis=1))))
            if np.isfinite(scale) and scale > 0.0:
                amplitude_curves.append(curve)
                amplitude_scales.append(scale)

    feature_count = 3
    if amplitude_curves:
        feature_rows = []
        for curve in amplitude_curves:
            row = [
                np.log(curve.duration_ms),
                np.log(max(1, len(curve.increment_value))),
            ]
            row.append(np.log(curve.chord_length_px))
            feature_rows.append(row)
        raw_features = np.asarray(feature_rows, dtype=np.float64)
        amplitude_feature_center = np.mean(raw_features, axis=0)
        amplitude_feature_scale = np.std(raw_features, axis=0)
        amplitude_feature_scale = np.where(
            amplitude_feature_scale < 1.0e-12,
            1.0,
            amplitude_feature_scale,
        )
        amplitude_design = np.column_stack(
            (
                np.ones(len(raw_features), dtype=np.float64),
                (raw_features - amplitude_feature_center[None, :])
                / amplitude_feature_scale[None, :],
            )
        )
        log_scales = np.log(np.asarray(amplitude_scales, dtype=np.float64))
        penalty = np.eye(amplitude_design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        amplitude_beta = np.linalg.solve(
            amplitude_design.T @ amplitude_design + penalty,
            amplitude_design.T @ log_scales,
        )
        amplitude_residual = log_scales - amplitude_design @ amplitude_beta
        amplitude_residual_quantiles = _empirical_quantiles(
            amplitude_residual,
            max(9, 2 * grid_size - 1),
        )
    else:
        amplitude_feature_center = np.zeros(feature_count, dtype=np.float64)
        amplitude_feature_scale = np.ones(feature_count, dtype=np.float64)
        amplitude_beta = np.zeros(feature_count + 1, dtype=np.float64)
        amplitude_residual_quantiles = np.asarray([0.0], dtype=np.float64)

    phase_interval_counts = np.zeros(phase_bin_count, dtype=np.int64)
    phase_active_counts = np.zeros(phase_bin_count, dtype=np.int64)
    initial_counts = np.zeros((phase_bin_count, 2), dtype=np.int64)
    transition_counts = np.zeros((phase_bin_count, 2, 2), dtype=np.int64)
    active_by_phase: list[list[np.ndarray]] = [
        [] for _ in range(phase_bin_count)
    ]
    normalized_by_curve: list[tuple[_TrainingCurve, np.ndarray, np.ndarray]] = []
    scale_by_identity = {
        id(curve): scale for curve, scale in zip(amplitude_curves, amplitude_scales)
    }
    for curve in modeling_curves:
        bins = np.minimum(
            (np.clip(curve.increment_phase, 0.0, 1.0) * phase_bin_count).astype(
                np.int64
            ),
            phase_bin_count - 1,
        )
        states = curve.increment_active.astype(np.int64)
        for interval_index, (phase_bin, state) in enumerate(zip(bins, states)):
            phase_interval_counts[phase_bin] += 1
            phase_active_counts[phase_bin] += int(state)
            if interval_index == 0:
                initial_counts[phase_bin, state] += 1
            else:
                transition_counts[phase_bin, states[interval_index - 1], state] += 1
        curve_scale = scale_by_identity.get(id(curve), 0.0)
        if curve_scale > 0.0:
            normalized = curve.increment_value / curve_scale
        else:
            normalized = np.zeros_like(curve.increment_value)
        normalized_by_curve.append((curve, bins, normalized))
        for phase_bin, state, value in zip(bins, states, normalized):
            if state:
                active_by_phase[phase_bin].append(value)

    total_intervals = int(np.sum(phase_interval_counts))
    total_active = int(np.sum(phase_active_counts))
    global_move_probability = (
        float(total_active / total_intervals) if total_intervals else 0.0
    )
    unconditional_move_probability = np.asarray(
        [
            _probability_from_counts(
                phase_active_counts[index],
                phase_interval_counts[index],
                global_move_probability,
            )
            for index in range(phase_bin_count)
        ],
        dtype=np.float64,
    )
    initial_move_probability = np.asarray(
        [
            _probability_from_counts(
                initial_counts[index, 1],
                np.sum(initial_counts[index]),
                unconditional_move_probability[index],
            )
            for index in range(phase_bin_count)
        ],
        dtype=np.float64,
    )
    global_transition = np.zeros((2, 2), dtype=np.float64)
    summed_transition = np.sum(transition_counts, axis=0)
    for previous in range(2):
        denominator = float(np.sum(summed_transition[previous]))
        fallback = global_move_probability
        move_probability = _probability_from_counts(
            summed_transition[previous, 1], denominator, fallback
        )
        global_transition[previous] = (1.0 - move_probability, move_probability)
    transition_probability = np.zeros((phase_bin_count, 2, 2), dtype=np.float64)
    for phase_bin in range(phase_bin_count):
        for previous in range(2):
            counts = transition_counts[phase_bin, previous]
            move_probability = _probability_from_counts(
                counts[1],
                np.sum(counts),
                global_transition[previous, 1],
            )
            transition_probability[phase_bin, previous] = (
                1.0 - move_probability,
                move_probability,
            )

    all_active_values = [
        np.asarray(value, dtype=np.float64)
        for phase_values in active_by_phase
        for value in phase_values
    ]
    global_active_array = (
        np.stack(all_active_values)
        if all_active_values
        else np.zeros((1, 2), dtype=np.float64)
    )
    global_active_mean = np.mean(global_active_array, axis=0)
    normalized_increment_mean = np.zeros((phase_bin_count, 2), dtype=np.float64)
    for phase_bin, values in enumerate(active_by_phase):
        normalized_increment_mean[phase_bin] = (
            np.mean(np.stack(values), axis=0) if values else global_active_mean
        )

    previous_residuals: list[np.ndarray] = []
    next_residuals: list[np.ndarray] = []
    residual_records: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    for curve, bins, normalized in normalized_by_curve:
        previous_residual: np.ndarray | None = None
        previous_was_active = False
        for phase_bin, state, value in zip(bins, curve.increment_active, normalized):
            if not state:
                previous_residual = None
                previous_was_active = False
                continue
            residual = value - normalized_increment_mean[phase_bin]
            if previous_was_active and previous_residual is not None:
                previous_residuals.append(previous_residual)
                next_residuals.append(residual)
                residual_records.append((phase_bin, residual, previous_residual))
            else:
                residual_records.append((phase_bin, residual, None))
            previous_residual = residual
            previous_was_active = True
    if previous_residuals:
        previous_matrix = np.stack(previous_residuals)
        next_matrix = np.stack(next_residuals)
        autoregression = np.linalg.solve(
            previous_matrix.T @ previous_matrix
            + np.eye(2, dtype=np.float64) * ridge,
            previous_matrix.T @ next_matrix,
        ).T
    else:
        autoregression = np.zeros((2, 2), dtype=np.float64)

    innovation_by_phase: list[list[np.ndarray]] = [
        [] for _ in range(phase_bin_count)
    ]
    for phase_bin, residual, previous in residual_records:
        innovation = (
            residual
            if previous is None
            else residual - autoregression @ previous
        )
        innovation_by_phase[phase_bin].append(innovation)
    all_innovations = [
        value for phase_values in innovation_by_phase for value in phase_values
    ]
    global_innovation = (
        np.stack(all_innovations)
        if all_innovations
        else np.zeros((1, 2), dtype=np.float64)
    )
    global_transform, global_inverse = _covariance_transform(global_innovation)
    innovation_transform = np.zeros((phase_bin_count, 2, 2), dtype=np.float64)
    inverse_by_phase = np.zeros_like(innovation_transform)
    for phase_bin, values in enumerate(innovation_by_phase):
        if len(values) >= 2:
            transform, inverse = _covariance_transform(np.stack(values))
        else:
            transform, inverse = global_transform, global_inverse
        innovation_transform[phase_bin] = transform
        inverse_by_phase[phase_bin] = inverse
    radii: list[float] = []
    for phase_bin, values in enumerate(innovation_by_phase):
        inverse = inverse_by_phase[phase_bin]
        for value in values:
            radii.append(float(np.linalg.norm(inverse @ value)))
    innovation_radius_quantiles = _empirical_quantiles(
        radii,
        max(9, 2 * grid_size - 1),
    )
    return _IncrementModel(
        phase_bin_count=phase_bin_count,
        amplitude_feature_center=amplitude_feature_center,
        amplitude_feature_scale=amplitude_feature_scale,
        amplitude_beta=amplitude_beta,
        amplitude_residual_quantiles=amplitude_residual_quantiles,
        initial_move_probability=initial_move_probability,
        transition_probability=transition_probability,
        unconditional_move_probability=unconditional_move_probability,
        normalized_increment_mean=normalized_increment_mean,
        innovation_transform=innovation_transform,
        autoregression=autoregression,
        innovation_radius_quantiles=innovation_radius_quantiles,
        phase_active_counts=phase_active_counts,
        phase_interval_counts=phase_interval_counts,
        transition_counts=transition_counts,
        training_event_count=len(modeling_curves),
        training_interval_count=total_intervals,
        active_interval_count=total_active,
        amplitude_event_count=len(amplitude_curves),
    )


def _fit_pressure_model(
    curves: Sequence[_TrainingCurve],
    *,
    grid_size: int,
) -> _PressureModel:
    means = np.asarray([curve.pressure_mean for curve in curves], dtype=np.float64)
    exact_zero = means == 0.0
    exact_one = means == 1.0
    interior = ~(exact_zero | exact_one)
    interior_quantiles = _empirical_quantiles(
        means[interior],
        max(9, 2 * grid_size - 1),
    ) if np.any(interior) else np.empty(0, dtype=np.float64)
    return _PressureModel(
        exact_zero_probability=float(np.mean(exact_zero)),
        exact_one_probability=float(np.mean(exact_one)),
        interior_mean_quantiles=interior_quantiles,
        training_event_count=len(curves),
        exact_zero_event_count=int(np.sum(exact_zero)),
        exact_one_event_count=int(np.sum(exact_one)),
        interior_event_count=int(np.sum(interior)),
    )


def _fit_shape_model(
    curves: Sequence[_TrainingCurve],
    *,
    grid_size: int,
) -> _ShapeModel:
    quantile_count = max(9, 2 * grid_size - 1)
    coordinates = np.concatenate(
        [curve.coordinate_values.reshape(-1) for curve in curves]
    )
    # Exact fractional-coordinate point masses are separated from the broad
    # continuous component at the largest empirical log-count elbow.  This
    # discovers the integer/half-pixel modes in HMOG without assuming 0.5 px.
    fractions = np.mod(coordinates, 1.0)
    fractions = np.where(fractions == 1.0, 0.0, fractions)
    fraction_values, fraction_counts = np.unique(fractions, return_counts=True)
    order = np.argsort(fraction_counts)[::-1]
    ordered_counts = fraction_counts[order]
    if len(ordered_counts) == 1:
        mode_count = 1
    else:
        log_gaps = np.log(ordered_counts[:-1]) - np.log(ordered_counts[1:])
        mode_count = int(np.argmax(log_gaps)) + 1
    mode_indices = order[:mode_count]
    coordinate_fraction_modes = fraction_values[mode_indices].astype(np.float64)
    coordinate_fraction_counts = fraction_counts[mode_indices].astype(np.int64)
    mode_order = np.argsort(coordinate_fraction_modes)
    coordinate_fraction_modes = coordinate_fraction_modes[mode_order]
    coordinate_fraction_counts = coordinate_fraction_counts[mode_order]
    best_count = int(np.sum(coordinate_fraction_counts))
    coordinate_fraction_probabilities = (
        coordinate_fraction_counts.astype(np.float64) / float(len(coordinates))
    )
    if len(coordinate_fraction_modes) == 1:
        best_quantum = 1.0
    else:
        circular = np.sort(
            np.concatenate(
                (coordinate_fraction_modes, coordinate_fraction_modes[:1] + 1.0)
            )
        )
        positive_spacing = np.diff(circular)
        positive_spacing = positive_spacing[positive_spacing > 0.0]
        best_quantum = float(np.min(positive_spacing))
    path_values = np.asarray([curve.path_chord_ratio for curve in curves])
    lateral_values = np.asarray(
        [curve.lateral_excursion_ratio for curve in curves]
    )
    lateral_rms_values = np.asarray(
        [curve.lateral_rms_ratio for curve in curves]
    )
    longitudinal_values = np.asarray(
        [curve.longitudinal_excursion_ratio for curve in curves]
    )
    reversal_values = np.asarray(
        [curve.longitudinal_reversal_fraction for curve in curves]
    )
    half_pixel_values = np.asarray(
        [curve.half_pixel_gridpoint_fraction for curve in curves]
    )
    metric_values = np.column_stack(
        (
            path_values,
            lateral_values,
            lateral_rms_values,
            longitudinal_values,
            reversal_values,
            half_pixel_values,
        )
    )
    rank_uniform = np.empty_like(metric_values, dtype=np.float64)
    for channel in range(metric_values.shape[1]):
        values = metric_values[:, channel]
        unique, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True
        )
        cumulative = np.cumsum(counts)
        starts = cumulative - counts
        average_rank = 0.5 * (starts + cumulative - 1)
        rank_uniform[:, channel] = (
            average_rank[inverse] + 0.5
        ) / float(len(values))
    spearman = np.eye(metric_values.shape[1], dtype=np.float64)
    for left in range(metric_values.shape[1]):
        for right in range(left + 1, metric_values.shape[1]):
            if (
                np.std(rank_uniform[:, left]) > 0.0
                and np.std(rank_uniform[:, right]) > 0.0
            ):
                correlation = float(
                    np.corrcoef(
                        rank_uniform[:, left], rank_uniform[:, right]
                    )[0, 1]
                )
            else:
                correlation = 0.0
            gaussian_correlation = 2.0 * np.sin(np.pi * correlation / 6.0)
            spearman[left, right] = gaussian_correlation
            spearman[right, left] = gaussian_correlation
    eigenvalues, eigenvectors = np.linalg.eigh(spearman)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).eps))
    covariance /= diagonal[:, None] * diagonal[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    metric_copula_transform = (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    ) @ eigenvectors.T
    return _ShapeModel(
        path_chord_quantiles=_empirical_quantiles(path_values, quantile_count),
        lateral_excursion_quantiles=_empirical_quantiles(
            lateral_values, quantile_count
        ),
        lateral_rms_quantiles=_empirical_quantiles(
            lateral_rms_values, quantile_count
        ),
        longitudinal_excursion_quantiles=_empirical_quantiles(
            longitudinal_values, quantile_count
        ),
        longitudinal_reversal_quantiles=_empirical_quantiles(
            reversal_values, quantile_count
        ),
        half_pixel_fraction_quantiles=_empirical_quantiles(
            half_pixel_values, quantile_count
        ),
        metric_copula_transform=metric_copula_transform,
        support_quantile=SHAPE_SUPPORT_QUANTILE,
        path_chord_hard_max=float(
            np.quantile(path_values, SHAPE_SUPPORT_QUANTILE)
        ),
        lateral_excursion_hard_max=float(
            np.quantile(lateral_values, SHAPE_SUPPORT_QUANTILE)
        ),
        lateral_rms_hard_max=float(
            np.quantile(lateral_rms_values, SHAPE_SUPPORT_QUANTILE)
        ),
        longitudinal_excursion_hard_max=float(
            np.quantile(longitudinal_values, SHAPE_SUPPORT_QUANTILE)
        ),
        longitudinal_reversal_hard_max=float(
            np.quantile(reversal_values, SHAPE_SUPPORT_QUANTILE)
        ),
        coordinate_lattice_quantum_px=float(best_quantum),
        coordinate_lattice_probability=float(best_count / len(coordinates)),
        coordinate_lattice_count=int(best_count),
        coordinate_value_count=int(len(coordinates)),
        coordinate_fraction_modes=coordinate_fraction_modes,
        coordinate_fraction_probabilities=coordinate_fraction_probabilities,
        coordinate_fraction_counts=coordinate_fraction_counts,
        training_event_count=len(curves),
    )


def _fit_condition(
    curves: Sequence[_TrainingCurve],
    *,
    grid_size: int,
    max_rank: int,
    ridge: float,
) -> _FunctionalModel:
    action = curves[0].action
    orientation_id = curves[0].orientation_id
    stationary_probability = (
        float(np.mean([curve.tap_stationary for curve in curves]))
        if action == "tap"
        else 0.0
    )
    moving_tap_curves = (
        [curve for curve in curves if not curve.tap_endpoint_equal]
        if action == "tap"
        else []
    )
    continuous_curves = (
        moving_tap_curves
        if action == "tap" and moving_tap_curves
        else list(curves)
    )
    durations = np.log(
        np.asarray([curve.duration_ms for curve in continuous_curves], dtype=np.float64)
    )
    feature_columns = [durations]
    if action != "tap":
        feature_columns.append(
            np.log(
                np.asarray(
                    [curve.chord_length_px for curve in continuous_curves],
                    dtype=np.float64,
                )
            )
        )
    raw_features = np.column_stack(feature_columns)
    feature_center = np.mean(raw_features, axis=0)
    feature_scale = np.std(raw_features, axis=0)
    feature_scale = np.where(feature_scale < 1.0e-6, 1.0, feature_scale)
    design = np.column_stack(
        (
            np.ones(len(continuous_curves), dtype=np.float64),
            (raw_features - feature_center[None, :]) / feature_scale[None, :],
        )
    )

    curve_tensor = np.stack([curve.curve for curve in continuous_curves])
    channel_scale = np.std(curve_tensor, axis=(0, 1))
    floors = np.asarray((1.0e-3, 1.0e-3, 0.10), dtype=np.float64)
    channel_scale = np.maximum(channel_scale, floors)
    response = (curve_tensor / channel_scale[None, None, :]).reshape(
        len(continuous_curves), -1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    residual = response - design @ beta
    available_rank = min(
        int(max_rank),
        max(0, len(continuous_curves) - 1),
        residual.shape[1],
    )
    if available_rank:
        _, singular_values, right = np.linalg.svd(residual, full_matrices=False)
        noise_loadings = (
            singular_values[:available_rank, None]
            / np.sqrt(max(1, len(continuous_curves) - 1))
            * right[:available_rank]
        )
        nonzero = np.linalg.norm(noise_loadings, axis=1) > 1.0e-12
        noise_loadings = noise_loadings[nonzero]
    else:
        noise_loadings = np.empty((0, response.shape[1]), dtype=np.float64)
    increment_curves = (
        moving_tap_curves if action == "tap" else list(curves)
    )
    increment_model = (
        _fit_increment_model(
            increment_curves,
            grid_size=grid_size,
            ridge=ridge,
        )
        if increment_curves
        else None
    )
    return _FunctionalModel(
        action=action,
        orientation_id=orientation_id,
        grid_size=grid_size,
        feature_center=feature_center,
        feature_scale=feature_scale,
        channel_scale=channel_scale,
        beta=beta,
        noise_loadings=noise_loadings,
        stationary_probability=stationary_probability,
        training_event_count=len(curves),
        continuous_event_count=len(continuous_curves),
        median_update_rate_hz=float(
            np.median([curve.update_rate_hz for curve in curves])
        ),
        increment_model=increment_model,
        pressure_model=_fit_pressure_model(curves, grid_size=grid_size),
        shape_model=(
            _fit_shape_model(
                moving_tap_curves if action == "tap" else curves,
                grid_size=grid_size,
            )
            if action != "tap" or moving_tap_curves
            else None
        ),
    )


def _maximum_residual_scale(
    baseline: np.ndarray,
    residual: np.ndarray,
    *,
    width_px: float,
    height_px: float,
) -> float:
    maximum = 1.0
    for axis, upper in ((0, width_px), (1, height_px)):
        base = baseline[:, axis]
        delta = residual[:, axis]
        positive = delta > 0.0
        negative = delta < 0.0
        if np.any(positive):
            maximum = min(
                maximum,
                float(np.min((upper - base[positive]) / delta[positive])),
            )
        if np.any(negative):
            maximum = min(
                maximum,
                float(np.min((0.0 - base[negative]) / delta[negative])),
            )
    maximum = max(0.0, min(1.0, maximum))
    if maximum < 1.0:
        maximum = float(np.nextafter(maximum, 0.0))
    return maximum


def _sample_move_states(
    model: _IncrementModel,
    phase_bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    states = np.zeros(len(phase_bins), dtype=np.bool_)
    previous = 0
    for index, phase_bin in enumerate(phase_bins):
        probability = (
            model.initial_move_probability[phase_bin]
            if index == 0
            else model.transition_probability[phase_bin, previous, 1]
        )
        states[index] = bool(rng.random() < probability)
        previous = int(states[index])
    if len(states) and not np.any(states):
        weights = model.unconditional_move_probability[phase_bins]
        total = float(np.sum(weights))
        if total > 0.0:
            selected = int(rng.choice(len(states), p=weights / total))
        else:
            selected = int(np.argmax(model.phase_active_counts[phase_bins]))
        states[selected] = True
    return states


def _sample_increment_local_curve(
    model: _IncrementModel,
    *,
    duration_ms: float,
    chord_length_px: float,
    u: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample row increments and bridge their global sum to ``(1, 0)``."""

    interval_count = len(u) - 1
    midpoint = 0.5 * (u[:-1] + u[1:])
    phase_bins = np.minimum(
        (np.clip(midpoint, 0.0, 1.0) * model.phase_bin_count).astype(np.int64),
        model.phase_bin_count - 1,
    )
    raw_features = [np.log(duration_ms), np.log(max(1, interval_count))]
    raw_features.append(np.log(chord_length_px))
    standardized = (
        np.asarray(raw_features, dtype=np.float64)
        - model.amplitude_feature_center
    ) / model.amplitude_feature_scale
    log_amplitude = float(
        np.concatenate((np.asarray([1.0]), standardized)) @ model.amplitude_beta
    )
    log_amplitude += _sample_empirical_quantiles(
        model.amplitude_residual_quantiles, rng
    )
    amplitude = float(np.exp(log_amplitude))
    states = _sample_move_states(model, phase_bins, rng)
    normalized_values = np.zeros((interval_count, 2), dtype=np.float64)
    previous_residual: np.ndarray | None = None
    for index, (phase_bin, active) in enumerate(zip(phase_bins, states)):
        if not active:
            previous_residual = None
            continue
        direction = rng.standard_normal(2)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 0.0:
            direction /= direction_norm
        radius = _sample_empirical_quantiles(
            model.innovation_radius_quantiles, rng
        )
        innovation = model.innovation_transform[phase_bin] @ direction * radius
        residual = innovation
        if previous_residual is not None:
            residual = residual + model.autoregression @ previous_residual
        normalized_values[index] = (
            model.normalized_increment_mean[phase_bin] + residual
        )
        previous_residual = residual
    increments = amplitude * normalized_values / float(max(1, interval_count))
    active_indices = np.flatnonzero(states)
    correction = np.asarray((1.0, 0.0), dtype=np.float64) - np.sum(
        increments, axis=0
    )
    bridge_weights = np.linalg.norm(increments[active_indices], axis=1)
    if float(np.sum(bridge_weights)) <= 0.0:
        bridge_weights = model.unconditional_move_probability[
            phase_bins[active_indices]
        ]
    if float(np.sum(bridge_weights)) <= 0.0:
        bridge_weights = np.ones(len(active_indices), dtype=np.float64)
    bridge_weights = bridge_weights / np.sum(bridge_weights)
    increments[active_indices] += bridge_weights[:, None] * correction[None, :]
    local_curve = np.vstack(
        (np.zeros((1, 2), dtype=np.float64), np.cumsum(increments, axis=0))
    )
    local_curve[0] = 0.0
    local_curve[-1] = (1.0, 0.0)
    return local_curve


def _path_chord_ratio(local_curve: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(local_curve, axis=0), axis=1)))


def _longitudinal_reversal_fraction(local_curve: np.ndarray) -> float:
    increments = np.diff(local_curve[:, 0])
    denominator = float(np.sum(np.abs(increments)))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(np.maximum(-increments, 0.0)) / denominator)


def _largest_supported_scale(
    baseline: np.ndarray,
    residual: np.ndarray,
    metric,
    maximum: float,
) -> float:
    if metric(baseline + residual) <= maximum:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if metric(baseline + middle * residual) <= maximum:
            low = middle
        else:
            high = middle
    return low


def _pause_preserving_support_curve(local_curve: np.ndarray) -> np.ndarray:
    """Return a monotone chord path with the proposal's exact zero-step mask."""

    increments = np.diff(local_curve, axis=0)
    active = np.linalg.norm(increments, axis=1) > 0.0
    weights = np.maximum(increments[:, 0], 0.0)
    weights[~active] = 0.0
    if float(np.sum(weights)) <= 0.0:
        weights = np.linalg.norm(increments, axis=1)
        weights[~active] = 0.0
    if float(np.sum(weights)) <= 0.0:
        weights = active.astype(np.float64)
    cumulative_progress = np.cumsum(weights, dtype=np.float64)
    total_progress = float(cumulative_progress[-1])
    cumulative_progress /= total_progress
    np.maximum(cumulative_progress, 0.0, out=cumulative_progress)
    np.minimum(cumulative_progress, 1.0, out=cumulative_progress)
    np.maximum.accumulate(cumulative_progress, out=cumulative_progress)
    cumulative_progress[-1] = 1.0
    support = np.column_stack(
        (
            np.concatenate(([0.0], cumulative_progress)),
            np.zeros(len(weights) + 1, dtype=np.float64),
        )
    )
    support[0] = (0.0, 0.0)
    support[-1] = (1.0, 0.0)
    return support


def _sample_joint_shape_targets(
    model: _ShapeModel,
    rng: np.random.Generator,
) -> np.ndarray:
    latent = model.metric_copula_transform @ rng.standard_normal(6)
    probabilities = np.asarray(
        [0.5 * (1.0 + math.erf(float(value) / np.sqrt(2.0))) for value in latent],
        dtype=np.float64,
    )
    upper = probabilities > 0.5
    probabilities[upper] = probabilities[upper] - (
        (1.0 - model.support_quantile)
        * (2.0 * probabilities[upper] - 1.0) ** 4
    )
    quantiles = (
        model.path_chord_quantiles,
        model.lateral_excursion_quantiles,
        model.lateral_rms_quantiles,
        model.longitudinal_excursion_quantiles,
        model.longitudinal_reversal_quantiles,
        model.half_pixel_fraction_quantiles,
    )
    targets = np.asarray(
        [
            _evaluate_empirical_quantiles(values, probability)
            for values, probability in zip(quantiles, probabilities)
        ],
        dtype=np.float64,
    )
    targets[0] = min(max(1.0, targets[0]), model.path_chord_hard_max)
    targets[1] = min(max(0.0, targets[1]), model.lateral_excursion_hard_max)
    targets[2] = min(max(0.0, targets[2]), model.lateral_rms_hard_max)
    targets[3] = min(max(0.0, targets[3]), model.longitudinal_excursion_hard_max)
    targets[4] = min(max(0.0, targets[4]), model.longitudinal_reversal_hard_max)
    targets[5] = min(max(0.0, targets[5]), 1.0)
    return targets


def _apply_train_fitted_shape_support(
    local_curve: np.ndarray,
    u: np.ndarray,
    duration_ms: float,
    model: _ShapeModel,
    rng: np.random.Generator,
    detector_sample_count: int | None = None,
) -> tuple[np.ndarray, float]:
    """Calibrate detector-grid shape to a joint train-fitted metric target."""

    support = _pause_preserving_support_curve(local_curve)
    supported = np.asarray(local_curve, dtype=np.float64).copy()
    targets = _sample_joint_shape_targets(model, rng)
    selected = _detector_zoh_indices(u, duration_ms, detector_sample_count)
    observed_lateral = supported[selected, 1]
    current_lateral = float(np.max(np.abs(observed_lateral)))
    target_lateral = float(targets[1])
    target_lateral_rms = min(float(targets[2]), target_lateral)
    if target_lateral <= 0.0 or current_lateral <= 0.0:
        supported[:, 1] = 0.0
    else:
        normalized = np.abs(supported[:, 1] / current_lateral)
        normalized_observed = normalized[selected]
        desired_ratio = target_lateral_rms / target_lateral

        def rms_ratio(exponent: float) -> float:
            return float(
                np.sqrt(np.mean(normalized_observed ** (2.0 * exponent)))
            )

        low_exponent = 1.0e-3
        high_exponent = 64.0
        if desired_ratio >= rms_ratio(low_exponent):
            exponent = low_exponent
        elif desired_ratio <= rms_ratio(high_exponent):
            exponent = high_exponent
        else:
            for _ in range(60):
                middle = 0.5 * (low_exponent + high_exponent)
                if rms_ratio(middle) > desired_ratio:
                    low_exponent = middle
                else:
                    high_exponent = middle
            exponent = 0.5 * (low_exponent + high_exponent)
        supported[:, 1] = (
            np.sign(supported[:, 1])
            * normalized**exponent
            * target_lateral
        )

    tangent_residual = supported[:, 0] - support[:, 0]

    def tangent_candidate(scale: float) -> np.ndarray:
        candidate = supported.copy()
        candidate[:, 0] = support[:, 0] + scale * tangent_residual
        candidate[0] = (0.0, 0.0)
        candidate[-1] = (1.0, 0.0)
        return candidate

    target_path = float(targets[0])
    low_scale = 0.0
    low_curve = tangent_candidate(low_scale)
    low_metrics = _detector_shape_metrics(
        low_curve, u, duration_ms, detector_sample_count
    )
    if low_metrics[0] >= target_path:
        supported = low_curve
    else:
        high_scale = 1.0
        high_curve = tangent_candidate(high_scale)
        high_metrics = _detector_shape_metrics(
            high_curve, u, duration_ms, detector_sample_count
        )
        while (
            high_metrics[0] < target_path
            and high_metrics[0] < model.path_chord_hard_max
            and high_metrics[3] < model.longitudinal_excursion_hard_max
            and high_metrics[4] < model.longitudinal_reversal_hard_max
            and high_scale < 64.0
        ):
            high_scale *= 2.0
            high_curve = tangent_candidate(high_scale)
            high_metrics = _detector_shape_metrics(
                high_curve, u, duration_ms, detector_sample_count
            )
        for _ in range(60):
            middle = 0.5 * (low_scale + high_scale)
            candidate = tangent_candidate(middle)
            metrics = _detector_shape_metrics(
                candidate, u, duration_ms, detector_sample_count
            )
            valid = (
                metrics[0] <= target_path
                and metrics[0] <= model.path_chord_hard_max
                and metrics[3] <= model.longitudinal_excursion_hard_max
                and metrics[4] <= model.longitudinal_reversal_hard_max
            )
            if valid:
                low_scale = middle
                low_curve = candidate
            else:
                high_scale = middle
        supported = low_curve
    supported[0] = (0.0, 0.0)
    supported[-1] = (1.0, 0.0)
    return supported, float(targets[5])


def _apply_tap_increment_support(
    local_curve: np.ndarray,
    u: np.ndarray,
    duration_ms: float,
    model: _ShapeModel,
    rng: np.random.Generator,
    detector_sample_count: int | None = None,
) -> tuple[np.ndarray, float]:
    """Make a moving tap monotone and calibrate lateral motion safely."""

    proposal = np.asarray(local_curve, dtype=np.float64)
    support = _pause_preserving_support_curve(proposal)
    result = support.copy()
    targets = _sample_joint_shape_targets(model, rng)
    target_lateral = min(
        max(0.0, float(targets[1])),
        max(0.0, float(model.lateral_excursion_hard_max)),
    )
    proposal_lateral = proposal[:, 1]
    raw_peak = float(np.max(np.abs(proposal_lateral)))
    if target_lateral > 0.0 and raw_peak > np.finfo(np.float64).tiny:
        normalized = proposal_lateral / raw_peak
        selected = _detector_zoh_indices(
            u,
            duration_ms,
            detector_sample_count,
        )
        detector_peak = float(np.max(np.abs(normalized[selected])))
        hard_amplitude = max(0.0, float(model.lateral_excursion_hard_max))
        if detector_peak > 0.0 and hard_amplitude > 0.0:
            # Compare by multiplication before division so even a subnormal
            # detector peak cannot overflow the scalar calculation.
            amplitude = (
                hard_amplitude
                if target_lateral >= hard_amplitude * detector_peak
                else target_lateral / detector_peak
            )
            result[:, 1] = normalized * amplitude
    result[0] = (0.0, 0.0)
    result[-1] = (1.0, 0.0)
    if not np.isfinite(result).all():
        raise ConditionalTouchGeneratorError(
            "tap increment support produced a non-finite coordinate"
        )
    return result, float(targets[5])


def _apply_smooth_global_shape_support(
    local_curve: np.ndarray,
    u: np.ndarray,
    duration_ms: float,
    model: _ShapeModel,
    detector_sample_count: int | None = None,
) -> np.ndarray:
    """Keep a low-frequency proposal inside train-real shape support."""

    baseline = np.column_stack((u, np.zeros(len(u), dtype=np.float64)))
    residual = np.asarray(local_curve, dtype=np.float64) - baseline

    def candidate(scale: float) -> np.ndarray:
        value = baseline + float(scale) * residual
        value[0] = (0.0, 0.0)
        value[-1] = (1.0, 0.0)
        return value

    def supported(value: np.ndarray) -> bool:
        metrics = _detector_shape_metrics(
            value, u, duration_ms, detector_sample_count
        )
        tolerance = np.finfo(np.float64).eps * 64.0
        return bool(
            metrics[0] <= model.path_chord_hard_max + tolerance
            and metrics[1] <= model.lateral_excursion_hard_max + tolerance
            and metrics[2] <= model.lateral_rms_hard_max + tolerance
            and metrics[3] <= model.longitudinal_excursion_hard_max + tolerance
            and metrics[4] <= model.longitudinal_reversal_hard_max + tolerance
        )

    full = candidate(1.0)
    if supported(full):
        return full
    low = 0.0
    high = 1.0
    result = candidate(low)
    for _ in range(32):
        middle = 0.5 * (low + high)
        value = candidate(middle)
        if supported(value):
            low = middle
            result = value
        else:
            high = middle
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _pressure_from_train_fitted_mean(
    base_logit: np.ndarray,
    model: _PressureModel,
    rng: np.random.Generator,
) -> np.ndarray:
    branch = float(rng.random())
    if branch < model.exact_zero_probability:
        return np.zeros(len(base_logit), dtype=np.float64)
    if branch < model.exact_zero_probability + model.exact_one_probability:
        return np.ones(len(base_logit), dtype=np.float64)
    if not len(model.interior_mean_quantiles):
        if model.exact_one_event_count >= model.exact_zero_event_count:
            return np.ones(len(base_logit), dtype=np.float64)
        return np.zeros(len(base_logit), dtype=np.float64)
    target_mean = _sample_empirical_quantiles(
        model.interior_mean_quantiles, rng
    )
    target_mean = float(np.clip(target_mean, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    low = -100.0
    high = 100.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if float(np.mean(_stable_sigmoid(base_logit + middle))) < target_mean:
            low = middle
        else:
            high = middle
    pressure = _stable_sigmoid(base_logit + 0.5 * (low + high))
    return pressure


def _apply_train_fitted_coordinate_lattice(
    points: np.ndarray,
    model: _ShapeModel,
    rng: np.random.Generator,
    *,
    width_px: float,
    height_px: float,
    u: np.ndarray,
    duration_ms: float,
    target_fraction: float,
    detector_sample_count: int | None = None,
) -> np.ndarray:
    """Match a train-fitted detector-grid lattice fraction without overfill."""

    result = np.asarray(points, dtype=np.float64).copy()
    if not len(model.coordinate_fraction_modes):
        return result

    def inside_hard_shape_support(candidate_points: np.ndarray) -> bool:
        metrics = _detector_shape_metrics(
            candidate_points, u, duration_ms, detector_sample_count
        )
        tolerance = np.finfo(np.float64).eps * 32.0
        return bool(
            metrics[0] <= model.path_chord_hard_max + tolerance
            and metrics[1] <= model.lateral_excursion_hard_max + tolerance
            and metrics[2] <= model.lateral_rms_hard_max + tolerance
            and metrics[3] <= model.longitudinal_excursion_hard_max + tolerance
            and metrics[4] <= model.longitudinal_reversal_hard_max + tolerance
        )

    runs: list[tuple[int, int]] = []
    index = 1
    while index < len(points) - 1:
        run_end = index + 1
        while run_end < len(points) - 1 and np.array_equal(
            points[run_end], points[index]
        ):
            run_end += 1
        runs.append((index, run_end))
        index = run_end
    run_axes = [
        (index, run_end, axis, upper)
        for index, run_end in runs
        for axis, upper in ((0, width_px), (1, height_px))
    ]
    rng.shuffle(run_axes)

    current_fraction = float(
        _detector_shape_metrics(
            result, u, duration_ms, detector_sample_count
        )[5]
    )
    current_distance = abs(current_fraction - target_fraction)
    for index, run_end, axis, upper in run_axes:
        candidate_result = result.copy()
        coordinate = float(points[index, axis])
        candidates: list[float] = []
        for mode in model.coordinate_fraction_modes:
            shifted = coordinate - float(mode)
            candidates.extend(
                (np.floor(shifted) + float(mode), np.ceil(shifted) + float(mode))
            )
        valid_candidates = np.asarray(
            [value for value in candidates if 0.0 <= value <= upper],
            dtype=np.float64,
        )
        if len(valid_candidates):
            candidate_result[index:run_end, axis] = valid_candidates[
                int(np.argmin(np.abs(valid_candidates - coordinate)))
            ]
        candidate_fraction = float(
            _detector_shape_metrics(
                candidate_result, u, duration_ms, detector_sample_count
            )[5]
        )
        candidate_distance = abs(candidate_fraction - target_fraction)
        if (
            candidate_distance + np.finfo(float).eps < current_distance
            and inside_hard_shape_support(candidate_result)
        ):
            result = candidate_result
            current_fraction = candidate_fraction
            current_distance = candidate_distance
            if current_distance <= 0.5 / len(
                _detector_zoh_indices(u, duration_ms, detector_sample_count)
            ):
                break
    result[0] = points[0]
    result[-1] = points[-1]
    return result


class ConditionalTouchGenerator:
    """Learned, seedable touch generator with no runtime donor lookup."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        models: Sequence[_FunctionalModel],
        *,
        metadata: Mapping[str, object],
        artifact_sha256: str | None = None,
    ) -> None:
        if not models:
            raise ConditionalTouchGeneratorError("generator has no fitted models")
        self._models = {
            (model.action, model.orientation_id): model for model in models
        }
        if len(self._models) != len(models):
            raise ConditionalTouchGeneratorError("duplicate fitted model condition")
        self._metadata = dict(metadata)
        self._artifact_sha256 = artifact_sha256

    @property
    def artifact_sha256(self) -> str | None:
        return self._artifact_sha256

    @property
    def metadata(self) -> dict[str, object]:
        result = json.loads(json.dumps(self._metadata, sort_keys=True))
        result["schema_version"] = self.schema_version
        result["artifact_sha256"] = self._artifact_sha256
        return result

    @property
    def training_summary(self) -> dict[str, object]:
        """Return audit-safe fit counts/configuration without raw event IDs."""

        return json.loads(json.dumps(self._metadata, sort_keys=True))

    @property
    def supported_conditions(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._models))

    @classmethod
    def fit_from_raw_rows(
        cls,
        rows: Mapping[str, Sequence[object] | np.ndarray],
        *,
        grid_size: int = 33,
        max_rank: int = 8,
        ridge: float = 1.0e-3,
        minimum_events: int = 3,
        tap_stationary_tolerance_px: float = 1.0e-6,
        training_source_sha256s: Sequence[str] = (),
    ) -> "ConditionalTouchGenerator":
        """Fit compact condition models from flat raw training rows.

        ``event_id`` is used only for grouping during this call.  It is not
        retained in memory or in the saved artifact.
        """

        missing = [name for name in _REQUIRED_ROW_FIELDS if name not in rows]
        if missing:
            raise ConditionalTouchGeneratorError(
                f"raw rows are missing required fields: {', '.join(missing)}"
            )
        if grid_size < 5:
            raise ConditionalTouchGeneratorError("grid_size must be at least 5")
        if max_rank < 0:
            raise ConditionalTouchGeneratorError("max_rank cannot be negative")
        if ridge <= 0.0 or not np.isfinite(ridge):
            raise ConditionalTouchGeneratorError("ridge must be finite and positive")
        if minimum_events < 1:
            raise ConditionalTouchGeneratorError("minimum_events must be positive")
        if tap_stationary_tolerance_px < 0.0:
            raise ConditionalTouchGeneratorError(
                "tap_stationary_tolerance_px cannot be negative"
            )

        arrays = {name: np.asarray(rows[name]) for name in _REQUIRED_ROW_FIELDS}
        lengths = {len(value) for value in arrays.values() if value.ndim == 1}
        if any(value.ndim != 1 for value in arrays.values()) or len(lengths) != 1:
            raise ConditionalTouchGeneratorError(
                "all required raw row fields must be equal-length vectors"
            )
        row_count = next(iter(lengths))
        grouped: dict[tuple[object, str, int], list[int]] = {}
        ignored_unsupported_rows = 0
        for index in range(row_count):
            action = _text(arrays["action"][index]).lower()
            if action not in SUPPORTED_ACTIONS:
                ignored_unsupported_rows += 1
                continue
            event_value = arrays["event_id"][index]
            if isinstance(event_value, np.generic):
                event_value = event_value.item()
            try:
                hash(event_value)
            except TypeError as error:
                raise ConditionalTouchGeneratorError("event_id values must be hashable") from error
            orientation_id = int(arrays["orientation_id"][index])
            grouped.setdefault((event_value, action, orientation_id), []).append(index)

        grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
        by_condition: dict[tuple[str, int], list[_TrainingCurve]] = {}
        rejected_reasons: dict[str, int] = {}
        for (_, action, orientation_id), indices in grouped.items():
            selected = np.asarray(indices, dtype=np.int64)
            try:
                curve = _event_curve(
                    action=action,
                    orientation_id=orientation_id,
                    t_ms=np.asarray(arrays["t_ms"][selected], dtype=np.float64),
                    x_px=np.asarray(arrays["x_px"][selected], dtype=np.float64),
                    y_px=np.asarray(arrays["y_px"][selected], dtype=np.float64),
                    pressure=np.asarray(
                        arrays["pressure"][selected], dtype=np.float64
                    ),
                    android_action=np.asarray(
                        arrays["android_action"][selected], dtype=np.int64
                    ),
                    grid=grid,
                    tap_stationary_tolerance_px=float(
                        tap_stationary_tolerance_px
                    ),
                )
            except (ConditionalTouchGeneratorError, ValueError) as error:
                reason = str(error)
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue
            by_condition.setdefault((action, orientation_id), []).append(curve)

        sparse = {
            f"{action}|{orientation}": len(curves)
            for (action, orientation), curves in by_condition.items()
            if len(curves) < minimum_events
        }
        if sparse:
            raise ConditionalTouchGeneratorError(
                "conditions below minimum_events: "
                + ", ".join(f"{key}={value}" for key, value in sorted(sparse.items()))
            )
        models = [
            _fit_condition(
                curves,
                grid_size=int(grid_size),
                max_rank=int(max_rank),
                ridge=float(ridge),
            )
            for _, curves in sorted(by_condition.items())
        ]
        if not models:
            detail = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(rejected_reasons.items())
            )
            raise ConditionalTouchGeneratorError(
                "no valid supported training events" + (f" ({detail})" if detail else "")
            )
        condition_counts = {
            f"{model.action}|{model.orientation_id}": model.training_event_count
            for model in models
        }
        stationary_probabilities = {
            f"{model.action}|{model.orientation_id}": model.stationary_probability
            for model in models
            if model.action == "tap"
        }
        increment_parameter_audit: dict[str, object] = {}
        pressure_parameter_audit: dict[str, object] = {}
        shape_parameter_audit: dict[str, object] = {}
        generation_strategies: dict[str, str] = {}
        tap_exact_endpoint_support: dict[str, object] = {}
        for model in models:
            key = f"{model.action}|{model.orientation_id}"
            if model.action == "tap":
                generation_strategies[key] = (
                    "exact_DOWN_UP_stationary_or_train_fitted_tap_increment"
                )
                condition_curves = by_condition[
                    (model.action, model.orientation_id)
                ]
                equal_stationary_count = sum(
                    curve.tap_endpoint_equal and curve.tap_stationary
                    for curve in condition_curves
                )
                equal_moved_count = sum(
                    curve.tap_endpoint_equal and not curve.tap_stationary
                    for curve in condition_curves
                )
                unequal_endpoint_count = sum(
                    not curve.tap_endpoint_equal for curve in condition_curves
                )
                tap_exact_endpoint_support[key] = {
                    "equal_endpoint_stationary_count": int(equal_stationary_count),
                    "equal_endpoint_moved_count": int(equal_moved_count),
                    "unequal_endpoint_count": int(unequal_endpoint_count),
                    "conditional_equal_endpoint_moved_probability": (
                        float(equal_moved_count / (equal_stationary_count + equal_moved_count))
                        if equal_stationary_count + equal_moved_count
                        else 0.0
                    ),
                    "generation_policy": (
                        "start=end selects observed stationary support; unequal endpoints select fitted local moving-tap increment support"
                    ),
                }
            elif model.increment_model is not None:
                generation_strategies[key] = (
                    "train_fitted_row_increment_endpoint_bridge"
                )
            else:
                generation_strategies[key] = (
                    "functional_regression_stochastic_pca_with_shape_support"
                )
            pressure_model = model.pressure_model
            pressure_parameter_audit[key] = {
                "exact_zero_probability": pressure_model.exact_zero_probability,
                "exact_one_probability": pressure_model.exact_one_probability,
                "interior_mean_quantiles": pressure_model.interior_mean_quantiles.tolist(),
                "training_event_count": pressure_model.training_event_count,
                "exact_zero_event_count": pressure_model.exact_zero_event_count,
                "exact_one_event_count": pressure_model.exact_one_event_count,
                "interior_event_count": pressure_model.interior_event_count,
                "exact_boundary_values_preserved": True,
            }
            if model.increment_model is not None:
                increment = model.increment_model
                increment_parameter_audit[key] = {
                    "phase_bin_count": increment.phase_bin_count,
                    "amplitude_feature_center": increment.amplitude_feature_center.tolist(),
                    "amplitude_feature_scale": increment.amplitude_feature_scale.tolist(),
                    "amplitude_beta": increment.amplitude_beta.tolist(),
                    "amplitude_residual_quantiles": increment.amplitude_residual_quantiles.tolist(),
                    "initial_move_probability": increment.initial_move_probability.tolist(),
                    "transition_probability": increment.transition_probability.tolist(),
                    "unconditional_move_probability": increment.unconditional_move_probability.tolist(),
                    "normalized_increment_mean": increment.normalized_increment_mean.tolist(),
                    "innovation_transform": increment.innovation_transform.tolist(),
                    "autoregression": increment.autoregression.tolist(),
                    "innovation_radius_quantiles": increment.innovation_radius_quantiles.tolist(),
                    "phase_active_counts": increment.phase_active_counts.tolist(),
                    "phase_interval_counts": increment.phase_interval_counts.tolist(),
                    "transition_counts": increment.transition_counts.tolist(),
                    "training_event_count": increment.training_event_count,
                    "training_interval_count": increment.training_interval_count,
                    "active_interval_count": increment.active_interval_count,
                    "amplitude_event_count": increment.amplitude_event_count,
                    "coordinate_quantization_applied": model.action != "tap",
                    "coordinate_quantization_model": (
                        "disabled for tap to preserve monotone local progress"
                        if model.action == "tap"
                        else "train-fitted detector-grid event fraction target"
                    ),
                    "coordinate_pause_definition_px": float(
                        tap_stationary_tolerance_px if model.action == "tap" else 0.0
                    ),
                    "endpoint_bridge": (
                        "pause-preserving monotone progress plus finite constant lateral scaling"
                        if model.action == "tap"
                        else "active-interval cumulative global bridge"
                    ),
                }
            if model.shape_model is not None:
                shape = model.shape_model
                shape_parameter_audit[key] = {
                    "path_chord_quantiles": shape.path_chord_quantiles.tolist(),
                    "lateral_excursion_quantiles": shape.lateral_excursion_quantiles.tolist(),
                    "lateral_rms_quantiles": shape.lateral_rms_quantiles.tolist(),
                    "longitudinal_excursion_quantiles": shape.longitudinal_excursion_quantiles.tolist(),
                    "longitudinal_reversal_quantiles": shape.longitudinal_reversal_quantiles.tolist(),
                    "half_pixel_fraction_quantiles": shape.half_pixel_fraction_quantiles.tolist(),
                    "metric_order": [
                        "path_chord",
                        "lateral_max_normalized",
                        "lateral_rms_normalized",
                        "longitudinal_excursion",
                        "reverse_step_fraction",
                        "detector_half_pixel_scalar_fraction",
                    ],
                    "metric_copula_transform": shape.metric_copula_transform.tolist(),
                    "joint_target_sampling": (
                        "gaussian rank copula with smooth monotone upper-tail compression to train q90"
                    ),
                    "calibrated_metrics": (
                        [
                            "lateral_max_normalized_constant_scale",
                        ]
                        if model.action == "tap"
                        else [
                            "path_chord",
                            "lateral_max_normalized",
                            "lateral_rms_normalized",
                            "detector_half_pixel_scalar_fraction",
                        ]
                    ),
                    "hard_support_only_metrics": (
                        [
                            "monotone_longitudinal_progress",
                            "lateral_rms_normalized",
                        ]
                        if model.action == "tap"
                        else [
                            "longitudinal_excursion",
                            "reverse_step_fraction",
                        ]
                    ),
                    "hard_support_quantile": shape.support_quantile,
                    "path_chord_hard_max": shape.path_chord_hard_max,
                    "lateral_excursion_hard_max": shape.lateral_excursion_hard_max,
                    "lateral_rms_hard_max": shape.lateral_rms_hard_max,
                    "longitudinal_excursion_hard_max": shape.longitudinal_excursion_hard_max,
                    "longitudinal_reversal_hard_max": shape.longitudinal_reversal_hard_max,
                    "coordinate_lattice_quantum_px": shape.coordinate_lattice_quantum_px,
                    "coordinate_lattice_probability": shape.coordinate_lattice_probability,
                    "coordinate_lattice_count": shape.coordinate_lattice_count,
                    "coordinate_value_count": shape.coordinate_value_count,
                    "coordinate_fraction_modes": shape.coordinate_fraction_modes.tolist(),
                    "coordinate_fraction_probabilities": shape.coordinate_fraction_probabilities.tolist(),
                    "coordinate_fraction_counts": shape.coordinate_fraction_counts.tolist(),
                    "coordinate_lattice_fit": {
                        "fraction_definition": "exact coordinate modulo 1 px",
                        "selection": "largest empirical sorted log-count elbow",
                        "manual_lattice_quantum_assumed": False,
                    },
                    "training_event_count": shape.training_event_count,
                    "support_source": "train_event_empirical_inverse_cdf",
                    "pause_schedule_support_conditioning": (
                        {
                            "source": "train_fitted_increment Markov states and positive progress weights",
                            "longitudinal_policy": "monotone pause-preserving cumulative support",
                            "lateral_policy": "single finite scalar normalization; no power transform",
                        }
                        if model.action == "tap"
                        else {
                            "metric": "max_abs_longitudinal_progress_minus_normalized_time",
                            "hard_max_source": "train_empirical_q90",
                            "maximum_rejection_attempts": MAX_SUPPORT_RESAMPLE_ATTEMPTS,
                            "numeric_fallback": "exact chronological chord",
                        }
                    ),
                }
        metadata = {
            "fit_config": {
                "grid_size": int(grid_size),
                "max_rank": int(max_rank),
                "ridge": float(ridge),
                "minimum_events": int(minimum_events),
                "tap_stationary_tolerance_px": float(
                    tap_stationary_tolerance_px
                ),
            },
            "training_source_sha256s": sorted(set(training_source_sha256s)),
            "accepted_event_counts": condition_counts,
            "accepted_event_count": int(sum(condition_counts.values())),
            "rejected_event_count": int(sum(rejected_reasons.values())),
            "rejected_event_reasons": rejected_reasons,
            "ignored_unsupported_row_count": int(ignored_unsupported_rows),
            "tap_stationary_probabilities": stationary_probabilities,
            "generation_strategies": generation_strategies,
            "increment_parameter_audit": increment_parameter_audit,
            "tap_exact_endpoint_support": tap_exact_endpoint_support,
            "pressure_parameter_audit": pressure_parameter_audit,
            "shape_parameter_audit": shape_parameter_audit,
            "raw_event_ids_retained": False,
            "runtime_donor_lookup_used": False,
            "generator_source_sha256": IMPORT_GENERATOR_SOURCE_SHA256,
            "android_touch_observation_source_sha256": (
                IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
            ),
            "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
        }
        return cls(models, metadata=metadata)

    def _serialize(self) -> dict[str, np.ndarray]:
        manifest_models: list[dict[str, object]] = []
        arrays: dict[str, np.ndarray] = {}
        for index, condition in enumerate(sorted(self._models)):
            model = self._models[condition]
            prefix = f"model_{index:03d}"
            manifest_models.append(
                {
                    "prefix": prefix,
                    "action": model.action,
                    "orientation_id": model.orientation_id,
                    "grid_size": model.grid_size,
                    "stationary_probability": model.stationary_probability,
                    "training_event_count": model.training_event_count,
                    "continuous_event_count": model.continuous_event_count,
                    "median_update_rate_hz": model.median_update_rate_hz,
                    "coordinate_strategy": (
                        "exact_endpoint_stationary_or_tap_increment_support"
                        if model.action == "tap"
                        else (
                            "row_increment_endpoint_bridge"
                            if model.increment_model is not None
                            else "functional_pca"
                        )
                    ),
                    "pressure": {
                        "exact_zero_probability": model.pressure_model.exact_zero_probability,
                        "exact_one_probability": model.pressure_model.exact_one_probability,
                        "training_event_count": model.pressure_model.training_event_count,
                        "exact_zero_event_count": model.pressure_model.exact_zero_event_count,
                        "exact_one_event_count": model.pressure_model.exact_one_event_count,
                        "interior_event_count": model.pressure_model.interior_event_count,
                    },
                    "shape_training_event_count": (
                        None
                        if model.shape_model is None
                        else model.shape_model.training_event_count
                    ),
                    "shape_coordinate_lattice": (
                        None
                        if model.shape_model is None
                        else {
                            "quantum_px": model.shape_model.coordinate_lattice_quantum_px,
                            "probability": model.shape_model.coordinate_lattice_probability,
                            "count": model.shape_model.coordinate_lattice_count,
                            "value_count": model.shape_model.coordinate_value_count,
                            "support_quantile": model.shape_model.support_quantile,
                            "path_chord_hard_max": model.shape_model.path_chord_hard_max,
                            "lateral_excursion_hard_max": model.shape_model.lateral_excursion_hard_max,
                            "lateral_rms_hard_max": model.shape_model.lateral_rms_hard_max,
                            "longitudinal_excursion_hard_max": model.shape_model.longitudinal_excursion_hard_max,
                            "longitudinal_reversal_hard_max": model.shape_model.longitudinal_reversal_hard_max,
                        }
                    ),
                }
            )
            arrays[f"{prefix}_feature_center"] = model.feature_center
            arrays[f"{prefix}_feature_scale"] = model.feature_scale
            arrays[f"{prefix}_channel_scale"] = model.channel_scale
            arrays[f"{prefix}_beta"] = model.beta
            arrays[f"{prefix}_noise_loadings"] = model.noise_loadings
            arrays[f"{prefix}_pressure_interior_mean_quantiles"] = (
                model.pressure_model.interior_mean_quantiles
            )
            if model.increment_model is not None:
                increment = model.increment_model
                manifest_models[-1]["increment"] = {
                    "phase_bin_count": increment.phase_bin_count,
                    "training_event_count": increment.training_event_count,
                    "training_interval_count": increment.training_interval_count,
                    "active_interval_count": increment.active_interval_count,
                    "amplitude_event_count": increment.amplitude_event_count,
                }
                for suffix in (
                    "amplitude_feature_center",
                    "amplitude_feature_scale",
                    "amplitude_beta",
                    "amplitude_residual_quantiles",
                    "initial_move_probability",
                    "transition_probability",
                    "unconditional_move_probability",
                    "normalized_increment_mean",
                    "innovation_transform",
                    "autoregression",
                    "innovation_radius_quantiles",
                    "phase_active_counts",
                    "phase_interval_counts",
                    "transition_counts",
                ):
                    arrays[f"{prefix}_increment_{suffix}"] = np.asarray(
                        getattr(increment, suffix)
                    )
            if model.shape_model is not None:
                shape = model.shape_model
                arrays[f"{prefix}_shape_path_chord_quantiles"] = (
                    shape.path_chord_quantiles
                )
                arrays[f"{prefix}_shape_lateral_excursion_quantiles"] = (
                    shape.lateral_excursion_quantiles
                )
                arrays[f"{prefix}_shape_lateral_rms_quantiles"] = (
                    shape.lateral_rms_quantiles
                )
                arrays[f"{prefix}_shape_longitudinal_excursion_quantiles"] = (
                    shape.longitudinal_excursion_quantiles
                )
                arrays[f"{prefix}_shape_longitudinal_reversal_quantiles"] = (
                    shape.longitudinal_reversal_quantiles
                )
                arrays[f"{prefix}_shape_half_pixel_fraction_quantiles"] = (
                    shape.half_pixel_fraction_quantiles
                )
                arrays[f"{prefix}_shape_metric_copula_transform"] = (
                    shape.metric_copula_transform
                )
                arrays[f"{prefix}_shape_coordinate_fraction_modes"] = (
                    shape.coordinate_fraction_modes
                )
                arrays[f"{prefix}_shape_coordinate_fraction_probabilities"] = (
                    shape.coordinate_fraction_probabilities
                )
                arrays[f"{prefix}_shape_coordinate_fraction_counts"] = (
                    shape.coordinate_fraction_counts
                )
        manifest = {
            "schema_version": self.schema_version,
            "generator_source_sha256": IMPORT_GENERATOR_SOURCE_SHA256,
            "android_touch_observation_source_sha256": (
                IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
            ),
            "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
            "metadata": self._metadata,
            "models": manifest_models,
        }
        arrays["manifest_json"] = np.asarray(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        return arrays

    def save(self, path: str | Path) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.savez_compressed(handle, **self._serialize())
        self._artifact_sha256 = _sha256_file(output_path)
        return self._artifact_sha256

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalTouchGenerator":
        artifact_path = Path(path)
        with np.load(artifact_path, allow_pickle=False) as archive:
            manifest = json.loads(str(np.asarray(archive["manifest_json"]).item()))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ConditionalTouchGeneratorError(
                    f"unsupported generator schema {manifest.get('schema_version')!r}"
                )
            expected_source_binding = {
                "generator_source_sha256": IMPORT_GENERATOR_SOURCE_SHA256,
                "android_touch_observation_source_sha256": (
                    IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
                ),
                "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
            }
            for field, expected in expected_source_binding.items():
                if manifest.get(field) != expected:
                    raise ConditionalTouchGeneratorError(
                        f"generator artifact {field} does not match imported code"
                    )
            metadata = manifest.get("metadata")
            if not isinstance(metadata, dict):
                raise ConditionalTouchGeneratorError(
                    "generator artifact metadata must be an object"
                )
            for field, expected in expected_source_binding.items():
                if metadata.get(field) != expected:
                    raise ConditionalTouchGeneratorError(
                        f"generator artifact metadata {field} does not match "
                        "imported code"
                    )
            models: list[_FunctionalModel] = []
            for item in manifest["models"]:
                prefix = item["prefix"]
                increment_item = item.get("increment")
                increment_model = None
                if increment_item is not None:
                    increment_arrays = {}
                    for suffix in (
                        "amplitude_feature_center",
                        "amplitude_feature_scale",
                        "amplitude_beta",
                        "amplitude_residual_quantiles",
                        "initial_move_probability",
                        "transition_probability",
                        "unconditional_move_probability",
                        "normalized_increment_mean",
                        "innovation_transform",
                        "autoregression",
                        "innovation_radius_quantiles",
                        "phase_active_counts",
                        "phase_interval_counts",
                        "transition_counts",
                    ):
                        increment_arrays[suffix] = np.asarray(
                            archive[f"{prefix}_increment_{suffix}"]
                        )
                    increment_model = _IncrementModel(
                        phase_bin_count=int(increment_item["phase_bin_count"]),
                        amplitude_feature_center=np.asarray(
                            increment_arrays["amplitude_feature_center"],
                            dtype=np.float64,
                        ),
                        amplitude_feature_scale=np.asarray(
                            increment_arrays["amplitude_feature_scale"],
                            dtype=np.float64,
                        ),
                        amplitude_beta=np.asarray(
                            increment_arrays["amplitude_beta"], dtype=np.float64
                        ),
                        amplitude_residual_quantiles=np.asarray(
                            increment_arrays["amplitude_residual_quantiles"],
                            dtype=np.float64,
                        ),
                        initial_move_probability=np.asarray(
                            increment_arrays["initial_move_probability"],
                            dtype=np.float64,
                        ),
                        transition_probability=np.asarray(
                            increment_arrays["transition_probability"],
                            dtype=np.float64,
                        ),
                        unconditional_move_probability=np.asarray(
                            increment_arrays["unconditional_move_probability"],
                            dtype=np.float64,
                        ),
                        normalized_increment_mean=np.asarray(
                            increment_arrays["normalized_increment_mean"],
                            dtype=np.float64,
                        ),
                        innovation_transform=np.asarray(
                            increment_arrays["innovation_transform"],
                            dtype=np.float64,
                        ),
                        autoregression=np.asarray(
                            increment_arrays["autoregression"], dtype=np.float64
                        ),
                        innovation_radius_quantiles=np.asarray(
                            increment_arrays["innovation_radius_quantiles"],
                            dtype=np.float64,
                        ),
                        phase_active_counts=np.asarray(
                            increment_arrays["phase_active_counts"], dtype=np.int64
                        ),
                        phase_interval_counts=np.asarray(
                            increment_arrays["phase_interval_counts"], dtype=np.int64
                        ),
                        transition_counts=np.asarray(
                            increment_arrays["transition_counts"], dtype=np.int64
                        ),
                        training_event_count=int(
                            increment_item["training_event_count"]
                        ),
                        training_interval_count=int(
                            increment_item["training_interval_count"]
                        ),
                        active_interval_count=int(
                            increment_item["active_interval_count"]
                        ),
                        amplitude_event_count=int(
                            increment_item["amplitude_event_count"]
                        ),
                    )
                pressure_item = item["pressure"]
                pressure_model = _PressureModel(
                    exact_zero_probability=float(
                        pressure_item["exact_zero_probability"]
                    ),
                    exact_one_probability=float(
                        pressure_item["exact_one_probability"]
                    ),
                    interior_mean_quantiles=np.asarray(
                        archive[f"{prefix}_pressure_interior_mean_quantiles"],
                        dtype=np.float64,
                    ),
                    training_event_count=int(pressure_item["training_event_count"]),
                    exact_zero_event_count=int(
                        pressure_item["exact_zero_event_count"]
                    ),
                    exact_one_event_count=int(pressure_item["exact_one_event_count"]),
                    interior_event_count=int(pressure_item["interior_event_count"]),
                )
                shape_model = None
                if item.get("shape_training_event_count") is not None:
                    lattice_item = item["shape_coordinate_lattice"]
                    shape_model = _ShapeModel(
                        path_chord_quantiles=np.asarray(
                            archive[f"{prefix}_shape_path_chord_quantiles"],
                            dtype=np.float64,
                        ),
                        lateral_excursion_quantiles=np.asarray(
                            archive[f"{prefix}_shape_lateral_excursion_quantiles"],
                            dtype=np.float64,
                        ),
                        lateral_rms_quantiles=np.asarray(
                            archive[f"{prefix}_shape_lateral_rms_quantiles"],
                            dtype=np.float64,
                        ),
                        longitudinal_excursion_quantiles=np.asarray(
                            archive[
                                f"{prefix}_shape_longitudinal_excursion_quantiles"
                            ],
                            dtype=np.float64,
                        ),
                        longitudinal_reversal_quantiles=np.asarray(
                            archive[
                                f"{prefix}_shape_longitudinal_reversal_quantiles"
                            ],
                            dtype=np.float64,
                        ),
                        half_pixel_fraction_quantiles=np.asarray(
                            archive[
                                f"{prefix}_shape_half_pixel_fraction_quantiles"
                            ],
                            dtype=np.float64,
                        ),
                        metric_copula_transform=np.asarray(
                            archive[f"{prefix}_shape_metric_copula_transform"],
                            dtype=np.float64,
                        ),
                        support_quantile=float(lattice_item["support_quantile"]),
                        path_chord_hard_max=float(
                            lattice_item["path_chord_hard_max"]
                        ),
                        lateral_excursion_hard_max=float(
                            lattice_item["lateral_excursion_hard_max"]
                        ),
                        lateral_rms_hard_max=float(
                            lattice_item["lateral_rms_hard_max"]
                        ),
                        longitudinal_excursion_hard_max=float(
                            lattice_item["longitudinal_excursion_hard_max"]
                        ),
                        longitudinal_reversal_hard_max=float(
                            lattice_item["longitudinal_reversal_hard_max"]
                        ),
                        coordinate_lattice_quantum_px=float(
                            lattice_item["quantum_px"]
                        ),
                        coordinate_lattice_probability=float(
                            lattice_item["probability"]
                        ),
                        coordinate_lattice_count=int(lattice_item["count"]),
                        coordinate_value_count=int(lattice_item["value_count"]),
                        coordinate_fraction_modes=np.asarray(
                            archive[f"{prefix}_shape_coordinate_fraction_modes"],
                            dtype=np.float64,
                        ),
                        coordinate_fraction_probabilities=np.asarray(
                            archive[
                                f"{prefix}_shape_coordinate_fraction_probabilities"
                            ],
                            dtype=np.float64,
                        ),
                        coordinate_fraction_counts=np.asarray(
                            archive[f"{prefix}_shape_coordinate_fraction_counts"],
                            dtype=np.int64,
                        ),
                        training_event_count=int(item["shape_training_event_count"]),
                    )
                models.append(
                    _FunctionalModel(
                        action=str(item["action"]),
                        orientation_id=int(item["orientation_id"]),
                        grid_size=int(item["grid_size"]),
                        feature_center=np.asarray(
                            archive[f"{prefix}_feature_center"], dtype=np.float64
                        ),
                        feature_scale=np.asarray(
                            archive[f"{prefix}_feature_scale"], dtype=np.float64
                        ),
                        channel_scale=np.asarray(
                            archive[f"{prefix}_channel_scale"], dtype=np.float64
                        ),
                        beta=np.asarray(
                            archive[f"{prefix}_beta"], dtype=np.float64
                        ),
                        noise_loadings=np.asarray(
                            archive[f"{prefix}_noise_loadings"], dtype=np.float64
                        ),
                        stationary_probability=float(item["stationary_probability"]),
                        training_event_count=int(item["training_event_count"]),
                        continuous_event_count=int(item["continuous_event_count"]),
                        median_update_rate_hz=float(item["median_update_rate_hz"]),
                        increment_model=increment_model,
                        pressure_model=pressure_model,
                        shape_model=shape_model,
                    )
                )
        return cls(
            models,
            metadata=metadata,
            artifact_sha256=_sha256_file(artifact_path),
        )

    def generate(
        self,
        *,
        action: str,
        orientation_id: int,
        start_xy_px: Sequence[float],
        end_xy_px: Sequence[float],
        direction: str | None,
        seed: int,
        t_ms: Iterable[float] | None = None,
        duration_ms: float | None = None,
        sample_count: int | None = None,
        minimum_residual_scale: float = 0.0,
        detector_sample_count: int | None = None,
    ) -> GeneratedTouch:
        action_text = str(action).lower()
        condition = (action_text, int(orientation_id))
        model = self._models.get(condition)
        if model is None:
            raise ConditionalTouchGeneratorError(
                f"no fitted model for action={action_text!r}, "
                f"orientation_id={orientation_id}"
            )
        start = _as_xy("start_xy_px", start_xy_px)
        end = _as_xy("end_xy_px", end_xy_px)
        width_px, height_px = screen_dimensions_for_orientation(orientation_id)
        for name, point in (("start", start), ("end", end)):
            if not (0.0 <= point[0] <= width_px and 0.0 <= point[1] <= height_px):
                raise ConditionalTouchGeneratorError(
                    f"{name} coordinate leaves the physical screen"
                )
        chord = end - start
        chord_length = float(np.linalg.norm(chord))
        realized_direction = _direction8(float(chord[0]), float(chord[1]))
        if action_text == "tap":
            tap_endpoint_equal = bool(np.array_equal(start, end))
            if tap_endpoint_equal:
                if direction not in (None, STATIONARY):
                    raise ConditionalTouchGeneratorError(
                        "equal-endpoint tap direction must be None or 'stationary'"
                    )
            elif direction not in (None, realized_direction):
                raise ConditionalTouchGeneratorError(
                    "moving tap direction must be None or match its endpoint sector"
                )
        else:
            if direction not in DIRECTION8:
                raise ConditionalTouchGeneratorError(
                    "scroll/swipe direction must be one of the eight sectors"
                )
            if chord_length <= 1.0e-9:
                raise ConditionalTouchGeneratorError(
                    "scroll/swipe endpoints must have nonzero displacement"
                )
            if direction != realized_direction:
                raise ConditionalTouchGeneratorError(
                    f"direction {direction!r} does not match endpoint sector "
                    f"{realized_direction!r}"
                )
        if not np.isfinite(minimum_residual_scale) or not (
            0.0 <= minimum_residual_scale <= 1.0
        ):
            raise ConditionalTouchGeneratorError(
                "minimum_residual_scale must lie in [0, 1]"
            )
        if detector_sample_count is not None and int(detector_sample_count) < 2:
            raise ConditionalTouchGeneratorError(
                "detector_sample_count must be at least 2"
            )

        if t_ms is not None:
            timeline = _validate_timeline(t_ms)
            actual_duration_ms = float(timeline[-1] - timeline[0])
            if duration_ms is not None and not np.isclose(
                float(duration_ms), actual_duration_ms, rtol=0.0, atol=1.0e-6
            ):
                raise ConditionalTouchGeneratorError(
                    "duration_ms does not match the supplied t_ms span"
                )
            if sample_count is not None and int(sample_count) != len(timeline):
                raise ConditionalTouchGeneratorError(
                    "sample_count does not match the supplied t_ms"
                )
        else:
            if duration_ms is None or not np.isfinite(duration_ms) or duration_ms <= 0.0:
                raise ConditionalTouchGeneratorError(
                    "positive duration_ms is required when t_ms is omitted"
                )
            actual_duration_ms = float(duration_ms)
            if sample_count is None:
                sample_count = max(
                    2,
                    int(
                        round(
                            actual_duration_ms
                            * model.median_update_rate_hz
                            / 1000.0
                        )
                    )
                    + 1,
                )
            if int(sample_count) < 2:
                raise ConditionalTouchGeneratorError("sample_count must be at least 2")
            timeline = np.linspace(
                0.0, actual_duration_ms, int(sample_count), dtype=np.float64
            )

        rng = np.random.default_rng(int(seed))
        # Train data has no moving equal-endpoint tap loops.  Equal requested
        # endpoints therefore select the stationary support, while unequal
        # requested endpoints use the separately fitted real moving-tap
        # residual model and still land exactly on DOWN/UP.
        tap_stationary = action_text == "tap" and bool(
            np.array_equal(start, end)
        )
        raw_features = [np.log(actual_duration_ms)]
        if action_text != "tap":
            raw_features.append(np.log(chord_length))
        standardized = (
            np.asarray(raw_features, dtype=np.float64) - model.feature_center
        ) / model.feature_scale
        design = np.concatenate((np.asarray([1.0]), standardized))
        response = design @ model.beta
        if len(model.noise_loadings):
            response = response + rng.standard_normal(
                len(model.noise_loadings)
            ) @ model.noise_loadings
        curve = response.reshape(model.grid_size, 3) * model.channel_scale[None, :]
        curve[0, :2] = 0.0
        curve[-1, :2] = 0.0
        u = (timeline - timeline[0]) / actual_duration_ms
        lattice_target_fraction: float | None = None
        fit_grid = np.linspace(0.0, 1.0, model.grid_size, dtype=np.float64)
        if tap_stationary:
            baseline = np.repeat(start[None, :], len(timeline), axis=0)
            residual = np.zeros_like(baseline)
        else:
            tangent = chord / chord_length
            normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
            # Taps and swipes are short enough that row-wise innovation plus
            # endpoint bridging creates visible high-frequency corners on the
            # detector grid.  Their fitted functional curve is already a
            # compact, low-frequency train-real residual model.  Use it
            # directly around the exact requested chord.  Scroll keeps the
            # increment/pause model that passed the endpoint-control smoke.
            smooth_exact_curve = action_text in {"tap", "swipe"}
            if smooth_exact_curve:
                coordinate_curve = np.column_stack(
                    (
                        np.interp(u, fit_grid, curve[:, 0]),
                        np.interp(u, fit_grid, curve[:, 1]),
                    )
                )
                local_curve = np.column_stack(
                    (u, np.zeros(len(u), dtype=np.float64))
                ) + coordinate_curve
                local_curve[0] = (0.0, 0.0)
                local_curve[-1] = (1.0, 0.0)
                if model.shape_model is not None:
                    local_curve = _apply_smooth_global_shape_support(
                        local_curve,
                        u,
                        actual_duration_ms,
                        model.shape_model,
                        detector_sample_count,
                    )
            elif model.increment_model is not None:
                if action_text == "tap":
                    local_curve = _sample_increment_local_curve(
                        model.increment_model,
                        duration_ms=actual_duration_ms,
                        chord_length_px=chord_length,
                        u=u,
                        rng=rng,
                    )
                else:
                    local_curve = None
                    for _ in range(MAX_SUPPORT_RESAMPLE_ATTEMPTS):
                        candidate_curve = _sample_increment_local_curve(
                            model.increment_model,
                            duration_ms=actual_duration_ms,
                            chord_length_px=chord_length,
                            u=u,
                            rng=rng,
                        )
                        if model.shape_model is None:
                            local_curve = candidate_curve
                            break
                        support_curve = _pause_preserving_support_curve(
                            candidate_curve
                        )
                        support_longitudinal = _detector_shape_metrics(
                            support_curve,
                            u,
                            actual_duration_ms,
                            detector_sample_count,
                        )[3]
                        if float(support_longitudinal) <= float(
                            np.nextafter(
                                model.shape_model.longitudinal_excursion_hard_max,
                                np.inf,
                            )
                        ):
                            local_curve = candidate_curve
                            break
                    if local_curve is None:
                        # Numeric safety for conditioning combinations outside
                        # the fitted row-count support: the chord is legal.
                        local_curve = np.column_stack(
                            (u, np.zeros(len(u), dtype=np.float64))
                        )
            else:
                if action_text == "tap":
                    raise ConditionalTouchGeneratorError(
                        "tap condition has no fitted unequal-endpoint support"
                    )
                coordinate_curve = np.column_stack(
                    (
                        np.interp(u, fit_grid, curve[:, 0]),
                        np.interp(u, fit_grid, curve[:, 1]),
                    )
                )
                local_curve = np.column_stack((u, np.zeros(len(u)))) + coordinate_curve
            if model.shape_model is not None and not smooth_exact_curve:
                if action_text == "tap":
                    local_curve, lattice_target_fraction = (
                        _apply_tap_increment_support(
                            local_curve,
                            u,
                            actual_duration_ms,
                            model.shape_model,
                            rng,
                            detector_sample_count,
                        )
                    )
                else:
                    local_curve, lattice_target_fraction = (
                        _apply_train_fitted_shape_support(
                            local_curve,
                            u,
                            actual_duration_ms,
                            model.shape_model,
                            rng,
                            detector_sample_count,
                        )
                    )
            screen_support = (
                np.column_stack((u, np.zeros(len(u), dtype=np.float64)))
                if smooth_exact_curve
                else _pause_preserving_support_curve(local_curve)
            )
            baseline = start[None, :] + screen_support[:, :1] * chord[None, :]
            local_residual = local_curve - screen_support
            residual = chord_length * (
                local_residual[:, :1] * tangent[None, :]
                + local_residual[:, 1:] * normal[None, :]
            )
        residual[0] = 0.0
        residual[-1] = 0.0
        residual_scale = _maximum_residual_scale(
            baseline,
            residual,
            width_px=width_px,
            height_px=height_px,
        )
        if residual_scale + 1.0e-12 < minimum_residual_scale:
            raise ConditionalTouchGeneratorError(
                "learned residual cannot fit the screen without excessive scaling"
            )
        points = baseline + residual_scale * residual
        points[0] = start
        points[-1] = end
        if (
            action_text != "tap"
            and model.shape_model is not None
            and not smooth_exact_curve
        ):
            if lattice_target_fraction is None:
                raise ConditionalTouchGeneratorError(
                    "shape calibration did not produce a lattice target"
                )
            points = _apply_train_fitted_coordinate_lattice(
                points,
                model.shape_model,
                rng,
                width_px=width_px,
                height_px=height_px,
                u=u,
                duration_ms=actual_duration_ms,
                target_fraction=lattice_target_fraction,
                detector_sample_count=detector_sample_count,
            )
            points[0] = start
            points[-1] = end
        if (
            np.any(points[:, 0] < 0.0)
            or np.any(points[:, 0] > width_px)
            or np.any(points[:, 1] < 0.0)
            or np.any(points[:, 1] > height_px)
        ):
            raise ConditionalTouchGeneratorError(
                "generated coordinates leave the screen after residual scaling"
            )

        pressure_logit = np.interp(u, fit_grid, curve[:, 2])
        pressure = _pressure_from_train_fitted_mean(
            pressure_logit,
            model.pressure_model,
            rng,
        )
        frame_index, frame_end = _frame_lifecycle(timeline)
        android_action = np.full(len(timeline), ACTION_MOVE, dtype=np.int64)
        android_action[0] = ACTION_DOWN
        android_action[-1] = ACTION_UP
        return GeneratedTouch(
            action=action_text,
            orientation_id=int(orientation_id),
            requested_direction=direction,
            realized_direction=realized_direction,
            t_ms=timeline,
            x_px=points[:, 0],
            y_px=points[:, 1],
            pressure=pressure,
            pointer_id=np.zeros(len(timeline), dtype=np.int64),
            android_action=android_action,
            frame_index=frame_index,
            frame_end=frame_end,
            residual_scale=float(residual_scale),
            tap_stationary_branch=tap_stationary,
        )


__all__ = [
    "ConditionalTouchGenerator",
    "ConditionalTouchGeneratorError",
    "DIRECTION8",
    "GeneratedTouch",
    "IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256",
    "IMPORT_GENERATOR_SOURCE_SHA256",
    "IMPORT_SOURCE_FINGERPRINT_SHA256",
    "SCHEMA_VERSION",
    "STATIONARY",
    "SUPPORTED_ACTIONS",
]
