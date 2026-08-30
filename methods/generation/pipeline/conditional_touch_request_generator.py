from __future__ import annotations

"""Compact train-fitted endpoint requests for conditional scroll generation.

The runtime artifact contains aggregate regression and inverse-CDF parameters
only.  Raw event identifiers and donor trajectories are used transiently while
fitting and are never serialized or consulted during generation.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:  # NormalDist was added in Python 3.8; production also runs on 3.7.
    from statistics import NormalDist
except ImportError:  # pragma: no cover - selected by the interpreter version.
    NormalDist = None  # type: ignore

from .android_touch_observation import (
    ACTION_DOWN,
    ACTION_MASK,
    ACTION_UP,
    screen_dimensions_for_orientation,
)


SCHEMA_VERSION = "conditional-touch-request-generator-v2-tap-full-pair-source-bound"
SUPPORTED_ACTIONS = ("scroll", "swipe", "tap")
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
RESIDUAL_QUANTILE_COUNT = 129
_NORMAL = NormalDist() if NormalDist is not None else None
_REQUIRED_ROW_FIELDS = (
    "event_id",
    "action",
    "orientation_id",
    "t_ms",
    "x_px",
    "y_px",
    "android_action",
)


class ConditionalTouchRequestGeneratorError(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedTouchRequest:
    action: str
    orientation_id: int
    direction: str
    start_xy_px: tuple[float, float]
    end_xy_px: tuple[float, float]
    duration_ms: float
    distance_px: float
    angle_rad: float
    available_distance_px: float
    conditional_support_probability: float
    feature_outside_training_range: bool
    seed: int
    stationary: bool
    stationary_probability: float
    endpoint_quantization_px: float


@dataclass(frozen=True)
class _ConditionModel:
    action: str
    orientation_id: int
    direction: str
    feature_center: np.ndarray
    feature_scale: np.ndarray
    feature_minimum: np.ndarray
    feature_maximum: np.ndarray
    angle_beta: np.ndarray
    distance_beta: np.ndarray
    angle_residual_quantiles: np.ndarray
    distance_residual_quantiles: np.ndarray
    gaussian_rank_correlation: float
    training_event_count: int


@dataclass(frozen=True)
class _TrainingEvent:
    action: str
    orientation_id: int
    direction: str
    duration_ms: float
    start_xy_px: np.ndarray
    angle_value: float
    log_distance: float


@dataclass(frozen=True)
class _TapTrainingEvent:
    orientation_id: int
    duration_ms: float
    start_xy_px: np.ndarray
    end_xy_px: np.ndarray
    stationary: bool
    direction: str | None
    angle_value: float | None
    log_distance: float | None


@dataclass(frozen=True)
class _TapStartModel:
    duration_center: float
    duration_scale: float
    duration_minimum: float
    duration_maximum: float
    coordinate_beta: np.ndarray
    coordinate_residual_quantiles: np.ndarray
    gaussian_rank_correlation: float
    coordinate_lattice_probabilities: np.ndarray
    training_event_count: int


@dataclass(frozen=True)
class _TapOrientationModel:
    orientation_id: int
    duration_center: float
    duration_scale: float
    duration_minimum: float
    duration_maximum: float
    stationary_beta: np.ndarray
    stationary_start: _TapStartModel
    moving_start: _TapStartModel
    moving_endpoint_lattice_probabilities: np.ndarray
    training_event_count: int
    stationary_event_count: int
    moving_event_count: int


@dataclass(frozen=True)
class _TapDirectionModel:
    feature_center: np.ndarray
    feature_scale: np.ndarray
    feature_minimum: np.ndarray
    feature_maximum: np.ndarray
    direction_beta: np.ndarray
    direction_counts: np.ndarray
    training_event_count: int


@dataclass(frozen=True)
class _EndpointSample:
    endpoint: np.ndarray
    distance_px: float
    angle_rad: float
    available_distance_px: float
    conditional_support_probability: float
    feature_outside_training_range: bool


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Bound once at import.  A process that imported older bytes cannot save or
# accept an artifact labeled with hashes read from later hot-edited files.
IMPORT_REQUEST_GENERATOR_SOURCE_SHA256 = _sha256_file(Path(__file__).resolve())
IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256 = _sha256_file(
    Path(__file__).resolve().with_name("android_touch_observation.py")
)


def _source_fingerprint_sha256(
    *,
    request_generator_source_sha256: str,
    android_touch_observation_source_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "android_touch_observation.py": android_touch_observation_source_sha256,
            "conditional_touch_request_generator.py": (
                request_generator_source_sha256
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


IMPORT_SOURCE_FINGERPRINT_SHA256 = _source_fingerprint_sha256(
    request_generator_source_sha256=IMPORT_REQUEST_GENERATOR_SOURCE_SHA256,
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
        raise ConditionalTouchRequestGeneratorError(
            "endpoint vector must be finite"
        )
    if float(np.hypot(dx, dy)) <= 0.0:
        raise ConditionalTouchRequestGeneratorError(
            "endpoint vector must be nonzero"
        )
    angle = float(np.arctan2(dy, dx))
    index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
    return DIRECTION8[index]


def _direction_center(direction: str) -> float:
    try:
        return float(DIRECTION8.index(direction) * np.pi / 4.0)
    except ValueError as error:
        raise ConditionalTouchRequestGeneratorError(
            f"unsupported direction {direction!r}"
        ) from error


def _wrap_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _as_start(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ConditionalTouchRequestGeneratorError(
            "start_xy_px must contain two finite coordinates"
        )
    return result


def _ray_screen_distance(
    start: np.ndarray,
    angle: float,
    *,
    width_px: float,
    height_px: float,
) -> float:
    dx = float(np.cos(angle))
    dy = float(np.sin(angle))
    intersections: list[float] = []
    if dx > 0.0:
        intersections.append((width_px - float(start[0])) / dx)
    elif dx < 0.0:
        intersections.append((0.0 - float(start[0])) / dx)
    if dy > 0.0:
        intersections.append((height_px - float(start[1])) / dy)
    elif dy < 0.0:
        intersections.append((0.0 - float(start[1])) / dy)
    finite = [
        float(value)
        for value in intersections
        if np.isfinite(value) and value >= 0.0
    ]
    return min(finite) if finite else 0.0


def _empirical_quantiles(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ConditionalTouchRequestGeneratorError(
            "cannot fit an empty or non-finite residual distribution"
        )
    probabilities = np.linspace(
        0.0, 1.0, RESIDUAL_QUANTILE_COUNT, dtype=np.float64
    )
    return np.asarray(np.quantile(array, probabilities), dtype=np.float64)


def _evaluate_quantiles(quantiles: np.ndarray, probability: float) -> float:
    values = np.asarray(quantiles, dtype=np.float64)
    position = float(np.clip(probability, 0.0, 1.0)) * float(len(values) - 1)
    lower = min(int(np.floor(position)), len(values) - 2)
    fraction = position - lower
    return float(values[lower] + fraction * (values[lower + 1] - values[lower]))


def _quantile_cdf(quantiles: np.ndarray, value: float) -> float:
    """Invert the piecewise-linear inverse CDF at an upper support bound."""

    values = np.asarray(quantiles, dtype=np.float64)
    probabilities = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    if value < float(values[0]):
        return 0.0
    if value >= float(values[-1]):
        return 1.0
    right = int(np.searchsorted(values, value, side="right"))
    left = max(0, right - 1)
    while right < len(values) and values[right] <= values[left]:
        right += 1
    if right >= len(values):
        return float(probabilities[left])
    fraction = float((value - values[left]) / (values[right] - values[left]))
    return float(
        probabilities[left]
        + fraction * (probabilities[right] - probabilities[left])
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and array[order[right]] == array[order[left]]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + right - 1) + 1.0
        left = right
    return ranks


def _normal_cdf(value: float) -> float:
    scalar = float(value)
    if _NORMAL is not None:
        return float(_NORMAL.cdf(scalar))
    # erfc is more accurate than ``1 + erf`` in the negative tail.
    return float(0.5 * math.erfc(-scalar / math.sqrt(2.0)))


def _normal_inverse(probability: float) -> float:
    # Acklam's rational approximation is the Python 3.7 fallback.  Its error
    # is far below the 129-bin empirical-CDF resolution used by this model.
    epsilon = float(np.finfo(np.float64).eps)
    value = float(np.clip(probability, epsilon, 1.0 - epsilon))
    if _NORMAL is not None:
        return float(_NORMAL.inv_cdf(value))
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    )
    lower_break = 0.02425
    upper_break = 1.0 - lower_break
    if value < lower_break:
        q = math.sqrt(-2.0 * math.log(value))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if value > upper_break:
        q = math.sqrt(-2.0 * math.log(1.0 - value))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = value - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _sigmoid(value: float) -> float:
    scalar = float(value)
    if scalar >= 0.0:
        inverse = math.exp(-min(scalar, 745.0))
        return float(1.0 / (1.0 + inverse))
    exponential = math.exp(max(scalar, -745.0))
    return float(exponential / (1.0 + exponential))


def _logit(value: float) -> float:
    epsilon = float(np.finfo(np.float64).eps)
    probability = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(math.log(probability / (1.0 - probability)))


def _gaussian_rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    count = len(first)
    first_probability = (_average_ranks(first) - 0.5) / float(count)
    second_probability = (_average_ranks(second) - 0.5) / float(count)
    first_gaussian = np.asarray(
        [_normal_inverse(value) for value in first_probability],
        dtype=np.float64,
    )
    second_gaussian = np.asarray(
        [_normal_inverse(value) for value in second_probability],
        dtype=np.float64,
    )
    if np.std(first_gaussian) <= 0.0 or np.std(second_gaussian) <= 0.0:
        return 0.0
    correlation = float(np.corrcoef(first_gaussian, second_gaussian)[0, 1])
    return float(np.clip(correlation, -0.999999, 0.999999))


def _ridge_inverse(design: np.ndarray, *, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T)


def _coordinate_lattice_probabilities(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64)
    integer = np.all(
        np.isclose(coordinates, np.round(coordinates), atol=1.0e-7), axis=1
    )
    half_grid = np.all(
        np.isclose(coordinates * 2.0, np.round(coordinates * 2.0), atol=1.0e-7),
        axis=1,
    )
    counts = np.asarray(
        (
            np.count_nonzero(~half_grid),
            np.count_nonzero(half_grid & ~integer),
            np.count_nonzero(integer),
        ),
        dtype=np.float64,
    )
    return counts / float(len(coordinates))


def _fit_tap_start_model(
    events: Sequence[_TapTrainingEvent],
    *,
    ridge: float,
) -> _TapStartModel:
    orientation_id = events[0].orientation_id
    width_px, height_px = screen_dimensions_for_orientation(orientation_id)
    log_duration = np.log(
        np.asarray([event.duration_ms for event in events], dtype=np.float64)
    )
    duration_center = float(np.mean(log_duration))
    duration_scale = float(np.std(log_duration))
    if duration_scale < 1.0e-12:
        duration_scale = 1.0
    design = np.column_stack(
        (
            np.ones(len(events), dtype=np.float64),
            (log_duration - duration_center) / duration_scale,
        )
    )
    starts = np.asarray([event.start_xy_px for event in events], dtype=np.float64)
    dimensions = np.asarray((width_px, height_px), dtype=np.float64)
    edge_fraction = 0.25 / dimensions
    fractions = np.clip(
        starts / dimensions[None, :],
        edge_fraction[None, :],
        1.0 - edge_fraction[None, :],
    )
    transformed = np.log(fractions / (1.0 - fractions))
    coordinate_beta = _ridge_inverse(design, ridge=ridge) @ transformed
    residual = transformed - design @ coordinate_beta
    probabilities = np.linspace(
        0.0, 1.0, RESIDUAL_QUANTILE_COUNT, dtype=np.float64
    )
    residual_quantiles = np.asarray(
        np.quantile(residual, probabilities, axis=0).T,
        dtype=np.float64,
    )
    return _TapStartModel(
        duration_center=duration_center,
        duration_scale=duration_scale,
        duration_minimum=float(np.min(log_duration)),
        duration_maximum=float(np.max(log_duration)),
        coordinate_beta=np.asarray(coordinate_beta, dtype=np.float64),
        coordinate_residual_quantiles=residual_quantiles,
        gaussian_rank_correlation=_gaussian_rank_correlation(
            residual[:, 0], residual[:, 1]
        ),
        coordinate_lattice_probabilities=_coordinate_lattice_probabilities(
            starts
        ),
        training_event_count=len(events),
    )


def _fit_binary_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    positive = float(np.mean(labels))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[0] = _logit(positive)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    for _ in range(50):
        probabilities = np.asarray(
            [_sigmoid(value) for value in design @ beta], dtype=np.float64
        )
        weights = np.clip(
            probabilities * (1.0 - probabilities), 1.0e-8, None
        )
        gradient = design.T @ (probabilities - labels) + penalty @ beta
        hessian = design.T @ (weights[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        beta -= update
        if float(np.max(np.abs(update))) < 1.0e-10:
            break
    return beta


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def _fit_softmax_ridge(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    ridge: float,
) -> np.ndarray:
    """Fit a baseline-class multinomial ridge model with Newton steps."""

    feature_count = design.shape[1]
    fitted_class_count = class_count - 1
    beta = np.zeros((fitted_class_count, feature_count), dtype=np.float64)
    targets = np.eye(class_count, dtype=np.float64)[labels]
    penalty_mask = np.ones(feature_count, dtype=np.float64)
    penalty_mask[0] = 0.0

    def objective(candidate: np.ndarray) -> float:
        scores = np.column_stack(
            (design @ candidate.T, np.zeros(len(design), dtype=np.float64))
        )
        probabilities = np.clip(_softmax(scores), 1.0e-15, 1.0)
        likelihood = -float(np.sum(targets * np.log(probabilities)))
        regularization = 0.5 * float(ridge) * float(
            np.sum(candidate[:, 1:] ** 2)
        )
        return likelihood + regularization

    for _ in range(40):
        scores = np.column_stack(
            (design @ beta.T, np.zeros(len(design), dtype=np.float64))
        )
        probabilities = _softmax(scores)
        gradient = (probabilities[:, :-1] - targets[:, :-1]).T @ design
        gradient += float(ridge) * beta * penalty_mask[None, :]
        hessian = np.zeros(
            (fitted_class_count * feature_count,) * 2, dtype=np.float64
        )
        for first_class in range(fitted_class_count):
            first_slice = slice(
                first_class * feature_count,
                (first_class + 1) * feature_count,
            )
            for second_class in range(fitted_class_count):
                second_slice = slice(
                    second_class * feature_count,
                    (second_class + 1) * feature_count,
                )
                weight = probabilities[:, first_class] * (
                    float(first_class == second_class)
                    - probabilities[:, second_class]
                )
                hessian[first_slice, second_slice] = (
                    design.T @ (weight[:, None] * design)
                )
            hessian[first_slice, first_slice] += np.diag(
                float(ridge) * penalty_mask
            )
        hessian += np.eye(len(hessian), dtype=np.float64) * 1.0e-10
        update = np.linalg.solve(hessian, gradient.reshape(-1)).reshape(
            fitted_class_count, feature_count
        )
        current = objective(beta)
        step = 1.0
        while step >= 1.0e-6 and objective(beta - step * update) > current:
            step *= 0.5
        beta -= step * update
        if float(np.max(np.abs(step * update))) < 1.0e-9:
            break
    result = np.zeros((feature_count, class_count), dtype=np.float64)
    result[:, :-1] = beta.T
    return result


def _fit_tap_direction_model(
    events: Sequence[_TapTrainingEvent],
    *,
    ridge: float,
) -> _TapDirectionModel:
    moving = [event for event in events if not event.stationary]
    raw_features = np.asarray(
        [
            (
                np.log(event.duration_ms),
                float(event.orientation_id == 1),
                float(event.orientation_id == 3),
            )
            for event in moving
        ],
        dtype=np.float64,
    )
    feature_center = np.mean(raw_features, axis=0)
    feature_scale = np.std(raw_features, axis=0)
    feature_scale = np.where(feature_scale < 1.0e-12, 1.0, feature_scale)
    design = np.column_stack(
        (
            np.ones(len(moving), dtype=np.float64),
            (raw_features - feature_center[None, :])
            / feature_scale[None, :],
        )
    )
    labels = np.asarray(
        [DIRECTION8.index(str(event.direction)) for event in moving],
        dtype=np.int64,
    )
    counts = np.bincount(labels, minlength=len(DIRECTION8)).astype(np.int64)
    return _TapDirectionModel(
        feature_center=feature_center,
        feature_scale=feature_scale,
        feature_minimum=np.min(raw_features, axis=0),
        feature_maximum=np.max(raw_features, axis=0),
        direction_beta=_fit_softmax_ridge(
            design,
            labels,
            class_count=len(DIRECTION8),
            ridge=ridge,
        ),
        direction_counts=counts,
        training_event_count=len(moving),
    )


def _fit_tap_motion_condition(
    events: Sequence[_TapTrainingEvent],
    *,
    direction: str,
    ridge: float,
) -> _ConditionModel:
    selected = [event for event in events if event.direction == direction]
    raw_features = []
    for event in selected:
        width_px, height_px = screen_dimensions_for_orientation(
            event.orientation_id
        )
        raw_features.append(
            (
                np.log(event.duration_ms),
                event.start_xy_px[0] / width_px,
                event.start_xy_px[1] / height_px,
            )
        )
    raw_feature_array = np.asarray(raw_features, dtype=np.float64)
    feature_center = np.mean(raw_feature_array, axis=0)
    feature_scale = np.std(raw_feature_array, axis=0)
    feature_scale = np.where(feature_scale < 1.0e-12, 1.0, feature_scale)
    design = np.column_stack(
        (
            np.ones(len(selected), dtype=np.float64),
            (raw_feature_array - feature_center[None, :])
            / feature_scale[None, :],
        )
    )
    angle_values = np.asarray(
        [float(event.angle_value) for event in selected], dtype=np.float64
    )
    distance_values = np.asarray(
        [float(event.log_distance) for event in selected], dtype=np.float64
    )
    inverse = _ridge_inverse(design, ridge=ridge)
    angle_beta = inverse @ angle_values
    distance_beta = inverse @ distance_values
    angle_residual = angle_values - design @ angle_beta
    distance_residual = distance_values - design @ distance_beta
    return _ConditionModel(
        action="tap",
        orientation_id=-1,
        direction=direction,
        feature_center=feature_center,
        feature_scale=feature_scale,
        feature_minimum=np.min(raw_feature_array, axis=0),
        feature_maximum=np.max(raw_feature_array, axis=0),
        angle_beta=angle_beta,
        distance_beta=distance_beta,
        angle_residual_quantiles=_empirical_quantiles(angle_residual),
        distance_residual_quantiles=_empirical_quantiles(distance_residual),
        gaussian_rank_correlation=_gaussian_rank_correlation(
            angle_residual, distance_residual
        ),
        training_event_count=len(selected),
    )


def _fit_condition(
    events: Sequence[_TrainingEvent],
    *,
    ridge: float,
) -> _ConditionModel:
    first = events[0]
    width_px, height_px = screen_dimensions_for_orientation(first.orientation_id)
    raw_features = np.asarray(
        [
            (
                np.log(event.duration_ms),
                event.start_xy_px[0] / width_px,
                event.start_xy_px[1] / height_px,
            )
            for event in events
        ],
        dtype=np.float64,
    )
    feature_center = np.mean(raw_features, axis=0)
    feature_scale = np.std(raw_features, axis=0)
    feature_scale = np.where(feature_scale < 1.0e-12, 1.0, feature_scale)
    design = np.column_stack(
        (
            np.ones(len(events), dtype=np.float64),
            (raw_features - feature_center[None, :])
            / feature_scale[None, :],
        )
    )
    angle_values = np.asarray([event.angle_value for event in events])
    distance_values = np.asarray([event.log_distance for event in events])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    inverse = np.linalg.solve(
        design.T @ design + penalty,
        design.T,
    )
    angle_beta = inverse @ angle_values
    distance_beta = inverse @ distance_values
    angle_residual = angle_values - design @ angle_beta
    distance_residual = distance_values - design @ distance_beta

    rank_angle = (_average_ranks(angle_residual) - 0.5) / float(len(events))
    rank_distance = (
        _average_ranks(distance_residual) - 0.5
    ) / float(len(events))
    gaussian_angle = np.asarray(
        [_normal_inverse(value) for value in rank_angle], dtype=np.float64
    )
    gaussian_distance = np.asarray(
        [_normal_inverse(value) for value in rank_distance], dtype=np.float64
    )
    if np.std(gaussian_angle) > 0.0 and np.std(gaussian_distance) > 0.0:
        correlation = float(
            np.corrcoef(gaussian_angle, gaussian_distance)[0, 1]
        )
    else:
        correlation = 0.0
    correlation = float(np.clip(correlation, -0.999999, 0.999999))
    return _ConditionModel(
        action=first.action,
        orientation_id=first.orientation_id,
        direction=first.direction,
        feature_center=feature_center,
        feature_scale=feature_scale,
        feature_minimum=np.min(raw_features, axis=0),
        feature_maximum=np.max(raw_features, axis=0),
        angle_beta=angle_beta,
        distance_beta=distance_beta,
        angle_residual_quantiles=_empirical_quantiles(angle_residual),
        distance_residual_quantiles=_empirical_quantiles(distance_residual),
        gaussian_rank_correlation=correlation,
        training_event_count=len(events),
    )


def _validate_model(model: _ConditionModel) -> None:
    vectors = {
        "feature_center": (model.feature_center, (3,)),
        "feature_scale": (model.feature_scale, (3,)),
        "feature_minimum": (model.feature_minimum, (3,)),
        "feature_maximum": (model.feature_maximum, (3,)),
        "angle_beta": (model.angle_beta, (4,)),
        "distance_beta": (model.distance_beta, (4,)),
        "angle_residual_quantiles": (
            model.angle_residual_quantiles,
            (RESIDUAL_QUANTILE_COUNT,),
        ),
        "distance_residual_quantiles": (
            model.distance_residual_quantiles,
            (RESIDUAL_QUANTILE_COUNT,),
        ),
    }
    for name, (value, shape) in vectors.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ConditionalTouchRequestGeneratorError(
                f"condition model {name} is malformed"
            )
    if np.any(model.feature_scale <= 0.0):
        raise ConditionalTouchRequestGeneratorError(
            "condition model feature_scale must be positive"
        )
    if np.any(model.feature_minimum > model.feature_maximum):
        raise ConditionalTouchRequestGeneratorError(
            "condition model feature range is malformed"
        )
    if np.any(np.diff(model.angle_residual_quantiles) < 0.0) or np.any(
        np.diff(model.distance_residual_quantiles) < 0.0
    ):
        raise ConditionalTouchRequestGeneratorError(
            "condition model residual quantiles must be nondecreasing"
        )
    valid_identity = (
        model.action in ("scroll", "swipe")
        and model.orientation_id in (0, 1, 3)
    ) or (model.action == "tap" and model.orientation_id == -1)
    if not (
        valid_identity
        and model.direction in DIRECTION8
        and model.training_event_count > 0
        and np.isfinite(model.gaussian_rank_correlation)
        and abs(model.gaussian_rank_correlation) < 1.0
    ):
        raise ConditionalTouchRequestGeneratorError(
            "condition model identity or scalar parameters are malformed"
        )


def _validate_probability_vector(value: np.ndarray, *, name: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (3,)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or not np.isclose(np.sum(array), 1.0, atol=1.0e-8)
    ):
        raise ConditionalTouchRequestGeneratorError(
            f"tap {name} probabilities are malformed"
        )


def _validate_tap_start_model(model: _TapStartModel) -> None:
    if (
        not np.isfinite(model.duration_center)
        or not np.isfinite(model.duration_scale)
        or model.duration_scale <= 0.0
        or not np.isfinite(model.duration_minimum)
        or not np.isfinite(model.duration_maximum)
        or model.duration_minimum > model.duration_maximum
        or model.training_event_count <= 0
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap start duration model is malformed"
        )
    if (
        np.asarray(model.coordinate_beta).shape != (2, 2)
        or not np.isfinite(model.coordinate_beta).all()
        or np.asarray(model.coordinate_residual_quantiles).shape
        != (2, RESIDUAL_QUANTILE_COUNT)
        or not np.isfinite(model.coordinate_residual_quantiles).all()
        or np.any(np.diff(model.coordinate_residual_quantiles, axis=1) < 0.0)
        or not np.isfinite(model.gaussian_rank_correlation)
        or abs(model.gaussian_rank_correlation) >= 1.0
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap start coordinate model is malformed"
        )
    _validate_probability_vector(
        model.coordinate_lattice_probabilities,
        name="start lattice",
    )


def _validate_tap_orientation_model(model: _TapOrientationModel) -> None:
    if (
        model.orientation_id not in (0, 1, 3)
        or not np.isfinite(model.duration_center)
        or not np.isfinite(model.duration_scale)
        or model.duration_scale <= 0.0
        or not np.isfinite(model.duration_minimum)
        or not np.isfinite(model.duration_maximum)
        or model.duration_minimum > model.duration_maximum
        or np.asarray(model.stationary_beta).shape != (2,)
        or not np.isfinite(model.stationary_beta).all()
        or model.training_event_count <= 0
        or model.stationary_event_count <= 0
        or model.moving_event_count <= 0
        or model.stationary_event_count + model.moving_event_count
        != model.training_event_count
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap orientation model is malformed"
        )
    _validate_tap_start_model(model.stationary_start)
    _validate_tap_start_model(model.moving_start)
    if (
        model.stationary_start.training_event_count
        != model.stationary_event_count
        or model.moving_start.training_event_count != model.moving_event_count
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap branch start counts do not match orientation counts"
        )
    _validate_probability_vector(
        model.moving_endpoint_lattice_probabilities,
        name="moving endpoint lattice",
    )


def _validate_tap_direction_model(model: _TapDirectionModel) -> None:
    vectors = (
        (model.feature_center, (3,)),
        (model.feature_scale, (3,)),
        (model.feature_minimum, (3,)),
        (model.feature_maximum, (3,)),
        (model.direction_beta, (4, len(DIRECTION8))),
        (model.direction_counts, (len(DIRECTION8),)),
    )
    if any(
        np.asarray(value).shape != shape or not np.isfinite(value).all()
        for value, shape in vectors
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap direction model arrays are malformed"
        )
    if (
        np.any(model.feature_scale <= 0.0)
        or np.any(model.feature_minimum > model.feature_maximum)
        or np.any(model.direction_counts <= 0)
        or int(np.sum(model.direction_counts)) != model.training_event_count
        or model.training_event_count <= 0
    ):
        raise ConditionalTouchRequestGeneratorError(
            "tap direction model values are malformed"
        )


def _sample_endpoint_from_condition(
    model: _ConditionModel,
    *,
    start: np.ndarray,
    duration_ms: float,
    width_px: float,
    height_px: float,
    rng: np.random.Generator,
) -> _EndpointSample:
    raw_features = np.asarray(
        (
            np.log(float(duration_ms)),
            start[0] / width_px,
            start[1] / height_px,
        ),
        dtype=np.float64,
    )
    standardized = (raw_features - model.feature_center) / model.feature_scale
    design = np.concatenate((np.asarray([1.0]), standardized))
    angle_mean = float(design @ model.angle_beta)
    distance_mean = float(design @ model.distance_beta)
    outside = bool(
        np.any(raw_features < model.feature_minimum)
        or np.any(raw_features > model.feature_maximum)
    )
    gaussian_angle = float(rng.standard_normal())
    angle_residual = _evaluate_quantiles(
        model.angle_residual_quantiles,
        _normal_cdf(gaussian_angle),
    )
    sector_limit = float(np.nextafter(1.0, 0.0))
    normalized_angle_offset = float(
        np.clip(
            np.tanh(angle_mean + angle_residual),
            -sector_limit,
            sector_limit,
        )
    )
    angle = float(
        _direction_center(model.direction)
        + normalized_angle_offset * np.pi / 8.0
    )
    available = _ray_screen_distance(
        start,
        angle,
        width_px=width_px,
        height_px=height_px,
    )
    if not np.isfinite(available) or available <= 0.0:
        raise ConditionalTouchRequestGeneratorError(
            "requested direction has no positive screen ray from start"
        )
    available_bound = float(np.nextafter(available, 0.0))
    if available_bound <= 0.0:
        raise ConditionalTouchRequestGeneratorError(
            "requested screen ray is below floating-point support"
        )
    maximum_residual = float(np.log(available_bound)) - distance_mean
    maximum_probability = _quantile_cdf(
        model.distance_residual_quantiles,
        maximum_residual,
    )
    if maximum_probability <= 0.0:
        raise ConditionalTouchRequestGeneratorError(
            "learned distance support cannot fit the screen from start"
        )
    correlation = model.gaussian_rank_correlation
    conditional_mean = correlation * gaussian_angle
    conditional_scale = float(np.sqrt(1.0 - correlation * correlation))
    gaussian_maximum = _normal_inverse(maximum_probability)
    conditional_maximum = _normal_cdf(
        (gaussian_maximum - conditional_mean) / conditional_scale
    )
    if conditional_maximum <= 0.0:
        raise ConditionalTouchRequestGeneratorError(
            "conditional learned distance support cannot fit the screen"
        )
    truncated_probability = max(
        float(np.finfo(np.float64).eps),
        float(rng.random()) * conditional_maximum,
    )
    gaussian_distance = conditional_mean + conditional_scale * _normal_inverse(
        truncated_probability
    )
    distance_residual = _evaluate_quantiles(
        model.distance_residual_quantiles,
        _normal_cdf(gaussian_distance),
    )
    distance = float(np.exp(distance_mean + distance_residual))
    if not np.isfinite(distance) or distance <= 0.0 or distance >= available:
        raise ConditionalTouchRequestGeneratorError(
            "sampled distance violated conditioned screen support"
        )
    endpoint = start + distance * np.asarray(
        (np.cos(angle), np.sin(angle)), dtype=np.float64
    )
    if not (
        np.isfinite(endpoint).all()
        and 0.0 <= endpoint[0] <= width_px
        and 0.0 <= endpoint[1] <= height_px
    ):
        raise ConditionalTouchRequestGeneratorError(
            "sampled endpoint leaves the screen without clipping"
        )
    if _direction8(*(endpoint - start)) != model.direction:
        raise ConditionalTouchRequestGeneratorError(
            "sampled endpoint left its requested direction sector"
        )
    return _EndpointSample(
        endpoint=endpoint,
        distance_px=distance,
        angle_rad=angle,
        available_distance_px=float(available),
        conditional_support_probability=float(maximum_probability),
        feature_outside_training_range=outside,
    )


def _sample_tap_start(
    model: _TapStartModel,
    *,
    duration_ms: float,
    width_px: float,
    height_px: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool, float]:
    log_duration = float(np.log(duration_ms))
    design = np.asarray(
        (1.0, (log_duration - model.duration_center) / model.duration_scale),
        dtype=np.float64,
    )
    mean = design @ model.coordinate_beta
    first_gaussian = float(rng.standard_normal())
    correlation = model.gaussian_rank_correlation
    second_gaussian = float(
        correlation * first_gaussian
        + np.sqrt(1.0 - correlation * correlation) * rng.standard_normal()
    )
    gaussian = (first_gaussian, second_gaussian)
    transformed = np.asarray(
        [
            mean[index]
            + _evaluate_quantiles(
                model.coordinate_residual_quantiles[index],
                _normal_cdf(gaussian[index]),
            )
            for index in range(2)
        ],
        dtype=np.float64,
    )
    dimensions = np.asarray((width_px, height_px), dtype=np.float64)
    start = dimensions * np.asarray(
        [_sigmoid(value) for value in transformed], dtype=np.float64
    )
    lattice_index = int(
        rng.choice(3, p=model.coordinate_lattice_probabilities)
    )
    quantum = (0.0, 0.5, 1.0)[lattice_index]
    if quantum > 0.0:
        start = np.round(start / quantum) * quantum
    if not (
        np.isfinite(start).all()
        and 0.0 <= start[0] <= width_px
        and 0.0 <= start[1] <= height_px
    ):
        raise ConditionalTouchRequestGeneratorError(
            "sampled tap start leaves the screen without clipping"
        )
    outside = not (
        model.duration_minimum <= log_duration <= model.duration_maximum
    )
    return start, bool(outside), float(quantum)


def _tap_stationary_probability(
    model: _TapOrientationModel,
    *,
    duration_ms: float,
) -> tuple[float, bool]:
    log_duration = float(np.log(duration_ms))
    design = np.asarray(
        (1.0, (log_duration - model.duration_center) / model.duration_scale),
        dtype=np.float64,
    )
    probability = _sigmoid(float(design @ model.stationary_beta))
    outside = not (
        model.duration_minimum <= log_duration <= model.duration_maximum
    )
    return probability, bool(outside)


def _tap_direction_probabilities(
    model: _TapDirectionModel,
    *,
    orientation_id: int,
    duration_ms: float,
) -> tuple[np.ndarray, bool]:
    raw_features = np.asarray(
        (
            np.log(duration_ms),
            float(orientation_id == 1),
            float(orientation_id == 3),
        ),
        dtype=np.float64,
    )
    design = np.concatenate(
        (
            np.asarray([1.0]),
            (raw_features - model.feature_center) / model.feature_scale,
        )
    )
    probabilities = _softmax(design @ model.direction_beta)
    outside = bool(
        np.any(raw_features < model.feature_minimum)
        or np.any(raw_features > model.feature_maximum)
    )
    return np.asarray(probabilities, dtype=np.float64), outside


def _sample_lattice_quantum(
    probabilities: np.ndarray,
    *,
    rng: np.random.Generator,
) -> float:
    index = int(rng.choice(3, p=probabilities))
    return float((0.0, 0.5, 1.0)[index])


class ConditionalTouchRequestGenerator:
    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        models: Sequence[_ConditionModel],
        *,
        tap_orientation_models: Sequence[_TapOrientationModel] = (),
        tap_direction_model: _TapDirectionModel | None = None,
        metadata: Mapping[str, object],
        artifact_sha256: str | None = None,
    ) -> None:
        if not models and not tap_orientation_models:
            raise ConditionalTouchRequestGeneratorError(
                "request generator has no fitted models"
            )
        for model in models:
            _validate_model(model)
        self._models = {
            (model.action, model.orientation_id, model.direction): model
            for model in models
        }
        if len(self._models) != len(models):
            raise ConditionalTouchRequestGeneratorError(
                "request generator has duplicate conditions"
            )
        for model in tap_orientation_models:
            _validate_tap_orientation_model(model)
        self._tap_orientation_models = {
            model.orientation_id: model for model in tap_orientation_models
        }
        if len(self._tap_orientation_models) != len(tap_orientation_models):
            raise ConditionalTouchRequestGeneratorError(
                "request generator has duplicate tap orientations"
            )
        if self._tap_orientation_models:
            if tap_direction_model is None:
                raise ConditionalTouchRequestGeneratorError(
                    "tap request generator has no direction model"
                )
            _validate_tap_direction_model(tap_direction_model)
            tap_motion_directions = {
                direction
                for action, orientation, direction in self._models
                if action == "tap" and orientation == -1
            }
            if tap_motion_directions != set(DIRECTION8):
                raise ConditionalTouchRequestGeneratorError(
                    "tap request generator must fit all moving directions"
                )
        elif tap_direction_model is not None or any(
            action == "tap" for action, _, _ in self._models
        ):
            raise ConditionalTouchRequestGeneratorError(
                "tap internal models exist without tap orientation models"
            )
        self._tap_direction_model = tap_direction_model
        self._metadata = json.loads(json.dumps(dict(metadata), sort_keys=True))
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
        return json.loads(json.dumps(self._metadata, sort_keys=True))

    @property
    def supported_conditions(self) -> tuple[tuple[str, int, str], ...]:
        public = [
            condition for condition in self._models if condition[0] != "tap"
        ]
        public.extend(
            ("tap", orientation_id, "full_pair")
            for orientation_id in self._tap_orientation_models
        )
        return tuple(sorted(public))

    @classmethod
    def fit_from_raw_rows(
        cls,
        rows: Mapping[str, Sequence[object] | np.ndarray],
        *,
        ridge: float = 1.0e-3,
        minimum_events_per_condition: int = 20,
        training_source_sha256s: Sequence[str] = (),
    ) -> "ConditionalTouchRequestGenerator":
        missing = [name for name in _REQUIRED_ROW_FIELDS if name not in rows]
        if missing:
            raise ConditionalTouchRequestGeneratorError(
                "raw rows are missing required fields: " + ", ".join(missing)
            )
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ConditionalTouchRequestGeneratorError(
                "ridge must be finite and positive"
            )
        if minimum_events_per_condition < 2:
            raise ConditionalTouchRequestGeneratorError(
                "minimum_events_per_condition must be at least two"
            )
        arrays = {name: np.asarray(rows[name]) for name in _REQUIRED_ROW_FIELDS}
        lengths = {len(value) for value in arrays.values() if value.ndim == 1}
        if any(value.ndim != 1 for value in arrays.values()) or len(lengths) != 1:
            raise ConditionalTouchRequestGeneratorError(
                "all required raw row fields must be equal-length vectors"
            )

        grouped: dict[tuple[object, str, int], list[int]] = {}
        ignored_unsupported_rows = 0
        for index in range(next(iter(lengths))):
            action = _text(arrays["action"][index]).lower()
            if action not in SUPPORTED_ACTIONS:
                ignored_unsupported_rows += 1
                continue
            event_id = arrays["event_id"][index]
            if isinstance(event_id, np.generic):
                event_id = event_id.item()
            try:
                hash(event_id)
            except TypeError as error:
                raise ConditionalTouchRequestGeneratorError(
                    "event_id values must be hashable"
                ) from error
            orientation_id = int(arrays["orientation_id"][index])
            grouped.setdefault((event_id, action, orientation_id), []).append(index)

        events_by_condition: dict[
            tuple[str, int, str], list[_TrainingEvent]
        ] = {}
        tap_events_by_orientation: dict[int, list[_TapTrainingEvent]] = {}
        tap_events: list[_TapTrainingEvent] = []
        rejected_reasons: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        for (_, action, orientation_id), indices in grouped.items():
            selected = np.asarray(indices, dtype=np.int64)
            try:
                width_px, height_px = screen_dimensions_for_orientation(
                    orientation_id
                )
            except ValueError:
                reject("unsupported orientation")
                continue
            t_ms = np.asarray(arrays["t_ms"][selected], dtype=np.float64)
            x_px = np.asarray(arrays["x_px"][selected], dtype=np.float64)
            y_px = np.asarray(arrays["y_px"][selected], dtype=np.float64)
            android_action = np.asarray(
                arrays["android_action"][selected], dtype=np.int64
            )
            if (
                len(t_ms) < 2
                or not np.isfinite(t_ms).all()
                or np.any(np.diff(t_ms) < 0.0)
                or not np.isfinite(x_px).all()
                or not np.isfinite(y_px).all()
            ):
                reject("invalid raw row values")
                continue
            if (
                np.any(x_px < 0.0)
                or np.any(x_px > width_px)
                or np.any(y_px < 0.0)
                or np.any(y_px > height_px)
            ):
                reject("training coordinates leave the screen")
                continue
            masked = np.bitwise_and(android_action, ACTION_MASK)
            down_rows = np.flatnonzero(masked == ACTION_DOWN)
            up_rows = np.flatnonzero(masked == ACTION_UP)
            if (
                len(down_rows) != 1
                or len(up_rows) != 1
                or int(up_rows[0]) <= int(down_rows[0])
            ):
                reject("event must have one ordered DOWN and UP")
                continue
            down, up = int(down_rows[0]), int(up_rows[0])
            duration_ms = float(t_ms[up] - t_ms[down])
            start = np.asarray((x_px[down], y_px[down]), dtype=np.float64)
            end = np.asarray((x_px[up], y_px[up]), dtype=np.float64)
            chord = end - start
            distance = float(np.linalg.norm(chord))
            if duration_ms <= 0.0:
                reject("event duration must be positive")
                continue
            if action == "tap" and distance == 0.0:
                tap_event = _TapTrainingEvent(
                    orientation_id=orientation_id,
                    duration_ms=duration_ms,
                    start_xy_px=start,
                    end_xy_px=end,
                    stationary=True,
                    direction=None,
                    angle_value=None,
                    log_distance=None,
                )
                tap_events.append(tap_event)
                tap_events_by_orientation.setdefault(
                    orientation_id, []
                ).append(tap_event)
                continue
            if distance <= 0.0:
                reject("event endpoint distance must be positive")
                continue
            direction = _direction8(float(chord[0]), float(chord[1]))
            angle = float(np.arctan2(chord[1], chord[0]))
            normalized_offset = _wrap_pi(
                angle - _direction_center(direction)
            ) / float(np.pi / 8.0)
            available = _ray_screen_distance(
                start,
                angle,
                width_px=width_px,
                height_px=height_px,
            )
            if available <= 0.0 or distance > available + 1.0e-7:
                reject("endpoint distance exceeds its screen ray")
                continue
            angle_value = float(
                np.arctanh(np.clip(normalized_offset, -0.999999, 0.999999))
            )
            if action == "tap":
                tap_event = _TapTrainingEvent(
                    orientation_id=orientation_id,
                    duration_ms=duration_ms,
                    start_xy_px=start,
                    end_xy_px=end,
                    stationary=False,
                    direction=direction,
                    angle_value=angle_value,
                    log_distance=float(np.log(distance)),
                )
                tap_events.append(tap_event)
                tap_events_by_orientation.setdefault(
                    orientation_id, []
                ).append(tap_event)
                continue
            event = _TrainingEvent(
                action=action,
                orientation_id=orientation_id,
                direction=direction,
                duration_ms=duration_ms,
                start_xy_px=start,
                angle_value=angle_value,
                log_distance=float(np.log(distance)),
            )
            events_by_condition.setdefault(
                (action, orientation_id, direction), []
            ).append(event)

        sparse = {
            f"{action}|{orientation}|{direction}": len(events)
            for (action, orientation, direction), events in events_by_condition.items()
            if len(events) < minimum_events_per_condition
        }
        if tap_events:
            for orientation_id, orientation_events in tap_events_by_orientation.items():
                stationary_count = sum(
                    event.stationary for event in orientation_events
                )
                moving_count = len(orientation_events) - stationary_count
                if stationary_count < minimum_events_per_condition:
                    sparse[
                        f"tap|{orientation_id}|stationary"
                    ] = stationary_count
                if moving_count < minimum_events_per_condition:
                    sparse[f"tap|{orientation_id}|moving"] = moving_count
            for direction in DIRECTION8:
                direction_count = sum(
                    (not event.stationary) and event.direction == direction
                    for event in tap_events
                )
                if direction_count < minimum_events_per_condition:
                    sparse[f"tap|pooled|{direction}"] = direction_count
        if sparse:
            raise ConditionalTouchRequestGeneratorError(
                "conditions below minimum_events_per_condition: "
                + ", ".join(
                    f"{condition}={count}"
                    for condition, count in sorted(sparse.items())
                )
            )
        models = [
            _fit_condition(events, ridge=float(ridge))
            for _, events in sorted(events_by_condition.items())
        ]
        tap_orientation_models: list[_TapOrientationModel] = []
        tap_direction_model = None
        if tap_events:
            for orientation_id, orientation_events in sorted(
                tap_events_by_orientation.items()
            ):
                stationary_events = [
                    event for event in orientation_events if event.stationary
                ]
                moving_events = [
                    event for event in orientation_events if not event.stationary
                ]
                log_duration = np.log(
                    np.asarray(
                        [event.duration_ms for event in orientation_events],
                        dtype=np.float64,
                    )
                )
                duration_center = float(np.mean(log_duration))
                duration_scale = float(np.std(log_duration))
                if duration_scale < 1.0e-12:
                    duration_scale = 1.0
                duration_design = np.column_stack(
                    (
                        np.ones(len(orientation_events), dtype=np.float64),
                        (log_duration - duration_center) / duration_scale,
                    )
                )
                stationary_labels = np.asarray(
                    [event.stationary for event in orientation_events],
                    dtype=np.float64,
                )
                tap_orientation_models.append(
                    _TapOrientationModel(
                        orientation_id=orientation_id,
                        duration_center=duration_center,
                        duration_scale=duration_scale,
                        duration_minimum=float(np.min(log_duration)),
                        duration_maximum=float(np.max(log_duration)),
                        stationary_beta=_fit_binary_logistic(
                            duration_design,
                            stationary_labels,
                            ridge=float(ridge),
                        ),
                        stationary_start=_fit_tap_start_model(
                            stationary_events, ridge=float(ridge)
                        ),
                        moving_start=_fit_tap_start_model(
                            moving_events, ridge=float(ridge)
                        ),
                        moving_endpoint_lattice_probabilities=(
                            _coordinate_lattice_probabilities(
                                np.asarray(
                                    [
                                        event.end_xy_px
                                        for event in moving_events
                                    ],
                                    dtype=np.float64,
                                )
                            )
                        ),
                        training_event_count=len(orientation_events),
                        stationary_event_count=len(stationary_events),
                        moving_event_count=len(moving_events),
                    )
                )
            tap_direction_model = _fit_tap_direction_model(
                tap_events, ridge=float(ridge)
            )
            models.extend(
                _fit_tap_motion_condition(
                    tap_events,
                    direction=direction,
                    ridge=float(ridge),
                )
                for direction in DIRECTION8
            )
        if not models and not tap_orientation_models:
            raise ConditionalTouchRequestGeneratorError(
                "no valid request events were fitted"
            )
        condition_counts = {
            f"{model.action}|{model.orientation_id}|{model.direction}": (
                model.training_event_count
            )
            for model in models
            if model.action != "tap"
        }
        condition_counts.update(
            {
                f"tap|{model.orientation_id}|full_pair": (
                    model.training_event_count
                )
                for model in tap_orientation_models
            }
        )
        internal_tap_motion_counts = {
            model.direction: model.training_event_count
            for model in models
            if model.action == "tap"
        }
        source_binding = {
            "request_generator_source_sha256": (
                IMPORT_REQUEST_GENERATOR_SOURCE_SHA256
            ),
            "android_touch_observation_source_sha256": (
                IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
            ),
            "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
        }
        metadata = {
            "fit_config": {
                "ridge": float(ridge),
                "minimum_events_per_condition": int(
                    minimum_events_per_condition
                ),
                "residual_quantile_count": RESIDUAL_QUANTILE_COUNT,
                "feature_order": [
                    "log_duration_ms",
                    "start_x_screen_fraction",
                    "start_y_screen_fraction",
                ],
                "condition_order": ["action", "orientation_id", "direction"],
                "tap_request_contract": (
                    "duration+orientation generate an unbound full start/end pair"
                ),
            },
            "accepted_event_counts": condition_counts,
            "accepted_event_count": int(sum(condition_counts.values())),
            "tap_internal_moving_direction_counts": internal_tap_motion_counts,
            "rejected_event_count": int(sum(rejected_reasons.values())),
            "rejected_event_reasons": rejected_reasons,
            "ignored_unsupported_row_count": int(ignored_unsupported_rows),
            "training_source_sha256s": sorted(set(training_source_sha256s)),
            "endpoint_policy": {
                "scroll": (
                    "tanh direction-sector angle plus screen-conditioned "
                    "truncated log-distance residual"
                ),
                "swipe": (
                    "tanh direction-sector angle plus screen-conditioned "
                    "truncated log-distance residual"
                ),
                "tap": (
                    "duration-conditioned full-pair start, exact stationary "
                    "point mass, and joint moving angle/log-distance with "
                    "screen-conditioned truncation"
                ),
            },
            "tap_external_bound_start_used": False,
            "raw_event_ids_retained": False,
            "raw_trajectories_retained": False,
            "runtime_donor_lookup_used": False,
            "coordinate_clipping_used": False,
            **source_binding,
        }
        return cls(
            models,
            tap_orientation_models=tap_orientation_models,
            tap_direction_model=tap_direction_model,
            metadata=metadata,
        )

    def _serialize(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        manifest_models: list[dict[str, object]] = []
        for index, condition in enumerate(sorted(self._models)):
            model = self._models[condition]
            prefix = f"model_{index:03d}"
            manifest_models.append(
                {
                    "prefix": prefix,
                    "action": model.action,
                    "orientation_id": model.orientation_id,
                    "direction": model.direction,
                    "gaussian_rank_correlation": (
                        model.gaussian_rank_correlation
                    ),
                    "training_event_count": model.training_event_count,
                }
            )
            for name in (
                "feature_center",
                "feature_scale",
                "feature_minimum",
                "feature_maximum",
                "angle_beta",
                "distance_beta",
                "angle_residual_quantiles",
                "distance_residual_quantiles",
            ):
                arrays[f"{prefix}_{name}"] = np.asarray(getattr(model, name))
        manifest_tap_orientations: list[dict[str, object]] = []
        for index, orientation_id in enumerate(
            sorted(self._tap_orientation_models)
        ):
            model = self._tap_orientation_models[orientation_id]
            prefix = f"tap_orientation_{index:02d}"
            arrays[f"{prefix}_stationary_beta"] = np.asarray(
                model.stationary_beta
            )
            arrays[
                f"{prefix}_moving_endpoint_lattice_probabilities"
            ] = np.asarray(model.moving_endpoint_lattice_probabilities)
            branch_manifest: dict[str, object] = {}
            for branch_name, start_model in (
                ("stationary", model.stationary_start),
                ("moving", model.moving_start),
            ):
                branch_prefix = f"{prefix}_{branch_name}_start"
                for name in (
                    "coordinate_beta",
                    "coordinate_residual_quantiles",
                    "coordinate_lattice_probabilities",
                ):
                    arrays[f"{branch_prefix}_{name}"] = np.asarray(
                        getattr(start_model, name)
                    )
                branch_manifest[branch_name] = {
                    "prefix": branch_prefix,
                    "duration_center": start_model.duration_center,
                    "duration_scale": start_model.duration_scale,
                    "duration_minimum": start_model.duration_minimum,
                    "duration_maximum": start_model.duration_maximum,
                    "gaussian_rank_correlation": (
                        start_model.gaussian_rank_correlation
                    ),
                    "training_event_count": start_model.training_event_count,
                }
            manifest_tap_orientations.append(
                {
                    "prefix": prefix,
                    "orientation_id": model.orientation_id,
                    "duration_center": model.duration_center,
                    "duration_scale": model.duration_scale,
                    "duration_minimum": model.duration_minimum,
                    "duration_maximum": model.duration_maximum,
                    "training_event_count": model.training_event_count,
                    "stationary_event_count": model.stationary_event_count,
                    "moving_event_count": model.moving_event_count,
                    "start_models": branch_manifest,
                }
            )
        tap_direction_manifest = None
        if self._tap_direction_model is not None:
            direction_prefix = "tap_direction"
            for name in (
                "feature_center",
                "feature_scale",
                "feature_minimum",
                "feature_maximum",
                "direction_beta",
                "direction_counts",
            ):
                arrays[f"{direction_prefix}_{name}"] = np.asarray(
                    getattr(self._tap_direction_model, name)
                )
            tap_direction_manifest = {
                "prefix": direction_prefix,
                "training_event_count": (
                    self._tap_direction_model.training_event_count
                ),
            }
        manifest = {
            "schema_version": self.schema_version,
            "request_generator_source_sha256": (
                IMPORT_REQUEST_GENERATOR_SOURCE_SHA256
            ),
            "android_touch_observation_source_sha256": (
                IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
            ),
            "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
            "metadata": self._metadata,
            "models": manifest_models,
            "tap_orientation_models": manifest_tap_orientations,
            "tap_direction_model": tap_direction_manifest,
        }
        arrays["manifest_json"] = np.asarray(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        return arrays

    def save(self, path: str | Path) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez_compressed(handle, **self._serialize())
        self._artifact_sha256 = _sha256_file(output)
        return self._artifact_sha256

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalTouchRequestGenerator":
        artifact = Path(path)
        try:
            archive_context = np.load(artifact, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ConditionalTouchRequestGeneratorError(
                f"cannot load request generator artifact: {artifact}"
            ) from error
        with archive_context as archive:
            try:
                manifest = json.loads(
                    str(np.asarray(archive["manifest_json"]).item())
                )
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                raise ConditionalTouchRequestGeneratorError(
                    "request generator artifact manifest is malformed"
                ) from error
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ConditionalTouchRequestGeneratorError(
                    f"unsupported request generator schema "
                    f"{manifest.get('schema_version')!r}"
                )
            expected_binding = {
                "request_generator_source_sha256": (
                    IMPORT_REQUEST_GENERATOR_SOURCE_SHA256
                ),
                "android_touch_observation_source_sha256": (
                    IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256
                ),
                "source_fingerprint_sha256": IMPORT_SOURCE_FINGERPRINT_SHA256,
            }
            for field, expected in expected_binding.items():
                if manifest.get(field) != expected:
                    raise ConditionalTouchRequestGeneratorError(
                        f"request artifact {field} does not match imported code"
                    )
            metadata = manifest.get("metadata")
            if not isinstance(metadata, dict):
                raise ConditionalTouchRequestGeneratorError(
                    "request artifact metadata must be an object"
                )
            for field, expected in expected_binding.items():
                if metadata.get(field) != expected:
                    raise ConditionalTouchRequestGeneratorError(
                        f"request artifact metadata {field} does not match "
                        "imported code"
                    )
            raw_models = manifest.get("models")
            if not isinstance(raw_models, list) or not raw_models:
                raise ConditionalTouchRequestGeneratorError(
                    "request artifact has no model manifest"
                )
            models: list[_ConditionModel] = []
            tap_orientation_models: list[_TapOrientationModel] = []
            tap_direction_model = None
            try:
                for item in raw_models:
                    if not isinstance(item, dict):
                        raise KeyError("model item")
                    prefix = str(item["prefix"])
                    values = {
                        name: np.asarray(
                            archive[f"{prefix}_{name}"], dtype=np.float64
                        )
                        for name in (
                            "feature_center",
                            "feature_scale",
                            "feature_minimum",
                            "feature_maximum",
                            "angle_beta",
                            "distance_beta",
                            "angle_residual_quantiles",
                            "distance_residual_quantiles",
                        )
                    }
                    models.append(
                        _ConditionModel(
                            action=str(item["action"]),
                            orientation_id=int(item["orientation_id"]),
                            direction=str(item["direction"]),
                            gaussian_rank_correlation=float(
                                item["gaussian_rank_correlation"]
                            ),
                            training_event_count=int(
                                item["training_event_count"]
                            ),
                            **values,
                        )
                    )
                raw_tap_orientations = manifest.get(
                    "tap_orientation_models", []
                )
                if not isinstance(raw_tap_orientations, list):
                    raise KeyError("tap_orientation_models")
                for item in raw_tap_orientations:
                    if not isinstance(item, dict):
                        raise KeyError("tap orientation item")
                    prefix = str(item["prefix"])
                    raw_start_models = item["start_models"]
                    if not isinstance(raw_start_models, dict):
                        raise KeyError("tap start_models")

                    def load_start(branch_name: str) -> _TapStartModel:
                        branch = raw_start_models[branch_name]
                        if not isinstance(branch, dict):
                            raise KeyError("tap start branch")
                        branch_prefix = str(branch["prefix"])
                        return _TapStartModel(
                            duration_center=float(branch["duration_center"]),
                            duration_scale=float(branch["duration_scale"]),
                            duration_minimum=float(
                                branch["duration_minimum"]
                            ),
                            duration_maximum=float(
                                branch["duration_maximum"]
                            ),
                            coordinate_beta=np.asarray(
                                archive[f"{branch_prefix}_coordinate_beta"],
                                dtype=np.float64,
                            ),
                            coordinate_residual_quantiles=np.asarray(
                                archive[
                                    f"{branch_prefix}_coordinate_residual_quantiles"
                                ],
                                dtype=np.float64,
                            ),
                            gaussian_rank_correlation=float(
                                branch["gaussian_rank_correlation"]
                            ),
                            coordinate_lattice_probabilities=np.asarray(
                                archive[
                                    f"{branch_prefix}_coordinate_lattice_probabilities"
                                ],
                                dtype=np.float64,
                            ),
                            training_event_count=int(
                                branch["training_event_count"]
                            ),
                        )

                    tap_orientation_models.append(
                        _TapOrientationModel(
                            orientation_id=int(item["orientation_id"]),
                            duration_center=float(item["duration_center"]),
                            duration_scale=float(item["duration_scale"]),
                            duration_minimum=float(
                                item["duration_minimum"]
                            ),
                            duration_maximum=float(
                                item["duration_maximum"]
                            ),
                            stationary_beta=np.asarray(
                                archive[f"{prefix}_stationary_beta"],
                                dtype=np.float64,
                            ),
                            stationary_start=load_start("stationary"),
                            moving_start=load_start("moving"),
                            moving_endpoint_lattice_probabilities=np.asarray(
                                archive[
                                    f"{prefix}_moving_endpoint_lattice_probabilities"
                                ],
                                dtype=np.float64,
                            ),
                            training_event_count=int(
                                item["training_event_count"]
                            ),
                            stationary_event_count=int(
                                item["stationary_event_count"]
                            ),
                            moving_event_count=int(item["moving_event_count"]),
                        )
                    )
                raw_tap_direction = manifest.get("tap_direction_model")
                if raw_tap_direction is not None:
                    if not isinstance(raw_tap_direction, dict):
                        raise KeyError("tap_direction_model")
                    prefix = str(raw_tap_direction["prefix"])
                    tap_direction_model = _TapDirectionModel(
                        feature_center=np.asarray(
                            archive[f"{prefix}_feature_center"],
                            dtype=np.float64,
                        ),
                        feature_scale=np.asarray(
                            archive[f"{prefix}_feature_scale"],
                            dtype=np.float64,
                        ),
                        feature_minimum=np.asarray(
                            archive[f"{prefix}_feature_minimum"],
                            dtype=np.float64,
                        ),
                        feature_maximum=np.asarray(
                            archive[f"{prefix}_feature_maximum"],
                            dtype=np.float64,
                        ),
                        direction_beta=np.asarray(
                            archive[f"{prefix}_direction_beta"],
                            dtype=np.float64,
                        ),
                        direction_counts=np.asarray(
                            archive[f"{prefix}_direction_counts"],
                            dtype=np.int64,
                        ),
                        training_event_count=int(
                            raw_tap_direction["training_event_count"]
                        ),
                    )
            except (KeyError, TypeError, ValueError) as error:
                raise ConditionalTouchRequestGeneratorError(
                    "request artifact model arrays are malformed"
                ) from error
        return cls(
            models,
            tap_orientation_models=tap_orientation_models,
            tap_direction_model=tap_direction_model,
            metadata=metadata,
            artifact_sha256=_sha256_file(artifact),
        )

    def generate(
        self,
        *,
        action: str,
        orientation_id: int,
        duration_ms: float,
        seed: int,
        direction: str | None = None,
        start_xy_px: Sequence[float] | None = None,
    ) -> GeneratedTouchRequest:
        action_text = str(action).lower()
        if not np.isfinite(duration_ms) or float(duration_ms) <= 0.0:
            raise ConditionalTouchRequestGeneratorError(
                "duration_ms must be finite and positive"
            )
        width_px, height_px = screen_dimensions_for_orientation(orientation_id)
        rng = np.random.default_rng(int(seed))
        if action_text == "tap":
            if direction is not None or start_xy_px is not None:
                raise ConditionalTouchRequestGeneratorError(
                    "tap full-pair requests do not accept a bound start or direction"
                )
            tap_model = self._tap_orientation_models.get(int(orientation_id))
            if tap_model is None or self._tap_direction_model is None:
                raise ConditionalTouchRequestGeneratorError(
                    f"no fitted tap full-pair model for orientation_id="
                    f"{orientation_id}"
                )
            stationary_probability, duration_outside = (
                _tap_stationary_probability(
                    tap_model, duration_ms=float(duration_ms)
                )
            )
            stationary = bool(rng.random() < stationary_probability)
            start_model = (
                tap_model.stationary_start
                if stationary
                else tap_model.moving_start
            )
            start, start_outside, start_quantum = _sample_tap_start(
                start_model,
                duration_ms=float(duration_ms),
                width_px=width_px,
                height_px=height_px,
                rng=rng,
            )
            if stationary:
                return GeneratedTouchRequest(
                    action="tap",
                    orientation_id=int(orientation_id),
                    direction="stationary",
                    start_xy_px=(float(start[0]), float(start[1])),
                    end_xy_px=(float(start[0]), float(start[1])),
                    duration_ms=float(duration_ms),
                    distance_px=0.0,
                    angle_rad=0.0,
                    available_distance_px=0.0,
                    conditional_support_probability=1.0,
                    feature_outside_training_range=bool(
                        duration_outside or start_outside
                    ),
                    seed=int(seed),
                    stationary=True,
                    stationary_probability=float(stationary_probability),
                    endpoint_quantization_px=float(start_quantum),
                )
            direction_probabilities, direction_outside = (
                _tap_direction_probabilities(
                    self._tap_direction_model,
                    orientation_id=int(orientation_id),
                    duration_ms=float(duration_ms),
                )
            )
            endpoint_quantum = _sample_lattice_quantum(
                tap_model.moving_endpoint_lattice_probabilities,
                rng=rng,
            )
            last_error = None
            for _ in range(64):
                direction_index = int(
                    rng.choice(len(DIRECTION8), p=direction_probabilities)
                )
                direction_text = DIRECTION8[direction_index]
                motion_model = self._models.get(
                    ("tap", -1, direction_text)
                )
                if motion_model is None:
                    raise ConditionalTouchRequestGeneratorError(
                        "tap moving direction model is missing"
                    )
                try:
                    sampled = _sample_endpoint_from_condition(
                        motion_model,
                        start=start,
                        duration_ms=float(duration_ms),
                        width_px=width_px,
                        height_px=height_px,
                        rng=rng,
                    )
                except ConditionalTouchRequestGeneratorError as error:
                    last_error = error
                    continue
                endpoint = sampled.endpoint.copy()
                if endpoint_quantum > 0.0:
                    endpoint = (
                        np.round(endpoint / endpoint_quantum)
                        * endpoint_quantum
                    )
                chord = endpoint - start
                distance = float(np.linalg.norm(chord))
                if (
                    not np.isfinite(endpoint).all()
                    or not 0.0 <= endpoint[0] <= width_px
                    or not 0.0 <= endpoint[1] <= height_px
                    or distance <= 0.0
                    or _direction8(float(chord[0]), float(chord[1]))
                    != direction_text
                ):
                    continue
                angle = float(np.arctan2(chord[1], chord[0]))
                available = _ray_screen_distance(
                    start,
                    angle,
                    width_px=width_px,
                    height_px=height_px,
                )
                return GeneratedTouchRequest(
                    action="tap",
                    orientation_id=int(orientation_id),
                    direction=direction_text,
                    start_xy_px=(float(start[0]), float(start[1])),
                    end_xy_px=(float(endpoint[0]), float(endpoint[1])),
                    duration_ms=float(duration_ms),
                    distance_px=distance,
                    angle_rad=angle,
                    available_distance_px=float(available),
                    conditional_support_probability=(
                        sampled.conditional_support_probability
                    ),
                    feature_outside_training_range=bool(
                        duration_outside
                        or start_outside
                        or direction_outside
                        or sampled.feature_outside_training_range
                    ),
                    seed=int(seed),
                    stationary=False,
                    stationary_probability=float(stationary_probability),
                    endpoint_quantization_px=float(endpoint_quantum),
                )
            detail = f": {last_error}" if last_error is not None else ""
            raise ConditionalTouchRequestGeneratorError(
                "could not sample a nonstationary in-screen tap endpoint" + detail
            )

        if direction is None or start_xy_px is None:
            raise ConditionalTouchRequestGeneratorError(
                f"{action_text} endpoint requests require start_xy_px and direction"
            )
        direction_text = str(direction)
        condition = (action_text, int(orientation_id), direction_text)
        model = self._models.get(condition)
        if model is None:
            raise ConditionalTouchRequestGeneratorError(
                f"no fitted endpoint model for action={action_text!r}, "
                f"orientation_id={orientation_id}, direction={direction_text!r}"
            )
        start = _as_start(start_xy_px)
        if not (
            0.0 <= start[0] <= width_px and 0.0 <= start[1] <= height_px
        ):
            raise ConditionalTouchRequestGeneratorError(
                "start coordinate leaves the physical screen"
            )
        sampled = _sample_endpoint_from_condition(
            model,
            start=start,
            duration_ms=float(duration_ms),
            width_px=width_px,
            height_px=height_px,
            rng=rng,
        )
        return GeneratedTouchRequest(
            action=action_text,
            orientation_id=int(orientation_id),
            direction=direction_text,
            start_xy_px=(float(start[0]), float(start[1])),
            end_xy_px=(
                float(sampled.endpoint[0]),
                float(sampled.endpoint[1]),
            ),
            duration_ms=float(duration_ms),
            distance_px=sampled.distance_px,
            angle_rad=sampled.angle_rad,
            available_distance_px=sampled.available_distance_px,
            conditional_support_probability=(
                sampled.conditional_support_probability
            ),
            feature_outside_training_range=(
                sampled.feature_outside_training_range
            ),
            seed=int(seed),
            stationary=False,
            stationary_probability=0.0,
            endpoint_quantization_px=0.0,
        )


__all__ = [
    "ConditionalTouchRequestGenerator",
    "ConditionalTouchRequestGeneratorError",
    "DIRECTION8",
    "GeneratedTouchRequest",
    "IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256",
    "IMPORT_REQUEST_GENERATOR_SOURCE_SHA256",
    "IMPORT_SOURCE_FINGERPRINT_SHA256",
    "RESIDUAL_QUANTILE_COUNT",
    "SCHEMA_VERSION",
    "SUPPORTED_ACTIONS",
]
