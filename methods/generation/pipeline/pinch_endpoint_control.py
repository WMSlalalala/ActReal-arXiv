from __future__ import annotations

"""Bounded endpoint control for complete two-pointer Android pinch rows.

The helper in this module is deliberately independent of the replay allocator.
It extracts the first and last *simultaneously live* pointer pairs, fits a
bounded endpoint deformation, and applies it without changing row timing or
Android lifecycle arrays.  Coordinates are never clipped.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .android_touch_observation import ACTION_CANCEL, ACTION_MASK, ACTION_UP


DEFAULT_MIN_PINCH_SCALE = 0.80
DEFAULT_MAX_PINCH_SCALE = 1.25
DEFAULT_STATIONARY_DIAGONAL_FRACTION = 0.02


class PinchEndpointControlError(ValueError):
    pass


Point = Tuple[float, float]
PointPair = Tuple[Point, Point]


@dataclass(frozen=True)
class PinchEndpointGeometry:
    """Ordered live-pointer geometry at the first and last pinch frames."""

    pointer_ids: tuple[int, int]
    start_points_px: PointPair
    end_points_px: PointPair
    start_center_px: Point
    end_center_px: Point
    start_span_px: float
    end_span_px: float
    start_frame_index: int
    end_frame_index: int
    start_t_ms: float
    end_t_ms: float

    @property
    def center_displacement_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.end_center_px, dtype=np.float64)
                - np.asarray(self.start_center_px, dtype=np.float64)
            )
        )

    @property
    def scale_direction(self) -> str:
        return "in" if self.end_span_px < self.start_span_px else "out"


@dataclass(frozen=True)
class PinchEndpointFit:
    """A bounded endpoint fit; middle-frame deformation remains donor-derived."""

    source: PinchEndpointGeometry
    target: PinchEndpointGeometry
    center_stationary: bool
    center_scale: float
    center_rotation_rad: float
    start_span_scale: float
    end_span_scale: float
    deformation_score: float
    minimum_scale: float
    maximum_scale: float


@dataclass(frozen=True)
class PinchEndpointTransformResult:
    """Transformed XY and auditable endpoint/deformation measurements."""

    x_px: np.ndarray
    y_px: np.ndarray
    fit: PinchEndpointFit
    start_endpoint_error_px: float
    end_endpoint_error_px: float
    coordinate_clipping_used: bool = False

    @property
    def maximum_endpoint_error_px(self) -> float:
        return max(self.start_endpoint_error_px, self.end_endpoint_error_px)


@dataclass(frozen=True)
class _LiveFrame:
    frame_index: int
    left: int
    right: int
    pointer_ids: tuple[int, int]
    selected_rows: tuple[int, int]
    points_px: np.ndarray
    center_px: np.ndarray
    span_px: float
    t_ms: float


def _as_1d(
    values: Iterable[object],
    *,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    output = np.asarray(values, dtype=dtype)
    if output.ndim != 1 or not len(output):
        raise PinchEndpointControlError(f"{name} must be a nonempty 1D array")
    return output


def _frame_bounds(frame_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.any(np.diff(frame_index) < 0):
        raise PinchEndpointControlError("frame indices are not ordered")
    changes = np.flatnonzero(frame_index[1:] != frame_index[:-1]) + 1
    bounds = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            changes,
            np.asarray([len(frame_index)], dtype=np.int64),
        )
    )
    return bounds[:-1], bounds[1:]


def _validated_row_arrays(
    *,
    t_ms: Iterable[object],
    frame_index: Iterable[object],
    pointer_id: Iterable[object],
    android_action: Iterable[object],
    x_px: Iterable[object],
    y_px: Iterable[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = _as_1d(t_ms, dtype=np.float64, name="t_ms")
    frames = _as_1d(frame_index, dtype=np.int64, name="frame_index")
    pointers = _as_1d(pointer_id, dtype=np.int64, name="pointer_id")
    actions = _as_1d(android_action, dtype=np.int64, name="android_action")
    x = _as_1d(x_px, dtype=np.float64, name="x_px")
    y = _as_1d(y_px, dtype=np.float64, name="y_px")
    if any(value.shape != time.shape for value in (frames, pointers, actions, x, y)):
        raise PinchEndpointControlError("pinch row arrays are misaligned")
    if (
        not np.isfinite(time).all()
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(np.diff(time) < 0.0)
    ):
        raise PinchEndpointControlError("pinch rows contain invalid values")
    return time, frames, pointers, actions & ACTION_MASK, x, y


def _live_frames(
    *,
    time: np.ndarray,
    frames: np.ndarray,
    pointers: np.ndarray,
    actions: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[_LiveFrame, ...]:
    starts, ends = _frame_bounds(frames)
    live_frames: list[_LiveFrame] = []
    canonical_ids: tuple[int, int] | None = None
    for left_value, right_value in zip(starts, ends):
        left, right = int(left_value), int(right_value)
        positions = np.arange(left, right, dtype=np.int64)
        selected: list[int] = []
        for pointer in np.unique(pointers[positions]):
            candidates = positions[pointers[positions] == pointer]
            live = candidates[~np.isin(actions[candidates], (ACTION_UP, ACTION_CANCEL))]
            if len(live):
                selected.append(int(live[-1]))
        if len(selected) < 2:
            continue
        chosen = tuple(
            sorted(selected, key=lambda row: int(pointers[row]))[:2]
        )
        pointer_pair = (int(pointers[chosen[0]]), int(pointers[chosen[1]]))
        if canonical_ids is None:
            canonical_ids = pointer_pair
        elif pointer_pair != canonical_ids:
            raise PinchEndpointControlError(
                "live pointer identity changed inside the pinch"
            )
        points = np.asarray(
            ((x[chosen[0]], y[chosen[0]]), (x[chosen[1]], y[chosen[1]])),
            dtype=np.float64,
        )
        span = float(np.linalg.norm(points[1] - points[0]))
        if not np.isfinite(span) or span <= 1.0e-9:
            raise PinchEndpointControlError("pinch has a zero live-pointer span")
        live_frames.append(
            _LiveFrame(
                frame_index=int(frames[left]),
                left=left,
                right=right,
                pointer_ids=pointer_pair,
                selected_rows=chosen,
                points_px=points,
                center_px=np.mean(points, axis=0),
                span_px=span,
                t_ms=float(np.max(time[np.asarray(chosen, dtype=np.int64)])),
            )
        )
    if len(live_frames) < 2:
        raise PinchEndpointControlError(
            "pinch needs at least two simultaneous live-pointer frames"
        )
    if live_frames[-1].t_ms <= live_frames[0].t_ms:
        raise PinchEndpointControlError(
            "pinch live-pointer endpoint time must increase"
        )
    return tuple(live_frames)


def _point_tuple(value: np.ndarray) -> Point:
    return (float(value[0]), float(value[1]))


def _point_pair_tuple(value: np.ndarray) -> PointPair:
    return (_point_tuple(value[0]), _point_tuple(value[1]))


def _geometry_from_live_frames(frames: tuple[_LiveFrame, ...]) -> PinchEndpointGeometry:
    first, last = frames[0], frames[-1]
    return PinchEndpointGeometry(
        pointer_ids=first.pointer_ids,
        start_points_px=_point_pair_tuple(first.points_px),
        end_points_px=_point_pair_tuple(last.points_px),
        start_center_px=_point_tuple(first.center_px),
        end_center_px=_point_tuple(last.center_px),
        start_span_px=float(first.span_px),
        end_span_px=float(last.span_px),
        start_frame_index=first.frame_index,
        end_frame_index=last.frame_index,
        start_t_ms=first.t_ms,
        end_t_ms=last.t_ms,
    )


def extract_live_two_pointer_endpoints(
    *,
    t_ms: Iterable[object],
    frame_index: Iterable[object],
    pointer_id: Iterable[object],
    android_action: Iterable[object],
    x_px: Iterable[object],
    y_px: Iterable[object],
) -> PinchEndpointGeometry:
    """Extract ordered endpoints, ignoring a trailing primary-UP coordinate."""

    time, frames, pointers, actions, x, y = _validated_row_arrays(
        t_ms=t_ms,
        frame_index=frame_index,
        pointer_id=pointer_id,
        android_action=android_action,
        x_px=x_px,
        y_px=y_px,
    )
    return _geometry_from_live_frames(
        _live_frames(
            time=time,
            frames=frames,
            pointers=pointers,
            actions=actions,
            x=x,
            y=y,
        )
    )


def endpoint_geometry_from_pairs(
    *,
    start_points_px: PointPair,
    end_points_px: PointPair,
    pointer_ids: tuple[int, int] = (0, 1),
) -> PinchEndpointGeometry:
    """Construct a target geometry when an API supplies four endpoint points."""

    start = np.asarray(start_points_px, dtype=np.float64)
    end = np.asarray(end_points_px, dtype=np.float64)
    if (
        start.shape != (2, 2)
        or end.shape != (2, 2)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or len(set(int(value) for value in pointer_ids)) != 2
    ):
        raise PinchEndpointControlError("pinch endpoint pairs are malformed")
    start_span = float(np.linalg.norm(start[1] - start[0]))
    end_span = float(np.linalg.norm(end[1] - end[0]))
    if start_span <= 1.0e-9 or end_span <= 1.0e-9:
        raise PinchEndpointControlError("pinch endpoint pair has zero span")
    return PinchEndpointGeometry(
        pointer_ids=(int(pointer_ids[0]), int(pointer_ids[1])),
        start_points_px=_point_pair_tuple(start),
        end_points_px=_point_pair_tuple(end),
        start_center_px=_point_tuple(np.mean(start, axis=0)),
        end_center_px=_point_tuple(np.mean(end, axis=0)),
        start_span_px=start_span,
        end_span_px=end_span,
        start_frame_index=0,
        end_frame_index=1,
        start_t_ms=0.0,
        end_t_ms=1.0,
    )


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _validate_scale_gate(minimum_scale: float, maximum_scale: float) -> None:
    if (
        not np.isfinite(minimum_scale)
        or not np.isfinite(maximum_scale)
        or minimum_scale <= 0.0
        or minimum_scale > 1.0
        or maximum_scale < 1.0
        or minimum_scale > maximum_scale
    ):
        raise PinchEndpointControlError(
            "scale gate must be finite, positive, and contain one"
        )


def _validate_endpoint_screen_bounds(
    geometry: PinchEndpointGeometry,
    *,
    screen_width_px: float,
    screen_height_px: float,
    role: str,
) -> None:
    points = np.asarray(
        geometry.start_points_px + geometry.end_points_px,
        dtype=np.float64,
    )
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > screen_width_px)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > screen_height_px)
    ):
        raise PinchEndpointControlError(f"{role} pinch endpoints leave the screen")


def fit_pinch_endpoint_control(
    source: PinchEndpointGeometry,
    target: PinchEndpointGeometry,
    *,
    screen_width_px: float,
    screen_height_px: float,
    minimum_scale: float = DEFAULT_MIN_PINCH_SCALE,
    maximum_scale: float = DEFAULT_MAX_PINCH_SCALE,
    stationary_diagonal_fraction: float = DEFAULT_STATIONARY_DIAGONAL_FRACTION,
) -> PinchEndpointFit:
    """Fit exact pinch endpoints while rejecting large spatial deformation."""

    _validate_scale_gate(float(minimum_scale), float(maximum_scale))
    if (
        not np.isfinite(screen_width_px)
        or not np.isfinite(screen_height_px)
        or screen_width_px <= 0.0
        or screen_height_px <= 0.0
        or not np.isfinite(stationary_diagonal_fraction)
        or stationary_diagonal_fraction < 0.0
    ):
        raise PinchEndpointControlError("screen/stationary configuration is invalid")
    _validate_endpoint_screen_bounds(
        source,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        role="source",
    )
    _validate_endpoint_screen_bounds(
        target,
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        role="target",
    )
    if source.scale_direction != target.scale_direction:
        raise PinchEndpointControlError("pinch in/out direction would change")

    diagonal = float(np.hypot(screen_width_px, screen_height_px))
    stationary_threshold = float(stationary_diagonal_fraction) * diagonal
    source_displacement = source.center_displacement_px
    target_displacement = target.center_displacement_px
    source_stationary = source_displacement <= stationary_threshold
    target_stationary = target_displacement <= stationary_threshold
    if source_stationary != target_stationary:
        raise PinchEndpointControlError(
            "source and target pinch center stationarity differ"
        )
    if source_stationary:
        center_scale = 1.0
    else:
        if source_displacement <= 1.0e-9 or target_displacement <= 1.0e-9:
            raise PinchEndpointControlError("moving pinch has zero center displacement")
        center_scale = target_displacement / source_displacement
    start_span_scale = target.start_span_px / source.start_span_px
    end_span_scale = target.end_span_px / source.end_span_px
    scales = (center_scale, start_span_scale, end_span_scale)
    if any(
        scale < float(minimum_scale) - 1.0e-12
        or scale > float(maximum_scale) + 1.0e-12
        for scale in scales
    ):
        raise PinchEndpointControlError(
            "pinch endpoint deformation leaves the bounded 0.80-1.25 scale gate"
        )

    source_delta = (
        np.asarray(source.end_center_px, dtype=np.float64)
        - np.asarray(source.start_center_px, dtype=np.float64)
    )
    target_delta = (
        np.asarray(target.end_center_px, dtype=np.float64)
        - np.asarray(target.start_center_px, dtype=np.float64)
    )
    if np.linalg.norm(source_delta) > 1.0e-9 and np.linalg.norm(target_delta) > 1.0e-9:
        center_rotation = _wrap_angle(
            float(np.arctan2(target_delta[1], target_delta[0]))
            - float(np.arctan2(source_delta[1], source_delta[0]))
        )
    else:
        center_rotation = 0.0
    deformation_score = max(abs(float(np.log(scale))) for scale in scales)
    return PinchEndpointFit(
        source=source,
        target=target,
        center_stationary=source_stationary,
        center_scale=float(center_scale),
        center_rotation_rad=center_rotation,
        start_span_scale=float(start_span_scale),
        end_span_scale=float(end_span_scale),
        deformation_score=deformation_score,
        minimum_scale=float(minimum_scale),
        maximum_scale=float(maximum_scale),
    )


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _pair_error(observed: np.ndarray, expected: PointPair) -> float:
    return float(
        np.max(
            np.linalg.norm(
                observed - np.asarray(expected, dtype=np.float64),
                axis=1,
            )
        )
    )


def apply_pinch_endpoint_control(
    *,
    t_ms: Iterable[object],
    frame_index: Iterable[object],
    pointer_id: Iterable[object],
    android_action: Iterable[object],
    x_px: Iterable[object],
    y_px: Iterable[object],
    fit: PinchEndpointFit,
    screen_width_px: float,
    screen_height_px: float,
) -> PinchEndpointTransformResult:
    """Apply a fitted endpoint deformation; return XY only and never clip."""

    time, frames, pointers, actions, x, y = _validated_row_arrays(
        t_ms=t_ms,
        frame_index=frame_index,
        pointer_id=pointer_id,
        android_action=android_action,
        x_px=x_px,
        y_px=y_px,
    )
    live_frames = _live_frames(
        time=time,
        frames=frames,
        pointers=pointers,
        actions=actions,
        x=x,
        y=y,
    )
    observed_source = _geometry_from_live_frames(live_frames)
    if observed_source.pointer_ids != fit.source.pointer_ids or not np.allclose(
        np.asarray(
            observed_source.start_points_px + observed_source.end_points_px,
            dtype=np.float64,
        ),
        np.asarray(
            fit.source.start_points_px + fit.source.end_points_px,
            dtype=np.float64,
        ),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise PinchEndpointControlError(
            "pinch fit does not belong to these source rows"
        )

    first, last = live_frames[0], live_frames[-1]
    source_center_start = first.center_px
    source_center_end = last.center_px
    target_center_start = np.asarray(fit.target.start_center_px, dtype=np.float64)
    target_center_end = np.asarray(fit.target.end_center_px, dtype=np.float64)
    center_matrix = fit.center_scale * _rotation(fit.center_rotation_rad)

    source_vectors = np.asarray(
        [frame.points_px[1] - frame.points_px[0] for frame in live_frames],
        dtype=np.float64,
    )
    source_angles = np.unwrap(np.arctan2(source_vectors[:, 1], source_vectors[:, 0]))
    source_log_spans = np.log(
        np.asarray([frame.span_px for frame in live_frames], dtype=np.float64)
    )
    target_start_vector = (
        np.asarray(fit.target.start_points_px[1], dtype=np.float64)
        - np.asarray(fit.target.start_points_px[0], dtype=np.float64)
    )
    target_end_vector = (
        np.asarray(fit.target.end_points_px[1], dtype=np.float64)
        - np.asarray(fit.target.end_points_px[0], dtype=np.float64)
    )
    target_start_angle = float(
        np.arctan2(target_start_vector[1], target_start_vector[0])
    )
    target_angle_delta = _wrap_angle(
        float(np.arctan2(target_end_vector[1], target_end_vector[0]))
        - target_start_angle
    )
    target_start_log_span = float(np.log(fit.target.start_span_px))
    target_end_log_span = float(np.log(fit.target.end_span_px))

    frame_transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    endpoint_denominator = last.t_ms - first.t_ms
    for index, frame in enumerate(live_frames):
        progress = float(
            np.clip(
                (frame.t_ms - first.t_ms) / endpoint_denominator,
                0.0,
                1.0,
            )
        )
        source_center_baseline = (
            (1.0 - progress) * source_center_start + progress * source_center_end
        )
        target_center_baseline = (
            (1.0 - progress) * target_center_start + progress * target_center_end
        )
        output_center = target_center_baseline + center_matrix @ (
            frame.center_px - source_center_baseline
        )
        source_angle_baseline = (
            (1.0 - progress) * source_angles[0]
            + progress * source_angles[-1]
        )
        output_angle = (
            target_start_angle
            + progress * target_angle_delta
            + (source_angles[index] - source_angle_baseline)
        )
        source_log_span_baseline = (
            (1.0 - progress) * source_log_spans[0]
            + progress * source_log_spans[-1]
        )
        output_log_span = (
            (1.0 - progress) * target_start_log_span
            + progress * target_end_log_span
            + (source_log_spans[index] - source_log_span_baseline)
        )
        output_span = float(np.exp(output_log_span))
        output_vector = output_span * np.asarray(
            (np.cos(output_angle), np.sin(output_angle)), dtype=np.float64
        )
        output_points = np.asarray(
            (output_center - output_vector / 2.0, output_center + output_vector / 2.0),
            dtype=np.float64,
        )
        if index == 0:
            output_points = np.asarray(fit.target.start_points_px, dtype=np.float64)
        elif index == len(live_frames) - 1:
            output_points = np.asarray(fit.target.end_points_px, dtype=np.float64)
        source_vector = frame.points_px[1] - frame.points_px[0]
        output_vector = output_points[1] - output_points[0]
        source_angle = float(np.arctan2(source_vector[1], source_vector[0]))
        output_angle = float(np.arctan2(output_vector[1], output_vector[0]))
        row_matrix = (
            float(np.linalg.norm(output_vector)) / frame.span_px
        ) * _rotation(output_angle - source_angle)
        row_translation = np.mean(output_points, axis=0) - row_matrix @ frame.center_px
        frame_transforms[frame.frame_index] = (row_matrix, row_translation)

    all_frame_values = np.unique(frames)
    live_frame_values = np.asarray(sorted(frame_transforms), dtype=np.int64)
    output_x = np.empty_like(x)
    output_y = np.empty_like(y)
    for frame_value in all_frame_values:
        positions = np.flatnonzero(frames == frame_value)
        if int(frame_value) in frame_transforms:
            matrix, translation = frame_transforms[int(frame_value)]
        else:
            nearest = int(
                live_frame_values[
                    np.argmin(np.abs(live_frame_values - int(frame_value)))
                ]
            )
            matrix, translation = frame_transforms[nearest]
        source_xy = np.column_stack((x[positions], y[positions]))
        output_xy = source_xy @ matrix.T + translation
        output_x[positions] = output_xy[:, 0]
        output_y[positions] = output_xy[:, 1]

    if (
        not np.isfinite(output_x).all()
        or not np.isfinite(output_y).all()
        or np.any(output_x < 0.0)
        or np.any(output_x > screen_width_px)
        or np.any(output_y < 0.0)
        or np.any(output_y > screen_height_px)
    ):
        raise PinchEndpointControlError(
            "bounded pinch deformation would leave the screen; clipping is forbidden"
        )
    output_geometry = extract_live_two_pointer_endpoints(
        t_ms=time,
        frame_index=frames,
        pointer_id=pointers,
        android_action=actions,
        x_px=output_x,
        y_px=output_y,
    )
    start_error = _pair_error(
        np.asarray(output_geometry.start_points_px, dtype=np.float64),
        fit.target.start_points_px,
    )
    end_error = _pair_error(
        np.asarray(output_geometry.end_points_px, dtype=np.float64),
        fit.target.end_points_px,
    )
    if max(start_error, end_error) > 1.0e-6:
        raise PinchEndpointControlError("pinch endpoint control missed its request")
    return PinchEndpointTransformResult(
        x_px=output_x,
        y_px=output_y,
        fit=fit,
        start_endpoint_error_px=start_error,
        end_endpoint_error_px=end_error,
        coordinate_clipping_used=False,
    )


__all__ = [
    "DEFAULT_MAX_PINCH_SCALE",
    "DEFAULT_MIN_PINCH_SCALE",
    "DEFAULT_STATIONARY_DIAGONAL_FRACTION",
    "PinchEndpointControlError",
    "PinchEndpointFit",
    "PinchEndpointGeometry",
    "PinchEndpointTransformResult",
    "apply_pinch_endpoint_control",
    "endpoint_geometry_from_pairs",
    "extract_live_two_pointer_endpoints",
    "fit_pinch_endpoint_control",
]
