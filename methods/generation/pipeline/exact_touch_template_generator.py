from __future__ import annotations

"""Exact, donor-based transport for one-finger touch templates.

This module is deliberately independent of the replay and learned-generator
implementations.  A donor contributes only its timing, pressure, and residual
path shape.  Requested geometry and duration are authoritative, coordinates
are never clipped, and all calculations remain small vector operations.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


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
SUPPORTED_ACTIONS = frozenset(("tap", "scroll", "swipe"))
SCHEMA_VERSION = "exact-touch-template-generator-v1"
# A caller that asks for the donor's own chord cannot hand back a bit-exact
# difference: ``start + chord`` rounds once in screen pixels, so recovering the
# chord as ``end - start`` lands within an ulp instead of on the nose.  Decide
# the pure-translation branch at a tolerance five orders below the pipeline's
# 5e-4 px endpoint contract; the requested endpoints are still written onto the
# output exactly, so nothing downstream loosens.
TAP_CHORD_MATCH_TOLERANCE_PX = 1.0e-9


class ExactTouchTemplateError(ValueError):
    """Raised when an exact touch request or its donor is invalid."""


@dataclass(frozen=True)
class ExactTouchTemplate:
    """One transported raw touch template and its deformation audit."""

    action: str
    direction: str
    duration_ms: float
    mode: str
    identity_transform: bool
    residual_scale: float
    t_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray


def _vector(name: str, values: Iterable[object]) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ExactTouchTemplateError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ExactTouchTemplateError(
            f"{name} must be a non-empty finite numeric vector"
        )
    return result


def _point(name: str, value: Sequence[float]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ExactTouchTemplateError(f"{name} must be a numeric XY pair") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ExactTouchTemplateError(f"{name} must be a finite XY pair")
    return result


def _direction8(delta: np.ndarray) -> str:
    angle = float(np.arctan2(float(delta[1]), float(delta[0])))
    index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
    return DIRECTION8[index]


def _request_direction(start: np.ndarray, end: np.ndarray) -> str:
    if np.array_equal(start, end):
        return STATIONARY
    return _direction8(end - start)


def _normalized_time(t_ms: np.ndarray) -> np.ndarray:
    if len(t_ms) < 2:
        raise ExactTouchTemplateError("template must contain at least two rows")
    relative = t_ms - float(t_ms[0])
    if np.any(np.diff(relative) < 0.0):
        raise ExactTouchTemplateError("template time must be nondecreasing")
    source_duration = float(relative[-1])
    if source_duration <= 0.0:
        raise ExactTouchTemplateError("template duration must be positive")
    result = relative / source_duration
    result[0] = 0.0
    result[-1] = 1.0
    return result


def _screen_residual_scale(
    baseline: np.ndarray,
    residual: np.ndarray,
    dimensions: np.ndarray,
) -> float:
    """Return one common residual scale proving every point is on screen."""

    scale = 1.0
    for axis in range(2):
        delta = residual[:, axis]
        positive = delta > 0.0
        negative = delta < 0.0
        if np.any(positive):
            scale = min(
                scale,
                float(
                    np.min(
                        (dimensions[axis] - baseline[positive, axis])
                        / delta[positive]
                    )
                ),
            )
        if np.any(negative):
            scale = min(
                scale,
                float(
                    np.min(
                        (0.0 - baseline[negative, axis]) / delta[negative]
                    )
                ),
            )
    scale = max(0.0, min(1.0, scale))
    if scale < 1.0:
        # Stay strictly inside the analytic boundary despite final additions.
        scale = float(np.nextafter(scale, 0.0))
    return scale


def _chord_frame_residual(
    points: np.ndarray,
    *,
    progress: np.ndarray,
    source_chord: np.ndarray,
    target_chord: np.ndarray,
) -> np.ndarray:
    source_length = float(np.linalg.norm(source_chord))
    target_length = float(np.linalg.norm(target_chord))
    if source_length <= 0.0:
        raise ExactTouchTemplateError(
            "scroll/swipe template must have a non-stationary chord"
        )
    if target_length <= 0.0:
        raise ExactTouchTemplateError(
            "scroll/swipe request must have a non-stationary chord"
        )
    source_unit = source_chord / source_length
    target_unit = target_chord / target_length
    source_normal = np.asarray(
        (-source_unit[1], source_unit[0]), dtype=np.float64
    )
    target_normal = np.asarray(
        (-target_unit[1], target_unit[0]), dtype=np.float64
    )
    source_offsets = points - points[0]
    source_longitudinal = source_offsets @ source_unit
    source_lateral = source_offsets @ source_normal
    longitudinal_residual = source_longitudinal - progress * source_length
    ratio = target_length / source_length
    result = ratio * (
        longitudinal_residual[:, None] * target_unit[None, :]
        + source_lateral[:, None] * target_normal[None, :]
    )
    result[0] = 0.0
    result[-1] = 0.0
    return result


def generate_exact_touch_template(
    *,
    action: str,
    start_xy_px: Sequence[float],
    end_xy_px: Sequence[float],
    direction: str,
    duration_ms: float,
    template_t_ms: Iterable[object],
    template_x_px: Iterable[object],
    template_y_px: Iterable[object],
    template_pressure: Iterable[object],
    screen_width_px: float,
    screen_height_px: float,
) -> ExactTouchTemplate:
    """Transport a single-pointer donor to exact geometry and duration."""

    action_name = str(action)
    if action_name not in SUPPORTED_ACTIONS:
        raise ExactTouchTemplateError(
            f"unsupported single-pointer action {action_name!r}"
        )
    if not isinstance(direction, str):
        raise ExactTouchTemplateError("direction must be explicit")
    try:
        width = float(screen_width_px)
        height = float(screen_height_px)
        duration = float(duration_ms)
    except (TypeError, ValueError) as exc:
        raise ExactTouchTemplateError(
            "screen dimensions and duration must be numeric"
        ) from exc
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ExactTouchTemplateError("screen dimensions must be finite and positive")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ExactTouchTemplateError("duration must be finite and positive")

    dimensions = np.asarray((width, height), dtype=np.float64)
    start = _point("start_xy_px", start_xy_px)
    end = _point("end_xy_px", end_xy_px)
    if (
        np.any(start < 0.0)
        or np.any(end < 0.0)
        or np.any(start > dimensions)
        or np.any(end > dimensions)
    ):
        raise ExactTouchTemplateError("requested endpoint leaves the screen")
    realized_direction = _request_direction(start, end)
    if direction != realized_direction:
        raise ExactTouchTemplateError(
            "direction conflicts with the requested endpoints"
        )
    if action_name in {"scroll", "swipe"} and realized_direction == STATIONARY:
        raise ExactTouchTemplateError(
            f"{action_name} requires a non-stationary endpoint request"
        )

    source_t = _vector("template_t_ms", template_t_ms)
    source_x = _vector("template_x_px", template_x_px)
    source_y = _vector("template_y_px", template_y_px)
    pressure = _vector("template_pressure", template_pressure)
    row_count = len(source_t)
    if any(len(values) != row_count for values in (source_x, source_y, pressure)):
        raise ExactTouchTemplateError("template vectors must have equal lengths")
    if np.any(pressure < 0.0) or np.any(pressure > 1.0):
        raise ExactTouchTemplateError("template pressure must lie in [0, 1]")
    source_points = np.column_stack((source_x, source_y))
    if np.any(source_points < 0.0) or np.any(source_points > dimensions[None, :]):
        raise ExactTouchTemplateError("template coordinate leaves the screen")
    progress = _normalized_time(source_t)
    source_relative_t = source_t - float(source_t[0])
    source_duration = float(source_relative_t[-1])
    source_chord = source_points[-1] - source_points[0]
    target_chord = end - start
    target_baseline = start[None, :] + progress[:, None] * target_chord[None, :]

    identity_transform = bool(
        np.array_equal(start, source_points[0])
        and np.array_equal(end, source_points[-1])
        and duration == source_duration
    )
    # A tap whose request already carries the donor's own DOWN-to-UP drift has
    # the donor's chord, so the entire donor path -- including the samples where
    # the finger never moved -- transports by pure translation and keeps the
    # zero-order-hold staircase a real observer produces.
    tap_same_chord = bool(
        action_name == "tap"
        and np.allclose(
            source_chord,
            target_chord,
            rtol=0.0,
            atol=TAP_CHORD_MATCH_TOLERANCE_PX,
        )
    )
    if identity_transform:
        mode = "identity_template"
        residual_scale = 1.0
        output_points = source_points.copy()
        output_t = source_relative_t.copy()
    elif tap_same_chord:
        mode = "tap_translation"
        residual_scale = 1.0
        output_points = source_points + (start - source_points[0])[None, :]
        if (
            np.any(output_points < 0.0)
            or np.any(output_points > dimensions[None, :])
        ):
            raise ExactTouchTemplateError(
                "pure tap translation would leave the screen"
            )
        output_t = progress * duration
    else:
        if action_name == "tap":
            mode = "tap_residual_bridge"
            source_baseline = (
                source_points[0][None, :]
                + progress[:, None] * source_chord[None, :]
            )
            residual = source_points - source_baseline
            residual[0] = 0.0
            residual[-1] = 0.0
        else:
            mode = f"{action_name}_chord_frame"
            residual = _chord_frame_residual(
                source_points,
                progress=progress,
                source_chord=source_chord,
                target_chord=target_chord,
            )
        residual_scale = _screen_residual_scale(
            target_baseline, residual, dimensions
        )
        output_points = target_baseline + residual_scale * residual
        output_t = progress * duration

    # Requests, not donors or accumulated arithmetic, own the endpoints.
    output_points[0] = start
    output_points[-1] = end
    output_t[0] = 0.0
    output_t[-1] = duration

    if (
        not np.isfinite(output_points).all()
        or np.any(output_points < 0.0)
        or np.any(output_points > dimensions[None, :])
    ):
        raise ExactTouchTemplateError("transported touch left the screen")
    if not np.isfinite(output_t).all() or np.any(np.diff(output_t) < 0.0):
        raise ExactTouchTemplateError("transported time is invalid")
    if not np.array_equal(output_points[0], start) or not np.array_equal(
        output_points[-1], end
    ):
        raise ExactTouchTemplateError("transport missed an exact endpoint")
    if output_t[0] != 0.0 or output_t[-1] != duration:
        raise ExactTouchTemplateError("transport missed the exact duration")
    if _request_direction(output_points[0], output_points[-1]) != direction:
        raise ExactTouchTemplateError("transport changed the requested direction")

    return ExactTouchTemplate(
        action=action_name,
        direction=direction,
        duration_ms=duration,
        mode=mode,
        identity_transform=identity_transform,
        residual_scale=float(residual_scale),
        t_ms=output_t,
        x_px=output_points[:, 0],
        y_px=output_points[:, 1],
        pressure=pressure.copy(),
    )


__all__ = [
    "DIRECTION8",
    "ExactTouchTemplate",
    "ExactTouchTemplateError",
    "SCHEMA_VERSION",
    "STATIONARY",
    "TAP_CHORD_MATCH_TOLERANCE_PX",
    "generate_exact_touch_template",
]
