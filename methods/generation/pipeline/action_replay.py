from __future__ import annotations

"""Leakage-safe replay of raw HMOG Android action primitives.

This module deliberately treats replay as an attack construction, not as a
trajectory generator.  Every primitive is a complete raw HMOG MotionEvent
contact from a training user.  The default replay leaves all row values
unchanged.  Supported spatial deformation is normally limited to a D4 rigid
isometry (axis swap/sign changes) followed by one common XY translation.
Explicit single-pointer endpoint control uses one bounded global similarity
transform.  The legacy
narrow integer-millisecond warp remains available, but full allocation keeps
raw gesture time exact and lets the common label-blind observer map it to the
detector endpoint.  No operation invents touch samples, clips a coordinate, or
interpolates a touch value.

The source-user split and the detector output split are intentionally
different concepts.  ``ActionReplayBank.from_hmog_npz`` accepts only source
training users.  ``partition_output_splits`` then assigns each primitive to
exactly one output split so a donor (or a transformed sibling) cannot occur in
both detector training and detector evaluation.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .android_touch_observation import (
    ACTION_CANCEL,
    ACTION_DOWN,
    ACTION_MASK,
    ACTION_POINTER_DOWN,
    ACTION_POINTER_UP,
    ACTION_UP,
    PERIOD_MS,
    TouchObservation,
    observe_android_rows,
    screen_dimensions_for_orientation,
)
from .pinch_endpoint_control import (
    PinchEndpointControlError,
    PinchEndpointFit,
    PinchEndpointGeometry,
    apply_pinch_endpoint_control,
    extract_live_two_pointer_endpoints,
    fit_pinch_endpoint_control,
)


SUPPORTED_ACTIONS = ("scroll", "swipe", "pinch")
OUTPUT_SPLITS = ("train", "development", "test")
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
PINCH_SCALE_DIRECTIONS = ("in", "out")

# Frozen train-corpus quartiles.  They are selection strata, not acceptance
# gates: continuous duration remains in every descriptor and can be used for
# nearest-neighbour assignment inside a bucket.
DURATION_BUCKET_EDGES_MS: Mapping[str, tuple[float, float, float]] = {
    "scroll": (402.0, 560.0, 975.0),
    "swipe": (237.0, 394.0, 661.0),
}
DURATION_BUCKET_NAMES = ("q0", "q1", "q2", "q3")
PINCH_STATIONARY_DIAGONAL_FRACTION = 0.02
DEFAULT_MAX_TIME_WARP = 1.18
ENDPOINT_MIN_SPATIAL_SCALE = 0.80
ENDPOINT_MAX_SPATIAL_SCALE = 1.25


class ActionReplayError(ValueError):
    pass


def transport_detector_touch_template(
    trajectory: np.ndarray,
    *,
    action: str,
    orientation_id: int,
    start_xy_px: Sequence[float],
    end_xy_px: Sequence[float],
    direction: str | None,
    duration_ms: float,
) -> tuple[np.ndarray, float]:
    """Map one detector-grid donor to exact requested geometry and duration."""

    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) < 2:
        raise ActionReplayError("touch template must have shape [samples, 9]")
    if not np.isfinite(values).all():
        raise ActionReplayError("touch template contains non-finite values")
    width, height = screen_dimensions_for_orientation(int(orientation_id))
    start = np.asarray(start_xy_px, dtype=np.float64)
    end = np.asarray(end_xy_px, dtype=np.float64)
    if (
        start.shape != (2,)
        or end.shape != (2,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
    ):
        raise ActionReplayError("requested endpoints must be finite XY pairs")
    if (
        np.any(start < 0.0)
        or np.any(end < 0.0)
        or start[0] > width
        or end[0] > width
        or start[1] > height
        or end[1] > height
    ):
        raise ActionReplayError("requested endpoint leaves the physical screen")
    duration = float(duration_ms)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ActionReplayError("requested duration must be finite and positive")
    target_chord = end - start
    target_length = float(np.linalg.norm(target_chord))
    realized = (
        STATIONARY
        if target_length <= 1.0e-12
        else _direction8(float(target_chord[0]), float(target_chord[1]))
    )
    if action == "tap":
        if direction not in (None, realized):
            raise ActionReplayError("tap direction conflicts with requested endpoints")
    elif action in {"swipe", "scroll"}:
        if target_length <= 1.0e-12 or direction != realized:
            raise ActionReplayError(
                f"{action} direction conflicts with requested endpoints"
            )
    else:
        raise ActionReplayError("template transport supports tap/scroll/swipe")

    dimensions = np.asarray((width, height), dtype=np.float64)
    source_points = values[:, 1:3].astype(np.float64) * dimensions[None, :]
    if (
        np.any(values[:, 1:3] < 0.0)
        or np.any(values[:, 1:3] > 1.0)
        or np.any(np.diff(values[:, 7].astype(np.float64)) < 0.0)
        or float(values[0, 7]) != 0.0
        or np.any(values[[0, -1], 0] <= 0.0)
        or np.any(values[[0, -1], 8] <= 0.0)
    ):
        raise ActionReplayError(
            "touch template violates screen, time, or active-endpoint invariants"
        )
    expected_dx_dy = np.zeros((len(values), 2), dtype=np.float32)
    source_consecutive = (values[1:, 0] > 0.0) & (values[:-1, 0] > 0.0)
    source_differences = np.diff(values[:, 1:3], axis=0)
    expected_dx_dy[1:][source_consecutive] = source_differences[
        source_consecutive
    ]
    if not np.array_equal(values[:, 5:7], expected_dx_dy):
        raise ActionReplayError("touch template has inconsistent dx/dy fields")
    source_start = source_points[0]
    source_end = source_points[-1]
    source_duration = float(values[-1, 7] - values[0, 7]) * 1000.0
    if source_duration <= 0.0:
        raise ActionReplayError("touch template duration must be positive")

    encoded_endpoints = np.asarray((start, end), dtype=np.float64)
    encoded_endpoints = (
        encoded_endpoints / dimensions[None, :]
    ).astype(np.float32)
    decoded_endpoints = (
        encoded_endpoints.astype(np.float64) * dimensions[None, :]
    )
    endpoint_encoding_errors = np.linalg.norm(
        decoded_endpoints - np.asarray((start, end), dtype=np.float64), axis=1
    )
    if float(np.max(endpoint_encoding_errors)) > 5.0e-4:
        raise ActionReplayError(
            "requested endpoint cannot be represented on the detector grid"
        )
    decoded_delta = decoded_endpoints[1] - decoded_endpoints[0]
    decoded_length = float(np.linalg.norm(decoded_delta))
    if target_length > 1.0e-12 and decoded_length <= 1.0e-12:
        raise ActionReplayError(
            "requested movement collapses on the detector grid"
        )
    if decoded_length > 1.0e-12:
        decoded_direction = _direction8(
            float(decoded_delta[0]), float(decoded_delta[1])
        )
        if decoded_direction != realized:
            raise ActionReplayError(
                "requested direction changes on the detector grid"
            )

    duration_seconds = np.float32(duration / 1000.0)
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ActionReplayError(
            "requested duration cannot be represented on the detector grid"
        )
    target_times = np.linspace(
        0.0, float(duration_seconds), len(values), dtype=np.float32
    )
    if (
        not np.isfinite(target_times).all()
        or abs(float(target_times[-1]) * 1000.0 - duration) > 1.0e-3
    ):
        raise ActionReplayError(
            "requested duration cannot be represented within one microsecond"
        )
    if (
        np.array_equal(encoded_endpoints, values[[0, -1], 1:3])
        and np.array_equal(target_times, values[:, 7])
    ):
        return values.copy(), 1.0

    u = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    source_chord = source_end - source_start
    source_length = float(np.linalg.norm(source_chord))
    source_baseline = source_start[None, :] + u[:, None] * source_chord[None, :]
    residual = source_points - source_baseline
    residual[0] = 0.0
    residual[-1] = 0.0
    if source_length > 1.0e-12 and target_length > 1.0e-12:
        source_unit = source_chord / source_length
        target_unit = target_chord / target_length
        cosine = float(np.dot(source_unit, target_unit))
        sine = float(source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0])
        rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        residual = residual @ rotation.T
    target_baseline = start[None, :] + u[:, None] * target_chord[None, :]

    residual_scale = 1.0
    for axis, upper in ((0, width), (1, height)):
        delta = residual[:, axis]
        positive = delta > 0.0
        negative = delta < 0.0
        if np.any(positive):
            residual_scale = min(
                residual_scale,
                float(np.min((upper - target_baseline[positive, axis]) / delta[positive])),
            )
        if np.any(negative):
            residual_scale = min(
                residual_scale,
                float(np.min((0.0 - target_baseline[negative, axis]) / delta[negative])),
            )
    residual_scale = max(0.0, min(1.0, residual_scale))
    if residual_scale < 1.0:
        residual_scale = float(np.nextafter(residual_scale, 0.0))
    points = target_baseline + residual_scale * residual
    points[0] = start
    points[-1] = end
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > width)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > height)
    ):
        raise ActionReplayError("transported donor left the screen")

    output = values.copy()
    output[:, 1:3] = (points / dimensions[None, :]).astype(np.float32)
    output[:, 5:7] = 0.0
    consecutive = (output[1:, 0] > 0.0) & (output[:-1, 0] > 0.0)
    differences = np.diff(output[:, 1:3], axis=0)
    output[1:, 5][consecutive] = differences[consecutive, 0]
    output[1:, 6][consecutive] = differences[consecutive, 1]
    output[:, 7] = target_times
    output_endpoints = (
        output[[0, -1], 1:3].astype(np.float64) * dimensions[None, :]
    )
    output_errors = np.linalg.norm(
        output_endpoints - np.asarray((start, end), dtype=np.float64), axis=1
    )
    if (
        not np.isfinite(output).all()
        or float(np.max(output_errors)) > 5.0e-4
        or (
            target_length > 1.0e-12
            and np.array_equal(output[0, 1:3], output[-1, 1:3])
        )
    ):
        raise ActionReplayError(
            "transported output cannot represent its exact request"
        )
    return output, residual_scale


@dataclass(frozen=True, order=True)
class ReplayBucket:
    action: str
    orientation_id: int
    direction: str
    duration_bucket: str | None = None
    pinch_scale_direction: str | None = None

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_ACTIONS:
            raise ActionReplayError(f"unsupported replay action {self.action!r}")
        if self.orientation_id not in (0, 1, 3):
            raise ActionReplayError("replay orientation must be 0, 1, or 3")
        allowed_directions = set(DIRECTION8)
        if self.action == "pinch":
            allowed_directions.add(STATIONARY)
            if self.direction not in allowed_directions:
                raise ActionReplayError("invalid pinch center direction")
            if self.duration_bucket is not None:
                raise ActionReplayError("pinch buckets do not use duration quartiles")
            if self.pinch_scale_direction not in PINCH_SCALE_DIRECTIONS:
                raise ActionReplayError("pinch bucket must be classified as in or out")
        else:
            if self.direction not in allowed_directions:
                raise ActionReplayError("invalid eight-sector direction")
            if self.duration_bucket not in DURATION_BUCKET_NAMES:
                raise ActionReplayError("scroll/swipe bucket needs a duration quartile")
            if self.pinch_scale_direction is not None:
                raise ActionReplayError("single-pointer bucket cannot have pinch scale")

    def stable_text(self) -> str:
        return "|".join(
            (
                self.action,
                str(self.orientation_id),
                self.direction,
                self.duration_bucket or "-",
                self.pinch_scale_direction or "-",
            )
        )


@dataclass(frozen=True)
class PrimitiveDescriptor:
    primitive_id: str
    action: str
    source_event_index: int
    source_event_id: str
    source_user_id: int
    source_session_id: int
    orientation_id: int
    duration_ms: float
    row_count: int
    frame_count: int
    observable_update_count: int
    observable_update_rate_hz: float
    start_xy_px: tuple[float, float]
    end_xy_px: tuple[float, float]
    displacement_px: float
    speed_px_per_s: float
    bbox_xyxy_px: tuple[float, float, float, float]
    bucket: ReplayBucket
    pinch_endpoint_geometry: PinchEndpointGeometry | None = None


@dataclass(frozen=True)
class ReplayRows:
    """Raw or minimally deformed rows for one complete Android contact."""

    primitive_id: str
    action: str
    orientation_id: int
    t_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray
    size: np.ndarray
    pointer_count: np.ndarray
    pointer_id: np.ndarray
    android_action: np.ndarray
    frame_index: np.ndarray
    active_mask: np.ndarray
    valid_mask: np.ndarray
    source_duration_ms: float
    replay_duration_ms: float
    time_warp_ratio: float
    translation_px: tuple[float, float]
    spatial_matrix_xy: tuple[tuple[int, int], tuple[int, int]] = (
        (1, 0),
        (0, 1),
    )
    spatial_transform_name: str = "identity"

@dataclass(frozen=True)
class ReplayObservation:
    descriptor: PrimitiveDescriptor
    rows: ReplayRows
    observation: TouchObservation
    detector_duration_ms: float


@dataclass(frozen=True)
class ReplayRequest:
    """Geometry and duration that a replay donor must preserve."""

    action: str
    orientation_id: int
    direction: str
    target_duration_ms: float
    pinch_scale_direction: str | None = None


@dataclass(frozen=True, order=True)
class ReplayGeometry:
    """Duration-independent Android gesture class used for capacity planning."""

    action: str
    orientation_id: int
    direction: str
    pinch_scale_direction: str | None = None

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_ACTIONS:
            raise ActionReplayError(f"unsupported replay action {self.action!r}")
        if self.orientation_id not in (0, 1, 3):
            raise ActionReplayError("replay orientation must be 0, 1, or 3")
        if self.action == "pinch":
            if self.direction not in set(DIRECTION8) | {STATIONARY}:
                raise ActionReplayError("invalid pinch center direction")
            if self.pinch_scale_direction not in PINCH_SCALE_DIRECTIONS:
                raise ActionReplayError("pinch geometry must preserve in/out")
        else:
            if self.direction not in DIRECTION8:
                raise ActionReplayError("invalid eight-sector direction")
            if self.pinch_scale_direction is not None:
                raise ActionReplayError(
                    "single-pointer geometry cannot have pinch scale"
                )

    @property
    def direction_orbit(self) -> str:
        return direction_orbit(self.direction)

    def stable_text(self) -> str:
        return "|".join(
            (
                self.action,
                str(self.orientation_id),
                self.direction,
                self.pinch_scale_direction or "-",
            )
        )


@dataclass(frozen=True)
class ReplayIsometry:
    """A fitted D4 transform plus optional lattice endpoint correction."""

    name: str
    matrix_xy: tuple[tuple[float, float], tuple[float, float]]
    translation_px: tuple[float, float]
    source_direction: str
    target_direction: str
    requested_anchor_px: tuple[float, float]
    output_anchor_px: tuple[float, float]
    anchor_error_px: float
    rank: int
    requested_endpoint_px: tuple[float, float] | None = None
    output_endpoint_px: tuple[float, float] | None = None
    endpoint_error_px: float = 0.0
    spatial_scale: float = 1.0
    requested_distance_ratio: float = 1.0
    endpoint_residual_px: tuple[float, float] = (0.0, 0.0)
    endpoint_residual_fraction: float = 0.0
    quantize_pixel_lattice: bool = False


@dataclass(frozen=True)
class IsometricReplayAllocation:
    descriptor: PrimitiveDescriptor
    isometry: ReplayIsometry

    @property
    def primitive_id(self) -> str:
        return self.descriptor.primitive_id


@dataclass(frozen=True)
class PinchEndpointReplayAllocation:
    """One unused pinch donor plus its bounded exact endpoint deformation."""

    descriptor: PrimitiveDescriptor
    endpoint_fit: PinchEndpointFit

    @property
    def primitive_id(self) -> str:
        return self.descriptor.primitive_id


# The eight symmetries of the square.  Coefficients are integral, so these
# transforms retain the raw coordinate lattice and never resample a path.
_D4_MATRICES: tuple[
    tuple[str, tuple[tuple[int, int], tuple[int, int]], int], ...
] = (
    ("identity", ((1, 0), (0, 1)), 0),
    ("reflect_x", ((-1, 0), (0, 1)), 1),
    ("reflect_y", ((1, 0), (0, -1)), 1),
    ("rotate_180", ((-1, 0), (0, -1)), 2),
    ("rotate_90", ((0, -1), (1, 0)), 3),
    ("rotate_270", ((0, 1), (-1, 0)), 3),
    ("reflect_diagonal", ((0, 1), (1, 0)), 4),
    ("reflect_antidiagonal", ((0, -1), (-1, 0)), 4),
)


def direction_orbit(direction: str) -> str:
    """Return the D4 orbit of an eight-sector (or stationary) direction."""

    if direction == STATIONARY:
        return STATIONARY
    if direction not in DIRECTION8:
        raise ActionReplayError(f"invalid direction {direction!r}")
    return (
        "cardinal"
        if direction in {"right", "down", "left", "up"}
        else "diagonal"
    )


def _geometry_for_descriptor(descriptor: PrimitiveDescriptor) -> ReplayGeometry:
    return ReplayGeometry(
        action=descriptor.action,
        orientation_id=descriptor.orientation_id,
        direction=descriptor.bucket.direction,
        pinch_scale_direction=descriptor.bucket.pinch_scale_direction,
    )


def _geometry_for_request(request: ReplayRequest | ReplayGeometry) -> ReplayGeometry:
    if isinstance(request, ReplayGeometry):
        return request
    return ReplayGeometry(
        action=request.action,
        orientation_id=request.orientation_id,
        direction=request.direction,
        pinch_scale_direction=request.pinch_scale_direction,
    )


def _geometry_orbit_key(geometry: ReplayGeometry) -> tuple[str, int, str, str]:
    return (
        geometry.action,
        geometry.orientation_id,
        geometry.direction_orbit,
        geometry.pinch_scale_direction or "-",
    )


def _small_integer_max_flow(
    type_capacities: Sequence[int],
    demand_capacities: Sequence[int],
    compatible: Sequence[Sequence[bool]],
) -> tuple[int, dict[tuple[int, int], int]]:
    """Solve the tiny donor-type/demand-node transportation problem."""

    type_count = len(type_capacities)
    demand_count = len(demand_capacities)
    source = type_count + demand_count
    sink = source + 1
    node_count = sink + 1
    residual: list[dict[int, int]] = [dict() for _ in range(node_count)]
    original: dict[tuple[int, int], int] = {}

    def edge(left: int, right: int, capacity: int) -> None:
        residual[left][right] = residual[left].get(right, 0) + int(capacity)
        residual[right].setdefault(left, 0)
        original[(left, right)] = original.get((left, right), 0) + int(capacity)

    for type_index, capacity in enumerate(type_capacities):
        edge(source, type_index, int(capacity))
    infinite = int(sum(demand_capacities))
    for type_index, row in enumerate(compatible):
        for demand_index, allowed in enumerate(row):
            if allowed:
                edge(type_index, type_count + demand_index, infinite)
    for demand_index, capacity in enumerate(demand_capacities):
        edge(type_count + demand_index, sink, int(capacity))

    total = 0
    while True:
        parent = [-1] * node_count
        parent[source] = source
        queue = [source]
        for left in queue:
            for right in sorted(residual[left]):
                if residual[left][right] > 0 and parent[right] < 0:
                    parent[right] = left
                    queue.append(right)
                    if right == sink:
                        break
            if parent[sink] >= 0:
                break
        if parent[sink] < 0:
            break
        amount = infinite
        cursor = sink
        while cursor != source:
            left = parent[cursor]
            amount = min(amount, residual[left][cursor])
            cursor = left
        cursor = sink
        while cursor != source:
            left = parent[cursor]
            residual[left][cursor] -= amount
            residual[cursor][left] = residual[cursor].get(left, 0) + amount
            cursor = left
        total += amount

    assignment: dict[tuple[int, int], int] = {}
    for type_index in range(type_count):
        for demand_index in range(demand_count):
            right = type_count + demand_index
            capacity = original.get((type_index, right), 0)
            if capacity:
                used = capacity - residual[type_index][right]
                if used:
                    assignment[(type_index, demand_index)] = int(used)
    return total, assignment


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _direction8(dx: float, dy: float) -> str:
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ActionReplayError("direction vector must be finite")
    angle = float(np.arctan2(dy, dx))
    index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
    return DIRECTION8[index]


def _fitted_isometries(
    descriptor: PrimitiveDescriptor,
    *,
    target_direction: str,
    target_anchor_px: tuple[float, float] | None = None,
    target_endpoint_px: tuple[float, float] | None = None,
    require_lattice_endpoints: bool = False,
) -> tuple[ReplayIsometry, ...]:
    """Enumerate exact-direction D4 transforms that fit the physical screen."""

    source_geometry = _geometry_for_descriptor(descriptor)
    if target_endpoint_px is None and direction_orbit(
        source_geometry.direction
    ) != direction_orbit(target_direction):
        return ()
    width, height = screen_dimensions_for_orientation(descriptor.orientation_id)
    source_anchor = np.asarray(descriptor.start_xy_px, dtype=np.float64)
    requested_anchor = np.asarray(
        descriptor.start_xy_px if target_anchor_px is None else target_anchor_px,
        dtype=np.float64,
    )
    if requested_anchor.shape != (2,) or not np.isfinite(requested_anchor).all():
        raise ActionReplayError("target anchor must contain two finite coordinates")
    if (
        requested_anchor[0] < 0.0
        or requested_anchor[0] > width
        or requested_anchor[1] < 0.0
        or requested_anchor[1] > height
    ):
        raise ActionReplayError("target anchor leaves the physical screen")
    x0, y0, x1, y1 = descriptor.bbox_xyxy_px
    corners = np.asarray(
        ((x0, y0), (x0, y1), (x1, y0), (x1, y1)),
        dtype=np.float64,
    )
    displacement = np.asarray(descriptor.end_xy_px, dtype=np.float64) - source_anchor
    fitted: list[ReplayIsometry] = []
    if target_endpoint_px is not None:
        endpoint = np.asarray(target_endpoint_px, dtype=np.float64)
        if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
            raise ActionReplayError("target endpoint must contain two finite coordinates")
        if np.any(endpoint < 0.0) or endpoint[0] > width or endpoint[1] > height:
            raise ActionReplayError("target endpoint leaves the physical screen")
        if not np.allclose(requested_anchor, np.rint(requested_anchor), atol=1.0e-6):
            raise ActionReplayError("endpoint control anchor must lie on the Android pixel lattice")
        if not np.allclose(endpoint, np.rint(endpoint), atol=1.0e-6):
            raise ActionReplayError("target endpoint must lie on the Android pixel lattice")
        requested_anchor = np.rint(requested_anchor)
        endpoint = np.rint(endpoint)
        source_norm = float(np.linalg.norm(displacement))
        target_displacement = endpoint - requested_anchor
        target_norm = float(np.linalg.norm(target_displacement))
        if source_norm <= 1.0e-9 or target_norm <= 1.0e-9:
            raise ActionReplayError("endpoint-bound replay needs non-stationary endpoints")
        distance_ratio = target_norm / source_norm
        for name, matrix_tuple, rank in _D4_MATRICES:
            matrix = np.asarray(matrix_tuple, dtype=np.float64)
            translation = requested_anchor - matrix @ source_anchor
            base_corners = corners @ matrix.T + translation
            base_endpoint = (
                matrix @ np.asarray(descriptor.end_xy_px, dtype=np.float64)
                + translation
            )
            residual = endpoint - base_endpoint
            if not np.allclose(residual, np.rint(residual), atol=1.0e-6):
                # Adding only integer corrections preserves the donor's
                # empirical physical-pixel fractional residue distribution.
                continue
            residual = np.rint(residual)
            residual_fraction = float(np.linalg.norm(residual) / target_norm)
            # A smooth integer correction may move every intermediate point
            # anywhere between zero and the endpoint residual.  This
            # conservative Minkowski bound proves no point can need clipping.
            minimum = np.min(base_corners, axis=0) + np.minimum(residual, 0.0)
            maximum = np.max(base_corners, axis=0) + np.maximum(residual, 0.0)
            if (
                np.any(minimum < 0.0)
                or maximum[0] > width
                or maximum[1] > height
                or residual_fraction > 0.10
            ):
                continue
            fitted.append(
                ReplayIsometry(
                    name=f"endpoint_d4_error_diffusion_{name}",
                    matrix_xy=matrix_tuple,
                    translation_px=(float(translation[0]), float(translation[1])),
                    source_direction=source_geometry.direction,
                    target_direction=target_direction,
                    requested_anchor_px=tuple(float(value) for value in requested_anchor),
                    output_anchor_px=tuple(float(value) for value in requested_anchor),
                    anchor_error_px=0.0,
                    rank=rank,
                    requested_endpoint_px=tuple(float(value) for value in endpoint),
                    output_endpoint_px=tuple(float(value) for value in endpoint),
                    endpoint_error_px=0.0,
                    spatial_scale=1.0,
                    requested_distance_ratio=distance_ratio,
                    endpoint_residual_px=(float(residual[0]), float(residual[1])),
                    endpoint_residual_fraction=residual_fraction,
                    quantize_pixel_lattice=True,
                )
            )
        return tuple(
            sorted(
                fitted,
                key=lambda value: (
                    value.endpoint_residual_fraction,
                    value.rank,
                    value.name,
                ),
            )
        )
    for name, matrix_tuple, rank in _D4_MATRICES:
        matrix = np.asarray(matrix_tuple, dtype=np.float64)
        transformed_displacement = matrix @ displacement
        if source_geometry.direction == STATIONARY:
            mapped_direction = STATIONARY
        else:
            mapped_direction = _direction8(
                float(transformed_displacement[0]),
                float(transformed_displacement[1]),
            )
        if mapped_direction != target_direction:
            continue
        transformed_corners = corners @ matrix.T
        minimum = np.min(transformed_corners, axis=0)
        maximum = np.max(transformed_corners, axis=0)
        lower = -minimum
        upper = np.asarray((width, height), dtype=np.float64) - maximum
        if np.any(lower > upper + 1.0e-9):
            continue
        desired_translation = requested_anchor - matrix @ source_anchor
        translation = np.minimum(np.maximum(desired_translation, lower), upper)
        output_anchor = matrix @ source_anchor + translation
        output_endpoint = (
            matrix @ np.asarray(descriptor.end_xy_px, dtype=np.float64)
            + translation
        )
        if require_lattice_endpoints and (
            not np.allclose(output_anchor, np.rint(output_anchor), atol=1.0e-6)
            or not np.allclose(
                output_endpoint, np.rint(output_endpoint), atol=1.0e-6
            )
        ):
            continue
        error = float(np.linalg.norm(output_anchor - requested_anchor))
        fitted.append(
            ReplayIsometry(
                name=name,
                matrix_xy=matrix_tuple,
                translation_px=(float(translation[0]), float(translation[1])),
                source_direction=source_geometry.direction,
                target_direction=target_direction,
                requested_anchor_px=(
                    float(requested_anchor[0]),
                    float(requested_anchor[1]),
                ),
                output_anchor_px=(float(output_anchor[0]), float(output_anchor[1])),
                anchor_error_px=error,
                rank=rank,
            )
        )
    return tuple(
        sorted(
            fitted,
            key=lambda value: (
                value.anchor_error_px,
                value.rank,
                value.name,
            ),
        )
    )


def fit_replay_isometry(
    descriptor: PrimitiveDescriptor,
    *,
    target_direction: str,
    target_anchor_px: tuple[float, float] | None = None,
) -> ReplayIsometry:
    """Fit the least-deforming legal D4 transform for one donor."""

    fitted = _fitted_isometries(
        descriptor,
        target_direction=target_direction,
        target_anchor_px=target_anchor_px,
    )
    if not fitted:
        raise ActionReplayError(
            "no in-screen D4 isometry maps donor direction "
            f"{descriptor.bucket.direction} to {target_direction}"
        )
    return fitted[0]


def duration_bucket(action: str, duration_ms: float) -> str:
    if action not in DURATION_BUCKET_EDGES_MS:
        raise ActionReplayError("duration quartiles exist only for scroll/swipe")
    if not np.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ActionReplayError("duration must be finite and positive")
    index = int(
        np.searchsorted(
            np.asarray(DURATION_BUCKET_EDGES_MS[action], dtype=np.float64),
            float(duration_ms),
            side="right",
        )
    )
    return DURATION_BUCKET_NAMES[index]


def _frame_bounds(frame_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(frame_index):
        raise ActionReplayError("primitive has no MotionEvent rows")
    if np.any(np.diff(frame_index.astype(np.int64)) < 0):
        raise ActionReplayError("frame indices are not ordered")
    changes = np.flatnonzero(frame_index[1:] != frame_index[:-1]) + 1
    bounds = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            changes,
            np.asarray([len(frame_index)], dtype=np.int64),
        )
    )
    return bounds[:-1], bounds[1:]


def _pinch_geometry(
    *,
    t_ms: np.ndarray,
    frame_index: np.ndarray,
    pointer_id: np.ndarray,
    android_action: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Return centers, spans, and detector-visible two-pointer update count."""

    starts, ends = _frame_bounds(frame_index)

    def simultaneous_geometry(
        left: int,
        right: int,
    ) -> tuple[np.ndarray, float, float] | None:
        positions = np.arange(int(left), int(right), dtype=np.int64)
        selected: list[int] = []
        for pointer in np.unique(pointer_id[positions]):
            candidates = positions[pointer_id[positions] == pointer]
            base = android_action[candidates].astype(np.int64) & ACTION_MASK
            live = candidates[~np.isin(base, (ACTION_UP, ACTION_CANCEL))]
            if len(live):
                selected.append(int(live[-1]))
        if len(selected) < 2:
            return None
        chosen = np.asarray(
            sorted(selected, key=lambda index: int(pointer_id[index]))[:2],
            dtype=np.int64,
        )
        points = np.column_stack((x_px[chosen], y_px[chosen])).astype(np.float64)
        return (
            np.mean(points, axis=0),
            float(np.linalg.norm(points[1] - points[0])),
            float(t_ms[int(chosen[-1])]),
        )

    geometries: list[tuple[np.ndarray, float, float]] = []
    for left, right in zip(starts, ends):
        geometry = simultaneous_geometry(int(left), int(right))
        if geometry is not None:
            geometries.append(geometry)
    first = geometries[0] if geometries else None
    last = geometries[-1] if geometries else None
    if first is None or last is None or first[1] <= 0.0 or last[1] <= 0.0:
        raise ActionReplayError("pinch lacks valid simultaneous two-pointer frames")
    update_times = np.asarray([value[2] for value in geometries], dtype=np.float64)
    if not np.isfinite(update_times).all():
        raise ActionReplayError("pinch has nonfinite update time")
    if update_times[0] > 0.0:
        update_times = np.concatenate((np.asarray([0.0]), update_times))
    observable_update_count = int(len(np.unique(update_times)))
    if observable_update_count <= 0:
        raise ActionReplayError("pinch has no observable updates")
    return first[0], last[0], first[1], last[1], observable_update_count


def classify_replay_request(
    *,
    action: str,
    orientation_id: int,
    target_duration_ms: float,
    x_px: Iterable[object],
    y_px: Iterable[object],
    pointer_id: Iterable[object] | None = None,
    android_action: Iterable[object] | None = None,
    frame_index: Iterable[object] | None = None,
) -> ReplayRequest:
    """Derive the label-preserving replay stratum from arbitrary raw rows."""

    if action not in SUPPORTED_ACTIONS:
        raise ActionReplayError(f"unsupported replay action {action!r}")
    if int(orientation_id) not in (0, 1, 3):
        raise ActionReplayError("request orientation must be 0, 1, or 3")
    duration = float(target_duration_ms)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ActionReplayError("request duration must be finite and positive")
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ActionReplayError("request XY rows are malformed")
    width, height = screen_dimensions_for_orientation(int(orientation_id))
    if (
        np.any(x < 0.0)
        or np.any(x > width)
        or np.any(y < 0.0)
        or np.any(y > height)
    ):
        raise ActionReplayError("request rows leave the physical screen")
    pinch_scale_direction: str | None = None
    if action in DURATION_BUCKET_EDGES_MS:
        dx = float(x[-1] - x[0])
        dy = float(y[-1] - y[0])
        if float(np.hypot(dx, dy)) <= 1.0e-9:
            raise ActionReplayError("single-pointer request has zero displacement")
        direction = _direction8(dx, dy)
    else:
        if pointer_id is None or android_action is None or frame_index is None:
            raise ActionReplayError("pinch request needs pointer/action/frame rows")
        pointers = np.asarray(pointer_id, dtype=np.int64)
        actions = np.asarray(android_action, dtype=np.int64)
        frames = np.asarray(frame_index, dtype=np.int64)
        if any(value.shape != x.shape for value in (pointers, actions, frames)):
            raise ActionReplayError("pinch request row arrays are misaligned")
        start_xy, end_xy, start_span, end_span, _ = _pinch_geometry(
            t_ms=np.arange(len(x), dtype=np.float64),
            frame_index=frames,
            pointer_id=pointers,
            android_action=actions,
            x_px=x,
            y_px=y,
        )
        dx, dy = end_xy - start_xy
        direction = (
            STATIONARY
            if float(np.hypot(dx, dy))
            <= PINCH_STATIONARY_DIAGONAL_FRACTION * float(np.hypot(width, height))
            else _direction8(float(dx), float(dy))
        )
        pinch_scale_direction = "in" if end_span < start_span else "out"
    return ReplayRequest(
        action=action,
        orientation_id=int(orientation_id),
        direction=direction,
        target_duration_ms=duration,
        pinch_scale_direction=pinch_scale_direction,
    )


class ActionReplayBank:
    """In-memory index over one raw HMOG action archive."""

    _EVENT_KEYS = (
        "event_id",
        "user_id",
        "session_id",
        "orientation_id",
        "touch_duration_ms",
        "event_offsets",
    )
    _FLAT_KEYS = (
        "flat_t_rel_ms",
        "flat_frame_index",
        "flat_pointer_count",
        "flat_pointer_id",
        "flat_action_code",
        "flat_x",
        "flat_y",
        "flat_pressure",
        "flat_size",
        "flat_active_mask",
        "flat_valid_mask",
    )

    def __init__(
        self,
        *,
        source_path: Path,
        source_sha256: str,
        action: str,
        arrays: Mapping[str, np.ndarray],
        descriptors: Sequence[PrimitiveDescriptor],
        rejected_counts: Mapping[str, int],
    ) -> None:
        self.source_path = Path(source_path)
        self.source_sha256 = str(source_sha256)
        self.action = str(action)
        self._arrays = dict(arrays)
        self.descriptors = tuple(descriptors)
        self.rejected_counts = dict(rejected_counts)
        self._by_id = {item.primitive_id: item for item in self.descriptors}
        if len(self._by_id) != len(self.descriptors):
            raise ActionReplayError("primitive identifiers are not unique")
        buckets: dict[ReplayBucket, list[str]] = {}
        for item in self.descriptors:
            buckets.setdefault(item.bucket, []).append(item.primitive_id)
        self._by_bucket = {
            bucket: tuple(sorted(identifiers))
            for bucket, identifiers in buckets.items()
        }

    @classmethod
    def from_hmog_npz(
        cls,
        path: str | Path,
        *,
        train_user_ids: Iterable[int],
        # Five-shot material is the victim's own recording, so the donor user is
        # the target user rather than a train user.  This keeps the default
        # train-only contract while letting a per-user pool name its own owner.
        additional_source_user_ids: Iterable[int] = (),
        allowed_source_event_ids: Iterable[object] | None = None,
        excluded_source_event_ids: Iterable[object] = (),
        expected_action: str | None = None,
    ) -> "ActionReplayBank":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        train_users = {int(value) for value in train_user_ids}
        train_users |= {int(value) for value in additional_source_user_ids}
        if not train_users:
            raise ActionReplayError("train_user_ids cannot be empty")
        allowed = (
            None
            if allowed_source_event_ids is None
            else {str(value) for value in allowed_source_event_ids}
        )
        if allowed is not None and not allowed:
            raise ActionReplayError("allowed_source_event_ids cannot be empty")
        excluded = {str(value) for value in excluded_source_event_ids}
        with np.load(source, allow_pickle=False) as archive:
            missing = [
                key
                for key in cls._EVENT_KEYS + cls._FLAT_KEYS + ("action_name",)
                if key not in archive.files
            ]
            if missing:
                raise ActionReplayError(f"raw archive is missing fields: {missing}")
            action = str(np.asarray(archive["action_name"]).item())
            arrays = {
                key: np.asarray(archive[key]).copy()
                for key in cls._EVENT_KEYS + cls._FLAT_KEYS
            }
        if action not in SUPPORTED_ACTIONS:
            raise ActionReplayError(f"archive action {action!r} is not replayable")
        if expected_action is not None and action != expected_action:
            raise ActionReplayError(
                f"archive action {action!r} does not match {expected_action!r}"
            )
        cls._validate_archive_shapes(arrays)
        source_digest = _sha256_file(source)
        descriptors: list[PrimitiveDescriptor] = []
        rejected: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        event_count = len(arrays["event_id"])
        offsets = arrays["event_offsets"]
        for event_index in range(event_count):
            user_id = int(arrays["user_id"][event_index])
            if user_id not in train_users:
                reject("non_train_user")
                continue
            event_id = str(arrays["event_id"][event_index])
            if allowed is not None and event_id not in allowed:
                reject("not_quality_accepted")
                continue
            if event_id in excluded:
                reject("excluded_genuine_source_event")
                continue
            orientation = int(arrays["orientation_id"][event_index])
            if orientation not in (0, 1, 3):
                reject("unsupported_orientation")
                continue
            left = int(offsets[event_index])
            right = int(offsets[event_index + 1])
            try:
                descriptor = cls._describe_event(
                    arrays=arrays,
                    source_sha256=source_digest,
                    action=action,
                    event_index=event_index,
                    event_id=event_id,
                    user_id=user_id,
                    orientation_id=orientation,
                    left=left,
                    right=right,
                )
            except ActionReplayError as error:
                reject(str(error))
                continue
            descriptors.append(descriptor)
        if not descriptors:
            raise ActionReplayError("no eligible train-only replay primitives")
        return cls(
            source_path=source,
            source_sha256=source_digest,
            action=action,
            arrays=arrays,
            descriptors=descriptors,
            rejected_counts=rejected,
        )

    @classmethod
    def _validate_archive_shapes(cls, arrays: Mapping[str, np.ndarray]) -> None:
        event_count = len(arrays["event_id"])
        for key in ("user_id", "session_id", "orientation_id", "touch_duration_ms"):
            if arrays[key].ndim != 1 or len(arrays[key]) != event_count:
                raise ActionReplayError(f"{key} length does not match event_id")
        offsets = arrays["event_offsets"]
        if (
            offsets.ndim != 1
            or len(offsets) != event_count + 1
            or int(offsets[0]) != 0
            or np.any(np.diff(offsets.astype(np.int64)) <= 0)
        ):
            raise ActionReplayError("event_offsets are not strict ragged offsets")
        flat_count = int(offsets[-1])
        for key in cls._FLAT_KEYS:
            if arrays[key].ndim != 1 or len(arrays[key]) != flat_count:
                raise ActionReplayError(f"{key} length does not match event_offsets")

    @classmethod
    def _describe_event(
        cls,
        *,
        arrays: Mapping[str, np.ndarray],
        source_sha256: str,
        action: str,
        event_index: int,
        event_id: str,
        user_id: int,
        orientation_id: int,
        left: int,
        right: int,
    ) -> PrimitiveDescriptor:
        if right - left < 2:
            raise ActionReplayError("too_few_rows")
        values = {
            key: np.asarray(arrays[key][left:right]) for key in cls._FLAT_KEYS
        }
        if not np.all(values["flat_valid_mask"] == 1):
            raise ActionReplayError("invalid_raw_row")
        t_ms = values["flat_t_rel_ms"].astype(np.float64)
        if not np.isfinite(t_ms).all() or np.any(np.diff(t_ms) < 0.0):
            raise ActionReplayError("unordered_event_time")
        t_rel_ms = t_ms - float(t_ms[0])
        x_px = values["flat_x"].astype(np.float64)
        y_px = values["flat_y"].astype(np.float64)
        pressure = values["flat_pressure"].astype(np.float64)
        size = values["flat_size"].astype(np.float64)
        if not (
            np.isfinite(x_px).all()
            and np.isfinite(y_px).all()
            and np.isfinite(pressure).all()
            and np.isfinite(size).all()
        ):
            raise ActionReplayError("nonfinite_raw_value")
        if np.any(pressure < 0.0) or np.any(pressure > 1.0):
            raise ActionReplayError("invalid_pressure")
        width, height = screen_dimensions_for_orientation(orientation_id)
        if (
            np.any(x_px < 0.0)
            or np.any(x_px > width)
            or np.any(y_px < 0.0)
            or np.any(y_px > height)
        ):
            raise ActionReplayError("source_out_of_screen")
        frame_index = values["flat_frame_index"].astype(np.int64)
        starts, ends = _frame_bounds(frame_index)
        base_action = values["flat_action_code"].astype(np.int64) & ACTION_MASK
        first_actions = set(base_action[int(starts[0]) : int(ends[0])].tolist())
        last_actions = set(base_action[int(starts[-1]) : int(ends[-1])].tolist())
        if ACTION_DOWN not in first_actions or ACTION_UP not in last_actions:
            raise ActionReplayError("incomplete_android_lifecycle")
        pointers = values["flat_pointer_id"].astype(np.int64)
        pointer_count = values["flat_pointer_count"].astype(np.int64)
        pinch_endpoint_geometry: PinchEndpointGeometry | None = None
        if action in ("scroll", "swipe"):
            if len(np.unique(pointers)) != 1 or np.any(pointer_count != 1):
                raise ActionReplayError("not_single_pointer")
            start_xy = np.asarray([x_px[0], y_px[0]], dtype=np.float64)
            end_xy = np.asarray([x_px[-1], y_px[-1]], dtype=np.float64)
            dx, dy = end_xy - start_xy
            direction = _direction8(float(dx), float(dy))
            pinch_direction = None
            update_times = t_rel_ms[ends - 1]
            if update_times[0] > 0.0:
                update_times = np.concatenate((np.asarray([0.0]), update_times))
            observable_update_count = int(len(np.unique(update_times)))
        else:
            if len(np.unique(pointers)) != 2 or np.max(pointer_count) != 2:
                raise ActionReplayError("not_two_pointer_pinch")
            if ACTION_POINTER_DOWN not in set(base_action.tolist()):
                raise ActionReplayError("missing_pointer_down")
            if ACTION_POINTER_UP not in set(base_action.tolist()):
                raise ActionReplayError("missing_pointer_up")
            (
                start_xy,
                end_xy,
                start_span,
                end_span,
                observable_update_count,
            ) = _pinch_geometry(
                t_ms=t_rel_ms,
                frame_index=frame_index,
                pointer_id=pointers,
                android_action=base_action,
                x_px=x_px,
                y_px=y_px,
            )
            try:
                pinch_endpoint_geometry = extract_live_two_pointer_endpoints(
                    t_ms=t_rel_ms,
                    frame_index=frame_index,
                    pointer_id=pointers,
                    android_action=base_action,
                    x_px=x_px,
                    y_px=y_px,
                )
            except PinchEndpointControlError:
                # Preserve the legacy replay pool.  Such a donor remains
                # usable by rigid D4 replay but is skipped by exact endpoint
                # allocation, whose stronger endpoint-time contract it lacks.
                pinch_endpoint_geometry = None
            dx, dy = end_xy - start_xy
            diagonal = float(np.hypot(width, height))
            direction = (
                STATIONARY
                if float(np.hypot(dx, dy)) <= PINCH_STATIONARY_DIAGONAL_FRACTION * diagonal
                else _direction8(float(dx), float(dy))
            )
            pinch_direction = "in" if end_span < start_span else "out"
        observed_duration = float(t_ms[-1] - t_ms[0])
        declared_duration = float(arrays["touch_duration_ms"][event_index])
        if (
            not np.isfinite(declared_duration)
            or declared_duration <= 0.0
            or observed_duration <= 0.0
            or abs(declared_duration - observed_duration) > 1.0
        ):
            raise ActionReplayError("duration_mismatch")
        displacement = float(np.hypot(dx, dy))
        speed = displacement / (declared_duration / 1000.0)
        observable_update_rate_hz = (
            float(observable_update_count) * 1000.0 / declared_duration
        )
        if (
            observable_update_count <= 0
            or not np.isfinite(observable_update_rate_hz)
            or observable_update_rate_hz <= 0.0
        ):
            raise ActionReplayError("invalid_observable_update_rate")
        bucket = ReplayBucket(
            action=action,
            orientation_id=orientation_id,
            direction=direction,
            duration_bucket=(
                duration_bucket(action, declared_duration)
                if action in DURATION_BUCKET_EDGES_MS
                else None
            ),
            pinch_scale_direction=pinch_direction,
        )
        primitive_id = sha256(
            f"{source_sha256}|{action}|{event_index}|{event_id}".encode("utf-8")
        ).hexdigest()
        return PrimitiveDescriptor(
            primitive_id=primitive_id,
            action=action,
            source_event_index=event_index,
            source_event_id=event_id,
            source_user_id=user_id,
            source_session_id=int(arrays["session_id"][event_index]),
            orientation_id=orientation_id,
            duration_ms=declared_duration,
            row_count=right - left,
            frame_count=len(starts),
            observable_update_count=observable_update_count,
            observable_update_rate_hz=observable_update_rate_hz,
            start_xy_px=(float(start_xy[0]), float(start_xy[1])),
            end_xy_px=(float(end_xy[0]), float(end_xy[1])),
            displacement_px=displacement,
            speed_px_per_s=speed,
            bbox_xyxy_px=(
                float(np.min(x_px)),
                float(np.min(y_px)),
                float(np.max(x_px)),
                float(np.max(y_px)),
            ),
            bucket=bucket,
            pinch_endpoint_geometry=pinch_endpoint_geometry,
        )

    @property
    def buckets(self) -> tuple[ReplayBucket, ...]:
        return tuple(sorted(self._by_bucket))

    def descriptor(self, primitive_id: str) -> PrimitiveDescriptor:
        try:
            return self._by_id[str(primitive_id)]
        except KeyError as error:
            raise ActionReplayError(f"unknown primitive {primitive_id!r}") from error

    def primitive_ids(self, bucket: ReplayBucket | None = None) -> tuple[str, ...]:
        if bucket is None:
            return tuple(sorted(self._by_id))
        return self._by_bucket.get(bucket, ())

    def raw_rows(self, primitive_id: str) -> ReplayRows:
        descriptor = self.descriptor(primitive_id)
        offsets = self._arrays["event_offsets"]
        left = int(offsets[descriptor.source_event_index])
        right = int(offsets[descriptor.source_event_index + 1])
        get = lambda name, dtype: np.asarray(  # noqa: E731
            self._arrays[name][left:right], dtype=dtype
        ).copy()
        t_ms = get("flat_t_rel_ms", np.float64)
        t_ms -= float(t_ms[0])
        return ReplayRows(
            primitive_id=descriptor.primitive_id,
            action=descriptor.action,
            orientation_id=descriptor.orientation_id,
            t_ms=t_ms,
            x_px=get("flat_x", np.float64),
            y_px=get("flat_y", np.float64),
            pressure=get("flat_pressure", np.float64),
            size=get("flat_size", np.float64),
            pointer_count=get("flat_pointer_count", np.int64),
            pointer_id=get("flat_pointer_id", np.int64),
            android_action=get("flat_action_code", np.int64),
            frame_index=get("flat_frame_index", np.int64),
            active_mask=get("flat_active_mask", np.uint8),
            valid_mask=get("flat_valid_mask", np.uint8),
            source_duration_ms=descriptor.duration_ms,
            replay_duration_ms=descriptor.duration_ms,
            time_warp_ratio=1.0,
            translation_px=(0.0, 0.0),
        )

    def partition_output_splits(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        seed: int = 42,
    ) -> "DonorSplitPools":
        split_weights = (
            {"train": 70.0, "development": 10.0, "test": 20.0}
            if weights is None
            else {str(key): float(value) for key, value in weights.items()}
        )
        if set(split_weights) != set(OUTPUT_SPLITS):
            raise ActionReplayError(
                f"output split weights must name exactly {OUTPUT_SPLITS!r}"
            )
        if any(not np.isfinite(value) or value <= 0 for value in split_weights.values()):
            raise ActionReplayError("output split weights must be finite and positive")
        total_weight = float(sum(split_weights.values()))
        by_split: dict[str, list[str]] = {split: [] for split in OUTPUT_SPLITS}
        by_split_bucket: dict[str, dict[ReplayBucket, tuple[str, ...]]] = {
            split: {} for split in OUTPUT_SPLITS
        }
        for bucket in self.buckets:
            identifiers = list(self.primitive_ids(bucket))
            identifiers.sort(
                key=lambda primitive_id: sha256(
                    f"{seed}|{bucket.stable_text()}|{primitive_id}".encode("utf-8")
                ).digest()
            )
            exact = {
                split: len(identifiers) * split_weights[split] / total_weight
                for split in OUTPUT_SPLITS
            }
            counts = {split: int(np.floor(exact[split])) for split in OUTPUT_SPLITS}
            remaining = len(identifiers) - sum(counts.values())
            remainder_order = sorted(
                OUTPUT_SPLITS,
                key=lambda split: (-(exact[split] - counts[split]), split),
            )
            for split in remainder_order[:remaining]:
                counts[split] += 1
            cursor = 0
            for split in OUTPUT_SPLITS:
                selected = tuple(identifiers[cursor : cursor + counts[split]])
                cursor += counts[split]
                by_split[split].extend(selected)
                if selected:
                    by_split_bucket[split][bucket] = selected
            if cursor != len(identifiers):
                raise AssertionError("split allocation did not consume a bucket")
        return DonorSplitPools(
            bank=self,
            seed=int(seed),
            by_split={key: tuple(value) for key, value in by_split.items()},
            by_split_bucket=by_split_bucket,
        )

    def partition_output_splits_for_demand(
        self,
        requests_by_split: Mapping[
            str, Sequence[ReplayRequest | ReplayGeometry]
        ],
        *,
        seed: int = 42,
    ) -> "DonorSplitPools":
        """Assign donor families to splits from actual D4-compatible demand."""

        unknown = set(requests_by_split) - set(OUTPUT_SPLITS)
        if unknown:
            raise ActionReplayError(f"unknown output splits in demand: {unknown}")
        demand: dict[
            tuple[str, int, str, str],
            dict[tuple[str, ReplayGeometry], int],
        ] = {}
        for split in OUTPUT_SPLITS:
            for request in requests_by_split.get(split, ()):  # type: ignore[arg-type]
                geometry = _geometry_for_request(request)
                if geometry.action != self.action:
                    raise ActionReplayError(
                        "demand action does not match replay bank action"
                    )
                key = _geometry_orbit_key(geometry)
                node = (split, geometry)
                groups = demand.setdefault(key, {})
                groups[node] = groups.get(node, 0) + 1

        donors_by_orbit: dict[tuple[str, int, str, str], list[str]] = {}
        for descriptor in self.descriptors:
            donors_by_orbit.setdefault(
                _geometry_orbit_key(_geometry_for_descriptor(descriptor)), []
            ).append(descriptor.primitive_id)

        reserved: dict[str, dict[ReplayGeometry, list[str]]] = {
            split: {} for split in OUTPUT_SPLITS
        }
        used: set[str] = set()
        for orbit_key, node_counts in sorted(demand.items()):
            identifiers = donors_by_orbit.get(orbit_key, [])
            requested = int(sum(node_counts.values()))
            if requested > len(identifiers):
                raise ActionReplayError(
                    "D4 orbit donor capacity is insufficient: "
                    f"{orbit_key}; requested={requested}; available={len(identifiers)}"
                )
            target_nodes = sorted(
                node_counts,
                key=lambda value: (
                    OUTPUT_SPLITS.index(value[0]),
                    value[1].stable_text(),
                ),
            )
            target_directions = sorted(
                {geometry.direction for _, geometry in target_nodes}
            )
            typed: dict[tuple[str, ...], list[str]] = {}
            for primitive_id in identifiers:
                descriptor = self.descriptor(primitive_id)
                feasible = tuple(
                    direction
                    for direction in target_directions
                    if _fitted_isometries(
                        descriptor,
                        target_direction=direction,
                    )
                )
                if feasible:
                    typed.setdefault(feasible, []).append(primitive_id)
            type_keys = sorted(typed)
            type_values: list[list[str]] = []
            for feasible in type_keys:
                values = typed[feasible]
                values.sort(
                    key=lambda primitive_id: sha256(
                        (
                            f"demand-partition|{seed}|{orbit_key}|"
                            f"{primitive_id}"
                        ).encode("utf-8")
                    ).digest()
                )
                type_values.append(values)
            compatible = [
                [geometry.direction in feasible for _, geometry in target_nodes]
                for feasible in type_keys
            ]
            total, assignment = _small_integer_max_flow(
                [len(values) for values in type_values],
                [node_counts[node] for node in target_nodes],
                compatible,
            )
            if total != requested:
                raise ActionReplayError(
                    "D4 in-screen donor matching is insufficient: "
                    f"{orbit_key}; matched={total}; requested={requested}"
                )
            cursors = [0] * len(type_values)
            for type_index in range(len(type_values)):
                for node_index, (split, geometry) in enumerate(target_nodes):
                    count = assignment.get((type_index, node_index), 0)
                    if not count:
                        continue
                    left = cursors[type_index]
                    right = left + count
                    selected = type_values[type_index][left:right]
                    if len(selected) != count:
                        raise AssertionError("demand flow over-consumed a donor type")
                    cursors[type_index] = right
                    reserved[split].setdefault(geometry, []).extend(selected)
                    overlap = used.intersection(selected)
                    if overlap:
                        raise AssertionError("demand planner reused a primitive")
                    used.update(selected)

        # Demand reservations are exact.  Surplus donors are still assigned to
        # one split so the legacy complete-partition invariant remains true.
        by_split: dict[str, list[str]] = {
            split: [
                primitive_id
                for geometry in sorted(reserved[split])
                for primitive_id in reserved[split][geometry]
            ]
            for split in OUTPUT_SPLITS
        }
        surplus = sorted(
            set(self.primitive_ids()) - used,
            key=lambda primitive_id: sha256(
                f"demand-surplus|{seed}|{primitive_id}".encode("utf-8")
            ).digest(),
        )
        # Surplus ownership is deterministic and approximately 70/10/20.  It
        # cannot alter a reserved request pool.
        thresholds = (70, 80)
        for primitive_id in surplus:
            value = int.from_bytes(
                sha256(f"demand-surplus-split|{seed}|{primitive_id}".encode()).digest()[:8],
                "big",
            ) % 100
            split = (
                "train"
                if value < thresholds[0]
                else "development"
                if value < thresholds[1]
                else "test"
            )
            by_split[split].append(primitive_id)

        by_split_bucket_lists: dict[str, dict[ReplayBucket, list[str]]] = {
            split: {} for split in OUTPUT_SPLITS
        }
        for split in OUTPUT_SPLITS:
            for primitive_id in by_split[split]:
                bucket = self.descriptor(primitive_id).bucket
                by_split_bucket_lists[split].setdefault(bucket, []).append(
                    primitive_id
                )
        return DonorSplitPools(
            bank=self,
            seed=int(seed),
            by_split={split: tuple(values) for split, values in by_split.items()},
            by_split_bucket={
                split: {
                    bucket: tuple(values)
                    for bucket, values in groups.items()
                }
                for split, groups in by_split_bucket_lists.items()
            },
            by_split_target_geometry={
                split: {
                    geometry: tuple(values)
                    for geometry, values in groups.items()
                }
                for split, groups in reserved.items()
            },
        )

    def common_observer_archive(self) -> dict[str, np.ndarray]:
        """Expose this already-loaded raw archive to the common observer."""

        values = dict(self._arrays)
        values["action_name"] = np.asarray(self.action)
        values["flat_key_index"] = np.full(
            len(values["flat_t_rel_ms"]), -1, dtype=np.int16
        )
        return values


class DonorSplitPools:
    """A complete, disjoint assignment of primitive families to output splits."""

    def __init__(
        self,
        *,
        bank: ActionReplayBank,
        seed: int,
        by_split: Mapping[str, tuple[str, ...]],
        by_split_bucket: Mapping[str, Mapping[ReplayBucket, tuple[str, ...]]],
        by_split_target_geometry: Mapping[
            str, Mapping[ReplayGeometry, tuple[str, ...]]
        ] | None = None,
    ) -> None:
        self.bank = bank
        self.seed = int(seed)
        self._by_split = {key: tuple(value) for key, value in by_split.items()}
        self._by_split_bucket = {
            split: {bucket: tuple(values) for bucket, values in groups.items()}
            for split, groups in by_split_bucket.items()
        }
        self._by_split_target_geometry = {
            split: {
                geometry: tuple(values) for geometry, values in groups.items()
            }
            for split, groups in (by_split_target_geometry or {}).items()
        }
        self._validate()

    def _validate(self) -> None:
        sets = {split: set(self._by_split[split]) for split in OUTPUT_SPLITS}
        for index, left in enumerate(OUTPUT_SPLITS):
            for right in OUTPUT_SPLITS[index + 1 :]:
                if sets[left] & sets[right]:
                    raise ActionReplayError("output donor pools overlap")
        union = set().union(*(sets[split] for split in OUTPUT_SPLITS))
        if union != set(self.bank.primitive_ids()):
            raise ActionReplayError("output donor pools do not cover the bank exactly")
        for split, groups in self._by_split_target_geometry.items():
            if split not in OUTPUT_SPLITS:
                raise ActionReplayError("demand plan names an unknown output split")
            planned: set[str] = set()
            for geometry, values in groups.items():
                if geometry.action != self.bank.action:
                    raise ActionReplayError("demand plan action differs from bank")
                if planned.intersection(values):
                    raise ActionReplayError("demand plan reuses a donor family")
                if not set(values).issubset(sets[split]):
                    raise ActionReplayError(
                        "demand reservation is not owned by its output split"
                    )
                planned.update(values)

    def primitive_ids(
        self,
        output_split: str,
        bucket: ReplayBucket | None = None,
    ) -> tuple[str, ...]:
        if output_split not in OUTPUT_SPLITS:
            raise ActionReplayError(f"unknown output split {output_split!r}")
        if bucket is None:
            return self._by_split[output_split]
        return self._by_split_bucket[output_split].get(bucket, ())

    @property
    def has_demand_plan(self) -> bool:
        return bool(self._by_split_target_geometry)

    def primitive_ids_for_target_geometry(
        self,
        output_split: str,
        geometry: ReplayGeometry,
    ) -> tuple[str, ...]:
        if output_split not in OUTPUT_SPLITS:
            raise ActionReplayError(f"unknown output split {output_split!r}")
        return self._by_split_target_geometry.get(output_split, {}).get(
            geometry, ()
        )

    def allocator(self, output_split: str, *, seed: int | None = None) -> "ReplayAllocator":
        return ReplayAllocator(
            pools=self,
            output_split=output_split,
            seed=self.seed if seed is None else int(seed),
        )


class ReplayAllocator:
    """Stateful without-replacement allocation within one output split."""

    def __init__(
        self,
        *,
        pools: DonorSplitPools,
        output_split: str,
        seed: int,
    ) -> None:
        if output_split not in OUTPUT_SPLITS:
            raise ActionReplayError(f"unknown output split {output_split!r}")
        self.pools = pools
        self.output_split = output_split
        self.seed = int(seed)
        self._remaining: dict[ReplayBucket, list[str]] = {}
        for bucket in pools.bank.buckets:
            values = list(pools.primitive_ids(output_split, bucket))
            values.sort(
                key=lambda primitive_id: sha256(
                    f"allocator|{seed}|{output_split}|{primitive_id}".encode("utf-8")
                ).digest(),
                reverse=True,
            )
            self._remaining[bucket] = values
        self._remaining_target_geometry: dict[ReplayGeometry, list[str]] = {}
        for geometry, identifiers in pools._by_split_target_geometry.get(  # noqa: SLF001
            output_split, {}
        ).items():
            values = list(identifiers)
            values.sort(
                key=lambda primitive_id: sha256(
                    (
                        f"geometry-allocator|{seed}|{output_split}|"
                        f"{geometry.stable_text()}|{primitive_id}"
                    ).encode("utf-8")
                ).digest(),
                reverse=True,
            )
            self._remaining_target_geometry[geometry] = values
        self._used: set[str] = set()

    @property
    def used_primitive_ids(self) -> frozenset[str]:
        return frozenset(self._used)

    def allocate(self, bucket: ReplayBucket) -> PrimitiveDescriptor:
        values = self._remaining.get(bucket)
        while values and values[-1] in self._used:
            values.pop()
        if not values:
            raise ActionReplayError(
                f"donor pool exhausted for {self.output_split} {bucket.stable_text()}"
            )
        primitive_id = values.pop()
        if primitive_id in self._used:
            raise AssertionError("without-replacement allocator reused a primitive")
        self._used.add(primitive_id)
        return self.pools.bank.descriptor(primitive_id)

    def _consume(self, bucket: ReplayBucket, primitive_id: str) -> None:
        if primitive_id in self._used:
            raise AssertionError("without-replacement allocator reused a primitive")
        values = self._remaining.get(bucket, [])
        try:
            values.remove(primitive_id)
        except ValueError as error:
            raise AssertionError("selected primitive left its source bucket") from error
        self._used.add(primitive_id)

    def allocate_request(
        self,
        *,
        orientation_id: int,
        direction: str,
        target_duration_ms: float,
        pinch_scale_direction: str | None = None,
        target_update_rate_hz: float | None = None,
        max_time_warp: float = DEFAULT_MAX_TIME_WARP,
    ) -> PrimitiveDescriptor:
        """Allocate an unused donor with exact source geometry."""

        if orientation_id not in (0, 1, 3):
            raise ActionReplayError("request orientation must be 0, 1, or 3")
        if not np.isfinite(target_duration_ms) or target_duration_ms <= 0.0:
            raise ActionReplayError("request duration must be finite and positive")
        if not np.isfinite(max_time_warp) or max_time_warp < 1.0:
            raise ActionReplayError("max_time_warp must be finite and at least one")
        if target_update_rate_hz is not None and (
            not np.isfinite(target_update_rate_hz) or target_update_rate_hz <= 0.0
        ):
            raise ActionReplayError(
                "target_update_rate_hz must be finite and positive"
            )
        action = self.pools.bank.action
        if action in DURATION_BUCKET_EDGES_MS:
            if direction not in DIRECTION8:
                raise ActionReplayError("request needs an eight-sector direction")
            if pinch_scale_direction is not None:
                raise ActionReplayError("single-pointer request cannot have pinch scale")
            exact_name = duration_bucket(action, target_duration_ms)
            exact_index = DURATION_BUCKET_NAMES.index(exact_name)
            candidate_groups = []
            for distance in range(len(DURATION_BUCKET_NAMES)):
                names = [
                    name
                    for name in DURATION_BUCKET_NAMES
                    if abs(DURATION_BUCKET_NAMES.index(name) - exact_index) == distance
                ]
                if names:
                    candidate_groups.append(
                        [
                            ReplayBucket(
                                action=action,
                                orientation_id=orientation_id,
                                direction=direction,
                                duration_bucket=name,
                            )
                            for name in names
                        ]
                    )
        else:
            if direction not in set(DIRECTION8) | {STATIONARY}:
                raise ActionReplayError("invalid pinch center direction")
            if pinch_scale_direction not in PINCH_SCALE_DIRECTIONS:
                raise ActionReplayError("pinch request must be classified in or out")
            candidate_groups = [[
                ReplayBucket(
                    action=action,
                    orientation_id=orientation_id,
                    direction=direction,
                    pinch_scale_direction=pinch_scale_direction,
                )
            ]]
        def compatible_available(
            buckets: Sequence[ReplayBucket],
        ) -> list[tuple[ReplayBucket, int, str]]:
            available: list[tuple[ReplayBucket, int, str]] = []
            for bucket in buckets:
                values = self._remaining.get(bucket, [])
                for index, primitive_id in enumerate(values):
                    if primitive_id not in self._used:
                        available.append((bucket, index, primitive_id))
            return available

        if target_update_rate_hz is None:
            available = []
            for group in candidate_groups:
                available = compatible_available(group)
                if available:
                    break
        else:
            available = compatible_available(
                [bucket for group in candidate_groups for bucket in group]
            )
        if available:
            # With a requested detector-grid rate, retained update count is
            # primary.  Otherwise, the first duration-bucket distance wins.
            # Actual source duration and identifier are deterministic ties.
            selected_bucket, selected_index, primitive_id = min(
                available,
                key=lambda item: (
                    (
                        abs(
                            self.pools.bank.descriptor(item[2]).observable_update_count
                            * 1000.0
                            / float(target_duration_ms)
                            - float(target_update_rate_hz)
                        )
                        if target_update_rate_hz is not None
                        else 0.0
                    ),
                    abs(
                        self.pools.bank.descriptor(item[2]).duration_ms
                        - float(target_duration_ms)
                    ),
                    item[2],
                ),
            )
            if self._remaining[selected_bucket][selected_index] != primitive_id:
                raise AssertionError("nearest-update donor index changed")
            self._consume(selected_bucket, primitive_id)
            return self.pools.bank.descriptor(primitive_id)
        raise ActionReplayError(
            "no unused donor for request geometry: "
            f"{action}|{orientation_id}|{direction}|{pinch_scale_direction or '-'}; "
            f"target_duration_ms={float(target_duration_ms):.6f}"
        )

    def allocate_isometric_request(
        self,
        *,
        orientation_id: int,
        direction: str,
        detector_duration_ms: float,
        pinch_scale_direction: str | None = None,
        target_update_count: float | None = None,
        target_update_rate_hz: float | None = None,
        target_anchor_px: tuple[float, float] | None = None,
        target_endpoint_px: tuple[float, float] | None = None,
        preferred_primitive_id: str | None = None,
        require_lattice_endpoints: bool = False,
    ) -> IsometricReplayAllocation:
        """Allocate one unused donor from the exact D4 geometry orbit."""

        duration = float(detector_duration_ms)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ActionReplayError(
                "detector_duration_ms must be finite and positive"
            )
        action = self.pools.bank.action
        geometry = ReplayGeometry(
            action=action,
            orientation_id=int(orientation_id),
            direction=direction,
            pinch_scale_direction=pinch_scale_direction,
        )
        if target_update_count is not None and (
            not np.isfinite(target_update_count) or target_update_count <= 0.0
        ):
            raise ActionReplayError(
                "target_update_count must be finite and positive"
            )
        if target_update_rate_hz is not None and (
            not np.isfinite(target_update_rate_hz)
            or target_update_rate_hz <= 0.0
        ):
            raise ActionReplayError(
                "target_update_rate_hz must be finite and positive"
            )
        implied_count = (
            None
            if target_update_rate_hz is None
            else float(target_update_rate_hz) * duration / 1000.0
        )
        if target_update_count is not None and implied_count is not None:
            tolerance = max(1.0e-6, 0.01 * float(target_update_count))
            if abs(float(target_update_count) - implied_count) > tolerance:
                raise ActionReplayError(
                    "target update count and rate describe different windows"
                )
        desired_count = (
            float(target_update_count)
            if target_update_count is not None
            else implied_count
        )

        if self.pools.has_demand_plan:
            planned = self._remaining_target_geometry.get(geometry, [])
            candidate_ids = [
                primitive_id
                for primitive_id in planned
                if primitive_id not in self._used
            ]
        else:
            planned = None
            candidate_ids = [
                primitive_id
                for primitive_id in self.pools.primitive_ids(self.output_split)
                if primitive_id not in self._used
                and (
                    (
                        target_endpoint_px is not None
                        and self.pools.bank.descriptor(primitive_id).orientation_id
                        == geometry.orientation_id
                        and self.pools.bank.descriptor(primitive_id).bucket.pinch_scale_direction
                        == geometry.pinch_scale_direction
                    )
                    or _geometry_orbit_key(
                        _geometry_for_descriptor(
                            self.pools.bank.descriptor(primitive_id)
                        )
                    )
                    == _geometry_orbit_key(geometry)
                )
            ]
        if preferred_primitive_id is not None and preferred_primitive_id in candidate_ids:
            preferred_descriptor = self.pools.bank.descriptor(
                preferred_primitive_id
            )
            preferred_fits = _fitted_isometries(
                preferred_descriptor,
                target_direction=direction,
                target_anchor_px=target_anchor_px,
                target_endpoint_px=target_endpoint_px,
                require_lattice_endpoints=require_lattice_endpoints,
            )
            if preferred_fits:
                preferred_isometry = preferred_fits[0]
                if (
                    target_endpoint_px is None
                    or (
                        ENDPOINT_MIN_SPATIAL_SCALE - 1.0e-12
                        <= preferred_isometry.requested_distance_ratio
                        <= ENDPOINT_MAX_SPATIAL_SCALE + 1.0e-12
                    )
                ):
                    if planned is not None:
                        planned.remove(preferred_primitive_id)
                    self._consume(
                        preferred_descriptor.bucket, preferred_primitive_id
                    )
                    return IsometricReplayAllocation(
                        descriptor=preferred_descriptor,
                        isometry=preferred_isometry,
                    )
        candidates: list[
            tuple[PrimitiveDescriptor, ReplayIsometry]
        ] = []
        for primitive_id in candidate_ids:
            descriptor = self.pools.bank.descriptor(primitive_id)
            fitted = _fitted_isometries(
                descriptor,
                target_direction=direction,
                target_anchor_px=target_anchor_px,
                target_endpoint_px=target_endpoint_px,
                require_lattice_endpoints=require_lattice_endpoints,
            )
            if fitted:
                isometry = fitted[0]
                if target_endpoint_px is not None and not (
                    ENDPOINT_MIN_SPATIAL_SCALE - 1.0e-12
                    <= isometry.requested_distance_ratio
                    <= ENDPOINT_MAX_SPATIAL_SCALE + 1.0e-12
                ):
                    continue
                candidates.append((descriptor, isometry))
        if not candidates:
            raise ActionReplayError(
                "no unused in-screen D4 donor for request geometry: "
                f"{geometry.stable_text()}"
            )
        descriptor, isometry = min(
            candidates,
            key=lambda item: (
                0 if item[0].primitive_id == preferred_primitive_id else 1,
                item[1].endpoint_residual_fraction,
                (
                    abs(
                        float(
                            np.log(
                                max(item[1].requested_distance_ratio, 1.0e-12)
                            )
                        )
                    )
                    if target_endpoint_px is not None
                    else 0.0
                ),
                (
                    abs(item[0].observable_update_count - desired_count)
                    if desired_count is not None
                    else 0.0
                ),
                item[1].anchor_error_px,
                item[1].endpoint_error_px,
                item[1].rank,
                abs(item[0].duration_ms - duration),
                item[0].primitive_id,
            ),
        )
        if planned is not None:
            try:
                planned.remove(descriptor.primitive_id)
            except ValueError as error:
                raise AssertionError(
                    "selected donor left its demand reservation"
                ) from error
        self._consume(descriptor.bucket, descriptor.primitive_id)
        return IsometricReplayAllocation(
            descriptor=descriptor,
            isometry=isometry,
        )

    def allocate_pinch_endpoint_request(
        self,
        *,
        orientation_id: int,
        direction: str,
        detector_duration_ms: float,
        pinch_scale_direction: str,
        target_geometry: PinchEndpointGeometry,
        target_update_count: float | None = None,
        target_update_rate_hz: float | None = None,
    ) -> PinchEndpointReplayAllocation:
        """Allocate an unused, in-screen pinch donor with exact endpoints."""

        if self.pools.bank.action != "pinch":
            raise ActionReplayError(
                "pinch endpoint allocation requires a pinch replay bank"
            )
        duration = float(detector_duration_ms)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ActionReplayError(
                "detector_duration_ms must be finite and positive"
            )
        geometry = ReplayGeometry(
            action="pinch",
            orientation_id=int(orientation_id),
            direction=direction,
            pinch_scale_direction=pinch_scale_direction,
        )
        if target_geometry.scale_direction != pinch_scale_direction:
            raise ActionReplayError(
                "target pinch endpoints differ from the requested in/out class"
            )
        width, height = screen_dimensions_for_orientation(int(orientation_id))
        center_delta = (
            np.asarray(target_geometry.end_center_px, dtype=np.float64)
            - np.asarray(target_geometry.start_center_px, dtype=np.float64)
        )
        expected_direction = (
            STATIONARY
            if float(np.linalg.norm(center_delta))
            <= PINCH_STATIONARY_DIAGONAL_FRACTION
            * float(np.hypot(width, height))
            else _direction8(float(center_delta[0]), float(center_delta[1]))
        )
        if expected_direction != direction:
            raise ActionReplayError(
                "target pinch endpoints differ from the requested center direction"
            )
        if target_update_count is not None and (
            not np.isfinite(target_update_count) or target_update_count <= 0.0
        ):
            raise ActionReplayError(
                "target_update_count must be finite and positive"
            )
        if target_update_rate_hz is not None and (
            not np.isfinite(target_update_rate_hz)
            or target_update_rate_hz <= 0.0
        ):
            raise ActionReplayError(
                "target_update_rate_hz must be finite and positive"
            )
        implied_count = (
            None
            if target_update_rate_hz is None
            else float(target_update_rate_hz) * duration / 1000.0
        )
        if target_update_count is not None and implied_count is not None:
            tolerance = max(1.0e-6, 0.01 * float(target_update_count))
            if abs(float(target_update_count) - implied_count) > tolerance:
                raise ActionReplayError(
                    "target update count and rate describe different windows"
                )
        desired_count = (
            float(target_update_count)
            if target_update_count is not None
            else implied_count
        )

        if self.pools.has_demand_plan:
            planned = self._remaining_target_geometry.get(geometry, [])
            candidate_ids = [
                primitive_id
                for primitive_id in planned
                if primitive_id not in self._used
            ]
        else:
            planned = None
            candidate_ids = [
                primitive_id
                for primitive_id in self.pools.primitive_ids(self.output_split)
                if primitive_id not in self._used
                and self.pools.bank.descriptor(primitive_id).orientation_id
                == int(orientation_id)
                and self.pools.bank.descriptor(
                    primitive_id
                ).bucket.pinch_scale_direction
                == pinch_scale_direction
            ]

        candidates: list[tuple[PrimitiveDescriptor, PinchEndpointFit]] = []
        for primitive_id in candidate_ids:
            descriptor = self.pools.bank.descriptor(primitive_id)
            source_geometry = descriptor.pinch_endpoint_geometry
            if source_geometry is None:
                continue
            try:
                fit = fit_pinch_endpoint_control(
                    source_geometry,
                    target_geometry,
                    screen_width_px=float(width),
                    screen_height_px=float(height),
                    minimum_scale=ENDPOINT_MIN_SPATIAL_SCALE,
                    maximum_scale=ENDPOINT_MAX_SPATIAL_SCALE,
                    stationary_diagonal_fraction=(
                        PINCH_STATIONARY_DIAGONAL_FRACTION
                    ),
                )
            except PinchEndpointControlError:
                continue
            candidates.append((descriptor, fit))
        candidates.sort(
            key=lambda item: (
                item[1].deformation_score,
                (
                    abs(item[0].observable_update_count - desired_count)
                    if desired_count is not None
                    else 0.0
                ),
                abs(item[0].duration_ms - duration),
                item[0].primitive_id,
            )
        )

        selected: tuple[PrimitiveDescriptor, PinchEndpointFit] | None = None
        for descriptor, fit in candidates:
            raw_rows = self.pools.bank.raw_rows(descriptor.primitive_id)
            try:
                apply_pinch_endpoint_control(
                    t_ms=raw_rows.t_ms,
                    frame_index=raw_rows.frame_index,
                    pointer_id=raw_rows.pointer_id,
                    android_action=raw_rows.android_action,
                    x_px=raw_rows.x_px,
                    y_px=raw_rows.y_px,
                    fit=fit,
                    screen_width_px=float(width),
                    screen_height_px=float(height),
                )
            except PinchEndpointControlError:
                continue
            selected = (descriptor, fit)
            break
        if selected is None:
            raise ActionReplayError(
                "no unused bounded in-screen pinch donor for request geometry: "
                f"{geometry.stable_text()}"
            )
        descriptor, fit = selected
        if planned is not None:
            try:
                planned.remove(descriptor.primitive_id)
            except ValueError as error:
                raise AssertionError(
                    "selected pinch donor left its demand reservation"
                ) from error
        self._consume(descriptor.bucket, descriptor.primitive_id)
        return PinchEndpointReplayAllocation(
            descriptor=descriptor,
            endpoint_fit=fit,
        )


def _integer_time_warp(
    t_ms: np.ndarray,
    *,
    target_duration_ms: int,
) -> np.ndarray:
    """Warp unique MotionEvent intervals without interpolating signal values."""

    values = np.asarray(t_ms, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ActionReplayError("time vector must be non-empty")
    relative = values - float(values[0])
    if np.any(np.diff(relative) < 0.0):
        raise ActionReplayError("MotionEvent times are not ordered")
    rounded = np.rint(relative).astype(np.int64)
    if not np.allclose(relative, rounded, atol=1.0e-6):
        raise ActionReplayError("HMOG replay times must lie on the integer-ms lattice")
    unique, inverse = np.unique(rounded, return_inverse=True)
    if len(unique) < 2 or int(unique[-1]) <= 0:
        raise ActionReplayError("primitive has no positive-duration timeline")
    if int(unique[-1]) == int(target_duration_ms):
        # The exact-replay path must remain byte-for-byte identical on the
        # source integer-ms lattice.  Largest-remainder allocation below is
        # only for an actual duration change.
        return rounded.astype(np.float64)
    intervals = np.diff(unique)
    if np.any(intervals <= 0):
        raise AssertionError("np.unique returned non-increasing times")
    minimum_duration = len(intervals)
    if target_duration_ms < minimum_duration:
        raise ActionReplayError(
            "target duration cannot retain every distinct MotionEvent timestamp"
        )
    distributable = int(target_duration_ms - minimum_duration)
    exact_extra = intervals.astype(np.float64) * (
        distributable / float(np.sum(intervals))
    )
    extra = np.floor(exact_extra).astype(np.int64)
    remainder = distributable - int(np.sum(extra))
    if remainder:
        order = np.lexsort(
            (
                np.arange(len(intervals), dtype=np.int64),
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
    if int(warped_unique[-1]) != int(target_duration_ms):
        raise AssertionError("integer time warp missed the requested endpoint")
    return warped_unique[inverse].astype(np.float64)


def transform_replay_rows(
    rows: ReplayRows,
    *,
    translation_px: tuple[float, float] = (0.0, 0.0),
    spatial_isometry: ReplayIsometry | None = None,
    pinch_endpoint_fit: PinchEndpointFit | None = None,
    target_duration_ms: int | None = None,
    max_time_warp: float = DEFAULT_MAX_TIME_WARP,
) -> ReplayRows:
    """Apply a D4 isometry/common translation and optional narrow time warp."""

    if pinch_endpoint_fit is not None and (
        spatial_isometry is not None or tuple(translation_px) != (0.0, 0.0)
    ):
        raise ActionReplayError(
            "pinch endpoint control cannot be combined with another transform"
        )
    if spatial_isometry is not None and tuple(translation_px) != (0.0, 0.0):
        raise ActionReplayError(
            "spatial_isometry already contains the common translation"
        )
    if pinch_endpoint_fit is not None:
        if rows.action != "pinch":
            raise ActionReplayError(
                "pinch endpoint control requires complete pinch rows"
            )
        width, height = screen_dimensions_for_orientation(rows.orientation_id)
        try:
            pinch_result = apply_pinch_endpoint_control(
                t_ms=rows.t_ms,
                frame_index=rows.frame_index,
                pointer_id=rows.pointer_id,
                android_action=rows.android_action,
                x_px=rows.x_px,
                y_px=rows.y_px,
                fit=pinch_endpoint_fit,
                screen_width_px=float(width),
                screen_height_px=float(height),
            )
        except PinchEndpointControlError as error:
            raise ActionReplayError(str(error)) from error
        matrix_tuple = ((1.0, 0.0), (0.0, 1.0))
        transform_name = "pinch_bounded_endpoint_residual"
        source_center = np.asarray(
            pinch_endpoint_fit.source.start_center_px, dtype=np.float64
        )
        target_center = np.asarray(
            pinch_endpoint_fit.target.start_center_px, dtype=np.float64
        )
        tx, ty = (float(value) for value in target_center - source_center)
        transformed_xy = np.column_stack(
            (pinch_result.x_px, pinch_result.y_px)
        )
    elif spatial_isometry is None:
        matrix_tuple = ((1, 0), (0, 1))
        transform_name = "identity"
        tx, ty = (float(translation_px[0]), float(translation_px[1]))
    else:
        matrix_tuple = spatial_isometry.matrix_xy
        transform_name = spatial_isometry.name
        tx, ty = spatial_isometry.translation_px
    if not np.isfinite(tx) or not np.isfinite(ty):
        raise ActionReplayError("translation must be finite")
    matrix = np.asarray(matrix_tuple, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ActionReplayError("spatial matrix must be finite and 2x2")
    gram = matrix.T @ matrix
    scale_sq = float(np.trace(gram) / 2.0)
    if scale_sq <= 0.0 or not np.allclose(gram, np.eye(2) * scale_sq, atol=1.0e-7):
        raise ActionReplayError("spatial matrix must be a global similarity transform")
    if pinch_endpoint_fit is None:
        xy = np.column_stack(
            (
                np.asarray(rows.x_px, dtype=np.float64),
                np.asarray(rows.y_px, dtype=np.float64),
            )
        )
        transformed_xy = xy @ matrix.T + np.asarray((tx, ty), dtype=np.float64)
    if spatial_isometry is not None and spatial_isometry.quantize_pixel_lattice:
        times = np.asarray(rows.t_ms, dtype=np.float64)
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            raise ActionReplayError("endpoint correction requires positive duration")
        progress = np.clip((times - times[0]) / duration, 0.0, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        residual = np.asarray(
            spatial_isometry.endpoint_residual_px, dtype=np.float64
        )
        correction = np.rint(smooth[:, None] * residual)
        transformed_xy = transformed_xy + correction
        requested_anchor = np.asarray(
            spatial_isometry.requested_anchor_px, dtype=np.float64
        )
        requested_endpoint = np.asarray(
            spatial_isometry.requested_endpoint_px, dtype=np.float64
        )
        if (
            not np.allclose(transformed_xy[0], requested_anchor, atol=1.0e-6)
            or not np.allclose(
                transformed_xy[-1], requested_endpoint, atol=1.0e-6
            )
        ):
            raise ActionReplayError("lattice endpoint correction missed its request")
    x_px = transformed_xy[:, 0]
    y_px = transformed_xy[:, 1]
    width, height = screen_dimensions_for_orientation(rows.orientation_id)
    if (
        np.any(x_px < 0.0)
        or np.any(x_px > width)
        or np.any(y_px < 0.0)
        or np.any(y_px > height)
    ):
        raise ActionReplayError("translation would leave the screen; clipping is forbidden")
    if not np.isfinite(max_time_warp) or max_time_warp < 1.0:
        raise ActionReplayError("max_time_warp must be finite and at least one")
    if target_duration_ms is None:
        t_ms = np.asarray(rows.t_ms, dtype=np.float64).copy()
        replay_duration = float(rows.source_duration_ms)
        ratio = 1.0
    else:
        target = int(target_duration_ms)
        if target <= 0 or float(target) != float(target_duration_ms):
            raise ActionReplayError("target_duration_ms must be a positive integer")
        ratio = target / float(rows.source_duration_ms)
        lower = 1.0 / float(max_time_warp)
        if ratio < lower - 1.0e-12 or ratio > max_time_warp + 1.0e-12:
            raise ActionReplayError(
                f"time warp {ratio:.6f} leaves [{lower:.6f}, {max_time_warp:.6f}]"
            )
        t_ms = _integer_time_warp(rows.t_ms, target_duration_ms=target)
        replay_duration = float(target)
    return ReplayRows(
        primitive_id=rows.primitive_id,
        action=rows.action,
        orientation_id=rows.orientation_id,
        t_ms=t_ms,
        x_px=x_px,
        y_px=y_px,
        pressure=np.asarray(rows.pressure).copy(),
        size=np.asarray(rows.size).copy(),
        pointer_count=np.asarray(rows.pointer_count).copy(),
        pointer_id=np.asarray(rows.pointer_id).copy(),
        android_action=np.asarray(rows.android_action).copy(),
        frame_index=np.asarray(rows.frame_index).copy(),
        active_mask=np.asarray(rows.active_mask).copy(),
        valid_mask=np.asarray(rows.valid_mask).copy(),
        source_duration_ms=float(rows.source_duration_ms),
        replay_duration_ms=replay_duration,
        time_warp_ratio=float(ratio),
        translation_px=(tx, ty),
        spatial_matrix_xy=matrix_tuple,
        spatial_transform_name=transform_name,
    )


def observe_replay_primitive(
    bank: ActionReplayBank,
    primitive_id: str,
    *,
    target_samples: int,
    replay_duration_ms: float | None = None,
    output_duration_ms: float | None = None,
    translation_px: tuple[float, float] = (0.0, 0.0),
    spatial_isometry: ReplayIsometry | None = None,
    pinch_endpoint_fit: PinchEndpointFit | None = None,
    max_time_warp: float = DEFAULT_MAX_TIME_WARP,
    period_ms: float = PERIOD_MS,
) -> ReplayObservation:
    """Replay raw rows, then map them onto an independent detector window."""

    if target_samples < 2:
        raise ActionReplayError("target_samples must be at least two")
    descriptor = bank.descriptor(primitive_id)
    if replay_duration_ms is None and output_duration_ms is None:
        raw_duration = float(descriptor.duration_ms)
        detector_duration = float((target_samples - 1) * period_ms)
    else:
        if replay_duration_ms is None or output_duration_ms is None:
            raise ActionReplayError(
                "provide both replay_duration_ms and output_duration_ms"
            )
        raw_duration = float(replay_duration_ms)
        detector_duration = float(output_duration_ms)
    if not np.isfinite(raw_duration) or raw_duration <= 0.0:
        raise ActionReplayError("replay_duration_ms must be finite and positive")
    if not np.isfinite(detector_duration) or detector_duration <= 0.0:
        raise ActionReplayError("output_duration_ms must be finite and positive")
    rounded_raw_duration = int(round(raw_duration))
    if abs(raw_duration - rounded_raw_duration) > 1.0e-6:
        raise ActionReplayError("replay_duration_ms must be integer milliseconds")
    rows = transform_replay_rows(
        bank.raw_rows(primitive_id),
        translation_px=translation_px,
        spatial_isometry=spatial_isometry,
        pinch_endpoint_fit=pinch_endpoint_fit,
        target_duration_ms=rounded_raw_duration,
        max_time_warp=max_time_warp,
    )
    if spatial_isometry is not None:
        classified = classify_replay_request(
            action=rows.action,
            orientation_id=rows.orientation_id,
            target_duration_ms=rows.replay_duration_ms,
            x_px=rows.x_px,
            y_px=rows.y_px,
            pointer_id=rows.pointer_id,
            android_action=rows.android_action,
            frame_index=rows.frame_index,
        )
        if classified.direction != spatial_isometry.target_direction:
            raise ActionReplayError(
                "post-transform direction differs from requested D4 class"
            )
        if (
            rows.action == "pinch"
            and classified.pinch_scale_direction
            != descriptor.bucket.pinch_scale_direction
        ):
            raise ActionReplayError("D4 transform changed pinch in/out class")
    if pinch_endpoint_fit is not None:
        classified = classify_replay_request(
            action=rows.action,
            orientation_id=rows.orientation_id,
            target_duration_ms=rows.replay_duration_ms,
            x_px=rows.x_px,
            y_px=rows.y_px,
            pointer_id=rows.pointer_id,
            android_action=rows.android_action,
            frame_index=rows.frame_index,
        )
        width, height = screen_dimensions_for_orientation(rows.orientation_id)
        target_center_delta = (
            np.asarray(
                pinch_endpoint_fit.target.end_center_px, dtype=np.float64
            )
            - np.asarray(
                pinch_endpoint_fit.target.start_center_px, dtype=np.float64
            )
        )
        target_direction = (
            STATIONARY
            if float(np.linalg.norm(target_center_delta))
            <= PINCH_STATIONARY_DIAGONAL_FRACTION
            * float(np.hypot(width, height))
            else _direction8(
                float(target_center_delta[0]), float(target_center_delta[1])
            )
        )
        if (
            classified.direction != target_direction
            or classified.pinch_scale_direction
            != pinch_endpoint_fit.target.scale_direction
        ):
            raise ActionReplayError(
                "post-transform pinch geometry differs from its endpoint request"
            )
    observation = observe_android_rows(
        action=rows.action,
        target_samples=target_samples,
        orientation_id=rows.orientation_id,
        t_ms=rows.t_ms,
        x_px=rows.x_px,
        y_px=rows.y_px,
        pressure=rows.pressure,
        pointer_id=rows.pointer_id,
        android_action=rows.android_action,
        frame_index=rows.frame_index,
        source_duration_ms=rows.replay_duration_ms,
        period_ms=period_ms,
        target_duration_ms=detector_duration,
    )
    return ReplayObservation(
        descriptor=descriptor,
        rows=rows,
        observation=observation,
        detector_duration_ms=detector_duration,
    )


__all__ = [
    "ActionReplayBank",
    "ActionReplayError",
    "DEFAULT_MAX_TIME_WARP",
    "DIRECTION8",
    "DonorSplitPools",
    "DURATION_BUCKET_EDGES_MS",
    "IsometricReplayAllocation",
    "PinchEndpointReplayAllocation",
    "PINCH_SCALE_DIRECTIONS",
    "PrimitiveDescriptor",
    "ReplayAllocator",
    "ReplayBucket",
    "ReplayGeometry",
    "ReplayIsometry",
    "ReplayObservation",
    "ReplayRequest",
    "ReplayRows",
    "STATIONARY",
    "direction_orbit",
    "duration_bucket",
    "classify_replay_request",
    "fit_replay_isometry",
    "observe_replay_primitive",
    "transform_replay_rows",
    "transport_detector_touch_template",
]
