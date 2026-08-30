from __future__ import annotations

"""Fail-closed provenance schema for sampled conditional-touch requests.

The carrier event supplies identity, IMU, and timing only.  A sampled request
is deliberately *not* represented as a replay of the carrier's original
event plan.  This module owns no model loading or generation code; it only
canonicalizes and binds the inputs and outputs of those two stages.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .android_touch_observation import (
    TouchObservationError,
    screen_dimensions_for_orientation,
)


REQUEST_PLAN_SCHEMA = "hmog_conditional_touch_request_plan_v1"
REQUEST_PLAN_HASH_DOMAIN = "hmog-conditional-touch-request-plan-v1"
RAW_T_MS_HASH_DOMAIN = "hmog-conditional-touch-request-plan-raw-t-ms-v1"
IDENTITY_ROLE = "carrier_only"
PLAN_REPLAY_SEMANTICS = "not_original_plan_replay"
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

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SEED = (1 << 63) - 1
_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "carrier_event_id",
        "identity_role",
        "plan_replay_semantics",
        "original_event_plan_sha256",
        "orientation_id",
        "action",
        "original_direction",
        "original_down_xy_px",
        "original_up_xy_px",
        "original_raw_t_ms_sha256",
        "original_raw_duration_ms",
        "sampled_start_xy_px",
        "sampled_end_xy_px",
        "sampled_direction",
        "request_model_file_sha256",
        "request_model_artifact_sha256",
        "request_model_schema_version",
        "request_model_source_fingerprint_sha256",
        "request_seed",
    }
)
_PLAN_KEYS = _PAYLOAD_KEYS | {"request_plan_sha256"}


class ConditionalTouchRequestPlanError(ValueError):
    """A request plan or its external binding violated the frozen contract."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ConditionalTouchRequestPlanError(
            "request plan is not canonical JSON"
        ) from error


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConditionalTouchRequestPlanError(
                f"duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ConditionalTouchRequestPlanError(
        f"non-finite JSON number is forbidden: {value}"
    )


def _load_json_text(text: str, *, subject: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except ConditionalTouchRequestPlanError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ConditionalTouchRequestPlanError(
            f"{subject} is not valid JSON"
        ) from error


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, subject: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ConditionalTouchRequestPlanError(
            f"{subject} fields changed (missing={missing}, extra={extra})"
        )


def _text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or len(value) > 512:
        raise ConditionalTouchRequestPlanError(
            f"{name} must be a non-empty JSON string of at most 512 characters"
        )
    if any(ord(character) < 32 for character in value):
        raise ConditionalTouchRequestPlanError(
            f"{name} contains a forbidden control character"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    if _SHA256_RE.fullmatch(result) is None:
        raise ConditionalTouchRequestPlanError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return result


def _native_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= int(value) <= maximum
    ):
        raise ConditionalTouchRequestPlanError(
            f"{name} must be a JSON integer in [{minimum}, {maximum}]"
        )
    return int(value)


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ConditionalTouchRequestPlanError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        condition = "finite and positive" if positive else "finite"
        raise ConditionalTouchRequestPlanError(f"{name} must be {condition}")
    return result


def _point(
    value: Any,
    *,
    name: str,
    width_px: float,
    height_px: float,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConditionalTouchRequestPlanError(
            f"{name} must contain exactly two coordinates"
        )
    result = [
        _number(value[0], name=f"{name}[0]"),
        _number(value[1], name=f"{name}[1]"),
    ]
    if not (0.0 <= result[0] <= width_px and 0.0 <= result[1] <= height_px):
        raise ConditionalTouchRequestPlanError(
            f"{name} leaves the physical screen"
        )
    return result


def _direction8(start: Sequence[float], end: Sequence[float]) -> str | None:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if float(np.hypot(dx, dy)) <= 0.0:
        return None
    angle = float(np.arctan2(dy, dx))
    index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
    return DIRECTION8[index]


def _direction(
    value: Any,
    *,
    name: str,
    action: str,
    start: Sequence[float],
    end: Sequence[float],
) -> str | None:
    if value is not None and (type(value) is not str or value not in DIRECTION8):
        raise ConditionalTouchRequestPlanError(
            f"{name} must be null or one of the eight direction sectors"
        )
    realized = _direction8(start, end)
    if action in {"scroll", "swipe"}:
        if realized is None:
            raise ConditionalTouchRequestPlanError(
                f"{name} cannot describe an equal-endpoint {action}"
            )
        if value != realized:
            raise ConditionalTouchRequestPlanError(
                f"{name} does not match its endpoint sector"
            )
    elif value is not None and value != realized:
        raise ConditionalTouchRequestPlanError(
            f"{name} does not match its tap endpoint sector"
        )
    return value


def _timeline(value: Iterable[object]) -> np.ndarray:
    try:
        raw = tuple(value)
    except (TypeError, ValueError) as error:
        raise ConditionalTouchRequestPlanError(
            "original_raw_t_ms must be a numeric sequence"
        ) from error
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.integer, np.floating))
        for item in raw
    ):
        raise ConditionalTouchRequestPlanError(
            "original_raw_t_ms must be a numeric sequence"
        )
    result = np.asarray(raw, dtype=np.float64)
    if (
        result.ndim != 1
        or len(result) < 2
        or not np.isfinite(result).all()
        or np.any(np.diff(result) < 0.0)
        or float(result[-1] - result[0]) <= 0.0
    ):
        raise ConditionalTouchRequestPlanError(
            "original_raw_t_ms must be finite, nondecreasing, and span time"
        )
    return result


def raw_t_ms_sha256(t_ms: Iterable[object]) -> str:
    """Hash a raw timeline with a platform-independent float64 encoding."""

    timeline = _timeline(t_ms)
    canonical = np.ascontiguousarray(timeline, dtype=np.dtype("<f8"))
    digest = hashlib.sha256()
    digest.update(RAW_T_MS_HASH_DOMAIN.encode("ascii"))
    digest.update(b"\0")
    digest.update(len(canonical).to_bytes(8, "big", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, _PAYLOAD_KEYS, subject="request-plan payload")
    schema = _text(value["schema_version"], name="schema_version")
    if schema != REQUEST_PLAN_SCHEMA:
        raise ConditionalTouchRequestPlanError(
            f"unsupported request-plan schema {schema!r}"
        )
    identity_role = _text(value["identity_role"], name="identity_role")
    if identity_role != IDENTITY_ROLE:
        raise ConditionalTouchRequestPlanError(
            f"identity_role must be {IDENTITY_ROLE!r}"
        )
    replay_semantics = _text(
        value["plan_replay_semantics"], name="plan_replay_semantics"
    )
    if replay_semantics != PLAN_REPLAY_SEMANTICS:
        raise ConditionalTouchRequestPlanError(
            f"plan_replay_semantics must be {PLAN_REPLAY_SEMANTICS!r}"
        )
    orientation_id = _native_int(
        value["orientation_id"],
        name="orientation_id",
        minimum=-1,
        maximum=3,
    )
    try:
        width_px, height_px = screen_dimensions_for_orientation(orientation_id)
    except TouchObservationError as error:
        raise ConditionalTouchRequestPlanError(
            f"unsupported orientation_id {orientation_id!r}"
        ) from error
    action = _text(value["action"], name="action").lower()
    if action not in SUPPORTED_ACTIONS:
        raise ConditionalTouchRequestPlanError(
            f"unsupported touch request action {action!r}"
        )
    original_down = _point(
        value["original_down_xy_px"],
        name="original_down_xy_px",
        width_px=width_px,
        height_px=height_px,
    )
    original_up = _point(
        value["original_up_xy_px"],
        name="original_up_xy_px",
        width_px=width_px,
        height_px=height_px,
    )
    sampled_start = _point(
        value["sampled_start_xy_px"],
        name="sampled_start_xy_px",
        width_px=width_px,
        height_px=height_px,
    )
    sampled_end = _point(
        value["sampled_end_xy_px"],
        name="sampled_end_xy_px",
        width_px=width_px,
        height_px=height_px,
    )
    original_direction = _direction(
        value["original_direction"],
        name="original_direction",
        action=action,
        start=original_down,
        end=original_up,
    )
    sampled_direction = _direction(
        value["sampled_direction"],
        name="sampled_direction",
        action=action,
        start=sampled_start,
        end=sampled_end,
    )
    return {
        "schema_version": schema,
        "carrier_event_id": _text(
            value["carrier_event_id"], name="carrier_event_id"
        ),
        "identity_role": identity_role,
        "plan_replay_semantics": replay_semantics,
        "original_event_plan_sha256": _sha256(
            value["original_event_plan_sha256"],
            name="original_event_plan_sha256",
        ),
        "orientation_id": orientation_id,
        "action": action,
        "original_direction": original_direction,
        "original_down_xy_px": original_down,
        "original_up_xy_px": original_up,
        "original_raw_t_ms_sha256": _sha256(
            value["original_raw_t_ms_sha256"],
            name="original_raw_t_ms_sha256",
        ),
        "original_raw_duration_ms": _number(
            value["original_raw_duration_ms"],
            name="original_raw_duration_ms",
            positive=True,
        ),
        "sampled_start_xy_px": sampled_start,
        "sampled_end_xy_px": sampled_end,
        "sampled_direction": sampled_direction,
        "request_model_file_sha256": _sha256(
            value["request_model_file_sha256"],
            name="request_model_file_sha256",
        ),
        "request_model_artifact_sha256": _sha256(
            value["request_model_artifact_sha256"],
            name="request_model_artifact_sha256",
        ),
        "request_model_schema_version": _text(
            value["request_model_schema_version"],
            name="request_model_schema_version",
        ),
        "request_model_source_fingerprint_sha256": _sha256(
            value["request_model_source_fingerprint_sha256"],
            name="request_model_source_fingerprint_sha256",
        ),
        "request_seed": _native_int(
            value["request_seed"],
            name="request_seed",
            minimum=0,
            maximum=_MAX_SEED,
        ),
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_payload(payload)
    digest = hashlib.sha256()
    digest.update(REQUEST_PLAN_HASH_DOMAIN.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(normalized).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditionalTouchRequestPlan:
    carrier_event_id: str
    original_event_plan_sha256: str
    orientation_id: int
    action: str
    original_direction: str | None
    original_down_xy_px: tuple[float, float]
    original_up_xy_px: tuple[float, float]
    original_raw_t_ms_sha256: str
    original_raw_duration_ms: float
    sampled_start_xy_px: tuple[float, float]
    sampled_end_xy_px: tuple[float, float]
    sampled_direction: str | None
    request_model_file_sha256: str
    request_model_artifact_sha256: str
    request_model_schema_version: str
    request_model_source_fingerprint_sha256: str
    request_seed: int
    request_plan_sha256: str
    schema_version: str = REQUEST_PLAN_SCHEMA
    identity_role: str = IDENTITY_ROLE
    plan_replay_semantics: str = PLAN_REPLAY_SEMANTICS

    def __post_init__(self) -> None:
        # Direct construction and dataclasses.replace are held to the same
        # contract as JSON loading; there is no temporarily "trusted" object.
        self.to_json_dict()

    @classmethod
    def create(
        cls,
        *,
        carrier_event_id: str,
        original_event_plan_sha256: str,
        orientation_id: int,
        action: str,
        original_direction: str | None,
        original_down_xy_px: Sequence[float],
        original_up_xy_px: Sequence[float],
        original_raw_t_ms: Iterable[object],
        original_raw_duration_ms: float,
        sampled_start_xy_px: Sequence[float],
        sampled_end_xy_px: Sequence[float],
        sampled_direction: str | None,
        request_model_file_sha256: str,
        request_model_artifact_sha256: str,
        request_model_schema_version: str,
        request_model_source_fingerprint_sha256: str,
        request_seed: int,
    ) -> "ConditionalTouchRequestPlan":
        timeline = _timeline(original_raw_t_ms)
        raw_duration_ms = _number(
            original_raw_duration_ms,
            name="original_raw_duration_ms",
            positive=True,
        )
        timeline_span_ms = float(timeline[-1] - timeline[0])
        if timeline_span_ms > raw_duration_ms + 1.0e-3:
            raise ConditionalTouchRequestPlanError(
                "original_raw_t_ms span exceeds original_raw_duration_ms"
            )
        payload = {
            "schema_version": REQUEST_PLAN_SCHEMA,
            "carrier_event_id": carrier_event_id,
            "identity_role": IDENTITY_ROLE,
            "plan_replay_semantics": PLAN_REPLAY_SEMANTICS,
            "original_event_plan_sha256": original_event_plan_sha256,
            "orientation_id": orientation_id,
            "action": action,
            "original_direction": original_direction,
            "original_down_xy_px": list(original_down_xy_px),
            "original_up_xy_px": list(original_up_xy_px),
            "original_raw_t_ms_sha256": raw_t_ms_sha256(timeline),
            # This is the source duration used by the Android observer.  It is
            # explicitly bound instead of inferred because some archives may
            # include a duration slightly longer than their last row span.
            "original_raw_duration_ms": raw_duration_ms,
            "sampled_start_xy_px": list(sampled_start_xy_px),
            "sampled_end_xy_px": list(sampled_end_xy_px),
            "sampled_direction": sampled_direction,
            "request_model_file_sha256": request_model_file_sha256,
            "request_model_artifact_sha256": request_model_artifact_sha256,
            "request_model_schema_version": request_model_schema_version,
            "request_model_source_fingerprint_sha256": (
                request_model_source_fingerprint_sha256
            ),
            "request_seed": request_seed,
        }
        normalized = _normalize_payload(payload)
        return cls.from_json_dict(
            {**normalized, "request_plan_sha256": _payload_sha256(normalized)}
        )

    @classmethod
    def from_json_dict(
        cls, value: Mapping[str, Any]
    ) -> "ConditionalTouchRequestPlan":
        if not isinstance(value, Mapping):
            raise ConditionalTouchRequestPlanError(
                "request plan must be a JSON object"
            )
        _require_exact_keys(value, _PLAN_KEYS, subject="request plan")
        payload = _normalize_payload(
            {key: value[key] for key in _PAYLOAD_KEYS}
        )
        declared = _sha256(
            value["request_plan_sha256"], name="request_plan_sha256"
        )
        expected = _payload_sha256(payload)
        if declared != expected:
            raise ConditionalTouchRequestPlanError(
                "request_plan_sha256 does not match canonical request fields"
            )
        return cls(
            carrier_event_id=str(payload["carrier_event_id"]),
            original_event_plan_sha256=str(
                payload["original_event_plan_sha256"]
            ),
            orientation_id=int(payload["orientation_id"]),
            action=str(payload["action"]),
            original_direction=payload["original_direction"],
            original_down_xy_px=tuple(payload["original_down_xy_px"]),
            original_up_xy_px=tuple(payload["original_up_xy_px"]),
            original_raw_t_ms_sha256=str(
                payload["original_raw_t_ms_sha256"]
            ),
            original_raw_duration_ms=float(
                payload["original_raw_duration_ms"]
            ),
            sampled_start_xy_px=tuple(payload["sampled_start_xy_px"]),
            sampled_end_xy_px=tuple(payload["sampled_end_xy_px"]),
            sampled_direction=payload["sampled_direction"],
            request_model_file_sha256=str(
                payload["request_model_file_sha256"]
            ),
            request_model_artifact_sha256=str(
                payload["request_model_artifact_sha256"]
            ),
            request_model_schema_version=str(
                payload["request_model_schema_version"]
            ),
            request_model_source_fingerprint_sha256=str(
                payload["request_model_source_fingerprint_sha256"]
            ),
            request_seed=int(payload["request_seed"]),
            request_plan_sha256=declared,
            schema_version=str(payload["schema_version"]),
            identity_role=str(payload["identity_role"]),
            plan_replay_semantics=str(payload["plan_replay_semantics"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "ConditionalTouchRequestPlan":
        value = _load_json_text(text, subject="request plan")
        if not isinstance(value, Mapping):
            raise ConditionalTouchRequestPlanError(
                "request plan must be a JSON object"
            )
        return cls.from_json_dict(value)

    def to_json_dict(self) -> dict[str, Any]:
        payload = _normalize_payload(
            {
                "schema_version": self.schema_version,
                "carrier_event_id": self.carrier_event_id,
                "identity_role": self.identity_role,
                "plan_replay_semantics": self.plan_replay_semantics,
                "original_event_plan_sha256": self.original_event_plan_sha256,
                "orientation_id": self.orientation_id,
                "action": self.action,
                "original_direction": self.original_direction,
                "original_down_xy_px": list(self.original_down_xy_px),
                "original_up_xy_px": list(self.original_up_xy_px),
                "original_raw_t_ms_sha256": self.original_raw_t_ms_sha256,
                "original_raw_duration_ms": self.original_raw_duration_ms,
                "sampled_start_xy_px": list(self.sampled_start_xy_px),
                "sampled_end_xy_px": list(self.sampled_end_xy_px),
                "sampled_direction": self.sampled_direction,
                "request_model_file_sha256": self.request_model_file_sha256,
                "request_model_artifact_sha256": (
                    self.request_model_artifact_sha256
                ),
                "request_model_schema_version": (
                    self.request_model_schema_version
                ),
                "request_model_source_fingerprint_sha256": (
                    self.request_model_source_fingerprint_sha256
                ),
                "request_seed": self.request_seed,
            }
        )
        expected = _payload_sha256(payload)
        if self.request_plan_sha256 != expected:
            raise ConditionalTouchRequestPlanError(
                "request plan object no longer matches its canonical hash"
            )
        return {**payload, "request_plan_sha256": self.request_plan_sha256}

    def to_json(self) -> str:
        return _canonical_json(self.to_json_dict())


def canonical_request_plan_sha256(
    value: ConditionalTouchRequestPlan | Mapping[str, Any],
) -> str:
    """Recompute the domain-separated digest from every canonical field."""

    if isinstance(value, ConditionalTouchRequestPlan):
        encoded = value.to_json_dict()
    elif isinstance(value, Mapping):
        encoded = dict(value)
    else:
        raise ConditionalTouchRequestPlanError(
            "request plan must be an object or field mapping"
        )
    if set(encoded) == set(_PLAN_KEYS):
        encoded.pop("request_plan_sha256")
    return _payload_sha256(encoded)


def validate_request_plans(
    values: Iterable[ConditionalTouchRequestPlan | Mapping[str, Any]],
) -> tuple[ConditionalTouchRequestPlan, ...]:
    """Validate a complete plan collection and reject carrier reuse."""

    plans: list[ConditionalTouchRequestPlan] = []
    carrier_ids: set[str] = set()
    for value in values:
        plan = (
            value
            if isinstance(value, ConditionalTouchRequestPlan)
            else ConditionalTouchRequestPlan.from_json_dict(value)
        )
        # Recompute even for an already-instantiated frozen object.
        plan.to_json_dict()
        if plan.carrier_event_id in carrier_ids:
            raise ConditionalTouchRequestPlanError(
                f"carrier event ID is reused: {plan.carrier_event_id}"
            )
        carrier_ids.add(plan.carrier_event_id)
        plans.append(plan)
    if not plans:
        raise ConditionalTouchRequestPlanError(
            "request-plan collection must not be empty"
        )
    return tuple(plans)


def load_request_plan_jsonl(path: str | Path) -> tuple[ConditionalTouchRequestPlan, ...]:
    """Load strict JSONL with duplicate-key and carrier-reuse rejection."""

    source = Path(path).resolve()
    plans: list[ConditionalTouchRequestPlan] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ConditionalTouchRequestPlanError(
                        f"blank request-plan JSONL row at line {line_number}"
                    )
                value = _load_json_text(
                    line, subject=f"request-plan JSONL line {line_number}"
                )
                if not isinstance(value, Mapping):
                    raise ConditionalTouchRequestPlanError(
                        f"request-plan JSONL line {line_number} is not an object"
                    )
                plans.append(ConditionalTouchRequestPlan.from_json_dict(value))
    except ConditionalTouchRequestPlanError:
        raise
    except OSError as error:
        raise ConditionalTouchRequestPlanError(
            f"cannot read request-plan JSONL: {source}"
        ) from error
    return validate_request_plans(plans)


def validate_request_plan_against_binding(
    plan: ConditionalTouchRequestPlan | Mapping[str, Any],
    *,
    carrier_event_id: str,
    original_event_plan_sha256: str,
    orientation_id: int,
    action: str,
    original_direction: str | None,
    original_down_xy_px: Sequence[float],
    original_up_xy_px: Sequence[float],
    original_raw_t_ms: Iterable[object],
    original_raw_duration_ms: float,
    sampled_start_xy_px: Sequence[float],
    sampled_end_xy_px: Sequence[float],
    sampled_direction: str | None,
    request_model_file_sha256: str,
    request_model_artifact_sha256: str,
    request_model_schema_version: str,
    request_model_source_fingerprint_sha256: str,
    request_seed: int,
) -> ConditionalTouchRequestPlan:
    """Rebuild the expected plan from live sources and compare every field."""

    observed = (
        plan
        if isinstance(plan, ConditionalTouchRequestPlan)
        else ConditionalTouchRequestPlan.from_json_dict(plan)
    )
    expected = ConditionalTouchRequestPlan.create(
        carrier_event_id=carrier_event_id,
        original_event_plan_sha256=original_event_plan_sha256,
        orientation_id=orientation_id,
        action=action,
        original_direction=original_direction,
        original_down_xy_px=original_down_xy_px,
        original_up_xy_px=original_up_xy_px,
        original_raw_t_ms=original_raw_t_ms,
        original_raw_duration_ms=original_raw_duration_ms,
        sampled_start_xy_px=sampled_start_xy_px,
        sampled_end_xy_px=sampled_end_xy_px,
        sampled_direction=sampled_direction,
        request_model_file_sha256=request_model_file_sha256,
        request_model_artifact_sha256=request_model_artifact_sha256,
        request_model_schema_version=request_model_schema_version,
        request_model_source_fingerprint_sha256=(
            request_model_source_fingerprint_sha256
        ),
        request_seed=request_seed,
    )
    if observed != expected:
        raise ConditionalTouchRequestPlanError(
            "request plan does not match its live carrier/model binding"
        )
    return observed


__all__ = [
    "ConditionalTouchRequestPlan",
    "ConditionalTouchRequestPlanError",
    "DIRECTION8",
    "IDENTITY_ROLE",
    "PLAN_REPLAY_SEMANTICS",
    "RAW_T_MS_HASH_DOMAIN",
    "REQUEST_PLAN_HASH_DOMAIN",
    "REQUEST_PLAN_SCHEMA",
    "SUPPORTED_ACTIONS",
    "canonical_request_plan_sha256",
    "load_request_plan_jsonl",
    "raw_t_ms_sha256",
    "validate_request_plan_against_binding",
    "validate_request_plans",
]
