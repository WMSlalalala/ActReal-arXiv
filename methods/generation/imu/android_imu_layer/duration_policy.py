from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def _finite_float(value: Any, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def xy_distance_px(
    xy_start: Optional[Sequence[float]] = None,
    xy_end: Optional[Sequence[float]] = None,
) -> Optional[float]:
    if xy_start is None or xy_end is None:
        return None
    if len(xy_start) < 2 or len(xy_end) < 2:
        return None
    x0, y0 = float(xy_start[0]), float(xy_start[1])
    x1, y1 = float(xy_end[0]), float(xy_end[1])
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    return math.hypot(x1 - x0, y1 - y0)


class DurationPolicy:
    """Duration policy learned from processed HMOG action data.

    The policy never uses detector test labels. It only maps Android API-level
    action parameters to a target event duration in milliseconds.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.actions = dict(config.get("actions", {}))
        self.hz = float(config.get("hz", 100.0))

    @classmethod
    def from_json(cls, path: str) -> "DurationPolicy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _clamp(self, action_cfg: Dict[str, Any], duration_ms: float, allow_extreme: bool = False) -> float:
        if allow_extreme:
            lo = _finite_float(action_cfg.get("min_ms"), 10.0)
            hi = _finite_float(action_cfg.get("max_ms"), max(lo, duration_ms))
        else:
            lo = _finite_float(action_cfg.get("p05_ms"), action_cfg.get("min_ms", 10.0))
            hi = _finite_float(action_cfg.get("p95_ms"), action_cfg.get("max_ms", max(lo, duration_ms)))
        if hi < lo:
            hi = lo
        return float(min(max(duration_ms, lo), hi))

    def duration_ms(
        self,
        action: str,
        xy_start: Optional[Sequence[float]] = None,
        xy_end: Optional[Sequence[float]] = None,
        requested_duration_ms: Optional[float] = None,
        pinch_start_span: Optional[float] = None,
        pinch_end_span: Optional[float] = None,
        allow_extreme: bool = False,
    ) -> float:
        if action not in self.actions:
            raise KeyError("unknown action %r; known=%s" % (action, sorted(self.actions)))
        cfg = self.actions[action]
        if requested_duration_ms is not None:
            return self._clamp(cfg, _finite_float(requested_duration_ms, cfg.get("median_ms", 100.0)), allow_extreme=True)

        policy_type = str(cfg.get("policy_type", "duration_only"))
        median = _finite_float(cfg.get("median_ms"), 100.0)

        if policy_type in {"linear_distance", "binned_distance"}:
            distance = xy_distance_px(xy_start, xy_end)
            if action == "pinch" and pinch_start_span is not None and pinch_end_span is not None:
                distance = abs(float(pinch_end_span) - float(pinch_start_span))
            if distance is None:
                duration = median
            elif policy_type == "linear_distance":
                intercept = _finite_float(cfg.get("intercept_ms"), median)
                slope = _finite_float(cfg.get("slope_ms_per_unit"), 0.0)
                duration = intercept + slope * float(distance)
            else:
                duration = self._duration_from_bins(cfg, float(distance), median)
            return self._clamp(cfg, duration, allow_extreme=allow_extreme)

        return self._clamp(cfg, median, allow_extreme=allow_extreme)

    @staticmethod
    def _duration_from_bins(cfg: Dict[str, Any], value: float, default: float) -> float:
        bins = cfg.get("bins", [])
        if not isinstance(bins, list) or not bins:
            return float(default)
        chosen = bins[-1]
        for row in bins:
            lo = _finite_float(row.get("lo"), -float("inf"))
            hi = _finite_float(row.get("hi"), float("inf"))
            if value >= lo and value <= hi:
                chosen = row
                break
        return _finite_float(chosen.get("median_ms"), default)
