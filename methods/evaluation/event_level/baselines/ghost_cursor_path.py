#!/usr/bin/env python3
"""ghost-cursor's pointer path, reproduced faithfully for offline use."""

from __future__ import annotations

import math
import random
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# The package imports its browser driver at module load.  None of the curve
# maths touches it, so a stand-in keeps the library's own files unmodified.
for _name, _attrs in (
    ("pyppeteer", ()),
    ("pyppeteer.page", ("Page",)),
    ("pyppeteer.element_handle", ("ElementHandle",)),
):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        for _attr in _attrs:
            setattr(_module, _attr, type(_attr, (), {}))
        sys.modules[_name] = _module

from pyppeteer_ghost_cursor.math import (  # noqa: E402
    Vector,
    bezierCurve,
    direction,
    magnitude,
    overshoot,
)
from pyppeteer_ghost_cursor.spoof import fitts  # noqa: E402

# spoof.ts:665, 416, 417 and the port's own lines 154, 166, 167.
OVERSHOOT_THRESHOLD = 500.0
OVERSHOOT_SPREAD = 10.0
OVERSHOOT_RADIUS = 120.0
DEFAULT_WIDTH = 100.0
MIN_STEPS = 25


def _arc_length_samples(curve, steps: int, resolution: int = 2048) -> np.ndarray:
    """Sample a curve at equal arc length, as the original's getLUT does."""

    dense_t = np.linspace(0.0, 1.0, resolution)
    dense = curve.evaluate_multi(dense_t)
    segment = np.hypot(np.diff(dense[0]), np.diff(dense[1]))
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    if arc[-1] <= 0.0:
        return np.repeat(dense[:, :1], steps, axis=1).T
    wanted = np.linspace(0.0, arc[-1], steps)
    return curve.evaluate_multi(np.interp(wanted, arc, dense_t)).T


DEGENERATE_CHORD_PX = 1.0e-9


def path(start, end, spread_override=None, width: float = DEFAULT_WIDTH) -> np.ndarray:
    """One ghost-cursor path, spaced by arc length.  Mirrors spoof.ts:318-343."""

    a = Vector(float(start[0]), float(start[1]))
    b = Vector(float(end[0]), float(end[1]))
    if magnitude(direction(a, b)) < DEGENERATE_CHORD_PX:
        return np.repeat(np.asarray([[a.x, a.y]], dtype=np.float64), 2, axis=0)
    curve = bezierCurve(a, b, spread_override)
    if spread_override is not None:
        # The port drops the override that the original honours
        # (`spreadOverride ?? clamp(...)`), and the overshoot correction depends
        # on it, so the curve is rebuilt here with the override applied.
        curve = _curve_with_spread(a, b, float(spread_override))
    length = float(curve.length) * 0.8
    base_time = random.random() * MIN_STEPS
    steps = math.ceil((math.log2(fitts(length, width) + 1) + base_time) * 3)
    steps = max(steps, 2)
    points = _arc_length_samples(curve, steps)
    return np.maximum(points, 0.0)  # clampPositive, spoof.ts:345


def _curve_with_spread(a: Vector, b: Vector, spread: float):
    from pyppeteer_ghost_cursor.math import generateBezierAnchors
    import bezier

    anchors = generateBezierAnchors(a, b, spread)
    nodes = np.asfortranarray(
        [
            [a.x, anchors[0].x, anchors[1].x, b.x],
            [a.y, anchors[0].y, anchors[1].y, b.y],
        ]
    )
    return bezier.Curve(nodes, degree=3)


def move(start, end, width: float = DEFAULT_WIDTH) -> np.ndarray:
    """The library's own move sequence, including overshoot.  spoof.ts:688-700."""

    a = Vector(float(start[0]), float(start[1]))
    b = Vector(float(end[0]), float(end[1]))
    if magnitude(direction(a, b)) > OVERSHOOT_THRESHOLD:
        scattered = overshoot(b, OVERSHOOT_RADIUS)
        first = path((a.x, a.y), (scattered.x, scattered.y), width=width)
        second = path(
            (first[-1][0], first[-1][1]),
            (b.x, b.y),
            spread_override=OVERSHOOT_SPREAD,
            width=width,
        )
        return np.vstack((first, second[1:]))
    return path((a.x, a.y), (b.x, b.y), width=width)


__all__ = ["move", "path", "OVERSHOOT_THRESHOLD", "OVERSHOOT_RADIUS", "OVERSHOOT_SPREAD"]
