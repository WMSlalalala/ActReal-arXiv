"""The canonical IMU window loaded from an offline-generated bundle.

Runtime serving reads only frozen bundles.  Diffusion generation is implemented
in ``methods/generation`` and is never invoked from this module.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import config


@dataclass
class IMUWindow:
    """One padding window at 100 Hz.

    ``samples`` is (T, 6): ax, ay, az, gx, gy, gz.  ``active_start`` and
    ``active_end`` index the gesture inside it, so the touch DOWN belongs at
    ``active_start`` frames after the window starts.
    """

    action: str
    samples: np.ndarray
    active_start: int
    active_end: int
    orientation_id: int
    requested_duration_ms: float
    source: str
    # Not every window is on the nominal 100 Hz step.  A released keystroke can
    # spread a thirty-second typing burst across a capped 512-frame window, and
    # delivering that at 100 Hz would compress the event six-fold and pull the
    # inertia away from the touch it belongs to.
    period_ms: float = config.GRID_PERIOD_MS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples.ndim != 2 or self.samples.shape[1] != 6:
            raise ValueError(f"samples must be (T,6), got {self.samples.shape}")
        if not 0 <= self.active_start <= self.active_end <= len(self.samples):
            raise ValueError(
                f"bad active span [{self.active_start},{self.active_end}) "
                f"for T={len(self.samples)}"
            )
        if not self.period_ms > 0:
            raise ValueError(f"period_ms must be positive, got {self.period_ms}")

    @property
    def frames(self) -> int:
        return int(len(self.samples))

    @property
    def active_frames(self) -> int:
        return int(self.active_end - self.active_start)

    @property
    def pre_roll_ms(self) -> float:
        return self.active_start * self.period_ms

    @property
    def post_roll_ms(self) -> float:
        return (self.frames - self.active_end) * self.period_ms

    @property
    def duration_ms(self) -> float:
        return self.frames * self.period_ms


def load_cached_window(path, *, action: str, exactness: str = "exact") -> IMUWindow:
    """Read one window off disk, keeping its own period rather than assuming 100 Hz."""

    with np.load(path) as data:
        samples = np.asarray(data["samples"], dtype=np.float32)
        return IMUWindow(
            action=action,
            samples=samples,
            active_start=int(data["active_start"]),
            active_end=int(data["active_end"]),
            orientation_id=int(data["orientation_id"]),
            requested_duration_ms=float(data["requested_duration_ms"]),
            period_ms=float(data["period_ms"]),
            source=f"cache:{exactness}:{Path(path).name}",
            metadata={"cache_path": str(path), "exactness": exactness},
        )
