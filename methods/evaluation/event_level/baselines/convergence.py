#!/usr/bin/env python3
"""Decide when a generator has finished learning, from its own samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def lag1(windows: np.ndarray, limit: int = 256) -> float:
    """Median lag-1 autocorrelation over the channels that actually vary."""

    values = []
    for window in windows[:limit]:
        for channel in range(window.shape[1]):
            series = window[:, channel]
            if series.std() > 1e-9:
                values.append(np.corrcoef(series[:-1], series[1:])[0, 1])
    return float(np.nanmedian(values)) if values else float("nan")


def _summaries(windows: np.ndarray) -> np.ndarray:
    """Per-window features an energy distance can be computed over cheaply."""

    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    step = np.abs(np.diff(windows, axis=1)).mean(axis=1)
    span = windows.max(axis=1) - windows.min(axis=1)
    return np.concatenate([mean, std, step, span], axis=1)


def energy_distance(a: np.ndarray, b: np.ndarray, limit: int = 400) -> float:
    """Standardised energy distance between two sets of window summaries."""

    x, y = _summaries(a[:limit]), _summaries(b[:limit])
    scale = np.concatenate([x, y]).std(axis=0) + 1e-9
    x, y = x / scale, y / scale

    def mean_pairwise(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)))

    return float(
        2.0 * mean_pairwise(x, y) - mean_pairwise(x, x) - mean_pairwise(y, y)
    )


def nearest_training_distance(generated: np.ndarray, training: np.ndarray, limit: int = 128) -> float:
    """Median distance from a sample to its closest training window.

    With five training windows memorisation is expected rather than a defect;
    this records how close the model actually gets so the paper can state it.
    """

    scale = float(training.std()) + 1e-9
    distances = []
    for sample in generated[:limit]:
        gaps = np.sqrt(((training - sample[None]) ** 2).mean(axis=(1, 2)))
        distances.append(float(gaps.min()) / scale)
    return float(np.median(distances)) if distances else float("nan")


@dataclass
class ConvergenceMonitor:
    """Track a generator's gap to the real data and say when it has plateaued."""

    real: np.ndarray
    patience: int = 4
    tolerance: float = 0.01
    minimum_checkpoints: int = 3
    history: list = field(default_factory=list)
    best_gap: float = float("inf")
    best_step: int = -1
    _since_best: int = 0

    def __post_init__(self) -> None:
        self._real_lag1 = lag1(self.real)
        self._real_std = self.real.std(axis=(0, 1))

    def observe(self, step: int, generated: np.ndarray) -> dict:
        generated_lag1 = lag1(generated)
        std_ratio = generated.std(axis=(0, 1)) / np.maximum(self._real_std, 1e-9)
        energy = energy_distance(self.real, generated)
        # A dead generator emits a constant, whose autocorrelation is undefined.
        # Left as nan it would poison the gap, and every comparison against a nan
        # is false -- so the checkpoint would be recorded as "not an improvement"
        # while never being rejected either, and if it came first there would be
        # no best checkpoint at all.  Charge it a full unit of error and say so.
        degenerate = not np.isfinite(generated_lag1)
        # Each term is a fraction-of-real error, so they add on one scale.
        lag_gap = (
            1.0
            if degenerate
            else abs(generated_lag1 - self._real_lag1) / max(abs(self._real_lag1), 1e-6)
        )
        std_gap = float(np.median(np.abs(std_ratio - 1.0)))
        gap = float(lag_gap + std_gap + energy)
        if not np.isfinite(gap):
            gap = float("inf")
        record = {
            "step": int(step),
            "lag1": generated_lag1,
            "real_lag1": self._real_lag1,
            "lag_gap": lag_gap,
            "degenerate": bool(degenerate),
            "std_ratio_median": float(np.median(std_ratio)),
            "std_gap": std_gap,
            "energy_distance": energy,
            "gap": gap,
        }
        self.history.append(record)
        if gap < self.best_gap - self.tolerance:
            self.best_gap, self.best_step, self._since_best = gap, int(step), 0
        else:
            self._since_best += 1
            if gap < self.best_gap:
                self.best_gap, self.best_step = gap, int(step)
        record["best_step"] = self.best_step
        record["since_best"] = self._since_best
        return record

    def should_stop(self) -> bool:
        return (
            len(self.history) >= self.minimum_checkpoints
            and self._since_best >= self.patience
        )

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.write_text(
            json.dumps(
                {
                    "real_lag1": self._real_lag1,
                    "real_std": self._real_std.tolist(),
                    "patience": self.patience,
                    "tolerance": self.tolerance,
                    "best_step": self.best_step,
                    "best_gap": self.best_gap,
                    "stopped_early": self.should_stop(),
                    "history": self.history,
                    **(extra or {}),
                },
                indent=2,
                sort_keys=True,
            )
        )


__all__ = [
    "ConvergenceMonitor",
    "energy_distance",
    "lag1",
    "nearest_training_distance",
]
