from __future__ import annotations

"""Small, non-formal checks for obvious touch-representation shortcuts.

The helpers in this module intentionally use only transparent one-dimensional
statistics.  They are useful for catching construction bugs such as one label
having no contact, a constant availability bit, or interpolated coordinates.
They do *not* establish that two trajectory populations are indistinguishable.
"""

from typing import Mapping, Sequence

import numpy as np

from .android_touch_observation import TouchObservation, active_xy_repeat_rate


FEATURE_NAMES = (
    "duration_seconds",
    "availability_fraction",
    "contact_fraction",
    "multi_pointer_fraction",
    "pointer_mean_active",
    "pointer_max",
    "active_xy_repeat_rate",
    "active_dxdy_nonzero_fraction",
    "all_dxdy_nonzero_fraction",
    "active_xy_unique_ratio",
    "contact_transition_rate",
    "source_update_rate",
    # Label-blind spatial and shape diagnostics on the common trajectory.
    # Positions are normalized screen-relative coordinates.  Path geometry is
    # restricted to adjacent samples in the same contact so that the empty
    # flight between two keystrokes is never treated as finger motion.
    "active_x_start",
    "active_y_start",
    "active_x_end",
    "active_y_end",
    "active_x_mean",
    "active_y_mean",
    "active_x_std",
    "active_y_std",
    "active_bbox_width",
    "active_bbox_height",
    "active_bbox_area",
    "within_contact_net_displacement",
    "within_contact_path_length",
    "within_contact_straightness",
    "active_pressure_mean",
    "active_pressure_std",
    "active_pressure_min",
    "active_pressure_max",
)


class ReplayLeakageSmokeError(ValueError):
    pass


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
    selected = np.asarray(denominator, dtype=bool)
    if not np.any(selected):
        return 0.0
    return float(np.mean(np.asarray(numerator, dtype=bool)[selected]))


def extract_obvious_shortcut_features(
    observation: TouchObservation,
) -> dict[str, float]:
    """Extract simple sampling/state features from one common observation."""

    trajectory = np.asarray(observation.trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 9 or len(trajectory) < 2:
        raise ReplayLeakageSmokeError("trajectory must have shape [samples, 9]")
    if not np.isfinite(trajectory).all():
        raise ReplayLeakageSmokeError("trajectory contains non-finite values")
    contact = trajectory[:, 0] > 0.5
    active_pairs = contact[1:] & contact[:-1]
    dxdy_nonzero = np.any(np.abs(trajectory[1:, 5:7]) > 1.0e-8, axis=1)
    active_xy = trajectory[contact, 1:3]
    if len(active_xy):
        # The source coordinates are float32, so byte-exact uniqueness is the
        # relevant check for zero-order-hold repetitions.
        unique_ratio = float(len(np.unique(active_xy, axis=0)) / len(active_xy))
        pointer_mean = float(np.mean(trajectory[contact, 4]))
        active_start = active_xy[0]
        active_end = active_xy[-1]
        active_mean = np.mean(active_xy, axis=0)
        active_std = np.std(active_xy, axis=0)
        active_min = np.min(active_xy, axis=0)
        active_max = np.max(active_xy, axis=0)
        active_bbox = active_max - active_min
        active_pressure = trajectory[contact, 3]
    else:
        unique_ratio = 0.0
        pointer_mean = 0.0
        active_start = np.zeros(2, dtype=np.float64)
        active_end = np.zeros(2, dtype=np.float64)
        active_mean = np.zeros(2, dtype=np.float64)
        active_std = np.zeros(2, dtype=np.float64)
        active_bbox = np.zeros(2, dtype=np.float64)
        active_pressure = np.zeros(1, dtype=np.float64)

    xy_steps = np.linalg.norm(
        trajectory[1:, 1:3] - trajectory[:-1, 1:3], axis=1
    )
    within_contact_path_length = float(np.sum(xy_steps[active_pairs]))
    segment_starts = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
    segment_ends = np.flatnonzero(contact & ~np.r_[contact[1:], False])
    if len(segment_starts) != len(segment_ends):
        raise ReplayLeakageSmokeError("contact segments are inconsistent")
    within_contact_net_displacement = float(
        sum(
            np.linalg.norm(
                trajectory[end, 1:3] - trajectory[start, 1:3]
            )
            for start, end in zip(segment_starts, segment_ends, strict=True)
        )
    )
    within_contact_straightness = (
        within_contact_net_displacement / within_contact_path_length
        if within_contact_path_length > 1.0e-12
        else 0.0
    )
    transitions = contact[1:] != contact[:-1]
    duration = float(trajectory[-1, 7] - trajectory[0, 7])
    if duration <= 0.0:
        raise ReplayLeakageSmokeError("trajectory duration must be positive")
    return {
        "duration_seconds": duration,
        "availability_fraction": float(np.mean(trajectory[:, 8] > 0.5)),
        "contact_fraction": float(np.mean(contact)),
        "multi_pointer_fraction": float(np.mean(trajectory[:, 4] > 1.5)),
        "pointer_mean_active": pointer_mean,
        "pointer_max": float(np.max(trajectory[:, 4])),
        "active_xy_repeat_rate": active_xy_repeat_rate(trajectory),
        "active_dxdy_nonzero_fraction": _safe_fraction(dxdy_nonzero, active_pairs),
        "all_dxdy_nonzero_fraction": float(np.mean(dxdy_nonzero)),
        "active_xy_unique_ratio": unique_ratio,
        "contact_transition_rate": float(np.mean(transitions)),
        "source_update_rate": float(observation.source_updates / len(trajectory)),
        "active_x_start": float(active_start[0]),
        "active_y_start": float(active_start[1]),
        "active_x_end": float(active_end[0]),
        "active_y_end": float(active_end[1]),
        "active_x_mean": float(active_mean[0]),
        "active_y_mean": float(active_mean[1]),
        "active_x_std": float(active_std[0]),
        "active_y_std": float(active_std[1]),
        "active_bbox_width": float(active_bbox[0]),
        "active_bbox_height": float(active_bbox[1]),
        "active_bbox_area": float(active_bbox[0] * active_bbox[1]),
        "within_contact_net_displacement": within_contact_net_displacement,
        "within_contact_path_length": within_contact_path_length,
        "within_contact_straightness": float(within_contact_straightness),
        "active_pressure_mean": float(np.mean(active_pressure)),
        "active_pressure_std": float(np.std(active_pressure)),
        "active_pressure_min": float(np.min(active_pressure)),
        "active_pressure_max": float(np.max(active_pressure)),
    }


def empirical_ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the two-sample empirical Kolmogorov--Smirnov distance."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b):
        raise ReplayLeakageSmokeError("KS inputs must be non-empty vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ReplayLeakageSmokeError("KS inputs must be finite")
    support = np.sort(np.unique(np.concatenate((a, b))))
    cdf_a = np.searchsorted(np.sort(a), support, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), support, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def orientation_free_univariate_auc(
    genuine: Sequence[float], replay: Sequence[float]
) -> float:
    """Return max(AUC, 1-AUC) for one feature, with ties worth one half."""

    negative = np.asarray(genuine, dtype=np.float64)
    positive = np.asarray(replay, dtype=np.float64)
    if negative.ndim != 1 or positive.ndim != 1 or not len(negative) or not len(positive):
        raise ReplayLeakageSmokeError("AUC inputs must be non-empty vectors")
    if not np.isfinite(negative).all() or not np.isfinite(positive).all():
        raise ReplayLeakageSmokeError("AUC inputs must be finite")
    # Pairwise comparison is deliberately simple; smoke sample sizes are small.
    comparison = positive[:, None] - negative[None, :]
    auc = float((np.sum(comparison > 0.0) + 0.5 * np.sum(comparison == 0.0)) / comparison.size)
    return max(auc, 1.0 - auc)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def compare_feature_groups(
    genuine_rows: Sequence[Mapping[str, float]],
    replay_rows: Sequence[Mapping[str, float]],
    *,
    auc_flag_threshold: float = 0.80,
    ks_flag_threshold: float = 0.50,
) -> dict[str, object]:
    """Summarize and flag conspicuous one-feature population differences.

    A flag is only a diagnostic candidate.  An unflagged result is reported as
    ``no_obvious_shortcut_detected`` rather than as a formal pass.
    """

    if not genuine_rows or not replay_rows:
        raise ReplayLeakageSmokeError("both feature groups must be non-empty")
    if not 0.5 <= auc_flag_threshold <= 1.0:
        raise ReplayLeakageSmokeError("AUC threshold must lie in [0.5, 1]")
    if not 0.0 <= ks_flag_threshold <= 1.0:
        raise ReplayLeakageSmokeError("KS threshold must lie in [0, 1]")
    expected = set(FEATURE_NAMES)
    for row in tuple(genuine_rows) + tuple(replay_rows):
        if set(row) != expected:
            raise ReplayLeakageSmokeError("feature row has the wrong schema")

    comparisons: dict[str, object] = {}
    flagged: list[str] = []
    for name in FEATURE_NAMES:
        genuine = np.asarray([float(row[name]) for row in genuine_rows])
        replay = np.asarray([float(row[name]) for row in replay_rows])
        auc = orientation_free_univariate_auc(genuine, replay)
        ks = empirical_ks_distance(genuine, replay)
        is_flagged = bool(auc >= auc_flag_threshold or ks >= ks_flag_threshold)
        if is_flagged:
            flagged.append(name)
        comparisons[name] = {
            "genuine": _summary(genuine),
            "replay": _summary(replay),
            "orientation_free_univariate_auc": auc,
            "empirical_ks_distance": ks,
            "obvious_shortcut_candidate": is_flagged,
        }
    return {
        "genuine_samples": len(genuine_rows),
        "replay_samples": len(replay_rows),
        "diagnostic_status": (
            "obvious_shortcut_candidates_found"
            if flagged
            else "no_obvious_shortcut_detected"
        ),
        "flagged_features": flagged,
        "thresholds": {
            "orientation_free_univariate_auc": float(auc_flag_threshold),
            "empirical_ks_distance": float(ks_flag_threshold),
        },
        "features": comparisons,
    }


__all__ = [
    "FEATURE_NAMES",
    "ReplayLeakageSmokeError",
    "compare_feature_groups",
    "empirical_ks_distance",
    "extract_obvious_shortcut_features",
    "orientation_free_univariate_auc",
]
