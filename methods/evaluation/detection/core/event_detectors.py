from __future__ import annotations

"""Registered Event PAD detector implementations and feature adapters.

All detector inputs use the canonical release contract:

* trajectory: contact, x_rel, y_rel, pressure, pointer_count, dx_rel,
  dy_rel, elapsed_seconds, availability
* IMU+XY+time: six IMU axes plus x_rel, y_rel, elapsed_seconds
* joint: six IMU axes plus the complete trajectory vector above

The paper-feature adapter never interprets trajectory columns as accelerometer
or gyroscope axes.  Sensor features are extracted from IMU only; the
trajectory-only paper baseline uses a Touchalytics-style stroke feature block.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn import functional as F


MODALITIES = (
    "trajectory_xytime",
    "imu_only",
    "imu_trajectory_xytime",
)
FEATURE_DETECTORS = ("hmog_style_svm", "hmog_style_rf")
PAPER_DETECTORS = ("paper_svm", "paper_xgboost")
DEEP_DETECTORS = ("behaveformer_stdat", "authconformer")
REGISTERED_DETECTORS = FEATURE_DETECTORS + PAPER_DETECTORS + DEEP_DETECTORS

TRAJECTORY_CHANNELS = 9
IMU_CHANNELS = 6
# The canonical detector grid is a regular 100 Hz clock.  The pure-IMU modality
# uses it to build a synthetic time axis so it never reads the touch stream.
DETECTOR_GRID_HZ = 100.0
# Frozen per-action detector window in 100 Hz samples.  Each value is the 95th
# percentile of GENUINE event length over the TRAIN users only, rounded up to a
# multiple of the four attention heads.  Longer events are right-truncated and
# shorter events are padded and masked, so retained samples keep their native
# rate.  Derived once from train data and frozen; never re-fit per run.
ACTION_WINDOW_SAMPLES = {
    "tap": 16,
    "scroll": 208,
    "swipe": 176,
    "pinch": 100,
    "keystroke": 512,
}
ACTION_WINDOW_SOURCE = (
    "train_split_genuine_length_p95_rounded_up_to_attention_heads_v1"
)
if any(value % 4 for value in ACTION_WINDOW_SAMPLES.values()):
    raise RuntimeError("detector window must divide the four attention heads")


def action_window_samples(action: str) -> int:
    """Return the frozen detector window for one action."""

    try:
        return ACTION_WINDOW_SAMPLES[action]
    except KeyError as exc:
        raise DetectorProtocolError(
            f"no frozen detector window for action {action!r}"
        ) from exc
FORMAL_MODALITY_CHANNELS = {
    "trajectory_xytime": (
        "contact_mask",
        "x_rel",
        "y_rel",
        "pressure",
        "pointer_count",
        "dx_rel",
        "dy_rel",
        "elapsed_seconds",
        "availability",
    ),
    "imu_only": (
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ),
    "imu_trajectory_xytime": (
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "contact_mask",
        "x_rel",
        "y_rel",
        "pressure",
        "pointer_count",
        "dx_rel",
        "dy_rel",
        "elapsed_seconds",
        "availability",
    ),
}
DEEP_INPUT_CHANNELS = {
    modality: len(channels)
    for modality, channels in FORMAL_MODALITY_CHANNELS.items()
}
if (
    tuple(FORMAL_MODALITY_CHANNELS) != MODALITIES
    or DEEP_INPUT_CHANNELS
    != {
        "trajectory_xytime": 9,
        "imu_only": 6,
        "imu_trajectory_xytime": 15,
    }
):
    raise RuntimeError("formal Event modality/channel registry drift")


class DetectorProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    family: str
    execution: str
    feature_recipe: str | None
    reference: str


DETECTOR_SPECS: Mapping[str, DetectorSpec] = {
    "hmog_style_svm": DetectorSpec(
        "hmog_style_svm",
        "feature",
        "cpu",
        "modality_statistics_v2",
        "HMOG-style statistical PAD adaptation; not an exact HMOG reproduction",
    ),
    "hmog_style_rf": DetectorSpec(
        "hmog_style_rf",
        "feature",
        "cpu",
        "modality_statistics_v2",
        "HMOG-style statistical PAD adaptation; not an exact HMOG reproduction",
    ),
    "paper_svm": DetectorSpec(
        "paper_svm",
        "paper",
        "cpu",
        "paper_modality_adapter_v2",
        "RiskCog-derived motion statistics plus Touchalytics-style touch features",
    ),
    "paper_xgboost": DetectorSpec(
        "paper_xgboost",
        "paper",
        "cpu",
        "paper_modality_adapter_v2",
        "RiskCog-derived motion statistics plus Touchalytics-style touch features",
    ),
    "behaveformer_stdat": DetectorSpec(
        "behaveformer_stdat",
        "deep",
        "physical_gpu",
        None,
        "BehaveFormer STDAT architecture adapted to PAD and canonical channels",
    ),
    "authconformer": DetectorSpec(
        "authconformer",
        "deep",
        "physical_gpu",
        None,
        "AuthConFormer architecture adapted to PAD and canonical channels",
    ),
}


def detector_spec(name: str) -> DetectorSpec:
    try:
        return DETECTOR_SPECS[name]
    except KeyError as exc:
        raise DetectorProtocolError(f"unregistered detector {name!r}") from exc


def modality_values(
    imu: np.ndarray,
    trajectory: np.ndarray,
    modality: str,
) -> np.ndarray:
    if (
        imu.ndim != 2
        or imu.shape[1] != IMU_CHANNELS
        or trajectory.shape != (len(imu), TRAJECTORY_CHANNELS)
    ):
        raise DetectorProtocolError("misaligned canonical Event signals")
    if modality == "trajectory_xytime":
        values = trajectory
    elif modality == "imu_only":
        values = imu
    elif modality == "imu_trajectory_xytime":
        values = np.concatenate([imu, trajectory], axis=1)
    else:
        raise DetectorProtocolError(f"unknown modality {modality!r}")
    if not np.isfinite(values).all():
        raise DetectorProtocolError("non-finite detector input")
    if np.any(trajectory[:, 1:3] < 0.0) or np.any(
        trajectory[:, 1:3] > 1.0
    ):
        raise DetectorProtocolError("detector x/y are not screen-relative")
    return values.astype(np.float32, copy=False)


def _safe_stats(signal: np.ndarray) -> list[float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 2:
        signal = np.pad(signal, (0, 2 - signal.size))
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    centered = signal - mean
    if std > 1.0e-12:
        normalized = centered / std
        skew = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4) - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0
    return [
        mean,
        std,
        float(np.min(signal)),
        float(np.max(signal)),
        float(np.median(signal)),
        float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)),
        float(np.sqrt(np.mean(signal**2))),
        float(np.mean(np.abs(centered))),
        skew,
        kurtosis,
    ]


def _generic_modality_features(values: np.ndarray) -> np.ndarray:
    delta = np.diff(values.astype(np.float64), axis=0)
    if len(delta) == 0:
        delta = np.zeros((1, values.shape[1]), dtype=np.float64)
    blocks: list[float] = []
    for channel in range(values.shape[1]):
        blocks.extend(_safe_stats(values[:, channel]))
        blocks.extend(
            [
                float(np.mean(delta[:, channel])),
                float(np.std(delta[:, channel])),
                float(np.mean(np.abs(delta[:, channel]))),
            ]
        )
    return np.nan_to_num(
        np.asarray(blocks, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _riskcog_sensor_features(imu: np.ndarray) -> np.ndarray:
    """RiskCog 56-D definitions over the complete canonical event."""

    def arss(sensor: np.ndarray) -> float:
        return float(
            np.sqrt(np.sum(np.asarray(sensor, np.float64) ** 2))
            / max(len(sensor), 1)
        )

    def axis_features(values: np.ndarray) -> list[float]:
        values = np.asarray(values, np.float64)
        if len(values) < 2:
            values = np.pad(values, (0, 2 - len(values)))
        count = len(values)
        mean = float(values.mean())
        std = float(
            np.sqrt(np.sum((values - mean) ** 2) / max(count - 1, 1))
        )
        if std > 1.0e-12:
            z = (values - mean) / std
            kurtosis = float(np.mean(z**4) - 3.0)
            skew = float(np.mean(z**3))
        else:
            kurtosis = 0.0
            skew = 0.0
        return [
            float(np.sqrt(np.mean(values**2))),
            mean,
            float(np.mean(np.abs(values - mean))),
            std,
            float(values.min()),
            float(values.max()),
            kurtosis,
            skew,
            float(np.sum(np.abs(np.diff(np.sign(values)))) / count),
        ]

    output = [arss(imu[:, :3]), arss(imu[:, 3:6])]
    for axis in range(IMU_CHANNELS):
        output.extend(axis_features(imu[:, axis]))
    return np.asarray(output, dtype=np.float32)


def _motion_spectral_features(
    imu: np.ndarray, elapsed_seconds: np.ndarray
) -> np.ndarray:
    """Generic phone-motion magnitude/spectral block at physical effective Hz."""

    duration = float(elapsed_seconds[-1] - elapsed_seconds[0])
    hz = (
        float((len(elapsed_seconds) - 1) / duration)
        if len(elapsed_seconds) > 1 and duration > 1.0e-9
        else 100.0
    )

    def sensor_features(sensor: np.ndarray) -> list[float]:
        magnitude = np.sqrt(
            np.sum(np.asarray(sensor, np.float64) ** 2, axis=1) + 1.0e-12
        )
        if len(magnitude) < 4:
            magnitude = np.pad(magnitude, (0, 4 - len(magnitude)))
        spectrum = np.abs(np.fft.rfft(magnitude - magnitude.mean()))
        frequencies = np.fft.rfftfreq(len(magnitude), d=1.0 / hz)
        if len(spectrum) > 1:
            order = np.argsort(spectrum[1:])[::-1]
            peak = float(spectrum[1:][order[0]])
            peak_hz = float(frequencies[1:][order[0]])
            second = (
                float(spectrum[1:][order[1]]) if len(order) > 1 else 0.0
            )
        else:
            peak = peak_hz = second = 0.0
        return [
            float(magnitude.mean()),
            float(magnitude.var()),
            float(magnitude.max()),
            float(magnitude.min()),
            peak,
            peak_hz,
            second,
        ]

    return np.asarray(
        sensor_features(imu[:, :3]) + sensor_features(imu[:, 3:6]),
        dtype=np.float32,
    )


def _touchalytics_style_features(trajectory: np.ndarray) -> np.ndarray:
    """Fixed relative-geometry stroke block for trajectory paper cells."""

    contact = trajectory[:, 0] >= 0.5
    positions = np.flatnonzero(contact)
    active = trajectory[positions] if len(positions) >= 2 else trajectory
    xy = active[:, 1:3].astype(np.float64)
    pressure = active[:, 3].astype(np.float64)
    pointers = active[:, 4].astype(np.float64)
    elapsed = active[:, 7].astype(np.float64)
    delta_xy = np.diff(xy, axis=0)
    delta_t = np.diff(elapsed)
    positive = delta_t > 1.0e-9
    segment_length = np.linalg.norm(delta_xy, axis=1)
    speed = (
        segment_length[positive] / delta_t[positive]
        if np.any(positive)
        else np.zeros(1, dtype=np.float64)
    )
    # Acceleration is descriptive here.  Differentiating the valid speed
    # sequence avoids a shape-dependent branch when an event contains a
    # repeated timestamp while preserving one deterministic feature recipe.
    acceleration = (
        np.diff(speed) if len(speed) > 1 else np.zeros(1, dtype=np.float64)
    )
    displacement = xy[-1] - xy[0]
    straight_distance = float(np.linalg.norm(displacement))
    path_length = float(np.sum(segment_length))
    duration = float(max(elapsed[-1] - elapsed[0], 0.0))
    angles = (
        np.unwrap(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]))
        if len(delta_xy)
        else np.zeros(1, dtype=np.float64)
    )
    angle_change = np.diff(angles)
    output = [
        duration,
        float(xy[0, 0]),
        float(xy[0, 1]),
        float(xy[-1, 0]),
        float(xy[-1, 1]),
        float(displacement[0]),
        float(displacement[1]),
        straight_distance,
        path_length,
        straight_distance / max(path_length, 1.0e-9),
        float(np.mean(speed)),
        float(np.std(speed)),
        float(np.max(speed)),
        float(np.mean(np.abs(acceleration))),
        float(np.std(acceleration)),
        float(np.mean(angles)),
        float(np.std(angles)),
        float(np.mean(np.abs(angle_change))) if len(angle_change) else 0.0,
        float(np.mean(pressure)),
        float(np.std(pressure)),
        float(np.min(pressure)),
        float(np.max(pressure)),
        float(np.mean(pointers)),
        float(np.max(pointers)),
        float(np.mean(xy[:, 0])),
        float(np.std(xy[:, 0])),
        float(np.mean(xy[:, 1])),
        float(np.std(xy[:, 1])),
        float(len(active)),
        float(np.mean(contact)),
    ]
    return np.nan_to_num(
        np.asarray(output, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def extract_event_features(
    detector: str,
    modality: str,
    imu: np.ndarray,
    trajectory: np.ndarray,
) -> np.ndarray:
    spec = detector_spec(detector)
    if spec.family == "deep":
        raise DetectorProtocolError("deep detector has no handcrafted features")
    canonical = modality_values(imu, trajectory, modality)
    if spec.family == "feature":
        result = _generic_modality_features(canonical)
    elif modality == "trajectory_xytime":
        result = _touchalytics_style_features(trajectory)
    elif modality == "imu_only":
        # Pure inertial features: this modality never reads the touch stream.
        # The detector grid is a regular 100 Hz clock, so the spectral block
        # gets a synthetic axis instead of the trajectory elapsed channel; on a
        # regular grid that yields the same effective rate.
        elapsed = np.arange(len(imu), dtype=np.float64) / DETECTOR_GRID_HZ
        result = np.concatenate(
            [
                _riskcog_sensor_features(imu),
                _motion_spectral_features(imu, elapsed),
            ]
        )
    else:
        result = np.concatenate(
            [
                _riskcog_sensor_features(imu),
                _motion_spectral_features(imu, trajectory[:, 7]),
                _generic_modality_features(trajectory[:, (1, 2, 7)]),
                _touchalytics_style_features(trajectory),
            ]
        )
    if not np.isfinite(result).all():
        raise DetectorProtocolError("non-finite extracted features")
    return result.astype(np.float32, copy=False)


def build_classical_estimator(name: str, seed: int) -> Any:
    spec = detector_spec(name)
    if spec.family not in {"feature", "paper"}:
        raise DetectorProtocolError(f"{name} is not a classical detector")
    if name in {"hmog_style_svm", "paper_svm"}:
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                random_state=seed,
                max_iter=20_000,
            ),
        )
    if name == "hmog_style_rf":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "paper_xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise DetectorProtocolError(
                f"xgboost is required and cannot be skipped: {exc}"
            ) from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    raise DetectorProtocolError(f"no estimator recipe for {name}")


def classical_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
    elif hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)[:, 1]
    else:
        raise DetectorProtocolError("estimator exposes no continuous score")
    result = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise DetectorProtocolError("non-finite classical score")
    return result


def _masked_window(
    values: torch.Tensor,
    mask: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut each event to the frozen per-action window and pad short events."""

    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise DetectorProtocolError("deep values/mask shape mismatch")
    if target < 2:
        raise DetectorProtocolError("detector window must hold two samples")
    rows = []
    keeps = []
    for event, valid in zip(values, mask):
        selected = event[valid]
        if len(selected) < 2:
            raise DetectorProtocolError("deep event has fewer than two samples")
        kept = selected[:target]
        keep = len(kept)
        if keep < target:
            padding = torch.zeros(
                (target - keep, kept.shape[1]),
                dtype=kept.dtype,
                device=kept.device,
            )
            kept = torch.cat([kept, padding], dim=0)
        rows.append(kept)
        keeps.append(keep)
    windowed = torch.stack(rows, dim=0)
    positions = torch.arange(target, device=values.device)
    lengths = torch.tensor(keeps, device=values.device).unsqueeze(1)
    return windowed, positions.unsqueeze(0) < lengths


class _TemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        values: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        return values + attended


class BehaveFormerSTDATPad(nn.Module):
    """STDAT-style temporal/channel attention over a frozen per-action window."""

    def __init__(
        self,
        input_channels: int,
        modality: str,
        action: str,
    ):
        super().__init__()
        target_samples = action_window_samples(action)
        self.action = action
        self.target_samples = target_samples
        self.modality = modality
        hidden = 64
        self.project = nn.Sequential(
            nn.Conv1d(input_channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GELU(),
        )
        self.position = nn.Parameter(
            torch.randn(1, target_samples, hidden) * 0.02
        )
        self.temporal = nn.ModuleList(
            [_TemporalAttention(hidden, 4, 0.1) for _ in range(3)]
        )
        self.channel = nn.ModuleList(
            [_TemporalAttention(target_samples, 4, 0.1) for _ in range(3)]
        )
        self.feed_forward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden * 2, hidden),
                )
                for _ in range(3)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values, valid = _masked_window(values, mask, self.target_samples)
        padded = ~valid
        hidden = self.project(values.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.position
        hidden = hidden.masked_fill(padded.unsqueeze(-1), 0.0)
        for temporal, channel, feed_forward in zip(
            self.temporal, self.channel, self.feed_forward
        ):
            # Temporal attention runs over time, so padded steps are keys that
            # must not be attended.  Channel attention runs over the channel
            # axis, where every position is a real channel; padded time slots
            # are already zeroed and are re-zeroed after each block.
            hidden = temporal(hidden, key_padding_mask=padded)
            hidden = channel(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = hidden + feed_forward(hidden)
            hidden = hidden.masked_fill(padded.unsqueeze(-1), 0.0)
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        pooled = (self.norm(hidden) * weights).sum(dim=1) / weights.sum(dim=1)
        return self.head(pooled).squeeze(1)


class AuthConFormerPad(nn.Module):
    """Convolution/Transformer PAD over a frozen per-action window."""

    def __init__(
        self,
        input_channels: int,
        modality: str,
        action: str,
    ):
        super().__init__()
        target_samples = action_window_samples(action)
        self.action = action
        self.target_samples = target_samples
        self.modality = modality
        hidden = 96
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden, 7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
        )
        reduced = math.ceil(math.ceil(target_samples / 2) / 2)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.position = nn.Parameter(
            torch.randn(1, reduced + 1, hidden) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values, valid = _masked_window(values, mask, self.target_samples)
        hidden = self.stem(values.transpose(1, 2)).transpose(1, 2)
        # The stem halves the sequence twice, so the validity mask is pooled
        # the same way: a reduced step is real when any source step was.
        reduced = valid.unsqueeze(1).to(values.dtype)
        for _ in range(2):
            reduced = F.max_pool1d(
                reduced, kernel_size=2, stride=2, ceil_mode=True
            )
        reduced_valid = reduced.squeeze(1) > 0.0
        cls = self.cls.expand(len(hidden), -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        cls_valid = torch.ones_like(reduced_valid[:, :1])
        padded = ~torch.cat([cls_valid, reduced_valid], dim=1)
        hidden = self.encoder(hidden, src_key_padding_mask=padded)
        return self.head(self.norm(hidden[:, 0])).squeeze(1)


def build_deep_detector(name: str, modality: str, action: str) -> nn.Module:
    spec = detector_spec(name)
    if spec.family != "deep":
        raise DetectorProtocolError(f"{name} is not a deep detector")
    try:
        channels = DEEP_INPUT_CHANNELS[modality]
    except KeyError as exc:
        raise DetectorProtocolError(f"unknown modality {modality}") from exc
    if name == "behaveformer_stdat":
        return BehaveFormerSTDATPad(channels, modality, action)
    if name == "authconformer":
        return AuthConFormerPad(channels, modality, action)
    raise DetectorProtocolError(f"no deep implementation for {name}")


def software_versions() -> dict[str, str | None]:
    try:
        import xgboost

        xgboost_version: str | None = xgboost.__version__
    except Exception:
        xgboost_version = None
    return {"xgboost": xgboost_version}


__all__ = [
    "DEEP_DETECTORS",
    "DEEP_INPUT_CHANNELS",
    "DETECTOR_SPECS",
    "DetectorProtocolError",
    "DetectorSpec",
    "FEATURE_DETECTORS",
    "FORMAL_MODALITY_CHANNELS",
    "MODALITIES",
    "PAPER_DETECTORS",
    "REGISTERED_DETECTORS",
    "build_classical_estimator",
    "build_deep_detector",
    "classical_scores",
    "detector_spec",
    "extract_event_features",
    "modality_values",
    "software_versions",
]
