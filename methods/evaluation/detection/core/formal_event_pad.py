from __future__ import annotations

"""Unified action-specific Event PAD protocol.

This is the only paper-facing Event PAD interface.  It dispatches one frozen
cell to the registered Feature, Paper, or Deep implementation while keeping a
single data loader, coordinate contract, split policy, threshold policy, and
metric implementation.
"""

import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import joblib
import numpy as np
import sklearn

# This must be set before the first CUDA context/CuBLAS operation.  Setting it
# only inside the training function is too late in callers that initialize
# CUDA while resolving the requested physical device.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .audit import sha256_file, write_json_new
from .event_detectors import (
    DEEP_DETECTORS,
    DEEP_INPUT_CHANNELS,
    DETECTOR_SPECS,
    FORMAL_MODALITY_CHANNELS,
    MODALITIES,
    REGISTERED_DETECTORS,
    build_classical_estimator,
    build_deep_detector,
    classical_scores,
    detector_spec,
    extract_event_features,
    modality_values,
    software_versions,
)
from .event_pad import (
    ACTIONS,
    COORDINATE_SCHEMA,
    MANIFEST_SCHEMA,
    SHARDED_MANIFEST_SCHEMA,
    EventPadError,
    EventPartition,
    EventPartitionIndex,
    _load_manifest,
    _load_ragged_signals,
    load_event_partition,
    operating_metrics,
    select_development_thresholds,
)
from .event_qualification_gate import (
    validate_dev_qualification_receipt,
)


SCOPES = ("balanced_small", "full")
FORMAL_EPOCHS = 20
FORMAL_BOOTSTRAP_REPLICATES = 10_000
CELL_SCHEMA = "formal_event_pad_action_cell_v5"
SCOPE_SCHEMA = "formal_event_pad_scope_aggregate_v5"
DUAL_SCHEMA = "formal_event_pad_dual_completion_v5"
ALL_DETECTOR_PROTOCOL = "registered_feature_paper_deep_v4"
DEEP_ONLY_PROTOCOL = "deep_xytime_two_registered_detectors_v1"
DETECTOR_PROTOCOLS = (ALL_DETECTOR_PROTOCOL, DEEP_ONLY_PROTOCOL)


@dataclass(frozen=True)
class EventCell:
    scope: str
    action: str
    modality: str
    detector: str
    family: str
    resource: str

    @property
    def cell_id(self) -> str:
        return (
            f"{self.action}__{self.modality}__{self.detector}"
        )


@dataclass(frozen=True)
class ActionMetadata:
    labels: np.ndarray
    user_ids: np.ndarray
    session_ids: np.ndarray
    event_ids: np.ndarray
    source_cluster_ids: np.ndarray


def _load_train_development_before_test_freeze(
    manifest_path: str | Path,
    *,
    scope: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Validate train/development without resolving the sealed test source."""

    manifest = Path(manifest_path).resolve()
    rows: dict[str, dict[str, Any]] = {}
    test_rows_ignored = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventPadError(
                    f"{manifest}:{line_number}: invalid manifest JSON"
                ) from exc
            if not isinstance(raw, dict):
                raise EventPadError(
                    f"{manifest}:{line_number}: manifest row is not an object"
                )
            schema = raw.get("schema_version")
            split = str(raw.get("split", ""))
            if schema not in {MANIFEST_SCHEMA, SHARDED_MANIFEST_SCHEMA}:
                raise EventPadError(
                    f"{manifest}:{line_number}: wrong Event manifest schema"
                )
            if split == "test":
                test_rows_ignored += 1
                continue
            if split not in {"train", "development"} or split in rows:
                raise EventPadError(
                    "pre-test manifest requires unique train/development rows"
                )
            users = raw.get("user_ids")
            if (
                not isinstance(users, list)
                or not users
                or len(users) != len(set(users))
            ):
                raise EventPadError(f"{split}: invalid manifest user IDs")
            if str(raw.get("scope")) != scope:
                raise EventPadError(
                    f"{split}: manifest scope does not match requested cell"
                )
            if schema == MANIFEST_SCHEMA:
                source = Path(str(raw.get("source", ""))).resolve()
                if (
                    not source.is_file()
                    or raw.get("source_sha256") != sha256_file(source)
                ):
                    raise EventPadError(
                        f"{split}: source missing or hash mismatch"
                    )
                rows[split] = {**raw, "source": source}
                continue
            shards = raw.get("shards")
            if not isinstance(shards, list) or not shards:
                raise EventPadError(f"{split}: no ragged shards")
            normalized_shards: list[dict[str, Any]] = []
            seen_sources: set[Path] = set()
            shard_users: list[str] = []
            event_total = 0
            for shard_number, shard in enumerate(shards):
                if not isinstance(shard, dict):
                    raise EventPadError(
                        f"{split}: shard {shard_number} is malformed"
                    )
                source = Path(str(shard.get("source", ""))).resolve()
                if (
                    source in seen_sources
                    or not source.is_file()
                    or shard.get("source_sha256") != sha256_file(source)
                ):
                    raise EventPadError(
                        f"{split}: shard source missing, duplicate, or changed"
                    )
                seen_sources.add(source)
                shard_user = str(shard.get("user_id", ""))
                shard_events = int(shard.get("events", -1))
                if not shard_user or shard_events < 1:
                    raise EventPadError(
                        f"{split}: invalid shard user/event binding"
                    )
                shard_users.append(shard_user)
                event_total += shard_events
                normalized_shards.append({**shard, "source": source})
            if (
                shard_users != users
                or len(shard_users) != len(set(shard_users))
                or event_total != int(raw.get("events", -1))
            ):
                raise EventPadError(
                    f"{split}: shard/user/event manifest mismatch"
                )
            rows[split] = {**raw, "shards": normalized_shards}
    if set(rows) != {"train", "development"} or test_rows_ignored != 1:
        raise EventPadError(
            "manifest must expose train/development and one sealed test row"
        )
    overlap = set(rows["train"]["user_ids"]) & set(
        rows["development"]["user_ids"]
    )
    if overlap:
        raise EventPadError(
            f"user leakage between train/development: {sorted(overlap)}"
        )
    return rows, {
        "manifest_test_rows_ignored_before_freeze": test_rows_ignored,
        "test_source_paths_resolved_before_freeze": 0,
        "test_signal_file_stat_or_hash_calls_before_freeze": 0,
        "test_signal_arrays_loaded_before_freeze": 0,
    }


def _load_manifest_after_pre_test_freeze(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    freeze_receipt_path: str | Path,
    scope: str,
) -> dict[str, dict[str, Any]]:
    """Open and integrity-check test only after a valid freeze receipt exists."""

    manifest = Path(manifest_path).resolve()
    freeze_path = Path(freeze_receipt_path).resolve()
    if not freeze_path.is_file():
        raise EventPadError(
            "sealed test cannot open before pre-test freeze receipt"
        )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventPadError("invalid pre-test freeze receipt") from exc
    if (
        not isinstance(freeze, dict)
        or freeze.get("frozen_before_test_open") is not True
        or freeze.get("manifest_sha256") != expected_manifest_sha256
        or freeze.get("manifest_test_rows_ignored_before_freeze") != 1
        or freeze.get("test_source_paths_resolved_before_freeze") != 0
        or freeze.get(
            "test_signal_file_stat_or_hash_calls_before_freeze"
        ) != 0
        or freeze.get("test_signal_arrays_loaded_before_freeze") != 0
    ):
        raise EventPadError("pre-test freeze receipt contract failed")
    if sha256_file(manifest) != expected_manifest_sha256:
        raise EventPadError("Event manifest changed after pre-test freeze")
    rows = _load_manifest(manifest)
    if any(str(row.get("scope")) != scope for row in rows.values()):
        raise EventPadError("manifest scope does not match requested cell")
    return rows


def event_cells(
    scope: str,
    detector_protocol: str = ALL_DETECTOR_PROTOCOL,
) -> list[EventCell]:
    if scope not in SCOPES:
        raise EventPadError(f"unknown Event scope {scope!r}")
    detectors = protocol_detectors(detector_protocol)
    cells = [
        EventCell(
            scope=scope,
            action=action,
            modality=modality,
            detector=detector,
            family=detector_spec(detector).family,
            resource=detector_spec(detector).execution,
        )
        for action in ACTIONS
        for modality in MODALITIES
        for detector in detectors
    ]
    expected = 90 if detector_protocol == ALL_DETECTOR_PROTOCOL else 30
    if (
        len(cells) != expected
        or len({cell.cell_id for cell in cells}) != expected
    ):
        raise EventPadError(
            f"registered Event scope must contain exactly {expected} cells"
        )
    return cells


def protocol_detectors(detector_protocol: str) -> tuple[str, ...]:
    """Return the exact detector registry for one frozen Event protocol."""

    if detector_protocol == ALL_DETECTOR_PROTOCOL:
        return REGISTERED_DETECTORS
    if detector_protocol == DEEP_ONLY_PROTOCOL:
        return DEEP_DETECTORS
    raise EventPadError(
        f"unknown Event detector protocol {detector_protocol!r}"
    )


def registered_event_cells(scope: str) -> list[EventCell]:
    """Backward-compatible complete Feature/Paper/Deep registry."""

    return event_cells(scope, ALL_DETECTOR_PROTOCOL)


def deep_event_cells(scope: str) -> list[EventCell]:
    """The user-confirmed Deep-only, XY+time formal Event registry."""

    return event_cells(scope, DEEP_ONLY_PROTOCOL)


def _seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _action_indices(
    partition: EventPartition | EventPartitionIndex, action: str
) -> np.ndarray:
    indices = np.flatnonzero(partition.actions == action)
    if len(indices) < 2:
        raise EventPadError(f"{partition.split}/{action}: no usable events")
    labels = partition.labels[indices]
    if set(labels.tolist()) != {0, 1}:
        raise EventPadError(
            f"{partition.split}/{action}: both genuine and fake are required"
        )
    return indices


def _metadata(
    partition: EventPartition | EventPartitionIndex, action: str
) -> ActionMetadata:
    indices = _action_indices(partition, action)
    return ActionMetadata(
        labels=partition.labels[indices],
        user_ids=partition.user_ids[indices],
        session_ids=partition.session_ids[indices],
        event_ids=partition.event_ids[indices],
        source_cluster_ids=partition.source_cluster_ids[indices],
    )


def _iter_action_batches(
    partition: EventPartition | EventPartitionIndex,
    action: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    selected = _action_indices(partition, action)
    rng = np.random.default_rng(seed)
    if isinstance(partition, EventPartition):
        order = np.arange(len(selected))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            raw = selected[positions]
            yield (
                partition.imu[raw],
                partition.trajectory[raw],
                partition.mask[raw],
                partition.labels[raw],
                positions,
            )
        return

    shard_order = np.arange(len(partition.shards))
    if shuffle:
        rng.shuffle(shard_order)
    for shard_number in shard_order:
        shard = partition.shards[int(shard_number)]
        global_start = int(shard["global_start"])
        global_end = global_start + int(shard["events"])
        view_positions = np.flatnonzero(
            (selected >= global_start) & (selected < global_end)
        )
        if not len(view_positions):
            continue
        local_indices = selected[view_positions] - global_start
        order = np.arange(len(local_indices))
        if shuffle:
            rng.shuffle(order)
        imu_flat, trajectory_flat, offsets = _load_ragged_signals(shard)
        for start in range(0, len(order), batch_size):
            take = order[start : start + batch_size]
            local = local_indices[take]
            positions = view_positions[take]
            lengths = np.asarray(
                [
                    int(offsets[index + 1] - offsets[index])
                    for index in local
                ],
                dtype=np.int64,
            )
            maximum = int(lengths.max())
            imu = np.zeros((len(local), maximum, 6), dtype=np.float32)
            trajectory = np.zeros(
                (len(local), maximum, 9), dtype=np.float32
            )
            mask = np.zeros((len(local), maximum), dtype=bool)
            for row, (index, length) in enumerate(zip(local, lengths)):
                left = int(offsets[index])
                right = int(offsets[index + 1])
                imu[row, :length] = imu_flat[left:right]
                trajectory[row, :length] = trajectory_flat[left:right]
                mask[row, :length] = True
            yield (
                imu,
                trajectory,
                mask,
                partition.labels[selected[positions]],
                positions,
            )


def _feature_matrix(
    partition: EventPartition | EventPartitionIndex,
    *,
    action: str,
    modality: str,
    detector: str,
) -> tuple[np.ndarray, ActionMetadata]:
    metadata = _metadata(partition, action)
    rows: list[np.ndarray | None] = [None] * len(metadata.labels)
    for imu, trajectory, mask, _labels, positions in _iter_action_batches(
        partition,
        action,
        batch_size=256,
        shuffle=False,
        seed=0,
    ):
        for batch_row, position in enumerate(positions):
            valid = mask[batch_row]
            rows[int(position)] = extract_event_features(
                detector,
                modality,
                imu[batch_row, valid],
                trajectory[batch_row, valid],
            )
    if any(row is None for row in rows):
        raise EventPadError("streaming feature extraction lost an event")
    matrix = np.stack([row for row in rows if row is not None]).astype(
        np.float32
    )
    if not np.isfinite(matrix).all():
        raise EventPadError("feature matrix is non-finite")
    return matrix, metadata


def _deep_values(
    imu: np.ndarray, trajectory: np.ndarray, modality: str
) -> np.ndarray:
    return np.stack(
        [
            modality_values(imu[index], trajectory[index], modality)
            for index in range(len(imu))
        ],
        axis=0,
    )


def _fit_normalizer(
    partition: EventPartition | EventPartitionIndex,
    *,
    action: str,
    modality: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Single source of truth: a second hand-written copy of this table silently
    # went stale when the IMU modality became six pure inertial channels.
    channels = DEEP_INPUT_CHANNELS[modality]
    total = np.zeros(channels, dtype=np.float64)
    square = np.zeros(channels, dtype=np.float64)
    count = 0
    for imu, trajectory, mask, _labels, _positions in _iter_action_batches(
        partition,
        action,
        batch_size=128,
        shuffle=False,
        seed=0,
    ):
        values = _deep_values(imu, trajectory, modality)
        active = values[mask]
        total += active.sum(axis=0, dtype=np.float64)
        square += np.square(active, dtype=np.float64).sum(axis=0)
        count += len(active)
    if count < 2:
        raise EventPadError("deep normalizer has insufficient training samples")
    mean = total / count
    variance = np.maximum(square / count - mean**2, 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _normalize_deep(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (values - mean[None, None, :]) / std[None, None, :]
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def _fit_deep(
    partition: EventPartition | EventPartitionIndex,
    *,
    action: str,
    modality: str,
    detector: str,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, list[float]]:
    mean, std = _fit_normalizer(
        partition, action=action, modality=modality
    )
    model = build_deep_detector(detector, modality, action).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
    )
    selected = _action_indices(partition, action)
    labels = partition.labels[selected]
    real = int(np.sum(labels == 0))
    fake = int(np.sum(labels == 1))
    if real < 1 or fake < 1:
        raise EventPadError("deep action cell requires both labels")
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for imu, trajectory, mask, batch_labels, _positions in (
            _iter_action_batches(
                partition,
                action,
                batch_size=32,
                shuffle=True,
                seed=seed + epoch,
            )
        ):
            values = _normalize_deep(
                _deep_values(imu, trajectory, modality),
                mask,
                mean,
                std,
            )
            tensor = torch.from_numpy(values).to(device)
            tensor_mask = torch.from_numpy(mask).to(device)
            target = torch.from_numpy(
                batch_labels.astype(np.float32)
            ).to(device)
            logits = model(tensor, tensor_mask)
            per_event = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            weights = torch.where(
                target >= 0.5,
                torch.full_like(target, (real + fake) / (2.0 * fake)),
                torch.full_like(target, (real + fake) / (2.0 * real)),
            )
            loss = torch.sum(per_event * weights) / torch.sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise EventPadError("deep epoch produced no batches")
        losses.append(float(np.mean(epoch_losses)))
    return model.eval(), mean, std, losses


@torch.no_grad()
def _score_deep(
    model: nn.Module,
    partition: EventPartition | EventPartitionIndex,
    *,
    action: str,
    modality: str,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, list[float], ActionMetadata]:
    metadata = _metadata(partition, action)
    scores = np.empty(len(metadata.labels), dtype=np.float64)
    latency = np.empty(len(metadata.labels), dtype=np.float64)
    for imu, trajectory, mask, _labels, positions in _iter_action_batches(
        partition,
        action,
        batch_size=64,
        shuffle=False,
        seed=0,
    ):
        values = _normalize_deep(
            _deep_values(imu, trajectory, modality), mask, mean, std
        )
        tensor = torch.from_numpy(values).to(device)
        tensor_mask = torch.from_numpy(mask).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        values_out = torch.sigmoid(model(tensor, tensor_mask))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        scores[positions] = values_out.cpu().numpy().astype(np.float64)
        latency[positions] = elapsed / max(len(positions), 1)
    return scores, latency.tolist(), metadata


def _descriptive_eer(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    candidates = np.concatenate(
        [
            np.unique(scores),
            [np.nextafter(float(np.max(scores)), np.inf)],
        ]
    )
    real = labels == 0
    fake = labels == 1
    frr = np.asarray(
        [np.mean(scores[real] >= threshold) for threshold in candidates]
    )
    far = np.asarray(
        [np.mean(scores[fake] < threshold) for threshold in candidates]
    )
    index = int(
        np.lexsort(
            (candidates, np.maximum(frr, far), np.abs(frr - far))
        )[0]
    )
    return {
        "eer": float((far[index] + frr[index]) / 2.0),
        "far": float(far[index]),
        "frr": float(frr[index]),
        "threshold": float(candidates[index]),
        "descriptive_test_sweep": True,
    }


def _cluster_bootstrap(
    metadata: ActionMetadata,
    scores: np.ndarray,
    threshold: float,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    users = np.unique(metadata.user_ids)
    if len(users) < 2:
        raise EventPadError("user-cluster bootstrap needs at least two users")
    real_den = np.zeros(len(users), dtype=np.int64)
    fake_den = np.zeros(len(users), dtype=np.int64)
    real_err = np.zeros(len(users), dtype=np.int64)
    fake_err = np.zeros(len(users), dtype=np.int64)
    for index, user in enumerate(users):
        selected = metadata.user_ids == user
        real = selected & (metadata.labels == 0)
        fake = selected & (metadata.labels == 1)
        real_den[index] = int(real.sum())
        fake_den[index] = int(fake.sum())
        real_err[index] = int(np.sum(scores[real] >= threshold))
        fake_err[index] = int(np.sum(scores[fake] < threshold))
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(users), size=(replicates, len(users)), endpoint=False
    )

    def interval(denominator: np.ndarray, errors: np.ndarray) -> dict[str, float]:
        sampled_denominator = denominator[draws].sum(axis=1)
        sampled_errors = errors[draws].sum(axis=1)
        valid = sampled_denominator > 0
        if not np.any(valid):
            raise EventPadError("bootstrap has no valid label denominator")
        values = sampled_errors[valid] / sampled_denominator[valid]
        return {
            "estimate": float(errors.sum() / denominator.sum()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }

    far = interval(fake_den, fake_err)
    frr = interval(real_den, real_err)
    return {
        "replicates_requested": int(replicates),
        "unit": "user_cluster",
        "far": far,
        "frr": frr,
        "fake_event_detection_rate": {
            "estimate": 1.0 - far["estimate"],
            "ci_low": 1.0 - far["ci_high"],
            "ci_high": 1.0 - far["ci_low"],
        },
    }


def _environment(device: torch.device) -> dict[str, Any]:
    properties = (
        torch.cuda.get_device_properties(device)
        if device.type == "cuda"
        else None
    )
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        **software_versions(),
        "device": str(device),
        "physical_device_index": (
            int(torch.cuda.current_device()) if device.type == "cuda" else None
        ),
        "physical_device_uuid": (
            str(properties.uuid) if properties is not None else None
        ),
        "physical_device_name": (
            str(properties.name) if properties is not None else None
        ),
    }


def run_formal_event_cell(
    *,
    manifest: str | Path,
    output_dir: str | Path,
    scope: str,
    action: str,
    modality: str,
    detector: str,
    device_name: str,
    epochs: int = FORMAL_EPOCHS,
    bootstrap_replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = 42,
    formal: bool = True,
    detector_protocol: str = ALL_DETECTOR_PROTOCOL,
    qualification_receipt: str | Path | None = None,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    if scope not in SCOPES or action not in ACTIONS:
        raise EventPadError("unregistered Event scope/action")
    allowed_detectors = protocol_detectors(detector_protocol)
    if modality not in MODALITIES or detector not in allowed_detectors:
        raise EventPadError("unregistered Event modality/detector")
    if formal and detector_protocol != DEEP_ONLY_PROTOCOL:
        raise EventPadError(
            "new formal Event runs require the Deep-only XY+time protocol"
        )
    if formal and (
        epochs != FORMAL_EPOCHS
        or bootstrap_replicates != FORMAL_BOOTSTRAP_REPLICATES
        or seed != 42
    ):
        raise EventPadError(
            "formal Event cell fixes 20 epochs, 10,000 CI draws, and seed 42"
        )
    spec = detector_spec(detector)
    device = torch.device(device_name)
    if spec.family == "deep":
        if formal and device.type != "cuda":
            raise EventPadError("formal deep Event detector requires physical CUDA")
        if formal and "CUDA_VISIBLE_DEVICES" in os.environ:
            raise EventPadError(
                "CUDA_VISIBLE_DEVICES remapping is forbidden for formal Event PAD"
            )
        if formal and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
            ":4096:8",
            ":16:8",
        }:
            raise EventPadError(
                "formal deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"
            )
        if formal and device.index is None:
            raise EventPadError(
                "formal deep Event detector requires an explicit physical CUDA index"
            )
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise EventPadError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
            if (
                device.index is not None
                and torch.cuda.current_device() != device.index
            ):
                raise EventPadError("physical CUDA device proof mismatch")
            torch.cuda.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats(device)
    elif device.type != "cpu":
        raise EventPadError("Feature/Paper Event detectors run on CPU")
    _seed(seed)
    qualification_binding: dict[str, str] | None = None
    if formal and detector_protocol == DEEP_ONLY_PROTOCOL:
        if qualification_receipt is None:
            raise EventPadError(
                "Deep-only formal Event cell requires the development "
                "qualification receipt before test access"
            )
        qualification_path = Path(qualification_receipt).resolve()
        validate_dev_qualification_receipt(
            qualification_path,
            manifest=manifest,
        )
        qualification_binding = {
            "path": str(qualification_path),
            "sha256": sha256_file(qualification_path),
        }
    elif qualification_receipt is not None:
        raise EventPadError(
            "development qualification receipt is only valid for the "
            "Deep-only formal Event protocol"
        )
    manifest_path = Path(manifest).resolve()
    manifest_sha256 = sha256_file(manifest_path)
    channel_schema = list(FORMAL_MODALITY_CHANNELS[modality])
    pre_test_rows, pre_test_access = (
        _load_train_development_before_test_freeze(
            manifest_path,
            scope=scope,
        )
    )
    train = load_event_partition(pre_test_rows["train"])
    development = load_event_partition(pre_test_rows["development"])
    losses: list[float] | None = None
    normalizer: dict[str, list[float]] | None = None
    fit_started = time.perf_counter()
    if spec.family == "deep":
        model, mean, std, losses = _fit_deep(
            train,
            action=action,
            modality=modality,
            detector=detector,
            device=device,
            epochs=epochs,
            seed=seed,
        )
        development_scores, _, development_metadata = _score_deep(
            model,
            development,
            action=action,
            modality=modality,
            device=device,
            mean=mean,
            std=std,
        )
        normalizer = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        train_features, train_metadata = _feature_matrix(
            train,
            action=action,
            modality=modality,
            detector=detector,
        )
        model = build_classical_estimator(detector, seed)
        model.fit(train_features, train_metadata.labels)
        development_features, development_metadata = _feature_matrix(
            development,
            action=action,
            modality=modality,
            detector=detector,
        )
        development_scores = classical_scores(model, development_features)
    fit_and_development_seconds = time.perf_counter() - fit_started
    thresholds = select_development_thresholds(
        development_metadata.labels, development_scores
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / (
        "checkpoint.pt" if spec.family == "deep" else "model.joblib"
    )
    if spec.family == "deep":
        torch.save(
            {
                "schema_version": CELL_SCHEMA,
                "detector_protocol": detector_protocol,
                "action": action,
                "modality": modality,
                "detector": detector,
                "coordinate_schema": COORDINATE_SCHEMA,
                "input_channel_schema": channel_schema,
                "input_channel_count": len(channel_schema),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "training_from_scratch": True,
                "preexisting_checkpoint_loaded": False,
                "preexisting_result_loaded": False,
                "historical_model_artifacts_read": [],
                "model_initialization": "fresh_seeded_initialization",
                "seed": seed,
                "normalizer": normalizer,
                "model_state": model.state_dict(),
            },
            model_path,
        )
    else:
        joblib.dump(model, model_path)
    environment = _environment(device)
    config = {
        "schema_version": CELL_SCHEMA,
        "formal_result": bool(formal),
        "detector_protocol": detector_protocol,
        "scope": scope,
        "action": action,
        "modality": modality,
        "detector": detector,
        "family": spec.family,
        "resource": spec.execution,
        "feature_recipe": spec.feature_recipe,
        "reference": spec.reference,
        "coordinate_schema": COORDINATE_SCHEMA,
        "time_schema": "elapsed_seconds_since_event_start_v1",
        "channel_count": len(channel_schema),
        "channel_schema": channel_schema,
        "formal_modalities": list(MODALITIES),
        "all_formal_modalities_include_xy_and_time": False,
        "deep_learning_only": detector_protocol == DEEP_ONLY_PROTOCOL,
        "plain_imu_only_allowed": True,
        "training_from_scratch": True,
        "preexisting_checkpoint_loaded": False,
        "preexisting_result_loaded": False,
        "historical_model_artifacts_read": [],
        "model_initialization": "fresh_seeded_initialization",
        "threshold_split": "development",
        "test_opened_before_freeze": False,
        "epochs": epochs if spec.family == "deep" else None,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "normalizer": normalizer,
        "environment": environment,
        "development_qualification": qualification_binding,
    }
    write_json_new(output / "frozen_config.json", config)
    write_json_new(
        output / "thresholds.json",
        {
            "schema_version": CELL_SCHEMA,
            "detector_protocol": detector_protocol,
            "selection_split": "development",
            "score_direction": "larger_is_more_fake",
            "manifest_sha256": manifest_sha256,
            "model_sha256": sha256_file(model_path),
            "frozen_config_sha256": sha256_file(
                output / "frozen_config.json"
            ),
            **thresholds,
        },
    )
    write_json_new(
        output / "pre_test_freeze_receipt.json",
        {
            "schema_version": CELL_SCHEMA,
            "detector_protocol": detector_protocol,
            "frozen_before_test_open": True,
            "test_signal_arrays_loaded": 0,
            **pre_test_access,
            "action": action,
            "modality": modality,
            "detector": detector,
            "coordinate_schema": COORDINATE_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "model_sha256": sha256_file(model_path),
            "frozen_config_sha256": sha256_file(
                output / "frozen_config.json"
            ),
            "thresholds_sha256": sha256_file(output / "thresholds.json"),
            "training_from_scratch": True,
            "preexisting_result_loaded": False,
            "development_qualification": qualification_binding,
        },
    )

    test_started = time.perf_counter()
    rows = _load_manifest_after_pre_test_freeze(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        freeze_receipt_path=output / "pre_test_freeze_receipt.json",
        scope=scope,
    )
    test = load_event_partition(rows["test"])
    if spec.family == "deep":
        test_scores, latency, test_metadata = _score_deep(
            model,
            test,
            action=action,
            modality=modality,
            device=device,
            mean=np.asarray(normalizer["mean"], dtype=np.float32),
            std=np.asarray(normalizer["std"], dtype=np.float32),
        )
    else:
        started = time.perf_counter_ns()
        test_features, test_metadata = _feature_matrix(
            test,
            action=action,
            modality=modality,
            detector=detector,
        )
        test_scores = classical_scores(model, test_features)
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        latency = [elapsed / max(len(test_scores), 1)] * len(test_scores)
    with (output / "test_scores.jsonl").open("x", encoding="utf-8") as handle:
        for index, score in enumerate(test_scores):
            handle.write(
                json.dumps(
                    {
                        "event_id": str(test_metadata.event_ids[index]),
                        "user_id": str(test_metadata.user_ids[index]),
                        "session_id": str(test_metadata.session_ids[index]),
                        "source_cluster_id": str(
                            test_metadata.source_cluster_ids[index]
                        ),
                        "action": action,
                        "label": int(test_metadata.labels[index]),
                        "fake_high_score": float(score),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    primary = operating_metrics(
        test_metadata.labels, test_scores, thresholds["eer"]
    )
    descriptive = _descriptive_eer(test_metadata.labels, test_scores)
    bootstrap = _cluster_bootstrap(
        test_metadata,
        test_scores,
        thresholds["eer"],
        replicates=bootstrap_replicates,
        seed=seed,
    )
    test_seconds = time.perf_counter() - test_started
    latency_values = np.asarray(latency, dtype=np.float64)
    artifact_paths = {
        "model": model_path,
        "frozen_config": output / "frozen_config.json",
        "thresholds": output / "thresholds.json",
        "pre_test_freeze_receipt": output / "pre_test_freeze_receipt.json",
        "test_scores": output / "test_scores.jsonl",
    }
    summary = {
        "schema_version": CELL_SCHEMA,
        "status": "pass",
        "formal_result": bool(formal),
        "detector_protocol": detector_protocol,
        "scope": scope,
        "action": action,
        "modality": modality,
        "detector": detector,
        "family": spec.family,
        "coordinate_schema": COORDINATE_SCHEMA,
        "channel_count": len(channel_schema),
        "channel_schema": channel_schema,
        "formal_modalities": list(MODALITIES),
        "all_formal_modalities_include_xy_and_time": False,
        "deep_learning_only": detector_protocol == DEEP_ONLY_PROTOCOL,
        "plain_imu_only_allowed": True,
        "training_from_scratch": True,
        "preexisting_checkpoint_loaded": False,
        "preexisting_result_loaded": False,
        "historical_model_artifacts_read": [],
        "primary_metrics": {
            "far": primary["far_fake_accepted"],
            "frr": primary["frr_real_rejected"],
            "fake_event_detection_rate":
                1.0 - primary["far_fake_accepted"],
            "roc_auc": float(roc_auc_score(test_metadata.labels, test_scores)),
            "development_eer_threshold": thresholds["eer"],
            "descriptive_test_eer": descriptive["eer"],
        },
        "bootstrap_95ci": bootstrap,
        "coverage": {
            "expected": int(len(test_metadata.labels)),
            "scored": int(len(test_scores)),
            "rate": 1.0,
        },
        "failure_counts": {"preprocessing": 0, "scoring": 0, "total": 0},
        "lineage_failures": 0,
        "latency_ms_per_event": {
            "p50": float(np.quantile(latency_values, 0.50)),
            "p95": float(np.quantile(latency_values, 0.95)),
        },
        "training_loss_by_epoch": losses,
        "physical_device_index": (
            int(torch.cuda.current_device()) if device.type == "cuda" else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "environment": environment,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model_sha256": sha256_file(model_path),
        "artifact_sha256": {
            name: sha256_file(path)
            for name, path in artifact_paths.items()
        },
        "runtime_seconds": {
            "fit_and_development": float(fit_and_development_seconds),
            "sealed_test_and_metrics": float(test_seconds),
            "total": float(time.perf_counter() - wall_started),
        },
        "test_threshold_selection_calls": 0,
        "test_opened_before_freeze": False,
        "development_qualification": qualification_binding,
    }
    write_json_new(output / "summary.json", summary)
    return summary


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EventPadError(f"missing Event artifact {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EventPadError(f"Event artifact is not an object: {path}")
    return value


def _finite_tree(value: Any, *, context: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not np.isfinite(float(value)):
            raise EventPadError(f"{context}: non-finite value")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, context=f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, context=f"{context}[{index}]")
        return
    raise EventPadError(f"{context}: unsupported value type")


def _artifact_protocol(value: Mapping[str, Any]) -> str:
    """Read the explicit detector protocol from a formal artifact."""

    return str(value.get("detector_protocol"))


def validate_event_cell_artifacts(
    summary_path: str | Path,
    cell: EventCell,
    *,
    detector_protocol: str,
    allowed_physical_gpus: set[int] | None = None,
) -> dict[str, Any]:
    """Validate one immutable Event cell and its protocol/hash/device chain."""

    protocol_detectors(detector_protocol)
    path = Path(summary_path).resolve()
    summary = _read_object(path)
    coverage = summary.get("coverage", {})
    failures = summary.get("failure_counts", {})
    if (
        summary.get("schema_version") != CELL_SCHEMA
        or summary.get("status") != "pass"
        or summary.get("formal_result") is not True
        or _artifact_protocol(summary)
        != detector_protocol
        or summary.get("scope") != cell.scope
        or summary.get("action") != cell.action
        or summary.get("modality") != cell.modality
        or summary.get("detector") != cell.detector
        or summary.get("family") != cell.family
        or summary.get("coordinate_schema") != COORDINATE_SCHEMA
        or not isinstance(coverage, Mapping)
        or not isinstance(coverage.get("expected"), int)
        or coverage.get("expected", 0) < 1
        or coverage.get("expected") != coverage.get("scored")
        or coverage.get("rate") != 1.0
        or not isinstance(failures, Mapping)
        or any(
            failures.get(field) != 0
            for field in ("preprocessing", "scoring", "total")
        )
        or summary.get("lineage_failures") != 0
        or summary.get("test_threshold_selection_calls") != 0
        or summary.get("test_opened_before_freeze") is not False
    ):
        raise EventPadError(f"{cell.scope}/{cell.cell_id}: completion gate failed")
    for field in (
        "primary_metrics",
        "bootstrap_95ci",
        "latency_ms_per_event",
    ):
        if field not in summary:
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: missing {field}"
            )
        _finite_tree(
            summary[field],
            context=f"{cell.scope}/{cell.cell_id}/{field}",
        )

    physical = summary.get("physical_device_index")
    environment = summary.get("environment")
    if cell.family == "deep":
        if (
            isinstance(physical, bool)
            or not isinstance(physical, int)
            or physical < 0
        ):
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: physical GPU proof failed"
            )
        if detector_protocol == DEEP_ONLY_PROTOCOL and (
            not isinstance(environment, Mapping)
            or environment.get("physical_device_index") != physical
            or environment.get("device") != f"cuda:{physical}"
            or not isinstance(
                environment.get("physical_device_uuid"), str
            )
            or not environment["physical_device_uuid"]
            or not isinstance(
                environment.get("physical_device_name"), str
            )
            or not environment["physical_device_name"]
        ):
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: physical GPU proof failed"
            )
        if (
            allowed_physical_gpus is not None
            and physical not in allowed_physical_gpus
        ):
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: GPU {physical} was not requested"
            )
        losses = summary.get("training_loss_by_epoch")
        if (
            not isinstance(losses, list)
            or len(losses) != FORMAL_EPOCHS
        ):
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: formal epoch record failed"
            )
        _finite_tree(
            losses,
            context=f"{cell.scope}/{cell.cell_id}/training_loss_by_epoch",
        )
    elif physical is not None:
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: CPU detector claimed a GPU"
        )

    # The following immutable artifact chain is mandatory for the new
    # Deep-only formal protocol.  It is intentionally not inferred from an
    # old 180-cell result.
    if detector_protocol != DEEP_ONLY_PROTOCOL:
        return summary
    root = path.parent
    expected_channels = list(FORMAL_MODALITY_CHANNELS[cell.modality])
    manifest = Path(str(summary.get("manifest", ""))).resolve()
    if (
        not manifest.is_file()
        or summary.get("manifest_sha256") != sha256_file(manifest)
        or summary.get("channel_count") != len(expected_channels)
        or summary.get("channel_schema") != expected_channels
        or summary.get("formal_modalities") != list(MODALITIES)
        or summary.get("all_formal_modalities_include_xy_and_time") is not False
        or summary.get("deep_learning_only") is not True
        or summary.get("plain_imu_only_allowed") is not True
        or summary.get("training_from_scratch") is not True
        or summary.get("preexisting_checkpoint_loaded") is not False
        or summary.get("preexisting_result_loaded") is not False
        or summary.get("historical_model_artifacts_read") != []
    ):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: XY+time/manifest/from-scratch "
            "binding failed"
        )
    qualification = summary.get("development_qualification")
    if not isinstance(qualification, Mapping):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: qualification binding missing"
        )
    qualification_path = Path(
        str(qualification.get("path", ""))
    ).resolve()
    if (
        set(qualification) != {"path", "sha256"}
        or not qualification_path.is_file()
        or qualification.get("sha256") != sha256_file(qualification_path)
    ):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: qualification hash binding failed"
        )
    validate_dev_qualification_receipt(
        qualification_path,
        manifest=manifest,
    )
    artifact_paths = {
        "model": root / "checkpoint.pt",
        "frozen_config": root / "frozen_config.json",
        "thresholds": root / "thresholds.json",
        "pre_test_freeze_receipt": root / "pre_test_freeze_receipt.json",
        "test_scores": root / "test_scores.jsonl",
    }
    hashes = summary.get("artifact_sha256")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(artifact_paths)
    ):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: artifact hash inventory failed"
        )
    for name, artifact in artifact_paths.items():
        if (
            not artifact.is_file()
            or not isinstance(hashes.get(name), str)
            or hashes[name] != sha256_file(artifact)
        ):
            raise EventPadError(
                f"{cell.scope}/{cell.cell_id}: {name} hash binding failed"
            )
    model_sha256 = hashes["model"]
    if summary.get("model_sha256") != model_sha256:
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: summary/model binding failed"
        )
    try:
        checkpoint = torch.load(
            artifact_paths["model"],
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: checkpoint metadata unreadable"
        ) from exc
    checkpoint_identity = {
        "action": cell.action,
        "modality": cell.modality,
        "detector": cell.detector,
    }
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != CELL_SCHEMA
        or checkpoint.get("detector_protocol") != detector_protocol
        or any(
            checkpoint.get(key) != value
            for key, value in checkpoint_identity.items()
        )
        or checkpoint.get("coordinate_schema") != COORDINATE_SCHEMA
        or checkpoint.get("input_channel_schema") != expected_channels
        or checkpoint.get("input_channel_count") != len(expected_channels)
        or checkpoint.get("manifest") != str(manifest)
        or checkpoint.get("manifest_sha256") != sha256_file(manifest)
        or checkpoint.get("training_from_scratch") is not True
        or checkpoint.get("preexisting_checkpoint_loaded") is not False
        or checkpoint.get("preexisting_result_loaded") is not False
        or checkpoint.get("historical_model_artifacts_read") != []
        or checkpoint.get("model_initialization")
        != "fresh_seeded_initialization"
        or checkpoint.get("seed") != 42
        or not isinstance(checkpoint.get("normalizer"), Mapping)
        or not isinstance(checkpoint.get("model_state"), Mapping)
    ):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: checkpoint "
            "XY+time/manifest/from-scratch binding failed"
        )
    try:
        architecture = build_deep_detector(
            cell.detector, cell.modality, cell.action
        )
        architecture.load_state_dict(
            checkpoint["model_state"], strict=True
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: checkpoint parameters do not "
            "strictly match the registered 9/9/15-channel architecture"
        ) from exc
    config = _read_object(artifact_paths["frozen_config"])
    threshold = _read_object(artifact_paths["thresholds"])
    freeze = _read_object(artifact_paths["pre_test_freeze_receipt"])
    identity = {
        "action": cell.action,
        "modality": cell.modality,
        "detector": cell.detector,
    }
    if (
        config.get("schema_version") != CELL_SCHEMA
        or config.get("formal_result") is not True
        or _artifact_protocol(config)
        != detector_protocol
        or config.get("scope") != cell.scope
        or any(config.get(key) != value for key, value in identity.items())
        or config.get("family") != cell.family
        or config.get("resource") != cell.resource
        or config.get("coordinate_schema") != COORDINATE_SCHEMA
        or config.get("channel_count") != len(expected_channels)
        or config.get("channel_schema") != expected_channels
        or config.get("formal_modalities") != list(MODALITIES)
        or config.get("all_formal_modalities_include_xy_and_time") is not False
        or config.get("deep_learning_only") is not True
        or config.get("plain_imu_only_allowed") is not True
        or config.get("training_from_scratch") is not True
        or config.get("preexisting_checkpoint_loaded") is not False
        or config.get("preexisting_result_loaded") is not False
        or config.get("historical_model_artifacts_read") != []
        or config.get("model_initialization")
        != "fresh_seeded_initialization"
        or config.get("manifest") != str(manifest)
        or config.get("manifest_sha256") != sha256_file(manifest)
        or config.get("model_file") != "checkpoint.pt"
        or config.get("model_sha256") != model_sha256
        or config.get("epochs") != FORMAL_EPOCHS
        or config.get("bootstrap_replicates")
        != FORMAL_BOOTSTRAP_REPLICATES
        or config.get("threshold_split") != "development"
        or config.get("test_opened_before_freeze") is not False
        or config.get("environment") != environment
        or config.get("development_qualification") != qualification
        or threshold.get("schema_version") != CELL_SCHEMA
        or _artifact_protocol(threshold)
        != detector_protocol
        or threshold.get("selection_split") != "development"
        or threshold.get("manifest_sha256") != sha256_file(manifest)
        or threshold.get("model_sha256") != model_sha256
        or threshold.get("frozen_config_sha256")
        != hashes["frozen_config"]
        or freeze.get("schema_version") != CELL_SCHEMA
        or _artifact_protocol(freeze)
        != detector_protocol
        or freeze.get("frozen_before_test_open") is not True
        or freeze.get("test_signal_arrays_loaded") != 0
        or freeze.get("manifest_test_rows_ignored_before_freeze") != 1
        or freeze.get("test_source_paths_resolved_before_freeze") != 0
        or freeze.get(
            "test_signal_file_stat_or_hash_calls_before_freeze"
        ) != 0
        or freeze.get("test_signal_arrays_loaded_before_freeze") != 0
        or any(freeze.get(key) != value for key, value in identity.items())
        or freeze.get("coordinate_schema") != COORDINATE_SCHEMA
        or freeze.get("manifest_sha256") != sha256_file(manifest)
        or freeze.get("model_sha256") != model_sha256
        or freeze.get("frozen_config_sha256")
        != hashes["frozen_config"]
        or freeze.get("thresholds_sha256") != hashes["thresholds"]
        or freeze.get("training_from_scratch") is not True
        or freeze.get("preexisting_result_loaded") is not False
        or freeze.get("development_qualification") != qualification
    ):
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: frozen artifact contract failed"
        )
    score_count = 0
    with artifact_paths["test_scores"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventPadError(
                    f"{cell.scope}/{cell.cell_id}: invalid score row "
                    f"{line_number}"
                ) from exc
            if (
                not isinstance(row, dict)
                or row.get("action") != cell.action
                or row.get("label") not in (0, 1)
                or any(
                    not isinstance(row.get(field), str)
                    or not row[field]
                    for field in (
                        "event_id",
                        "user_id",
                        "session_id",
                        "source_cluster_id",
                    )
                )
                or isinstance(row.get("fake_high_score"), bool)
                or not isinstance(row.get("fake_high_score"), (int, float))
                or not np.isfinite(float(row["fake_high_score"]))
            ):
                raise EventPadError(
                    f"{cell.scope}/{cell.cell_id}: invalid score row "
                    f"{line_number}"
                )
            score_count += 1
    if score_count != coverage["scored"]:
        raise EventPadError(
            f"{cell.scope}/{cell.cell_id}: score denominator mismatch"
        )
    return summary


def _validated_release(
    release_path: Path,
    scope: str,
    detector_protocol: str,
) -> tuple[dict[str, Any], Path, str]:
    release = _read_object(release_path)
    manifest = Path(str(release.get("event_manifest", ""))).resolve()
    if (
        scope not in SCOPES
        or release.get("status") != "pass"
        or release.get("formal_result") is not True
        or release.get("scope") != scope
        or release.get("coordinate_schema") != COORDINATE_SCHEMA
        or release.get("lineage_failures") != 0
        or release.get("preprocessing_failures") != 0
        or not manifest.is_file()
    ):
        raise EventPadError(f"{scope}: Event release did not pass")
    if detector_protocol == DEEP_ONLY_PROTOCOL and (
        release.get("formal_modalities") != list(MODALITIES)
        or release.get("all_formal_modalities_include_xy_and_time") is not False
        or release.get("plain_imu_only_allowed") is not True
    ):
        raise EventPadError(
            f"{scope}: Event release is not the exact three-input "
            "XY+time protocol"
        )
    digest = sha256_file(manifest)
    if release.get("event_manifest_sha256") != digest:
        raise EventPadError(f"{scope}: Event release manifest hash mismatch")
    return release, manifest, digest


def aggregate_scope(
    *,
    scope: str,
    release_path: str | Path,
    results_root: str | Path,
    output_path: str | Path,
    detector_protocol: str = ALL_DETECTOR_PROTOCOL,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    release_path = Path(release_path).resolve()
    _release, manifest, manifest_sha256 = _validated_release(
        release_path, scope, detector_protocol
    )
    root = Path(results_root).resolve()
    cells = []
    expected_cells = len(event_cells(scope, detector_protocol))
    for cell in event_cells(scope, detector_protocol):
        summary_path = (
            root
            / cell.action
            / cell.modality
            / cell.detector
            / "summary.json"
        )
        summary = validate_event_cell_artifacts(
            summary_path,
            cell,
            detector_protocol=detector_protocol,
        )
        if (
            detector_protocol == DEEP_ONLY_PROTOCOL
            and (
                summary.get("manifest") != str(manifest)
                or summary.get("manifest_sha256") != manifest_sha256
            )
        ):
            raise EventPadError(
                f"{scope}/{cell.cell_id}: release/manifest binding failed"
            )
        cells.append(
            {
                "cell_id": cell.cell_id,
                "action": cell.action,
                "modality": cell.modality,
                "detector": cell.detector,
                "family": cell.family,
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "manifest_sha256": manifest_sha256,
            }
        )
    aggregate = {
        "schema_version": SCOPE_SCHEMA,
        "status": "pass",
        "formal_result": True,
        "scope": scope,
        "detector_protocol": detector_protocol,
        "coordinate_schema": COORDINATE_SCHEMA,
        "expected_cells": expected_cells,
        "completed_cells": len(cells),
        "release": str(release_path),
        "release_sha256": sha256_file(release_path),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "cells": cells,
    }
    if len(cells) != expected_cells:
        raise EventPadError(
            f"{scope}: incomplete {expected_cells}-cell Event grid"
        )
    destination = Path(output_path).resolve()
    if destination.exists() and reuse_existing:
        existing = _read_object(destination)
        if existing != aggregate:
            raise EventPadError(
                f"{scope}: existing scope aggregate does not match resumed cells"
            )
    else:
        write_json_new(destination, aggregate)
    return aggregate


def aggregate_dual(
    *,
    release_root: str | Path,
    results_root: str | Path,
    output_path: str | Path,
    detector_protocol: str = ALL_DETECTOR_PROTOCOL,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    release_root = Path(release_root).resolve()
    results_root = Path(results_root).resolve()
    scopes = {}
    per_scope = len(event_cells(SCOPES[0], detector_protocol))
    for scope in SCOPES:
        aggregate_path = (
            results_root / scope / "scope_aggregate.json"
        )
        aggregate = aggregate_scope(
            scope=scope,
            release_path=release_root / scope / "release.json",
            results_root=results_root / scope,
            output_path=aggregate_path,
            detector_protocol=detector_protocol,
            reuse_existing=reuse_existing,
        )
        scopes[scope] = {
            "status": "pass",
            "formal_result": True,
            "expected_cells": per_scope,
            "completed_cells": per_scope,
            "aggregate": str(aggregate_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "release": aggregate["release"],
            "release_sha256": aggregate["release_sha256"],
            "manifest": aggregate["manifest"],
            "manifest_sha256": aggregate["manifest_sha256"],
        }
    completion = {
        "schema_version": DUAL_SCHEMA,
        "status": "pass",
        "formal_result": True,
        "detector_protocol": detector_protocol,
        "coordinate_schema": COORDINATE_SCHEMA,
        "expected_cells": per_scope * len(SCOPES),
        "completed_cells": per_scope * len(SCOPES),
        "scope_order": list(SCOPES),
        "lineage_failures": 0,
        "failure_counts": {
            "preprocessing": 0,
            "scoring": 0,
            "total": 0,
        },
        "scopes": scopes,
    }
    destination = Path(output_path).resolve()
    if destination.exists() and reuse_existing:
        existing = _read_object(destination)
        if existing != completion:
            raise EventPadError(
                "existing dual completion does not match resumed cells"
            )
    else:
        write_json_new(destination, completion)
    return completion


__all__ = [
    "CELL_SCHEMA",
    "ALL_DETECTOR_PROTOCOL",
    "DEEP_ONLY_PROTOCOL",
    "DETECTOR_PROTOCOLS",
    "DEEP_DETECTORS",
    "DETECTOR_SPECS",
    "DUAL_SCHEMA",
    "EventCell",
    "FORMAL_BOOTSTRAP_REPLICATES",
    "FORMAL_EPOCHS",
    "REGISTERED_DETECTORS",
    "SCOPE_SCHEMA",
    "SCOPES",
    "aggregate_dual",
    "aggregate_scope",
    "deep_event_cells",
    "event_cells",
    "protocol_detectors",
    "registered_event_cells",
    "run_formal_event_cell",
    "validate_event_cell_artifacts",
]
