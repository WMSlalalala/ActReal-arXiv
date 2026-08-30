from __future__ import annotations

"""Shared event dataset contracts, loaders, and frozen-threshold metrics."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
SPLITS = ("train", "development", "test")
REAL_LABEL = 0
FAKE_LABEL = 1
TRAJECTORY_CHANNELS = 9
IMU_CHANNELS = 6
SCHEMA = "joint_event_pad_partition_v2"
MANIFEST_SCHEMA = "joint_event_pad_manifest_v1"
SHARDED_MANIFEST_SCHEMA = "joint_event_pad_manifest_v2"
RAGGED_SHARD_SCHEMA = "joint_event_pad_ragged_shard_v1"
COORDINATE_SCHEMA = "screen_relative_xy_v1"
TIME_SCHEMA = "elapsed_seconds_since_event_start_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EventPadError(ValueError):
    pass


@dataclass(frozen=True)
class EventPartition:
    split: str
    imu: np.ndarray
    trajectory: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    user_ids: np.ndarray
    session_ids: np.ndarray
    event_ids: np.ndarray
    actions: np.ndarray
    coordinate_schema: str
    time_schema: str
    source: Path
    source_sha256: str
    source_cluster_ids: np.ndarray | None = None


@dataclass(frozen=True)
class EventPartitionIndex:
    """Signal-lazy metadata index for a list of ragged on-disk shards."""

    split: str
    shards: tuple[dict[str, Any], ...]
    shard_starts: tuple[int, ...]
    labels: np.ndarray
    user_ids: np.ndarray
    session_ids: np.ndarray
    event_ids: np.ndarray
    actions: np.ndarray
    source_cluster_ids: np.ndarray
    coordinate_schema: str
    time_schema: str
    source: Path
    source_sha256: str


def _load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest = Path(path).resolve()
    rows: dict[str, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            schema = row.get("schema_version")
            if schema not in {MANIFEST_SCHEMA, SHARDED_MANIFEST_SCHEMA}:
                raise EventPadError(f"{manifest}:{line_number}: wrong schema")
            split = str(row.get("split", ""))
            if split not in SPLITS or split in rows:
                raise EventPadError("manifest requires exactly one row per split")
            users = row.get("user_ids")
            if not isinstance(users, list) or not users or len(users) != len(set(users)):
                raise EventPadError(f"{split}: invalid manifest user_ids")
            if schema == MANIFEST_SCHEMA:
                source = Path(str(row.get("source", ""))).resolve()
                if (
                    not source.is_file()
                    or sha256_file(source) != row.get("source_sha256")
                ):
                    raise EventPadError(
                        f"{split}: source missing or hash mismatch"
                    )
                rows[split] = {**row, "source": source}
            else:
                shards = row.get("shards")
                if not isinstance(shards, list) or not shards:
                    raise EventPadError(f"{split}: no ragged shards")
                normalized_shards: list[dict[str, Any]] = []
                seen_sources: set[Path] = set()
                shard_users: list[str] = []
                event_total = 0
                for shard_number, shard in enumerate(shards):
                    if not isinstance(shard, dict):
                        raise EventPadError(
                            f"{split}: shard {shard_number} is not an object"
                        )
                    source = Path(str(shard.get("source", ""))).resolve()
                    if (
                        source in seen_sources
                        or not source.is_file()
                        or sha256_file(source) != shard.get("source_sha256")
                    ):
                        raise EventPadError(
                            f"{split}: shard source missing, duplicate, or changed"
                        )
                    seen_sources.add(source)
                    shard_user = str(shard.get("user_id", ""))
                    if not shard_user:
                        raise EventPadError(
                            f"{split}: shard lacks a user binding"
                        )
                    shard_users.append(shard_user)
                    shard_events = int(shard.get("events", -1))
                    if shard_events < 1:
                        raise EventPadError(
                            f"{split}: shard has invalid event count"
                        )
                    event_total += shard_events
                    normalized_shards.append({**shard, "source": source})
                if (
                    shard_users != users
                    or len(shard_users) != len(set(shard_users))
                    or event_total != int(row.get("events", -1))
                ):
                    raise EventPadError(
                        f"{split}: shard/user/event manifest mismatch"
                    )
                rows[split] = {**row, "shards": normalized_shards}
    if set(rows) != set(SPLITS):
        raise EventPadError("manifest must contain train/development/test")
    for left, right in (("train", "development"), ("train", "test"), ("development", "test")):
        overlap = set(rows[left]["user_ids"]) & set(rows[right]["user_ids"])
        if overlap:
            raise EventPadError(f"user leakage between {left}/{right}: {sorted(overlap)}")
    return rows


def _manifest_partition_digest(row: Mapping[str, Any]) -> str:
    if row.get("schema_version") == SHARDED_MANIFEST_SCHEMA:
        return hashlib.sha256(
            "".join(
                str(shard["source_sha256"])
                for shard in row["shards"]
            ).encode("ascii")
        ).hexdigest()
    return str(row["source_sha256"])


def load_partition(row: Mapping[str, Any]) -> EventPartition:
    source = Path(row["source"]).resolve()
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "imu", "trajectory", "mask", "label", "user_id", "session_id",
            "event_id", "action", "schema_version", "coordinate_schema",
            "time_schema",
        }
        if required - set(archive.files):
            raise EventPadError(f"{source}: missing arrays {sorted(required-set(archive.files))}")
        if str(np.asarray(archive["schema_version"]).item()) != SCHEMA:
            raise EventPadError(f"{source}: wrong partition schema")
        coordinate_schema = str(
            np.asarray(archive["coordinate_schema"]).item()
        )
        time_schema = str(np.asarray(archive["time_schema"]).item())
        if coordinate_schema != COORDINATE_SCHEMA:
            raise EventPadError(
                f"{source}: coordinate_schema must be "
                f"{COORDINATE_SCHEMA!r}"
            )
        if time_schema != TIME_SCHEMA:
            raise EventPadError(
                f"{source}: time_schema must be {TIME_SCHEMA!r}"
            )
        imu = np.asarray(archive["imu"], dtype=np.float32)
        trajectory = np.asarray(archive["trajectory"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
        labels = np.asarray(archive["label"], dtype=np.int64)
        users = np.asarray(archive["user_id"]).astype(str)
        sessions = np.asarray(archive["session_id"]).astype(str)
        events = np.asarray(archive["event_id"]).astype(str)
        actions = np.asarray(archive["action"]).astype(str)
        source_clusters = (
            np.asarray(archive["source_cluster_id"]).astype(str)
            if "source_cluster_id" in archive.files
            else events.copy()
        )
    n = labels.size
    if (
        imu.ndim != 3 or imu.shape != (n, mask.shape[1], IMU_CHANNELS)
        or trajectory.shape != (n, mask.shape[1], TRAJECTORY_CHANNELS)
        or mask.ndim != 2 or mask.shape[0] != n
        or any(
            value.shape != (n,)
            for value in (
                users,
                sessions,
                events,
                actions,
                source_clusters,
            )
        )
    ):
        raise EventPadError(f"{source}: inconsistent event shapes")
    if n == 0 or mask.shape[1] < 8 or np.any(mask.sum(1) < 2):
        raise EventPadError(f"{source}: empty or too-short event")
    if not np.isfinite(imu).all() or not np.isfinite(trajectory).all():
        raise EventPadError(f"{source}: non-finite signal")
    if not set(np.unique(labels)).issubset({REAL_LABEL, FAKE_LABEL}):
        raise EventPadError(f"{source}: labels must use real=0/fake=1")
    if set(np.unique(labels)) != {REAL_LABEL, FAKE_LABEL}:
        raise EventPadError(f"{source}: both labels are required")
    if not set(np.unique(actions)).issubset(set(ACTIONS)):
        raise EventPadError(f"{source}: invalid action")
    if len(set(events.tolist())) != n:
        raise EventPadError(f"{source}: duplicate event_id")
    if set(users.tolist()) != set(row["user_ids"]):
        raise EventPadError(f"{source}: manifest/partition users mismatch")
    if np.any(imu[~mask] != 0) or np.any(trajectory[~mask] != 0):
        raise EventPadError(f"{source}: padding must be explicit zero")
    valid_xy = trajectory[:, :, 1:3][mask]
    if valid_xy.size and (
        np.any(valid_xy < 0.0) or np.any(valid_xy > 1.0)
    ):
        raise EventPadError(
            f"{source}: valid trajectory x/y must be screen-relative "
            "coordinates in [0,1]"
        )
    return EventPartition(
        split=str(row["split"]), imu=imu, trajectory=trajectory, mask=mask,
        labels=labels, user_ids=users, session_ids=sessions, event_ids=events,
        actions=actions, coordinate_schema=coordinate_schema,
        time_schema=time_schema, source=source,
        source_sha256=sha256_file(source),
        source_cluster_ids=source_clusters,
    )


def load_partition_index(
    row: Mapping[str, Any],
) -> EventPartitionIndex:
    if row.get("schema_version") != SHARDED_MANIFEST_SCHEMA:
        raise EventPadError("partition index requires the sharded manifest")
    labels_parts: list[np.ndarray] = []
    users_parts: list[np.ndarray] = []
    sessions_parts: list[np.ndarray] = []
    events_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    clusters_parts: list[np.ndarray] = []
    normalized_shards: list[dict[str, Any]] = []
    starts: list[int] = []
    total = 0
    seen_events: set[str] = set()
    seen_clusters_by_label: set[tuple[str, int]] = set()
    for shard in row["shards"]:
        source = Path(shard["source"]).resolve()
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "coordinate_schema",
                "time_schema",
                "scope",
                "split",
                "imu_flat",
                "trajectory_flat",
                "offsets",
                "label",
                "user_id",
                "session_id",
                "event_id",
                "action",
                "source_cluster_id",
                "sample_idx",
                "cross_modal_pair_id",
            }
            if required - set(archive.files):
                raise EventPadError(
                    f"{source}: incomplete ragged shard"
                )
            if (
                str(np.asarray(archive["schema_version"]).item())
                != RAGGED_SHARD_SCHEMA
                or str(np.asarray(archive["coordinate_schema"]).item())
                != COORDINATE_SCHEMA
                or str(np.asarray(archive["time_schema"]).item())
                != TIME_SCHEMA
                or str(np.asarray(archive["split"]).item())
                != str(row["split"])
            ):
                raise EventPadError(f"{source}: ragged shard contract mismatch")
            labels = np.asarray(archive["label"], dtype=np.int64)
            users = np.asarray(archive["user_id"]).astype(str)
            sessions = np.asarray(archive["session_id"]).astype(str)
            events = np.asarray(archive["event_id"]).astype(str)
            actions = np.asarray(archive["action"]).astype(str)
            clusters = np.asarray(
                archive["source_cluster_id"]
            ).astype(str)
            sample_indices = np.asarray(
                archive["sample_idx"], dtype=np.int64
            )
            pair_ids = np.asarray(
                archive["cross_modal_pair_id"]
            ).astype(str)
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
        n = len(labels)
        if (
            n != int(shard["events"])
            or any(
                value.shape != (n,)
                for value in (
                    users,
                    sessions,
                    events,
                    actions,
                    clusters,
                    sample_indices,
                    pair_ids,
                )
            )
            or offsets.shape != (n + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 2)
            or not set(labels.tolist()).issubset({REAL_LABEL, FAKE_LABEL})
            or not set(actions.tolist()).issubset(set(ACTIONS))
            or set(users.tolist()) != {str(shard["user_id"])}
            or np.any(sample_indices < 0)
            or any(len(value) != 64 for value in pair_ids.tolist())
        ):
            raise EventPadError(f"{source}: invalid ragged metadata")
        if any(event in seen_events for event in events.tolist()):
            raise EventPadError(f"{source}: duplicate event ID across shards")
        seen_events.update(events.tolist())
        # Genuine source clusters must be unique because no control reuse is
        # permitted. Fake cluster IDs are also unique event IDs.
        cluster_label = set(zip(clusters.tolist(), labels.tolist()))
        if len(cluster_label) != n or seen_clusters_by_label & cluster_label:
            raise EventPadError(
                f"{source}: source cluster reuse/duplication detected"
            )
        seen_clusters_by_label.update(cluster_label)
        starts.append(total)
        total += n
        normalized_shards.append(
            {
                **dict(shard),
                "source": source,
                "offsets": offsets,
                "global_start": starts[-1],
            }
        )
        labels_parts.append(labels)
        users_parts.append(users)
        sessions_parts.append(sessions)
        events_parts.append(events)
        actions_parts.append(actions)
        clusters_parts.append(clusters)
    if total != int(row.get("events", -1)):
        raise EventPadError("ragged manifest/index event count mismatch")
    labels = np.concatenate(labels_parts)
    users = np.concatenate(users_parts)
    sessions = np.concatenate(sessions_parts)
    events = np.concatenate(events_parts)
    actions = np.concatenate(actions_parts)
    clusters = np.concatenate(clusters_parts)
    if set(labels.tolist()) != {REAL_LABEL, FAKE_LABEL}:
        raise EventPadError("ragged partition requires both labels")
    if set(users.tolist()) != set(row["user_ids"]):
        raise EventPadError("ragged manifest/partition users mismatch")
    digest = hashlib.sha256(
        "".join(
            str(shard["source_sha256"]) for shard in row["shards"]
        ).encode("ascii")
    ).hexdigest()
    return EventPartitionIndex(
        split=str(row["split"]),
        shards=tuple(normalized_shards),
        shard_starts=tuple(starts),
        labels=labels,
        user_ids=users,
        session_ids=sessions,
        event_ids=events,
        actions=actions,
        source_cluster_ids=clusters,
        coordinate_schema=COORDINATE_SCHEMA,
        time_schema=TIME_SCHEMA,
        source=Path(row["shards"][0]["source"]).resolve().parent,
        source_sha256=digest,
    )


def load_event_partition(
    row: Mapping[str, Any],
) -> EventPartition | EventPartitionIndex:
    if row.get("schema_version") == SHARDED_MANIFEST_SCHEMA:
        return load_partition_index(row)
    return load_partition(row)


def _load_ragged_signals(
    shard: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = Path(shard["source"]).resolve()
    with np.load(source, allow_pickle=False) as archive:
        imu = np.asarray(archive["imu_flat"], dtype=np.float32)
        trajectory = np.asarray(
            archive["trajectory_flat"], dtype=np.float32
        )
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
    if (
        imu.ndim != 2
        or imu.shape[1] != IMU_CHANNELS
        or trajectory.shape != (len(imu), TRAJECTORY_CHANNELS)
        or offsets.shape != (int(shard["events"]) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(imu)
        or np.any(np.diff(offsets) < 2)
        or not np.isfinite(imu).all()
        or not np.isfinite(trajectory).all()
    ):
        raise EventPadError(f"{source}: invalid ragged signals")
    valid_xy = trajectory[:, 1:3]
    if np.any(valid_xy < 0.0) or np.any(valid_xy > 1.0):
        raise EventPadError(
            f"{source}: ragged trajectory x/y outside [0,1]"
        )
    for left, right in zip(offsets[:-1], offsets[1:]):
        time_values = trajectory[int(left):int(right), 7]
        if (
            abs(float(time_values[0])) > 1.0e-7
            or np.any(np.diff(time_values) < 0.0)
        ):
            raise EventPadError(
                f"{source}: invalid relative event time channel"
            )
    return imu, trajectory, offsets


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    return np.concatenate([unique, [np.nextafter(unique[-1], np.inf)]])


def operating_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    real = labels == REAL_LABEL
    fake = labels == FAKE_LABEL
    return {
        "threshold": float(threshold),
        "far_fake_accepted": float(np.mean(scores[fake] < threshold)),
        "frr_real_rejected": float(np.mean(scores[real] >= threshold)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
    }


def select_development_thresholds(
    labels: np.ndarray, scores: np.ndarray, target_frr: float = 0.05
) -> dict[str, float]:
    thresholds = _threshold_candidates(scores)
    real = np.sort(scores[labels == REAL_LABEL])
    fake = np.sort(scores[labels == FAKE_LABEL])
    if not len(real) or not len(fake):
        raise EventPadError(
            "development threshold selection requires both labels"
        )
    frr = (
        len(real) - np.searchsorted(real, thresholds, side="left")
    ) / len(real)
    far = np.searchsorted(
        fake, thresholds, side="left"
    ) / len(fake)
    eer_order = np.lexsort((thresholds, np.maximum(frr, far), np.abs(frr-far)))
    eligible = np.flatnonzero(frr <= target_frr + 1e-15)
    if not len(eligible):
        raise EventPadError("no development threshold satisfies FRR budget")
    budget_order = np.lexsort(
        (thresholds[eligible], -frr[eligible], far[eligible])
    )
    return {
        "eer": float(thresholds[int(eer_order[0])]),
        "frr5": float(thresholds[int(eligible[int(budget_order[0])])]),
        "target_frr": float(target_frr),
    }

