"""Per-user five-shot attack material as a transform source pool."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .audit import sha256_array

SCHEMA = "hmog_fiveshot_attack_material_v3"
SELECTION_DOMAIN = "hmog-fiveshot-material-selection-v1"
SHOTS_PER_GROUP = 5
# Even assignment puts every shot exactly at its cap, so any substitution forced
# by geometry has to exceed it.  A tenth above the even share leaves room for the
# handful that are genuinely needed while still catching a runaway: the bug that
# motivated the cap in the first place drove one shot to 58 uses against 40.
SUBSTITUTION_HEADROOM = 0.1


class FiveShotMaterialError(RuntimeError):
    """Raised when the frozen material cannot be loaded or bound exactly."""


@dataclass(frozen=True)
class MaterialEvent:
    """One frozen real event plus the signals a transform needs."""

    event_id: str
    user_id: str
    action: str
    split: str
    shot_ordinal: int
    source_cluster_id: str
    samples: int
    duration_ms: float
    imu: np.ndarray
    trajectory: np.ndarray
    row: Mapping[str, Any]
    imu_timestamp_ns: np.ndarray | None = None

    @property
    def imu_is_native_slice(self) -> bool:
        """Report whether the IMU came from the native stream, not the shard."""

        return self.imu_timestamp_ns is not None


class FiveShotMaterialPool:
    """Frozen per-(user, action) material with deterministic selection."""

    def __init__(
        self,
        events: Mapping[tuple[str, str], tuple[MaterialEvent, ...]],
        *,
        release: Mapping[str, Any],
        maximum_uses_per_shot: int,
    ) -> None:
        self._events = dict(events)
        self._release = dict(release)
        self._maximum_uses = int(maximum_uses_per_shot)
        if self._maximum_uses < 1:
            raise FiveShotMaterialError("material reuse cap must be positive")
        self._uses: Counter[tuple[str, str, int]] = Counter()
        self._substitutions = 0
        self._substitutions_by_group: Counter[tuple[str, str]] = Counter()

    @property
    def release(self) -> Mapping[str, Any]:
        return dict(self._release)

    @property
    def maximum_uses_per_shot(self) -> int:
        return self._maximum_uses

    @property
    def substitution_ceiling(self) -> int:
        """Return the highest use count a substitution may create."""

        return max(
            self._maximum_uses + 1,
            int(self._maximum_uses * (1.0 + SUBSTITUTION_HEADROOM)),
        )

    def groups(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._events))

    def shots(self, *, user_id: str, action: str) -> tuple[MaterialEvent, ...]:
        try:
            return self._events[(str(user_id), str(action))]
        except KeyError as exc:
            raise FiveShotMaterialError(
                f"no frozen material for {user_id}/{action}"
            ) from exc

    def plan(
        self, *, user_id: str, action: str, event_ids: Iterable[str], seed: int
    ) -> dict[str, MaterialEvent]:
        """Assign every output event of one group to a shot, evenly."""

        shots = self.shots(user_id=user_id, action=action)
        identities = [str(value) for value in event_ids]
        if len(set(identities)) != len(identities):
            raise FiveShotMaterialError(
                f"{user_id}/{action} output event IDs repeat"
            )
        ordered = sorted(
            identities,
            key=lambda identity: hashlib.sha256(
                f"{SELECTION_DOMAIN}|{int(seed)}|{user_id}|{action}|{identity}".encode(
                    "utf-8"
                )
            ).digest(),
        )
        assignment: dict[str, MaterialEvent] = {}
        for position, identity in enumerate(ordered):
            chosen = shots[position % len(shots)]
            assignment[identity] = chosen
            key = (str(user_id), str(action), int(chosen.shot_ordinal))
            self._uses[key] += 1
            if self._uses[key] > self._maximum_uses:
                raise FiveShotMaterialError(
                    f"{user_id}/{action} shot {chosen.shot_ordinal} exceeded "
                    f"its {self._maximum_uses}-use cap"
                )
        return assignment

    def plan_matched(
        self,
        *,
        user_id: str,
        action: str,
        event_scale: Mapping[str, float],
        shot_scale: Mapping[int, float],
        seed: int,
    ) -> dict[str, MaterialEvent]:
        """Assign a group along one ordered scalar instead of by hash."""

        shots = self.shots(user_id=user_id, action=action)
        identities = [str(value) for value in event_scale]
        if len(set(identities)) != len(identities):
            raise FiveShotMaterialError(
                f"{user_id}/{action} output event IDs repeat"
            )
        if not identities:
            return {}
        unscaled = [
            shot.shot_ordinal
            for shot in shots
            if int(shot.shot_ordinal) not in shot_scale
        ]
        if unscaled:
            raise FiveShotMaterialError(
                f"{user_id}/{action} shots {sorted(unscaled)} have no match scale"
            )
        for identity in identities:
            value = float(event_scale[identity])
            if not np.isfinite(value):
                raise FiveShotMaterialError(
                    f"{user_id}/{action}/{identity} has a non-finite match scale"
                )
        ordered_events = sorted(
            identities,
            key=lambda identity: (
                float(event_scale[identity]),
                hashlib.sha256(
                    f"{SELECTION_DOMAIN}|{int(seed)}|{user_id}|{action}|"
                    f"{identity}".encode("utf-8")
                ).digest(),
            ),
        )
        ordered_shots = sorted(
            shots,
            key=lambda shot: (
                float(shot_scale[int(shot.shot_ordinal)]),
                int(shot.shot_ordinal),
            ),
        )
        total = len(ordered_events)
        count = len(ordered_shots)
        assignment: dict[str, MaterialEvent] = {}
        for position, identity in enumerate(ordered_events):
            chosen = ordered_shots[position * count // total]
            assignment[identity] = chosen
            key = (str(user_id), str(action), int(chosen.shot_ordinal))
            self._uses[key] += 1
            if self._uses[key] > self._maximum_uses:
                raise FiveShotMaterialError(
                    f"{user_id}/{action} shot {chosen.shot_ordinal} exceeded "
                    f"its {self._maximum_uses}-use cap"
                )
        return assignment

    def uses(self, *, user_id: str, action: str, shot_ordinal: int) -> int:
        """Report how many outputs a shot has been booked for so far."""

        return int(self._uses[(str(user_id), str(action), int(shot_ordinal))])

    def record_substitution(
        self, *, user_id: str, action: str, previous: int, chosen: int
    ) -> None:
        """Move one recorded use when a request cannot take its assigned shot."""

        previous_key = (str(user_id), str(action), int(previous))
        chosen_key = (str(user_id), str(action), int(chosen))
        if int(previous) == int(chosen):
            raise FiveShotMaterialError(
                f"{user_id}/{action} substitution names one shot twice"
            )
        if self._uses[previous_key] < 1:
            raise FiveShotMaterialError(
                f"{user_id}/{action} shot {previous} has no use to give back"
            )
        group = (str(user_id), str(action))
        allowance = self.substitution_ceiling
        if self._uses[chosen_key] + 1 > allowance:
            raise FiveShotMaterialError(
                f"{user_id}/{action} shot {chosen} would reach "
                f"{self._uses[chosen_key] + 1} uses, past the {allowance} a "
                f"{self._maximum_uses}-use cap allows for substitutions"
            )
        self._uses[previous_key] -= 1
        self._uses[chosen_key] += 1
        self._substitutions += 1
        self._substitutions_by_group[group] += 1

    def usage_audit(self) -> dict[str, Any]:
        """Report reuse so a release can prove the cap was respected."""

        per_action: dict[str, list[int]] = defaultdict(list)
        for (_user, action, _ordinal), count in self._uses.items():
            per_action[action].append(int(count))
        summary = {}
        for action, counts in sorted(per_action.items()):
            array = np.asarray(counts, dtype=np.int64)
            summary[action] = {
                "shots_used": int(len(array)),
                "total_uses": int(array.sum()),
                "minimum_uses": int(array.min()),
                "median_uses": float(np.median(array)),
                "maximum_uses": int(array.max()),
            }
        return {
            "selection_rule": (
                "hash-ordered round-robin over sha256(domain|seed|user|action|"
                "output_event_id) for single-pointer actions; span-sorted block "
                "matching with the same hash as tie-break for pinch; both are "
                "independent of any detector score and of generation outcome"
            ),
            "maximum_uses_per_shot": self._maximum_uses,
            "substitution_ceiling": self.substitution_ceiling,
            "substituted_assignments": int(self._substitutions),
            "substituted_groups": int(len(self._substitutions_by_group)),
            "maximum_substitutions_in_one_group": int(
                max(self._substitutions_by_group.values(), default=0)
            ),
            "by_action": summary,
        }


def _load_shard_signals(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Read every material event's signals, one shard open per shard."""

    from .replay_dataset_builder import InputShard, _load_shard

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["split"]),
                str(row["user_id"]),
                str(row["shard_source"]),
                str(row["shard_source_sha256"]),
            )
        ].append(row)
    signals: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (split, user_id, source, sha256), group in grouped.items():
        shard = _load_shard(
            InputShard(
                split=split,
                user_id=user_id,
                path=Path(source),
                sha256=sha256,
                manifest_row={},
            ),
            signals=True,
        )
        for row in group:
            index = int(row["shard_index"])
            imu, trajectory = shard.event_signal(index)
            # Keystroke rows publish two IMU digests: `imu_sha256` binds the
            # native slice they actually use, and `shard_imu_sha256` binds the
            # shard event they were selected from.  Checking the shard bytes
            # against the native digest would fail every keystroke row.
            shard_imu_digest = str(
                row.get("shard_imu_sha256", row["imu_sha256"])
            )
            if (
                sha256_array(imu) != shard_imu_digest
                or sha256_array(trajectory) != str(row["trajectory_sha256"])
            ):
                raise FiveShotMaterialError(
                    f"{row['event_id']}: material signal digest changed"
                )
            signals[(source, index)] = (
                np.asarray(imu, dtype=np.float32),
                np.asarray(trajectory, dtype=np.float32),
            )
    return signals


def _recovered_touch(
    rows: Iterable[Mapping[str, Any]],
    signals: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    bindings: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Render every donor's touch the way the genuine class is rendered."""

    from .android_touch_observation import TouchObservationError
    from .genuine_touch_recovery import (
        GenuineTouchRecoveryError,
        load_raw_trajectory_archive,
        observe_genuine_binding,
    )

    archives: dict[Path, Mapping[str, np.ndarray]] = {}
    recovered: dict[str, np.ndarray] = {}
    for row in rows:
        cluster = str(row["source_cluster_id"])
        binding = bindings.get(cluster)
        if binding is None:
            raise FiveShotMaterialError(
                f"{row['event_id']}: no genuine binding to render the donor from"
            )
        frozen = signals[(str(row["shard_source"]), int(row["shard_index"]))][1]
        source = Path(binding.raw_trajectory_source)
        if source not in archives:
            archives[source] = load_raw_trajectory_archive(source)
        try:
            observation = observe_genuine_binding(
                binding,
                target_samples=int(len(frozen)),
                target_duration_ms=float(row["duration_ms"]),
                archive=archives[source],
            )
        except (GenuineTouchRecoveryError, TouchObservationError) as exc:
            raise FiveShotMaterialError(
                f"{row['event_id']}: donor touch could not be recovered: {exc}"
            ) from exc
        trajectory = np.asarray(observation.trajectory, dtype=np.float32)
        if trajectory.shape != frozen.shape:
            raise FiveShotMaterialError(
                f"{row['event_id']}: recovered donor changed shape "
                f"{frozen.shape} -> {trajectory.shape}"
            )
        recovered[str(row["event_id"])] = trajectory
    return recovered


NATIVE_IMU_SOURCE_KIND = "native_continuous_100hz_slice"


def _load_native_imu(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the native 100 Hz IMU slice behind every keystroke material row."""

    native: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("imu_source_kind", "")) != NATIVE_IMU_SOURCE_KIND:
            continue
        grouped[
            (str(row["native_stream"]), str(row["native_stream_sha256"]))
        ].append(row)
    for (stream, digest), group in grouped.items():
        path = Path(stream)
        if not path.is_file():
            raise FiveShotMaterialError(f"native IMU stream is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = {"imu", "timestamp_ns"} - set(archive.files)
            if missing:
                raise FiveShotMaterialError(
                    f"{path}: native stream lacks {sorted(missing)}"
                )
            samples = np.asarray(archive["imu"])
            timeline = np.asarray(archive["timestamp_ns"])
        for row in group:
            start = int(row["native_start_sample"])
            end = int(row["native_end_sample"])
            imu = np.ascontiguousarray(samples[start:end], dtype=np.float32)
            stamps = np.ascontiguousarray(timeline[start:end])
            if (
                sha256_array(imu) != str(row["imu_sha256"])
                or sha256_array(stamps) != str(row["imu_timestamp_ns_sha256"])
            ):
                raise FiveShotMaterialError(
                    f"{row['event_id']}: native IMU digest changed"
                )
            if len(imu) != int(row["samples"]):
                raise FiveShotMaterialError(
                    f"{row['event_id']}: native IMU length differs from its row"
                )
            native[str(row["event_id"])] = (imu, stamps)
    return native


def load_fiveshot_material(
    material_dir: str | Path,
    *,
    maximum_uses_per_shot: int,
    actions: Iterable[str] | None = None,
    genuine_touch_bindings: Mapping[str, Any] | None = None,
) -> FiveShotMaterialPool:
    """Load the frozen material and bind every signal digest on the way in."""

    root = Path(material_dir).resolve()
    manifest = root / "material_manifest.jsonl"
    release_path = root / "release.json"
    if not manifest.is_file() or not release_path.is_file():
        raise FiveShotMaterialError(f"five-shot material is missing: {root}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != SCHEMA:
        raise FiveShotMaterialError(
            f"unexpected material schema: {release.get('schema_version')!r}"
        )
    wanted = None if actions is None else {str(value) for value in actions}
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if wanted is not None:
        rows = [row for row in rows if str(row["action"]) in wanted]
    if not rows:
        raise FiveShotMaterialError("five-shot material selected no rows")
    signals = _load_shard_signals(rows)
    native = _load_native_imu(rows)
    recovered = (
        {}
        if genuine_touch_bindings is None
        else _recovered_touch(rows, signals, genuine_touch_bindings)
    )

    grouped: dict[tuple[str, str], list[MaterialEvent]] = defaultdict(list)
    for row in rows:
        imu, trajectory = signals[
            (str(row["shard_source"]), int(row["shard_index"]))
        ]
        trajectory = recovered.get(str(row["event_id"]), trajectory)
        timeline = None
        if str(row["event_id"]) in native:
            imu, timeline = native[str(row["event_id"])]
        grouped[(str(row["user_id"]), str(row["action"]))].append(
            MaterialEvent(
                event_id=str(row["event_id"]),
                user_id=str(row["user_id"]),
                action=str(row["action"]),
                split=str(row["split"]),
                shot_ordinal=int(row["shot_ordinal"]),
                source_cluster_id=str(row["source_cluster_id"]),
                samples=int(row["samples"]),
                duration_ms=float(row["duration_ms"]),
                imu=imu,
                trajectory=trajectory,
                row=dict(row),
                imu_timestamp_ns=timeline,
            )
        )
    events: dict[tuple[str, str], tuple[MaterialEvent, ...]] = {}
    for key, group in grouped.items():
        ordered = tuple(sorted(group, key=lambda event: event.shot_ordinal))
        if len(ordered) != SHOTS_PER_GROUP or [
            event.shot_ordinal for event in ordered
        ] != list(range(SHOTS_PER_GROUP)):
            raise FiveShotMaterialError(
                f"{key[0]}/{key[1]} does not hold exactly five ordered shots"
            )
        events[key] = ordered
    return FiveShotMaterialPool(
        events,
        release=release,
        maximum_uses_per_shot=maximum_uses_per_shot,
    )


__all__ = [
    "FiveShotMaterialError",
    "FiveShotMaterialPool",
    "MaterialEvent",
    "SCHEMA",
    "load_fiveshot_material",
]
