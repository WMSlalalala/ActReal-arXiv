"""Read generated events out of the frozen release.

The release ships 97,297 fake events as ragged per-user shards.  Each event
carries a trajectory and an IMU window on the same 100 Hz grid and with the
same row count, already paired -- so one event is a complete, internally
consistent action, and the first end-to-end delivery does not have to wait for
the online generator.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from . import config
from .imu_source import IMUWindow
from .touch_track import TouchTrack, from_observation

FAKE = 1
GENUINE = 0


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    action: str
    label: int
    user_id: str
    session_id: str
    trajectory: np.ndarray
    imu: np.ndarray
    shard: Path
    index: int

    @property
    def frames(self) -> int:
        return int(len(self.trajectory))

    @property
    def span_ms(self) -> float:
        """What this event's own clock spans, read off its elapsed column.

        Not every event is on a 10 ms step: a long keystroke is spread across a
        capped window, so the span has to come from the event rather than from
        its row count.
        """

        return float(self.trajectory[-1, 7]) * 1000.0

    def touch_track(self, orientation_id: int = 0) -> TouchTrack:
        return from_observation(
            self.trajectory,
            action=self.action,
            orientation_id=orientation_id,
            source=f"released:{self.event_id}",
        )

    @property
    def period_ms(self) -> float:
        """This event's own grid step, which is not always the nominal 10 ms."""

        if self.frames < 2:
            return config.GRID_PERIOD_MS
        return self.span_ms / (self.frames - 1)

    def imu_window(self, orientation_id: int = 0) -> IMUWindow:
        contact = self.trajectory[:, 0] > 0.5
        hits = np.flatnonzero(contact)
        start, end = (int(hits[0]), int(hits[-1]) + 1) if hits.size else (0, self.frames)
        period = self.period_ms
        return IMUWindow(
            action=self.action,
            samples=np.asarray(self.imu, dtype=np.float32),
            active_start=start,
            active_end=end,
            orientation_id=orientation_id,
            requested_duration_ms=(end - start) * period,
            period_ms=period,
            source=f"released:{self.event_id}",
            metadata={"label": int(self.label), "user_id": self.user_id},
        )


@functools.lru_cache(maxsize=8)
def _bundle_map() -> dict[str, str]:
    path = config.DATASETS_ROOT / "ACTION_BUNDLE_MAP.json"
    data = json.loads(path.read_text())
    return {a: v["dataset_bundle"] for a, v in data["actions"].items()}


def bundle_for(action: str) -> str:
    mapping = _bundle_map()
    if action not in mapping:
        raise KeyError(f"unknown action {action!r}; known={sorted(mapping)}")
    return mapping[action]


def shards_for(action: str) -> list[Path]:
    root = config.DATASETS_ROOT / bundle_for(action) / "shards"
    return sorted(root.glob("hmog_u*.npz"))


class EventStore:
    """Lazy reader over one action's shards."""

    def __init__(self, action: str):
        self.action = action
        self.shards = shards_for(action)
        if not self.shards:
            raise FileNotFoundError(f"no shards for action {action!r}")

    def iter_events(
        self,
        *,
        label: Optional[int] = FAKE,
        limit: Optional[int] = None,
        shards: Optional[Sequence[Path]] = None,
    ) -> Iterator[StoredEvent]:
        seen = 0
        for shard in shards or self.shards:
            with np.load(shard, allow_pickle=True) as data:
                actions = data["action"]
                labels = data["label"]
                offsets = data["offsets"]
                keep = actions == self.action
                if label is not None:
                    keep &= labels == label
                if not keep.any():
                    continue
                traj = data["trajectory_flat"]
                imu = data["imu_flat"]
                event_ids = data["event_id"]
                users = data["user_id"]
                sessions = data["session_id"]
                for i in np.flatnonzero(keep):
                    s, e = int(offsets[i]), int(offsets[i + 1])
                    yield StoredEvent(
                        event_id=str(event_ids[i]),
                        action=str(actions[i]),
                        label=int(labels[i]),
                        user_id=str(users[i]),
                        session_id=str(sessions[i]),
                        trajectory=np.asarray(traj[s:e], dtype=np.float32),
                        imu=np.asarray(imu[s:e], dtype=np.float32),
                        shard=shard,
                        index=int(i),
                    )
                    seen += 1
                    if limit is not None and seen >= limit:
                        return

    def first(self, **kwargs) -> StoredEvent:
        for event in self.iter_events(limit=1, **kwargs):
            return event
        raise LookupError(f"no event matched for action {self.action!r}")

    def sample(self, rng: np.random.Generator, *, label: int = FAKE, pool: int = 64) -> StoredEvent:
        """Pick one event without reading every shard."""

        shard = self.shards[int(rng.integers(len(self.shards)))]
        events = list(self.iter_events(label=label, limit=pool, shards=[shard]))
        if not events:
            raise LookupError(f"shard {shard.name} has no {self.action} events")
        return events[int(rng.integers(len(events)))]
