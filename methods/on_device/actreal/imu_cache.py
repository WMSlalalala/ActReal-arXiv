"""Pre-generated inertia, indexed by what an agent can ask for.

Sampling a window takes about a second on a GPU, which is too slow for an
agent's action path.

So the windows are generated ahead of time, over a grid of the things an action
can vary in: how long it lasts, which way it goes, how far it travels.  At run
time the request is quantised onto that grid and a window is drawn from the
cell.  A miss is returned explicitly; the formal serving path does not generate
new samples after the cache is frozen.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from . import config

DIRECTIONS8 = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")

# Actions whose cells are split by direction and travel, not by duration alone.
TRAVELLING_ACTIONS = frozenset({"scroll", "swipe", "pinch"})

# Duration grids, in milliseconds: (low, high, step).  The step is chosen to be
# finer than the spread of the human durations themselves, so quantising costs
# less than the natural variation between two of the same person's gestures.
# The top of each range is the action's own inertia window: a gesture longer
# than its window cannot be generated, and one shorter than the victim's own
# recordings cannot be served without stretching somebody's timing.
DURATION_GRIDS: dict[str, tuple[float, float, float]] = {
    "tap": (40.0, 340.0, 20.0),
    "scroll": (120.0, 1760.0, 40.0),
    "swipe": (120.0, 1640.0, 40.0),
    "pinch": (200.0, 1150.0, 50.0),
    "keystroke": (200.0, 2560.0, 200.0),
}

# Travel bins in source pixels, kept as a record of what a window was generated
# at -- not as part of its identity.  The generator's xy conditioning is coarse,
# and an agent's coordinates reach the physical layer as a duration and a
# direction; splitting the grid by travel as well multiplied it sevenfold for
# nothing and still left the band an agent actually uses (640-1000 px) empty.
DISTANCE_EDGES = (0.0, 60.0, 150.0, 320.0, 640.0, 1000.0, 1400.0, 1920.0)


def direction8(start: Sequence[float], end: Sequence[float]) -> Optional[str]:
    """Which of eight compass sectors a gesture travels in, or None if it does not."""

    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    # Screen y grows downward, so north is negative dy.
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0
    index = int((angle + 22.5) // 45.0) % 8
    return DIRECTIONS8[index]


def duration_bin(action: str, duration_ms: float) -> int:
    low, high, step = DURATION_GRIDS[action]
    clamped = min(max(float(duration_ms), low), high)
    return int(round((clamped - low) / step))


def duration_for_bin(action: str, index: int) -> float:
    low, high, step = DURATION_GRIDS[action]
    return min(low + index * step, high)


def duration_bin_count(action: str) -> int:
    low, high, step = DURATION_GRIDS[action]
    return int(round((high - low) / step)) + 1


def distance_bin(distance_px: float) -> int:
    for index, edge in enumerate(DISTANCE_EDGES[1:]):
        if distance_px < edge:
            return index
    return len(DISTANCE_EDGES) - 2


def distance_for_bin(index: int) -> float:
    lo = DISTANCE_EDGES[index]
    hi = DISTANCE_EDGES[min(index + 1, len(DISTANCE_EDGES) - 1)]
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class CacheKey:
    """What identifies a window: how long it lasts and which way it goes.

    Travel is deliberately absent.  The coordinates an agent names are used to
    work out a duration and a heading, and those are what the generator is
    conditioned on in any useful way; keeping travel in the identity split the
    grid seven ways without making any window more suitable.
    """

    action: str
    orientation_id: int
    duration_bin: int
    direction: Optional[str]

    @classmethod
    def for_request(
        cls,
        action: str,
        *,
        duration_ms: float,
        orientation_id: int = 0,
        start: Optional[Sequence[float]] = None,
        end: Optional[Sequence[float]] = None,
    ) -> "CacheKey":
        direction = None
        if start is not None and end is not None:
            direction = direction8(start, end)
        elif action in TRAVELLING_ACTIONS:
            # A travelling action's cells are split by heading, so a key built
            # without geometry names a cell that was never generated.
            raise ValueError(
                f"{action} needs start and end to build a cache key; "
                "without them the direction is unknown"
            )
        return cls(
            action=action,
            orientation_id=int(orientation_id),
            duration_bin=duration_bin(action, duration_ms),
            direction=direction,
        )

    def as_tuple(self) -> tuple:
        return (
            self.action,
            self.orientation_id,
            self.duration_bin,
            self.direction or "",
        )

    def relative_dir(self) -> Path:
        return Path(
            self.action,
            f"ori{self.orientation_id}",
            f"dur{self.duration_bin:03d}",
            self.direction or "none",
        )

    def neighbours(self) -> Iterator["CacheKey"]:
        """Cells to try when this one is empty, nearest first.

        Duration is relaxed before direction: a gesture 20 ms off is closer to
        the request than one heading 45 degrees elsewhere.
        """

        from dataclasses import replace

        for delta in (1, -1, 2, -2):
            index = self.duration_bin + delta
            if 0 <= index < duration_bin_count(self.action):
                yield replace(self, duration_bin=index)
        if self.direction is not None:
            here = DIRECTIONS8.index(self.direction)
            for delta in (1, -1):
                yield replace(self, direction=DIRECTIONS8[(here + delta) % 8])


@dataclass
class CacheHit:
    path: Path
    key: CacheKey
    exactness: str  # "exact" | "neighbour"
    requested: CacheKey

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "key": asdict(self.key),
            "exactness": self.exactness,
            "requested": asdict(self.requested),
        }


class ImuCache:
    """Read side of the pre-generated grid."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            action TEXT NOT NULL,
            orientation_id INTEGER NOT NULL,
            duration_bin INTEGER NOT NULL,
            direction TEXT NOT NULL,
            -- Recorded, not part of the identity: what travel this window was
            -- generated at, so a later analysis can ask whether it mattered.
            travel_bin INTEGER NOT NULL DEFAULT -1,
            path TEXT NOT NULL UNIQUE,
            frames INTEGER NOT NULL,
            active_start INTEGER NOT NULL,
            active_end INTEGER NOT NULL,
            requested_duration_ms REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS samples_cell
            ON samples (action, orientation_id, duration_bin, direction);
    """

    def __init__(self, root: "str | Path" = None, seed: int = 0):
        self.root = Path(root or config.IMU_CACHE_ROOT)
        self.db_path = self.root / "index.sqlite"
        self._rng = random.Random(seed)
        self.hits = 0
        self.neighbour_hits = 0
        self.misses = 0
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
            # Two builders on one root is a mistake, but a build that dies on
            # a momentary lock after an hour of GPU time is a worse one.
            self._conn.execute("PRAGMA busy_timeout = 30000")
            self._conn.executescript(self.SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(
        self,
        key: CacheKey,
        path: Path,
        *,
        frames: int,
        active_start: int,
        active_end: int,
        requested_duration_ms: float,
        travel_bin: int = -1,
    ) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT OR REPLACE INTO samples "
            "(action, orientation_id, duration_bin, direction, travel_bin, path, "
            " frames, active_start, active_end, requested_duration_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                *key.as_tuple(),
                int(travel_bin),
                str(path.relative_to(self.root)) if path.is_absolute() else str(path),
                int(frames),
                int(active_start),
                int(active_end),
                float(requested_duration_ms),
            ),
        )

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def _paths_in(self, key: CacheKey) -> list[str]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT path FROM samples WHERE action=? AND orientation_id=? AND "
            "duration_bin=? AND direction=?",
            key.as_tuple(),
        ).fetchall()
        return [row[0] for row in rows]

    def lookup(self, key: CacheKey) -> Optional[CacheHit]:
        paths = self._paths_in(key)
        if paths:
            self.hits += 1
            return CacheHit(self.root / self._rng.choice(paths), key, "exact", key)
        for neighbour in key.neighbours():
            paths = self._paths_in(neighbour)
            if paths:
                self.neighbour_hits += 1
                return CacheHit(
                    self.root / self._rng.choice(paths), neighbour, "neighbour", key
                )
        self.misses += 1
        return None

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.neighbour_hits + self.misses
        return {
            "requests": total,
            "exact": self.hits,
            "neighbour": self.neighbour_hits,
            "miss": self.misses,
            "hit_rate": round((self.hits + self.neighbour_hits) / total, 4) if total else None,
            "exact_rate": round(self.hits / total, 4) if total else None,
        }

    def coverage(self) -> dict[str, Any]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT action, COUNT(*), COUNT(DISTINCT "
            "duration_bin || '/' || direction) "
            "FROM samples GROUP BY action"
        ).fetchall()
        return {
            action: {"samples": count, "cells": cells} for action, count, cells in rows
        }

    def write_coverage(self, path: Optional[Path] = None) -> Path:
        target = Path(path or self.root / "COVERAGE.json")
        target.write_text(json.dumps(self.coverage(), indent=2))
        return target
