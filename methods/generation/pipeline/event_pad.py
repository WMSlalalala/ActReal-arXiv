"""Event data contracts required while rebuilding generated datasets."""

from __future__ import annotations

import sys
from pathlib import Path

_METHODS_ROOT = Path(__file__).resolve().parents[2]
if str(_METHODS_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHODS_ROOT))

from shared.event_data import (  # noqa: E402,F401
    ACTIONS,
    COORDINATE_SCHEMA,
    FAKE_LABEL,
    IMU_CHANNELS,
    MANIFEST_SCHEMA,
    RAGGED_SHARD_SCHEMA,
    REAL_LABEL,
    SCHEMA,
    SHARDED_MANIFEST_SCHEMA,
    SPLITS,
    TIME_SCHEMA,
    TRAJECTORY_CHANNELS,
    EventPadError,
    EventPartition,
    EventPartitionIndex,
    _load_manifest,
    _load_ragged_signals,
    load_event_partition,
    load_partition,
    load_partition_index,
    operating_metrics,
    select_development_thresholds,
)

__all__ = [
    "ACTIONS", "COORDINATE_SCHEMA", "FAKE_LABEL", "IMU_CHANNELS",
    "MANIFEST_SCHEMA", "RAGGED_SHARD_SCHEMA", "REAL_LABEL", "SCHEMA",
    "SHARDED_MANIFEST_SCHEMA", "SPLITS", "TIME_SCHEMA",
    "TRAJECTORY_CHANNELS", "EventPadError", "EventPartition",
    "EventPartitionIndex", "_load_manifest", "_load_ragged_signals",
    "load_event_partition", "load_partition", "load_partition_index",
    "operating_metrics", "select_development_thresholds",
]
