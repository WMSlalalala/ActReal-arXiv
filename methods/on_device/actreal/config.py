"""Repository-relative ActReal paths with ``ACTREAL_*`` overrides."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    override = os.environ.get("ACTREAL_" + name)
    return Path(override) if override else default


# --- roots -----------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]

GENERATION_ROOT = _env_path("GENERATION_ROOT", REPOSITORY_ROOT / "methods" / "generation")
DURATION_POLICY = _env_path(
    "DURATION_POLICY",
    GENERATION_ROOT / "imu" / "configs" / "duration_policy.json",
)

# --- release artefacts -----------------------------------------------------

DATASETS_ROOT = _env_path("DATASETS_ROOT", REPOSITORY_ROOT / "data" / "event_level")

# --- our own outputs -------------------------------------------------------

IMU_CACHE_ROOT = _env_path(
    "IMU_CACHE_ROOT", REPOSITORY_ROOT / ".actreal-work" / "imu_cache"
)
ADB = _env_path("ADB", Path("adb"))

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")

# Padding-window length in 100 Hz frames, per action.  Fixed by the checkpoints;
# see upstream_carrier_imu/README_CN.md.
WINDOW_FRAMES = {"tap": 35, "scroll": 179, "swipe": 167, "pinch": 116, "keystroke": 256}
GRID_HZ = 100.0
GRID_PERIOD_MS = 1000.0 / GRID_HZ
