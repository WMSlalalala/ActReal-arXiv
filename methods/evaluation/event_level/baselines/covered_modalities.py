#!/usr/bin/env python3
"""Which modalities a built attack may legitimately be reported on."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(os.environ.get(
    "ACTREAL_BASELINE_WORK_ROOT", REPO / ".actreal-work" / "baselines"
))


def swapped_channels(method: str) -> dict:
    """{'imu': bool, 'trajectory_xy': bool} as the build recorded it."""

    for release in sorted((ROOT / method).glob("*/release.json")):
        if release.is_file():
            return json.loads(release.read_text()).get("baseline_swapped", {})
    return {}


def covered(method: str) -> list:
    channels = swapped_channels(method)
    imu = bool(channels.get("imu"))
    trajectory = bool(channels.get("trajectory_xy"))
    modalities = []
    if trajectory:
        modalities.append("trajectory_xytime")
    if imu:
        modalities.append("imu_only")
    if imu and trajectory:
        modalities.append("imu_trajectory_xytime")
    return modalities


def main() -> None:
    if len(sys.argv) == 2:
        print(" ".join(covered(sys.argv[1])))
        return
    for path in sorted(ROOT.glob("*")):
        if path.name.startswith("_") or not (path / "bundle_manifest.json").is_file():
            continue
        channels = swapped_channels(path.name)
        print(f"  {path.name:22s} swapped={channels}  -> {covered(path.name)}")


if __name__ == "__main__":
    main()
