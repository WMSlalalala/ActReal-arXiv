#!/usr/bin/env python3
"""Export the genuine windows a generator is fitted on, for scoring.

The same call the generator drivers make, written to disk so the quality
metrics score exactly the distribution the generator saw -- not a differently
resampled version of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))
from common.hmog_baseline_common import ACTIONS, collect_genuine_windows  # noqa: E402

TRAJECTORY_COLUMNS = (1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for action in ACTIONS:
        for kind in ("trajectory", "imu"):
            windows, users = collect_genuine_windows(
                args.dataset_dir, args.split, action, kind
            )
            if kind == "trajectory":
                windows = windows[:, :, TRAJECTORY_COLUMNS]
            path = args.out_dir / f"real_{args.split}_{action}_{kind}.npy"
            np.save(path, windows.astype(np.float32))
            print(f"{path.name}: {windows.shape} from {len(set(users))} users")


if __name__ == "__main__":
    main()
