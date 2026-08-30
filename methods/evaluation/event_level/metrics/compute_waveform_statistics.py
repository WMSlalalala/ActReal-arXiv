#!/usr/bin/env python3
"""Compute the IMU waveform statistics reported for learned baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


EVENT_LEVEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_LEVEL))
from baselines.convergence import lag1  # noqa: E402


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
METHODS = {"diffusion_ts": "diffts", "tts_gan": "ttsgan"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--real-window-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    sources = []
    for action in ACTIONS:
        real_path = args.real_window_dir / f"real_train_{action}_imu.npy"
        real = np.load(real_path, mmap_mode="r")
        real_lag1 = lag1(real)
        real_source = {
            "path": real_path.relative_to(args.baseline_root).as_posix(),
            "sha256": file_sha256(real_path),
        }

        for method, directory in METHODS.items():
            method_root = args.baseline_root / directory
            samples_path = method_root / f"samples_{action}_imu.npy"
            summary_path = method_root / f"summary_{action}_imu.json"
            generated = np.load(samples_path, mmap_mode="r")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ratio = np.asarray(summary["generated_std"], dtype=float) / np.asarray(
                summary["real_std"], dtype=float
            )
            if not np.allclose(real.std(axis=(0, 1)), summary["real_std"]):
                raise SystemExit(f"real std mismatch for {method}/{action}")

            rows.append(
                {
                    "method": method,
                    "action": action,
                    "genuine_lag1": real_lag1,
                    "generated_lag1": lag1(generated),
                    "std_ratio_by_axis": ratio.tolist(),
                    "std_ratio_median": float(np.median(ratio)),
                    "std_ratio_min": float(np.min(ratio)),
                    "std_ratio_max": float(np.max(ratio)),
                }
            )
            sources.extend(
                [
                    {
                        "path": samples_path.relative_to(args.baseline_root).as_posix(),
                        "sha256": file_sha256(samples_path),
                    },
                    {
                        "path": summary_path.relative_to(args.baseline_root).as_posix(),
                        "sha256": file_sha256(summary_path),
                    },
                ]
            )
        sources.append(real_source)

    diffusion = [row for row in rows if row["method"] == "diffusion_ts"]
    tts_gan = [row for row in rows if row["method"] == "tts_gan"]
    result = {
        "schema": "actreal_baseline_imu_waveform_statistics_v1",
        "lag1_window_limit": 256,
        "rows": rows,
        "paper_ranges": {
            "diffusion_ts_generated_lag1": [
                min(row["generated_lag1"] for row in diffusion),
                max(row["generated_lag1"] for row in diffusion),
            ],
            "diffusion_ts_action_median_std_ratio": [
                min(row["std_ratio_median"] for row in diffusion),
                max(row["std_ratio_median"] for row in diffusion),
            ],
            "tts_gan_axis_std_ratio": [
                min(min(row["std_ratio_by_axis"]) for row in tts_gan),
                max(max(row["std_ratio_by_axis"]) for row in tts_gan),
            ],
        },
        "sources": sorted(sources, key=lambda row: row["path"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
