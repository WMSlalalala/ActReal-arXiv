"""Recompute the released ninety-cell table from the released scores."""

from __future__ import annotations

import collections
import gzip
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CELLS = REPO / "evaluation" / "event_level" / "actreal" / "cells"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = ("trajectory_xytime", "imu_only", "imu_trajectory_xytime")
DETECTORS = (
    "hmog_style_svm", "hmog_style_rf", "paper_svm",
    "paper_xgboost", "behaveformer_stdat", "authconformer",
)


def _scores(cell: Path):
    """Open a cell's per-event test scores, compressed or not."""

    packed = cell / "test_scores.jsonl.gz"
    if packed.is_file():
        return gzip.open(packed, "rt")
    return (cell / "test_scores.jsonl").open()


def far_and_frr(cell: Path) -> tuple[float, float]:
    thresholds = json.loads((cell / "thresholds.json").read_text())
    if thresholds["score_direction"] != "larger_is_more_fake":
        raise SystemExit(f"unexpected score direction in {cell.name}")
    cut = float(thresholds["frr5"])
    fake = accepted = genuine = rejected = 0
    for line in _scores(cell):
        row = json.loads(line)
        score = float(row["fake_high_score"])
        if int(row["label"]) == 1:
            fake += 1
            accepted += score < cut
        else:
            genuine += 1
            rejected += score >= cut
    return accepted / fake, rejected / genuine


def main() -> None:
    table: dict[tuple[str, str, str], tuple[float, float]] = {}
    missing: list[str] = []
    for action in ACTIONS:
        for modality in MODALITIES:
            for detector in DETECTORS:
                name = f"{action}__{modality}__{detector}"
                cell = CELLS / name
                if not (cell / "test_scores.jsonl.gz").is_file() and not (
                    cell / "test_scores.jsonl"
                ).is_file():
                    missing.append(name)
                    continue
                table[(action, modality, detector)] = far_and_frr(cell)

    print("FAR at the development-selected FRR=5% threshold")
    print(f"{len(table)}/90 cells present")
    if missing:
        print(f"MISSING {len(missing)} -- these are holes, not zeroes:")
        for name in missing:
            print(f"   {name}")
    print()

    header = f"{'action':<10s}{'modality':<24s}" + "".join(
        f"{d[:12]:>13s}" for d in DETECTORS
    )
    print(header)
    print("-" * len(header))
    for action in ACTIONS:
        for modality in MODALITIES:
            row = f"{action:<10s}{modality:<24s}"
            for detector in DETECTORS:
                value = table.get((action, modality, detector))
                row += "        ....." if value is None else f"{value[0]:>13.4f}"
            print(row)
        print()

    far = sorted(v[0] for v in table.values())
    if not far:
        return
    n = len(far)
    print(
        f"all {n}: min {far[0]:.4f}  median {statistics.median(far):.4f}  "
        f"mean {statistics.mean(far):.4f}  max {far[-1]:.4f}"
    )
    frr = sorted(v[1] for v in table.values())
    print(f"realised test FRR: median {statistics.median(frr):.4f} (target 0.05)")

    for label, index in (("action", 0), ("modality", 1), ("detector", 2)):
        grouped = collections.defaultdict(list)
        for key, value in table.items():
            grouped[key[index]].append(value[0])
        print(
            f"\n{label:<24s}{'n':>4s}{'min':>9s}{'median':>9s}"
            f"{'mean':>9s}{'max':>9s}"
        )
        for key, values in sorted(
            grouped.items(), key=lambda kv: -statistics.median(kv[1])
        ):
            values = sorted(values)
            print(
                f"{key:<24s}{len(values):>4d}{values[0]:>9.3f}"
                f"{statistics.median(values):>9.3f}{statistics.mean(values):>9.3f}"
                f"{values[-1]:>9.3f}"
            )


if __name__ == "__main__":
    main()
