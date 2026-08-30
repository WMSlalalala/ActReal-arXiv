#!/usr/bin/env python3
"""Assemble baseline tables beside a locally regenerated ActReal score tree."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from covered_modalities import covered  # noqa: E402
from release_registry import (  # noqa: E402
    ACTIONS,
    DETECTORS,
    MODALITIES,
    action_to_bundle,
    reference_scores,
)

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(os.environ.get(
    "ACTREAL_BASELINE_WORK_ROOT", REPO / ".actreal-work" / "baselines"
))


def _load_scores(cell: Path):
    """The cell's test scores, split by label. None if the cell is incomplete."""
    thresholds_path = cell / "thresholds.json"
    scores_path = cell / "test_scores.jsonl"
    if not scores_path.is_file():
        scores_path = cell / "test_scores.jsonl.gz"
    if not thresholds_path.is_file() or not scores_path.is_file():
        return None
    thresholds = json.loads(thresholds_path.read_text())
    if thresholds["score_direction"] != "larger_is_more_fake":
        raise SystemExit(f"unexpected score direction in {cell}")
    opener = gzip.open if scores_path.suffix == ".gz" else open
    fake, genuine = [], []
    with opener(scores_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            (fake if int(row["label"]) == 1 else genuine).append(float(row["fake_high_score"]))
    if not fake or not genuine:
        return None
    return thresholds, fake, genuine


def read_cell(cell: Path, metric: str = "far5") -> tuple | None:
    """One cell's number, at one of two operating points."""
    loaded = _load_scores(cell)
    if loaded is None:
        return None
    thresholds, fake, genuine = loaded

    if metric == "far5":
        cut = float(thresholds["frr5"])
        accepted = sum(1 for s in fake if s < cut)
        rejected = sum(1 for s in genuine if s >= cut)
        return accepted / len(fake), rejected / len(genuine)

    if metric == "test_eer":
        import numpy as _np
        f = _np.asarray(fake); g = _np.asarray(genuine)
        # Sweep every distinct score as a candidate cut and keep the one where
        # the two error rates come closest; EER is their average there.
        best = None
        for cut in _np.unique(_np.concatenate([f, g])):
            far = float((f < cut).mean())
            frr = float((g >= cut).mean())
            gap = abs(far - frr)
            if best is None or gap < best[0]:
                best = (gap, float(cut), far, frr)
        _gap, cut, far, frr = best
        return (far + frr) / 2.0, cut, far, frr

    raise SystemExit(f"unknown metric {metric!r}")


def collect(method: str, metric: str = "far5") -> tuple:
    """{(action, modality, detector) -> number} plus the cells that are missing.

    `metric` picks the operating point -- see read_cell."""

    root = ROOT / method
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{method} has no bundle_manifest.json -- build it first")
    manifest = json.loads(manifest_path.read_text())

    scores, missing, unswapped = {}, [], []
    for action in ACTIONS:
        # Only the modalities this method's own build says it changed.  A
        # trajectory-only method leaves the inertial channel exactly as the
        # release wrote it, so its imu_only cells reproduce the release rather
        # than measuring the method -- and a joint cell for a single-channel
        # method is half release, an attack nobody proposed.
        for modality in covered(method):
            bundle = action_to_bundle(modality)[action]
            entry = manifest["bundles"].get(bundle)
            if entry is None:
                missing.append(f"{action}/{modality}: bundle {bundle} absent")
                continue
            if action not in entry["owned_actions"]:
                missing.append(f"{action}/{modality}: {bundle} does not own it")
                continue
            tally = entry.get("swapped", {}).get(action, {})
            swapped = tally.get("imu_swapped", 0) + tally.get("trajectory_swapped", 0)
            if tally.get("fake", 0) and not swapped:
                unswapped.append(f"{action}/{modality}: declined in {bundle}")
                continue
            cells_dir = root / f"cells_{bundle}_{modality}" / "cells"
            for detector in DETECTORS:
                cell = cells_dir / f"{action}__{modality}__{detector}"
                result = read_cell(cell, metric)
                if result is None:
                    missing.append(f"{action}__{modality}__{detector}")
                else:
                    scores[(action, modality, detector)] = result[0]
    return scores, missing, unswapped


def render(title: str, scores: dict, reference: dict) -> None:
    print(f"\n### {title}")
    for modality in MODALITIES:
        print(f"\n  [{modality}]")
        print("    " + "action".ljust(11) + "".join(d[:9].ljust(11) for d in DETECTORS))
        for action in ACTIONS:
            row = "    " + action.ljust(11)
            for detector in DETECTORS:
                value = scores.get((action, modality, detector))
                row += (f"{value:.3f}" if value is not None else "  -  ").ljust(11)
            print(row)
    values = list(scores.values())
    if values:
        paired = [
            (v, reference[k]["far"]) for k, v in scores.items() if k in reference
        ]
        print(f"\n  cells {len(values)}/90   median FAR {statistics.median(values):.3f}"
              f"   at or above 0.60: {sum(1 for v in values if v >= 0.6)}")
        if paired:
            wins = sum(1 for mine, theirs in paired if mine >= theirs)
            gap = statistics.median(theirs - mine for mine, theirs in paired)
            print(f"  vs release on the {len(paired)} shared cells: "
                  f"baseline wins {wins}, median gap {gap:+.3f} in the release's favour")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*",
                        help="baseline names under the baseline work root; default is all built")
    parser.add_argument(
        "--reference-score-root", type=Path, default=None,
        help=("regenerated ActReal cell scores; default: ACTREAL_EVENT_SCORE_ROOT "
              "or .actreal-work/event_level/actreal/cells"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    reference = reference_scores(args.reference_score_root)
    # `_excluded/` holds completed diagnostics that are not paper baselines.
    methods = args.methods or sorted(
        path.name
        for path in ROOT.glob("*")
        if path.name != "_excluded" and (path / "bundle_manifest.json").is_file()
    )
    if not methods:
        raise SystemExit(f"nothing built under {ROOT}")

    print("=" * 78)
    print("FAR at the development-selected FRR=5% threshold")
    print("reference = locally regenerated ActReal scores at checkpointed thresholds")
    print("=" * 78)
    render("release (source method)", {k: v["far"] for k, v in reference.items()},
           reference)

    everything = {}
    for method in methods:
        scores, missing, unswapped = collect(method)
        render(method, scores, reference)
        if unswapped:
            print(f"  declined by this baseline: {'; '.join(unswapped)}")
        if missing:
            print(f"  {len(missing)} cell(s) not yet available, e.g. {missing[:3]}")
        everything[method] = {f"{a}__{m}__{d}": v for (a, m, d), v in scores.items()}

    if args.json_out:
        everything["__release__"] = {
            f"{a}__{m}__{d}": v["far"] for (a, m, d), v in reference.items()
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(everything, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
