#!/usr/bin/env python3
"""Build the component-ablation table from regenerated detector cells.

ASR is computed from each cell's `test_scores.jsonl` at its own `thresholds.json`
frr5. Outputs are working artifacts under ``.actreal-work`` by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPOSITORY_ROOT / "methods" / "evaluation" / "event_level" / "baselines"
sys.path.insert(0, str(BASELINE_DIR))

from release_registry import reference_scores  # noqa: E402

ABLATION_CELL_ROOT = Path(os.environ.get(
    "ACTREAL_ABLATION_CELL_ROOT",
    REPOSITORY_ROOT / ".actreal-work" / "ablation_cells",
))
if not ABLATION_CELL_ROOT.is_absolute():
    ABLATION_CELL_ROOT = REPOSITORY_ROOT / ABLATION_CELL_ROOT


def cell_asr(cell_dir: Path) -> dict | None:
    """ASR and realised FRR at this cell's own frr5 cut."""
    thresholds = cell_dir / "thresholds.json"
    if not thresholds.is_file():
        return None
    threshold = float(json.loads(thresholds.read_text(encoding="utf-8"))["frr5"])
    plain, gz = cell_dir / "test_scores.jsonl", cell_dir / "test_scores.jsonl.gz"
    if plain.is_file():
        handle, opener = plain, open
    elif gz.is_file():
        handle, opener = gz, gzip.open
    else:
        return None
    fake: list[float] = []
    genuine: list[float] = []
    with opener(handle, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            score = row.get("fake_high_score", row.get("score"))
            (fake if int(row["label"]) == 1 else genuine).append(float(score))
    if not fake:
        return None
    fake_a = np.asarray(fake)
    genuine_a = np.asarray(genuine)
    return {
        "threshold_frr5": threshold,
        "asr_at_frr5": float((fake_a < threshold).mean()),
        "frr_realised": float((genuine_a >= threshold).mean()) if len(genuine_a) else None,
        "n_fake": len(fake_a),
        "n_genuine": len(genuine_a),
    }


def arm_cells(arm: dict, actions: list[str], detectors: list[str],
              cell_root: Path) -> dict:
    """Locate this arm's 24 cells, in its own bundle and nowhere else."""
    found = {}
    root = cell_root / arm["id"]
    for action in actions:
        for detector in detectors:
            name = f"{action}__imu_only__{detector}"
            matches = sorted(root.rglob(f"{name}/thresholds.json"))
            if len(matches) > 1:
                raise RuntimeError(f"multiple final cells for {arm['id']}/{name}: {matches}")
            if not matches:
                continue
            cell = matches[0].parent
            result = cell_asr(cell)
            if result is not None:
                result["bundle"] = arm["id"]
                result["path"] = str(cell)
                found[(action, detector)] = result
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / ".actreal-work" / "ablation" / "components",
    )
    parser.add_argument(
        "--cell-root", type=Path, default=ABLATION_CELL_ROOT,
        help=("ablation score-cell root; default: ACTREAL_ABLATION_CELL_ROOT "
              "or .actreal-work/ablation_cells"),
    )
    parser.add_argument(
        "--reference-score-root", type=Path, default=None,
        help=("regenerated A1 event-score cells; default: ACTREAL_EVENT_SCORE_ROOT "
              "or .actreal-work/event_level/actreal/cells"),
    )
    args = parser.parse_args()

    spec = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    actions: list[str] = spec["actions"]
    detectors: list[str] = spec["detectors"]

    reference = {
        (action, detector): {"asr_at_frr5": result["far"]}
        for (action, modality, detector), result
        in reference_scores(args.reference_score_root).items()
        if action in actions and modality == "imu_only" and detector in detectors
    }
    expected_reference = {
        (action, detector) for action in actions for detector in detectors
    }
    if set(reference) != expected_reference:
        raise RuntimeError("regenerated A1 reference does not contain the expected 24 cells")
    ref_24 = float(np.mean([v["asr_at_frr5"] for v in reference.values()]))

    rows = []
    cell_rows = []
    for arm in spec["arms"]:
        found = arm_cells(arm, actions, detectors, args.cell_root)
        for (action, detector), result in sorted(found.items()):
            cell_rows.append({
                "arm": arm["id"], "paper_row": arm["paper_row"],
                "action": action, "modality": "imu_only", "detector": detector,
                "is_null_action": action not in arm["ablated_actions"],
                **{k: result[k] for k in ("threshold_frr5", "asr_at_frr5",
                                          "frr_realised", "n_fake", "n_genuine",
                                          "bundle")},
            })
        complete = len(found) == len(actions) * len(detectors)
        ablated = [v["asr_at_frr5"] for (a, _), v in found.items()
                   if a in arm["ablated_actions"]]
        ref_same = float(np.mean([reference[k]["asr_at_frr5"] for k in found
                                  if k in reference])) if found else None
        rows.append({
            "arm": arm["id"],
            "paper_row": arm["paper_row"],
            "removes": arm["removes"],
            "cells_found": len(found),
            "cells_expected": len(actions) * len(detectors),
            "complete_24": complete,
            "actions_present": sorted({a for a, _ in found}),
            "bundles": sorted({v["bundle"] for v in found.values()}),
            "asr_24": float(np.mean([v["asr_at_frr5"] for v in found.values()])) if complete else None,
            "asr_on_cells_found": float(np.mean([v["asr_at_frr5"] for v in found.values()])) if found else None,
            "reference_on_same_cells": ref_same,
            "delta_asr": (float(np.mean([v["asr_at_frr5"] for v in found.values()])) - ref_same)
                    if found and ref_same is not None else None,
            "asr_ablated_actions_only": float(np.mean(ablated)) if ablated else None,
            "tap_is_null": "tap" not in arm["ablated_actions"],
            "tap_asr": float(np.mean([v["asr_at_frr5"] for (a, _), v in found.items()
                                      if a == "tap"])) if any(a == "tap" for a, _ in found) else None,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate_results.json").write_text(json.dumps({
        "schema": "actreal_objective_component_ablation_v1",
        "modality": "imu_only",
        "actions": actions,
        "detectors": detectors,
        "cells_per_arm": len(actions) * len(detectors),
        "operating_point": spec["operating_point"],
        "asr_source": ("computed from each cell's test_scores.jsonl at its own "
                        "thresholds.json frr5"),
        "reference_A1_24_cells_recomputed": ref_24,
        "rows": rows,
    }, indent=1), encoding="utf-8")

    if cell_rows:
        with (args.out / "cell_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0]))
            writer.writeheader()
            writer.writerows(cell_rows)

    print(f"A1 reference, 24 cells: recomputed {ref_24:.6f}")
    print(f"\n{'arm':13s} {'paper':6s} {'cells':>6s} {'ASR':>8s} {'dASR':>8s} "
          f"{'abl-only':>9s} {'tap':>7s}  actions")
    for row in rows:
        asr = row["asr_on_cells_found"]
        print(f"{row['arm']:13s} {row['paper_row'][:6]:6s} "
              f"{row['cells_found']:3d}/{row['cells_expected']:<2d} "
              f"{'--' if asr is None else format(asr, '8.4f')} "
              f"{'--' if row['delta_asr'] is None else format(row['delta_asr'], '+8.4f')} "
              f"{'--' if row['asr_ablated_actions_only'] is None else format(row['asr_ablated_actions_only'], '9.4f')} "
              f"{'--' if row['tap_asr'] is None else format(row['tap_asr'], '7.4f')}"
              f"  {','.join(row['actions_present']) or '(none yet)'}"
              f"{'' if row['complete_24'] else '   PARTIAL'}")
    print("\nno within-arm interval: one training run per arm, and sampling is "
          "deterministic given a checkpoint")
    print(f"\n-> {args.out}/aggregate_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
