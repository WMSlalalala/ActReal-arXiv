#!/usr/bin/env python3
"""Build the baseline, reference-count, and session result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[4]
EVALUATION = REPO / "evaluation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_baselines() -> None:
    directory = EVALUATION / "event_level" / "baselines"
    rows = read_csv(directory / "cell_results.csv")
    names = [
        ("diffts_trajectory", "diffusion_ts_touch"),
        ("ghostcursor", "ghost_cursor_touch"),
        ("pyclick", "pyclick_touch"),
        ("diffts_imu", "diffusion_ts_imu"),
        ("imagentime", "imagen_time_imu"),
        ("ttsgan", "tts_gan_imu"),
        ("diffts_both", "diffusion_ts_independent_dual_stream"),
    ]
    output = []
    for source_name, display_name in names:
        selected = [row for row in rows if row["method"] == source_name]
        if not selected:
            raise SystemExit(f"missing baseline rows for {source_name}")
        output.append(
            {
                "method": display_name,
                "source_method": source_name,
                "modality": selected[0]["modality"],
                "actions": len({row["action"] for row in selected}),
                "detectors": len({row["detector"] for row in selected}),
                "cells": len(selected),
                "far_at_frr5": mean(float(row["far_at_frr5"]) for row in selected),
                "eer_test": mean(float(row["eer_test"]) for row in selected),
            }
        )
    write_json(
        directory / "aggregate_results.json",
        {
            "schema": "actreal_event_level_baseline_aggregate_v1",
            "metrics": ["far_at_frr5", "eer_test"],
            "rows": output,
            "source": "evaluation/event_level/baselines/cell_results.csv",
        },
    )


def build_reference_count() -> None:
    directory = EVALUATION / "ablation" / "reference_count"
    ablation = read_csv(directory / "cell_results.csv")
    actreal = read_csv(EVALUATION / "event_level" / "actreal" / "cell_results.csv")
    actions = ("tap", "scroll", "swipe", "pinch")
    full = [
        row
        for row in actreal
        if row["action"] in actions and row["modality"] == "imu_only"
    ]
    sources = {
        "k0": "abl_noshot_adv",
        "k1": "abl_krefs1",
        "k3": "abl_krefs3",
        "k8": "abl_krefs8",
    }
    groups = {"k5": full}
    for arm, source in sources.items():
        groups[arm] = [row for row in ablation if row["method"] == source]
    if any(len(rows) != 24 for rows in groups.values()):
        raise SystemExit({arm: len(rows) for arm, rows in groups.items()})
    reference_far = mean(float(row["far_at_frr5"]) for row in groups["k5"])
    output = []
    for arm in ("k5", "k0", "k1", "k3", "k8"):
        rows = groups[arm]
        far = mean(float(row["far_at_frr5"]) for row in rows)
        output.append(
            {
                "arm": arm,
                "source_method": "actreal_release" if arm == "k5" else sources[arm],
                "far_at_frr5": far,
                "eer_test": mean(float(row["eer_test"]) for row in rows),
                "delta_far": far - reference_far,
            }
        )
    write_json(
        directory / "aggregate_results.json",
        {
            "schema": "actreal_imu_reference_count_ablation_v1",
            "modality": "imu_only",
            "actions": list(actions),
            "detectors": 6,
            "cells_per_arm": 24,
            "reference_arm": "k5",
            "rows": output,
            "sources": {
                "k5": "evaluation/event_level/actreal/cell_results.csv",
                "k0_k1_k3_k8": "evaluation/ablation/reference_count/cell_results.csv",
            },
        },
    )


def build_session() -> None:
    directory = EVALUATION / "session_level"
    source = json.loads((directory / "aggregate_results.json").read_text(encoding="utf-8"))
    final = source["final_numbers"]["at_frr5"]
    names = (
        ("S1_COUNT", "count"),
        ("S2_MEAN", "mean"),
        ("S4_TRIMMED", "trimmed_mean_top_third"),
    )
    rows = []
    for source_name, result_name in names:
        row = final[source_name]
        caught_low, caught_high = row["ci95_userbootstrap"]
        rows.append(
            {
                "aggregation": result_name,
                "genuine_frr": row["realised_session_fa"],
                "session_far": 1.0 - row["caught"],
                "session_far_ci95": [1.0 - caught_high, 1.0 - caught_low],
            }
        )
    write_json(
        directory / "paper_results.json",
        {
            "schema": "actreal_session_level_paper_results_v1",
            "genuine_sessions": source["meta"]["n_sessions"],
            "evaluation_users": source["meta"]["n_users"],
            "attack_session_reconstructions": 20,
            "balanced_user_partitions_per_reconstruction": 200,
            "users_per_partition_side": 10,
            "modality_detector_cells": 18,
            "rows": rows,
            "source": "evaluation/session_level/aggregate_results.json",
            "conversion": "session_far = 1 - caught",
        },
    )


def main() -> None:
    build_baselines()
    build_reference_count()
    build_session()


if __name__ == "__main__":
    main()
