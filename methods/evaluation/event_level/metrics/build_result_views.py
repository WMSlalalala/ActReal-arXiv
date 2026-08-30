#!/usr/bin/env python3
"""Rebuild the machine-readable result views used by the paper."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVALUATION = REPO / "evaluation"
CELLS = EVALUATION / "event_level" / "actreal" / "cells"
SCORES = EVALUATION / "event_level" / "actreal"
RELEASE_CSV = SCORES / "cell_results.csv"
INVENTORY = EVALUATION / "provenance" / "result_inventory.json"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_or_none(values):
    values = list(values)
    return mean(values) if values else None


def cell_records():
    with RELEASE_CSV.open("r", encoding="utf-8", newline="") as handle:
        release = {
            (row["action"], row["modality"], row["detector"]): row
            for row in csv.DictReader(handle)
        }
    records = []
    users = []
    for cell_dir in sorted(path for path in CELLS.iterdir() if path.is_dir()):
        summary = read_json(cell_dir / "summary.json")
        thresholds = read_json(cell_dir / "thresholds.json")
        threshold = float(thresholds["frr5"])
        rows = list(score_rows(cell_dir / "test_scores.jsonl.gz"))

        fake = [row for row in rows if int(row["label"]) == 1]
        genuine = [row for row in rows if int(row["label"]) == 0]
        far = mean_or_none(float(row["fake_high_score"]) < threshold for row in fake)
        frr = mean_or_none(float(row["fake_high_score"]) >= threshold for row in genuine)

        action = summary["action"]
        modality = summary["modality"]
        detector = summary["detector"]
        primary = summary["primary_metrics"]
        release_row = release[(action, modality, detector)]
        release_far = float(release_row["far_at_frr5"])
        if abs(release_far - far) > 1e-12:
            raise SystemExit(
                f"FRR5 FAR disagrees with release index for {cell_dir.name}: "
                f"{far} != {release_far}"
            )
        records.append(
            {
                "cell": cell_dir.name,
                "action": action,
                "modality": modality,
                "detector": detector,
                "family": summary.get("family"),
                "frr5_threshold": threshold,
                "far_at_frr5": far,
                "frr_at_frr5": frr,
                "eer_test": float(release_row["eer_test"]),
                "summary_eer_test": float(primary["descriptive_test_eer"]),
                "roc_auc_test": float(primary["roc_auc"]),
                "summary_far_at_dev_eer": float(primary["far"]),
                "summary_frr_at_dev_eer": float(primary["frr"]),
                "test_fake_events": len(fake),
                "test_genuine_events": len(genuine),
                "train_scores_available": (cell_dir / "train_scores.jsonl.gz").exists(),
                "dev_scores_available": (cell_dir / "dev_scores.jsonl.gz").exists(),
                "test_scores_available": True,
            }
        )

        by_user = defaultdict(lambda: {"fake": [], "genuine": []})
        for row in rows:
            group = "fake" if int(row["label"]) == 1 else "genuine"
            by_user[str(row["user_id"])][group].append(float(row["fake_high_score"]))
        for user_id, values in sorted(by_user.items()):
            if values["fake"]:
                users.append(
                    {
                        "cell": cell_dir.name,
                        "action": action,
                        "modality": modality,
                        "detector": detector,
                        "user_id": user_id,
                        "metric": "FAR",
                        "events": len(values["fake"]),
                        "value": mean(score < threshold for score in values["fake"]),
                    }
                )
            if values["genuine"]:
                users.append(
                    {
                        "cell": cell_dir.name,
                        "action": action,
                        "modality": modality,
                        "detector": detector,
                        "user_id": user_id,
                        "metric": "FRR",
                        "events": len(values["genuine"]),
                        "value": mean(score >= threshold for score in values["genuine"]),
                    }
                )
    return records, users


def aggregate(records):
    dimensions = {
        "overall": (),
        "action": ("action",),
        "modality": ("modality",),
        "detector": ("detector",),
        "family": ("family",),
        "action_x_modality": ("action", "modality"),
        "action_x_detector": ("action", "detector"),
        "modality_x_detector": ("modality", "detector"),
        "action_x_modality_x_detector": ("action", "modality", "detector"),
    }
    output = {}
    for name, keys in dimensions.items():
        groups = defaultdict(list)
        for record in records:
            key = "all" if not keys else " | ".join(str(record[item]) for item in keys)
            groups[key].append(record)
        output[name] = {
            key: {
                "cells": len(rows),
                "far_at_frr5": mean(row["far_at_frr5"] for row in rows),
                "frr_at_frr5": mean(row["frr_at_frr5"] for row in rows),
                "eer_test": mean(row["eer_test"] for row in rows),
                "roc_auc_test": mean(row["roc_auc_test"] for row in rows),
            }
            for key, rows in sorted(groups.items())
        }
    return output


def user_aggregates(users):
    grouped = defaultdict(list)
    for row in users:
        grouped[(row["metric"], row["user_id"])].append(float(row["value"]))
    rows = []
    for (metric, user_id), values in sorted(grouped.items()):
        rows.append(
            {
                "user_id": user_id,
                "metric": metric,
                "cells": len(values),
                "value": mean(values),
            }
        )
    return rows


def stability(user_rows):
    far = {row["user_id"]: row["value"] for row in user_rows if row["metric"] == "FAR"}
    values = list(far.values())
    point = mean(values)
    sample_sd = stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    leave_one_out = {
        user_id: mean(value for other, value in far.items() if other != user_id)
        for user_id in sorted(far)
    }
    return {
        "users": len(values),
        "mean_far": point,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "normal_95ci": [point - 1.96 * standard_error, point + 1.96 * standard_error],
        "leave_one_user_out": leave_one_out,
        "leave_one_user_out_min": min(leave_one_out.values()),
        "leave_one_user_out_max": max(leave_one_out.values()),
        "leave_one_user_out_range": max(leave_one_out.values()) - min(leave_one_out.values()),
        "leave_one_user_out_max_absolute_change": max(
            abs(value - point) for value in leave_one_out.values()
        ),
    }


def bootstrap(records, users, *, replicates=2000, seed=42):
    by_cell_metric_user = {}
    user_sets = defaultdict(set)
    for row in users:
        key = (row["cell"], row["metric"], row["user_id"])
        by_cell_metric_user[key] = (int(row["events"]), float(row["value"]))
        user_sets[(row["cell"], row["metric"])].add(row["user_id"])

    common = {frozenset(value) for value in user_sets.values()}
    if len(common) != 1:
        raise SystemExit(f"cell metrics do not share one user set: {len(common)}")
    user_ids = sorted(next(iter(common)))
    cells = [record["cell"] for record in records]
    modality = {record["cell"]: record["modality"] for record in records}
    record_by_cell = {record["cell"]: record for record in records}

    samples = defaultdict(list)
    per_cell_samples = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        names = [user_ids[index] for index in rng.choice(len(user_ids), len(user_ids), replace=True)]
        far = {}
        frr = {}
        for cell in cells:
            for metric, target in (("FAR", far), ("FRR", frr)):
                selected = [by_cell_metric_user[(cell, metric, name)] for name in names]
                denominator = sum(events for events, _value in selected)
                target[cell] = sum(events * value for events, value in selected) / denominator
            per_cell_samples[(cell, "FAR")].append(far[cell])
            per_cell_samples[(cell, "FRR")].append(frr[cell])
        samples["overall_far"].append(mean(far.values()))
        samples["overall_frr"].append(mean(frr.values()))
        for name in ("trajectory_xytime", "imu_only", "imu_trajectory_xytime"):
            samples[f"{name}_far"].append(mean(value for cell, value in far.items() if modality[cell] == name))
        samples["imu_minus_joint_far"].append(
            samples["imu_only_far"][-1] - samples["imu_trajectory_xytime_far"][-1]
        )

    def band(values):
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    overall_far = mean(record["far_at_frr5"] for record in records)
    overall_frr = mean(record["frr_at_frr5"] for record in records)
    imu = mean(record["far_at_frr5"] for record in records if record["modality"] == "imu_only")
    joint = mean(
        record["far_at_frr5"]
        for record in records
        if record["modality"] == "imu_trajectory_xytime"
    )
    points = {
        "overall_far": overall_far,
        "overall_frr": overall_frr,
        "trajectory_xytime_far": mean(
            record["far_at_frr5"] for record in records if record["modality"] == "trajectory_xytime"
        ),
        "imu_only_far": imu,
        "imu_trajectory_xytime_far": joint,
        "imu_minus_joint_far": imu - joint,
    }
    output = {
        "schema": "actreal_user_clustered_bootstrap_v1",
        "unit": "test_user",
        "users": len(user_ids),
        "replicates": replicates,
        "seed": seed,
        "shared_user_resample_across_cells": True,
        "metrics": {
            key: {"point": point, "ci95": band(samples[key])}
            for key, point in points.items()
        },
        "per_cell": {},
    }
    for cell in sorted(cells):
        output["per_cell"][cell] = {
            "far_at_frr5": {
                "point": record_by_cell[cell]["far_at_frr5"],
                "ci95": band(per_cell_samples[(cell, "FAR")]),
            },
            "frr_at_frr5": {
                "point": record_by_cell[cell]["frr_at_frr5"],
                "ci95": band(per_cell_samples[(cell, "FRR")]),
            },
        }
    return output


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_inventory():
    files = []
    for path in sorted(item for item in EVALUATION.rglob("*") if item.is_file()):
        relative = path.relative_to(REPO).as_posix()
        if path == INVENTORY:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema": "actreal_result_inventory_v2",
        "files": files,
        "counts": {
            "files": len(files),
            "main_cells": len([path for path in CELLS.iterdir() if path.is_dir()]),
            "train_score_files": len(list(CELLS.glob("*/train_scores.jsonl.gz"))),
            "development_score_files": len(list(CELLS.glob("*/dev_scores.jsonl.gz"))),
            "test_score_files": len(list(CELLS.glob("*/test_scores.jsonl.gz"))),
        },
    }


def normalize_release_map():
    path = EVALUATION / "provenance" / "release_cell_map.json"
    value = read_json(path)
    for key in (
        "_README",
        "r1_baseline_do_not_use",
        "far5_mean_r1_baseline",
        "why_it_fools_you",
    ):
        value.pop(key, None)
    relative_root = "evaluation/event_level/actreal/cells"
    value["release_scores_root"] = relative_root
    for cell, record in value["cells"].items():
        record["scores"] = f"{relative_root}/{cell}/test_scores.jsonl.gz"
        record["thresholds"] = f"{relative_root}/{cell}/thresholds.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    normalize_release_map()
    records, users = cell_records()
    if len(records) != 90:
        raise SystemExit(f"expected 90 cells, found {len(records)}")

    cell_fields = [
        "cell",
        "action",
        "modality",
        "detector",
        "family",
        "frr5_threshold",
        "far_at_frr5",
        "frr_at_frr5",
        "eer_test",
        "summary_eer_test",
        "roc_auc_test",
        "summary_far_at_dev_eer",
        "summary_frr_at_dev_eer",
        "test_fake_events",
        "test_genuine_events",
        "train_scores_available",
        "dev_scores_available",
        "test_scores_available",
    ]
    write_csv(SCORES / "cell_results.csv", records, cell_fields)
    write_csv(
        SCORES / "user_results.csv",
        users,
        ["cell", "action", "modality", "detector", "user_id", "metric", "events", "value"],
    )
    aggregate_users = user_aggregates(users)
    write_csv(
        SCORES / "user_aggregate_results.csv",
        aggregate_users,
        ["user_id", "metric", "cells", "value"],
    )
    with (SCORES / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate(records), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (SCORES / "user_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(stability(aggregate_users), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (SCORES / "user_clustered_ci.json").open("w", encoding="utf-8") as handle:
        json.dump(bootstrap(records, users), handle, indent=2, sort_keys=True)
        handle.write("\n")
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", encoding="utf-8") as handle:
        json.dump(build_inventory(), handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
