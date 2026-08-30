#!/usr/bin/env python3
"""Score FAR for the 18-run real-device matrix with six frozen detectors.

This is the release entry point for the paper's on-device FAR result.  It loads
all six frozen detectors without fitting a model, reselecting a threshold, or
using phone calibration.  The released output contains the primary 18-run
aggregate, its compact FAR table, a score-blind cohort audit, and a
reproducibility receipt.
System-IME keystroke observations come from the audited executor receipts
because an ordinary Activity cannot observe those raw MotionEvents.

The detector thresholds remain the formal HMOG development-split FRR5
operating points recorded in the 90-cell release registry.  FAR is the fraction
of ActReal event--detector decisions accepted as human at those frozen cuts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
import torch


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = ("trajectory_xytime", "imu_only", "imu_trajectory_xytime")
DETECTORS = (
    "hmog_style_svm",
    "hmog_style_rf",
    "paper_svm",
    "paper_xgboost",
    "behaveformer_stdat",
    "authconformer",
)
CLASSICAL = DETECTORS[:4]
PERIOD_MS = 10.0
NS_PER_MS = 1_000_000.0
EXPECTED_RAW_GESTURE_EVENTS = {"tap": 1800, "scroll": 102, "swipe": 9, "pinch": 70}
EXPECTED_COMPLETE_RAW_GESTURE_EVENTS = {
    "tap": 1236,
    "scroll": 70,
    "swipe": 2,
    "pinch": 60,
}
EXPECTED_PARTITION_GESTURE_EVENTS = {
    "tap": 1012,
    "scroll": 68,
    "swipe": 2,
    "pinch": 60,
}

MOTION_CODES = {
    "HISTORY": -1,
    "ACTION_DOWN": 0,
    "ACTION_UP": 1,
    "ACTION_MOVE": 2,
    "ACTION_CANCEL": 3,
    "ACTION_OUTSIDE": 4,
    "ACTION_POINTER_DOWN": 5,
    "ACTION_POINTER_UP": 6,
    "ACTION_HOVER_MOVE": 7,
    "ACTION_SCROLL": 8,
    "ACTION_HOVER_ENTER": 9,
    "ACTION_HOVER_EXIT": 10,
    "ACTION_BUTTON_PRESS": 11,
    "ACTION_BUTTON_RELEASE": 12,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def framework_task(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_injected_" not in stem:
        raise RuntimeError(f"not an injected report: {path}")
    return tuple(stem.split("_injected_", 1))  # type: ignore[return-value]


def paired_imu(session: Path) -> tuple[np.ndarray, np.ndarray]:
    per: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in read_csv(session / "imu.csv"):
        sensor = row["sensor"]
        if sensor not in {"accelerometer", "gyroscope"}:
            continue
        per[sensor].append(
            (
                float(row["event_elapsed_ns"]) / NS_PER_MS,
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            )
        )
    if not per["accelerometer"] or not per["gyroscope"]:
        raise RuntimeError(f"{session}: missing accelerometer or gyroscope")
    acc = np.asarray(sorted(per["accelerometer"]), dtype=np.float64)
    gyr = np.asarray(sorted(per["gyroscope"]), dtype=np.float64)
    t_ms = acc[:, 0]
    right = np.clip(np.searchsorted(gyr[:, 0], t_ms), 0, len(gyr) - 1)
    left = np.clip(right - 1, 0, len(gyr) - 1)
    use_left = np.abs(gyr[left, 0] - t_ms) <= np.abs(gyr[right, 0] - t_ms)
    index = np.where(use_left, left, right)
    return t_ms, np.concatenate([acc[:, 1:4], gyr[index, 1:4]], axis=1)


def interpolate(t_ms: np.ndarray, values: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    if grid_ms[0] < t_ms[0] or grid_ms[-1] > t_ms[-1]:
        raise RuntimeError("event grid falls outside the captured IMU stream")
    result = np.empty((len(grid_ms), values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        result[:, channel] = np.interp(grid_ms, t_ms, values[:, channel])
    return result


def motion_code(value: str) -> int:
    upper = value.upper()
    for name, code in MOTION_CODES.items():
        if upper.startswith(name):
            return code
    raise ValueError(f"unknown Android motion action {value!r}")


def on_grid(
    detector_grid_clock: Any,
    t_src_ms: np.ndarray,
    values: np.ndarray,
    target: int,
) -> np.ndarray:
    """Match the released phone scorer's native-IMU-to-detector-grid operator."""

    if target < 2 or len(t_src_ms) < 2:
        return np.repeat(values[:1], max(target, 1), axis=0).astype(np.float32)
    span_ms = float(t_src_ms[-1] - t_src_ms[0])
    grid_ms = (
        t_src_ms[0]
        + detector_grid_clock(target, span_ms if span_ms > 0.0 else 1.0) * 1000.0
    )
    output = np.empty((target, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(grid_ms, t_src_ms, values[:, channel])
    return output


def gesture_signal(
    event: dict[str, str],
    touch_rows: list[dict[str, str]],
    imu_t_ms: np.ndarray,
    imu_values: np.ndarray,
    *,
    touch_observation: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the exact release observer input from target-Activity rows."""

    start_ms = float(event["start_elapsed_ns"]) / NS_PER_MS
    end_ms = float(event["end_elapsed_ns"]) / NS_PER_MS
    duration_ms = end_ms - start_ms
    target = max(2, int(round(duration_ms / PERIOD_MS)) + 1)

    imu_span = (imu_t_ms >= start_ms) & (imu_t_ms <= end_ms)
    if int(imu_span.sum()) < 1:
        raise RuntimeError(f"{event['event_id']}: no captured IMU row in event span")
    imu = on_grid(
        touch_observation.detector_grid_clock,
        imu_t_ms[imu_span] - start_ms,
        imu_values[imu_span],
        target,
    )

    rows = sorted(
        touch_rows,
        key=lambda row: (
            int(row["event_elapsed_ns"]),
            int(row.get("pointer_index", 0)),
            int(row.get("pointer_id", 0)),
        ),
    )
    if not rows:
        raise RuntimeError(f"{event['event_id']}: no application-visible touch rows")
    orientation_id = int(float(event["orientation_id"]))
    hmog_width, hmog_height = touch_observation.screen_dimensions_for_orientation(
        orientation_id
    )
    timestamps = np.asarray(
        [int(row["event_elapsed_ns"]) for row in rows], dtype=np.int64
    )
    _unique, frame_index = np.unique(timestamps, return_inverse=True)
    raw_x_px = np.asarray(
        [
        (float(row["raw_x_px"]) / float(row["display_width_px"])) * hmog_width
        for row in rows
        ],
        dtype=np.float64,
    )
    raw_y_px = np.asarray(
        [
        (float(row["raw_y_px"]) / float(row["display_height_px"])) * hmog_height
        for row in rows
        ],
        dtype=np.float64,
    )
    x_outside = (raw_x_px < 0.0) | (raw_x_px > hmog_width)
    y_outside = (raw_y_px < 0.0) | (raw_y_px > hmog_height)
    clipped_rows = x_outside | y_outside
    x_px = np.clip(raw_x_px, 0.0, hmog_width)
    y_px = np.clip(raw_y_px, 0.0, hmog_height)
    coordinate_audit = {
        "policy": "legacy phone boundary clipping before canonical ZOH observer",
        "rows": len(rows),
        "clipped_rows": int(clipped_rows.sum()),
        "clipped_coordinate_values": int(x_outside.sum() + y_outside.sum()),
        "x_clipped_values": int(x_outside.sum()),
        "y_clipped_values": int(y_outside.sum()),
        "x_hmog_min_before": float(raw_x_px.min()),
        "x_hmog_max_before": float(raw_x_px.max()),
        "y_hmog_min_before": float(raw_y_px.min()),
        "y_hmog_max_before": float(raw_y_px.max()),
        "hmog_width": float(hmog_width),
        "hmog_height": float(hmog_height),
    }
    try:
        observation = touch_observation.observe_android_rows(
            action=str(event["action"]),
            target_samples=target,
            orientation_id=orientation_id,
            t_ms=(timestamps.astype(np.float64) / NS_PER_MS) - start_ms,
            x_px=x_px,
            y_px=y_px,
            pressure=[float(row["pressure"]) for row in rows],
            pointer_id=[int(row["pointer_id"]) for row in rows],
            android_action=[motion_code(row["motion_action"]) for row in rows],
            frame_index=frame_index,
            source_duration_ms=duration_ms,
            target_duration_ms=duration_ms,
        )
    except Exception as error:
        raise RuntimeError(
            f"{event['event_id']} {event['action']}: canonical touch observer failed; "
            f"x=[{raw_x_px.min()}, {raw_x_px.max()}]/{hmog_width}, "
            f"y=[{raw_y_px.min()}, {raw_y_px.max()}]/{hmog_height}: {error}"
        ) from error
    return (
        imu.astype(np.float32),
        observation.trajectory.astype(np.float32),
        coordinate_audit,
    )


def align_served_report_records(
    report: dict[str, Any], events: list[dict[str, str]], tolerance_ms: float = 100.0
) -> tuple[list[tuple[dict[str, Any], dict[str, str], float]], list[dict[str, Any]]]:
    """Bind each served physical action to the Activity's resulting event."""

    gestures = [event for event in events if event["action"] != "keystroke"]
    unused = {event["event_id"] for event in gestures}
    aligned: list[tuple[dict[str, Any], dict[str, str], float]] = []
    exclusions: list[dict[str, Any]] = []
    for record in report["records"]:
        receipt = record.get("receipt")
        bundle = (record.get("plan") or {}).get("bundle") or {}
        if (
            not bool(record.get("served"))
            or not isinstance(receipt, dict)
            or receipt.get("action") == "keystroke"
        ):
            continue
        expected_ms = (
            float(receipt["imu_start_elapsed_ns"]) / NS_PER_MS
            + float(bundle.get("pre_roll_ms") or 0.0)
        )
        candidates = sorted(
            (
                abs(float(event["start_elapsed_ns"]) / NS_PER_MS - expected_ms),
                event,
            )
            for event in gestures
            if event["event_id"] in unused
        )
        if candidates and candidates[0][0] <= tolerance_ms:
            delta_ms = float(candidates[0][0])
            event = candidates[0][1]
            unused.remove(event["event_id"])
            aligned.append((record, event, delta_ms))
        else:
            exclusions.append(
                {
                    "record_index": int(record["index"]),
                    "api": str(record["api"]),
                    "receipt_action": str(receipt["action"]),
                    "reason": "no_matching_target_application_event",
                    "nearest_start_delta_ms": (
                        float(candidates[0][0]) if candidates else None
                    ),
                }
            )
    return aligned, exclusions


def release_exclusive_partition(
    events: list[dict[str, str]],
    touch_by_id: dict[str, list[dict[str, str]]],
    task_events: list[dict[str, str]],
    partition_module: Any,
    *,
    eligible_run_ids: set[str] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Apply the canonical score-blind release partition to one matrix run."""

    candidate_events = [
        event
        for event in events
        if event.get("task") != "fewshot_calibration"
        and (
            eligible_run_ids is None
            or event.get("run_id") in eligible_run_ids
        )
    ]
    candidate_by_id = {event["event_id"]: event for event in candidate_events}
    task_touch: dict[str, dict[str, Any]] = {}
    missing_touch_rows: list[str] = []
    for event in candidate_events:
        rows = touch_by_id.get(event["event_id"], [])
        if not rows:
            if event.get("action") in ACTIONS[:-1]:
                missing_touch_rows.append(event["event_id"])
            continue
        phases = {row.get("phase", "") for row in rows}
        if len(phases) != 1:
            raise RuntimeError(f"{event['event_id']}: touch rows span phases {phases}")
        task_touch[event["event_id"]] = {
            "event": event,
            "rows": rows,
            "phase": next(iter(phases)),
        }

    not_counted, receipt_audit = partition_module.map_not_counted_receipts(
        task_events, task_touch
    )
    task_keys = [
        event for event in candidate_events if event.get("action") == "keystroke"
    ]
    key_overlaps: dict[str, list[str]] = defaultdict(list)
    selected: dict[str, str] = {}
    exclusion_reasons: Counter[str] = Counter()
    event_intervals: dict[str, tuple[str, int, int]] = {}
    for event_id, item in task_touch.items():
        event = item["event"]
        action = str(event.get("action", ""))
        if action not in ACTIONS[:-1]:
            continue
        geometry = partition_module.event_geometry(event, item["rows"])
        valid, _rule = partition_module.collector_valid(action, geometry)
        reasons = []
        if not valid:
            reasons.append("fails_collector_physical_validity")
        if event_id in not_counted:
            reasons.append("collector_gesture_not_counted")
        start_ns = int(event["start_elapsed_ns"])
        end_ns = int(event["end_elapsed_ns"])
        for key in task_keys:
            if key.get("run_id") != event.get("run_id"):
                continue
            if partition_module.overlaps(
                start_ns,
                end_ns,
                int(key["start_elapsed_ns"]),
                int(key["end_elapsed_ns"]),
            ):
                key_overlaps[event_id].append(key["event_id"])
        if key_overlaps[event_id]:
            reasons.append("overlaps_higher_priority_keystroke")
        if reasons:
            exclusion_reasons.update(reasons)
            continue
        selected[event_id] = action
        event_intervals[event_id] = (str(event.get("run_id", "")), start_ns, end_ns)

    blocked: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for action in ("pinch", "swipe", "scroll", "tap"):
        current = [
            (event_id, event_intervals[event_id])
            for event_id, label in selected.items()
            if label == action
        ]
        for event_id, (run_id, start_ns, end_ns) in sorted(
            current, key=lambda item: item[1]
        ):
            if any(
                partition_module.overlaps(start_ns, end_ns, left, right)
                for left, right in blocked[run_id]
            ):
                del selected[event_id]
                exclusion_reasons["overlaps_higher_priority_nonkey_action"] += 1
            else:
                blocked[run_id].append((start_ns, end_ns))

    raw_counts = Counter(
        event.get("action")
        for event in candidate_events
        if event.get("action") in ACTIONS[:-1]
    )
    selected_counts = Counter(selected.values())
    if missing_touch_rows:
        exclusion_reasons["missing_application_touch_rows"] += len(missing_touch_rows)
    audit = {
        "raw_counts": {action: raw_counts[action] for action in ACTIONS[:-1]},
        "selected_counts": {
            action: selected_counts[action] for action in ACTIONS[:-1]
        },
        "missing_touch_event_ids": sorted(missing_touch_rows),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "gesture_not_counted_receipts": len(receipt_audit),
        "keystroke_overlaps": sum(bool(value) for value in key_overlaps.values()),
        "candidate_events": len(candidate_by_id),
    }
    return selected, audit


def official_complete_partition(
    session: Path,
    destination: Path,
    partition_script: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(partition_script),
            str(session),
            "--out",
            str(destination),
            "--policy",
            "release_exclusive",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"official partition failed for {session}: {completed.stderr.strip()}"
        )
    return json.loads(destination.read_text(encoding="utf-8"))


def build_gesture_events(
    matrix_roots: dict[str, Path],
    *,
    touch_observation: Any,
    partition_module: Any,
    partition_script: Path,
    partition_output: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    set[str],
]:
    """Build the full-18 primary and app-complete-12 sensitivity cohorts."""

    output = {action: [] for action in ACTIONS[:-1]}
    exclusions: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    cohort_audit: dict[str, Any] = {"sessions": {}}
    complete_sessions: set[str] = set()
    total_raw: Counter[str] = Counter()
    total_complete_raw: Counter[str] = Counter()
    total_primary: Counter[str] = Counter()
    total_sensitivity: Counter[str] = Counter()
    source_pairs = {
        "primary": Counter(),
        "sensitivity": Counter(),
    }

    for device, matrix_root in matrix_roots.items():
        summary_path = matrix_root / "reports" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_runs = {
            row["cell"]: row
            for row in summary["cells"]
            if row.get("condition") == "injected"
        }
        if len(task_runs) != 9:
            raise RuntimeError(f"{summary_path}: expected nine injected task runs")
        for report_path in sorted((matrix_root / "reports").glob("*_injected_*.json")):
            framework, task = framework_task(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            matrix_cell = task_runs[f"{framework}_injected_{task}"]
            if not (
                bool(matrix_cell["ok"])
                and int(matrix_cell["returncode"]) == 0
                and not bool(matrix_cell["timed_out"])
            ):
                raise RuntimeError(f"{report_path}: matrix task run was not successful")
            session = (matrix_root / str(report["session"])).resolve()
            events = read_csv(session / "events.csv")
            task_events = read_csv(session / "task_events.csv")
            completed_run_ids = {
                row["run_id"]
                for row in task_events
                if row.get("run_id") and row.get("event") == "task_complete"
            }
            if completed_run_ids:
                complete_sessions.add(str(session))
            touch_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in read_csv(session / "touch.csv"):
                if row.get("event_id"):
                    touch_by_id[row["event_id"]].append(row)

            primary_ids, primary_audit = release_exclusive_partition(
                events,
                touch_by_id,
                task_events,
                partition_module,
                eligible_run_ids=None,
            )
            sensitivity_ids: dict[str, str] = {}
            sensitivity_audit = None
            official_path = None
            if completed_run_ids:
                sensitivity_ids, sensitivity_audit = release_exclusive_partition(
                    events,
                    touch_by_id,
                    task_events,
                    partition_module,
                    eligible_run_ids=completed_run_ids,
                )
                official_path = (
                    partition_output / f"{device}__{framework}__{task}.json"
                )
                official = official_complete_partition(
                    session, official_path, partition_script
                )
                official_ids = {
                    event_id: str(row["action"])
                    for event_id, row in official["events"].items()
                    if row.get("action") in ACTIONS[:-1]
                }
                if official_ids != sensitivity_ids:
                    raise RuntimeError(
                        f"{session}: local release partition differs from official script"
                    )

            aligned, report_exclusions = align_served_report_records(report, events)
            receipt_by_event = {
                event["event_id"]: (record, delta_ms)
                for record, event, delta_ms in aligned
            }
            for event_id, action in primary_ids.items():
                cohorts = [
                    "primary",
                    *(["sensitivity"] if event_id in sensitivity_ids else []),
                ]
                if event_id in receipt_by_event:
                    record, _delta_ms = receipt_by_event[event_id]
                    source_action = str(record["receipt"]["action"])
                else:
                    source_action = "no_served_receipt"
                for cohort in cohorts:
                    source_pairs[cohort][(source_action, action)] += 1
            for item in report_exclusions:
                exclusions.append(
                    {
                        "device": device,
                        "framework": framework,
                        "task": task,
                        "session": str(session),
                        **item,
                    }
                )

            raw_counts = Counter(primary_audit["raw_counts"])
            total_raw.update(raw_counts)
            total_primary.update(primary_audit["selected_counts"])
            if sensitivity_audit is not None:
                total_complete_raw.update(sensitivity_audit["raw_counts"])
                total_sensitivity.update(sensitivity_audit["selected_counts"])
            imu_t_ms, imu_values = paired_imu(session)
            session_key = f"{device}__{framework}__{task}"
            sources[session_key] = {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "session": str(session),
                "app_task_complete": bool(completed_run_ids),
                "official_partition": str(official_path) if official_path else None,
                "served_non_keystroke_records": len(aligned) + len(report_exclusions),
                "app_visible_aligned_records": len(aligned),
            }
            cohort_audit["sessions"][session_key] = {
                "primary": primary_audit,
                "sensitivity": sensitivity_audit,
            }
            event_by_id = {event["event_id"]: event for event in events}
            for event_id, action in primary_ids.items():
                event = event_by_id[event_id]
                imu, trajectory, coordinate_audit = gesture_signal(
                    event,
                    touch_by_id[event_id],
                    imu_t_ms,
                    imu_values,
                    touch_observation=touch_observation,
                )
                receipt = receipt_by_event.get(event_id)
                output[action].append(
                    {
                        "event_id": event_id,
                        "action": action,
                        "device": device,
                        "framework": framework,
                        "task": task,
                        "session": str(session),
                        "cohorts": [
                            "primary",
                            *(["sensitivity"] if event_id in sensitivity_ids else []),
                        ],
                        "executor_action": (
                            str(receipt[0]["receipt"]["action"]) if receipt else None
                        ),
                        "report_record_index": (
                            int(receipt[0]["index"]) if receipt else None
                        ),
                        "receipt_to_app_start_delta_ms": (
                            float(receipt[1]) if receipt else None
                        ),
                        "vantage": "target-application MotionEvent and SensorEvent capture",
                        "coordinate_preclip_audit": coordinate_audit,
                        "imu": imu,
                        "trajectory": trajectory,
                    }
                )

    def ordered(counter: Counter[str]) -> dict[str, int]:
        return {action: int(counter[action]) for action in ACTIONS[:-1]}

    raw_counts = ordered(total_raw)
    complete_raw_counts = ordered(total_complete_raw)
    primary_counts = ordered(total_primary)
    sensitivity_counts = ordered(total_sensitivity)
    if raw_counts != EXPECTED_RAW_GESTURE_EVENTS:
        raise RuntimeError(
            f"raw full-18 partition drift: expected {EXPECTED_RAW_GESTURE_EVENTS}, got {raw_counts}"
        )
    if complete_raw_counts != EXPECTED_COMPLETE_RAW_GESTURE_EVENTS:
        raise RuntimeError(
            "raw app-complete partition drift: expected "
            f"{EXPECTED_COMPLETE_RAW_GESTURE_EVENTS}, got {complete_raw_counts}"
        )
    if sensitivity_counts != EXPECTED_PARTITION_GESTURE_EVENTS:
        raise RuntimeError(
            "app-complete release partition drift: expected "
            f"{EXPECTED_PARTITION_GESTURE_EVENTS}, got {sensitivity_counts}"
        )
    if primary_counts != {"tap": 1548, "scroll": 100, "swipe": 9, "pinch": 70}:
        raise RuntimeError(f"full-18 release partition drift: got {primary_counts}")
    cohort_audit.update(
        {
            "raw_full18_counts": raw_counts,
            "raw_app_complete12_counts": complete_raw_counts,
            "primary_full18_counts": primary_counts,
            "sensitivity_app_complete12_counts": sensitivity_counts,
            "source_action_to_app_action_counts": {
                cohort: {
                    f"{source}__to__{target}": count
                    for (source, target), count in sorted(counter.items())
                }
                for cohort, counter in source_pairs.items()
            },
            "source_action_conflicts_are_not_a_partition_filter": True,
        }
    )
    clipped_events = [
        {
            "device": event["device"],
            "framework": event["framework"],
            "task": event["task"],
            "session": event["session"],
            "event_id": event["event_id"],
            "action": event["action"],
            "cohorts": event["cohorts"],
            **event["coordinate_preclip_audit"],
        }
        for action_events in output.values()
        for event in action_events
        if event["coordinate_preclip_audit"]["clipped_rows"] > 0
    ]
    cohort_audit["coordinate_preclip"] = {
        "policy": (
            "normalize application raw coordinates by the reported display bounds, "
            "clip to the HMOG plane boundary as in the frozen phone scorer, then "
            "apply the canonical Android zero-order-hold observer"
        ),
        "score_dependent": False,
        "by_cohort": {
            cohort: {
                "affected_events": sum(cohort in row["cohorts"] for row in clipped_events),
                "clipped_rows": sum(
                    int(row["clipped_rows"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
                "clipped_coordinate_values": sum(
                    int(row["clipped_coordinate_values"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
                "x_hmog_min_before": min(
                    float(row["x_hmog_min_before"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
                "x_hmog_max_before": max(
                    float(row["x_hmog_max_before"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
                "y_hmog_min_before": min(
                    float(row["y_hmog_min_before"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
                "y_hmog_max_before": max(
                    float(row["y_hmog_max_before"])
                    for row in clipped_events
                    if cohort in row["cohorts"]
                ),
            }
            for cohort in ("primary", "sensitivity")
        },
        "events": clipped_events,
    }
    return output, exclusions, sources, cohort_audit, complete_sessions


def lcs_pairs(expected: list[int], observed: list[int]) -> list[tuple[int, int]]:
    """Align receipt records to app episodes without shifting after a loss."""

    n, m = len(expected), len(observed)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for left in range(n - 1, -1, -1):
        for right in range(m - 1, -1, -1):
            if expected[left] == observed[right]:
                table[left][right] = 1 + table[left + 1][right + 1]
            else:
                table[left][right] = max(
                    table[left + 1][right], table[left][right + 1]
                )
    result: list[tuple[int, int]] = []
    left = right = 0
    while left < n and right < m:
        if (
            expected[left] == observed[right]
            and table[left][right] == 1 + table[left + 1][right + 1]
        ):
            result.append((left, right))
            left += 1
            right += 1
        elif table[left + 1][right] >= table[left][right + 1]:
            left += 1
        else:
            right += 1
    return result


def reconstruct_keystroke_touch(
    *,
    grid_ms: np.ndarray,
    callback_rows: list[dict[str, str]],
    keys: list[dict[str, Any]],
    width_px: float,
    height_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct app-unobservable IME contacts from executor receipts."""

    if len(callback_rows) != len(keys):
        raise RuntimeError("callback/key count mismatch")
    callback_rows = sorted(
        callback_rows, key=lambda row: int(row["event_elapsed_ns"])
    )
    trajectory = np.zeros((len(grid_ms), 9), dtype=np.float32)
    trajectory[:, 7] = ((grid_ms - grid_ms[0]) / 1000.0).astype(np.float32)
    trajectory[:, 8] = 1.0
    contact_samples = 0
    for row, key in zip(callback_rows, keys):
        release_ms = float(row["event_elapsed_ns"]) / NS_PER_MS
        down_ms = release_ms - float(key["held_ms"])
        x_px = float(key["x"])
        y_px = float(key["y"])
        if not (0.0 <= x_px <= width_px and 0.0 <= y_px <= height_px):
            raise RuntimeError("executor key coordinate lies outside the display")
        inside = np.flatnonzero((grid_ms >= down_ms) & (grid_ms < release_ms))
        if not len(inside):
            inside = np.asarray([int(np.argmin(np.abs(grid_ms - down_ms)))])
        trajectory[inside, 0] = 1.0
        trajectory[inside, 1] = np.float32(x_px / width_px)
        trajectory[inside, 2] = np.float32(y_px / height_px)
        trajectory[inside, 3] = 1.0
        trajectory[inside, 4] = 1.0
        contact_samples += int(len(inside))
    contact = trajectory[:, 0] > 0.5
    consecutive = contact[1:] & contact[:-1]
    differences = np.diff(trajectory[:, 1:3], axis=0)
    trajectory[1:, 5:7][consecutive] = differences[consecutive]
    return trajectory, {
        "contacts": len(keys),
        "contact_samples": contact_samples,
        "reconstruction": "executor_receipt_aligned_to_textwatcher",
    }


def reconstruct_keystroke_events(
    matrix_roots: dict[str, Path], complete_sessions: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_per_task = {"shopping": 3, "search": 3, "social": 9}
    events_out: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_hashes: dict[str, Any] = {}
    for device, matrix_root in matrix_roots.items():
        reports = sorted((matrix_root / "reports").glob("*_injected_*.json"))
        if len(reports) != 9:
            raise RuntimeError(f"{matrix_root}: expected nine injected reports")
        for report_path in reports:
            framework, task = framework_task(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            session = (matrix_root / str(report["session"])).resolve()
            type_records = [
                row for row in report["records"] if row.get("api") == "type"
            ]
            if len(type_records) != expected_per_task[task]:
                raise RuntimeError(f"{report_path}: unexpected typing record count")
            app_events = sorted(
                [
                    row
                    for row in read_csv(session / "events.csv")
                    if row["action"] == "keystroke"
                ],
                key=lambda row: int(row["start_elapsed_ns"]),
            )
            callbacks: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in read_csv(session / "keystroke.csv"):
                callbacks[row["event_id"]].append(row)
            expected_lengths = [
                int(row["arguments"]["text_length"]) for row in type_records
            ]
            observed_lengths = [int(float(row["n_keys"])) for row in app_events]
            pairs = lcs_pairs(expected_lengths, observed_lengths)
            paired_report = {left for left, _right in pairs}
            for index in sorted(set(range(len(type_records))) - paired_report):
                exclusions.append(
                    {
                        "device": device,
                        "framework": framework,
                        "task": task,
                        "reason": "missing_matching_target_app_episode",
                        "expected_text_length": expected_lengths[index],
                    }
                )
            imu_t_ms, imu_values = paired_imu(session)
            source_hashes[f"{device}__{framework}__{task}"] = {
                "report_sha256": sha256_file(report_path),
                "events_sha256": sha256_file(session / "events.csv"),
                "keystroke_sha256": sha256_file(session / "keystroke.csv"),
                "imu_sha256": sha256_file(session / "imu.csv"),
                "aligned_events": len(pairs),
            }
            for report_index, event_index in pairs:
                record = type_records[report_index]
                event = app_events[event_index]
                keys = list(record["arguments"]["keys"])
                callback_rows = callbacks.get(event["event_id"], [])
                if (
                    len(keys) != int(float(event["n_keys"]))
                    or len(callback_rows) != len(keys)
                ):
                    exclusions.append(
                        {
                            "device": device,
                            "framework": framework,
                            "task": task,
                            "reason": "key_callback_count_mismatch",
                            "report_keys": len(keys),
                            "app_keys": int(float(event["n_keys"])),
                            "callbacks": len(callback_rows),
                        }
                    )
                    continue
                if not bool(record["arguments"].get("touch_realised")):
                    raise RuntimeError("typing receipt lacks realized touch")
                receipt = record["receipt"]
                start_ms = float(receipt["imu_start_elapsed_ns"]) / NS_PER_MS
                frames = int(receipt["imu_frames"])
                period_ms = float(receipt["imu"]["period_ms"])
                if not math.isclose(period_ms, PERIOD_MS):
                    raise RuntimeError("keystroke receipt grid is not 100 Hz")
                grid_ms = start_ms + np.arange(frames, dtype=np.float64) * period_ms
                if grid_ms[0] < imu_t_ms[0] or grid_ms[-1] > imu_t_ms[-1]:
                    raise RuntimeError("receipt IMU window lies outside app capture")
                imu = interpolate(imu_t_ms, imu_values, grid_ms)
                trajectory, touch_audit = reconstruct_keystroke_touch(
                    grid_ms=grid_ms,
                    callback_rows=callback_rows,
                    keys=keys,
                    width_px=float(event["display_width_px"]),
                    height_px=float(event["display_height_px"]),
                )
                events_out.append(
                    {
                        "event_id": event["event_id"],
                        "action": "keystroke",
                        "device": device,
                        "framework": framework,
                        "task": task,
                        "session": str(session),
                        "cohorts": [
                            "primary",
                            *(
                                ["sensitivity"]
                                if str(session) in complete_sessions
                                else []
                            ),
                        ],
                        "vantage": (
                            "executor pointer receipt plus target-application "
                            "SensorEvent capture"
                        ),
                        "touch_audit": touch_audit,
                        "imu": imu,
                        "trajectory": trajectory,
                    }
                )
    if len(events_out) != 82:
        raise RuntimeError(
            f"expected 82 audited keystroke events, found {len(events_out)}"
        )
    sensitivity_count = sum(
        "sensitivity" in event["cohorts"] for event in events_out
    )
    if sensitivity_count != 60:
        raise RuntimeError(
            f"expected 60 app-complete keystrokes, found {sensitivity_count}"
        )
    return events_out, {
        "expected_events": 90,
        "scored_events": len(events_out),
        "excluded_events": 90 - len(events_out),
        "exclusion_reason_counts": dict(
            sorted(Counter(row["reason"] for row in exclusions).items())
        ),
        "sources": source_hashes,
    }


def pad_batch(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    length = max(len(value) for value in values)
    channels = values[0].shape[1]
    batch = np.zeros((len(values), length, channels), dtype=np.float32)
    mask = np.zeros((len(values), length), dtype=bool)
    for index, value in enumerate(values):
        batch[index, : len(value)] = value
        mask[index, : len(value)] = True
    return batch, mask


def load_models(
    artifact: Path,
    event_detectors: Any,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = json.loads(
        (
            artifact
            / "methods/evaluation/event_level/common/release_cell_map.json"
        ).read_text(encoding="utf-8")
    )["cells"]
    models: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    fingerprints: dict[str, Any] = {}
    for action in ACTIONS:
        for modality in MODALITIES:
            for detector in DETECTORS:
                cell_id = f"{action}__{modality}__{detector}"
                cell = release[cell_id]
                model_root = artifact / str(cell["model_dir"])
                if detector in CLASSICAL:
                    model_path = model_root / "model.joblib"
                    models[cell_id] = joblib.load(model_path)
                else:
                    model_path = model_root / "checkpoint.pt"
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    model = event_detectors.build_deep_detector(detector, modality, action).to(device)
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    models[cell_id] = model.eval()
                    checkpoints[cell_id] = checkpoint
                fingerprints[cell_id] = {
                    "model": model_path.relative_to(artifact).as_posix(),
                    "model_sha256": sha256_file(model_path),
                    "formal_hmog_frr5_threshold": float(cell["frr5"]),
                    "threshold_source": "release development split; no phone calibration",
                }
    return models, checkpoints, {"cells": release, "fingerprints": fingerprints}


@torch.no_grad()
def score_deep_events(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    arrays: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    normalizer = checkpoint["normalizer"]
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    scores = []
    for offset in range(0, len(arrays), 64):
        batch, mask = pad_batch(arrays[offset : offset + 64])
        batch = (batch - mean[None, None, :]) / std[None, None, :]
        batch[~mask] = 0.0
        tensor = torch.from_numpy(batch).to(device)
        tensor_mask = torch.from_numpy(mask).to(device)
        scores.extend(torch.sigmoid(model(tensor, tensor_mask)).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def aggregate_rows(
    rows: list[dict[str, Any]],
    detectors: tuple[str, ...],
    *,
    cohort: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action in ACTIONS:
        result[action] = {}
        for modality in MODALITIES:
            selected = [
                row
                for row in rows
                if row["action"] == action
                and row["modality"] == modality
                and row["detector"] in detectors
                and cohort in row["cohorts"]
            ]
            event_ids = {
                (row["device"], row["framework"], row["task"], row["event_id"])
                for row in selected
            }
            if len(selected) != len(event_ids) * len(detectors):
                raise RuntimeError(f"decision accounting drift for {action}/{modality}")
            accepted = sum(bool(row["accepted_as_human"]) for row in selected)
            by_detector = {}
            for detector in detectors:
                subset = [row for row in selected if row["detector"] == detector]
                detector_accepted = sum(bool(row["accepted_as_human"]) for row in subset)
                by_detector[detector] = {
                    "accepted": detector_accepted,
                    "decisions": len(subset),
                    "far": detector_accepted / len(subset),
                }
            result[action][modality] = {
                "events": len(event_ids),
                "accepted": accepted,
                "decisions": len(selected),
                "far": accepted / len(selected),
                "by_detector": by_detector,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="artifact repository root (default: repository containing this script)",
    )
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--s21-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    detector_code = args.artifact / "methods/evaluation/detection/core"
    pipeline_code = args.artifact / "methods/generation/pipeline"
    on_device_code = args.artifact / "methods/evaluation/on_device"
    for code_root in (detector_code, pipeline_code, on_device_code):
        sys.path.insert(0, str(code_root))
    import event_detectors  # noqa: PLC0415
    import android_touch_observation  # noqa: PLC0415
    import partition_task_actions  # noqa: PLC0415

    matrix_roots = {"pixel10": args.pixel_root.resolve(), "s21": args.s21_root.resolve()}
    partition_script = on_device_code / "partition_task_actions.py"
    with tempfile.TemporaryDirectory(prefix="actreal-matrix-partitions-") as tmp:
        gesture_events, exclusions, sources, cohort_audit, complete_sessions = (
            build_gesture_events(
                matrix_roots,
                touch_observation=android_touch_observation,
                partition_module=partition_task_actions,
                partition_script=partition_script,
                partition_output=Path(tmp),
            )
        )
    keystroke_events, keystroke_audit = reconstruct_keystroke_events(
        matrix_roots, complete_sessions
    )
    events = {**gesture_events, "keystroke": keystroke_events}
    cohort_audit["keystroke_reconstruction"] = keystroke_audit
    cohort_audit["keystroke_counts"] = {
        cohort: sum(cohort in event["cohorts"] for event in events["keystroke"])
        for cohort in ("primary", "sensitivity")
    }
    cohort_counts = {
        cohort: {
            action: sum(cohort in event["cohorts"] for event in events[action])
            for action in ACTIONS
        }
        for cohort in ("primary", "sensitivity")
    }
    expected_cohort_counts = {
        "primary": {
            "tap": 1548,
            "scroll": 100,
            "swipe": 9,
            "pinch": 70,
            "keystroke": 82,
        },
        "sensitivity": {
            "tap": 1012,
            "scroll": 68,
            "swipe": 2,
            "pinch": 60,
            "keystroke": 60,
        },
    }
    if cohort_counts != expected_cohort_counts:
        raise RuntimeError(
            f"final cohort drift: expected {expected_cohort_counts}, got {cohort_counts}"
        )

    device = torch.device(args.device)
    models, checkpoints, protocol = load_models(args.artifact, event_detectors, device)
    release_cells = protocol["cells"]
    release_map_path = (
        args.artifact
        / "methods/evaluation/event_level/common/release_cell_map.json"
    )
    rows: list[dict[str, Any]] = []
    for action in ACTIONS:
        action_events = events[action]
        for modality in MODALITIES:
            arrays = [
                event_detectors.modality_values(
                    event["imu"], event["trajectory"], modality
                )
                for event in action_events
            ]
            for detector in DETECTORS:
                cell_id = f"{action}__{modality}__{detector}"
                if detector in CLASSICAL:
                    features = np.stack(
                        [
                            event_detectors.extract_event_features(
                                detector,
                                modality,
                                event["imu"],
                                event["trajectory"],
                            )
                            for event in action_events
                        ]
                    ).astype(np.float32)
                    scores = event_detectors.classical_scores(models[cell_id], features)
                else:
                    scores = score_deep_events(
                        models[cell_id], checkpoints[cell_id], arrays, device
                    )
                threshold = float(release_cells[cell_id]["frr5"])
                for event, score in zip(action_events, scores):
                    rows.append(
                        {
                            "action": action,
                            "modality": modality,
                            "detector": detector,
                            "cell_id": cell_id,
                            "event_id": event["event_id"],
                            "device": event["device"],
                            "framework": event["framework"],
                            "task": event["task"],
                            "session": event["session"],
                            "cohorts": event["cohorts"],
                            "vantage": event["vantage"],
                            "coordinate_preclip_audit": event.get(
                                "coordinate_preclip_audit"
                            ),
                            "samples": len(event["imu"]),
                            "score": float(score),
                            "threshold": threshold,
                            "threshold_source": "formal_hmog_frr5",
                            "phone_threshold_adjustment_used": False,
                            "accepted_as_human": bool(float(score) < threshold),
                        }
                    )
            print(f"scored {action}/{modality}: {len(action_events)} events", flush=True)

    primary_result = {
        "event_counts": cohort_counts["primary"],
        "decisions_per_action_modality_six_detector": {
            action: cohort_counts["primary"][action] * len(DETECTORS)
            for action in ACTIONS
        },
        "six_detector": aggregate_rows(rows, DETECTORS, cohort="primary"),
    }
    aggregate = {
        "schema": "actreal_matrix_far_six_detector_v1",
        "operating_point": {
            "threshold": "formal HMOG development-split FRR5",
            "accept_rule": "score < threshold",
            "phone_calibration_used": False,
            "model_refit": False,
            "threshold_reselection": False,
        },
        "scope": {
            "devices": list(matrix_roots),
            "frameworks": sorted({row["framework"] for row in rows}),
            "actions": list(ACTIONS),
            "modalities": list(MODALITIES),
            "detectors": list(DETECTORS),
            "aggregation_unit": (
                "pooled event-detector decisions (event-micro FAR); each event "
                "contributes one decision from every detector"
            ),
            "includes_app_observed_execution_period_filler": True,
            "served_action_only_metric": False,
            "touch_observer": (
                "legacy phone boundary clipping followed by the canonical Android "
                "zero-order-hold release observer"
            ),
            "gesture_partition": (
                "score-blind release-exclusive partition of target-application "
                "events; source-action disagreement is audited, not used as a filter"
            ),
            "cohort": (
                "all 18 matrix runs accepted by the matrix run receipt; app-visible "
                "release-exclusive gesture events plus 82 audited keystroke events"
            ),
        },
        "primary": primary_result,
        "attack": primary_result["six_detector"],
        "partition_exclusion_counts": dict(
            sorted(Counter(row["reason"] for row in exclusions).items())
        ),
    }
    aggregate_path = args.out / "matrix_far_aggregate.json"
    cohort_audit_path = args.out / "matrix_cohort_audit.json"
    far_table_json_path = args.out / "action_far_table.json"
    far_table_csv_path = args.out / "action_far_table.csv"
    # The app-complete subset is used only as an internal partition cross-check;
    # it is intentionally absent from the release result and audit.
    cohort_audit.pop("raw_app_complete12_counts", None)
    cohort_audit.pop("sensitivity_app_complete12_counts", None)
    for session_audit in cohort_audit.get("sessions", {}).values():
        session_audit.pop("sensitivity", None)
    source_pairs = cohort_audit.get("source_action_to_app_action_counts", {})
    if isinstance(source_pairs, dict):
        cohort_audit["source_action_to_app_action_counts"] = {
            "primary": source_pairs.get("primary", {})
        }
    coordinate = cohort_audit.get("coordinate_preclip", {})
    if isinstance(coordinate, dict):
        coordinate["by_cohort"] = {
            "primary": coordinate.get("by_cohort", {}).get("primary", {})
        }
        for event in coordinate.get("events", []):
            event.pop("session", None)
            event["cohorts"] = ["primary"]
    cohort_audit["keystroke_counts"] = {
        "primary": cohort_audit.get("keystroke_counts", {}).get("primary", 0)
    }
    write_json(aggregate_path, aggregate)
    write_json(cohort_audit_path, cohort_audit)
    action_labels = {
        "tap": "Tap",
        "scroll": "Scroll",
        "swipe": "Swipe",
        "pinch": "Pinch",
        "keystroke": "Keystroke",
    }
    modality_labels = {
        "Touch": "trajectory_xytime",
        "IMU": "imu_only",
        "Joint": "imu_trajectory_xytime",
    }
    far_rows = [
        {
            "Action": action_labels[action],
            **{
                f"{label}_FAR": primary_result["six_detector"][action][modality][
                    "far"
                ]
                for label, modality in modality_labels.items()
            },
        }
        for action in ACTIONS
    ]
    write_json(
        far_table_json_path,
        {
            "schema": "actreal_on_device_far_table_v1",
            "source": aggregate_path.name,
            "metric": "FAR",
            "definition": (
                "Fraction of ActReal event-detector decisions accepted as human"
            ),
            "aggregation": (
                "Event-micro across the six frozen detector decisions for each event"
            ),
            "operating_point": aggregate["operating_point"],
            "event_counts": {
                action_labels[action]: primary_result["event_counts"][action]
                for action in ACTIONS
            },
            "detectors": list(DETECTORS),
            "rows": far_rows,
        },
    )
    with far_table_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["Action", "Touch FAR", "IMU FAR", "Joint FAR"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in far_rows:
            writer.writerow(
                {
                    "Action": row["Action"],
                    "Touch FAR": f'{row["Touch_FAR"]:.6f}',
                    "IMU FAR": f'{row["IMU_FAR"]:.6f}',
                    "Joint FAR": f'{row["Joint_FAR"]:.6f}',
                }
            )
    receipt = {
        "schema": "actreal_matrix_far_six_detector_v1_receipt",
        "script": "methods/evaluation/on_device/score_matrix_far.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "compute_device": str(device),
        "release_cell_map": {
            "path": release_map_path.relative_to(args.artifact).as_posix(),
            "sha256": sha256_file(release_map_path),
        },
        "models": protocol["fingerprints"],
        "canonical_touch_observer": {
            "path": "methods/generation/pipeline/android_touch_observation.py",
            "sha256": sha256_file(Path(android_touch_observation.__file__).resolve()),
        },
        "official_partition_builder": {
            "path": "methods/evaluation/on_device/partition_task_actions.py",
            "sha256": sha256_file(partition_script.resolve()),
        },
        "coordinate_preclip_audit": cohort_audit["coordinate_preclip"]["by_cohort"],
        "sources": {
            key: {
                "report_sha256": value["report_sha256"],
                "app_task_complete": value["app_task_complete"],
                "served_non_keystroke_records": value["served_non_keystroke_records"],
                "app_visible_aligned_records": value["app_visible_aligned_records"],
            }
            for key, value in sources.items()
        },
        "keystroke_reconstruction": keystroke_audit,
        "outputs": {
            aggregate_path.name: sha256_file(aggregate_path),
            cohort_audit_path.name: sha256_file(cohort_audit_path),
            far_table_json_path.name: sha256_file(far_table_json_path),
            far_table_csv_path.name: sha256_file(far_table_csv_path),
        },
    }
    write_json(args.out / "matrix_run_receipt.json", receipt)
    print("primary", cohort_counts["primary"])
    for action in ACTIONS:
        print(
            action,
            {
                modality: round(
                    primary_result["six_detector"][action][modality]["far"], 6
                )
                for modality in MODALITIES
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
