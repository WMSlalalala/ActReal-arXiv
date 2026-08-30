#!/usr/bin/env python3
"""Export de-identified, event-aligned on-device data used by ActReal."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PHONE_CODE = REPOSITORY_ROOT / "methods" / "on_device" / "evaluation"
sys.path.insert(0, str(PHONE_CODE))
from session_io import (  # noqa: E402
    ACTIONS,
    completed_run_ids,
    event_conditions,
    event_window,
    formal_geometry,
    open_session,
    selected_events,
    sensor_index,
    validate_run_health,
)


SCHEMA = "actreal_on_device_processed_v1"
TASKS = ("simulated_shopping", "simulated_search", "simulated_social")
DEVICE_NAMES = {"frankel": "pixel10", "o1q": "s21"}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def motion_code(value: str) -> int:
    names = {
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
    for name, code in names.items():
        if value.upper().startswith(name):
            return code
    raise ValueError(f"unknown motion action: {value!r}")


def canonical_runs(session, split: str):
    completed = completed_run_ids(session)
    if split == "fewshot":
        candidates = []
        for run_id in sorted(completed):
            rows = selected_events(session, task="fewshot_calibration", run_id=run_id)
            counts = Counter(row["action"] for row in rows)
            if rows and all(counts[action] == 5 for action in ACTIONS):
                candidates.append((max(int(row["start_elapsed_ns"]) for row in rows), run_id))
        if not candidates:
            raise ValueError("no completed five-shot run with exactly five events per action")
        return [max(candidates)[1]]

    # Freeze the first completed run for each task so accidental repetitions do not
    # increase that participant-device pair's weight.
    result = []
    for task in TASKS:
        completions = [
            row for row in session.task_events
            if row.get("event") == "task_complete"
            and row.get("task") == task
            and row.get("run_id") in completed
        ]
        if not completions:
            raise ValueError(f"no completed run for {task}")
        completions.sort(key=lambda row: int(row["event_elapsed_ns"]))
        result.append(completions[0]["run_id"])
    return result


def grouped_rows(path: Path):
    result = defaultdict(list)
    for row in read_csv(path):
        event_id = row.get("event_id", "")
        if event_id:
            result[event_id].append(row)
    return result


def empty_imu(action: str):
    t, _, _ = formal_geometry(action)
    return (
        np.empty((0, t, 6), dtype=np.float32),
        np.empty((0, t), dtype=np.uint8),
        np.empty((0, t), dtype=np.uint8),
    )


def build_archive(session, events, action: str, participant: str, device: str, split: str):
    sensors = sensor_index(session)
    action_events = [row for row in events if row["action"] == action]
    imu, imu_mask, action_mask = empty_imu(action)
    active_len = np.empty(0, dtype=np.int32)
    duration_ms = np.empty(0, dtype=np.float32)
    orientation_id = np.empty(0, dtype=np.int8)
    tasks = np.empty(0, dtype="U1")
    xy = np.empty((0, 4), dtype=np.float32)
    n_keys = np.empty(0, dtype=np.int16)
    n_letters = np.empty(0, dtype=np.int16)
    if action_events:
        windows = [event_window(event, sensors) for event in action_events]
        imu = np.stack([item["window"] for item in windows]).astype(np.float32)
        imu_mask = np.stack([item["valid_mask"] for item in windows]).astype(np.uint8)
        action_mask = np.stack([item["mask"] for item in windows]).astype(np.uint8)
        active_len = np.asarray([int(item["active_len"]) for item in windows], dtype=np.int32)
        duration_ms = np.asarray([float(item["duration_ms"]) for item in windows], dtype=np.float32)
        conditions = [event_conditions(event) for event in action_events]
        orientation_id = np.asarray([item["orientation_id"] for item in conditions], dtype=np.int8)
        tasks = np.asarray([event["task"] for event in action_events])
        xy = np.stack([item["xy"] for item in conditions]).astype(np.float32)
        n_keys = np.asarray([item["n_keys"] for item in conditions], dtype=np.int16)
        n_letters = np.asarray([item["n_letters"] for item in conditions], dtype=np.int16)

    values = {
        "schema": np.asarray(SCHEMA),
        "participant_id": np.asarray(participant),
        "device": np.asarray(device),
        "split": np.asarray(split),
        "action": np.asarray(action),
        "sample_count": np.asarray(len(action_events), dtype=np.int32),
        "task": tasks,
        "imu": imu,
        "imu_valid_mask": imu_mask,
        "action_mask": action_mask,
        "active_len": active_len,
        "duration_ms": duration_ms,
        "orientation_id": orientation_id,
        "xy_hmog": xy,
        "n_keys": n_keys,
        "n_letters": n_letters,
    }

    if action == "keystroke":
        by_event = grouped_rows(session.root / "keystroke.csv")
        offsets = [0]
        rel_ms, before, after, added, removed = [], [], [], [], []
        for event in action_events:
            start = int(event["start_elapsed_ns"])
            rows = sorted(by_event[event["event_id"]], key=lambda row: int(row["event_elapsed_ns"]))
            for row in rows:
                rel_ms.append((int(row["event_elapsed_ns"]) - start) / 1_000_000.0)
                before.append(int(row["before_count"]))
                after.append(int(row["after_count"]))
                added.append(int(row["added_count"]))
                removed.append(int(row["removed_count"]))
            offsets.append(len(rel_ms))
        values.update({
            "touch_representation": np.asarray("redacted_edit_timing"),
            "touch_offsets": np.asarray(offsets, dtype=np.int32),
            "touch_relative_ms": np.asarray(rel_ms, dtype=np.float32),
            "touch_before_count": np.asarray(before, dtype=np.int16),
            "touch_after_count": np.asarray(after, dtype=np.int16),
            "touch_added_count": np.asarray(added, dtype=np.int16),
            "touch_removed_count": np.asarray(removed, dtype=np.int16),
        })
    else:
        by_event = grouped_rows(session.root / "touch.csv")
        offsets = [0]
        rel_ms, pointer_id, pointer_count, x_norm, y_norm, pressure, size, codes = ([] for _ in range(8))
        for event in action_events:
            start = int(event["start_elapsed_ns"])
            rows = sorted(
                by_event[event["event_id"]],
                key=lambda row: int(row["event_elapsed_ns"]),
            )
            if not rows:
                raise ValueError(f"{action} event has no application-visible touch rows")
            for row in rows:
                width, height = float(row["display_width_px"]), float(row["display_height_px"])
                rel_ms.append((int(row["event_elapsed_ns"]) - start) / 1_000_000.0)
                pointer_id.append(int(row["pointer_id"]))
                pointer_count.append(int(row["pointer_count"]))
                x_norm.append(float(row["raw_x_px"]) / width)
                y_norm.append(float(row["raw_y_px"]) / height)
                pressure.append(float(row["pressure"]))
                size.append(float(row["size"]))
                codes.append(motion_code(row["motion_action"]))
            offsets.append(len(rel_ms))
        values.update({
            "touch_representation": np.asarray("application_motion_event_normalized"),
            "touch_offsets": np.asarray(offsets, dtype=np.int32),
            "touch_relative_ms": np.asarray(rel_ms, dtype=np.float32),
            "touch_pointer_id": np.asarray(pointer_id, dtype=np.int16),
            "touch_pointer_count": np.asarray(pointer_count, dtype=np.int8),
            "touch_x_normalized": np.asarray(x_norm, dtype=np.float32),
            "touch_y_normalized": np.asarray(y_norm, dtype=np.float32),
            "touch_pressure": np.asarray(pressure, dtype=np.float32),
            "touch_size": np.asarray(size, dtype=np.float32),
            "touch_motion_action": np.asarray(codes, dtype=np.int8),
        })
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-participants", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    canonical_task_runs = 0

    participant_numbers = sorted(
        int(path.name.removeprefix("human"))
        for path in args.source.glob("human[0-9]*")
        if path.is_dir() and path.name.removeprefix("human").isdigit()
    )
    for participant_num in participant_numbers:
        participant = f"P{participant_num:02d}"
        source_root = args.source / f"human{participant_num}"
        sessions = sorted({path.parent for path in source_root.rglob("manifest.json")})
        # Ignore nested duplicate copies by their repeated session_id.
        seen_session_ids = set()
        for source_session in sessions:
            try:
                with open_session(source_session) as session:
                    session_id = str(session.manifest["session_id"])
                    if session_id in seen_session_ids:
                        continue
                    seen_session_ids.add(session_id)
                    raw_device = str(session.manifest.get("device", ""))
                    if raw_device not in DEVICE_NAMES:
                        raise ValueError(f"unrecognized device {raw_device!r}")
                    device = DEVICE_NAMES[raw_device]
                    for split in ("fewshot", "test"):
                        runs = canonical_runs(session, split)
                        validate_run_health(session, runs)
                        events = [
                            event for run in runs
                            for event in selected_events(session, run_id=run)
                        ]
                        events.sort(key=lambda row: int(row["start_elapsed_ns"]))
                        for action in ACTIONS:
                            values = build_archive(session, events, action, participant, device, split)
                            destination = args.out / participant / device / split / f"{action}.npz"
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with destination.open("wb") as handle:
                                np.savez_compressed(handle, **values)
                            rows.append({
                                "participant_id": participant,
                                "device": device,
                                "split": split,
                                "action": action,
                                "samples": int(values["sample_count"]),
                                "touch_rows": int(len(values["touch_relative_ms"])),
                                "imu_frames": int(np.asarray(values["imu_valid_mask"]).sum()),
                                "relative_path": destination.relative_to(args.out).as_posix(),
                                "bytes": destination.stat().st_size,
                            })
                        if split == "test":
                            canonical_task_runs += len(runs)
            except Exception as exc:
                errors.append({"participant_id": participant, "source_index": len(seen_session_ids), "error": str(exc)})

    rows.sort(key=lambda row: (row["participant_id"], row["device"], row["split"], row["action"]))
    sample_totals = {
        split: {
            action: sum(
                int(row["samples"])
                for row in rows
                if row["split"] == split and row["action"] == action
            )
            for action in ACTIONS
        }
        for split in ("fewshot", "test")
    }
    fields = list(rows[0]) if rows else []
    with (args.out / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": SCHEMA,
        "protocol_expected_participants": int(args.expected_participants),
        "exported_participants": len({row["participant_id"] for row in rows}),
        "exported_participant_ids": sorted({row["participant_id"] for row in rows}),
        "expected_devices_per_participant": ["pixel10", "s21"],
        "exported_devices": len({(row["participant_id"], row["device"]) for row in rows}),
        "exported_canonical_task_runs": canonical_task_runs,
        "expected_files_per_device": 10,
        "exported_npz_files": len(rows),
        "missing_protocol_participant_ids": [
            f"P{index:02d}"
            for index in range(int(args.expected_participants))
            if index not in participant_numbers
        ],
        "canonical_test_rule": "earliest completed run per participant-device-task",
        "fewshot_rule": "latest completed calibration with exactly five events per action",
        "sample_totals": sample_totals,
        "imu_window_frames": {
            action: int(formal_geometry(action)[0]) for action in ACTIONS
        },
        "imu_sampling_hz": 100,
        "touch_timestamps": "milliseconds relative to each event start",
        "contains_raw_csv": False,
        "contains_entered_text": False,
        "contains_absolute_paths": False,
        "errors": errors,
    }
    (args.out / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if errors:
        print(json.dumps(summary, indent=2))
        raise SystemExit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
