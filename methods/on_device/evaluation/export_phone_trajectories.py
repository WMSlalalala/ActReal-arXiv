#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from session_io import (
    ACTIONS,
    HMOG_PORTRAIT_HEIGHT_PX,
    HMOG_PORTRAIT_WIDTH_PX,
    event_conditions,
    open_session,
    selected_events,
    sha256_file,
    validate_run_health,
)


SCHEMA = "phone_completed_trajectory_archive_v1"
TOUCH_ACTIONS = ("tap", "scroll", "swipe", "pinch")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def motion_action_code(value: str) -> int:
    value = value.upper()
    if value == "HISTORY":
        return -1
    names = (
        "ACTION_DOWN",
        "ACTION_UP",
        "ACTION_MOVE",
        "ACTION_CANCEL",
        "ACTION_OUTSIDE",
        "ACTION_POINTER_DOWN",
        "ACTION_POINTER_UP",
        "ACTION_HOVER_MOVE",
        "ACTION_SCROLL",
        "ACTION_HOVER_ENTER",
        "ACTION_HOVER_EXIT",
        "ACTION_BUTTON_PRESS",
        "ACTION_BUTTON_RELEASE",
    )
    for code, name in enumerate(names):
        if value.startswith(name):
            return code
    raise ValueError("unknown Android motion action %r" % value)


def finite_float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("non-finite %s in touch/edit row" % key)
    return value


def save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def touch_archive(
    action: str,
    events: Sequence[Mapping[str, str]],
    rows_by_event: Mapping[str, List[Mapping[str, str]]],
) -> Dict[str, np.ndarray]:
    offsets = [0]
    flat_elapsed_ns: List[int] = []
    flat_relative_ms: List[float] = []
    flat_pointer_id: List[int] = []
    flat_pointer_count: List[int] = []
    flat_x_px: List[float] = []
    flat_y_px: List[float] = []
    flat_x_hmog: List[float] = []
    flat_y_hmog: List[float] = []
    flat_pressure: List[float] = []
    flat_size: List[float] = []
    flat_motion_action_code: List[int] = []
    event_xy = []
    event_xy_raw = []
    for event in events:
        event_id = event["event_id"]
        rows = list(rows_by_event.get(event_id, []))
        if not rows:
            raise ValueError("%s event %s has no raw touch rows" % (action, event_id))
        rows = [
            row
            for _, row in sorted(
                enumerate(rows),
                key=lambda pair: (int(pair[1]["event_elapsed_ns"]), pair[0]),
            )
        ]
        start_ns = int(event["start_elapsed_ns"])
        end_ns = int(event["end_elapsed_ns"])
        if int(rows[0]["event_elapsed_ns"]) < start_ns - 1_000_000:
            raise ValueError("touch trajectory begins before its event")
        if int(rows[-1]["event_elapsed_ns"]) > end_ns + 1_000_000:
            raise ValueError("touch trajectory ends after its event")
        orientation = int(float(event["orientation_id"]))
        target_width = (
            HMOG_PORTRAIT_HEIGHT_PX
            if orientation in (1, 3)
            else HMOG_PORTRAIT_WIDTH_PX
        )
        target_height = (
            HMOG_PORTRAIT_WIDTH_PX
            if orientation in (1, 3)
            else HMOG_PORTRAIT_HEIGHT_PX
        )
        for row in rows:
            elapsed_ns = int(row["event_elapsed_ns"])
            x = finite_float(row, "raw_x_px")
            y = finite_float(row, "raw_y_px")
            width = finite_float(row, "display_width_px")
            height = finite_float(row, "display_height_px")
            if width <= 0 or height <= 0:
                raise ValueError("invalid display geometry in raw touch row")
            flat_elapsed_ns.append(elapsed_ns)
            flat_relative_ms.append((elapsed_ns - start_ns) / 1_000_000.0)
            flat_pointer_id.append(int(row["pointer_id"]))
            flat_pointer_count.append(int(row["pointer_count"]))
            flat_x_px.append(x)
            flat_y_px.append(y)
            flat_x_hmog.append(x * target_width / width)
            flat_y_hmog.append(y * target_height / height)
            flat_pressure.append(finite_float(row, "pressure"))
            flat_size.append(finite_float(row, "size"))
            flat_motion_action_code.append(motion_action_code(row["motion_action"]))
        offsets.append(len(flat_elapsed_ns))
        conditions = event_conditions(event)
        event_xy.append(np.asarray(conditions["xy"], dtype=np.float32))
        event_xy_raw.append(np.asarray(conditions["xy_raw"], dtype=np.float32))
    return {
        "schema": np.asarray(SCHEMA),
        "action": np.asarray(action),
        "trajectory_kind": np.asarray("raw_motion_event_per_pointer"),
        "event_offsets": np.asarray(offsets, dtype=np.int64),
        "event_id": np.asarray([row["event_id"] for row in events]),
        "task": np.asarray([row["task"] for row in events]),
        "posture": np.asarray([row.get("posture", "") for row in events]),
        "run_id": np.asarray([row.get("run_id", "") for row in events]),
        "duration_ms": np.asarray(
            [float(row["duration_ms"]) for row in events], dtype=np.float32
        ),
        "orientation_id": np.asarray(
            [int(float(row["orientation_id"])) for row in events], dtype=np.int64
        ),
        "event_xy_hmog": np.stack(event_xy).astype(np.float32),
        "event_xy_raw_px": np.stack(event_xy_raw).astype(np.float32),
        "flat_elapsed_ns": np.asarray(flat_elapsed_ns, dtype=np.int64),
        "flat_relative_ms": np.asarray(flat_relative_ms, dtype=np.float32),
        "flat_pointer_id": np.asarray(flat_pointer_id, dtype=np.int64),
        "flat_pointer_count": np.asarray(flat_pointer_count, dtype=np.int64),
        "flat_x_raw_px": np.asarray(flat_x_px, dtype=np.float32),
        "flat_y_raw_px": np.asarray(flat_y_px, dtype=np.float32),
        "flat_x_hmog": np.asarray(flat_x_hmog, dtype=np.float32),
        "flat_y_hmog": np.asarray(flat_y_hmog, dtype=np.float32),
        "flat_pressure": np.asarray(flat_pressure, dtype=np.float32),
        "flat_size": np.asarray(flat_size, dtype=np.float32),
        "flat_motion_action_code": np.asarray(
            flat_motion_action_code, dtype=np.int16
        ),
    }


def keystroke_archive(
    events: Sequence[Mapping[str, str]],
    rows_by_event: Mapping[str, List[Mapping[str, str]]],
) -> Dict[str, np.ndarray]:
    offsets = [0]
    elapsed_ns: List[int] = []
    relative_ms: List[float] = []
    before_count: List[int] = []
    after_count: List[int] = []
    added_count: List[int] = []
    removed_count: List[int] = []
    for event in events:
        rows = sorted(
            rows_by_event.get(event["event_id"], []),
            key=lambda row: int(row["event_elapsed_ns"]),
        )
        if not rows:
            raise ValueError(
                "keystroke event %s has no redacted edit rows" % event["event_id"]
            )
        start_ns = int(event["start_elapsed_ns"])
        for row in rows:
            timestamp = int(row["event_elapsed_ns"])
            elapsed_ns.append(timestamp)
            relative_ms.append((timestamp - start_ns) / 1_000_000.0)
            before_count.append(int(row["before_count"]))
            after_count.append(int(row["after_count"]))
            added_count.append(int(row["added_count"]))
            removed_count.append(int(row["removed_count"]))
        offsets.append(len(elapsed_ns))
    return {
        "schema": np.asarray(SCHEMA),
        "action": np.asarray("keystroke"),
        "trajectory_kind": np.asarray("redacted_textwatcher_edit_timing"),
        "event_offsets": np.asarray(offsets, dtype=np.int64),
        "event_id": np.asarray([row["event_id"] for row in events]),
        "task": np.asarray([row["task"] for row in events]),
        "posture": np.asarray([row.get("posture", "") for row in events]),
        "run_id": np.asarray([row.get("run_id", "") for row in events]),
        "duration_ms": np.asarray(
            [float(row["duration_ms"]) for row in events], dtype=np.float32
        ),
        "orientation_id": np.asarray(
            [int(float(row["orientation_id"])) for row in events], dtype=np.int64
        ),
        "n_keys": np.asarray(
            [int(float(row["n_keys"])) for row in events], dtype=np.int64
        ),
        "n_letters": np.asarray(
            [int(float(row["n_letters"])) for row in events], dtype=np.int64
        ),
        "flat_elapsed_ns": np.asarray(elapsed_ns, dtype=np.int64),
        "flat_relative_ms": np.asarray(relative_ms, dtype=np.float32),
        "flat_before_count": np.asarray(before_count, dtype=np.int64),
        "flat_after_count": np.asarray(after_count, dtype=np.int64),
        "flat_added_count": np.asarray(added_count, dtype=np.int64),
        "flat_removed_count": np.asarray(removed_count, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export completed phone runs as audited variable-length trajectory archives."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["simulated_shopping", "simulated_search", "simulated_social"],
    )
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("trajectory output must be a new empty directory")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "tasks": list(args.tasks),
        "actions": {},
        "limitations": [
            "Non-key actions preserve the complete in-app raw per-pointer MotionEvent trajectory.",
            "Keystroke uses a redacted TextWatcher edit-timing sequence because the system IME does not expose its raw touch trajectory to this Activity.",
            "HMOG-scaled coordinates are deterministic derived fields; raw phone pixels are preserved separately.",
        ],
    }
    with open_session(args.session) as session:
        manifest.update(
            {
                "source_session": str(session.source_path),
                "source_session_sha256": session.source_sha256,
                "session_id": session.manifest["session_id"],
                "profile_id": session.manifest["profile_id"],
            }
        )
        selected = [
            row
            for task in args.tasks
            for row in selected_events(
                session, task=task, actions=ACTIONS, completed_only=True
            )
        ]
        selected.sort(key=lambda row: int(row["start_elapsed_ns"]))
        if not selected:
            raise ValueError("no events from completed requested task runs")
        validate_run_health(
            session, [row["run_id"] for row in selected if row.get("run_id")]
        )
        by_action = {
            action: [row for row in selected if row["action"] == action]
            for action in ACTIONS
        }
        missing = [action for action, rows in by_action.items() if not rows]
        if missing:
            raise ValueError(
                "completed task runs contain no trajectory for: %s"
                % ", ".join(missing)
            )
        touch_by_event: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
        for row in read_csv(session.root / "touch.csv"):
            if row["event_id"]:
                touch_by_event[row["event_id"]].append(row)
        edits_by_event: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
        for row in read_csv(session.root / "keystroke.csv"):
            if row["event_id"]:
                edits_by_event[row["event_id"]].append(row)

        for action in ACTIONS:
            path = args.out / ("phone_trajectory_%s.npz" % action)
            values = (
                keystroke_archive(by_action[action], edits_by_event)
                if action == "keystroke"
                else touch_archive(action, by_action[action], touch_by_event)
            )
            save_npz(path, values)
            manifest["actions"][action] = {
                "events": len(by_action[action]),
                "flat_rows": int(len(values["flat_elapsed_ns"])),
                "archive": str(path),
                "archive_sha256": sha256_file(path),
                "trajectory_kind": str(
                    np.asarray(values["trajectory_kind"]).item()
                ),
            }
    (args.out / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
