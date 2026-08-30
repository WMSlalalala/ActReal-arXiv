#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from session_io import (
    ACTIONS,
    event_conditions,
    event_window,
    open_session,
    selected_events,
    sensor_index,
    sha256_file,
    completed_run_ids,
    validate_run_health,
)


SCHEMA = "phone_five_shot_refs_v1"


def save_npz(path: Path, values: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an Android export and materialize exactly five phone refs per action."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise FileExistsError("%s is not empty; use a new directory" % args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    with open_session(args.session) as session:
        sensors = sensor_index(session)
        completed = completed_run_ids(session)
        calibration_run_ids = sorted(
            set(
                row.get("run_id", "")
                for row in session.events
                if row.get("task") == "fewshot_calibration" and row.get("run_id")
            )
        )
        calibration_runs_audit = {}
        candidates = []
        for run_id in calibration_run_ids:
            run_events = selected_events(
                session, task="fewshot_calibration", run_id=run_id
            )
            counts = Counter(row["action"] for row in run_events)
            is_completed = run_id in completed
            calibration_runs_audit[run_id] = {
                "completed_marker": is_completed,
                "counts": {action: counts[action] for action in ACTIONS},
            }
            if is_completed and all(counts[action] == 5 for action in ACTIONS):
                candidates.append((max(int(row["start_elapsed_ns"]) for row in run_events),
                                   run_id, run_events))
        if not candidates:
            raise ValueError(
                "no completed run_id contains exactly five accepted events per action"
            )
        _, selected_run_id, events = max(candidates)
        health = validate_run_health(session, [selected_run_id])
        postures = sorted(set(row.get("posture", "") for row in events))
        if len(postures) != 1:
            raise ValueError("five-shot run mixes posture values: %s" % postures)
        audit: Dict[str, Any] = {
            "schema": SCHEMA,
            "source_session": str(session.source_path),
            "source_session_sha256": session.source_sha256,
            "session_id": session.manifest["session_id"],
            "profile_id": session.manifest["profile_id"],
            "selected_run_id": selected_run_id,
            "posture": postures[0],
            "capture_health": health[selected_run_id],
            "calibration_runs_audit": calibration_runs_audit,
            "actions": {},
        }
        for action in ACTIONS:
            action_events = [row for row in events if row["action"] == action]
            windows: List[np.ndarray] = []
            masks: List[np.ndarray] = []
            valid_masks: List[np.ndarray] = []
            active_len: List[int] = []
            duration_ms: List[float] = []
            orientation_id: List[int] = []
            xy: List[np.ndarray] = []
            n_keys: List[int] = []
            n_letters: List[int] = []
            event_ids: List[str] = []
            for event in action_events:
                built = event_window(event, sensors)
                conditions = event_conditions(event)
                windows.append(built["window"])
                masks.append(built["mask"])
                valid_masks.append(built["valid_mask"])
                active_len.append(int(built["active_len"]))
                duration_ms.append(float(built["duration_ms"]))
                orientation_id.append(int(conditions["orientation_id"]))
                xy.append(np.asarray(conditions["xy"], dtype=np.float32))
                n_keys.append(int(conditions["n_keys"]))
                n_letters.append(int(conditions["n_letters"]))
                event_ids.append(str(event["event_id"]))
            path = args.out / ("refs_%s.npz" % action)
            save_npz(
                path,
                {
                    "schema": np.asarray(SCHEMA),
                    "source_session_sha256": np.asarray(session.source_sha256),
                    "profile_id": np.asarray(str(session.manifest["profile_id"])),
                    "run_id": np.asarray(selected_run_id),
                    "posture": np.asarray(postures[0]),
                    "action": np.asarray(action),
                    "windows": np.stack(windows).astype(np.float32),
                    "mask": np.stack(masks).astype(np.uint8),
                    "valid_mask": np.stack(valid_masks).astype(np.uint8),
                    "active_len": np.asarray(active_len, dtype=np.int64),
                    "duration_ms": np.asarray(duration_ms, dtype=np.float32),
                    "orientation_id": np.asarray(orientation_id, dtype=np.int64),
                    "xy": np.stack(xy).astype(np.float32),
                    "n_keys": np.asarray(n_keys, dtype=np.int64),
                    "n_letters": np.asarray(n_letters, dtype=np.int64),
                    "event_id": np.asarray(event_ids),
                },
            )
            audit["actions"][action] = {
                "count": 5,
                "T": int(windows[0].shape[0]),
                "active_len": active_len,
                "duration_ms": duration_ms,
                "event_id": event_ids,
                "refs_file": str(path),
                "refs_sha256": sha256_file(path),
                "full_window_sensor_coverage": [
                    float(np.mean(valid)) for valid in valid_masks
                ],
            }
        (args.out / "refs_manifest.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
