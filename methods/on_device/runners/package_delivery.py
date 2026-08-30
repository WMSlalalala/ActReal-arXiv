#!/usr/bin/env python3
"""Check a collected session, split it by run, and package it for upload.

This is the last thing that runs before data leaves the machine, so it is
written to be suspicious.  Every check below exists because the failure it
catches has actually happened on this project and was invisible in the
summary numbers:

- a run whose touch stream is empty while its IMU stream is perfect, because
  the soft keyboard was covering the half of the screen the gestures aimed at;
- a stream whose rows all carry source 8194 rather than 4098, because the
  virtual device was missing INPUT_PROP_DIRECT and the whole session was
  recorded as a touchpad;
- an IMU stream whose sample spacing has a long flat gap in the middle, because
  the background player stopped during the agent's thinking;
- a session that looks complete and contains two victims, because a new
  capture session was never opened between them.

Nothing here judges whether the data looks human.  That is the detector's
question and the detector is not here; this only answers whether what was
collected is what was meant to be collected.

    python methods/on_device/runners/package_delivery.py SESSION_DIR [SESSION_DIR ...] --out delivery/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parents[1] / ".actreal-work"
sys.path.insert(0, str(ROOT))

# Rows that belong to one run and are therefore split; everything else
# describes the device or the session and is copied whole.
PER_RUN = ("events.csv", "touch.csv", "imu.csv", "keystroke.csv", "task_events.csv",
           "capture_health.csv", "accessibility.csv")
WHOLE = ("manifest.json", "sensor_info.csv", "export_audit.json")

SOURCE_TOUCHSCREEN = 4098      # what a digitiser reports
SOURCE_MOUSE = 8194            # what a device without INPUT_PROP_DIRECT reports


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, [row for row in reader]


def column(header: list[str], row: list[str], name: str) -> Optional[str]:
    if name not in header:
        return None
    index = header.index(name)
    return row[index] if len(row) > index else None


def require(header: list[str], rows: list[list[str]], *names: str) -> Optional[str]:
    """Refuse to check a stream whose columns are not the ones assumed.

    The first version of this file asked for ``timestamp_ns`` and ``source``.
    The recorder writes ``event_elapsed_ns`` and has no source column at all,
    so every sample rate came back None, every continuity check was skipped,
    and the run finished by printing "no problems found" -- a clean bill of
    health from a check that never executed. A missing column is now a finding,
    not a silent pass.
    """

    missing = [n for n in names if n not in header]
    if not header and rows:
        return "the file has rows but no header"
    if missing:
        return (f"expected columns {missing} are not in this file "
                f"(it has {header[:8]}...); the checks that use them did not run")
    return None


def check_touch(header: list[str], rows: list[list[str]]) -> dict[str, Any]:
    """What the touch stream says about the path it travelled."""

    if not rows:
        return {"rows": 0, "verdict": "no touch was recorded for this run"}
    complaint = require(header, rows, "pointer_id", "size", "raw_x_px", "raw_y_px")
    if complaint:
        return {"rows": len(rows), "problem": complaint}

    sizes = [float(s) for r in rows
             if (s := column(header, r, "size")) not in (None, "")]
    pointers = [int(p) for r in rows
                if (p := column(header, r, "pointer_id")) not in (None, "")]
    counts = [int(c) for r in rows
              if (c := column(header, r, "pointer_count")) not in (None, "")]
    out: dict[str, Any] = {
        "rows": len(rows),
        "distinct_pointers": len(set(pointers)),
        "max_pointer_count": max(counts, default=0),
        "size_median": round(statistics.median(sizes), 6) if sizes else None,
        "actions": dict(Counter(column(header, r, "motion_action") for r in rows).most_common(6)),
    }
    # There is no `source` column here, so the touchpad-vs-touchscreen check
    # that caught INPUT_PROP_DIRECT cannot be made from this file. Said out
    # loud rather than quietly dropped, because its absence is the difference
    # between "checked and fine" and "not checked".
    out["not_checked"] = (
        "input source is not recorded in touch.csv, so whether these arrived as "
        "SOURCE_TOUCHSCREEN cannot be confirmed from the session alone"
    )
    return out


def check_imu(header: list[str], rows: list[list[str]]) -> dict[str, Any]:
    """Rate and continuity, per sensor, which is where an idle stream shows its seams."""

    if not rows:
        return {"rows": 0, "verdict": "no inertia was recorded for this run"}
    complaint = require(header, rows, "event_elapsed_ns", "sensor")
    if complaint:
        return {"rows": len(rows), "problem": complaint}

    # Split by sensor first. The file interleaves accelerometer and gyroscope
    # rows, so pooling them reports twice the true rate and a median gap of
    # zero -- which reads as a healthy 200Hz stream on a phone sampling at 100.
    per_sensor: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        sensor = column(header, row, "sensor") or "?"
        value = column(header, row, "event_elapsed_ns")
        if value:
            try:
                per_sensor[sensor].append(float(value) / 1e6)
            except ValueError:
                pass

    out: dict[str, Any] = {"rows": len(rows), "sensors": {}}
    worst_silence = 0.0
    for sensor, stamps in sorted(per_sensor.items()):
        if len(stamps) < 3:
            out["sensors"][sensor] = {"rows": len(stamps)}
            continue
        stamps.sort()
        gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        span_s = (stamps[-1] - stamps[0]) / 1000.0
        if not gaps or span_s <= 0:
            out["sensors"][sensor] = {"rows": len(stamps)}
            continue
        silence = max(gaps) / 1000.0
        worst_silence = max(worst_silence, silence)
        out["sensors"][sensor] = {
            "rows": len(stamps),
            "span_s": round(span_s, 1),
            "rate_hz": round(len(stamps) / span_s, 1),
            "median_gap_ms": round(statistics.median(gaps), 2),
            "longest_silence_s": round(silence, 2),
        }
    out["longest_silence_s"] = round(worst_silence, 2)
    rates = [s.get("rate_hz") for s in out["sensors"].values() if s.get("rate_hz")]
    out["rate_hz"] = round(statistics.median(rates), 1) if rates else None

    # A second of nothing is not a sample-rate wobble; it is the stream having
    # stopped, and the agent's thinking is exactly where that used to happen.
    if worst_silence > 1.0:
        out["problem"] = (
            f"the inertial stream went quiet for {worst_silence:.1f}s; a phone "
            f"nobody was holding is the single most conspicuous thing a session "
            f"can contain"
        )
    return out


def inspect(session: Path) -> dict[str, Any]:
    """Everything worth knowing about one session, run by run."""

    report: dict[str, Any] = {"session": session.name, "runs": {}, "problems": []}

    manifest = session / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            report["participant"] = data.get("participant_id") or data.get("participant")
            report["device"] = data.get("device_model") or data.get("model")
        except json.JSONDecodeError:
            report["problems"].append("manifest.json is not readable JSON")

    events_header, events = read_rows(session / "events.csv")
    if not events:
        report["problems"].append("events.csv is empty; this session recorded nothing")
        return report

    by_run: dict[str, Counter] = defaultdict(Counter)
    tasks_of: dict[str, set] = defaultdict(set)
    for row in events:
        run = column(events_header, row, "run_id") or ""
        by_run[run][column(events_header, row, "action") or "?"] += 1
        task = column(events_header, row, "task")
        if task:
            tasks_of[run].add(task)

    touch_header, touch_rows = read_rows(session / "touch.csv")
    imu_header, imu_rows = read_rows(session / "imu.csv")
    task_header, task_rows = read_rows(session / "task_events.csv")

    completed = {
        column(task_header, r, "run_id")
        for r in task_rows
        if column(task_header, r, "event") == "task_complete"
    }

    for run, actions in by_run.items():
        run_touch = [r for r in touch_rows if column(touch_header, r, "run_id") == run]
        run_imu = [r for r in imu_rows if column(imu_header, r, "run_id") == run]
        entry = {
            "tasks": sorted(tasks_of.get(run, [])),
            "actions": dict(actions),
            "completed": run in completed,
            "touch": check_touch(touch_header, run_touch),
            "imu": check_imu(imu_header, run_imu),
        }
        for half in ("touch", "imu"):
            if "problem" in entry[half]:
                report["problems"].append(f"{run[:20]}: {entry[half]['problem']}")
        report["runs"][run] = entry

    # One session should be one person.  Two participants in one directory is
    # not a labelling annoyance -- the detector scores against a personal model,
    # so a mixed session is scored against a model that fits neither half.
    participants = {
        column(events_header, r, "participant_id")
        for r in events
        if column(events_header, r, "participant_id")
    }
    if len(participants) > 1:
        report["problems"].append(
            f"this session contains {len(participants)} participants "
            f"({sorted(participants)}); it was never split and belongs to nobody"
        )
    return report


def split_runs(session: Path, out_root: Path) -> list[str]:
    """One directory per run, each a session in its own right."""

    header, events = read_rows(session / "events.csv")
    runs = []
    for row in events:
        run = column(header, row, "run_id")
        if run and run not in runs:
            runs.append(run)

    written = []
    for run in runs:
        target = out_root / session.name / run
        target.mkdir(parents=True, exist_ok=True)
        for name in WHOLE:
            if (session / name).is_file():
                shutil.copy2(session / name, target / name)
        for name in PER_RUN:
            source = session / name
            if not source.is_file():
                continue
            file_header, rows = read_rows(source)
            if "run_id" not in file_header:
                shutil.copy2(source, target / name)
                continue
            index = file_header.index("run_id")
            with (target / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(file_header)
                writer.writerows(r for r in rows if len(r) > index and r[index] == run)
        written.append(run)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "delivery")
    parser.add_argument("--split", action="store_true", help="also write one directory per run")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    everything: dict[str, Any] = {"sessions": [], "problems": []}

    for session in args.sessions:
        if not session.is_dir():
            print(f"{session}: not a directory", file=sys.stderr)
            continue
        report = inspect(session)
        if args.split:
            report["split_runs"] = split_runs(session, args.out)
        everything["sessions"].append(report)
        everything["problems"].extend(f"{session.name}: {p}" for p in report["problems"])

        print(f"\n=== {session.name}  participant={report.get('participant', '?')} ===")
        for run, entry in report["runs"].items():
            touch, imu = entry["touch"], entry["imu"]
            print(f"  {run[:26]:<26} {','.join(entry['tasks']) or '?':<20} "
                  f"{'complete' if entry['completed'] else 'incomplete':<11} "
                  f"touch {touch['rows']:>5} (ptr<={touch.get('max_pointer_count', '?')}) "
                  f"imu {imu['rows']:>6} @{imu.get('rate_hz') or '?'}Hz "
                  f"quiet<={imu.get('longest_silence_s', '?')}s")
            print(f"      actions: {entry['actions']}")
        for problem in report["problems"]:
            print(f"  PROBLEM  {problem}")

    (args.out / "delivery_report.json").write_text(
        json.dumps(everything, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nreport -> {args.out / 'delivery_report.json'}")
    if everything["problems"]:
        print(f"{len(everything['problems'])} problems found; the data is not clean",
              file=sys.stderr)
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
