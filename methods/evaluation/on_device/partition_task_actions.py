#!/usr/bin/env python3
"""Build the canonical detector-independent task-action partition.

Touches marked ``unclassified`` remain excluded because they have matching
``gesture_not_counted`` receipts and fail the collector's minimum
displacement/span rules. This script never uses authentication scores to decide
a label.

Three auditable policies are available:

``physical_valid``
    Keep completed-task touches whose recorded action passes the collector's
    physical validity rule, including valid off-phase gestures.

``protocol_accepted``
    Additionally exclude every touch paired with a collector
    ``gesture_not_counted`` receipt, without applying typing-window
    exclusivity.  This is retained as an ablation.

``release_exclusive``
    Also remove a touch whose time span intersects a task keystroke episode.
    This is the primary release-aligned phone policy: the authoritative HMOG
    preprocessor reserves complete typing-event intervals before applying
    pinch > swipe > scroll > tap to all remaining candidates.  The phone-side
    interval is the collector's submitted TextWatcher episode, which closes at
    Search/Send and is a conservative proxy for HMOG's first-down/last-up burst.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


GESTURES = ("tap", "scroll", "swipe", "pinch")
CALIBRATION_TASK = "fewshot_calibration"
RECEIPT_TOLERANCE_NS = 100_000_000  # receipt is emitted beside touch finalization


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_details(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in value.split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            out[key] = val
    return out


def parse_notes(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in value.split(";"):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def event_geometry(event: dict[str, str], rows: list[dict[str, str]]) -> dict:
    """Reconstruct only quantities used by the collector validity contract."""
    notes = parse_notes(event.get("notes", ""))
    x0, y0 = float(event["x_start"]), float(event["y_start"])
    x1, y1 = float(event["x_end"]), float(event["y_end"])
    dx, dy = x1 - x0, y1 - y0
    duration_ms = (
        int(event["end_elapsed_ns"]) - int(event["start_elapsed_ns"])
    ) / 1_000_000.0
    max_pointers = max(int(row["pointer_count"]) for row in rows)
    max_distance = notes.get("max_distance_px")
    if max_distance is None:
        max_distance = max(
            math.hypot(float(row["raw_x_px"]) - x0, float(row["raw_y_px"]) - y0)
            for row in rows
        )
    span_start = notes.get("pinch_start_span_px")
    span_end = notes.get("pinch_end_span_px")
    span_delta = (
        abs(span_end - span_start)
        if span_start is not None and span_end is not None
        else None
    )
    return {
        "duration_ms": duration_ms,
        "dx_px": dx,
        "dy_px": dy,
        "max_distance_from_start_px": max_distance,
        "max_pointers": max_pointers,
        "pinch_start_span_px": span_start,
        "pinch_end_span_px": span_end,
        "pinch_span_delta_px": span_delta,
        "touch_rows": len(rows),
    }


def collector_valid(action: str, g: dict) -> tuple[bool, str]:
    """Task-independent part of GestureCaptureView.isValidGesture."""
    duration = float(g["duration_ms"])
    pointers = int(g["max_pointers"])
    if duration <= 0 or duration > 10_000:
        return False, "duration outside (0, 10000] ms"
    if action == "tap":
        ok = pointers == 1 and float(g["max_distance_from_start_px"]) <= 70 and duration <= 1_000
        return ok, "one pointer, max distance <=70 px, duration <=1000 ms"
    if action == "scroll":
        ok = pointers == 1 and abs(float(g["dy_px"])) >= 120
        return ok, "one pointer and absolute vertical endpoint displacement >=120 px"
    if action == "swipe":
        dx, dy = abs(float(g["dx_px"])), abs(float(g["dy_px"]))
        ok = pointers == 1 and dx >= 120 and dx >= dy
        return ok, "one pointer, absolute horizontal displacement >=120 px and >= vertical"
    if action == "pinch":
        delta = g["pinch_span_delta_px"]
        ok = pointers >= 2 and delta is not None and float(delta) >= 30
        return ok, "at least two pointers and absolute start/end span change >=30 px"
    return False, "collector did not assign a known gesture action"


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Release preprocessor's half-open [start, end) overlap rule."""
    return a0 < b1 and b0 < a1


def map_not_counted_receipts(
    task_events: list[dict[str, str]],
    task_touch: dict[str, dict],
) -> tuple[dict[str, dict], list[dict]]:
    """Pair each receipt with the adjacent raw event, with strict guards."""
    receipts = [row for row in task_events if row.get("event") == "gesture_not_counted"]
    used: set[str] = set()
    matched: dict[str, dict] = {}
    receipt_audit: list[dict] = []
    for receipt in receipts:
        details = parse_details(receipt.get("details", ""))
        when = int(receipt["event_elapsed_ns"])
        candidates = []
        for event_id, item in task_touch.items():
            event = item["event"]
            if event_id in used:
                continue
            if event.get("run_id") != receipt.get("run_id"):
                continue
            if event.get("task") != receipt.get("task"):
                continue
            if event.get("action") != details.get("action"):
                continue
            if item["phase"] != details.get("phase"):
                continue
            delta = when - int(event["end_elapsed_ns"])
            candidates.append((abs(delta), delta, event_id))
        if not candidates:
            raise ValueError(f"no raw event matches gesture_not_counted receipt {receipt}")
        candidates.sort()
        absolute_delta, signed_delta, event_id = candidates[0]
        if absolute_delta > RECEIPT_TOLERANCE_NS:
            raise ValueError(
                f"receipt/event separation {absolute_delta / 1e6:.3f} ms exceeds guard"
            )
        if len(candidates) > 1 and candidates[1][0] == absolute_delta:
            raise ValueError(f"ambiguous gesture_not_counted receipt {receipt}")
        used.add(event_id)
        audit = {
            "event_id": event_id,
            "receipt_elapsed_ns": when,
            "event_end_elapsed_ns": int(task_touch[event_id]["event"]["end_elapsed_ns"]),
            "receipt_minus_event_end_ms": signed_delta / 1_000_000.0,
            "details": details,
        }
        matched[event_id] = audit
        receipt_audit.append(audit)
    return matched, receipt_audit


def map_cancelled_task_orphans(
    task_events: list[dict[str, str]],
    task_orphans: dict[str, list[dict[str, str]]],
    completed: set[str],
) -> list[dict]:
    """Allow only raw-only task touches with an exact motion_cancelled receipt.

    ACTION_CANCEL deliberately does not create an ``events.csv`` gesture.  The
    raw MotionEvent lifecycle remains in ``touch.csv`` and the collector emits
    a contemporaneous ``motion_cancelled`` task receipt.  Such a lifecycle is
    frozen as an excluded sample; it is never assigned a gesture label.
    """
    receipts_by_event_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for receipt in task_events:
        if receipt.get("event") != "motion_cancelled":
            continue
        details_items = receipt.get("details", "").split(";")
        if details_items and details_items[0] and "=" not in details_items[0]:
            receipts_by_event_id[details_items[0]].append(receipt)

    audits: list[dict] = []
    for event_id, rows in sorted(task_orphans.items()):
        tasks = {row.get("task", "") for row in rows}
        phases = {row.get("phase", "") for row in rows}
        run_ids = {row.get("run_id", "") for row in rows}
        motion_actions = [row.get("motion_action", "") for row in rows]
        if len(tasks) != 1 or len(phases) != 1 or len(run_ids) != 1:
            raise ValueError(f"task orphan {event_id} spans task/phase/run metadata")
        if motion_actions[0] != "ACTION_DOWN" or motion_actions[-1] != "ACTION_CANCEL":
            raise ValueError(
                f"task orphan {event_id} is not a DOWN-to-CANCEL lifecycle"
            )
        if motion_actions.count("ACTION_DOWN") != 1 or motion_actions.count("ACTION_CANCEL") != 1:
            raise ValueError(f"task orphan {event_id} has ambiguous cancel lifecycle")

        task = next(iter(tasks))
        phase = next(iter(phases))
        run_id = next(iter(run_ids))
        terminal_elapsed_ns = max(int(row["event_elapsed_ns"]) for row in rows)
        candidates = []
        for receipt in receipts_by_event_id.get(event_id, []):
            details = parse_details(receipt.get("details", ""))
            if receipt.get("task") != task or receipt.get("run_id") != run_id:
                continue
            if details.get("phase") != phase:
                continue
            delta = int(receipt["event_elapsed_ns"]) - terminal_elapsed_ns
            if 0 <= delta <= RECEIPT_TOLERANCE_NS:
                candidates.append((delta, receipt, details))
        if len(candidates) != 1:
            raise ValueError(
                f"task orphan {event_id} has {len(candidates)} exact motion_cancelled receipts"
            )
        delta, receipt, details = candidates[0]
        audits.append({
            "event_id": event_id,
            "exclusion_reason": "motion_cancelled_before_event_finalization",
            "task": task,
            "phase": phase,
            "run_id": run_id,
            "completed_task_run": run_id in completed,
            "touch_rows": len(rows),
            "first_motion_action": motion_actions[0],
            "terminal_motion_action": motion_actions[-1],
            "terminal_elapsed_ns": terminal_elapsed_ns,
            "receipt_elapsed_ns": int(receipt["event_elapsed_ns"]),
            "receipt_minus_terminal_ms": delta / 1_000_000.0,
            "receipt_details": details,
        })
    return audits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("physical_valid", "protocol_accepted", "release_exclusive"),
        default="release_exclusive",
    )
    args = parser.parse_args()

    manifest = json.loads((args.session / "manifest.json").read_text(encoding="utf-8"))
    events_rows = read_csv(args.session / "events.csv")
    events = {row["event_id"]: row for row in events_rows}
    task_events = read_csv(args.session / "task_events.csv")
    completed = {
        row["run_id"]
        for row in task_events
        if row.get("run_id") and row.get("event") == "task_complete"
    }

    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.session / "touch.csv"):
        if row.get("event_id"):
            by_event[row["event_id"]].append(row)

    task_touch: dict[str, dict] = {}
    cancelled_task_orphan_rows: dict[str, list[dict[str, str]]] = {}
    calibration_orphan_touch_event_ids: list[str] = []
    for event_id, rows in by_event.items():
        event = events.get(event_id)
        if event is None:
            # A rejected calibration attempt may have raw MotionEvent rows but
            # deliberately no accepted events.csv record.  It cannot enter the
            # task partition and is recorded explicitly instead of making a
            # score-blind task-only partition impossible.  A task orphan is
            # allowed only when its raw lifecycle and exact motion_cancelled
            # receipt prove that the collector intentionally did not finalize a
            # gesture event; it remains excluded and receives no inferred label.
            tasks = {row.get("task", "") for row in rows}
            if tasks == {CALIBRATION_TASK}:
                calibration_orphan_touch_event_ids.append(event_id)
                continue
            cancelled_task_orphan_rows[event_id] = rows
            continue
        if event.get("task") == CALIBRATION_TASK or event.get("run_id") not in completed:
            continue
        phases = {row.get("phase", "") for row in rows}
        if len(phases) != 1:
            raise ValueError(f"touch event {event_id} spans phases {phases}")
        task_touch[event_id] = {
            "event": event,
            "rows": rows,
            "phase": next(iter(phases)),
        }
    if not task_touch:
        raise ValueError("no completed-task touch events")

    cancelled_task_orphans = map_cancelled_task_orphans(
        task_events, cancelled_task_orphan_rows, completed
    )

    not_counted, receipt_audit = map_not_counted_receipts(task_events, task_touch)

    task_keys = [
        row for row in events_rows
        if row.get("action") == "keystroke"
        and row.get("task") != CALIBRATION_TASK
        and row.get("run_id") in completed
    ]
    key_overlaps: dict[str, list[str]] = defaultdict(list)
    for event_id, item in task_touch.items():
        event = item["event"]
        a0, a1 = int(event["start_elapsed_ns"]), int(event["end_elapsed_ns"])
        for key in task_keys:
            if key.get("run_id") != event.get("run_id"):
                continue
            b0, b1 = int(key["start_elapsed_ns"]), int(key["end_elapsed_ns"])
            if overlaps(a0, a1, b0, b1):
                key_overlaps[event_id].append(key["event_id"])

    output_events: dict[str, dict] = {}
    for event_id, item in task_touch.items():
        event = item["event"]
        action = event.get("action", "")
        geometry = event_geometry(event, item["rows"])
        valid, validity_rule = collector_valid(action, geometry)
        exclusions: list[str] = []
        if not valid:
            exclusions.append("fails_collector_physical_validity")
        if args.policy == "release_exclusive" and event_id in key_overlaps:
            exclusions.append("overlaps_higher_priority_keystroke")
        if args.policy in ("protocol_accepted", "release_exclusive") and event_id in not_counted:
            exclusions.append("collector_gesture_not_counted")
        final_action = action if valid and not exclusions else "excluded"
        output_events[event_id] = {
            **geometry,
            "phase": item["phase"],
            "run_id": event.get("run_id", ""),
            "recorded_action": action,
            "action": final_action,
            "collector_physical_valid": valid,
            "collector_validity_rule": validity_rule,
            "gesture_not_counted_receipt": event_id in not_counted,
            "overlapping_keystroke_event_ids": key_overlaps.get(event_id, []),
            "exclusion_reasons": exclusions,
        }

    # The release next applies pinch > swipe > scroll > tap.  It is vacuous in
    # the five audited phone sessions (there are no touch/touch overlaps), but
    # implementing it keeps the policy true if this script is reused.
    if args.policy == "release_exclusive":
        blocked: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for priority_action in ("pinch", "swipe", "scroll", "tap"):
            current = [
                (event_id, row) for event_id, row in output_events.items()
                if row["action"] == priority_action
            ]
            kept = []
            for event_id, row in sorted(
                current,
                key=lambda item: (
                    item[1]["run_id"],
                    int(task_touch[item[0]]["event"]["start_elapsed_ns"]),
                    int(task_touch[item[0]]["event"]["end_elapsed_ns"]),
                ),
            ):
                event = task_touch[event_id]["event"]
                start, end = int(event["start_elapsed_ns"]), int(event["end_elapsed_ns"])
                if any(overlaps(start, end, other_start, other_end)
                       for other_start, other_end in blocked[row["run_id"]]):
                    row["action"] = "excluded"
                    row["exclusion_reasons"].append("overlaps_higher_priority_nonkey_action")
                else:
                    kept.append((row["run_id"], start, end))
            for run_id, start, end in kept:
                blocked[run_id].append((start, end))

    recorded_counts = Counter(row["recorded_action"] for row in output_events.values())
    final_counts = Counter(
        row["action"] for row in output_events.values() if row["action"] in GESTURES
    )
    exclusion_counts = Counter(
        reason for row in output_events.values() for reason in row["exclusion_reasons"]
    )
    exclusion_counts.update(
        row["exclusion_reason"] for row in cancelled_task_orphans
    )
    excluded_ids = [
        event_id for event_id, row in output_events.items() if row["action"] == "excluded"
    ]
    excluded_ids.extend(row["event_id"] for row in cancelled_task_orphans)
    payload = {
        "schema": "task_action_partition_v2",
        "profile_id": manifest.get("profile_id"),
        "session_id": manifest.get("session_id"),
        "policy_name": args.policy,
        "raw_data_mutated": False,
        "detector_scores_used_for_labels": False,
        "partition_builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "completed_task_run_ids": sorted(completed),
        "policy": {
            "physical_validity": {
                "tap": "one pointer, max distance <=70 px, duration <=1000 ms",
                "scroll": "one pointer, absolute vertical endpoint displacement >=120 px",
                "swipe": "one pointer, absolute horizontal endpoint displacement >=120 px and >= vertical",
                "pinch": "at least two pointers and absolute start/end span change >=30 px",
                "unclassified": "never promoted; excluded",
            },
            "protocol_receipts": (
                "exclude every gesture_not_counted event"
                if args.policy in ("protocol_accepted", "release_exclusive")
                else "retain physically valid off-phase events as sensitivity analysis"
            ),
            "mutual_exclusion": (
                "the authoritative release preprocessor reserves complete typing-event "
                "intervals, then applies pinch > swipe > scroll > tap"
            ),
            "typing_window_exclusivity": (
                "exclude every touch whose interval overlaps a keystroke episode"
                if args.policy == "release_exclusive"
                else "not applied in this ablation"
            ),
            "typing_boundary_mapping": (
                "phone interval is first TextWatcher edit through submitText/Search/Send; "
                "HMOG interval is first key DOWN through last key UP. The phone interval "
                "is therefore a conservative event-adapter proxy, not an exact key-up boundary."
            ),
            "release_preprocessor": "external/preprocess.py",
            "outcome_independence": "no detector score, threshold, FRR, or FAR is consulted",
            "semantic_limit": (
                "collector swipe/scroll are horizontal/vertical; released HMOG labels "
                "are StrokeEvent/fling versus ScrollEvent, so equality is a mapping assumption"
            ),
        },
        "counts_recorded_touch": dict(sorted(recorded_counts.items())),
        "counts_final_touch": dict(sorted(final_counts.items())),
        "task_keystroke_events": len(task_keys),
        "gesture_not_counted_receipts": len(receipt_audit),
        "touches_overlapping_keystrokes": len(key_overlaps),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "excluded_event_ids": sorted(excluded_ids),
        "calibration_orphan_touch_event_ids": sorted(
            calibration_orphan_touch_event_ids
        ),
        "cancelled_task_orphans": cancelled_task_orphans,
        "receipt_event_map": receipt_audit,
        "events": output_events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({
        "profile_id": payload["profile_id"],
        "session_id": payload["session_id"],
        "policy_name": payload["policy_name"],
        "counts_recorded_touch": payload["counts_recorded_touch"],
        "counts_final_touch": payload["counts_final_touch"],
        "task_keystroke_events": payload["task_keystroke_events"],
        "gesture_not_counted_receipts": payload["gesture_not_counted_receipts"],
        "touches_overlapping_keystrokes": payload["touches_overlapping_keystrokes"],
        "exclusion_reason_counts": payload["exclusion_reason_counts"],
        "out": str(args.out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
