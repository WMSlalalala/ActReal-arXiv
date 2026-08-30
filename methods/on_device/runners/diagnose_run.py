"""Read a finished run and say what went wrong, and whose fault it was.

Supervising a campaign by eye does not scale: a session is fifty-odd entries of
trajectory, most of them fine, and the two that matter are buried.  Worse, the
interesting question is not *which* actions failed but *why they were allowed
to* -- and the three causes want different responses:

``app``       the target ignores an input that any app would normally honour.
              The Enter key on this study app is the example: neither
              KEYCODE_ENTER nor KEYCODE_SEARCH submits a search, only the
              on-screen button does, so every search costs the agent an
              iteration discovering that again.  Nothing in the injection
              stack can fix this; the task instruction has to say so.

``injection`` the contact went out but landed somewhere stale -- the screen
              moved between the agent's screenshot and the action.  This is
              ours, and it is what emptied the idle filler of scrolls.

``agent``     the model misread the screen or chose a bad target.  Not a bug in
              anything; it is what the framework is, and the run recovers.

The split matters because only the first two are actionable, and confusing them
wastes a campaign.  A run stopped by app-design friction looks exactly like a
run stopped by bad injection if all you read is "three consecutive failures".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Matched against the reflector's own words.  Ordered: the first rule that
# matches wins, so the specific ones come before the general.
RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("app", "key event ignored by the app",
     re.compile(r"\b(enter|keyboard key|key event|return key)\b.{0,80}"
                r"(no (visible )?(change|effect)|did not|not submit|nothing)", re.I | re.S)),
    ("app", "Back left the task and reset its progress",
     re.compile(r"\bback\b.{0,120}(exit|left|leav|out of the task|returned to the "
                r"(app )?(home|main)|task list|restart)", re.I | re.S)),
    ("app", "control present but inert / precondition unmet",
     re.compile(r"(button|control).{0,60}(disabled|greyed|grayed|not (yet )?(active|enabled)|"
                r"unlocks once|still visible.{0,40}unchecked)", re.I | re.S)),
    ("injection", "screen moved under the action",
     re.compile(r"(screen|content|page|list).{0,40}"
                r"(shifted|scrolled|moved|changed position)", re.I | re.S)),
    ("agent", "wrong target chosen",
     re.compile(r"(wrong|incorrect|different) (element|item|button|result|position)|"
                r"tapped .{0,30}instead", re.I | re.S)),
)


def classify(feedback: str) -> tuple[str, str]:
    for who, label, pattern in RULES:
        if pattern.search(feedback or ""):
            return who, label
    return "agent", "unclassified"


def load_steps(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("steps", [])


def walk(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Pair each action with the verdict that follows it.

    The trajectory does not nest them: an action is written when it is chosen
    and the verdict lands in a later entry carrying the same step number, so
    the pairing is by step rather than by position.
    """

    actions: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    stop: Optional[str] = None
    for entry in steps:
        key = entry.get("step")
        if entry.get("action_object"):
            if key not in actions:
                order.append(key)
            actions.setdefault(key, {})["action"] = entry["action_object"]
        if entry.get("outcome"):
            if key not in actions:
                order.append(key)
            slot = actions.setdefault(key, {})
            slot["outcome"] = entry["outcome"]
            slot["feedback"] = entry.get("error_description") or ""
        for field in ("stop_reason", "finish_reason"):
            if entry.get(field):
                stop = entry[field]
    return [dict(step=k, **actions[k]) for k in order if k in actions], stop


def report(path: Path, verbose: bool = False) -> dict[str, Any]:
    steps = load_steps(path)
    paired, stop = walk(steps)

    blame: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    grades: Counter[str] = Counter()

    for item in paired:
        outcome = item.get("outcome")
        if not outcome:
            continue
        grades[outcome] += 1
        if outcome != "C":
            continue
        who, label = classify(item.get("feedback", ""))
        blame[who] += 1
        labels[label] += 1
        name = (item.get("action") or {}).get("name", "?")
        failures.append({
            "step": item["step"], "action": name, "who": who,
            "label": label, "feedback": item.get("feedback", "")[:200],
        })

    total = sum(grades.values())
    ok = grades.get("A", 0)
    print(f"Trajectory  {path.parent.parent.name}/{path.parent.name}")
    print(
        f"Actions  {len(paired)}, graded {total} -> successful {ok}, "
        f"partial {grades.get('B', 0)}, failed {grades.get('C', 0)}"
    )
    if stop:
        print(f"Stopped  {stop}")
    print()
    if not failures:
        print("No failed actions.")
    else:
        print("Failure attribution:")
        for who in ("app", "injection", "agent"):
            if blame[who]:
                tag = {
                    "app": "app design",
                    "injection": "injection/coordinates",
                    "agent": "agent decision",
                }[who]
                print(f"  {tag:<24} {blame[who]}")
        print()
        for f in failures:
            tag = {
                "app": "app design",
                "injection": "injection/coordinates",
                "agent": "agent decision",
            }[f["who"]]
            print(f"  step {f['step']:<3} {f['action']:<20} [{tag}] {f['label']}")
            if verbose:
                print(f"        {f['feedback']}")
    return {
        "path": str(path), "actions": len(paired), "grades": dict(grades),
        "blame": dict(blame), "labels": dict(labels), "failures": failures,
        "stop": stop,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", type=Path, help="framework steps.json to diagnose")
    parser.add_argument("--verbose", action="store_true",
                        help="print the reflector's own words for each failure")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    path = args.steps
    if not path.exists():
        print("no trajectory found", file=sys.stderr)
        return 2
    result = report(path, verbose=args.verbose)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
