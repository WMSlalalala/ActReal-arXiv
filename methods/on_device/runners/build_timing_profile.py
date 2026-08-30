#!/usr/bin/env python3
"""Collect one held-out victim's action durations from their five recordings.

No curve is fitted through travel.  Measured on the 70 training users alone --
never on the held-out ones -- travel and duration do not go together: scroll
gives a log correlation of -0.055 over 39,094 training events and -0.086 within
a training user at the median; swipe gives +0.102 and -0.107.  People cover
more ground by moving faster.

So the centre of a duration comes from the victim's own five recordings, and
the scatter around it from how much a training user varies within themselves.
Nothing is read off the held-out population.

The victim must come from the held-out split: that is who the attack targets.
A victim whose recordings run past an action's inertia window cannot be used
for that action, and this reports which ones do.

    python methods/on_device/runners/build_timing_profile.py --material MATERIAL --rank
    python methods/on_device/runners/build_timing_profile.py --material MATERIAL --victim PROFILE
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from actreal.duration_law import (
    DurationLawError,
    TimingBook,
    build_timing,
    window_cap_ms,
)

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def load_material(path: Path) -> dict[tuple[str, str], list[dict]]:
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["user_id"], row["action"])].append(row)
    return rows


def book_for(material, victim: str) -> tuple[TimingBook, list[str]]:
    timings = {}
    problems = []
    for action in ACTIONS:
        shots = material.get((victim, action), [])
        durations = [s["raw_duration_ms"] for s in shots if s.get("raw_duration_ms")]
        ids = [s.get("event_id", "") for s in shots if s.get("raw_duration_ms")]
        try:
            timings[action] = build_timing(
                durations, action=action, victim=victim, source_event_ids=ids
            )
        except DurationLawError as error:
            problems.append(str(error))
    return TimingBook(timings, victim=victim), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material",
        type=Path,
        required=True,
        help="licensed five-shot material_manifest.jsonl",
    )
    parser.add_argument("--victim", help="five-shot profile identifier")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / ".actreal-work" / "timing_profile.json",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="list usable held-out victims, don't write anything",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="which split a victim may come from; the attack targets a held-out user",
    )
    parser.add_argument(
        "--gestures",
        default="tap,scroll,swipe",
        help="actions that must fit their window for a victim to be usable",
    )
    args = parser.parse_args()

    material = load_material(args.material)
    required = [a.strip() for a in args.gestures.split(",") if a.strip()]

    allowed = {
        user
        for (user, _), shots in material.items()
        if shots and shots[0].get("split") == args.split
    }
    if not allowed:
        print(f"no users in split {args.split!r}", file=sys.stderr)
        return 2

    if args.rank:
        rows = []
        for victim in sorted(allowed):
            book, _ = book_for(material, victim)
            unfit = [a for a in required if a in book.actions and not book.get(a).fits_window()]
            missing = [a for a in required if a not in book.actions]
            if unfit or missing:
                continue
            medians = {a: book.get(a).median_ms for a in required}
            headroom = min(
                (window_cap_ms(a) or 1e9) / book.get(a).span_ms[1] for a in required
            )
            rows.append((victim, headroom, medians))
        rows.sort(key=lambda r: -r[1])
        print(f"{len(rows)}/{len(allowed)} {args.split} victims fit every window for {required}")
        print(f"{'victim':11s} {'window headroom':>15s}   " + "  ".join(f"{a} med" for a in required))
        for victim, headroom, medians in rows[:12]:
            mids = "  ".join(f"{medians[a]:7.0f}ms" for a in required)
            print(f"{victim:11s} {headroom:14.2f}x   {mids}")
        print("\ncaps: " + ", ".join(f"{a} {window_cap_ms(a):.0f}ms" for a in ACTIONS))
        return 0

    if not args.victim:
        parser.error("--victim is required unless --rank is used")
    if args.victim not in allowed:
        print(
            f"{args.victim} is not in the {args.split} split; the attack targets a "
            f"held-out user, so pick one of {sorted(allowed)[:5]}...",
            file=sys.stderr,
        )
        return 2

    book, problems = book_for(material, args.victim)
    for action in book.actions:
        timing = book.get(action)
        low, high = timing.span_ms
        cap = window_cap_ms(action)
        flag = "" if timing.fits_window() else f"  <-- exceeds the {cap:.0f}ms window"
        print(
            f"{args.victim}/{action:10s} {timing.count} shots  "
            f"{low:6.0f}-{high:6.0f}ms  median {timing.median_ms:6.0f}ms  "
            f"log spread {timing.log_spread:.3f}{flag}"
        )
    for problem in problems:
        print(f"  missing: {problem}")

    unfit = book.unfit_actions()
    target = args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": "actreal_timing_profile_v1",
                "note": (
                    "centre from the victim's own five recordings, scatter from the "
                    "70 training users' within-user spread; travel decides speed, "
                    "not duration (train-only scroll travel-duration r = -0.055 "
                    "over 39,094 events, -0.086 within a user)"
                ),
                "victim": args.victim,
                "split": args.split,
                "unfit_actions": unfit,
                "actions": book.as_dict(),
            },
            indent=1,
        )
    )
    print(f"\nwrote {target}")
    if unfit:
        print(f"  unusable actions for this victim: {unfit} (recordings exceed the window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
