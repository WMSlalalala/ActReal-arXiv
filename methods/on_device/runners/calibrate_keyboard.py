#!/usr/bin/env python3
"""Measure where this phone's keys are by pressing them and reading the field.

The obvious way to do this is the accessibility tree, the way the idle filler
finds inert regions.  It does not work: ``uiautomator dump`` returns the focused
window only, and the IME is a window of its own, so the keyboard's package never
appears in the dump at all -- verified on this device with Gboard in front.

The next obvious way is to measure a screenshot.  That was tried and it failed
in a way worth recording, because it failed *quietly*: the letter rows came out
50 px high, which still put every key's centre inside its key, so tapping the
centres typed a-z correctly and the map looked right.  What it could not
survive was scatter.  Contacts aimed a little above centre crossed into the
number row, and typing came back with o as 9, p as 0, t as 5 -- each the key
directly overhead.  A calibration that passes its own check and then fails in
use is worse than one that fails immediately.

So the keys are found the way the input stack sees them: press a point, read
what arrived.  A column is walked in steps and the character that comes back
names the row, which puts the boundaries within one step and needs no
assumption about what the keyboard looks like.

    python methods/on_device/runners/calibrate_keyboard.py --probe
    python methods/on_device/runners/calibrate_keyboard.py --probe --step 8 --out config/keymap.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actreal.device import Adb  # noqa: E402
from actreal.keyboard import KeyMap, to_dict  # noqa: E402

REMOTE = "/sdcard/actreal_probe.xml"
SEARCH = "am start -a android.search.action.GLOBAL_SEARCH"

# The layout this measures: three letter rows and the bar under them.  Which
# characters sit where is not in question -- it is QWERTY -- so what is measured
# is only where the rows begin and end and how wide a key is.
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
ROW_X0 = (58, 112, 218)
KEY_W = 107


def field_text(adb: Adb) -> str:
    """Whatever the focused text field currently holds, or ""."""

    got = adb.shell(f"uiautomator dump {REMOTE}")
    if not got.ok or "dumped to" not in (got.stdout or ""):
        return ""
    xml = adb.shell(f"cat {REMOTE}").stdout or ""
    adb.shell(f"rm -f {REMOTE}")
    best = ""
    for node in re.findall(r"<node\b[^>]*>", xml):
        if 'class="android.widget.EditText"' in node or 'focused="true"' in node:
            text = re.search(r'text="([^"]*)"', node)
            if text and len(text.group(1)) > len(best):
                best = text.group(1)
    return best


def clear(adb: Adb, count: int = 40) -> None:
    adb.shell(" ; ".join(["input keyevent 67"] * count))


def probe_column(adb: Adb, x: int, y0: int, y1: int, step: int) -> list[tuple[int, str]]:
    """Tap down a column and report which character each point produced."""

    ys = list(range(y0, y1 + 1, step))
    clear(adb)
    adb.shell(" ; ".join(f"input tap {x} {y}" for y in ys))
    time.sleep(1.5)
    got = field_text(adb)
    # One press, one character -- when that does not hold the read is unusable
    # and saying so beats lining up whatever did arrive against the wrong point.
    if len(got) != len(ys):
        return [(y, "") for y in ys]
    return list(zip(ys, got))


def boundary(readings: list[tuple[int, str]], upper: str, lower: str) -> Optional[int]:
    """The first y that produced ``lower`` after a run of ``upper``."""

    seen_upper = False
    for y, ch in readings:
        if ch == upper:
            seen_upper = True
        elif seen_upper and ch == lower:
            return y
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "config" / "keymap.json")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--probe", action="store_true",
                        help="walk a column and read back what each point typed")
    parser.add_argument("--step", type=int, default=10, help="probe step in pixels")
    parser.add_argument("--from-y", type=int, default=1580)
    parser.add_argument("--to-y", type=int, default=1900)
    args = parser.parse_args()

    adb = Adb(serial=args.serial)
    if not adb.devices():
        print("no device visible to adb", file=sys.stderr)
        return 2
    print(f"input method : {adb.shell('settings get secure default_input_method').stdout.strip()}")

    adb.shell(SEARCH)
    time.sleep(3.0)
    if "mInputShown=true" not in (adb.shell("dumpsys input_method").stdout or ""):
        print("no text field took focus; the keyboard never appeared", file=sys.stderr)
        return 3

    if not args.probe:
        print("pass --probe: the accessibility tree does not contain the IME on "
              "this device, so there is nothing else to read", file=sys.stderr)
        return 4

    # The o column: 'o' in the top row, 'k' below it, and the number row above.
    x = 914
    readings = probe_column(adb, x, args.from_y, args.to_y, args.step)
    if not any(ch for _y, ch in readings):
        print("the field did not report its text, so the probe cannot be read",
              file=sys.stderr)
        print("readings:", readings, file=sys.stderr)
        return 5
    print("probe at x=%d:" % x)
    print("   " + " ".join(f"{y}:{ch or '?'}" for y, ch in readings))

    top = boundary(readings, "9", "o")
    bottom = boundary(readings, "o", "k")
    if top is None or bottom is None:
        print(f"could not find both boundaries (top={top}, bottom={bottom})",
              file=sys.stderr)
        return 6
    height = bottom - top
    print(f"row 1        : {top}..{bottom}  height {height}")

    chars: dict[str, tuple[int, int, int, int]] = {}
    for index, (row, x0) in enumerate(zip(ROWS, ROW_X0)):
        cy = top + height * index + height // 2
        for position, ch in enumerate(row):
            cx = x0 + position * KEY_W
            chars[ch] = (cx - KEY_W // 2, cy - height // 2,
                         cx + KEY_W // 2, cy + height // 2)
    bar = top + height * 3
    chars[" "] = (390, bar, 798, bar + height)
    chars[","] = (165, bar, 272, bar + height)
    chars["."] = (807, bar, 914, bar + height)

    keymap = KeyMap(
        package=(adb.shell("settings get secure default_input_method").stdout or "").strip().split("/")[0],
        chars=chars,
        shift=(31, top + height * 2, 138, top + height * 3),
        delete=(940, top + height * 2, 1047, top + height * 3),
        symbols=(31, bar, 138, bar + height),
        area=(4, top, 1076, bar + height),
    )

    payload = to_dict(keymap)
    payload["provenance"] = {
        "method": "touch probe",
        "probe_x": x, "step_px": args.step,
        "row1_top": top, "row1_bottom": bottom, "row_height": height,
        "readings": [{"y": y, "char": ch} for y, ch in readings],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved        : {args.out}")

    clear(adb)
    adb.shell("input keyevent KEYCODE_HOME")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
