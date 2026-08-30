#!/usr/bin/env python3
"""Find the hold length at which this keyboard stops reading a press as a letter.

Typing by contact put the right text on screen and then, once in a dozen keys,
the wrong one: "laptop" came back "lapt9p", and 9 is what the o key carries as
its secondary label. Gboard hands over the secondary character when a press is
held long enough, so the question is where that line sits -- and it has to be
measured on *this* path, because the earlier sweep used ``input swipe``, which
is Android synthesising events well above the driver, not our uinput device
reporting as the digitiser.

Presses of increasing length go into one field, all on the same key, and what
comes out says where the boundary is: a run of the letter followed by a run of
the digit, with the change at the threshold.

    python methods/on_device/runners/hold_sweep.py --bundles-dir BUNDLES --key p
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parents[1] / ".actreal-work"
sys.path.insert(0, str(ROOT))

from actreal import background, configs  # noqa: E402
from actreal.device import Adb  # noqa: E402
from actreal.keyboard import load_keymap  # noqa: E402
from actreal.mapping import ScreenMapping  # noqa: E402
from actreal.planner import ActionPlanner, BundleLibrary  # noqa: E402
from actreal.session import play_action  # noqa: E402
from actreal.typing_bundle import compose_typing  # noqa: E402
from runners.joint_inject import probe, resumed  # noqa: E402

PACKAGE = "com.sensorworldmodel.collector"
TASK_ENTRY = (540, 1698)
SEARCH_FIELD = (400, 361)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--key", default="p")
    parser.add_argument("--holds", default="40,80,120,160,200,240,300")
    parser.add_argument("--gap-ms", type=float, default=600.0)
    parser.add_argument("--shot", type=Path, default=WORK_ROOT / "debug" / "hold_sweep.png")
    args = parser.parse_args()

    holds = [float(v) for v in args.holds.split(",")]
    keymap = load_keymap()
    if keymap is None or args.key not in keymap.chars:
        print("no keymap, or the key is not on it", file=sys.stderr)
        return 3

    adb = Adb()
    adb.shell(f"am force-stop {PACKAGE}")
    time.sleep(2.0)
    adb.shell("input keyevent 224")
    adb.shell(f"am start -n {PACKAGE}/.MainActivity")
    time.sleep(6.0)
    config = configs.rooted(adb, package=PACKAGE, period_ms=10.0,
                            device_w=1080, device_h=2424, probe_report=probe(adb))
    adb.shell(f"input tap {TASK_ENTRY[0]} {TASK_ENTRY[1]}")
    time.sleep(7.0)
    if resumed(adb) != ".SimulatedTaskActivity":
        print("the task did not open", file=sys.stderr)
        return 4

    library = BundleLibrary(args.bundles_dir, seed=0)
    planner = ActionPlanner(library, ScreenMapping.isotropic(device_w=1080, device_h=2424))
    background.install(config.imu, library, period_ms=10.0)
    config.imu.set_mode("injected")
    if hasattr(config.touch, "open_device"):
        config.touch.open_device()

    adb.shell(f"input tap {SEARCH_FIELD[0]} {SEARCH_FIELD[1]}")
    time.sleep(2.0)

    rng = random.Random(5)
    rect = keymap.chars[args.key]
    centre = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    sequence = [(args.key, centre) for _ in holds]
    gaps = [args.gap_ms] * max(0, len(holds) - 1)
    donors = [planner.plan("tap", centre, duration_ms=h).bundle for h in holds[:6]]
    bundle = compose_typing(sequence=sequence, gaps_ms=gaps, holds_ms=holds,
                            press_bundles=donors, rng=rng, mapping=planner.mapping)
    if bundle is None:
        print("could not compose", file=sys.stderr)
        return 5

    print(f"key {args.key!r} at {centre}, holds {holds} ms")
    try:
        play_action(config, bundle,
                    read_uptime_ms=lambda: config.imu.read_clock().uptime_ms,
                    lead_ms=400.0, set_mode=False)
        time.sleep(bundle.imu_duration_ms / 1000.0 + 2.0)
    finally:
        if hasattr(config.touch, "close_device"):
            config.touch.close_device()

    args.shot.parent.mkdir(parents=True, exist_ok=True)
    adb.shell("screencap -p /sdcard/hold_sweep.png")
    adb.run("pull", "/sdcard/hold_sweep.png", str(args.shot))
    adb.shell("rm -f /sdcard/hold_sweep.png")
    print(f"screenshot: {args.shot}")
    print("read it left to right against the hold list above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
