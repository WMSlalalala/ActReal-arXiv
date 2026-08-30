#!/usr/bin/env python3
"""Type a known string by contact and see whether the field agrees.

Everything about typing by key press can be checked on the host except the two
things that matter: whether the composed gesture survives the injection path --
fourteen separate contacts inside one bundle, which nothing else in this
project produces -- and whether the coordinates measured off a screenshot are
the coordinates the input stack resolves to keys.

So this plays one string into the study app's own search field and photographs
the result.  If the field reads back what was asked for, the touch channel
carried the text; the hook's own counters say whether the inertia went with it.

    python methods/on_device/runners/typing_check.py --bundles-dir BUNDLES --text "wireless mouse"
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
from actreal.planner import ActionPlanner, BundleLibrary  # noqa: E402
from actreal.rhythm import load_rhythm, press_durations  # noqa: E402
from actreal.mapping import ScreenMapping  # noqa: E402
from actreal.session import play_action  # noqa: E402
from actreal.typing_bundle import compose_typing, describe  # noqa: E402
from runners.joint_inject import probe, resumed  # noqa: E402

PACKAGE = "com.sensorworldmodel.collector"
TASK_ENTRY = (540, 1698)          # the Amazon card on the study app's home
SEARCH_FIELD = (400, 361)         # its search box, once the task is open


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--victim", required=True, help="five-shot profile identifier")
    parser.add_argument("--text", default="wireless mouse")
    parser.add_argument("--lead-ms", type=float, default=400.0)
    parser.add_argument("--shot", type=Path, default=None)
    args = parser.parse_args()

    adb = Adb()
    if not adb.devices():
        print("no device visible to adb", file=sys.stderr)
        return 2

    keymap = load_keymap()
    rhythm = load_rhythm(args.victim)
    pool = press_durations(args.victim)
    if keymap is None:
        print("no config/keymap.json -- run methods/on_device/runners/calibrate_keyboard.py",
              file=sys.stderr)
        return 3
    if rhythm is None or not pool:
        print(f"no measured typing rhythm for {args.victim}", file=sys.stderr)
        return 3
    ok, missing = keymap.covers(args.text)
    if not ok:
        print(f"this keyboard cannot reach {''.join(missing)!r}", file=sys.stderr)
        return 3
    print(f"keymap        : {len(keymap.chars)} characters, "
          f"shift {'yes' if keymap.shift else 'no'}")
    print(f"rhythm        : {rhythm.summary()}")

    adb.shell(f"am force-stop {PACKAGE}")
    time.sleep(2.0)
    adb.shell("input keyevent 224")
    adb.shell(f"am start -n {PACKAGE}/.MainActivity")
    time.sleep(6.0)

    config = configs.rooted(adb, package=PACKAGE, period_ms=10.0,
                            device_w=1080, device_h=2424, probe_report=probe(adb))
    print(f"configuration : {config.name}  touch {config.touch.describe()['backend']}")

    adb.shell(f"input tap {TASK_ENTRY[0]} {TASK_ENTRY[1]}")
    time.sleep(7.0)
    where = resumed(adb)
    if where != ".SimulatedTaskActivity":
        print(f"the task did not open ({where})", file=sys.stderr)
        return 4

    library = BundleLibrary(args.bundles_dir, seed=0)
    planner = ActionPlanner(library, ScreenMapping.isotropic(device_w=1080, device_h=2424))
    background.install(config.imu, library, period_ms=10.0)
    config.imu.set_mode("injected")
    if hasattr(config.touch, "open_device"):
        config.touch.open_device()

    # The field has to be focused and the keyboard up, or the calibrated key
    # rectangles describe a screen that is not the one on the phone.
    adb.shell(f"input tap {SEARCH_FIELD[0]} {SEARCH_FIELD[1]}")
    time.sleep(2.0)
    shown = adb.shell("dumpsys input_method")
    print(f"keyboard      : {'up' if 'mInputShown=true' in (shown.stdout or '') else 'DOWN'}")

    rng = random.Random(11)
    sequence = keymap.sequence(args.text, rng)
    gaps = rhythm.intervals(len(sequence), rng)
    holds = [pool[rng.randrange(len(pool))] for _ in sequence]
    donors = [planner.plan("tap", point, duration_ms=held).bundle
              for (_l, point), held in zip(sequence[:8], holds[:8])]
    bundle = compose_typing(sequence=sequence, gaps_ms=gaps, holds_ms=holds,
                            press_bundles=donors, rng=rng, mapping=planner.mapping)
    if bundle is None:
        print("the typing bundle could not be composed", file=sys.stderr)
        return 5
    print(f"bundle        : {describe(bundle)}")

    before = config.imu.session.status()["stats"]
    try:
        play_action(config, bundle,
                    read_uptime_ms=lambda: config.imu.read_clock().uptime_ms,
                    lead_ms=args.lead_ms, set_mode=False)
        time.sleep(bundle.imu_duration_ms / 1000.0 + 2.0)
    finally:
        if hasattr(config.touch, "close_device"):
            config.touch.close_device()

    stats = config.imu.session.status()["stats"]
    print("\n=== inertia ===")
    for key in ("events", "replaced", "from_window", "from_background", "no_frame"):
        print(f"  {key:<18} +{stats.get(key, 0) - before.get(key, 0)}")

    shot = args.shot or (WORK_ROOT / "debug" / "typing_check.png")
    shot.parent.mkdir(parents=True, exist_ok=True)
    adb.shell("screencap -p /sdcard/typing_check.png")
    adb.run("pull", "/sdcard/typing_check.png", str(shot))
    adb.shell("rm -f /sdcard/typing_check.png")
    print(f"\nscreenshot    : {shot}")
    print(f"expected text : {args.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
