#!/usr/bin/env python3
"""One action, both halves, one timeline -- configuration two end to end.

The point of this file is the word *joint*. A touch trajectory and an inertial
window are not two injections that happen to run near each other; they are one
gesture, and a detector's strongest signal is the correlation between them --
the phone tilts because the finger pressed. Playing them independently produces
a recording where the hand moves and the device does not, which is worse than
either half alone.

So both are scheduled against a single measured timebase:

    touch   -> uinput virtual device (kernel input pipeline, SOURCE_TOUCHSCREEN)
    inertia -> android::SensorEventQueue::read, intercepted in the target
    clock   -> both Android clocks read inside the target process

and the target app is unmodified during execution. The packaged collector APK
records the replacement values because the sensor read is rewritten on the way
out.

Two device-specific rules are obeyed here, both learned the hard way:

  * attach before the task starts. SimulatedTaskActivity polls
    monitorCaptureLiveness every 500ms and finishes the run if the main thread
    stalls; frida's attach stalls it briefly, so attaching mid-run makes the app
    abort its own recording and looks exactly like a crash.
  * never the Java bridge. It aborts this ART under load (SIGABRT from the
    agent's own mapping) after reporting itself healthy.

    python methods/on_device/runners/joint_inject.py --bundles-dir BUNDLES --action tap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actreal import background
from actreal import configs
from actreal.bundle import load_bundle
from actreal.device import Adb, probe
from actreal.session import play_action

PACKAGE = "com.sensorworldmodel.collector"
TASK_ENTRY = (540, 1698)


def keyboard_down(adb: Adb) -> bool:
    for _ in range(3):
        shown = any("mInputShown=true" in line
                    for line in adb.shell("dumpsys input_method").text.splitlines())
        if not shown:
            return True
        adb.shell("input keyevent 111")
        time.sleep(1.5)
    return False


def resumed(adb: Adb) -> str:
    for line in adb.shell("dumpsys activity activities").text.splitlines():
        if "topResumedActivity" in line:
            for piece in line.replace("/", " ").split():
                if piece.startswith(".") and piece.endswith("Activity"):
                    return piece
    return "<unknown>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--action", default="tap")
    parser.add_argument("--lead-ms", type=float, default=400.0)
    args = parser.parse_args()

    adb = Adb()
    if not adb.devices():
        print("no device visible to adb", file=sys.stderr)
        return 2

    # Fresh process, sitting on the launcher screen with no run in progress:
    # this is the only safe moment to attach.
    adb.shell(f"am force-stop {PACKAGE}")
    time.sleep(2.0)
    adb.shell("input keyevent 224")
    adb.shell(f"am start -n {PACKAGE}/.MainActivity")
    time.sleep(6.0)

    config = configs.rooted(
        adb, package=PACKAGE, period_ms=10.0,
        device_w=1080, device_h=2424, probe_report=probe(adb),
    )
    touch, imu = config.touch.describe(), config.imu.describe()
    print(f"configuration : {config.name}   reach: {config.reach}")
    print(f"touch         : {touch['backend']}/{touch.get('mode')}")
    print(f"inertia       : {imu['hook_status'].get('backend')} hook, "
          f"third-party capable {imu['third_party_capable']}")
    print(f"clock         : offset {config.timebase.offset_ns} ns, "
          f"trustworthy {config.timebase.trustworthy}")

    # Only now open the task: the hook is already in place, so the stall its
    # installation causes has already happened and no run was lost to it.
    adb.shell(f"input tap {TASK_ENTRY[0]} {TASK_ENTRY[1]}")
    time.sleep(7.0)
    where = resumed(adb)
    print(f"screen        : {where}, keyboard down {keyboard_down(adb)}")
    if where != ".SimulatedTaskActivity":
        print(f"the task did not open ({where}); nothing would be recorded",
              file=sys.stderr)
        return 1

    candidates = sorted(args.bundles_dir.glob(f"{args.action}_*.json"))
    if not candidates:
        print(f"no {args.action} bundles under {args.bundles_dir}", file=sys.stderr)
        return 2
    bundle = load_bundle(candidates[0])

    # Between actions the app must still be receiving something of ours.
    # Injected mode has taken its real sensors away, and a recording that goes
    # quiet whenever nothing is being played is the most conspicuous thing a
    # session can contain -- one measured gap ran to 257 seconds. The idle
    # stream is built from the bundles' own pre-roll, so it is this victim's
    # hand at rest rather than a synthetic hum.
    from actreal.planner import BundleLibrary
    library = BundleLibrary(args.bundles_dir, seed=0)
    report = background.install(config.imu, library, period_ms=10.0)
    print(f"background    : {report.get('frames')} frames from "
          f"{report.get('source', 'bundle pre-roll')}")

    config.imu.set_mode("injected")
    if hasattr(config.touch, "open_device"):
        config.touch.open_device()
    before = config.imu.session.status()["stats"]

    print(f"\nbundle        : {bundle.bundle_id} {bundle.action}, "
          f"{bundle.imu_frames} inertial frames")
    try:
        play_action(
            config, bundle,
            read_uptime_ms=lambda: config.imu.read_clock().uptime_ms,
            lead_ms=args.lead_ms, set_mode=False,
        )
        time.sleep(bundle.imu_duration_ms / 1000.0 + 2.0)
    finally:
        if hasattr(config.touch, "close_device"):
            config.touch.close_device()

    after = config.imu.session.status()
    stats = after["stats"]
    print("\n=== what the hook did ===")
    for key in ("reads", "events", "replaced", "from_window", "from_background",
                "passed", "no_frame", "size_mismatch"):
        moved = stats.get(key, 0) - before.get(key, 0)
        print(f"  {key:<18} {stats.get(key, 0):>8}   (+{moved} during this action)")
    print(f"  rates              {after.get('rates')}")

    replaced = stats.get("replaced", 0) - before.get("replaced", 0)
    if replaced <= 0:
        print("\nno sample was replaced: the window did not line up with any "
              "delivery, or the app is not recording", file=sys.stderr)
        return 1
    print(f"\n{replaced} inertial samples carried the selected profile while the "
          f"touch played through the kernel input pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
