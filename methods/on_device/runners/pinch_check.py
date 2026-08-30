#!/usr/bin/env python3
"""Pinch the article chart and see whether it actually zooms.

The Google round asks for a pinch on the chart, the agent aimed squarely at it,
and the chart did not move -- twice, both scored failures. The panel's own
gesture handling turned out to claim the touch stream only from the *second*
finger, by which point the enclosing ScrollView had already decided the gesture
was a scroll and kept it. That is now claimed from the first contact instead,
and this drives the same path the agent does so the change is checked rather
than assumed.

    python methods/on_device/runners/pinch_check.py --bundles-dir BUNDLES --at 540,2230
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parents[1] / ".actreal-work"
sys.path.insert(0, str(ROOT))

from actreal import background, configs  # noqa: E402
from actreal.device import Adb  # noqa: E402
from actreal.mapping import ScreenMapping  # noqa: E402
from actreal.planner import ActionPlanner, BundleLibrary  # noqa: E402
from actreal.session import play_action  # noqa: E402
from runners.joint_inject import probe, resumed  # noqa: E402

PACKAGE = "com.sensorworldmodel.collector"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--at", default="540,2230", help="pinch centre, x,y")
    parser.add_argument("--direction", default="out", choices=("in", "out"))
    parser.add_argument("--shot", type=Path, default=WORK_ROOT / "debug" / "pinch_check.png")
    parser.add_argument("--before", type=Path, default=WORK_ROOT / "debug" / "pinch_before.png")
    args = parser.parse_args()

    x, y = (float(v) for v in args.at.split(","))
    adb = Adb()
    if not adb.devices():
        print("no device visible to adb", file=sys.stderr)
        return 2
    where = resumed(adb)
    print(f"screen        : {where}")
    if where != ".SimulatedTaskActivity":
        print("open the task first; this only plays the gesture", file=sys.stderr)
        return 3

    config = configs.rooted(adb, package=PACKAGE, period_ms=10.0,
                            device_w=1080, device_h=2424, probe_report=probe(adb))
    library = BundleLibrary(args.bundles_dir, seed=0)
    planner = ActionPlanner(library, ScreenMapping.isotropic(device_w=1080, device_h=2424))
    background.install(config.imu, library, period_ms=10.0)
    config.imu.set_mode("injected")
    if hasattr(config.touch, "open_device"):
        config.touch.open_device()

    args.before.parent.mkdir(parents=True, exist_ok=True)
    adb.shell("screencap -p /sdcard/pinch_before.png")
    adb.run("pull", "/sdcard/pinch_before.png", str(args.before))

    plan = planner.plan("pinch", (x, y), duration_ms=None) if args.direction == "out" \
        else planner.plan("pinch", (x, y))
    bundle = plan.bundle
    print(f"bundle        : {bundle.bundle_id} {bundle.action}, "
          f"{len(bundle.touch.points)} contacts points")
    try:
        play_action(config, bundle,
                    read_uptime_ms=lambda: config.imu.read_clock().uptime_ms,
                    lead_ms=400.0, set_mode=False)
        time.sleep(bundle.imu_duration_ms / 1000.0 + 1.5)
    finally:
        if hasattr(config.touch, "close_device"):
            config.touch.close_device()

    adb.shell("screencap -p /sdcard/pinch_after.png")
    adb.run("pull", "/sdcard/pinch_after.png", str(args.shot))
    adb.shell("rm -f /sdcard/pinch_before.png /sdcard/pinch_after.png")
    print(f"before        : {args.before}")
    print(f"after         : {args.shot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
