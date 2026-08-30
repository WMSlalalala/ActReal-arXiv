#!/usr/bin/env python3
"""Pull every session the campaign recorded, one at a time, through the app.

The study app writes into internal storage, which nothing on the device can read
without root, and exports only the session that is currently open. A campaign
that opens nine sessions therefore leaves eight of them unreachable through the
UI by the time it finishes -- and this project has already lost a victim's runs
that way once.

Root changes that. The sessions directory is readable directly now, so this
copies them out wholesale instead of driving nine export dialogues, and falls
back to the export flow only if the directory cannot be read.

    python methods/on_device/runners/export_all.py --out OUTPUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parents[1] / ".actreal-work"
sys.path.insert(0, str(ROOT))

from actreal.device import Adb

PACKAGE = "com.sensorworldmodel.collector"
# Where the app keeps them. Read through su, because the app's data directory is
# not world-readable and this is the whole reason the export dialogue existed.
REMOTE = f"/data/data/{PACKAGE}/files/sessions"
STAGING = "/data/local/tmp/actreal_sessions"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "phone_sessions")
    parser.add_argument("--remote", default=REMOTE)
    args = parser.parse_args()

    adb = Adb()
    if not adb.devices():
        print("no device visible to adb", file=sys.stderr)
        return 2

    listing = adb.shell(f'su -c "ls -1 {args.remote} 2>/dev/null"').text
    names = [line.strip() for line in listing.splitlines() if line.strip()]
    if not names:
        # Some Android builds place app-specific exports in external storage.
        alt = f"/sdcard/Android/data/{PACKAGE}/files/sessions"
        listing = adb.shell(f'su -c "ls -1 {alt} 2>/dev/null"').text
        names = [line.strip() for line in listing.splitlines() if line.strip()]
        if names:
            args.remote = alt
    if not names:
        print(f"no sessions under {args.remote}", file=sys.stderr)
        return 1
    print(f"{len(names)} sessions on the device")

    # Copy into a world-readable staging directory first: adb pull runs as the
    # shell user and cannot reach into the app's private storage, so pulling
    # straight from there produces zero bytes and no error.
    adb.shell(f'su -c "rm -rf {STAGING}; mkdir -p {STAGING}"')
    adb.shell(f'su -c "cp -r {args.remote}/. {STAGING}/"')
    adb.shell(f'su -c "chmod -R 777 {STAGING}"')

    args.out.mkdir(parents=True, exist_ok=True)
    pulled: list[dict] = []
    for name in names:
        target = args.out / name
        adb.run("pull", f"{STAGING}/{name}", str(target))
        csvs = sorted(p.name for p in target.glob("*.csv")) if target.is_dir() else []
        size = sum(p.stat().st_size for p in target.rglob("*")) if target.is_dir() else 0
        pulled.append({"session": name, "csv": len(csvs), "bytes": size})
        print(f"  {name:<40} {len(csvs)} csv  {size / 1048576:.1f} MB")

    adb.shell(f'su -c "rm -rf {STAGING}"')
    report = args.out / "pulled.json"
    report.write_text(json.dumps(pulled, indent=2), encoding="utf-8")

    empty = [p for p in pulled if p["csv"] == 0]
    print(f"\n{len(pulled)} pulled, {sum(p['bytes'] for p in pulled) / 1048576:.0f} MB total")
    if empty:
        print(f"{len(empty)} came out with no CSV at all: "
              f"{[p['session'] for p in empty]}", file=sys.stderr)
        return 1
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
