#!/usr/bin/env python3
"""Build one baseline against the frozen release, bundle by bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from release_registry import bundles  # noqa: E402

PYTHON = "python3"
REPO = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = Path(os.environ.get(
    "ACTREAL_BASELINE_WORK_ROOT", REPO / ".actreal-work" / "baselines"
))


def already_built(directory: Path) -> bool:
    # baseline_counts.json is written last by finalise_dataset, so its presence
    # is the completion marker -- a half-built directory does not have it.
    return (directory / "baseline_counts.json").is_file()


def run(command: list, log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        help="output name, e.g. diffts_imu or csdi_unconditional")
    parser.add_argument(
        "--builder",
        choices=("sample_bank", "pyclick", "ghostcursor"),
        required=True,
    )
    parser.add_argument("--banks", type=Path, default=None,
                        help="sample bank pickle, for --builder sample_bank")
    parser.add_argument("--binding", type=Path, default=None)
    parser.add_argument("--method-json", default="{}")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--kind", choices=("trajectory", "imu", "both"), default="imu",
                        help="which channels this baseline claims, for the verifier")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    root = OUTPUT_ROOT / args.method
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"method": args.method, "builder": args.builder, "bundles": {}}
    failures = []
    declined = []

    for name, source, owned in bundles():
        output = root / name
        if already_built(output):
            print(f"  {name}: already built", flush=True)
        else:
            print(f"  {name}: building (owns {', '.join(owned)})", flush=True)
            if args.builder == "sample_bank":
                command = [
                    PYTHON, str(BASE / "build_sample_bank_baseline.py"),
                    "--source-dir", str(source), "--output-dir", str(output),
                    "--banks", str(args.banks), "--binding", str(args.binding),
                    "--method-name", args.method, "--method-json", args.method_json,
                    "--owned-actions", ",".join(owned), "--workers", str(args.workers),
                ]
            else:
                # The two cursor-path baselines take the same arguments; each
                # declines keystroke itself, since neither models a path with no
                # travel between key presses.
                script = {
                    "pyclick": "build_pyclick_baseline.py",
                    "ghostcursor": "build_ghostcursor_baseline.py",
                }[args.builder]
                command = [
                    PYTHON, str(BASE / script),
                    "--source-dir", str(source), "--output-dir", str(output),
                    "--binding", str(args.binding),
                    "--owned-actions", ",".join(owned), "--workers", str(args.workers),
                ]
            if not run(command, root / f"build_{name}.log"):
                failures.append(f"build {name}")
                print(f"  {name}: BUILD FAILED -- see {root / f'build_{name}.log'}",
                      flush=True)
                continue

        if not args.skip_verify:
            verified = run(
                [
                    PYTHON, str(BASE / "verify_harness.py"),
                    "--source-dir", str(source), "--built-dir", str(output),
                    "--kind", args.kind, "--routing-shards", "3",
                ],
                root / f"verify_{name}.log",
            )
            if not verified:
                failures.append(f"verify {name}")
                print(f"  {name}: VERIFY FAILED -- see {root / f'verify_{name}.log'}",
                      flush=True)

        counts_path = output / "baseline_counts.json"
        entry = {"source": str(source), "owned_actions": list(owned),
                 "output": str(output)}
        if counts_path.is_file():
            counts = json.loads(counts_path.read_text())
            per_action = counts.get("per_action", {})
            # Record what was actually swapped per action, so a summariser can
            # refuse to report an action this bundle only carried along.
            entry["swapped"] = {
                action: {
                    "fake": tally.get("fake", 0),
                    "imu_swapped": tally.get("imu_swapped", 0),
                    "trajectory_swapped": tally.get("trajectory_swapped", 0),
                }
                for action, tally in sorted(per_action.items())
            }
            for action in owned:
                tally = per_action.get(action, {})
                swapped = tally.get("imu_swapped", 0) + tally.get("trajectory_swapped", 0)
                if not tally.get("fake", 0) or swapped:
                    continue
                # An action the bundle owns but this baseline swapped nothing
                # for.  That is not a failure -- a method that does not model an
                # action should decline it rather than invent something its
                # authors never proposed -- but it must never be summarised as a
                # result either, so it is recorded here and the summariser
                # refuses to report it.
                declined.append(f"{name}/{action}")
                print(f"  {name}: declines {action} ({tally['fake']} fake events "
                      "left as the carrier's)", flush=True)
        manifest["bundles"][name] = entry

    manifest["failures"] = failures
    manifest["declined_by_method"] = declined
    (root / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwrote {root / 'bundle_manifest.json'}")
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(len(failures))
    if declined:
        print(f"declined by this method (never summarised as results): {', '.join(declined)}")
    print("all bundles built and verified")


if __name__ == "__main__":
    main()
