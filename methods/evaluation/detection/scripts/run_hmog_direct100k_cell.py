#!/usr/bin/env python3
"""Train one non-formal detector cell from the direct 100k dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from detection.core.event_detectors import REGISTERED_DETECTORS
from detection.core.event_pad import ACTIONS, EventPadError
from detection.core.formal_event_pad import (
    ALL_DETECTOR_PROTOCOL,
    SCOPES,
    run_formal_event_cell,
)


DIRECT_MODALITIES = (
    "trajectory_xytime",
    "imu_only",
    "imu_trajectory_xytime",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--modality", choices=DIRECT_MODALITIES, required=True)
    parser.add_argument("--detector", choices=REGISTERED_DETECTORS, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_formal_event_cell(
        manifest=args.manifest,
        output_dir=args.output_dir,
        scope=args.scope,
        action=args.action,
        modality=args.modality,
        detector=args.detector,
        device_name=args.device,
        epochs=args.epochs,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        formal=False,
        detector_protocol=ALL_DETECTOR_PROTOCOL,
        qualification_receipt=None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except EventPadError as exc:
        print(f"DIRECT DETECTOR CELL ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
