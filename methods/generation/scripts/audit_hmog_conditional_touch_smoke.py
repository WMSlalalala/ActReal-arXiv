#!/usr/bin/env python3
"""Audit a conditional-touch smoke dataset before detector training.

Hard failures (event/binding drift, six-axis or time-axis drift, endpoint/OOB,
or exact-pressure violations) exit 1.  Shape distributions are deliberately
report-only and must still be followed by the registered detector gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.conditional_touch_smoke_audit import (
    ConditionalTouchSmokeAuditError,
    audit_conditional_touch_smoke,
    write_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed v2 conditional-touch smoke precheck. Distribution "
            "summaries are informational and do not replace detector gates."
        )
    )
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-release",
        type=Path,
        help="optional release.json binding the baseline manifest/provenance",
    )
    parser.add_argument(
        "--candidate-release",
        type=Path,
        help="optional release.json binding the candidate manifest/provenance",
    )
    parser.add_argument(
        "--baseline-provenance",
        type=Path,
        help="optional baseline provenance.jsonl (improves orientation metadata)",
    )
    parser.add_argument(
        "--candidate-provenance",
        type=Path,
        help=(
            "optional candidate provenance.jsonl; when supplied, raw requested/"
            "generated endpoints become hard invariants"
        ),
    )
    parser.add_argument(
        "--expected-events",
        type=int,
        default=1500,
        help="exact event count required in each dataset (default: 1500)",
    )
    parser.add_argument(
        "--detector-endpoint-tolerance-px",
        type=float,
        default=1.0e-4,
        help=(
            "float32 detector-grid endpoint tolerance in physical pixels; raw "
            "provenance endpoints must remain exactly equal (default: 1e-4)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write deterministic JSON here; stdout is used when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = audit_conditional_touch_smoke(
            baseline_manifest=args.baseline_manifest,
            candidate_manifest=args.candidate_manifest,
            expected_events=args.expected_events,
            baseline_release=args.baseline_release,
            candidate_release=args.candidate_release,
            baseline_provenance_path=args.baseline_provenance,
            candidate_provenance_path=args.candidate_provenance,
            detector_endpoint_tolerance_px=args.detector_endpoint_tolerance_px,
        )
    except ConditionalTouchSmokeAuditError as exc:
        print(f"audit input error: {exc}", file=sys.stderr)
        return 2
    if args.output is not None:
        write_report(report, args.output)
        print(args.output.resolve())
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
