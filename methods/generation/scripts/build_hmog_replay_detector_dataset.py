#!/usr/bin/env python3
"""Build a non-destructive raw-replay detector dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.replay_dataset_builder import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_PATH,
    ReplayDatasetBuildError,
    build_full_100k_replay_dataset,
    build_smoke_replay_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace direct100k trajectories with common-observer genuine raw "
            "touch plus mode-specific replay/generation. The input is never "
            "modified."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-release", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-genuine-manifest", type=Path)
    parser.add_argument("--dispatch-quality-ledger", type=Path)
    parser.add_argument("--raw-trajectory-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument(
        "--conditional-touch-model",
        type=Path,
        help=(
            "Frozen train-fitted conditional tap/scroll/swipe model; required "
            "for --smoke-small and not used by --full-100k."
        ),
    )
    parser.add_argument(
        "--conditional-touch-request-model",
        type=Path,
        help=(
            "Optional train-fitted tap full-pair and scroll/swipe endpoint "
            "request model for smoke mode."
        ),
    )
    parser.add_argument(
        "--smoke-reference-manifest",
        type=Path,
        help=(
            "Optional prior smoke event manifest whose exact split/user/action/"
            "label event IDs are reused; forbidden for --full-100k."
        ),
    )
    parser.add_argument("--users-per-split", type=int, default=2)
    parser.add_argument("--events-per-label-action-user", type=int, default=2)
    parser.add_argument(
        "--fiveshot-material-dir",
        type=Path,
        required=True,
        help=(
            "frozen five-shot attack material; every fake touch signal is a "
            "transform of the named victim's own five real events"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel shard rebuilds for --full-100k; 0 uses every core but "
            "two, 1 forces the single-process path. Each input shard is one "
            "victim and is rebuilt independently, so the result does not "
            "depend on this."
        ),
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--smoke-small",
        action="store_true",
        help="Build the small diagnostic selection.",
    )
    modes.add_argument(
        "--full-100k",
        action="store_true",
        help=(
            "Preserve every input event and all 100,000 fake events after exact "
            "no-replacement replay-capacity preflight."
        ),
    )
    args = parser.parse_args()
    if args.full_100k and args.smoke_reference_manifest is not None:
        parser.error(
            "--smoke-reference-manifest is only valid with --smoke-small"
        )
    common = {
        "input_manifest": args.input_manifest,
        "input_release": args.input_release,
        "output_dir": args.output_dir,
        "native_genuine_manifest": args.native_genuine_manifest,
        "dispatch_quality_ledger": args.dispatch_quality_ledger,
        "raw_trajectory_root": args.raw_trajectory_root,
        "split_path": args.split_json,
        "fiveshot_material_dir": args.fiveshot_material_dir,
        "seed": args.seed,
    }
    if args.full_100k:
        release = build_full_100k_replay_dataset(
            **common, workers=args.workers or None
        )
    else:
        release = build_smoke_replay_dataset(
            **common,
            conditional_touch_model=args.conditional_touch_model,
            conditional_touch_request_model=(
                args.conditional_touch_request_model
            ),
            smoke_reference_manifest=args.smoke_reference_manifest,
            users_per_split=args.users_per_split,
            events_per_label_action_user=args.events_per_label_action_user,
        )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ReplayDatasetBuildError as exc:
        print(f"REPLAY DATASET BUILD ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
