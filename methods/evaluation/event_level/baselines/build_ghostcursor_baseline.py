#!/usr/bin/env python3
"""Fill every fake touch carrier with a ghost-cursor pointer path."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))
from common.hmog_baseline_common import (  # noqa: E402
    finalise_dataset,
    fit_to_screen,
    iter_shards,
    load_shard,
    merge_counts,
    rewrite_shard_arrays,
    save_shard,
)

PATH_ACTIONS = frozenset(("tap", "scroll", "swipe", "pinch"))
_BINDING: dict | None = None


def screen_dimensions(orientation_id: int) -> tuple[float, float]:
    if int(orientation_id) in (1, 3):
        return 1920.0, 1080.0
    return 1080.0, 1920.0


def _resample_to(points: np.ndarray, samples: int) -> np.ndarray:
    """Put the library's own sample count on the carrier's row count."""

    if len(points) == samples:
        return points
    source = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, samples)
    return np.column_stack(
        [np.interp(target, source, points[:, axis]) for axis in range(2)]
    )


_OWNED = None


def generate_trajectory(event: dict):
    if event["action"] not in PATH_ACTIONS:
        return None
    # A release bundle carries all five actions but publishes only the ones it
    # owns; the rest are declined so no number is produced against a carrier the
    # release does not report from this bundle.
    if _OWNED is not None and event["action"] not in _OWNED:
        return None
    import ghost_cursor_path

    orientation, start_px, end_px = _BINDING[event["event_id"]]
    width, height = screen_dimensions(orientation)
    dimensions = np.asarray((width, height))
    carrier = np.asarray(event["trajectory"][:, 1:3], dtype=np.float64) * dimensions
    if start_px is not None and end_px is not None:
        start, end = np.asarray(start_px, float), np.asarray(end_px, float)
    else:
        start, end = carrier[0], carrier[-1]

    digest = hashlib.sha256(f"ghostcursor|{event['event_id']}".encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    random.seed(seed)
    np.random.seed(seed)

    points = ghost_cursor_path.move(start, end)
    placed = _resample_to(np.asarray(points, dtype=np.float64), int(event["samples"]))
    # The library clamps at zero but knows nothing of the far edges, and its
    # anchors can reach past them; shrink about the bound start rather than clip.
    fitted, _ = fit_to_screen(placed, start, dimensions)
    fitted[0] = start
    fitted[-1] = end
    return fitted / dimensions


def _process(job):
    source, output = job
    arrays = load_shard(Path(source))
    counts = rewrite_shard_arrays(arrays, trajectory_generator=generate_trajectory)
    return Path(source).name, save_shard(Path(output), arrays), counts


def _initialise(binding_path: str, owned: str = "") -> None:
    global _BINDING, _OWNED
    _OWNED = frozenset(a for a in owned.split(",") if a) or None
    with open(binding_path, "rb") as handle:
        _BINDING = pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--owned-actions",
        default="",
        help=(
            "comma-separated actions this release bundle owns; every other "
            "action is declined"
        ),
    )
    args = parser.parse_args()

    (args.output_dir / "shards").mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(path), str(args.output_dir / "shards" / path.name))
        for path in iter_shards(args.source_dir)
    ]
    digests: dict[str, str] = {}
    totals: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise,
        initargs=(str(args.binding), args.owned_actions),
    ) as pool:
        for name, digest, counts in pool.map(_process, jobs):
            digests[name] = digest
            merge_counts(totals, counts)

    finalise_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shard_digests=digests,
        counts=totals,
        method_name="ghost_cursor_fitts_overshoot",
        method_detail={
            "repository": "https://github.com/Xetera/ghost-cursor",
            "python_port": "python-ghost-cursor 0.1.1 (pyppeteer_ghost_cursor)",
            "entry_point": "ghost_cursor_path.move(start, end)",
            "reproduces": "spoof.ts path() and the move() overshoot sequence",
            "port_deviations_corrected": [
                "arc-length sampling restored in place of the port's parameter "
                "spacing (measured 1.20x velocity distortion at 12 samples)",
                "spreadOverride honoured, which the port drops and the "
                "overshoot correction depends on",
            ],
            "constants": {
                "overshoot_threshold_px": 500,
                "overshoot_radius_px": 120,
                "overshoot_spread": 10,
                "default_width_px": 100,
                "min_steps": 25,
            },
            "actions_generated": sorted(PATH_ACTIONS),
            "actions_declined": ["keystroke"],
            "victim_events_used": 0,
            "training": "none",
        },
        swapped={"trajectory_xy": True, "imu": False},
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
