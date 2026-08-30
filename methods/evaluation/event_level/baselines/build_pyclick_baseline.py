#!/usr/bin/env python3
"""Fill every fake touch carrier with a pyclick Bezier bot path."""

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

# The pinned library checkout is vendored under evaluation/third_party.
PYCLICK_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "pyclick"
PATH_ACTIONS = frozenset(("tap", "scroll", "swipe", "pinch"))

_BINDING: dict | None = None


def screen_dimensions(orientation_id: int) -> tuple[float, float]:
    """HMOG Galaxy S4 pixels, matching the repository's own observer."""

    if int(orientation_id) in (1, 3):
        return 1920.0, 1080.0
    return 1080.0, 1920.0


def _event_seed(event_id: str) -> int:
    digest = hashlib.sha256(f"pyclick|{event_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _endpoints(event: dict) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    orientation, start_px, end_px = _BINDING[event["event_id"]]
    width, height = screen_dimensions(orientation)
    carrier = np.asarray(event["trajectory"][:, 1:3], dtype=np.float64)
    if start_px is not None and end_px is not None:
        start = np.asarray(start_px, dtype=np.float64)
        end = np.asarray(end_px, dtype=np.float64)
    else:
        # Pinch is dispatched as an area, not as one pointer's two ends, so its
        # carrier centroid path supplies the endpoints the bot has to join.
        start = carrier[0] * (width, height)
        end = carrier[-1] * (width, height)
    return (
        (int(round(start[0])), int(round(start[1]))),
        (int(round(end[0])), int(round(end[1]))),
        width,
        height,
    )


def _import_human_curve():
    """Import the library's curve generator without its GUI driver."""

    import types

    sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
    from pyclick import HumanCurve

    return HumanCurve


_OWNED = None


def generate_trajectory(event: dict):
    if event["action"] not in PATH_ACTIONS:
        return None
    # A release bundle carries all five actions but publishes only the ones it
    # owns; the rest are declined so no number is produced against a carrier the
    # release does not report from this bundle.
    if _OWNED is not None and event["action"] not in _OWNED:
        return None
    HumanCurve = _import_human_curve()

    start, end, width, height = _endpoints(event)
    seed = _event_seed(event["event_id"])
    random.seed(seed)
    np.random.seed(seed)
    curve = HumanCurve(start, end, targetPoints=int(event["samples"]))
    points = np.asarray(curve.points, dtype=np.float64)
    if len(points) != int(event["samples"]):
        raise RuntimeError("pyclick returned the wrong number of points")
    # The library's knots may sit up to 100 px outside the endpoints' bounding
    # box, so a stroke near an edge can leave the screen.  Shrink it about its
    # start rather than clipping, matching the paper's own placement policy.
    dimensions = np.asarray((width, height))
    fitted, _ = fit_to_screen(points, np.asarray(start, dtype=np.float64), dimensions)
    return fitted / dimensions


def _process(path_and_output: tuple[str, str]) -> tuple[str, str, dict]:
    source, output = path_and_output
    arrays = load_shard(Path(source))
    counts = rewrite_shard_arrays(arrays, trajectory_generator=generate_trajectory)
    digest = save_shard(Path(output), arrays)
    return Path(source).name, digest, counts


def _initialise(binding_path: str, owned: str = "") -> None:
    global _BINDING, _OWNED
    _OWNED = frozenset(a for a in owned.split(",") if a) or None
    sys.path.insert(0, str(PYCLICK_ROOT))
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
            print(f"{name} done", flush=True)

    finalise_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shard_digests=digests,
        counts=totals,
        method_name="pyclick_bezier_human_curve",
        method_detail={
            "repository": "https://github.com/patrikoss/pyclick",
            "entry_point": "pyclick.HumanCurve(fromPoint, toPoint, targetPoints)",
            "library_modified": False,
            "defaults_used": (
                "knotsCount=2, distortionMean=1, distortionStdev=1, "
                "distortionFrequency=0.5, tweening=easeOutQuad"
            ),
            "actions_generated": sorted(PATH_ACTIONS),
            "actions_declined": ["keystroke"],
            "endpoints": "bound target request px, rounded to integer pixels",
            "seed": "sha256('pyclick|' + event_id)[:4]",
        },
        swapped={"trajectory_xy": True, "imu": False},
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
