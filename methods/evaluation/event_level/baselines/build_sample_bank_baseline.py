#!/usr/bin/env python3
"""Fill fake carriers from an unconditional generator's sample bank."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))
from common.hmog_baseline_common import (  # noqa: E402
    ACTIONS,
    finalise_dataset,
    fit_to_screen,
    iter_shards,
    linear_resample,
    load_shard,
    merge_counts,
    rewrite_shard_arrays,
    save_shard,
    zoh_resample,
)

_STATE: dict = {}


# Actions a caller can refuse to generate.  Declining is recorded per action and
# excluded from every summary, because a declined action still carries the
# source method's own attack and must never be read as a baseline result.
def _declined(action: str) -> bool:
    return action in _STATE.get("decline", frozenset())


def screen_dimensions(orientation_id: int) -> tuple[float, float]:
    if int(orientation_id) in (1, 3):
        return 1920.0, 1080.0
    return 1080.0, 1920.0


# The fake events split 70/10/20 by user, so the bank is cut the same way and a
# window drawn for a training fake can never reappear in a test fake.
SPLIT_SHARE = {
    "train": (0.0, 0.7),
    "development": (0.7, 0.8),
    "test": (0.8, 1.0),
}
# Below this ratio of chord length to overall path extent, matching the chord
# would blow the shape up rather than place it.
MINIMUM_CHORD_RATIO = 0.05
# How many bank windows a carrier may look at before keeping the best fitting
# one.  The source method picks its material from a five-shot span ladder; this
# is the same idea with the bank standing in for the ladder.
PLACEMENT_DRAWS = 8
# Outside this band the required scale is not placing a stroke any more.
SCALE_BAND = (1.0 / 3.0, 3.0)


def _draw_index(event_id: str, kind: str, total: int, split: str) -> int:
    low, high = SPLIT_SHARE.get(split, (0.0, 1.0))
    lower, upper = int(low * total), int(high * total)
    if upper - lower < 1:
        lower, upper = 0, total
    digest = hashlib.sha256(f"{kind}|{event_id}".encode()).digest()
    return lower + int.from_bytes(digest[:8], "big") % (upper - lower)


def _bind_to_endpoints(
    path_px: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    first: int,
    last: int,
) -> tuple[np.ndarray, bool]:
    """Move a generated path onto the carrier's endpoints, shape preserved."""

    anchor = path_px[first]
    source_chord = path_px[last] - anchor
    target_chord = end - start
    source_length = float(np.linalg.norm(source_chord))
    target_length = float(np.linalg.norm(target_chord))
    extent = float(np.max(np.linalg.norm(path_px - anchor, axis=1)))
    degenerate = (
        target_length < 1.0e-9
        or source_length < max(1.0e-9, MINIMUM_CHORD_RATIO * extent)
    )
    if degenerate:
        # Either the carrier asks for no displacement (a tap), or the generated
        # path returns so close to where it started that its chord carries no
        # usable direction.  Translation is the only transform defined here.
        return path_px - anchor + start, True
    scale = target_length / source_length
    rotation = float(
        np.arctan2(target_chord[1], target_chord[0])
        - np.arctan2(source_chord[1], source_chord[0])
    )
    cosine, sine = np.cos(rotation), np.sin(rotation)
    matrix = scale * np.asarray(((cosine, -sine), (sine, cosine)))
    placed = (path_px - anchor) @ matrix.T + start
    placed[first] = start
    placed[last] = end
    return placed, False


def generate_trajectory(event: dict):
    if _declined(event["action"]):
        return None
    bank = _STATE["trajectory"].get(event["action"])
    if bank is None:
        return None
    samples = int(event["samples"])
    orientation, start_px, end_px = _STATE["binding"][event["event_id"]]
    width, height = screen_dimensions(orientation)
    dimensions = np.asarray((width, height))
    carrier = np.asarray(event["trajectory"][:, 1:3], dtype=np.float64) * dimensions

    contact = np.flatnonzero(np.asarray(event["trajectory"][:, 0]) > 0.0)
    if len(contact) >= 2:
        first, last = int(contact[0]), int(contact[-1])
    else:
        first, last = 0, samples - 1
    if start_px is not None and end_px is not None:
        start, end = np.asarray(start_px, float), np.asarray(end_px, float)
    else:
        # Pinch and keystroke are not dispatched as one pointer's two ends, so
        # the carrier's own first and last contact positions stand in.
        start, end = carrier[first], carrier[last]
    target_length = float(np.linalg.norm(end - start))

    # Keep the draw whose chord needs the scale closest to 1.  A tap asks for no
    # displacement at all, so every draw is equally good and the first is taken.
    best_path = None
    best_cost = None
    for attempt in range(1 if target_length < 1.0e-9 else PLACEMENT_DRAWS):
        drawn = _draw_index(
            event["event_id"], f"trajectory#{attempt}", len(bank), event["split"]
        )
        candidate = (
            zoh_resample(bank[drawn], samples).astype(np.float64) * dimensions
        )
        source_length = float(
            np.linalg.norm(candidate[last] - candidate[first])
        )
        cost = (
            np.inf
            if source_length < 1.0e-9 or target_length < 1.0e-9
            else abs(np.log(target_length / source_length))
        )
        if best_cost is None or cost < best_cost:
            best_path, best_cost = candidate, cost
        if target_length < 1.0e-9 or cost <= abs(np.log(SCALE_BAND[1])):
            break

    placed, _ = _bind_to_endpoints(best_path, start, end, first, last)
    # Never clip: shrink about the bound start until the whole path is on
    # screen, the same way the paper's own transporter keeps its strokes inside.
    # Only the rows this generator owns constrain the fit -- the rest are
    # restored from the carrier a moment later and must not drag the scale down.
    owned = np.asarray(event["trajectory"][:, 0]) > 0.0
    fitted, _ = fit_to_screen(placed, start, dimensions, considered=owned)
    return fitted / dimensions


def generate_imu(event: dict):
    if _declined(event["action"]):
        return None
    bank = _STATE["imu"].get(event["action"])
    if bank is None:
        return None
    sample = bank[_draw_index(event["event_id"], "imu", len(bank), event["split"])]
    return linear_resample(sample, int(event["samples"]))


def _process(job: tuple[str, str, bool, bool]) -> tuple[str, str, dict]:
    source, output, do_trajectory, do_imu = job
    arrays = load_shard(Path(source))
    counts = rewrite_shard_arrays(
        arrays,
        trajectory_generator=generate_trajectory if do_trajectory else None,
        imu_generator=generate_imu if do_imu else None,
    )
    return Path(source).name, save_shard(Path(output), arrays), counts


def _initialise(banks_path: str, binding_path: str, decline: str) -> None:
    with open(banks_path, "rb") as handle:
        banks = pickle.load(handle)
    with open(binding_path, "rb") as handle:
        _STATE["binding"] = pickle.load(handle)
    _STATE["trajectory"] = banks.get("trajectory", {})
    _STATE["imu"] = banks.get("imu", {})
    _STATE["decline"] = frozenset(a for a in decline.split(",") if a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-json", default="{}")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--decline",
        default="",
        help="comma-separated actions this baseline does not address",
    )
    parser.add_argument(
        "--owned-actions",
        default="",
        help=(
            "comma-separated actions this bundle owns in the release; every "
            "other action is declined, so the build leaves untouched the events "
            "the release does not report from this bundle"
        ),
    )
    args = parser.parse_args()

    # A release bundle carries all five actions' events but only publishes the
    # ones it owns.  Swapping an unowned action would generate a number against
    # a carrier the paper never reports, so everything unowned is declined --
    # which leaves the carrier's own signal in place and is recorded as declined
    # rather than being summarised as a baseline result.
    if args.owned_actions:
        owned = {a for a in args.owned_actions.split(",") if a}
        unknown = owned - set(ACTIONS)
        if unknown:
            raise SystemExit(f"unknown action(s) in --owned-actions: {sorted(unknown)}")
        declined = set(a for a in args.decline.split(",") if a) | (set(ACTIONS) - owned)
        args.decline = ",".join(sorted(declined))

    with open(args.banks, "rb") as handle:
        banks = pickle.load(handle)
    do_trajectory = bool(banks.get("trajectory"))
    do_imu = bool(banks.get("imu"))
    if not do_trajectory and not do_imu:
        raise SystemExit("sample bank is empty")

    (args.output_dir / "shards").mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(path),
            str(args.output_dir / "shards" / path.name),
            do_trajectory,
            do_imu,
        )
        for path in iter_shards(args.source_dir)
    ]
    digests: dict[str, str] = {}
    totals: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise,
        initargs=(str(args.banks), str(args.binding), args.decline),
    ) as pool:
        for name, digest, counts in pool.map(_process, jobs):
            digests[name] = digest
            merge_counts(totals, counts)

    detail = json.loads(args.method_json)
    detail["actions_declined"] = sorted(a for a in args.decline.split(",") if a)
    detail["sample_bank"] = str(args.banks)
    detail["bank_sizes"] = {
        kind: {action: int(len(values)) for action, values in mapping.items()}
        for kind, mapping in banks.items()
    }
    detail["endpoint_binding"] = (
        "single rotation + uniform scale + translation onto the carrier's "
        "requested endpoints; translation only when the request has no chord"
    )
    finalise_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shard_digests=digests,
        counts=totals,
        method_name=args.method_name,
        method_detail=detail,
        swapped={"trajectory_xy": do_trajectory, "imu": do_imu},
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
