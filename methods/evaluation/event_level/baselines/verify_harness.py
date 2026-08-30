#!/usr/bin/env python3
"""Prove the swap harness changes what it is asked to and nothing else."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))
from common.hmog_baseline_common import (  # noqa: E402
    iter_shards,
    load_shard,
    rewrite_shard_arrays,
)

UNCLAIMED_TRAJECTORY_COLUMNS = (0, 3, 4, 7, 8)
STRUCTURE_KEYS = (
    "offsets",
    "label",
    "action",
    "user_id",
    "session_id",
    "event_id",
    "split",
    "sample_idx",
    "cross_modal_pair_id",
    "source_cluster_id",
)


def _fingerprint(event_id: str) -> float:
    """A value unique to one event, in the (0, 1) range coordinates live in."""

    digest = hashlib.sha256(event_id.encode()).digest()
    return 0.05 + 0.9 * (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF)


def check_routing(source_dir: Path, shards: int) -> list[str]:
    """Every generated value must land in the rows of the event that asked."""

    failures = []
    for path in list(iter_shards(source_dir))[:shards]:
        original = load_shard(path)
        rewritten = load_shard(path)
        rewrite_shard_arrays(
            rewritten,
            trajectory_generator=lambda event: np.full(
                (event["samples"], 2), _fingerprint(event["event_id"]), np.float32
            ),
            imu_generator=lambda event: np.full(
                (event["samples"], 6), _fingerprint(event["event_id"]), np.float32
            ),
        )
        offsets = original["offsets"]
        for index in range(len(original["label"])):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            event_id = str(original["event_id"][index])
            genuine = int(original["label"][index]) == 0
            imu_rows = rewritten["imu_flat"][start:stop]
            if genuine:
                if not np.array_equal(imu_rows, original["imu_flat"][start:stop]):
                    failures.append(f"{path.name}: genuine {event_id} was written")
                continue
            expected = np.float32(_fingerprint(event_id))
            if not np.allclose(imu_rows, expected, atol=1e-6):
                failures.append(
                    f"{path.name}: {event_id} received another event's IMU payload"
                )
            # The trajectory path restores off-contact rows and held rows from
            # the carrier, so only the rows it is free to write are checked.
            carrier = original["trajectory_flat"][start:stop]
            free = carrier[:, 0] > 0.0
            if len(carrier) > 1:
                free[1:] &= ~np.all(np.diff(carrier[:, 1:3], axis=0) == 0.0, axis=1)
            written = rewritten["trajectory_flat"][start:stop][free][:, 1:3]
            if len(written) and not np.allclose(written, expected, atol=1e-6):
                failures.append(
                    f"{path.name}: {event_id} received another event's trajectory"
                )
    return failures


def check_against_source(source_dir: Path, built_dir: Path, kind: str) -> list[str]:
    failures = []
    for path in iter_shards(source_dir):
        built_path = built_dir / "shards" / path.name
        if not built_path.is_file():
            failures.append(f"{path.name}: missing from {built_dir}")
            continue
        source = load_shard(path)
        built = load_shard(built_path)
        for key in STRUCTURE_KEYS:
            if key in source and not np.array_equal(source[key], built[key]):
                failures.append(f"{path.name}: {key} differs")
        offsets = source["offsets"]
        genuine = np.flatnonzero(source["label"] == 0)
        fake = np.flatnonzero(source["label"] == 1)
        rows = np.zeros(len(source["trajectory_flat"]), dtype=bool)
        for index in genuine:
            rows[int(offsets[index]) : int(offsets[index + 1])] = True
        for key in ("trajectory_flat", "imu_flat"):
            if not np.array_equal(source[key][rows], built[key][rows]):
                failures.append(f"{path.name}: genuine rows changed in {key}")
        fake_rows = ~rows
        if kind in ("trajectory", "both"):
            source_unclaimed = source["trajectory_flat"][fake_rows][
                :, UNCLAIMED_TRAJECTORY_COLUMNS
            ]
            built_unclaimed = built["trajectory_flat"][fake_rows][
                :, UNCLAIMED_TRAJECTORY_COLUMNS
            ]
            if not np.array_equal(source_unclaimed, built_unclaimed):
                failures.append(f"{path.name}: a trajectory swap moved a column "
                                f"it does not own")
            if kind == "trajectory" and not np.array_equal(
                source["imu_flat"][fake_rows], built["imu_flat"][fake_rows]
            ):
                failures.append(f"{path.name}: a trajectory swap changed IMU")
        if kind == "imu":
            if not np.array_equal(
                source["trajectory_flat"][fake_rows],
                built["trajectory_flat"][fake_rows],
            ):
                failures.append(f"{path.name}: an IMU swap changed the trajectory")
        if kind in ("trajectory", "both"):
            # A row with no contact carries an exact (0, 0) in this dataset, and
            # ~80% of keystroke rows are such rows.  A generated path written
            # over them would be a tell no amount of model quality could undo,
            # so the restore has to be provable, not assumed.
            off_contact = source["trajectory_flat"][:, 0] <= 0.0
            if off_contact.any() and not np.array_equal(
                built["trajectory_flat"][off_contact][:, 1:3],
                source["trajectory_flat"][off_contact][:, 1:3],
            ):
                failures.append(
                    f"{path.name}: a no-contact row lost the carrier's (0, 0)"
                )
        if not np.isfinite(built["trajectory_flat"]).all():
            failures.append(f"{path.name}: non-finite trajectory")
        if not np.isfinite(built["imu_flat"]).all():
            failures.append(f"{path.name}: non-finite IMU")
        coordinates = built["trajectory_flat"][:, 1:3]
        if coordinates.min() < 0.0 or coordinates.max() > 1.0:
            failures.append(f"{path.name}: coordinate outside the screen")
    return failures


def check_swap_happened(source_dir: Path, built_dir: Path, kind: str) -> list[str]:
    """A dataset identical to its source would pass every other check."""

    failures = []
    counts = json.loads((built_dir / "baseline_counts.json").read_text())
    per_action = counts.get("per_action", {})
    observed = {"trajectory": 0, "imu": 0}
    # A carrier with one or two freely-observed rows leaves the generator
    # nothing to choose: the first is the anchor and the last is the bound
    determined = 0
    for path in iter_shards(source_dir):
        source = load_shard(path)
        built = load_shard(built_dir / "shards" / path.name)
        offsets = source["offsets"]
        for index in np.flatnonzero(source["label"] == 1):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            carrier = source["trajectory_flat"][start:stop]
            free = carrier[:, 0] > 0.0
            if len(carrier) > 1:
                free[1:] &= ~np.all(np.diff(carrier[:, 1:3], axis=0) == 0.0, axis=1)
            if int(free.sum()) <= 2:
                determined += 1
            if not np.array_equal(
                carrier[:, 1:3], built["trajectory_flat"][start:stop, 1:3]
            ):
                observed["trajectory"] += 1
            if not np.array_equal(
                source["imu_flat"][start:stop], built["imu_flat"][start:stop]
            ):
                observed["imu"] += 1
    print(
        f"    fake carriers whose observed path is fully determined by the "
        f"anchor and the bound target (<=2 free rows): {determined}"
    )
    for channel in ("trajectory", "imu"):
        claimed = int(counts.get(f"{channel}_swapped", 0))
        if claimed == 0:
            continue
        expected = claimed - (determined if channel == "trajectory" else 0)
        if observed[channel] < 0.99 * expected:
            failures.append(
                f"{built_dir.name}: claims {claimed} {channel} swaps and at most "
                f"{determined} are target-determined, so at least {expected:.0f} "
                f"events should differ, but only {observed[channel]} do"
            )
    declined = {
        action: tally
        for action, tally in per_action.items()
        if tally.get("trajectory_declined") or tally.get("imu_declined")
    }
    if declined:
        print(
            f"    declined actions (carry the source method's own signal, "
            f"must not be read as baseline results): {sorted(declined)}"
        )
    return failures


def check_clock_fingerprint(built_dir: Path, shards: int = 12) -> list[str]:
    """Refuse a dataset whose fake events wear a giveaway clock."""

    failures = []
    tallies: dict = {}
    for path in list(iter_shards(built_dir))[:shards]:
        arrays = load_shard(path)
        offsets = arrays["offsets"]
        trajectory = arrays["trajectory_flat"]
        for index in range(len(arrays["label"])):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            count = stop - start
            if count < 2:
                continue
            ideal = np.linspace(0.0, (count - 1) * 10.0 / 1000.0, count, dtype=np.float32)
            exact = np.array_equal(
                trajectory[start:stop, 7].view(np.int32), ideal.view(np.int32)
            )
            key = str(arrays["action"][index])
            tally = tallies.setdefault(key, {"fake": [0, 0], "genuine": [0, 0]})
            side = tally["fake" if int(arrays["label"][index]) == 1 else "genuine"]
            side[1] += 1
            side[0] += int(exact)

    for action, tally in sorted(tallies.items()):
        fake_hits, fake_total = tally["fake"]
        genuine_hits, genuine_total = tally["genuine"]
        if fake_total < 50 or genuine_total < 50:
            continue
        fake_rate = fake_hits / fake_total
        genuine_rate = genuine_hits / genuine_total
        if fake_rate > 0.5 and fake_rate > genuine_rate * 3:
            failures.append(
                f"{built_dir.name}/{action}: {100 * fake_rate:.1f}% of fake events "
                f"have a bit-exact decimal clock against {100 * genuine_rate:.1f}% "
                "of genuine -- the carrier stamps elapsed deterministically and a "
                "tree detector can read that instead of the attack"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--built-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--kind", action="append", default=[], choices=("trajectory", "imu", "both")
    )
    parser.add_argument("--routing-shards", type=int, default=5)
    args = parser.parse_args()

    if len(args.built_dir) != len(args.kind):
        raise SystemExit(
            f"{len(args.built_dir)} --built-dir but {len(args.kind)} --kind: "
            "each dataset needs its own kind or it would go unchecked"
        )

    failures = check_routing(args.source_dir, args.routing_shards)
    print(
        f"[1] payload routing over {args.routing_shards} shards: "
        f"{'PASS' if not failures else 'FAIL'}"
    )
    for built, kind in zip(args.built_dir, args.kind):
        problems = check_against_source(args.source_dir, built, kind)
        print(f"[2,3] {built.name} ({kind}) untouched-invariants: "
              f"{'PASS' if not problems else 'FAIL'}")
        swap_problems = check_swap_happened(args.source_dir, built, kind)
        print(f"[4] {built.name} ({kind}) swap really happened: "
              f"{'PASS' if not swap_problems else 'FAIL'}")
        clock_problems = check_clock_fingerprint(built)
        print(f"[5] {built.name} carrier clock is not a giveaway: "
              f"{'PASS' if not clock_problems else 'FAIL'}")
        failures.extend(problems + swap_problems + clock_problems)

    if failures:
        print("\n".join(failures[:40]))
        raise SystemExit(f"{len(failures)} failures")
    print("all checks passed")


if __name__ == "__main__":
    main()
