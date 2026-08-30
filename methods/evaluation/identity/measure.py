#!/usr/bin/env python3
"""Recompute the generated-event target-user distances in Appendix E.2.

The required HMOG-derived event shards are licensed inputs and are not
redistributed with this artifact. Point ``--event-data-root`` (or the
``ACTREAL_EVENT_DATA_ROOT`` environment variable) at a local directory whose
bundle subdirectories contain ``shards/hmog_u*.npz``. Only the distance columns
shown in the paper are written; no auxiliary identity or detector metrics are
exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
METHODS = REPO / "methods"
if str(METHODS) not in sys.path:
    sys.path.insert(0, str(METHODS))

from evaluation.detection.core.event_detectors import extract_event_features  # noqa: E402


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
FEATURE_SPACES = {
    "hmog": "hmog_style_svm",
    "ttos": "paper_svm",
}


def load_bundle_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        action: payload["actions"][action]["imu_only"]
        for action in ACTIONS
    }


def bundle_shards(event_data_root: Path, bundle: str) -> Path:
    candidates = (
        event_data_root / bundle / "shards",
        event_data_root / bundle,
    )
    for candidate in candidates:
        if any(candidate.glob("hmog_u*.npz")):
            return candidate
    raise FileNotFoundError(
        f"no hmog_u*.npz shards found for bundle {bundle!r} under "
        f"{event_data_root}"
    )


def load_action_events(
    shard_dir: Path,
    action: str,
    split: str,
) -> dict[str, dict[str, list]]:
    per_user: dict[str, dict[str, list]] = {}
    for shard in sorted(shard_dir.glob("hmog_u*.npz")):
        with np.load(shard, allow_pickle=True) as store:
            if str(store["split"]) != split:
                continue
            offsets = store["offsets"]
            for index, label in enumerate(store["label"]):
                if str(store["action"][index]) != action:
                    continue
                user = str(store["user_id"][index])
                side = "genuine" if int(label) == 0 else "fake"
                bucket = per_user.setdefault(
                    user, {"genuine": [], "fake": []}
                )
                bucket[side].append(
                    (
                        store["imu_flat"][offsets[index] : offsets[index + 1]],
                        store["trajectory_flat"][
                            offsets[index] : offsets[index + 1]
                        ],
                    )
                )
    return per_user


def featurize(events: list, detector: str) -> np.ndarray:
    return np.asarray(
        [
            extract_event_features(detector, "imu_only", imu, trajectory)
            for imu, trajectory in events
        ],
        dtype=np.float64,
    )


def pair_distances(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(
        left[:, None, :] - right[None, :, :], axis=2
    ).reshape(-1)


def compute_action(
    per_user: dict[str, dict[str, list]],
    detector: str,
    max_per_user: int,
    expected_users: int,
) -> tuple[float, float]:
    genuine: dict[str, np.ndarray] = {}
    generated: dict[str, np.ndarray] = {}
    for user in sorted(per_user):
        genuine_events = per_user[user]["genuine"][:max_per_user]
        generated_events = per_user[user]["fake"][:max_per_user]
        if len(genuine_events) < 2 or len(generated_events) < 2:
            continue
        genuine[user] = featurize(genuine_events, detector)
        generated[user] = featurize(generated_events, detector)

    users = sorted(genuine)
    if len(users) != expected_users:
        raise ValueError(
            f"expected {expected_users} usable users, found {len(users)}"
        )

    genuine_pool = np.concatenate([genuine[user] for user in users], axis=0)
    center = genuine_pool.mean(axis=0)
    scale = genuine_pool.std(axis=0)
    scale[scale == 0] = 1.0
    genuine_z = {user: (genuine[user] - center) / scale for user in users}
    generated_z = {user: (generated[user] - center) / scale for user in users}

    target_parts: list[np.ndarray] = []
    other_parts: list[np.ndarray] = []
    for user in users:
        target_parts.append(
            pair_distances(generated_z[user], genuine_z[user])
        )
        for other in users:
            if other != user:
                other_parts.append(
                    pair_distances(generated_z[user], genuine_z[other])
                )

    return (
        float(np.concatenate(target_parts).mean()),
        float(np.concatenate(other_parts).mean()),
    )


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-data-root",
        type=Path,
        default=os.environ.get("ACTREAL_EVENT_DATA_ROOT"),
        help="local root containing licensed HMOG-derived bundle directories",
    )
    parser.add_argument(
        "--bundle-map",
        type=Path,
        default=REPO / "data" / "event_level" / "ACTION_BUNDLE_MAP.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory outside the immutable artifact checkout",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-per-user", type=int, default=40)
    parser.add_argument("--expected-users", type=int, default=20)
    args = parser.parse_args()
    if args.event_data_root is None:
        parser.error(
            "provide --event-data-root or set ACTREAL_EVENT_DATA_ROOT; "
            "licensed HMOG-derived shards are not included"
        )

    bundle_map = load_bundle_map(args.bundle_map)
    values: dict[str, dict[str, tuple[float, float]]] = {}
    for action in ACTIONS:
        per_user = load_action_events(
            bundle_shards(args.event_data_root, bundle_map[action]),
            action,
            args.split,
        )
        values[action] = {
            name: compute_action(
                per_user,
                detector,
                args.max_per_user,
                args.expected_users,
            )
            for name, detector in FEATURE_SPACES.items()
        }

    generated_rows: list[list[str]] = []
    for action in ACTIONS:
        hmog = values[action]["hmog"]
        ttos = values[action]["ttos"]
        generated_rows.append(
            [
                action.title(),
                f"{hmog[0]:.3f}",
                f"{hmog[1]:.3f}",
                f"{ttos[0]:.3f}",
                f"{ttos[1]:.3f}",
            ]
        )

    write_csv(
        args.out / "generated_target_structure.csv",
        [
            "action",
            "hmog_d_fake",
            "hmog_d_inter",
            "ttos_d_fake",
            "ttos_d_inter",
        ],
        generated_rows,
    )
    print(f"wrote the generated target-user structure table to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
