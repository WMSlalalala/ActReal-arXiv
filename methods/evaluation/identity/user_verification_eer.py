#!/usr/bin/env python3
"""Recompute the frozen-five-shot user-verification EERs in Appendix E.1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


REPO = Path(__file__).resolve().parents[3]
METHODS = REPO / "methods"
if str(METHODS) not in sys.path:
    sys.path.insert(0, str(METHODS))

from evaluation.detection.core.event_detectors import (  # noqa: E402
    extract_event_features,
)


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
FEATURE_SPACES = {
    "hmog_style": "hmog_style_svm",
    "ttos_style": "paper_svm",
}


@dataclass(frozen=True)
class Event:
    user_id: str
    session_id: str
    event_id: str
    source_cluster_id: str
    imu: np.ndarray
    trajectory: np.ndarray


def load_bundle_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        action: str(payload["actions"][action]["imu_only"])
        for action in ACTIONS
    }


def find_shards(event_data_root: Path, bundle: str) -> list[Path]:
    for directory in (
        event_data_root / bundle / "shards",
        event_data_root / bundle,
    ):
        shards = sorted(directory.glob("hmog_u*.npz"))
        if shards:
            return shards
    raise FileNotFoundError(
        f"no hmog_u*.npz files for {bundle!r} under {event_data_root}"
    )


def load_genuine_events(
    event_data_root: Path,
    bundle: str,
    action: str,
    split: str,
) -> list[Event]:
    events: list[Event] = []
    physical_ids: set[tuple[str, str, str]] = set()
    event_ids: set[str] = set()
    for shard in find_shards(event_data_root, bundle):
        with np.load(shard, allow_pickle=False) as store:
            if str(np.asarray(store["split"]).item()) != split:
                continue
            offsets = np.asarray(store["offsets"], dtype=np.int64)
            labels = np.asarray(store["label"], dtype=np.int64)
            actions = np.asarray(store["action"]).astype(str)
            selected = np.flatnonzero((labels == 0) & (actions == action))
            for index in selected:
                user_id = str(store["user_id"][index])
                session_id = str(store["session_id"][index])
                event_id = str(store["event_id"][index])
                source_cluster_id = str(store["source_cluster_id"][index])
                physical_id = (user_id, session_id, source_cluster_id)
                if physical_id in physical_ids or event_id in event_ids:
                    raise ValueError(
                        f"duplicate genuine event for {action}: {event_id}"
                    )
                physical_ids.add(physical_id)
                event_ids.add(event_id)
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                imu = np.asarray(
                    store["imu_flat"][start:stop], dtype=np.float32
                )
                trajectory = np.asarray(
                    store["trajectory_flat"][start:stop], dtype=np.float32
                )
                if imu.shape != (len(trajectory), 6):
                    raise ValueError(f"misaligned event arrays for {event_id}")
                events.append(
                    Event(
                        user_id=user_id,
                        session_id=session_id,
                        event_id=event_id,
                        source_cluster_id=source_cluster_id,
                        imu=imu,
                        trajectory=trajectory,
                    )
                )
    if not events:
        raise ValueError(f"no {split} genuine events for {action}/{bundle}")
    return events


def load_fiveshot_manifest(
    path: Path,
    split: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        action: {} for action in ACTIONS
    }
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["split"]) != split:
                continue
            action = str(row["action"])
            result[action].setdefault(str(row["user_id"]), []).append(row)
    for action in ACTIONS:
        for user, rows in result[action].items():
            rows.sort(key=lambda row: int(row["shot_ordinal"]))
            ordinals = [int(row["shot_ordinal"]) for row in rows]
            if len(rows) != 5 or ordinals != list(range(5)):
                raise ValueError(f"{action}/{user} does not have shots 0..4")
            if len({str(row["event_id"]) for row in rows}) != 5:
                raise ValueError(f"{action}/{user} repeats a five-shot event")
    return result


def identity_balanced_weights(
    users: np.ndarray,
    target: str,
) -> np.ndarray:
    users = np.asarray(users).astype(str)
    positive = users == target
    if not np.any(positive):
        raise ValueError(f"no positive examples for {target}")
    impostors = sorted(set(users[~positive]))
    if not impostors:
        raise ValueError(f"no impostor identities for {target}")
    weights = np.zeros(len(users), dtype=np.float64)
    weights[positive] = 0.5 / int(np.sum(positive))
    identity_mass = 0.5 / len(impostors)
    for impostor in impostors:
        selected = users == impostor
        weights[selected] = identity_mass / int(np.sum(selected))
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("identity-balanced weights do not sum to one")
    return weights


def interpolated_eer(
    users: np.ndarray,
    target: str,
    scores: np.ndarray,
) -> tuple[float, float]:
    users = np.asarray(users).astype(str)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (users == target).astype(np.int64)
    weights = identity_balanced_weights(users, target)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite verifier score")
    fpr, tpr, _thresholds = roc_curve(
        labels,
        scores,
        sample_weight=weights,
        drop_intermediate=False,
    )
    fnr = 1.0 - tpr
    difference = fpr - fnr
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1.0e-15))
    if len(exact):
        index = int(exact[0])
        eer = float((fpr[index] + fnr[index]) / 2.0)
    else:
        crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
        if not len(crossings):
            index = int(np.argmin(np.abs(difference)))
            eer = float((fpr[index] + fnr[index]) / 2.0)
        else:
            left = int(crossings[0])
            right = left + 1
            alpha = float(
                -difference[left] / (difference[right] - difference[left])
            )
            crossing_fpr = fpr[left] + alpha * (fpr[right] - fpr[left])
            crossing_fnr = fnr[left] + alpha * (fnr[right] - fnr[left])
            eer = float((crossing_fpr + crossing_fnr) / 2.0)
    auc = float(roc_auc_score(labels, scores, sample_weight=weights))
    return eer, auc


def fit_scores(
    train_features: np.ndarray,
    train_users: np.ndarray,
    test_features: np.ndarray,
    target: str,
    seed: int,
) -> np.ndarray:
    labels = (train_users == target).astype(np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"one-class enrollment data for {target}")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight=None,
            dual=False,
            random_state=seed,
            max_iter=20_000,
        ),
    )
    model.fit(
        train_features,
        labels,
        linearsvc__sample_weight=identity_balanced_weights(
            train_users, target
        ),
    )
    scores = np.asarray(
        model.decision_function(test_features), dtype=np.float64
    ).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite SVM margins for {target}")
    return scores


def featurize(events: list[Event], detector_name: str) -> np.ndarray:
    features = np.asarray(
        [
            extract_event_features(
                detector_name, "imu_only", event.imu, event.trajectory
            )
            for event in events
        ],
        dtype=np.float64,
    )
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("invalid extracted feature matrix")
    return features


def bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("participant bootstrap requires at least two users")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = values[draws].mean(axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def evaluate_action(
    action: str,
    feature_space: str,
    features: np.ndarray,
    events: list[Event],
    reference_rows: dict[str, list[dict[str, Any]]],
    expected_users: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_by_id = {event.event_id: index for index, event in enumerate(events)}
    identities = sorted({event.user_id for event in events})
    if len(identities) != expected_users or set(reference_rows) != set(identities):
        raise ValueError(f"{action}: five-shot users do not match test users")

    enrollment_indices: list[int] = []
    enrollment_sessions: dict[str, set[str]] = {}
    for user in identities:
        enrollment_sessions[user] = set()
        for row in reference_rows[user]:
            event_id = str(row["event_id"])
            if event_id not in event_by_id:
                raise ValueError(f"{action}: frozen reference missing: {event_id}")
            index = event_by_id[event_id]
            event = events[index]
            if event.user_id != user or event.session_id != str(row["session_id"]):
                raise ValueError(f"{action}: frozen reference identity mismatch")
            enrollment_indices.append(index)
            enrollment_sessions[user].add(event.session_id)
    if (
        len(enrollment_indices) != expected_users * 5
        or len(set(enrollment_indices)) != len(enrollment_indices)
    ):
        raise ValueError(f"{action}: invalid frozen five-shot index set")

    enrollment_set = set(enrollment_indices)
    evaluation_indices = [
        index
        for index, event in enumerate(events)
        if index not in enrollment_set
        and event.session_id not in enrollment_sessions[event.user_id]
    ]
    train_users = np.asarray(
        [events[index].user_id for index in enrollment_indices]
    )
    test_users = np.asarray(
        [events[index].user_id for index in evaluation_indices]
    )
    if any(int(np.sum(train_users == user)) != 5 for user in identities):
        raise RuntimeError(f"{action}: enrollment is not exactly five-shot")

    user_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(identities):
        genuine_count = int(np.sum(test_users == target))
        impostor_identities = len(set(test_users[test_users != target]))
        if genuine_count == 0 or impostor_identities != expected_users - 1:
            raise ValueError(
                f"{action}/{target}: incomplete held-session evaluation pool"
            )
        scores = fit_scores(
            features[enrollment_indices],
            train_users,
            features[evaluation_indices],
            target,
            seed + target_index,
        )
        eer, auc = interpolated_eer(test_users, target, scores)
        user_rows.append(
            {
                "action": action,
                "feature_space": feature_space,
                "user_id": target,
                "enrollment_events": 5,
                "enrollment_sessions": len(enrollment_sessions[target]),
                "genuine_attempts": genuine_count,
                "impostor_identities": impostor_identities,
                "eer": eer,
                "roc_auc": auc,
            }
        )

    eers = np.asarray([float(row["eer"]) for row in user_rows])
    aucs = np.asarray([float(row["roc_auc"]) for row in user_rows])
    ci_low, ci_high = bootstrap_mean(eers, bootstrap_replicates, seed)
    summary = {
        "action": action,
        "feature_space": feature_space,
        "classifier": "identity_balanced_ovr_linear_svm",
        "users": len(user_rows),
        "macro_eer": float(eers.mean()),
        "user_clustered_ci_low": ci_low,
        "user_clustered_ci_high": ci_high,
        "macro_roc_auc": float(aucs.mean()),
        "min_user_eer": float(eers.min()),
        "max_user_eer": float(eers.max()),
        "genuine_events": len(evaluation_indices),
    }
    return user_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    event_root = os.environ.get("ACTREAL_EVENT_DATA_ROOT")
    manifest = os.environ.get("ACTREAL_FIVESHOT_MANIFEST")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-data-root",
        type=Path,
        default=Path(event_root) if event_root else None,
    )
    parser.add_argument(
        "--bundle-map",
        type=Path,
        default=REPO / "data" / "event_level" / "ACTION_BUNDLE_MAP.json",
    )
    parser.add_argument(
        "--fiveshot-manifest",
        type=Path,
        default=Path(manifest) if manifest else None,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-users", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_825)
    args = parser.parse_args()
    if args.event_data_root is None:
        parser.error(
            "provide --event-data-root or set ACTREAL_EVENT_DATA_ROOT"
        )
    if args.fiveshot_manifest is None:
        args.fiveshot_manifest = (
            args.event_data_root
            / "fiveshot_material"
            / "material_manifest.jsonl"
        )
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def main() -> int:
    args = parse_args()
    bundle_map = load_bundle_map(args.bundle_map)
    references = load_fiveshot_manifest(args.fiveshot_manifest, args.split)
    all_user_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for action_index, action in enumerate(ACTIONS):
        events = load_genuine_events(
            args.event_data_root,
            bundle_map[action],
            action,
            args.split,
        )
        if len({event.user_id for event in events}) != args.expected_users:
            raise ValueError(f"{action}: unexpected number of test users")
        for feature_index, (feature_space, detector_name) in enumerate(
            FEATURE_SPACES.items()
        ):
            user_rows, summary = evaluate_action(
                action,
                feature_space,
                featurize(events, detector_name),
                events,
                references[action],
                args.expected_users,
                args.bootstrap_replicates,
                args.seed + 100 * action_index + feature_index,
            )
            all_user_rows.extend(user_rows)
            summaries.append(summary)

    write_csv(args.out / "per_user_eer.csv", all_user_rows)
    write_csv(args.out / "summary.csv", summaries)

    print("Action      HMOG-style EER (95% CI)       TToS-style EER (95% CI)")
    for action in ACTIONS:
        selected = {
            row["feature_space"]: row
            for row in summaries
            if row["action"] == action
        }
        hmog = selected["hmog_style"]
        ttos = selected["ttos_style"]
        print(
            f"{action.title():<11} "
            f"{100 * hmog['macro_eer']:5.1f} "
            f"[{100 * hmog['user_clustered_ci_low']:4.1f}, "
            f"{100 * hmog['user_clustered_ci_high']:4.1f}]          "
            f"{100 * ttos['macro_eer']:5.1f} "
            f"[{100 * ttos['user_clustered_ci_low']:4.1f}, "
            f"{100 * ttos['user_clustered_ci_high']:4.1f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
