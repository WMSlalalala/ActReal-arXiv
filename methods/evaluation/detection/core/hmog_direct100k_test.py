from __future__ import annotations

"""Independent model replay for direct100k detector test sets.

Both detector families replay the same way: the training cell's frozen model
artifact is reloaded, the test split is scored again, and the replayed scores
must reproduce the training-time scores.  Only loading and scoring differ
between a torch checkpoint and a fitted classical estimator; every metric,
curve and bootstrap below that point is family agnostic.
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .audit import sha256_file
from .event_detectors import (
    DEEP_DETECTORS,
    FORMAL_MODALITY_CHANNELS,
    REGISTERED_DETECTORS,
    build_deep_detector,
    classical_scores,
    detector_spec,
    extract_event_features,
)
from .event_pad import (
    ACTIONS,
    EventPadError,
    _load_manifest,
    load_event_partition,
    operating_metrics,
)
from .formal_event_pad import (
    _descriptive_eer,
    _iter_action_batches,
    _metadata,
    _score_deep,
)


DIRECT_MODALITIES = (
    "trajectory_xytime",
    "imu_only",
    "imu_trajectory_xytime",
)
CELL_TEST_SCHEMA = "hmog_direct100k_checkpoint_test_v1"
DETECTOR_GROUPS = {
    "deep": "deep_pad",
    "feature": "feature_pad",
    "paper": "paper_pad",
}


class Direct100kTestError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Direct100kTestError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _score_rows(
    *,
    metadata: Any,
    scores: np.ndarray,
    action: str,
) -> list[dict[str, Any]]:
    return [
        {
            "event_id": str(metadata.event_ids[index]),
            "user_id": str(metadata.user_ids[index]),
            "session_id": str(metadata.session_ids[index]),
            "source_cluster_id": str(metadata.source_cluster_ids[index]),
            "action": action,
            "label": int(metadata.labels[index]),
            "fake_high_score": float(scores[index]),
        }
        for index in range(len(scores))
    ]


def _verify_training_scores(
    path: Path,
    replay_rows: list[dict[str, Any]],
) -> float:
    if not path.is_file():
        raise Direct100kTestError(f"training test score dump is missing: {path}")
    original = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(original) != len(replay_rows):
        raise Direct100kTestError("checkpoint replay score coverage changed")
    maximum = 0.0
    for before, after in zip(original, replay_rows, strict=True):
        identity = ("event_id", "user_id", "label", "action")
        if any(before.get(key) != after.get(key) for key in identity):
            raise Direct100kTestError("checkpoint replay event order changed")
        difference = abs(
            float(before["fake_high_score"])
            - float(after["fake_high_score"])
        )
        maximum = max(maximum, difference)
    if maximum > 1.0e-6:
        raise Direct100kTestError(
            f"checkpoint replay differs from training-time scores: {maximum}"
        )
    return maximum


def _curve_rows(
    labels: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, float | int]]:
    thresholds = np.concatenate(
        [
            np.unique(np.asarray(scores, dtype=np.float64)),
            [np.nextafter(float(np.max(scores)), np.inf)],
        ]
    )
    real = labels == 0
    fake = labels == 1
    real_scores = np.sort(scores[real])
    fake_scores = np.sort(scores[fake])
    frr = (
        len(real_scores)
        - np.searchsorted(real_scores, thresholds, side="left")
    ) / len(real_scores)
    far = np.searchsorted(
        fake_scores, thresholds, side="left"
    ) / len(fake_scores)
    return [
        {
            "threshold_fake_score": float(threshold),
            "frr": float(real_rejected),
            "far": float(fake_accepted),
            "fake_detection_rate": float(1.0 - fake_accepted),
            "n_real": int(real.sum()),
            "n_fake": int(fake.sum()),
        }
        for threshold, real_rejected, fake_accepted in zip(
            thresholds, frr, far, strict=True
        )
    ]


def _user_bootstrap(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    user_ids: np.ndarray,
    eer_threshold: float,
    target_threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    users = np.unique(user_ids.astype(str))
    if len(users) < 2:
        raise Direct100kTestError("test bootstrap requires at least two users")
    real_indices = {
        user: np.flatnonzero((user_ids == user) & (labels == 0))
        for user in users
    }
    fake_indices = {
        user: np.flatnonzero((user_ids == user) & (labels == 1))
        for user in users
    }
    rng = np.random.default_rng(seed)
    real_draws = rng.integers(0, len(users), size=(replicates, len(users)))
    fake_draws = rng.integers(0, len(users), size=(replicates, len(users)))
    values: dict[str, list[float]] = {
        "eer_far": [],
        "eer_frr": [],
        "target_far": [],
        "target_frr": [],
        "auc": [],
    }
    for real_draw, fake_draw in zip(real_draws, fake_draws, strict=True):
        selected_real = np.concatenate(
            [real_indices[users[index]] for index in real_draw]
        )
        selected_fake = np.concatenate(
            [fake_indices[users[index]] for index in fake_draw]
        )
        if not len(selected_real) or not len(selected_fake):
            raise Direct100kTestError("bootstrap draw lacks a label class")
        real_scores = scores[selected_real]
        fake_scores = scores[selected_fake]
        values["eer_far"].append(
            float(np.mean(fake_scores < eer_threshold))
        )
        values["eer_frr"].append(
            float(np.mean(real_scores >= eer_threshold))
        )
        values["target_far"].append(
            float(np.mean(fake_scores < target_threshold))
        )
        values["target_frr"].append(
            float(np.mean(real_scores >= target_threshold))
        )
        values["auc"].append(
            float(
                roc_auc_score(
                    np.concatenate(
                        [
                            np.zeros(len(selected_real), dtype=np.int64),
                            np.ones(len(selected_fake), dtype=np.int64),
                        ]
                    ),
                    np.concatenate([real_scores, fake_scores]),
                )
            )
        )

    def interval(key: str) -> dict[str, float]:
        array = np.asarray(values[key], dtype=np.float64)
        return {
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
        }

    return {
        "n_boot": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": "test_user",
        "real_fake_resampling": "independent_with_replacement",
        "n_real_users": int(
            len(np.unique(user_ids[labels == 0].astype(str)))
        ),
        "n_fake_users": int(
            len(np.unique(user_ids[labels == 1].astype(str)))
        ),
        **{f"{key}_{bound}": value for key in values for bound, value in interval(key).items()},
    }


def _score_classical(
    estimator: Any,
    partition: Any,
    *,
    action: str,
    modality: str,
    detector: str,
) -> tuple[np.ndarray, list[float], Any]:
    """Replay a fitted classical estimator with the same contract as deep."""

    metadata = _metadata(partition, action)
    scores = np.empty(len(metadata.labels), dtype=np.float64)
    latency = np.empty(len(metadata.labels), dtype=np.float64)
    for imu, trajectory, mask, _labels, positions in _iter_action_batches(
        partition,
        action,
        batch_size=64,
        shuffle=False,
        seed=0,
    ):
        started = time.perf_counter_ns()
        features = np.stack(
            [
                extract_event_features(
                    detector,
                    modality,
                    imu[index][mask[index]],
                    trajectory[index][mask[index]],
                )
                for index in range(len(imu))
            ],
            axis=0,
        )
        batch = classical_scores(estimator, features)
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        scores[positions] = np.asarray(batch, dtype=np.float64)
        latency[positions] = elapsed / max(len(positions), 1)
    return scores, latency.tolist(), metadata


def test_checkpoint_cell(
    *,
    manifest: str | Path,
    training_cell_dir: str | Path,
    output_dir: str | Path,
    device_name: str,
    bootstrap_replicates: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    manifest_path = Path(manifest).resolve()
    training = Path(training_cell_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise Direct100kTestError(f"test output already exists: {output}")
    summary = _load_json(training / "summary.json")
    thresholds = _load_json(training / "thresholds.json")
    family = str(summary.get("family", ""))
    if family not in DETECTOR_GROUPS:
        raise Direct100kTestError(f"unregistered detector family {family!r}")
    checkpoint_path = training / (
        "checkpoint.pt" if family == "deep" else "model.joblib"
    )
    if not checkpoint_path.is_file():
        raise Direct100kTestError(f"model artifact is missing: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        summary.get("status") != "pass"
        or summary.get("training_from_scratch") is not True
        or summary.get("model_sha256") != checkpoint_sha256
        or thresholds.get("model_sha256") != checkpoint_sha256
        or summary.get("manifest_sha256") != sha256_file(manifest_path)
        or thresholds.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise Direct100kTestError("training artifacts are not internally bound")
    action = str(summary.get("action", ""))
    modality = str(summary.get("modality", ""))
    detector = str(summary.get("detector", ""))
    if (
        action not in ACTIONS
        or modality not in DIRECT_MODALITIES
        or detector not in REGISTERED_DETECTORS
        or detector_spec(detector).family != family
    ):
        raise Direct100kTestError("unregistered direct100k test cell")

    device = torch.device(device_name)
    if family == "deep":
        if device.type == "cuda":
            if not torch.cuda.is_available() or device.index is None:
                raise Direct100kTestError("explicit CUDA device is required")
            torch.cuda.set_device(device)
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            payload.get("action") != action
            or payload.get("modality") != modality
            or payload.get("detector") != detector
            or payload.get("manifest_sha256") != sha256_file(manifest_path)
            or payload.get("input_channel_schema")
            != list(FORMAL_MODALITY_CHANNELS[modality])
        ):
            raise Direct100kTestError("checkpoint identity/channel binding failed")
        model = build_deep_detector(detector, modality, action).to(device)
        incompatible = model.load_state_dict(payload["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise Direct100kTestError(
                "checkpoint state dictionary is incompatible"
            )
        model.eval()
        normalizer = payload.get("normalizer")
        if not isinstance(normalizer, dict):
            raise Direct100kTestError("checkpoint normalizer is missing")
    else:
        # joblib.dump stores the bare fitted estimator, so identity cannot be
        # re-read from the artifact the way a torch payload carries it.  The
        # summary/threshold pair already binds both this file's digest and the
        # manifest digest, and the channel schema is checked against the
        # modality the summary declares.
        if summary.get("channel_schema") != list(
            FORMAL_MODALITY_CHANNELS[modality]
        ):
            raise Direct100kTestError("classical channel binding failed")
        model = joblib.load(checkpoint_path)

    rows = _load_manifest(manifest_path)
    test = load_event_partition(rows["test"])
    if family == "deep":
        scores, latency, metadata = _score_deep(
            model,
            test,
            action=action,
            modality=modality,
            device=device,
            mean=np.asarray(normalizer["mean"], dtype=np.float32),
            std=np.asarray(normalizer["std"], dtype=np.float32),
        )
    else:
        scores, latency, metadata = _score_classical(
            model,
            test,
            action=action,
            modality=modality,
            detector=detector,
        )
    replay_rows = _score_rows(
        metadata=metadata, scores=scores, action=action
    )
    maximum_difference = _verify_training_scores(
        training / "test_scores.jsonl", replay_rows
    )
    output.mkdir(parents=True)
    score_path = output / "test_scores.jsonl"
    with score_path.open("x", encoding="utf-8") as handle:
        for row in replay_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    curve_rows = _curve_rows(metadata.labels, scores)
    curve_path = output / "curves.csv"
    with curve_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    eer_threshold = float(thresholds["eer"])
    target_threshold = float(thresholds["frr5"])
    eer_metrics = operating_metrics(metadata.labels, scores, eer_threshold)
    target_metrics = operating_metrics(
        metadata.labels, scores, target_threshold
    )
    descriptive = _descriptive_eer(metadata.labels, scores)
    auc = float(roc_auc_score(metadata.labels, scores))
    bootstrap = _user_bootstrap(
        labels=metadata.labels,
        scores=scores,
        user_ids=metadata.user_ids,
        eer_threshold=eer_threshold,
        target_threshold=target_threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    result = {
        "schema_version": CELL_TEST_SCHEMA,
        "status": "pass",
        "action": action,
        "modality": modality,
        "variant": {
            "trajectory_xytime": "trajectory_only",
            "imu_only": "imu_only",
            "imu_trajectory_xytime": "trajectory_plus_imu",
        }[modality],
        "detector_group": DETECTOR_GROUPS[family],
        "detector": detector,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_cell": str(training),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_reloaded": True,
        "detector_family": family,
        "model_artifact": checkpoint_path.name,
        "model_refit_during_test": False,
        "threshold_source": "development",
        "test_split_only": True,
        "training_time_score_replay_max_abs_difference": maximum_difference,
        "training_time_score_replay_pass": True,
        "n_real": int(np.sum(metadata.labels == 0)),
        "n_fake": int(np.sum(metadata.labels == 1)),
        "n_test_users": int(len(np.unique(metadata.user_ids))),
        "auc": auc,
        "descriptive_test_eer": descriptive,
        "development_eer_threshold": eer_threshold,
        "development_frr5_threshold": target_threshold,
        "eer_far": float(eer_metrics["far_fake_accepted"]),
        "eer_frr": float(eer_metrics["frr_real_rejected"]),
        "eer_fake_detection_rate": float(
            1.0 - float(eer_metrics["far_fake_accepted"])
        ),
        "target_far": float(target_metrics["far_fake_accepted"]),
        "target_frr": float(target_metrics["frr_real_rejected"]),
        "target_fake_detection_rate": float(
            1.0 - float(target_metrics["far_fake_accepted"])
        ),
        "bootstrap": bootstrap,
        "latency_ms_per_event": {
            "p50": float(np.quantile(latency, 0.50)),
            "p95": float(np.quantile(latency, 0.95)),
        },
        "test_scores": str(score_path),
        "test_scores_sha256": sha256_file(score_path),
        "curves": str(curve_path),
        "curves_sha256": sha256_file(curve_path),
    }
    _write_json(output / "summary.json", result)
    return result


__all__ = [
    "CELL_TEST_SCHEMA",
    "DIRECT_MODALITIES",
    "Direct100kTestError",
    "test_checkpoint_cell",
]
