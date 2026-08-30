from __future__ import annotations

"""Fail-closed gate for the non-paper-facing Event development qualification."""

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score

from .audit import sha256_file
from .event_pad import ACTIONS, EventPadError


QUALIFICATION_SCHEMA = "event_pad_dev_only_qualification_v3"
QUALIFICATION_DETECTOR_PROTOCOL = (
    "deep_xytime_two_registered_detectors_v1"
)
QUALIFICATION_DETECTOR = "behaveformer_stdat"
QUALIFICATION_CONFIRMATORY_DETECTORS = ("authconformer",)
QUALIFICATION_MODALITIES = ("trajectory_xytime", "imu_only")
QUALIFICATION_EPOCHS = 5
QUALIFICATION_SEED = 42
QUALIFICATION_AUC_MARGIN = 0.04
QUALIFICATION_SCOPE = "full"
QUALIFICATION_DECISION_RULE = (
    "allow_iff_each_action_abs_delta_auc_trajectory_minus_imu_lte_0.04"
)
QUALIFICATION_SCORE_USE_POLICY = (
    "release_level_gate_only_no_per_event_selection_drop_reweight_or_relabel"
)
_AUC_ATOL = 1.0e-12


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EventPadError(f"missing development qualification: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventPadError(
            f"invalid development qualification JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise EventPadError("development qualification must be an object")
    return value


def qualification_decision(
    action_auc: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Apply the preregistered action-level symmetric same-level gate."""

    if set(action_auc) != set(ACTIONS):
        raise EventPadError("qualification requires exactly five actions")
    results: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        values = action_auc[action]
        if set(values) != set(QUALIFICATION_MODALITIES):
            raise EventPadError(
                f"{action}: qualification requires trajectory and IMU AUC"
            )
        try:
            trajectory = float(values["trajectory_xytime"])
            imu = float(values["imu_only"])
        except (TypeError, ValueError) as exc:
            raise EventPadError(
                f"{action}: invalid qualification AUC"
            ) from exc
        if (
            not np.isfinite(trajectory)
            or not np.isfinite(imu)
            or not 0.0 <= trajectory <= 1.0
            or not 0.0 <= imu <= 1.0
        ):
            raise EventPadError(f"{action}: invalid qualification AUC")
        delta = trajectory - imu
        within_band = (
            abs(delta) <= QUALIFICATION_AUC_MARGIN + _AUC_ATOL
        )
        if delta > QUALIFICATION_AUC_MARGIN + _AUC_ATOL:
            classification = "trajectory_too_easy_iterate"
        elif delta < -QUALIFICATION_AUC_MARGIN - _AUC_ATOL:
            classification = "trajectory_too_hard_iterate"
        else:
            classification = "within_symmetric_margin"
        results[action] = {
            "trajectory_dev_roc_auc": trajectory,
            "imu_dev_roc_auc": imu,
            "delta_auc_trajectory_minus_imu": delta,
            "symmetric_margin": QUALIFICATION_AUC_MARGIN,
            "within_symmetric_margin": within_band,
            "trajectory_imu_same_level": within_band,
            "classification": classification,
        }
    return results, all(
        result["trajectory_imu_same_level"] for result in results.values()
    )


def validate_dev_qualification_receipt(
    receipt_path: str | Path,
    *,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a qualification and, optionally, one formal manifest binding."""

    path = Path(receipt_path).resolve()
    value = _read_object(path)
    test_access = value.get("test_signal_access")
    manifests = value.get("qualified_manifests")
    action_results = value.get("action_results")
    environment = value.get("environment")
    if (
        value.get("schema_version") != QUALIFICATION_SCHEMA
        or value.get("status") != "pass"
        or value.get("formal_result") is not False
        or value.get("paper_facing") is not False
        or value.get("detector_protocol")
        != QUALIFICATION_DETECTOR_PROTOCOL
        or value.get("scope") != QUALIFICATION_SCOPE
        or value.get("detector") != QUALIFICATION_DETECTOR
        or value.get("qualification_detector_role")
        != "fixed_generator_development_gate"
        or value.get("confirmatory_detectors") != list(
            QUALIFICATION_CONFIRMATORY_DETECTORS
        )
        or value.get("confirmatory_detector_role")
        != "formal_confirmatory_only_never_generator_iteration"
        or value.get("formal_test_feedback_to_generator_forbidden")
        is not True
        or value.get("modalities") != list(QUALIFICATION_MODALITIES)
        or value.get("actions") != list(ACTIONS)
        or value.get("qualification_cells")
        != len(ACTIONS) * len(QUALIFICATION_MODALITIES)
        or value.get("epochs") != QUALIFICATION_EPOCHS
        or value.get("seed") != QUALIFICATION_SEED
        or value.get("auc_margin") != QUALIFICATION_AUC_MARGIN
        or value.get("decision_rule") != QUALIFICATION_DECISION_RULE
        or value.get("development_score_use_policy")
        != QUALIFICATION_SCORE_USE_POLICY
        or value.get("per_event_fake_selection_from_scores") is not False
        or value.get("per_event_fake_drop_from_scores") is not False
        or value.get("per_event_fake_reweight_from_scores") is not False
        or value.get("per_event_fake_relabel_from_scores") is not False
        or not isinstance(test_access, Mapping)
        or test_access.get("manifest_test_rows_ignored") != 1
        or test_access.get("test_source_paths_resolved") != 0
        or test_access.get("test_signal_file_stat_or_hash_calls") != 0
        or test_access.get("test_signal_arrays_loaded") != 0
        or test_access.get("test_scoring_calls") != 0
        or not isinstance(manifests, Mapping)
        or set(manifests) != {"balanced_small", "full"}
        or not isinstance(action_results, Mapping)
        or set(action_results) != set(ACTIONS)
        or not isinstance(environment, Mapping)
        or environment.get("device") != (
            f"cuda:{environment.get('physical_device_index')}"
        )
        or not isinstance(environment.get("physical_device_index"), int)
        or isinstance(environment.get("physical_device_index"), bool)
        or environment.get("physical_device_index", -1) < 0
        or not isinstance(environment.get("physical_device_uuid"), str)
        or not environment.get("physical_device_uuid")
        or not isinstance(environment.get("physical_device_name"), str)
        or not environment.get("physical_device_name")
    ):
        raise EventPadError("development qualification contract failed")

    auc_inputs: dict[str, dict[str, float]] = {}
    for action in ACTIONS:
        result = action_results[action]
        if not isinstance(result, Mapping):
            raise EventPadError(
                f"{action}: malformed development qualification result"
            )
        auc_inputs[action] = {
            "trajectory_xytime": result.get("trajectory_dev_roc_auc"),
            "imu_only": result.get("imu_dev_roc_auc"),
        }
    recomputed, allowed = qualification_decision(auc_inputs)
    for action in ACTIONS:
        observed = action_results[action]
        expected = recomputed[action]
        for key, expected_value in expected.items():
            observed_value = observed.get(key)
            if isinstance(expected_value, float):
                if (
                    isinstance(observed_value, bool)
                    or not isinstance(observed_value, (int, float))
                    or not np.isclose(
                        float(observed_value),
                        expected_value,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise EventPadError(
                        f"{action}: qualification decision mismatch"
                    )
            elif observed_value != expected_value:
                raise EventPadError(
                    f"{action}: qualification decision mismatch"
                )
    if (
        value.get("formal_60_allowed") is not allowed
        or value.get("decision") != (
            "allow_formal_60" if allowed else "iterate_generator"
        )
    ):
        raise EventPadError("development qualification launch decision mismatch")
    if value.get("all_actions_within_symmetric_margin") is not all(
        result["within_symmetric_margin"] for result in recomputed.values()
    ):
        raise EventPadError(
            "development qualification symmetric-band summary mismatch"
        )
    losses = value.get("training_loss_by_action_modality")
    if not isinstance(losses, Mapping) or set(losses) != set(ACTIONS):
        raise EventPadError("development qualification loss inventory failed")
    for action in ACTIONS:
        action_losses = losses[action]
        if (
            not isinstance(action_losses, Mapping)
            or set(action_losses) != set(QUALIFICATION_MODALITIES)
        ):
            raise EventPadError(
                f"{action}: development qualification loss inventory failed"
            )
        for modality in QUALIFICATION_MODALITIES:
            values = action_losses[modality]
            if (
                not isinstance(values, list)
                or len(values) != QUALIFICATION_EPOCHS
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not np.isfinite(float(item))
                    for item in values
                )
            ):
                raise EventPadError(
                    f"{action}/{modality}: qualification epoch record failed"
                )

    score_path = Path(
        str(value.get("development_scores_file", ""))
    ).resolve()
    if (
        not score_path.is_file()
        or score_path.parent != path.parent
        or value.get("development_scores_sha256")
        != sha256_file(score_path)
    ):
        raise EventPadError(
            "development qualification score artifact binding failed"
        )
    score_rows: dict[
        tuple[str, str],
        dict[tuple[str, str, str, str, int], float],
    ] = {
        (action, modality): {}
        for action in ACTIONS
        for modality in QUALIFICATION_MODALITIES
    }
    with score_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventPadError(
                    f"invalid development score row {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise EventPadError(
                    f"invalid development score row {line_number}"
                )
            key = (row.get("action"), row.get("modality"))
            if (
                row.get("split") != "development"
                or key not in score_rows
                or row.get("label") not in (0, 1)
                or any(
                    not isinstance(row.get(field), str) or not row[field]
                    for field in (
                        "user_id",
                        "session_id",
                        "event_id",
                        "source_cluster_id",
                    )
                )
                or isinstance(row.get("fake_high_score"), bool)
                or not isinstance(row.get("fake_high_score"), (int, float))
                or not np.isfinite(float(row["fake_high_score"]))
            ):
                raise EventPadError(
                    f"invalid development score row {line_number}"
                )
            slot = (
                row["user_id"],
                row["session_id"],
                row["event_id"],
                row["source_cluster_id"],
                int(row["label"]),
            )
            if slot in score_rows[key]:
                raise EventPadError(
                    f"duplicate development score slot at row {line_number}"
                )
            score_rows[key][slot] = float(row["fake_high_score"])
    for action in ACTIONS:
        trajectory_rows = score_rows[(action, "trajectory_xytime")]
        imu_rows = score_rows[(action, "imu_only")]
        trajectory_slots = set(trajectory_rows)
        imu_slots = set(imu_rows)
        if not trajectory_slots or trajectory_slots != imu_slots:
            raise EventPadError(
                f"{action}: development qualification score pairing failed"
            )
        ordered_slots = sorted(trajectory_slots)
        labels = np.asarray(
            [slot[-1] for slot in ordered_slots], dtype=np.int64
        )
        if set(labels.tolist()) != {0, 1}:
            raise EventPadError(
                f"{action}: development qualification requires both labels"
            )
        for modality, rows_for_modality in (
            ("trajectory_xytime", trajectory_rows),
            ("imu_only", imu_rows),
        ):
            observed_auc = float(
                roc_auc_score(
                    labels,
                    np.asarray(
                        [
                            rows_for_modality[slot]
                            for slot in ordered_slots
                        ],
                        dtype=np.float64,
                    ),
                )
            )
            declared_auc = recomputed[action][
                f"{'trajectory' if modality == 'trajectory_xytime' else 'imu'}"
                "_dev_roc_auc"
            ]
            if not np.isclose(
                observed_auc,
                declared_auc,
                rtol=0.0,
                atol=_AUC_ATOL,
            ):
                raise EventPadError(
                    f"{action}/{modality}: development AUC does not "
                    "reproduce from bound scores"
                )
    for scope, binding in manifests.items():
        if not isinstance(binding, Mapping):
            raise EventPadError(
                f"{scope}: malformed qualification manifest binding"
            )
        bound_path = Path(str(binding.get("path", ""))).resolve()
        if (
            not bound_path.is_file()
            or binding.get("sha256") != sha256_file(bound_path)
        ):
            raise EventPadError(
                f"{scope}: qualification manifest binding failed"
            )
    if manifest is not None:
        requested = Path(manifest).resolve()
        if not requested.is_file():
            raise EventPadError(
                f"formal Event manifest is missing: {requested}"
            )
        matching = [
            binding
            for binding in manifests.values()
            if Path(str(binding["path"])).resolve() == requested
        ]
        if (
            len(matching) != 1
            or matching[0]["sha256"] != sha256_file(requested)
        ):
            raise EventPadError(
                "development qualification does not bind the formal manifest"
            )
    if not allowed:
        raise EventPadError(
            "development qualification requires generator iteration; "
            "formal 60-cell test access is forbidden"
        )
    return value


__all__ = [
    "QUALIFICATION_AUC_MARGIN",
    "QUALIFICATION_DECISION_RULE",
    "QUALIFICATION_CONFIRMATORY_DETECTORS",
    "QUALIFICATION_DETECTOR",
    "QUALIFICATION_DETECTOR_PROTOCOL",
    "QUALIFICATION_EPOCHS",
    "QUALIFICATION_MODALITIES",
    "QUALIFICATION_SCHEMA",
    "QUALIFICATION_SCOPE",
    "QUALIFICATION_SCORE_USE_POLICY",
    "QUALIFICATION_SEED",
    "qualification_decision",
    "validate_dev_qualification_receipt",
]
