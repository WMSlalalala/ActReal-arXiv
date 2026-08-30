from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_hmog_conditional_touch_smoke import (
    build_argument_parser,
    main as audit_main,
)
from pipeline.conditional_touch_request_plan import (
    ConditionalTouchRequestPlan,
)
from pipeline.conditional_touch_smoke_audit import (
    audit_conditional_touch_smoke,
    sha256_file,
)


class ConditionalTouchSmokeAuditTests(unittest.TestCase):
    def _trajectory(
        self,
        action: str,
        *,
        candidate_fake: bool,
        bad_pressure: bool,
        moving_tap: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = np.asarray((200.0, 400.0), dtype=np.float64)
        if action == "tap":
            end = (
                np.asarray((212.0, 400.0), dtype=np.float64)
                if moving_tap
                else start.copy()
            )
            middle = np.asarray((202.0, 399.5), dtype=np.float64)
        elif action == "scroll":
            end = np.asarray((205.0, 200.0), dtype=np.float64)
            middle = np.asarray(
                (230.0 if candidate_fake else 203.0, 300.0), dtype=np.float64
            )
        else:
            end = np.asarray((500.0, 450.0), dtype=np.float64)
            middle = np.asarray(
                (350.0, 480.0 if candidate_fake else 425.0), dtype=np.float64
            )
        pixels = np.vstack((start, start, middle, end, end))
        trajectory = np.zeros((len(pixels), 9), dtype=np.float32)
        trajectory[:, 0] = 1.0
        trajectory[:, 1] = pixels[:, 0] / 1080.0
        trajectory[:, 2] = pixels[:, 1] / 1920.0
        trajectory[:, 3] = 1.0
        if bad_pressure:
            trajectory[2, 3] = np.float32(0.99999)
        trajectory[:, 4] = 1.0
        trajectory[:, 7] = np.arange(len(pixels), dtype=np.float32) * 0.01
        trajectory[:, 8] = 1.0
        return trajectory, start, end

    def _write_dataset(
        self,
        root: Path,
        *,
        candidate: bool,
        bad_imu: bool = False,
        bad_pressure: bool = False,
        id_drift: bool = False,
        binding_drift: bool = False,
        moving_tap: bool = False,
        tap_request_plan: str = "none",
    ) -> tuple[Path, Path, Path]:
        root.mkdir()
        shards = root / "shards"
        shards.mkdir()
        manifest_rows = []
        provenance_rows = []
        split_users = {
            "train": "hmog_u000",
            "development": "hmog_u030",
            "test": "synthetic_user_006",
        }
        for split_index, (split, user) in enumerate(split_users.items()):
            event_ids: list[str] = []
            actions: list[str] = []
            labels: list[int] = []
            imu_parts: list[np.ndarray] = []
            trajectory_parts: list[np.ndarray] = []
            offsets = [0]
            for action_index, action in enumerate(("tap", "scroll", "swipe")):
                for label in (0, 1):
                    event_id = f"event-{split_index}-{action}-{label}"
                    if id_drift and candidate and split_index == 0 and action == "tap" and label == 1:
                        event_id += "-replacement"
                    stored_action = (
                        "swipe"
                        if binding_drift
                        and candidate
                        and split_index == 0
                        and action == "scroll"
                        and label == 1
                        else action
                    )
                    is_candidate_fake = candidate and label == 1
                    trajectory, start, end = self._trajectory(
                        action,
                        candidate_fake=is_candidate_fake,
                        bad_pressure=bad_pressure and is_candidate_fake,
                        moving_tap=(
                            moving_tap
                            and is_candidate_fake
                            and action == "tap"
                        ),
                    )
                    imu = np.full(
                        (len(trajectory), 6),
                        split_index * 100 + action_index * 10 + label,
                        dtype=np.float32,
                    )
                    if bad_imu and is_candidate_fake:
                        imu[0, 0] += np.float32(0.25)
                    event_ids.append(event_id)
                    actions.append(stored_action)
                    labels.append(label)
                    imu_parts.append(imu)
                    trajectory_parts.append(trajectory)
                    offsets.append(offsets[-1] + len(trajectory))
                    row = {
                        "event_id": event_id,
                        "split": split,
                        "user_id": user,
                        "action": stored_action,
                        "label": label,
                        "coordinate_clipping_used": False,
                    }
                    if label == 1:
                        actual = trajectory[[0, -1], 1:3].astype(np.float64)
                        actual *= np.asarray((1080.0, 1920.0))
                        row["donor"] = {
                            "conditioning_action": action,
                            "conditioning_orientation_id": 0,
                            "conditioning_direction": (
                                "right"
                                if action == "tap" and not np.array_equal(start, end)
                                else None
                            ),
                            "requested_start_px": start.tolist(),
                            "requested_end_px": end.tolist(),
                            "raw_output_start_px": start.tolist(),
                            "raw_output_end_px": end.tolist(),
                            "detector_output_start_px": actual[0].tolist(),
                            "detector_output_end_px": actual[1].tolist(),
                            "coordinate_clipping_used": False,
                        }
                        if (
                            action == "tap"
                            and not np.array_equal(start, end)
                            and tap_request_plan != "none"
                        ):
                            carrier_event_id = (
                                event_id
                                if tap_request_plan != "wrong_carrier"
                                else f"wrong-{event_id}"
                            )
                            plan = ConditionalTouchRequestPlan.create(
                                carrier_event_id=carrier_event_id,
                                original_event_plan_sha256="a" * 64,
                                orientation_id=0,
                                action="tap",
                                original_direction=None,
                                original_down_xy_px=(200.0, 400.0),
                                original_up_xy_px=(200.0, 400.0),
                                original_raw_t_ms=(0.0, 10.0, 20.0, 30.0, 40.0),
                                original_raw_duration_ms=40.0,
                                sampled_start_xy_px=start,
                                sampled_end_xy_px=end,
                                sampled_direction="right",
                                request_model_file_sha256="b" * 64,
                                request_model_artifact_sha256="c" * 64,
                                request_model_schema_version="request-model-v1",
                                request_model_source_fingerprint_sha256="d" * 64,
                                request_seed=17 + split_index,
                            )
                            request_plan = plan.to_json_dict()
                            if tap_request_plan == "malformed":
                                request_plan["unexpected"] = True
                            row["donor"]["request_plan"] = request_plan
                    else:
                        row["donor"] = {}
                    provenance_rows.append(row)
            shard = shards / f"{user}.npz"
            np.savez_compressed(
                shard,
                schema_version=np.asarray("joint_event_pad_ragged_shard_v1"),
                coordinate_schema=np.asarray("screen_relative_xy_v1"),
                time_schema=np.asarray("elapsed_seconds_since_event_start_v1"),
                scope=np.asarray("balanced_small"),
                split=np.asarray(split),
                imu_flat=np.concatenate(imu_parts),
                trajectory_flat=np.concatenate(trajectory_parts),
                offsets=np.asarray(offsets, dtype=np.int64),
                label=np.asarray(labels, dtype=np.int64),
                user_id=np.asarray([user] * len(labels)),
                session_id=np.zeros(len(labels), dtype=np.int64),
                event_id=np.asarray(event_ids),
                action=np.asarray(actions),
                source_cluster_id=np.asarray(event_ids),
                sample_idx=np.arange(len(labels), dtype=np.int64),
                cross_modal_pair_id=np.asarray([f"pair-{value}" for value in event_ids]),
            )
            manifest_rows.append(
                {
                    "schema_version": "joint_event_pad_manifest_v2",
                    "scope": "balanced_small",
                    "split": split,
                    "events": len(event_ids),
                    "user_ids": [user],
                    "shards": [
                        {
                            "source": str(shard.resolve()),
                            "source_sha256": sha256_file(shard),
                            "user_id": user,
                            "events": len(event_ids),
                        }
                    ],
                }
            )
        manifest = root / "event_manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        provenance = root / "provenance.jsonl"
        provenance.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in provenance_rows
            ),
            encoding="utf-8",
        )
        release = root / "release.json"
        release.write_text(
            json.dumps(
                {
                    "schema_version": "hmog_direct100k_detector_dataset_v1",
                    "event_manifest_sha256": sha256_file(manifest),
                    "provenance_sha256": sha256_file(provenance),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest, release, provenance

    def test_passes_exact_bindings_signals_endpoints_and_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_dataset(root / "baseline", candidate=False)
            candidate = self._write_dataset(root / "candidate", candidate=True)
            report = audit_conditional_touch_smoke(
                baseline_manifest=baseline[0],
                baseline_release=baseline[1],
                baseline_provenance_path=baseline[2],
                candidate_manifest=candidate[0],
                candidate_release=candidate[1],
                candidate_provenance_path=candidate[2],
                expected_events=18,
            )
        self.assertEqual(report["status"], "pass")
        invariants = report["hard_invariants"]
        self.assertTrue(invariants["exact_event_id_set"])
        self.assertTrue(invariants["exact_split_user_action_label_binding"])
        self.assertEqual(invariants["fake_touch_events_compared"], 9)
        self.assertEqual(invariants["imu_flat_exact_events"], 9)
        self.assertEqual(invariants["time_axis_exact_events"], 9)
        self.assertEqual(invariants["maximum_raw_endpoint_error_px"], 0.0)
        self.assertTrue(report["signal_semantics"]["imu_only_is_pure_imu"])
        scroll = report["distribution_diagnostics"]["actions"]["scroll"]
        self.assertEqual(scroll["groups"]["candidate_fake"]["events"], 3)
        self.assertIn(
            "zero_step_fraction",
            scroll["groups"]["candidate_fake"]["metrics"],
        )

    def test_fails_closed_on_imu_and_exact_pressure_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_dataset(root / "baseline", candidate=False)
            candidate = self._write_dataset(
                root / "candidate",
                candidate=True,
                bad_imu=True,
                bad_pressure=True,
            )
            report = audit_conditional_touch_smoke(
                baseline_manifest=baseline[0],
                candidate_manifest=candidate[0],
                candidate_provenance_path=candidate[2],
                expected_events=18,
            )
        self.assertEqual(report["status"], "fail")
        failures = {
            row["code"]: row["count"]
            for row in report["hard_invariants"]["failures"]
        }
        self.assertEqual(failures["fake_touch_imu_flat_mismatch"], 9)
        self.assertEqual(failures["candidate_active_pressure_not_exactly_one"], 9)

    def test_fails_closed_on_event_id_and_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_dataset(root / "baseline", candidate=False)
            candidate = self._write_dataset(
                root / "candidate",
                candidate=True,
                id_drift=True,
                binding_drift=True,
            )
            report = audit_conditional_touch_smoke(
                baseline_manifest=baseline[0],
                candidate_manifest=candidate[0],
                expected_events=18,
            )
        failures = {
            row["code"]: row["count"]
            for row in report["hard_invariants"]["failures"]
        }
        self.assertEqual(failures["candidate_missing_event_ids"], 1)
        self.assertEqual(failures["candidate_extra_event_ids"], 1)
        self.assertEqual(failures["event_binding_mismatch"], 1)
        self.assertEqual(report["status"], "fail")

    def test_moving_tap_requires_strictly_bound_request_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_dataset(root / "baseline", candidate=False)
            valid = self._write_dataset(
                root / "valid",
                candidate=True,
                moving_tap=True,
                tap_request_plan="valid",
            )
            valid_report = audit_conditional_touch_smoke(
                baseline_manifest=baseline[0],
                candidate_manifest=valid[0],
                candidate_provenance_path=valid[2],
                expected_events=18,
            )
        self.assertEqual(valid_report["status"], "pass")
        invariants = valid_report["hard_invariants"]
        self.assertEqual(invariants["moving_tap_events"], 3)
        self.assertEqual(invariants["moving_tap_request_plans_verified"], 3)

    def test_moving_tap_rejects_missing_malformed_or_mismatched_plan(self) -> None:
        for variant in ("none", "malformed", "wrong_carrier"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline = self._write_dataset(root / "baseline", candidate=False)
                candidate = self._write_dataset(
                    root / "candidate",
                    candidate=True,
                    moving_tap=True,
                    tap_request_plan=variant,
                )
                report = audit_conditional_touch_smoke(
                    baseline_manifest=baseline[0],
                    candidate_manifest=candidate[0],
                    candidate_provenance_path=candidate[2],
                    expected_events=18,
                )
            failures = {
                row["code"]: row["count"]
                for row in report["hard_invariants"]["failures"]
            }
            self.assertEqual(report["status"], "fail")
            self.assertEqual(failures["candidate_request_plan_mismatch"], 3)
            self.assertNotIn("candidate_endpoint_mismatch", failures)

    def test_cli_help_and_failure_exit_are_explicit(self) -> None:
        help_text = build_argument_parser().format_help()
        self.assertIn("--candidate-provenance", help_text)
        self.assertIn("--expected-events", help_text)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write_dataset(root / "baseline", candidate=False)
            candidate = self._write_dataset(
                root / "candidate", candidate=True, bad_pressure=True
            )
            output = root / "audit.json"
            code = audit_main(
                [
                    "--baseline-manifest",
                    str(baseline[0]),
                    "--candidate-manifest",
                    str(candidate[0]),
                    "--candidate-provenance",
                    str(candidate[2]),
                    "--expected-events",
                    "18",
                    "--output",
                    str(output),
                ]
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(saved["status"], "fail")


if __name__ == "__main__":
    unittest.main()
