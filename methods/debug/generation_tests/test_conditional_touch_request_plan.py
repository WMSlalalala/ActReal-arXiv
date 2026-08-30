from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.conditional_touch_request_plan import (
    ConditionalTouchRequestPlan,
    ConditionalTouchRequestPlanError,
    IDENTITY_ROLE,
    PLAN_REPLAY_SEMANTICS,
    REQUEST_PLAN_SCHEMA,
    canonical_request_plan_sha256,
    load_request_plan_jsonl,
    raw_t_ms_sha256,
    validate_request_plan_against_binding,
    validate_request_plans,
)


class ConditionalTouchRequestPlanTest(unittest.TestCase):
    @staticmethod
    def _arguments(
        *, carrier_event_id: str = "hmog-composed-carrier-17", request_seed: int = 91
    ) -> dict[str, object]:
        return {
            "carrier_event_id": carrier_event_id,
            "original_event_plan_sha256": "a" * 64,
            "orientation_id": 0,
            "action": "scroll",
            "original_direction": "up",
            "original_down_xy_px": (240.0, 1500.0),
            "original_up_xy_px": (250.0, 900.0),
            "original_raw_t_ms": np.asarray(
                (1000.0, 1000.0, 1071.5, 1300.0), dtype=np.float64
            ),
            "original_raw_duration_ms": 300.0,
            "sampled_start_xy_px": (280.0, 1470.0),
            "sampled_end_xy_px": (310.0, 870.0),
            "sampled_direction": "up",
            "request_model_file_sha256": "b" * 64,
            "request_model_artifact_sha256": "c" * 64,
            "request_model_schema_version": (
                "conditional-touch-request-generator-v1-source-bound"
            ),
            "request_model_source_fingerprint_sha256": "d" * 64,
            "request_seed": request_seed,
        }

    def test_canonical_round_trip_binds_carrier_model_and_raw_timeline(self) -> None:
        arguments = self._arguments()
        plan = ConditionalTouchRequestPlan.create(**arguments)
        self.assertEqual(plan.schema_version, REQUEST_PLAN_SCHEMA)
        self.assertEqual(plan.identity_role, IDENTITY_ROLE)
        self.assertEqual(plan.plan_replay_semantics, PLAN_REPLAY_SEMANTICS)
        self.assertEqual(
            plan.original_raw_t_ms_sha256,
            raw_t_ms_sha256(arguments["original_raw_t_ms"]),
        )
        self.assertEqual(plan.original_raw_duration_ms, 300.0)
        self.assertEqual(len(plan.request_plan_sha256), 64)
        self.assertEqual(
            canonical_request_plan_sha256(plan), plan.request_plan_sha256
        )
        self.assertEqual(ConditionalTouchRequestPlan.from_json(plan.to_json()), plan)
        self.assertEqual(
            ConditionalTouchRequestPlan.from_json_dict(plan.to_json_dict()), plan
        )
        self.assertEqual(
            ConditionalTouchRequestPlan.create(**arguments).request_plan_sha256,
            plan.request_plan_sha256,
        )
        with self.assertRaises(FrozenInstanceError):
            plan.request_seed = 92  # type: ignore[misc]
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "canonical hash"
        ):
            replace(plan, request_seed=92)

    def test_every_payload_field_and_declared_digest_are_fail_closed(self) -> None:
        plan = ConditionalTouchRequestPlan.create(**self._arguments())
        original = plan.to_json_dict()
        updates: dict[str, object] = {
            "schema_version": "wrong-schema",
            "carrier_event_id": "different-carrier",
            "identity_role": "original_event",
            "plan_replay_semantics": "original_plan_replay",
            "original_event_plan_sha256": "e" * 64,
            "orientation_id": 1,
            "action": "swipe",
            "original_direction": "right",
            "original_down_xy_px": [241.0, 1500.0],
            "original_up_xy_px": [251.0, 900.0],
            "original_raw_t_ms_sha256": "f" * 64,
            "original_raw_duration_ms": 301.0,
            "sampled_start_xy_px": [281.0, 1470.0],
            "sampled_end_xy_px": [311.0, 870.0],
            "sampled_direction": "right",
            "request_model_file_sha256": "1" * 64,
            "request_model_artifact_sha256": "2" * 64,
            "request_model_schema_version": "different-model-schema",
            "request_model_source_fingerprint_sha256": "3" * 64,
            "request_seed": 92,
            "request_plan_sha256": "4" * 64,
        }
        self.assertEqual(set(updates), set(original))
        for field, replacement in updates.items():
            with self.subTest(field=field):
                tampered = json.loads(json.dumps(original))
                tampered[field] = replacement
                with self.assertRaises(ConditionalTouchRequestPlanError):
                    ConditionalTouchRequestPlan.from_json_dict(tampered)

        missing = dict(original)
        missing.pop("request_seed")
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "fields changed"
        ):
            ConditionalTouchRequestPlan.from_json_dict(missing)
        extra = dict(original)
        extra["unbound"] = True
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "fields changed"
        ):
            ConditionalTouchRequestPlan.from_json_dict(extra)

    def test_live_binding_detects_raw_timeline_or_model_tamper(self) -> None:
        arguments = self._arguments()
        plan = ConditionalTouchRequestPlan.create(**arguments)
        self.assertEqual(
            validate_request_plan_against_binding(plan, **arguments), plan
        )
        changed_time = dict(arguments)
        changed_time["original_raw_t_ms"] = (1000.0, 1000.0, 1072.5, 1300.0)
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "live carrier/model binding"
        ):
            validate_request_plan_against_binding(plan, **changed_time)
        changed_model = dict(arguments)
        changed_model["request_model_file_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "live carrier/model binding"
        ):
            validate_request_plan_against_binding(plan, **changed_model)

    def test_tap_plan_can_bind_a_learned_non_equal_full_pair(self) -> None:
        arguments = self._arguments()
        arguments.update(
            {
                "action": "tap",
                "original_direction": None,
                "original_down_xy_px": (500.0, 900.0),
                "original_up_xy_px": (500.0, 900.0),
                "sampled_start_xy_px": (470.0, 850.0),
                "sampled_end_xy_px": (476.0, 854.0),
                # Tap has no directional command even when its learned UP
                # point differs from DOWN.
                "sampled_direction": None,
            }
        )
        plan = ConditionalTouchRequestPlan.create(**arguments)
        self.assertEqual(plan.action, "tap")
        self.assertIsNone(plan.original_direction)
        self.assertIsNone(plan.sampled_direction)
        self.assertNotEqual(plan.sampled_start_xy_px, plan.sampled_end_xy_px)
        self.assertEqual(
            validate_request_plan_against_binding(plan, **arguments), plan
        )

    def test_collection_and_jsonl_require_unique_carrier_ids(self) -> None:
        first = ConditionalTouchRequestPlan.create(**self._arguments())
        second = ConditionalTouchRequestPlan.create(
            **self._arguments(carrier_event_id="hmog-composed-carrier-18", request_seed=92)
        )
        self.assertEqual(validate_request_plans((first, second)), (first, second))
        duplicate = ConditionalTouchRequestPlan.create(
            **self._arguments(carrier_event_id=first.carrier_event_id, request_seed=93)
        )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "carrier event ID is reused"
        ):
            validate_request_plans((first, duplicate))

        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "plans.jsonl"
            valid.write_text(
                first.to_json() + "\n" + second.to_json() + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_request_plan_jsonl(valid), (first, second))
            reused = Path(directory) / "reused.jsonl"
            reused.write_text(
                first.to_json() + "\n" + duplicate.to_json() + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ConditionalTouchRequestPlanError, "carrier event ID is reused"
            ):
                load_request_plan_jsonl(reused)

    def test_json_and_semantic_validation_reject_ambiguous_inputs(self) -> None:
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "duplicate JSON object key"
        ):
            ConditionalTouchRequestPlan.from_json(
                '{"schema_version":"x","schema_version":"y"}'
            )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "non-finite JSON number"
        ):
            ConditionalTouchRequestPlan.from_json('{"value":NaN}')
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "nondecreasing"
        ):
            ConditionalTouchRequestPlan.create(
                **{
                    **self._arguments(),
                    "original_raw_t_ms": (0.0, 2.0, 1.0),
                }
            )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "span exceeds"
        ):
            ConditionalTouchRequestPlan.create(
                **{
                    **self._arguments(),
                    "original_raw_duration_ms": 299.0,
                }
            )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "endpoint sector"
        ):
            ConditionalTouchRequestPlan.create(
                **{
                    **self._arguments(),
                    "sampled_direction": "right",
                }
            )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "physical screen"
        ):
            ConditionalTouchRequestPlan.create(
                **{
                    **self._arguments(),
                    "sampled_end_xy_px": (310.0, -1.0),
                }
            )
        with self.assertRaisesRegex(
            ConditionalTouchRequestPlanError, "JSON integer"
        ):
            ConditionalTouchRequestPlan.create(
                **{**self._arguments(), "request_seed": True}
            )


if __name__ == "__main__":
    unittest.main()
