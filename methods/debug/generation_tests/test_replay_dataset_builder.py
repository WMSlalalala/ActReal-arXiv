from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import pipeline.replay_dataset_builder as replay_builder
from pipeline.conditional_touch_generator import GeneratedTouch
from pipeline.genuine_touch_recovery import GenuineTouchBinding
from pipeline.replay_dataset_builder import (
    CONDITIONAL_TOUCH_ACTIONS,
    EXACT_TOUCH_TEMPLATE_ACTIONS,
    InputShard,
    LoadedShard,
    RawTimingReference,
    RebuiltEventSignal,
    ReplayDatasetBuildError,
    RawWindowRatioSampler,
    TapDonor,
    TapReplayAllocator,
    _observe_conditional_touch_target,
    _action_target_anchor_px,
    _detector_window_duration_ms,
    _exact_touch_imu_provenance_is_valid,
    _fiveshot_timing_samples,
    _finalize_smoke_reference_audit,
    _is_hmog_ascii_letter_keycode,
    _load_smoke_reference_selection,
    _pair_id,
    _pinch_target_endpoint_geometry,
    _fiveshot_tap_drift_request,
    _single_pointer_target_control_points_px,
    _rebuild_reference_locked_smoke_shard,
    _validate_full_input_selection,
    _validate_donor_provenance,
    build_full_100k_replay_dataset,
    build_smoke_replay_dataset,
    load_android_target,
    observe_bound_android_target,
    _write_output_shard,
)


class ReplayDatasetBuilderTests(unittest.TestCase):
    @staticmethod
    def _single_pointer_target(
        *,
        action: str,
        x_px: tuple[float, ...],
        y_px: tuple[float, ...],
        android_action: tuple[int, ...],
        direction: str | None = None,
    ) -> replay_builder.AndroidTarget:
        count = len(x_px)
        t_ms = np.linspace(0.0, 20.0, count, dtype=np.float64)
        return replay_builder.AndroidTarget(
            action=action,
            orientation_id=0,
            direction=direction,
            pinch_scale_direction=None,
            keycodes=(),
            trajectory_source=Path("/not-opened/generated-target.npz"),
            trajectory_source_sha256="0" * 64,
            trajectory_archive_index=0,
            raw_duration_ms=20.0,
            t_ms=t_ms,
            x_px=np.asarray(x_px, dtype=np.float64),
            y_px=np.asarray(y_px, dtype=np.float64),
            pressure=np.full(count, 0.5, dtype=np.float64),
            pointer_id=np.zeros(count, dtype=np.int64),
            android_action=np.asarray(android_action, dtype=np.int64),
            key_index=np.full(count, -1, dtype=np.int64),
            frame_index=np.arange(count, dtype=np.int64),
            frame_end=np.ones(count, dtype=np.uint8),
            bound_event_plan_sha256="1" * 64,
        )

    @staticmethod
    def _donor_template(
        x_px: tuple[float, ...], y_px: tuple[float, ...]
    ) -> np.ndarray:
        """Build one nine-column detector-grid template from screen pixels."""

        count = len(x_px)
        values = np.zeros((count, 9), dtype=np.float32)
        values[:, 0] = 1.0
        values[:, 1] = np.asarray(x_px, dtype=np.float64) / 1080.0
        values[:, 2] = np.asarray(y_px, dtype=np.float64) / 1920.0
        values[:, 3] = 0.5
        values[:, 4] = 1.0
        values[1:, 5:7] = np.diff(values[:, 1:3], axis=0)
        values[:, 7] = np.arange(count, dtype=np.float32) / 100.0
        values[:, 8] = 1.0
        return values

    def _tap_target(self, x: float, y: float) -> replay_builder.AndroidTarget:
        return self._single_pointer_target(
            action="tap",
            x_px=(x, x, x),
            y_px=(y, y, y),
            android_action=(0, 2, 1),
        )

    def test_fiveshot_tap_request_carries_the_donor_drift(self) -> None:
        """The bound target reports one point; the donor supplies the drift."""

        target = self._tap_target(820.0, 755.0)
        start_px, end_px = _single_pointer_target_control_points_px(target)
        self.assertEqual(start_px, end_px)
        template = self._donor_template((300.0, 302.0, 306.0),
                                        (400.0, 401.0, 404.0))
        request, requested_end, direction, audit = _fiveshot_tap_drift_request(
            template, target, start_px=start_px
        )
        self.assertIs(request, template)
        self.assertEqual(direction, "down_right")
        self.assertEqual(audit["tap_donor_drift_scale"], 1.0)
        expected_drift = float(np.hypot(6.0, 4.0))
        self.assertAlmostEqual(audit["tap_donor_drift_px"], expected_drift, 4)
        self.assertAlmostEqual(
            audit["tap_requested_drift_px"], expected_drift, 4
        )
        self.assertNotEqual(requested_end, start_px)
        # The DOWN anchor is untouched, so the gesture still reaches the view
        # Android hit-tests, and the chord is the donor's own.
        self.assertAlmostEqual(requested_end[0] - start_px[0], 6.0, places=4)
        self.assertAlmostEqual(requested_end[1] - start_px[1], 4.0, places=4)

    def test_fiveshot_tap_request_holds_still_for_a_still_donor(self) -> None:
        target = self._tap_target(820.0, 755.0)
        start_px, _ = _single_pointer_target_control_points_px(target)
        template = self._donor_template((300.0, 305.0, 300.0),
                                        (400.0, 400.0, 400.0))
        request, requested_end, direction, audit = _fiveshot_tap_drift_request(
            template, target, start_px=start_px
        )
        self.assertIs(request, template)
        self.assertIsNone(direction)
        self.assertEqual(requested_end, start_px)
        self.assertEqual(audit["tap_donor_drift_px"], 0.0)
        self.assertEqual(audit["tap_requested_drift_px"], 0.0)

    def test_fiveshot_tap_request_shrinks_past_the_touch_slop(self) -> None:
        """A donor beyond the slop budget is shrunk, never flattened."""

        target = self._tap_target(820.0, 755.0)
        start_px, _ = _single_pointer_target_control_points_px(target)
        template = self._donor_template((300.0, 320.0, 340.0),
                                        (400.0, 400.0, 400.0))
        request, requested_end, direction, audit = _fiveshot_tap_drift_request(
            template, target, start_px=start_px
        )
        self.assertEqual(direction, "right")
        self.assertLess(audit["tap_donor_drift_scale"], 1.0)
        self.assertAlmostEqual(audit["tap_donor_drift_px"], 40.0, places=3)
        limit = replay_builder.FIVESHOT_TAP_DRIFT_LIMIT_PX
        margin = replay_builder.FIVESHOT_TAP_SCREEN_MARGIN_PX
        self.assertLess(audit["tap_requested_drift_px"], limit)
        self.assertGreater(audit["tap_requested_drift_px"], limit - 2.0 * margin)
        # Shrinking keeps the donor's held shape: the midpoint stays at half the
        # chord and dx/dy still describe the coordinates they were rebuilt from.
        scaled = request[:, 1:3].astype(np.float64) * (1080.0, 1920.0)
        self.assertAlmostEqual(
            float(scaled[1, 0] - scaled[0, 0]),
            float(scaled[2, 0] - scaled[0, 0]) / 2.0,
            places=4,
        )
        expected = np.zeros((3, 2), dtype=np.float32)
        expected[1:] = np.diff(request[:, 1:3], axis=0)
        np.testing.assert_array_equal(request[:, 5:7], expected)

    def test_fiveshot_tap_request_keeps_every_sample_on_screen(self) -> None:
        target = self._tap_target(1079.0, 755.0)
        start_px, _ = _single_pointer_target_control_points_px(target)
        template = self._donor_template((300.0, 305.0, 310.0),
                                        (400.0, 400.0, 400.0))
        request, requested_end, _, audit = _fiveshot_tap_drift_request(
            template, target, start_px=start_px
        )
        self.assertLess(audit["tap_donor_drift_scale"], 1.0)
        self.assertLessEqual(requested_end[0], 1080.0)
        translated = (
            request[:, 1:3].astype(np.float64) * (1080.0, 1920.0)
        )
        translated = translated - translated[0] + np.asarray(start_px)
        self.assertTrue(np.all(translated[:, 0] <= 1080.0))

    def test_single_pointer_controls_use_down_and_up_semantics(self) -> None:
        target = self._single_pointer_target(
            action="swipe",
            x_px=(999.0, 100.0, 300.0, 500.0, 888.0),
            y_px=(999.0, 400.0, 400.0, 400.0, 888.0),
            android_action=(2, 0, 2, 1, 2),
            direction="right",
        )
        self.assertEqual(
            _single_pointer_target_control_points_px(target),
            ((100.0, 400.0), (500.0, 400.0)),
        )
        with self.assertRaisesRegex(
            ReplayDatasetBuildError, "begin at DOWN and end at UP"
        ):
            _observe_conditional_touch_target(
                mock.Mock(),
                target,
                target_samples=3,
                target_duration_ms=20.0,
                generator_seed=7,
            )

    def test_smoke_reference_selection_freezes_exact_group_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen_sources: dict[str, list[InputShard]] = {
                split: [] for split in replay_builder.SPLITS
            }
            manifest_rows = []
            expected_first_ids: set[str] = set()
            provenance_rows: list[dict[str, object]] = []
            for split_index, split in enumerate(replay_builder.SPLITS):
                user_id = f"hmog_u{90 + split_index:03d}"
                event_ids: list[str] = []
                actions: list[str] = []
                labels: list[int] = []
                for action in replay_builder.ACTIONS:
                    for label in (0, 1):
                        for ordinal in range(2):
                            event_id = (
                                f"reference/{split}/{user_id}/{action}/"
                                f"{label}/{ordinal}"
                            )
                            event_ids.append(event_id)
                            actions.append(action)
                            labels.append(label)
                            expected_first_ids.add(event_id)
                            provenance_rows.append(
                                {
                                    "event_id": event_id,
                                    "donor": {
                                        "target_binding": {
                                            "orientation_id": 0,
                                        }
                                    },
                                }
                            )
                event_count = len(event_ids)
                shard_path = root / f"{split}.npz"
                np.savez_compressed(
                    shard_path,
                    schema_version=np.asarray(
                        replay_builder.INPUT_SHARD_SCHEMA
                    ),
                    coordinate_schema=np.asarray("screen_relative_v1"),
                    time_schema=np.asarray("elapsed_seconds_v1"),
                    scope=np.asarray(replay_builder.SMOKE_MANIFEST_SCOPE),
                    split=np.asarray(split),
                    imu_flat=np.zeros((event_count * 2, 6), dtype=np.float32),
                    trajectory_flat=np.zeros(
                        (event_count * 2, 9), dtype=np.float32
                    ),
                    offsets=np.arange(
                        0, event_count * 2 + 1, 2, dtype=np.int64
                    ),
                    label=np.asarray(labels, dtype=np.int8),
                    user_id=np.asarray([user_id] * event_count),
                    session_id=np.asarray(["s"] * event_count),
                    event_id=np.asarray(event_ids),
                    action=np.asarray(actions),
                    source_cluster_id=np.asarray(
                        [f"cluster-{index}" for index in range(event_count)]
                    ),
                    sample_idx=np.arange(event_count, dtype=np.int64),
                    cross_modal_pair_id=np.asarray(
                        [f"pair-{index}" for index in range(event_count)]
                    ),
                )
                digest = replay_builder.sha256_file(shard_path)
                shard_row = {
                    "actions": list(replay_builder.ACTIONS),
                    "events": event_count,
                    "fake": event_count // 2,
                    "genuine": event_count // 2,
                    "source": str(shard_path),
                    "source_sha256": digest,
                    "user_id": user_id,
                }
                chosen_sources[split].append(
                    InputShard(
                        split=split,
                        user_id=user_id,
                        path=shard_path,
                        sha256=digest,
                        manifest_row=shard_row,
                    )
                )
                manifest_rows.append(
                    {
                        "schema_version": replay_builder.INPUT_MANIFEST_SCHEMA,
                        "scope": replay_builder.SMOKE_MANIFEST_SCOPE,
                        "formal_result": False,
                        "split": split,
                        "events": event_count,
                        "fake_events": event_count // 2,
                        "genuine_events": event_count // 2,
                        "user_ids": [user_id],
                        "shards": [shard_row],
                    }
                )
            manifest = root / "reference_manifest.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in manifest_rows
                ),
                encoding="utf-8",
            )
            (root / "provenance.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in provenance_rows
                ),
                encoding="utf-8",
            )
            selection, audit, touch_requests, touch_templates = (
                _load_smoke_reference_selection(
                reference_manifest=manifest,
                chosen_sources=chosen_sources,
                events_per_label_action_user=2,
                )
            )
            selected_ids = {
                event_id
                for event_ids in selection.values()
                for event_id in event_ids
            }
            self.assertEqual(selected_ids, expected_first_ids)
            self.assertEqual(len(selection), 30)
            self.assertEqual(audit["frozen_events"], 60)
            self.assertTrue(audit["current_input_binding_exact_match"])
            self.assertEqual(len(touch_requests), 12)
            self.assertEqual(len(touch_templates), 12)
            with self.assertRaisesRegex(
                ReplayDatasetBuildError, "event count must exactly equal"
            ):
                _load_smoke_reference_selection(
                    reference_manifest=manifest,
                    chosen_sources=chosen_sources,
                    events_per_label_action_user=1,
                )

    def test_full_mode_forbids_smoke_reference_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            reference.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                ReplayDatasetBuildError, "forbidden in full_100k"
            ):
                build_full_100k_replay_dataset(
                    input_manifest=root / "missing-input.jsonl",
                    output_dir=root / "new-output",
                    smoke_reference_manifest=reference,
                )

    def test_reference_locked_rebuild_fails_closed_and_finalizes_audit(self) -> None:
        split = "train"
        user_id = "hmog_u090"
        selected_event_ids: list[str] = []
        actions: list[str] = []
        labels: list[int] = []
        selection = {}
        for action in replay_builder.ACTIONS:
            for label in (0, 1):
                event_id = f"frozen/{action}/{label}"
                selected_event_ids.append(event_id)
                actions.append(action)
                labels.append(label)
                selection[(split, user_id, action, label)] = (event_id,)
        fallback_event_id = "unfrozen/fallback/tap/0"
        event_ids = [*selected_event_ids, fallback_event_id]
        actions.append("tap")
        labels.append(0)
        source = InputShard(
            split=split,
            user_id=user_id,
            path=Path("/not-opened/reference-locked-input.npz"),
            sha256="0" * 64,
            manifest_row={},
        )
        arrays = {
            "label": np.asarray(labels, dtype=np.int8),
            "action": np.asarray(actions),
            "event_id": np.asarray(event_ids),
        }
        shard = LoadedShard(source=source, arrays=arrays)

        class StubContext:
            def __init__(self, fail_event_id: str | None) -> None:
                self.fail_event_id = fail_event_id
                self.calls: list[str] = []

            def rebuild(
                self, *, shard: LoadedShard, index: int
            ) -> tuple[RebuiltEventSignal, dict[str, object]]:
                event_id = str(shard.arrays["event_id"][index])
                self.calls.append(event_id)
                if event_id == self.fail_event_id:
                    raise ReplayDatasetBuildError("injected frozen failure")
                action = str(shard.arrays["action"][index])
                label = int(shard.arrays["label"][index])
                return (
                    RebuiltEventSignal(
                        imu=np.zeros((2, 6), dtype=np.float32),
                        trajectory=np.zeros((2, 9), dtype=np.float32),
                    ),
                    {
                        "event_id": event_id,
                        "split": split,
                        "user_id": user_id,
                        "action": action,
                        "label": label,
                    },
                )

        failing = StubContext(selected_event_ids[0])
        with self.assertRaisesRegex(
            ReplayDatasetBuildError,
            "replacement is forbidden",
        ):
            _rebuild_reference_locked_smoke_shard(
                context=failing,
                shard=shard,
                selection=selection,
                record_provenance=lambda row: None,
            )
        self.assertEqual(failing.calls, [selected_event_ids[0]])
        self.assertNotIn(fallback_event_id, failing.calls)

        successful = StubContext(None)
        provenance_rows: list[dict[str, object]] = []
        selected, signals, observed = _rebuild_reference_locked_smoke_shard(
            context=successful,
            shard=shard,
            selection=selection,
            record_provenance=provenance_rows.append,
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(signals), 10)
        self.assertEqual(len(provenance_rows), 10)
        self.assertNotIn(fallback_event_id, successful.calls)
        frozen_digest = replay_builder._smoke_reference_selection_sha256(
            selection
        )
        audit = _finalize_smoke_reference_audit(
            expected=selection,
            observed=observed,
            audit={
                "status": "frozen_pending_rebuild",
                "frozen_event_ids_sha256": frozen_digest,
            },
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["rebuilt_events"], 10)
        self.assertEqual(audit["output_event_ids_sha256"], frozen_digest)
        self.assertTrue(audit["output_event_ids_exact_match"])
        self.assertFalse(audit["replacement_after_rebuild_failure_used"])

    def test_conditional_touch_uses_exact_requested_geometry(self) -> None:
        class StubGenerator:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def generate(self, **kwargs: object) -> GeneratedTouch:
                self.requests.append(dict(kwargs))
                t_ms = np.asarray(kwargs["t_ms"], dtype=np.float64)
                start = np.asarray(kwargs["start_xy_px"], dtype=np.float64)
                end = np.asarray(kwargs["end_xy_px"], dtype=np.float64)
                u = np.linspace(0.0, 1.0, len(t_ms), dtype=np.float64)
                points = start[None, :] + u[:, None] * (end - start)[None, :]
                endpoint_equal = bool(np.array_equal(start, end))
                if kwargs["action"] == "tap" and not endpoint_equal and len(points) > 2:
                    points[1:-1, 1] += 2.5
                delta = end - start
                if endpoint_equal:
                    realized_direction = "stationary"
                else:
                    labels = (
                        "right",
                        "down_right",
                        "down",
                        "down_left",
                        "left",
                        "up_left",
                        "up",
                        "up_right",
                    )
                    angle = float(np.arctan2(delta[1], delta[0]))
                    realized_direction = labels[
                        int(
                            np.floor(
                                (angle + np.pi / 8.0) / (np.pi / 4.0)
                            )
                        )
                        % len(labels)
                    ]
                frame_index = np.arange(len(t_ms), dtype=np.int64)
                actions = np.full(len(t_ms), 2, dtype=np.int64)
                actions[0] = 0
                actions[-1] = 1
                return GeneratedTouch(
                    action=str(kwargs["action"]),
                    orientation_id=int(kwargs["orientation_id"]),
                    requested_direction=kwargs["direction"],
                    realized_direction=realized_direction,
                    t_ms=t_ms,
                    x_px=points[:, 0],
                    y_px=points[:, 1],
                    pressure=np.full(len(t_ms), 0.5, dtype=np.float64),
                    pointer_id=np.zeros(len(t_ms), dtype=np.int64),
                    android_action=actions,
                    frame_index=frame_index,
                    frame_end=np.ones(len(t_ms), dtype=np.bool_),
                    residual_scale=1.0,
                    tap_stationary_branch=(
                        kwargs["action"] == "tap" and endpoint_equal
                    ),
                )

        generator = StubGenerator()
        cases = (
            self._single_pointer_target(
                action="tap",
                x_px=(123.25, 126.0, 130.0),
                y_px=(456.75, 457.0, 460.0),
                android_action=(0, 2, 1),
            ),
            self._single_pointer_target(
                action="swipe",
                x_px=(100.25, 250.0, 500.75),
                y_px=(600.5, 600.5, 600.5),
                android_action=(0, 2, 1),
                direction="right",
            ),
        )
        for target in cases:
            observation, details = _observe_conditional_touch_target(
                generator,
                target,
                target_samples=7,
                target_duration_ms=60.0,
                generator_seed=19,
            )
            start, lifecycle_end = _single_pointer_target_control_points_px(
                target
            )
            requested_end = lifecycle_end
            self.assertEqual(details["requested_start_px"], list(start))
            self.assertEqual(details["requested_end_px"], list(requested_end))
            self.assertLessEqual(details["maximum_endpoint_error_px"], 5.0e-4)
            self.assertEqual(observation.trajectory.shape, (7, 9))
        self.assertEqual(generator.requests[0]["start_xy_px"], (123.25, 456.75))
        self.assertEqual(generator.requests[0]["end_xy_px"], (130.0, 460.0))
        self.assertEqual(generator.requests[1]["start_xy_px"], (100.25, 600.5))
        self.assertEqual(generator.requests[1]["end_xy_px"], (500.75, 600.5))
        self.assertEqual(generator.requests[0]["detector_sample_count"], 7)
        self.assertEqual(generator.requests[1]["detector_sample_count"], 7)

    def test_conditional_touch_accepts_explicit_moving_tap_request(self) -> None:
        target = self._single_pointer_target(
            action="tap",
            x_px=(123.25, 126.0, 130.0),
            y_px=(456.75, 457.0, 460.0),
            android_action=(0, 2, 1),
        )

        class ExplicitTapGenerator:
            def __init__(self) -> None:
                self.request: dict[str, object] | None = None

            def generate(self, **kwargs: object) -> GeneratedTouch:
                self.request = dict(kwargs)
                t_ms = np.asarray(kwargs["t_ms"], dtype=np.float64)
                start = np.asarray(kwargs["start_xy_px"], dtype=np.float64)
                end = np.asarray(kwargs["end_xy_px"], dtype=np.float64)
                u = np.linspace(0.0, 1.0, len(t_ms), dtype=np.float64)
                points = start[None, :] + u[:, None] * (end - start)[None, :]
                actions = np.full(len(t_ms), 2, dtype=np.int64)
                actions[0] = 0
                actions[-1] = 1
                return GeneratedTouch(
                    action="tap",
                    orientation_id=0,
                    requested_direction=kwargs["direction"],
                    realized_direction="right",
                    t_ms=t_ms,
                    x_px=points[:, 0],
                    y_px=points[:, 1],
                    pressure=np.full(len(t_ms), 0.5, dtype=np.float64),
                    pointer_id=np.zeros(len(t_ms), dtype=np.int64),
                    android_action=actions,
                    frame_index=np.arange(len(t_ms), dtype=np.int64),
                    frame_end=np.ones(len(t_ms), dtype=np.bool_),
                    residual_scale=1.0,
                    tap_stationary_branch=False,
                )

        generator = ExplicitTapGenerator()
        observation, details = _observe_conditional_touch_target(
            generator,
            target,
            target_samples=7,
            target_duration_ms=60.0,
            generator_seed=29,
            requested_start_xy=(200.25, 300.5),
            requested_end_xy=(212.25, 300.5),
            requested_direction="right",
        )
        self.assertIsNotNone(generator.request)
        self.assertEqual(generator.request["start_xy_px"], (200.25, 300.5))
        self.assertEqual(generator.request["end_xy_px"], (212.25, 300.5))
        self.assertEqual(generator.request["direction"], "right")
        self.assertEqual(details["requested_start_px"], [200.25, 300.5])
        self.assertEqual(details["requested_end_px"], [212.25, 300.5])
        self.assertEqual(details["realized_direction"], "right")
        self.assertFalse(details["tap_stationary_branch"])
        self.assertLessEqual(details["maximum_endpoint_error_px"], 5.0e-4)
        self.assertEqual(observation.trajectory.shape, (7, 9))

    def test_conditional_touch_rejects_partial_or_mismatched_explicit_request(self) -> None:
        target = self._single_pointer_target(
            action="tap",
            x_px=(123.25, 126.0, 130.0),
            y_px=(456.75, 457.0, 460.0),
            android_action=(0, 2, 1),
        )
        with self.assertRaisesRegex(
            ReplayDatasetBuildError,
            "requires both start and end",
        ):
            _observe_conditional_touch_target(
                mock.Mock(),
                target,
                target_samples=7,
                target_duration_ms=60.0,
                generator_seed=29,
                requested_start_xy=(200.25, 300.5),
            )
        with self.assertRaisesRegex(
            ReplayDatasetBuildError,
            "tap direction does not match requested endpoints",
        ):
            _observe_conditional_touch_target(
                mock.Mock(),
                target,
                target_samples=7,
                target_duration_ms=60.0,
                generator_seed=29,
                requested_start_xy=(200.25, 300.5),
                requested_end_xy=(212.25, 300.5),
                requested_direction="left",
            )

    def test_hmog_letter_keycode_uses_ascii_not_android_range(self) -> None:
        for code in (65, 90, 97, 122):
            self.assertTrue(_is_hmog_ascii_letter_keycode(code))
        for code in (-5, -1, 29, 32, 46, 54, 64, 91, 96, 123):
            self.assertFalse(_is_hmog_ascii_letter_keycode(code))

    def test_writer_rebuilds_offsets_for_variable_length_paired_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = InputShard(
                split="train",
                user_id="u1",
                path=root / "input.npz",
                sha256="0" * 64,
                manifest_row={},
            )
            old_lengths = (3, 4)
            old_offsets = np.asarray([0, 3, 7], dtype=np.int64)
            arrays = {
                "coordinate_schema": np.asarray("screen_relative_v1"),
                "time_schema": np.asarray("elapsed_seconds_v1"),
                "offsets": old_offsets,
                "imu_flat": np.zeros((sum(old_lengths), 6), dtype=np.float32),
                "trajectory_flat": np.zeros(
                    (sum(old_lengths), 9), dtype=np.float32
                ),
                "label": np.asarray([0, 1], dtype=np.int8),
                "user_id": np.asarray(["u1", "u1"]),
                "session_id": np.asarray(["s1", "s1"]),
                "event_id": np.asarray(["genuine", "fake-key"]),
                "action": np.asarray(["tap", "keystroke"]),
                "source_cluster_id": np.asarray(["c1", "c2"]),
                "sample_idx": np.asarray([0, 1], dtype=np.int64),
                "cross_modal_pair_id": np.asarray(["old1", "old2"]),
            }
            shard = LoadedShard(source=source, arrays=arrays)
            first_imu = np.ones((3, 6), dtype=np.float32)
            first_trajectory = np.ones((3, 9), dtype=np.float32)
            second_imu = np.full((5, 6), 2.0, dtype=np.float32)
            second_trajectory = np.full((5, 9), 3.0, dtype=np.float32)
            output = root / "output.npz"
            _write_output_shard(
                output_path=output,
                shard=shard,
                selected=[1, 0],
                signals={
                    0: RebuiltEventSignal(first_imu, first_trajectory),
                    1: RebuiltEventSignal(second_imu, second_trajectory),
                },
            )
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["offsets"], np.asarray([0, 3, 8])
                )
                self.assertEqual(archive["imu_flat"].shape, (8, 6))
                self.assertEqual(archive["trajectory_flat"].shape, (8, 9))
                np.testing.assert_array_equal(
                    archive["cross_modal_pair_id"].astype(str),
                    np.asarray(
                        [
                            _pair_id("genuine", first_imu, first_trajectory),
                            _pair_id("fake-key", second_imu, second_trajectory),
                        ]
                    ),
                )

            with self.assertRaisesRegex(
                ReplayDatasetBuildError, "exactly match"
            ):
                _write_output_shard(
                    output_path=root / "extra.npz",
                    shard=shard,
                    selected=[0],
                    signals={
                        0: RebuiltEventSignal(first_imu, first_trajectory),
                        1: RebuiltEventSignal(second_imu, second_trajectory),
                    },
                )
            with self.assertRaisesRegex(ReplayDatasetBuildError, "out of range"):
                _write_output_shard(
                    output_path=root / "negative.npz",
                    shard=shard,
                    selected=[-1],
                    signals={-1: RebuiltEventSignal(first_imu, first_trajectory)},
                )

    def _trajectory_source(
        self,
        root: Path,
        *,
        action: str,
        x: list[float],
        y: list[float],
        pointers: list[int],
        actions: list[int],
        frames: list[int],
        key_indices: list[int] | None = None,
        keycodes: list[int] | None = None,
    ) -> Path:
        count = len(x)
        path = root / f"{action}.npz"
        np.savez_compressed(
            path,
            action=np.asarray(action),
            duration_ms=np.asarray([100.0], dtype=np.float32),
            orientation_id=np.asarray([0], dtype=np.int8),
            user_id=np.asarray([0], dtype=np.int16),
            split_id=np.asarray([0], dtype=np.int8),
            sample_index=np.asarray([0], dtype=np.int32),
            event_plan_sha256=np.zeros((1, 32), dtype=np.uint8),
            clipped_point_count=np.asarray([0], dtype=np.int32),
            clipped_point_rate=np.asarray([0.0], dtype=np.float64),
            geometry_valid=np.asarray([1], dtype=np.uint8),
            geometry_outlier=np.asarray([0], dtype=np.uint8),
            geometry_exclusion_code=np.asarray([0], dtype=np.int8),
            pre_projection_oob_point_count=np.asarray([0], dtype=np.int32),
            pre_projection_oob_point_rate=np.asarray([0.0], dtype=np.float64),
            typed_target_dispatch_feasibility_gate_pass=np.asarray([True]),
            generated_dispatch_quality_gate_pass=np.asarray([True]),
            target_clipping_applied=np.asarray([False]),
            slot_dropped=np.asarray([False]),
            physical_clip_count=np.asarray([0], dtype=np.int32),
            physical_clip_rate=np.asarray([0.0], dtype=np.float64),
            physical_clipped_coordinate_value_count=np.asarray([0], dtype=np.int32),
            android_offsets=np.asarray([0, count], dtype=np.int64),
            flat_android_t_ms=np.linspace(0.0, 100.0, count, dtype=np.float32),
            flat_android_x=np.asarray(x, dtype=np.float32),
            flat_android_y=np.asarray(y, dtype=np.float32),
            flat_android_pressure=np.ones(count, dtype=np.float32),
            flat_android_pointer_id=np.asarray(pointers, dtype=np.int8),
            flat_android_action=np.asarray(actions, dtype=np.int16),
            flat_android_key_index=np.asarray(
                [-1] * count if key_indices is None else key_indices,
                dtype=np.int16,
            ),
            flat_android_keycode=np.asarray(
                [-1] * count if keycodes is None else keycodes,
                dtype=np.int32,
            ),
            flat_android_frame_index=np.asarray(frames, dtype=np.int32),
            flat_android_frame_end=np.ones(count, dtype=np.uint8),
        )
        return path

    def _joint(
        self,
        root: Path,
        *,
        event_id: str,
        action: str,
        source: Path,
        source_sha256: str | None = None,
    ) -> Path:
        path = root / f"{event_id}.npz"
        digest = (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if source_sha256 is None
            else source_sha256
        )
        np.savez_compressed(
            path,
            source_action_label=np.asarray(action),
            trajectory_source=np.asarray(str(source)),
            trajectory_source_sha256=np.asarray(digest),
            trajectory_archive_index=np.asarray(0, dtype=np.int32),
            orientation_id=np.asarray(0, dtype=np.int8),
            paired_sample_index=np.asarray(0, dtype=np.int32),
            shared_event_plan_sha256=np.asarray("0" * 64),
            physical_out_of_bounds_point_count=np.asarray(0, dtype=np.int32),
            physical_out_of_bounds_point_rate=np.asarray(0.0),
            physical_clip_count=np.asarray(0, dtype=np.int32),
            physical_clip_rate=np.asarray(0.0),
            physical_clipped_coordinate_value_count=np.asarray(0, dtype=np.int32),
            typed_target_dispatch_feasibility_gate_pass=np.asarray(True),
            generated_dispatch_quality_gate_pass=np.asarray(True),
            target_clipping_applied=np.asarray(False),
            slot_dropped=np.asarray(False),
        )
        return path

    def test_android_target_rejects_trajectory_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._trajectory_source(
                root,
                action="scroll",
                x=[100.0, 50.0],
                y=[100.0, 100.0],
                pointers=[0, 0],
                actions=[0, 1],
                frames=[0, 1],
            )
            self._joint(
                root,
                event_id="fake-scroll",
                action="scroll",
                source=source,
                source_sha256="a" * 64,
            )
            with self.assertRaisesRegex(
                ReplayDatasetBuildError, "source hash mismatch"
            ):
                load_android_target(
                    event_id="fake-scroll",
                    action="scroll",
                    target_duration_ms=100.0,
                    joint_events_root=root,
                    trajectory_cache={},
                )

    def test_scroll_direction_is_derived_from_bound_android_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._trajectory_source(
                root,
                action="scroll",
                x=[100.0, 50.0],
                y=[100.0, 100.0],
                pointers=[0, 0],
                actions=[0, 1],
                frames=[0, 1],
            )
            self._joint(root, event_id="fake-scroll", action="scroll", source=source)
            target = load_android_target(
                event_id="fake-scroll",
                action="scroll",
                target_duration_ms=100.0,
                joint_events_root=root,
                trajectory_cache={},
            )
            self.assertEqual(target.direction, "left")
            self.assertEqual(target.orientation_id, 0)
            self.assertEqual(_action_target_anchor_px(target), (100.0, 100.0))

    def test_pinch_target_ignores_trailing_single_pointer_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._trajectory_source(
                root,
                action="pinch",
                x=[10.0, 20.0, 10.0, 30.0, 900.0],
                y=[10.0, 10.0, 10.0, 10.0, 900.0],
                pointers=[0, 1, 0, 1, 0],
                actions=[0, 5, 2, 2, 1],
                frames=[0, 0, 1, 1, 2],
            )
            self._joint(root, event_id="fake-pinch", action="pinch", source=source)
            target = load_android_target(
                event_id="fake-pinch",
                action="pinch",
                target_duration_ms=100.0,
                joint_events_root=root,
                trajectory_cache={},
            )
            self.assertEqual(target.direction, "stationary")
            self.assertEqual(target.pinch_scale_direction, "out")
            self.assertEqual(_action_target_anchor_px(target), (15.0, 10.0))
            endpoints = _pinch_target_endpoint_geometry(target)
            self.assertEqual(
                endpoints.start_points_px,
                ((10.0, 10.0), (20.0, 10.0)),
            )
            self.assertEqual(
                endpoints.end_points_px,
                ((10.0, 10.0), (30.0, 10.0)),
            )

    def test_keystroke_keycodes_keep_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._trajectory_source(
                root,
                action="keystroke",
                x=[100.0, 100.0, 200.0, 200.0],
                y=[1500.0] * 4,
                pointers=[0] * 4,
                actions=[0, 1, 0, 1],
                frames=[0, 1, 2, 3],
                key_indices=[4, 4, 7, 7],
                keycodes=[97, 97, 98, 98],
            )
            self._joint(
                root, event_id="fake-keystroke", action="keystroke", source=source
            )
            target = load_android_target(
                event_id="fake-keystroke",
                action="keystroke",
                target_duration_ms=100.0,
                joint_events_root=root,
                trajectory_cache={},
            )
            self.assertEqual(target.keycodes, (97, 98))

    def test_tap_allocator_never_reuses_a_binding(self) -> None:
        bindings = []
        for index in range(200):
            bindings.append(
                GenuineTouchBinding(
                    source_cluster_id=f"cluster-{index}",
                    user_id="1",
                    split="train",
                    session_id="s",
                    action="tap",
                    source_event_id=f"event-{index}",
                    source_event_ordinal=index,
                    orientation_id=0,
                    start_sample=0,
                    end_sample=11,
                    raw_trajectory_source=Path("/tmp/raw.npz"),
                    raw_trajectory_source_sha256="0" * 64,
                    raw_trajectory_event_index=index,
                    raw_event_sha256=f"{index:064x}",
                )
            )
        allocator = TapReplayAllocator(
            [TapDonor(binding=value, raw_duration_ms=100.0) for value in bindings],
            output_split="development",
            split_seed=42,
        )
        first = allocator.allocate(orientation_id=0, target_duration_ms=100.0)
        second = allocator.allocate(orientation_id=0, target_duration_ms=100.0)
        self.assertNotEqual(
            first.binding.source_event_id, second.binding.source_event_id
        )

    def test_duration_ratio_sampler_prefers_exact_geometry_then_orientation(self) -> None:
        sampler = RawWindowRatioSampler(
            exact_groups={
                ("scroll", 0, "right", "-"): [1.1] * 8,
            },
            orientation_groups={("scroll", 0): [1.4, 1.5]},
            raw_duration_by_cluster={"one": 100.0},
            raw_archive_cache={},
        )
        exact = sampler.sample(
            action="scroll",
            orientation_id=0,
            direction="right",
            event_id="a",
            seed=42,
        )
        fallback = sampler.sample(
            action="scroll",
            orientation_id=0,
            direction="left",
            event_id="b",
            seed=42,
        )
        self.assertEqual(
            exact.conditioning, "action_orientation_direction_pinch_scale"
        )
        self.assertAlmostEqual(exact.raw_to_window_ratio, 1.1)
        self.assertEqual(fallback.conditioning, "action_orientation")
        self.assertIn(fallback.raw_to_window_ratio, (1.4, 1.5))

    def test_keystroke_timing_reference_is_key_count_feasible(self) -> None:
        sampler = RawWindowRatioSampler(
            exact_groups={},
            orientation_groups={
                ("keystroke", 0): [
                    RawTimingReference(0.1, 100.0, "bad", key_count=10),
                    RawTimingReference(2.0, 80.0, "good", key_count=10),
                ]
            },
            raw_duration_by_cluster={"bad": 100.0, "good": 2000.0},
            raw_archive_cache={},
        )
        selected = sampler.sample(
            action="keystroke",
            orientation_id=0,
            event_id="target",
            seed=42,
            target_duration_ms=1000.0,
            key_count=10,
        )
        self.assertEqual(selected.reference_source_cluster_id, "good")
        self.assertEqual(selected.reference_key_count, 10)
        self.assertEqual(
            selected.conditioning, "action_orientation_key_count_feasible"
        )

    def test_keystroke_carrier_keeps_duration_and_sample_count_coupled(self) -> None:
        ten_seconds = RawTimingReference(
            1.0,
            50.0,
            "ten-seconds",
            key_count=10,
            raw_duration_ms=10_000.0,
            window_duration_ms=10_000.0,
            window_sample_count=512,
            observable_update_count=500,
        )
        sampler = RawWindowRatioSampler(
            exact_groups={},
            orientation_groups={("keystroke", 0): [ten_seconds]},
            raw_duration_by_cluster={"ten-seconds": 10_000.0},
            raw_archive_cache={},
        )
        selected = sampler.sample_keystroke_carrier(
            orientation_id=0,
            key_count=10,
            event_id="target",
            seed=42,
        )
        self.assertEqual(selected.reference_raw_duration_ms, 10_000.0)
        self.assertEqual(selected.reference_window_duration_ms, 10_000.0)
        self.assertEqual(selected.reference_window_sample_count, 512)
        self.assertEqual(selected.reference_observable_update_count, 500)
        self.assertEqual(
            selected.conditioning,
            "action_orientation_key_count_complete_carrier",
        )

        unconditioned = sampler.sample_keystroke_carrier(
            orientation_id=0,
            key_count=None,
            event_id="new-target-sequence",
            seed=42,
        )
        self.assertEqual(unconditioned.reference_key_count, 10)
        self.assertEqual(
            unconditioned.conditioning,
            "action_orientation_complete_carrier",
        )

    def test_detector_duration_uses_endpoint_and_raw_fallback_only_when_given(self) -> None:
        trajectory = np.zeros((8, 9), dtype=np.float32)
        with self.assertRaisesRegex(ReplayDatasetBuildError, "endpoint"):
            _detector_window_duration_ms(trajectory)
        self.assertEqual(
            _detector_window_duration_ms(
                trajectory, fallback_duration_ms=2501.0
            ),
            2501.0,
        )
        trajectory[:, 7] = np.linspace(0.0, 0.06, 8)
        self.assertAlmostEqual(
            _detector_window_duration_ms(trajectory), 60.0, places=3
        )

    def test_bound_tap_uses_common_zoh_and_binding_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._trajectory_source(
                root,
                action="tap",
                x=[100.0, 200.0],
                y=[300.0, 300.0],
                pointers=[0, 0],
                actions=[0, 1],
                frames=[0, 1],
            )
            self._joint(root, event_id="fake-tap", action="tap", source=source)
            target = load_android_target(
                event_id="fake-tap",
                action="tap",
                target_duration_ms=60.0,
                joint_events_root=root,
                trajectory_cache={},
                expected_user_id="hmog_u000",
                expected_split="train",
                expected_sample_idx=0,
                expected_cross_modal_pair_id="0" * 64,
            )
            observed = observe_bound_android_target(
                target, target_samples=7, target_duration_ms=60.0
            )
            self.assertAlmostEqual(float(observed.trajectory[-1, 7]), 0.06)
            active_x = observed.trajectory[observed.trajectory[:, 0] > 0, 1]
            self.assertLessEqual(len(np.unique(active_x)), 2)
            with self.assertRaisesRegex(
                ReplayDatasetBuildError, "cross-modal binding"
            ):
                load_android_target(
                    event_id="fake-tap",
                    action="tap",
                    target_duration_ms=60.0,
                    joint_events_root=root,
                    trajectory_cache={},
                    expected_user_id="hmog_u000",
                    expected_split="train",
                    expected_sample_idx=0,
                    expected_cross_modal_pair_id="1" * 64,
                )

    def test_provenance_rejects_cross_split_donor_family(self) -> None:
        rows = []
        for split in ("train", "test"):
            rows.append(
                {
                    "label": 1,
                    "split": split,
                    "rebuild_method": "train_raw_tap_replay",
                    "donor": {"source_event_id": "same"},
                }
            )
        with self.assertRaisesRegex(ReplayDatasetBuildError, "leakage"):
            _validate_donor_provenance(
                rows, selected_genuine_source_event_ids=set()
            )

    def test_bound_tap_provenance_is_not_counted_as_human_donor(self) -> None:
        result = _validate_donor_provenance(
            [
                {
                    "label": 1,
                    "split": "train",
                    "rebuild_method": "bound_fake_tap_android_zoh",
                    "donor": {"human_replay": False},
                }
            ],
            selected_genuine_source_event_ids=set(),
        )
        self.assertEqual(result["tap_primitives"], 0)

    def test_conditional_touch_dispatch_is_retired(self) -> None:
        """No action may be routed to the conditional generator any more."""

        self.assertEqual(CONDITIONAL_TOUCH_ACTIONS, frozenset())
        self.assertEqual(
            EXACT_TOUCH_TEMPLATE_ACTIONS,
            frozenset(("tap", "scroll", "swipe")),
        )
        row = {
            "label": 1,
            "split": "train",
            "action": "scroll",
            "input_imu_sha256": "same-imu",
            "output_imu_sha256": "same-imu",
            "rebuild_method": "conditional_touch_generator_model",
            "donor": {
                "role": "frozen_conditional_touch_generator_model",
                "human_replay": False,
                "runtime_donor_used": False,
                "model_used": True,
            },
        }
        with self.assertRaises(ReplayDatasetBuildError):
            _validate_donor_provenance(
                [row], selected_genuine_source_event_ids=set()
            )
    def test_isometric_action_provenance_is_audited_as_one_donor(self) -> None:
        donor = {
            "primitive_id": "primitive-one",
            "source_event_id": "human-event-one",
            "spatial_transform_name": "rotate_90",
            "spatial_matrix_xy": [[0, -1], [1, 0]],
            "translation_px": [100.0, 200.0],
            "requested_anchor_px": [300.0, 400.0],
            "output_anchor_px": [299.8, 400.1],
            "time_warp_ratio": 1.0,
            "scale_used": False,
            "coordinate_clipping_used": False,
            "anchor_error_px": 0.25,
        }
        result = _validate_donor_provenance(
            [
                {
                    "label": 1,
                    "split": "train",
                    "rebuild_method": "train_raw_action_isometric_replay",
                    "donor": donor,
                }
            ],
            selected_genuine_source_event_ids=set(),
        )
        self.assertEqual(result["action_primitives"], 1)

        malformed = dict(donor)
        malformed["time_warp_ratio"] = 1.01
        with self.assertRaisesRegex(
            ReplayDatasetBuildError, "provenance is incomplete"
        ):
            _validate_donor_provenance(
                [
                    {
                        "label": 1,
                        "split": "train",
                        "rebuild_method": "train_raw_action_isometric_replay",
                        "donor": malformed,
                    }
                ],
                selected_genuine_source_event_ids=set(),
            )

    def test_pinch_endpoint_provenance_is_audited_as_one_donor(self) -> None:
        donor = {
            "role": "train_raw_action_pinch_endpoint_replay",
            "primitive_id": "primitive-pinch",
            "source_event_id": "human-pinch",
            "pinch_source_start_points_px": [[100.0, 500.0], [300.0, 500.0]],
            "pinch_source_end_points_px": [[200.0, 500.0], [600.0, 500.0]],
            "pinch_requested_start_points_px": [
                [200.0, 600.0],
                [400.0, 600.0],
            ],
            "pinch_requested_end_points_px": [
                [300.0, 580.0],
                [300.0, 1020.0],
            ],
            "pinch_center_scale": 1.0,
            "pinch_start_span_scale": 1.0,
            "pinch_end_span_scale": 1.1,
            "pinch_deformation_score": float(np.log(1.1)),
            "pinch_minimum_scale": 0.80,
            "pinch_maximum_scale": 1.25,
            "pinch_start_endpoint_error_px": 0.0,
            "pinch_end_endpoint_error_px": 0.0,
            "pinch_maximum_endpoint_error_px": 0.0,
            "spatial_transform_name": "pinch_bounded_endpoint_residual",
            "spatial_matrix_xy": None,
            "translation_px": [100.0, 100.0],
            "requested_anchor_px": [300.0, 600.0],
            "requested_endpoint_px": [300.0, 800.0],
            "time_warp_ratio": 1.0,
            "scale_used": True,
            "spatial_scale": 1.0,
            "coordinate_clipping_used": False,
            "pixel_lattice_correction_used": False,
        }
        result = _validate_donor_provenance(
            [
                {
                    "label": 1,
                    "split": "train",
                    "rebuild_method": (
                        "train_raw_action_pinch_endpoint_replay"
                    ),
                    "donor": donor,
                }
            ],
            selected_genuine_source_event_ids=set(),
        )
        self.assertEqual(result["action_primitives"], 1)

        malformed = dict(donor)
        malformed["pinch_end_span_scale"] = 1.5
        with self.assertRaisesRegex(
            ReplayDatasetBuildError, "pinch endpoint replay provenance"
        ):
            _validate_donor_provenance(
                [
                    {
                        "label": 1,
                        "split": "train",
                        "rebuild_method": (
                            "train_raw_action_pinch_endpoint_replay"
                        ),
                        "donor": malformed,
                    }
                ],
                selected_genuine_source_event_ids=set(),
            )

    def test_full_selection_rejects_non_70_10_20_user_policy_first(self) -> None:
        shards = {"train": [], "development": [], "test": []}
        shards["train"] = [
            InputShard(
                split="train",
                user_id=f"hmog_u{index:03d}",
                path=Path(f"/missing/{index}.npz"),
                sha256="0" * 64,
                manifest_row={},
            )
            for index in range(69)
        ]
        with self.assertRaisesRegex(ReplayDatasetBuildError, "user policy"):
            _validate_full_input_selection(shards)

    def test_existing_output_is_rejected_before_input_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayDatasetBuildError, "already exists"):
                build_smoke_replay_dataset(
                    input_manifest=Path(directory) / "missing.jsonl",
                    output_dir=directory,
                )


class FiveShotGestureTimingProvenanceTest(unittest.TestCase):
    """A gesture that changes length has to account for the inertia it kept."""

    def _row(self, **timing: object) -> dict[str, object]:
        window = {
            "carrier_window_samples": 179,
            "carrier_active_samples": 16,
            "carrier_active_span": [80, 96],
            "requested_samples": 90,
            "cut_span": [43, 133],
            "capped_to_window": False,
        }
        plan = {
            "requested_travel_px": 31.2,
            "law_source_event_ids": ["s0", "s1", "s2", "s3", "s4"],
            "carrier_duration_ms": 60.0,
            "carrier_sample_count": 8,
            "requested_samples": 90,
            "capped_to_carrier_window": False,
            "carrier_imu_source": "/cache/user_000/scroll/train/sample_0000.npz",
            "carrier_imu_window": window,
            "spread_policy": "loo_residual",
            "law_log_offset": -0.4137,
            "law_residual_draw": 8123456789,
            "law_residual_index": 4,
            "law_residual_spread": 0.6287,
            "floored_to_reportable": False,
        }
        plan.update(timing)
        return {
            "input_imu_sha256": "a" * 64,
            "output_imu_sha256": "b" * 64,
            "output_samples": 90,
            "donor": {"target_binding": {"fiveshot_gesture_timing": plan}},
        }

    def test_a_gesture_keeping_the_carrier_inertia_must_keep_it_exactly(
        self,
    ) -> None:
        row = {
            "input_imu_sha256": "a" * 64,
            "output_imu_sha256": "a" * 64,
            "donor": {"target_binding": {}},
        }
        self.assertTrue(_exact_touch_imu_provenance_is_valid(row))
        row["output_imu_sha256"] = "b" * 64
        self.assertFalse(_exact_touch_imu_provenance_is_valid(row))

    def test_a_retimed_gesture_accounts_for_the_window_it_was_cut_from(
        self,
    ) -> None:
        self.assertTrue(_exact_touch_imu_provenance_is_valid(self._row()))

    def test_a_cut_that_does_not_match_the_rows_written_is_refused(self) -> None:
        row = self._row()
        row["output_samples"] = 89
        self.assertFalse(_exact_touch_imu_provenance_is_valid(row))

    def test_a_cut_reaching_outside_its_window_is_refused(self) -> None:
        row = self._row()
        row["donor"]["target_binding"]["fiveshot_gesture_timing"][
            "carrier_imu_window"
        ]["cut_span"] = [120, 210]
        self.assertFalse(_exact_touch_imu_provenance_is_valid(row))

    def test_an_unclaimed_cap_is_refused(self) -> None:
        """Capping is the one case the law does not get what it asked for."""

        row = self._row(requested_samples=200)
        self.assertFalse(_exact_touch_imu_provenance_is_valid(row))
        row = self._row(requested_samples=200, capped_to_carrier_window=True)
        self.assertTrue(_exact_touch_imu_provenance_is_valid(row))

    def test_a_law_built_on_one_recording_is_refused(self) -> None:
        row = self._row(law_source_event_ids=["s0"])
        self.assertFalse(_exact_touch_imu_provenance_is_valid(row))

    def test_a_curve_only_build_may_not_have_moved_the_reading(self) -> None:
        """Only a spread build draws a departure, and it says which it drew."""

        curve = dict(spread_policy="none", law_log_offset=0.0, law_residual_index=-1)
        self.assertTrue(_exact_touch_imu_provenance_is_valid(self._row(**curve)))
        self.assertFalse(
            _exact_touch_imu_provenance_is_valid(
                self._row(**{**curve, "law_log_offset": -0.4137})
            )
        )
        self.assertFalse(
            _exact_touch_imu_provenance_is_valid(
                self._row(**{**curve, "law_residual_index": 2})
            )
        )

    def test_only_a_spread_build_can_ask_for_an_unreportable_gesture(self) -> None:
        """A departure at the short end can undershoot the grid; the curve cannot."""

        self.assertTrue(
            _exact_touch_imu_provenance_is_valid(
                self._row(floored_to_reportable=True)
            )
        )
        self.assertFalse(
            _exact_touch_imu_provenance_is_valid(
                self._row(
                    spread_policy="none",
                    law_log_offset=0.0,
                    law_residual_index=-1,
                    floored_to_reportable=True,
                )
            )
        )

    def test_an_undeclared_spread_policy_is_refused(self) -> None:
        self.assertFalse(
            _exact_touch_imu_provenance_is_valid(self._row(spread_policy="jitter"))
        )


class FiveShotTimingSamplesTest(unittest.TestCase):
    def test_a_duration_and_a_row_count_are_the_same_statement(self) -> None:
        for duration, samples in ((60.0, 7), (890.0, 90), (1390.0, 140)):
            with self.subTest(duration=duration):
                self.assertEqual(_fiveshot_timing_samples(duration), samples)

    def test_a_duration_that_cannot_be_reported_is_refused(self) -> None:
        for duration in (0.0, -10.0, float("nan")):
            with self.subTest(duration=duration):
                with self.assertRaises(ReplayDatasetBuildError):
                    _fiveshot_timing_samples(duration)


if __name__ == "__main__":
    unittest.main()
