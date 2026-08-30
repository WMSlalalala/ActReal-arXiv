from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.android_touch_observation import (
    screen_dimensions_for_orientation,
)
from pipeline.action_replay import (
    ActionReplayBank,
    ActionReplayError,
    ReplayBucket,
    ReplayGeometry,
    classify_replay_request,
    duration_bucket,
    observe_replay_primitive,
    transform_replay_rows,
    transport_detector_touch_template,
)
from pipeline.pinch_endpoint_control import endpoint_geometry_from_pairs


def _single_pointer_event(
    event_id: int,
    *,
    user_id: int = 0,
    duration_ms: int = 500,
    start_xy: tuple[float, float] = (100.0, 900.0),
    end_xy: tuple[float, float] = (400.0, 900.0),
    orientation_id: int = 0,
    complete: bool = True,
) -> dict[str, object]:
    middle = (np.asarray(start_xy) + np.asarray(end_xy)) / 2.0
    return {
        "event_id": event_id,
        "user_id": user_id,
        "session_id": 1 + event_id % 3,
        "orientation_id": orientation_id,
        "duration_ms": duration_ms,
        "t_ms": [0, duration_ms // 2, duration_ms],
        "frame_index": [0, 1, 2],
        "pointer_count": [1, 1, 1],
        "pointer_id": [0, 0, 0],
        "android_action": [0, 2, 1 if complete else 2],
        "x_px": [start_xy[0], float(middle[0]), end_xy[0]],
        "y_px": [start_xy[1], float(middle[1]), end_xy[1]],
        "pressure": [0.7, 0.8, 0.6],
        "size": [0.02, 0.03, 0.02],
    }


def _detector_touch_template(
    points_px: list[tuple[float, float]],
    *,
    orientation_id: int = 0,
    duration_ms: float = 400.0,
) -> np.ndarray:
    """Return one active detector-grid trajectory with coupled touch fields."""

    width, height = screen_dimensions_for_orientation(orientation_id)
    points = np.asarray(points_px, dtype=np.float64)
    trajectory = np.zeros((len(points), 9), dtype=np.float32)
    trajectory[:, 0] = 1.0
    trajectory[:, 1] = points[:, 0] / float(width)
    trajectory[:, 2] = points[:, 1] / float(height)
    trajectory[:, 3] = np.linspace(0.55, 0.75, len(points), dtype=np.float32)
    trajectory[:, 4] = 1.0
    trajectory[1:, 5:7] = np.diff(trajectory[:, 1:3], axis=0)
    trajectory[:, 7] = np.linspace(
        0.0, duration_ms / 1000.0, len(points), dtype=np.float32
    )
    trajectory[:, 8] = 1.0
    return trajectory


def _pinch_event(
    event_id: int,
    *,
    scale_direction: str,
    center_direction: str,
) -> dict[str, object]:
    if scale_direction == "out":
        start = ((100.0, 500.0), (300.0, 500.0))
        end = ((200.0, 500.0), (600.0, 500.0))
    else:
        start = ((500.0, 500.0), (900.0, 500.0))
        end = ((400.0, 500.0), (600.0, 500.0))
    if center_direction == "stationary":
        start = ((300.0, 500.0), (700.0, 500.0))
        end = ((400.0, 500.0), (600.0, 500.0))
    # Frame zero deliberately contains both ACTION_DOWN and POINTER_DOWN rows,
    # exactly as mixed-timestamp HMOG frames can.  The final primary UP remains
    # in the raw lifecycle but must not become a one-pointer pinch centroid.
    return {
        "event_id": event_id,
        "user_id": 0,
        "session_id": 10,
        "orientation_id": 0,
        "duration_ms": 120,
        "t_ms": [0, 0, 0, 50, 50, 100, 100, 120],
        "frame_index": [0, 0, 0, 1, 1, 2, 2, 3],
        "pointer_count": [1, 2, 2, 2, 2, 2, 2, 1],
        "pointer_id": [0, 0, 1, 0, 1, 0, 1, 0],
        "android_action": [0, 5, 5, 2, 2, 6, 6, 1],
        "x_px": [
            start[0][0],
            start[0][0],
            start[1][0],
            (start[0][0] + end[0][0]) / 2,
            (start[1][0] + end[1][0]) / 2,
            end[0][0],
            end[1][0],
            end[0][0],
        ],
        "y_px": [500.0] * 8,
        "pressure": [1.0] * 8,
        "size": [0.02] * 8,
    }


def _write_archive(path: Path, action: str, events: list[dict[str, object]]) -> None:
    offsets = [0]
    flat: dict[str, list[object]] = {
        "flat_t_rel_ms": [],
        "flat_frame_index": [],
        "flat_pointer_count": [],
        "flat_pointer_id": [],
        "flat_action_code": [],
        "flat_x": [],
        "flat_y": [],
        "flat_pressure": [],
        "flat_size": [],
        "flat_active_mask": [],
        "flat_valid_mask": [],
    }
    mapping = {
        "flat_t_rel_ms": "t_ms",
        "flat_frame_index": "frame_index",
        "flat_pointer_count": "pointer_count",
        "flat_pointer_id": "pointer_id",
        "flat_action_code": "android_action",
        "flat_x": "x_px",
        "flat_y": "y_px",
        "flat_pressure": "pressure",
        "flat_size": "size",
    }
    for event in events:
        count = len(event["t_ms"])  # type: ignore[arg-type]
        for destination, source in mapping.items():
            flat[destination].extend(event[source])  # type: ignore[arg-type]
        flat["flat_active_mask"].extend([1] * count)
        flat["flat_valid_mask"].extend([1] * count)
        offsets.append(offsets[-1] + count)
    np.savez(
        path,
        action_name=np.asarray(action),
        event_id=np.asarray([event["event_id"] for event in events], dtype=np.int64),
        user_id=np.asarray([event["user_id"] for event in events], dtype=np.int32),
        session_id=np.asarray([event["session_id"] for event in events], dtype=np.int16),
        orientation_id=np.asarray(
            [event["orientation_id"] for event in events], dtype=np.int8
        ),
        touch_duration_ms=np.asarray(
            [event["duration_ms"] for event in events], dtype=np.int32
        ),
        event_offsets=np.asarray(offsets, dtype=np.int64),
        flat_t_rel_ms=np.asarray(flat["flat_t_rel_ms"], dtype=np.int64),
        flat_frame_index=np.asarray(flat["flat_frame_index"], dtype=np.int32),
        flat_pointer_count=np.asarray(flat["flat_pointer_count"], dtype=np.int8),
        flat_pointer_id=np.asarray(flat["flat_pointer_id"], dtype=np.int16),
        flat_action_code=np.asarray(flat["flat_action_code"], dtype=np.int16),
        flat_x=np.asarray(flat["flat_x"], dtype=np.float32),
        flat_y=np.asarray(flat["flat_y"], dtype=np.float32),
        flat_pressure=np.asarray(flat["flat_pressure"], dtype=np.float32),
        flat_size=np.asarray(flat["flat_size"], dtype=np.float32),
        flat_active_mask=np.asarray(flat["flat_active_mask"], dtype=np.uint8),
        flat_valid_mask=np.asarray(flat["flat_valid_mask"], dtype=np.uint8),
    )


class ActionReplayTest(unittest.TestCase):
    def test_detector_touch_template_identity_is_bitwise_exact(self) -> None:
        template = _detector_touch_template(
            [(100.0, 900.0), (180.0, 875.0), (400.0, 900.0)]
        )
        width, height = screen_dimensions_for_orientation(0)
        dimensions = np.asarray((width, height), dtype=np.float64)
        start = template[0, 1:3].astype(np.float64) * dimensions
        end = template[-1, 1:3].astype(np.float64) * dimensions
        duration_ms = float(template[-1, 7] - template[0, 7]) * 1000.0

        transported, residual_scale = transport_detector_touch_template(
            template,
            action="swipe",
            orientation_id=0,
            start_xy_px=start,
            end_xy_px=end,
            direction="right",
            duration_ms=duration_ms,
        )

        np.testing.assert_array_equal(transported, template)
        self.assertFalse(np.shares_memory(transported, template))
        self.assertEqual(residual_scale, 1.0)

    def test_detector_touch_template_places_tap_at_arbitrary_xy_and_duration(
        self,
    ) -> None:
        template = _detector_touch_template(
            [
                (100.0, 500.0),
                (100.0, 500.0),
                (103.0, 498.0),
                (100.0, 500.0),
            ],
            duration_ms=80.0,
        )
        target = np.asarray((347.25, 812.5), dtype=np.float64)
        duration_ms = 137.25

        transported, residual_scale = transport_detector_touch_template(
            template,
            action="tap",
            orientation_id=0,
            start_xy_px=target,
            end_xy_px=target,
            direction=None,
            duration_ms=duration_ms,
        )

        width, height = screen_dimensions_for_orientation(0)
        points = transported[:, 1:3].astype(np.float64) * np.asarray(
            (width, height), dtype=np.float64
        )
        np.testing.assert_allclose(
            points[[0, -1]],
            np.repeat(target[None, :], 2, axis=0),
            rtol=0.0,
            atol=1.0e-4,
        )
        np.testing.assert_array_equal(points[1], points[0])
        self.assertAlmostEqual(float(transported[-1, 7]) * 1000.0, duration_ms, places=4)
        np.testing.assert_array_equal(transported[:, [0, 3, 4, 8]], template[:, [0, 3, 4, 8]])
        self.assertEqual(residual_scale, 1.0)

    def test_detector_touch_template_swipe_has_exact_endpoints_and_direction(
        self,
    ) -> None:
        template = _detector_touch_template(
            [
                (100.0, 900.0),
                (180.0, 875.0),
                (300.0, 915.0),
                (400.0, 900.0),
            ]
        )
        start = np.asarray((700.0, 1400.0), dtype=np.float64)
        end = np.asarray((660.0, 500.0), dtype=np.float64)

        transported, residual_scale = transport_detector_touch_template(
            template,
            action="swipe",
            orientation_id=0,
            start_xy_px=start,
            end_xy_px=end,
            direction="up",
            duration_ms=321.0,
        )

        width, height = screen_dimensions_for_orientation(0)
        points = transported[:, 1:3].astype(np.float64) * np.asarray(
            (width, height), dtype=np.float64
        )
        np.testing.assert_allclose(points[0], start, rtol=0.0, atol=1.0e-4)
        np.testing.assert_allclose(points[-1], end, rtol=0.0, atol=1.0e-4)
        self.assertTrue(np.all(points >= 0.0))
        self.assertTrue(np.all(points[:, 0] <= width))
        self.assertTrue(np.all(points[:, 1] <= height))
        self.assertAlmostEqual(float(transported[-1, 7]), 0.321, places=6)
        self.assertGreater(residual_scale, 0.0)

        with self.assertRaisesRegex(ActionReplayError, "direction conflicts"):
            transport_detector_touch_template(
                template,
                action="swipe",
                orientation_id=0,
                start_xy_px=start,
                end_xy_px=end,
                direction="down",
                duration_ms=321.0,
            )

    def test_train_only_exclusion_and_disjoint_output_pools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scroll.npz"
            events = [
                _single_pointer_event(
                    index,
                    start_xy=(100.0 + index, 900.0),
                    end_xy=(400.0 + index, 900.0),
                )
                for index in range(30)
            ]
            events.append(_single_pointer_event(100, user_id=9))
            events.append(_single_pointer_event(101, complete=False))
            _write_archive(path, "scroll", events)
            bank = ActionReplayBank.from_hmog_npz(
                path,
                train_user_ids=[0],
                allowed_source_event_ids=[*range(30), 101],
                excluded_source_event_ids=[0],
            )
            self.assertEqual(len(bank.descriptors), 29)
            self.assertTrue(all(item.source_user_id == 0 for item in bank.descriptors))
            self.assertNotIn("0", {item.source_event_id for item in bank.descriptors})
            self.assertEqual(bank.rejected_counts["non_train_user"], 1)
            self.assertEqual(bank.rejected_counts["incomplete_android_lifecycle"], 1)

            quality_bank = ActionReplayBank.from_hmog_npz(
                path,
                train_user_ids=[0],
                allowed_source_event_ids=[1, 2],
            )
            self.assertEqual(
                {item.source_event_id for item in quality_bank.descriptors},
                {"1", "2"},
            )
            self.assertEqual(quality_bank.rejected_counts["not_quality_accepted"], 29)

            pools = bank.partition_output_splits(seed=123)
            split_sets = {
                split: set(pools.primitive_ids(split))
                for split in ("train", "development", "test")
            }
            self.assertFalse(split_sets["train"] & split_sets["development"])
            self.assertFalse(split_sets["train"] & split_sets["test"])
            self.assertFalse(split_sets["development"] & split_sets["test"])
            self.assertEqual(set().union(*split_sets.values()), set(bank.primitive_ids()))

            bucket = bank.descriptors[0].bucket
            allocator = pools.allocator("test", seed=5)
            available = len(pools.primitive_ids("test", bucket))
            allocated = [allocator.allocate(bucket).primitive_id for _ in range(available)]
            self.assertEqual(len(allocated), len(set(allocated)))
            with self.assertRaises(ActionReplayError):
                allocator.allocate(bucket)

            # q0 has no donor in this fixture.  Request allocation keeps the
            # same orientation+direction and falls back to the adjacent q1
            # duration bucket within the explicit warp limit, without reuse.
            request_allocator = pools.allocator("train", seed=6)
            first = request_allocator.allocate_request(
                orientation_id=0,
                direction="right",
                target_duration_ms=400.0,
                max_time_warp=1.3,
            )
            second = request_allocator.allocate_request(
                orientation_id=0,
                direction="right",
                target_duration_ms=400.0,
                max_time_warp=1.3,
            )
            self.assertEqual(first.bucket.duration_bucket, "q1")
            self.assertNotEqual(first.primitive_id, second.primitive_id)

    def test_exact_translation_and_narrow_integer_time_warp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "swipe.npz"
            event = _single_pointer_event(
                1,
                duration_ms=100,
                start_xy=(100.0, 900.0),
                end_xy=(300.0, 900.0),
            )
            # Use a non-grid middle timestamp so zero-order hold is observable.
            event["t_ms"] = [0, 25, 100]
            _write_archive(path, "swipe", [event])
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            primitive_id = bank.descriptors[0].primitive_id
            raw = bank.raw_rows(primitive_id)

            exact = transform_replay_rows(raw)
            np.testing.assert_array_equal(exact.x_px, raw.x_px)
            np.testing.assert_array_equal(exact.y_px, raw.y_px)
            np.testing.assert_array_equal(exact.android_action, raw.android_action)
            np.testing.assert_array_equal(exact.pressure, raw.pressure)
            self.assertEqual(exact.time_warp_ratio, 1.0)

            shifted = transform_replay_rows(raw, translation_px=(10.0, -5.0))
            np.testing.assert_allclose(shifted.x_px, raw.x_px + 10.0)
            np.testing.assert_allclose(shifted.y_px, raw.y_px - 5.0)
            np.testing.assert_array_equal(shifted.frame_index, raw.frame_index)
            with self.assertRaises(ActionReplayError):
                transform_replay_rows(raw, translation_px=(-101.0, 0.0))
            with self.assertRaises(ActionReplayError):
                transform_replay_rows(raw, translation_px=(-100.0 - 1.0e-7, 0.0))

            warped = transform_replay_rows(raw, target_duration_ms=110)
            self.assertEqual(warped.t_ms[0], 0.0)
            self.assertEqual(warped.t_ms[-1], 110.0)
            self.assertTrue(np.all(np.diff(np.unique(warped.t_ms)) >= 1.0))
            np.testing.assert_array_equal(warped.x_px, raw.x_px)
            with self.assertRaises(ActionReplayError):
                transform_replay_rows(
                    raw,
                    target_duration_ms=110,
                    max_time_warp=1.05,
                )

            replay = observe_replay_primitive(
                bank,
                primitive_id,
                target_samples=11,
                replay_duration_ms=100.0,
                output_duration_ms=100.0,
            )
            np.testing.assert_array_equal(replay.rows.t_ms, raw.t_ms)
            # The middle update at 25 ms is held on the 30..90 ms grid.  No XY
            # interpolation is introduced by replay construction.
            self.assertEqual(replay.observation.source_updates, 3)
            self.assertGreater(
                np.sum(np.diff(replay.observation.touch[:, 1]) == 0.0),
                0,
            )

    def test_scroll_swipe_bucket_contract(self) -> None:
        self.assertEqual(duration_bucket("scroll", 401.0), "q0")
        self.assertEqual(duration_bucket("scroll", 402.0), "q1")
        self.assertEqual(duration_bucket("swipe", 394.0), "q2")
        bucket = ReplayBucket(
            action="scroll",
            orientation_id=0,
            direction="up_left",
            duration_bucket="q3",
        )
        self.assertEqual(bucket.direction, "up_left")
        with self.assertRaises(ActionReplayError):
            ReplayBucket(
                action="scroll",
                orientation_id=0,
                direction="stationary",
                duration_bucket="q0",
            )
        with self.assertRaises(ActionReplayError):
            ReplayBucket(
                action="pinch",
                orientation_id=0,
                direction="not_a_direction",
                pinch_scale_direction="in",
            )

        request = classify_replay_request(
            action="scroll",
            orientation_id=0,
            target_duration_ms=500,
            x_px=[300, 200, 100],
            y_px=[900, 800, 700],
        )
        self.assertEqual(request.direction, "up_left")
        self.assertEqual(request.target_duration_ms, 500.0)

    def test_pinch_uses_scale_and_center_direction_and_drops_single_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pinch.npz"
            _write_archive(
                path,
                "pinch",
                [
                    _pinch_event(1, scale_direction="out", center_direction="right"),
                    _pinch_event(2, scale_direction="in", center_direction="left"),
                    _pinch_event(3, scale_direction="in", center_direction="stationary"),
                ],
            )
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            self.assertTrue(
                all(item.observable_update_count == 3 for item in bank.descriptors)
            )
            self.assertTrue(
                all(item.observable_update_rate_hz == 25.0 for item in bank.descriptors)
            )
            observed = {
                (item.bucket.pinch_scale_direction, item.bucket.direction)
                for item in bank.descriptors
            }
            self.assertIn(("out", "right"), observed)
            self.assertIn(("in", "left"), observed)
            self.assertIn(("in", "stationary"), observed)

            target = _pinch_event(
                9,
                scale_direction="out",
                center_direction="right",
            )
            request = classify_replay_request(
                action="pinch",
                orientation_id=0,
                target_duration_ms=120,
                x_px=target["x_px"],
                y_px=target["y_px"],
                pointer_id=target["pointer_id"],
                android_action=target["android_action"],
                frame_index=target["frame_index"],
            )
            self.assertEqual(request.direction, "right")
            self.assertEqual(request.pinch_scale_direction, "out")

            primitive = next(
                item for item in bank.descriptors if item.bucket.direction == "right"
            )
            replay = observe_replay_primitive(
                bank,
                primitive.primitive_id,
                target_samples=13,
                replay_duration_ms=120.0,
                output_duration_ms=120.0,
            )
            self.assertEqual(replay.observation.source_updates, 3)
            self.assertTrue(np.all(replay.observation.touch[:, 4] == 2.0))
            # The complete raw ACTION_UP row is retained for audit/replay even
            # though the detector observation correctly excludes its centroid.
            self.assertEqual(int(replay.rows.android_action[-1]) & 0xFF, 1)

    def test_allocator_matches_detector_update_count_without_raw_duration_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scroll.npz"
            sparse = _single_pointer_event(1, duration_ms=500)
            dense = _single_pointer_event(2, duration_ms=520)
            dense.update(
                {
                    "t_ms": [0, 100, 200, 300, 400, 520],
                    "frame_index": [0, 1, 2, 3, 4, 5],
                    "pointer_count": [1] * 6,
                    "pointer_id": [0] * 6,
                    "android_action": [0, 2, 2, 2, 2, 1],
                    "x_px": np.linspace(100.0, 400.0, 6).tolist(),
                    "y_px": [900.0] * 6,
                    "pressure": [0.7] * 6,
                    "size": [0.02] * 6,
                }
            )
            _write_archive(path, "scroll", [sparse, dense])
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            by_count = {item.observable_update_count: item for item in bank.descriptors}
            self.assertEqual(set(by_count), {3, 6})
            self.assertAlmostEqual(by_count[3].observable_update_rate_hz, 6.0)
            self.assertAlmostEqual(
                by_count[6].observable_update_rate_hz,
                6 * 1000.0 / 520.0,
            )

            pools = bank.partition_output_splits(
                weights={"train": 98, "development": 1, "test": 1},
                seed=3,
            )
            selected = pools.allocator("train", seed=4).allocate_request(
                orientation_id=0,
                direction="right",
                target_duration_ms=500.0,
                target_update_rate_hz=12.0,
                max_time_warp=1.18,
            )
            # The 520 ms source is farther in duration, but its six retained
            # updates become exactly 12 Hz after replay at the requested 500 ms.
            self.assertEqual(selected.observable_update_count, 6)
            self.assertAlmostEqual(
                selected.observable_update_count * 1000.0 / 500.0,
                12.0,
            )
            replay = observe_replay_primitive(
                bank,
                selected.primitive_id,
                target_samples=51,
                replay_duration_ms=selected.duration_ms,
                output_duration_ms=500.0,
                max_time_warp=1.0,
            )
            self.assertEqual(replay.observation.source_updates, 6)
            self.assertEqual(replay.rows.time_warp_ratio, 1.0)
            self.assertEqual(replay.rows.t_ms[-1], 520.0)

            incompatible_path = Path(temporary) / "incompatible_scroll.npz"
            _write_archive(
                incompatible_path,
                "scroll",
                [_single_pointer_event(3, duration_ms=800)],
            )
            incompatible_bank = ActionReplayBank.from_hmog_npz(
                incompatible_path,
                train_user_ids=[0],
            )
            incompatible_allocator = incompatible_bank.partition_output_splits().allocator(
                "train"
            )
            retained = incompatible_allocator.allocate_request(
                orientation_id=0,
                direction="right",
                target_duration_ms=500.0,
                max_time_warp=1.18,
            )
            # Allocation no longer rejects an otherwise valid human gesture
            # just because its raw duration differs from the detector window.
            self.assertEqual(retained.duration_ms, 800.0)

    def test_demand_partition_and_d4_replay_preserve_exact_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scroll.npz"
            events = [
                _single_pointer_event(
                    index,
                    start_xy=(100.0 + index, 900.0),
                    end_xy=(400.0 + index, 900.0),
                )
                for index in range(10)
            ]
            _write_archive(path, "scroll", events)
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            target_geometry = ReplayGeometry(
                action="scroll",
                orientation_id=0,
                direction="up",
            )
            pools = bank.partition_output_splits_for_demand(
                {
                    "development": [target_geometry] * 5,
                    "train": (),
                    "test": (),
                },
                seed=9,
            )
            reserved = pools.primitive_ids_for_target_geometry(
                "development", target_geometry
            )
            self.assertEqual(len(reserved), 5)
            split_sets = {
                split: set(pools.primitive_ids(split))
                for split in ("train", "development", "test")
            }
            self.assertFalse(split_sets["train"] & split_sets["development"])
            self.assertFalse(split_sets["train"] & split_sets["test"])
            self.assertFalse(split_sets["development"] & split_sets["test"])
            self.assertEqual(set().union(*split_sets.values()), set(bank.primitive_ids()))

            allocator = pools.allocator("development", seed=10)
            allocations = [
                allocator.allocate_isometric_request(
                    orientation_id=0,
                    direction="up",
                    detector_duration_ms=500.0,
                    target_update_count=3.0,
                    target_anchor_px=(500.0, 1000.0),
                )
                for _ in range(5)
            ]
            self.assertEqual(
                len({allocation.primitive_id for allocation in allocations}), 5
            )
            with self.assertRaisesRegex(ActionReplayError, "no unused"):
                allocator.allocate_isometric_request(
                    orientation_id=0,
                    direction="up",
                    detector_duration_ms=500.0,
                )

            allocation = allocations[0]
            raw = bank.raw_rows(allocation.primitive_id)
            replay = observe_replay_primitive(
                bank,
                allocation.primitive_id,
                target_samples=51,
                replay_duration_ms=allocation.descriptor.duration_ms,
                output_duration_ms=500.0,
                spatial_isometry=allocation.isometry,
                max_time_warp=1.0,
            )
            request = classify_replay_request(
                action="scroll",
                orientation_id=0,
                target_duration_ms=500.0,
                x_px=replay.rows.x_px,
                y_px=replay.rows.y_px,
            )
            self.assertEqual(request.direction, "up")
            np.testing.assert_array_equal(replay.rows.t_ms, raw.t_ms)
            np.testing.assert_array_equal(replay.rows.pressure, raw.pressure)
            np.testing.assert_array_equal(
                replay.rows.android_action, raw.android_action
            )
            self.assertAlmostEqual(allocation.isometry.anchor_error_px, 0.0)
            self.assertEqual(allocation.isometry.output_anchor_px, (500.0, 1000.0))
            raw_steps = np.hypot(np.diff(raw.x_px), np.diff(raw.y_px))
            replay_steps = np.hypot(
                np.diff(replay.rows.x_px), np.diff(replay.rows.y_px)
            )
            np.testing.assert_allclose(replay_steps, raw_steps)

    def test_endpoint_similarity_matches_start_end_and_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "swipe.npz"
            _write_archive(
                path,
                "swipe",
                [_single_pointer_event(1, start_xy=(100.0, 900.0), end_xy=(400.0, 900.0))],
            )
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            allocator = bank.partition_output_splits().allocator("train")
            start = (250.0, 800.0)
            end = (260.0, 500.0)
            allocation = allocator.allocate_isometric_request(
                orientation_id=0,
                direction="up",
                detector_duration_ms=500.0,
                target_anchor_px=start,
                target_endpoint_px=end,
            )
            replay = observe_replay_primitive(
                bank,
                allocation.primitive_id,
                target_samples=51,
                replay_duration_ms=allocation.descriptor.duration_ms,
                output_duration_ms=500.0,
                spatial_isometry=allocation.isometry,
                max_time_warp=1.0,
            )
            np.testing.assert_allclose((replay.rows.x_px[0], replay.rows.y_px[0]), start)
            np.testing.assert_allclose((replay.rows.x_px[-1], replay.rows.y_px[-1]), end)
            self.assertAlmostEqual(
                float(np.hypot(replay.rows.x_px[-1] - replay.rows.x_px[0], replay.rows.y_px[-1] - replay.rows.y_px[0])),
                float(np.hypot(end[0] - start[0], end[1] - start[1])),
            )
            self.assertLess(allocation.isometry.endpoint_error_px, 1.0e-6)

    def test_endpoint_control_rejects_fractional_and_out_of_support_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "swipe.npz"
            _write_archive(
                path,
                "swipe",
                [_single_pointer_event(1, start_xy=(100.0, 900.0), end_xy=(400.0, 900.0))],
            )
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            with self.assertRaisesRegex(ActionReplayError, "pixel lattice"):
                bank.partition_output_splits().allocator("train").allocate_isometric_request(
                    orientation_id=0,
                    direction="right",
                    detector_duration_ms=500.0,
                    target_anchor_px=(100.5, 900.0),
                    target_endpoint_px=(400.0, 900.0),
                )
            with self.assertRaisesRegex(ActionReplayError, "no unused"):
                bank.partition_output_splits().allocator("train").allocate_isometric_request(
                    orientation_id=0,
                    direction="right",
                    detector_duration_ms=500.0,
                    target_anchor_px=(100.0, 900.0),
                    target_endpoint_px=(600.0, 900.0),
                )

    def test_d4_pinch_preserves_in_out_and_rejects_unfittable_axis_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pinch_path = Path(temporary) / "pinch.npz"
            _write_archive(
                pinch_path,
                "pinch",
                [_pinch_event(1, scale_direction="out", center_direction="right")],
            )
            pinch_bank = ActionReplayBank.from_hmog_npz(
                pinch_path, train_user_ids=[0]
            )
            pinch_allocator = pinch_bank.partition_output_splits(
                weights={"train": 98, "development": 1, "test": 1}
            ).allocator("train")
            allocation = pinch_allocator.allocate_isometric_request(
                orientation_id=0,
                direction="up",
                pinch_scale_direction="out",
                detector_duration_ms=120.0,
                target_anchor_px=(500.0, 1000.0),
            )
            replay = observe_replay_primitive(
                pinch_bank,
                allocation.primitive_id,
                target_samples=13,
                replay_duration_ms=allocation.descriptor.duration_ms,
                output_duration_ms=120.0,
                spatial_isometry=allocation.isometry,
                max_time_warp=1.0,
            )
            classified = classify_replay_request(
                action="pinch",
                orientation_id=0,
                target_duration_ms=120.0,
                x_px=replay.rows.x_px,
                y_px=replay.rows.y_px,
                pointer_id=replay.rows.pointer_id,
                android_action=replay.rows.android_action,
                frame_index=replay.rows.frame_index,
            )
            self.assertEqual(classified.direction, "up")
            self.assertEqual(classified.pinch_scale_direction, "out")

            tall_path = Path(temporary) / "tall_scroll.npz"
            _write_archive(
                tall_path,
                "scroll",
                [
                    _single_pointer_event(
                        2,
                        start_xy=(500.0, 100.0),
                        end_xy=(500.0, 1700.0),
                    )
                ],
            )
            tall_bank = ActionReplayBank.from_hmog_npz(
                tall_path, train_user_ids=[0]
            )
            tall_allocator = tall_bank.partition_output_splits(
                weights={"train": 98, "development": 1, "test": 1}
            ).allocator("train")
            with self.assertRaisesRegex(ActionReplayError, "in-screen D4"):
                tall_allocator.allocate_isometric_request(
                    orientation_id=0,
                    direction="right",
                    detector_duration_ms=500.0,
                )

    def test_bounded_pinch_allocator_controls_all_four_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pinch.npz"
            _write_archive(
                path,
                "pinch",
                [_pinch_event(1, scale_direction="out", center_direction="right")],
            )
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            allocator = bank.partition_output_splits(
                weights={"train": 98, "development": 1, "test": 1}
            ).allocator("train")
            target = endpoint_geometry_from_pairs(
                start_points_px=((200.0, 600.0), (400.0, 600.0)),
                end_points_px=((300.0, 580.0), (300.0, 1020.0)),
            )
            allocation = allocator.allocate_pinch_endpoint_request(
                orientation_id=0,
                direction="down",
                pinch_scale_direction="out",
                detector_duration_ms=120.0,
                target_geometry=target,
            )
            raw = bank.raw_rows(allocation.primitive_id)
            replay = observe_replay_primitive(
                bank,
                allocation.primitive_id,
                target_samples=13,
                replay_duration_ms=120.0,
                output_duration_ms=120.0,
                pinch_endpoint_fit=allocation.endpoint_fit,
                max_time_warp=1.0,
            )
            np.testing.assert_allclose(
                np.column_stack((replay.rows.x_px[[1, 2]], replay.rows.y_px[[1, 2]])),
                target.start_points_px,
            )
            np.testing.assert_allclose(
                np.column_stack((replay.rows.x_px[[5, 6]], replay.rows.y_px[[5, 6]])),
                target.end_points_px,
            )
            np.testing.assert_array_equal(replay.rows.t_ms, raw.t_ms)
            np.testing.assert_array_equal(
                replay.rows.android_action, raw.android_action
            )
            np.testing.assert_array_equal(replay.rows.pressure, raw.pressure)
            self.assertAlmostEqual(allocation.endpoint_fit.center_scale, 1.0)
            self.assertAlmostEqual(allocation.endpoint_fit.end_span_scale, 1.1)

    def test_raw_replay_duration_is_independent_of_detector_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "swipe.npz"
            event = _single_pointer_event(1, duration_ms=650)
            event["t_ms"] = [0, 325, 650]
            _write_archive(path, "swipe", [event])
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            primitive = bank.descriptors[0]

            replay = observe_replay_primitive(
                bank,
                primitive.primitive_id,
                target_samples=51,
                replay_duration_ms=650.0,
                output_duration_ms=500.0,
                max_time_warp=1.0,
            )
            self.assertEqual(replay.rows.source_duration_ms, 650.0)
            self.assertEqual(replay.rows.replay_duration_ms, 650.0)
            self.assertEqual(replay.detector_duration_ms, 500.0)
            self.assertEqual(replay.rows.t_ms[-1], 650.0)
            self.assertAlmostEqual(float(replay.observation.trajectory[-1, 7]), 0.5)
            self.assertEqual(replay.observation.source_updates, 3)

            default_replay = observe_replay_primitive(
                bank,
                primitive.primitive_id,
                target_samples=51,
                max_time_warp=1.0,
            )
            self.assertEqual(default_replay.rows.replay_duration_ms, 650.0)
            self.assertEqual(default_replay.detector_duration_ms, 500.0)

            with self.assertRaisesRegex(ActionReplayError, "provide both"):
                observe_replay_primitive(
                    bank,
                    primitive.primitive_id,
                    target_samples=51,
                    replay_duration_ms=650.0,
                )

    def test_capped_replay_tokens_keep_explicit_long_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "long_scroll.npz"
            event = _single_pointer_event(
                1,
                duration_ms=12_000,
                start_xy=(100.0, 900.0),
                end_xy=(300.0, 700.0),
            )
            event["t_ms"] = [0, 6_000, 12_000]
            _write_archive(path, "scroll", [event])
            bank = ActionReplayBank.from_hmog_npz(path, train_user_ids=[0])
            replay = observe_replay_primitive(
                bank,
                bank.descriptors[0].primitive_id,
                target_samples=256,
                replay_duration_ms=12_000.0,
                output_duration_ms=12_000.0,
                max_time_warp=1.0,
            )
            self.assertEqual(len(replay.observation.trajectory), 256)
            self.assertAlmostEqual(
                float(replay.observation.trajectory[-1, 7]),
                12.0,
            )
            np.testing.assert_array_equal(
                replay.rows.t_ms,
                np.asarray([0.0, 6_000.0, 12_000.0]),
            )


if __name__ == "__main__":
    unittest.main()
