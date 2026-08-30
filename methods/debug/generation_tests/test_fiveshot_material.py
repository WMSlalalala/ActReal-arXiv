from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.audit import sha256_array
from pipeline.replay_dataset_builder import INPUT_SHARD_SCHEMA
from pipeline.fiveshot_material import (
    SCHEMA,
    FiveShotMaterialError,
    FiveShotMaterialPool,
    MaterialEvent,
    load_fiveshot_material,
)


def _event(user: str, action: str, ordinal: int) -> MaterialEvent:
    imu = np.full((4, 6), float(ordinal), dtype=np.float32)
    trajectory = np.full((4, 9), float(ordinal), dtype=np.float32)
    return MaterialEvent(
        event_id=f"{user}-{action}-{ordinal}",
        user_id=user,
        action=action,
        split="train",
        shot_ordinal=ordinal,
        source_cluster_id=f"cluster-{ordinal}",
        samples=4,
        duration_ms=40.0,
        imu=imu,
        trajectory=trajectory,
        row={},
    )


def _pool(*, cap: int, users: tuple[str, ...] = ("u0",)) -> FiveShotMaterialPool:
    events = {
        (user, "tap"): tuple(_event(user, "tap", index) for index in range(5))
        for user in users
    }
    return FiveShotMaterialPool(events, release={}, maximum_uses_per_shot=cap)


class BalancedAssignmentTest(unittest.TestCase):
    def test_two_hundred_events_use_each_shot_exactly_forty_times(self) -> None:
        pool = _pool(cap=40)
        pool.plan(
            user_id="u0",
            action="tap",
            event_ids=[f"fake-{index:04d}" for index in range(200)],
            seed=42,
        )
        audit = pool.usage_audit()["by_action"]["tap"]
        self.assertEqual(audit["shots_used"], 5)
        self.assertEqual(audit["total_uses"], 200)
        self.assertEqual(audit["minimum_uses"], 40)
        self.assertEqual(audit["maximum_uses"], 40)

    def test_assignment_ignores_input_order(self) -> None:
        identities = [f"fake-{index:04d}" for index in range(50)]
        forward = _pool(cap=40).plan(
            user_id="u0", action="tap", event_ids=identities, seed=42
        )
        reverse = _pool(cap=40).plan(
            user_id="u0", action="tap", event_ids=list(reversed(identities)), seed=42
        )
        for identity in identities:
            with self.subTest(identity=identity):
                self.assertEqual(
                    forward[identity].shot_ordinal,
                    reverse[identity].shot_ordinal,
                )

    def test_a_different_seed_gives_a_different_assignment(self) -> None:
        identities = [f"fake-{index:04d}" for index in range(50)]
        first = _pool(cap=40).plan(
            user_id="u0", action="tap", event_ids=identities, seed=1
        )
        second = _pool(cap=40).plan(
            user_id="u0", action="tap", event_ids=identities, seed=2
        )
        self.assertTrue(
            any(
                first[identity].shot_ordinal != second[identity].shot_ordinal
                for identity in identities
            )
        )

    def test_exceeding_the_cap_is_refused(self) -> None:
        pool = _pool(cap=2)
        with self.assertRaises(FiveShotMaterialError):
            pool.plan(
                user_id="u0",
                action="tap",
                event_ids=[f"fake-{index}" for index in range(20)],
                seed=1,
            )

    def test_repeated_output_identity_is_refused(self) -> None:
        pool = _pool(cap=40)
        with self.assertRaises(FiveShotMaterialError):
            pool.plan(
                user_id="u0", action="tap", event_ids=["a", "b", "a"], seed=1
            )

    def test_unknown_group_is_refused(self) -> None:
        pool = _pool(cap=40)
        with self.assertRaises(FiveShotMaterialError):
            pool.shots(user_id="u0", action="pinch")


class MaterialLoadingTest(unittest.TestCase):
    def test_a_changed_signal_digest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release.json").write_text(
                json.dumps({"schema_version": SCHEMA}), encoding="utf-8"
            )
            (root / "material_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "split": "train",
                        "user_id": "u0",
                        "action": "tap",
                        "shot_ordinal": 0,
                        "event_id": "e0",
                        "source_cluster_id": "c0",
                        "shard_source": str(root / "missing.npz"),
                        "shard_source_sha256": "0" * 64,
                        "shard_index": 0,
                        "samples": 4,
                        "duration_ms": 40.0,
                        "imu_sha256": sha256_array(np.zeros((4, 6), dtype=np.float32)),
                        "trajectory_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                load_fiveshot_material(root, maximum_uses_per_shot=40)

    def test_a_foreign_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release.json").write_text(
                json.dumps({"schema_version": "something_else"}), encoding="utf-8"
            )
            (root / "material_manifest.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(FiveShotMaterialError):
                load_fiveshot_material(root, maximum_uses_per_shot=40)


def _pinch_pool(spans: tuple[float, ...], *, cap: int = 40) -> FiveShotMaterialPool:
    events = {
        ("u0", "pinch"): tuple(
            MaterialEvent(
                event_id=f"u0-pinch-{index}",
                user_id="u0",
                action="pinch",
                split="train",
                shot_ordinal=index,
                source_cluster_id=f"cluster-{index}",
                samples=4,
                duration_ms=40.0,
                imu=np.zeros((4, 6), dtype=np.float32),
                trajectory=np.zeros((4, 9), dtype=np.float32),
                row={"pinch_start_span_px": float(span)},
            )
            for index, span in enumerate(spans)
        )
    }
    return FiveShotMaterialPool(events, release={}, maximum_uses_per_shot=cap)


class SpanMatchedAssignmentTest(unittest.TestCase):
    SPANS = (200.0, 300.0, 400.0, 600.0, 900.0)

    def _plan(self, requests: dict[str, float], *, cap: int = 40, seed: int = 42):
        pool = _pinch_pool(self.SPANS, cap=cap)
        assignment = pool.plan_matched(
            user_id="u0",
            action="pinch",
            event_scale=requests,
            shot_scale={index: span for index, span in enumerate(self.SPANS)},
            seed=seed,
        )
        return pool, assignment

    def test_two_hundred_requests_use_each_shot_exactly_forty_times(self) -> None:
        requests = {f"fake-{i:04d}": 150.0 + 5.0 * i for i in range(200)}
        pool, _ = self._plan(requests)
        audit = pool.usage_audit()["by_action"]["pinch"]
        self.assertEqual(audit["shots_used"], 5)
        self.assertEqual(audit["total_uses"], 200)
        self.assertEqual(audit["minimum_uses"], 40)
        self.assertEqual(audit["maximum_uses"], 40)

    def test_small_requests_get_small_material(self) -> None:
        requests = {f"fake-{i:04d}": 150.0 + 5.0 * i for i in range(200)}
        _, assignment = self._plan(requests)
        ordered = sorted(requests, key=lambda key: requests[key])
        self.assertEqual(assignment[ordered[0]].shot_ordinal, 0)
        self.assertEqual(assignment[ordered[-1]].shot_ordinal, 4)
        ordinals = [assignment[key].shot_ordinal for key in ordered]
        self.assertEqual(ordinals, sorted(ordinals))

    def test_matching_beats_round_robin_on_rescale(self) -> None:
        requests = {f"fake-{i:04d}": 150.0 + 5.0 * i for i in range(200)}
        _, matched = self._plan(requests)
        round_robin = _pinch_pool(self.SPANS).plan(
            user_id="u0", action="pinch", event_ids=list(requests), seed=42
        )
        worst = {}
        for name, plan in (("matched", matched), ("hash", round_robin)):
            worst[name] = max(
                max(
                    requests[key] / float(plan[key].row["pinch_start_span_px"]),
                    float(plan[key].row["pinch_start_span_px"]) / requests[key],
                )
                for key in requests
            )
        self.assertLess(worst["matched"], worst["hash"])

    def test_assignment_ignores_request_order(self) -> None:
        requests = {f"fake-{i:04d}": 150.0 + 5.0 * i for i in range(50)}
        _, forward = self._plan(requests)
        _, reverse = self._plan(dict(reversed(list(requests.items()))))
        for key in requests:
            with self.subTest(key=key):
                self.assertEqual(
                    forward[key].shot_ordinal, reverse[key].shot_ordinal
                )

    def test_ties_are_broken_deterministically_by_the_frozen_hash(self) -> None:
        requests = {f"fake-{i:04d}": 400.0 for i in range(200)}
        _, first = self._plan(requests)
        _, second = self._plan(requests)
        self.assertEqual(
            {k: v.shot_ordinal for k, v in first.items()},
            {k: v.shot_ordinal for k, v in second.items()},
        )
        counts = Counter(v.shot_ordinal for v in first.values())
        self.assertEqual(sorted(counts.values()), [40] * 5)

    def test_a_shot_without_a_match_scale_is_refused(self) -> None:
        pool = _pinch_pool(self.SPANS)
        with self.assertRaises(FiveShotMaterialError):
            pool.plan_matched(
                user_id="u0",
                action="pinch",
                event_scale={"fake-0": 300.0},
                shot_scale={0: 200.0},
                seed=42,
            )

    def test_a_non_finite_request_scale_is_refused(self) -> None:
        pool = _pinch_pool(self.SPANS)
        with self.assertRaises(FiveShotMaterialError):
            pool.plan_matched(
                user_id="u0",
                action="pinch",
                event_scale={"fake-0": float("nan")},
                shot_scale={i: s for i, s in enumerate(self.SPANS)},
                seed=42,
            )


class SubstitutionBookkeepingTest(unittest.TestCase):
    def _planned(self) -> FiveShotMaterialPool:
        pool = _pinch_pool((200.0, 300.0, 400.0, 600.0, 900.0))
        pool.plan_matched(
            user_id="u0",
            action="pinch",
            event_scale={f"fake-{i:04d}": 150.0 + 5.0 * i for i in range(200)},
            shot_scale={0: 200.0, 1: 300.0, 2: 400.0, 3: 600.0, 4: 900.0},
            seed=42,
        )
        return pool

    def test_a_substitution_keeps_the_total_and_moves_one_use(self) -> None:
        pool = self._planned()
        pool.record_substitution(
            user_id="u0", action="pinch", previous=0, chosen=1
        )
        audit = pool.usage_audit()
        self.assertEqual(audit["by_action"]["pinch"]["total_uses"], 200)
        self.assertEqual(audit["by_action"]["pinch"]["minimum_uses"], 39)
        self.assertEqual(audit["by_action"]["pinch"]["maximum_uses"], 41)
        self.assertEqual(audit["substituted_assignments"], 1)

    def test_a_shot_stops_at_the_substitution_ceiling(self) -> None:
        pool = self._planned()
        self.assertEqual(pool.substitution_ceiling, 44)
        for donor in (0, 2, 3, 4):
            pool.record_substitution(
                user_id="u0", action="pinch", previous=donor, chosen=1
            )
        self.assertEqual(
            pool.uses(user_id="u0", action="pinch", shot_ordinal=1), 44
        )
        with self.assertRaises(FiveShotMaterialError):
            pool.record_substitution(
                user_id="u0", action="pinch", previous=0, chosen=1
            )

    def test_uses_reports_the_running_count(self) -> None:
        pool = self._planned()
        self.assertEqual(
            [
                pool.uses(user_id="u0", action="pinch", shot_ordinal=index)
                for index in range(5)
            ],
            [40] * 5,
        )
        self.assertEqual(
            pool.uses(user_id="u0", action="pinch", shot_ordinal=99), 0
        )

    def test_substituting_a_shot_for_itself_is_refused(self) -> None:
        pool = self._planned()
        with self.assertRaises(FiveShotMaterialError):
            pool.record_substitution(
                user_id="u0", action="pinch", previous=1, chosen=1
            )

    def test_giving_back_a_use_that_was_never_taken_is_refused(self) -> None:
        pool = _pinch_pool((200.0, 300.0, 400.0, 600.0, 900.0))
        with self.assertRaises(FiveShotMaterialError):
            pool.record_substitution(
                user_id="u0", action="pinch", previous=0, chosen=1
            )


class KeystrokeNativeImuLoadingTest(unittest.TestCase):
    """The keystroke rows bind two different IMU signals; both must resolve."""

    def _release(self, root: Path) -> None:
        (root / "release.json").write_text(
            json.dumps({"schema_version": SCHEMA}), encoding="utf-8"
        )

    def _shard(self, root: Path, imu: np.ndarray, trajectory: np.ndarray) -> None:
        np.savez(
            root / "shard.npz",
            schema_version=np.asarray(INPUT_SHARD_SCHEMA),
            coordinate_schema=np.asarray("screen_relative"),
            time_schema=np.asarray("elapsed_seconds"),
            scope=np.asarray("balanced_small"),
            split=np.asarray("train"),
            imu_flat=imu.astype(np.float32),
            trajectory_flat=trajectory.astype(np.float32),
            offsets=np.asarray([0, len(imu)], dtype=np.int64),
            label=np.asarray([0], dtype=np.int64),
            user_id=np.asarray(["u0"]),
            session_id=np.asarray(["u0_s0"]),
            event_id=np.asarray(["e0"]),
            action=np.asarray(["keystroke"]),
            source_cluster_id=np.asarray(["c0"]),
            sample_idx=np.asarray([0], dtype=np.int64),
            cross_modal_pair_id=np.asarray(["p0"]),
        )

    def _material(self, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shard_imu = np.arange(24, dtype=np.float32).reshape(4, 6)
        trajectory = np.arange(36, dtype=np.float32).reshape(4, 9)
        native_imu = np.full((7, 6), 3.5, dtype=np.float32)
        timeline = np.arange(7, dtype=np.int64) * 10_000_000
        self._shard(root, shard_imu, trajectory)
        np.savez(root / "native.npz", imu=native_imu, timestamp_ns=timeline)
        return native_imu, timeline, trajectory

    def _row(
        self, root: Path, native_imu, timeline, trajectory, shard_imu, ordinal=0
    ) -> dict:
        return {
            "split": "train",
            "user_id": "u0",
            "action": "keystroke",
            "shot_ordinal": int(ordinal),
            "event_id": f"e{int(ordinal)}",
            "source_cluster_id": f"c{int(ordinal)}",
            "shard_source": str(root / "shard.npz"),
            "shard_source_sha256": "0" * 64,
            "shard_index": 0,
            "samples": int(len(native_imu)),
            "duration_ms": 70.0,
            "imu_source_kind": "native_continuous_100hz_slice",
            "native_stream": str(root / "native.npz"),
            "native_stream_sha256": "0" * 64,
            "native_start_sample": 0,
            "native_end_sample": int(len(native_imu)),
            "imu_sha256": sha256_array(native_imu),
            "imu_timestamp_ns_sha256": sha256_array(timeline),
            "shard_imu_sha256": sha256_array(shard_imu),
            "trajectory_sha256": sha256_array(trajectory),
        }

    def test_a_keystroke_row_resolves_its_native_imu_and_shard_trajectory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release(root)
            native_imu, timeline, trajectory = self._material(root)
            shard_imu = np.arange(24, dtype=np.float32).reshape(4, 6)
            (root / "material_manifest.jsonl").write_text(
                "".join(
                    json.dumps(
                        self._row(
                            root,
                            native_imu,
                            timeline,
                            trajectory,
                            shard_imu,
                            ordinal=ordinal,
                        )
                    )
                    + "\n"
                    for ordinal in range(5)
                ),
                encoding="utf-8",
            )
            pool = load_fiveshot_material(root, maximum_uses_per_shot=40)
            shot = pool.shots(user_id="u0", action="keystroke")[0]
            self.assertTrue(shot.imu_is_native_slice)
            np.testing.assert_array_equal(shot.imu, native_imu)
            np.testing.assert_array_equal(shot.imu_timestamp_ns, timeline)
            np.testing.assert_array_equal(shot.trajectory, trajectory)
            self.assertEqual(shot.samples, len(native_imu))
            self.assertEqual(len(shot.trajectory), 4)

    def test_a_changed_native_stream_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release(root)
            native_imu, timeline, trajectory = self._material(root)
            shard_imu = np.arange(24, dtype=np.float32).reshape(4, 6)
            np.savez(
                root / "native.npz",
                imu=native_imu + 1.0,
                timestamp_ns=timeline,
            )
            (root / "material_manifest.jsonl").write_text(
                "".join(
                    json.dumps(
                        self._row(
                            root,
                            native_imu,
                            timeline,
                            trajectory,
                            shard_imu,
                            ordinal=ordinal,
                        )
                    )
                    + "\n"
                    for ordinal in range(5)
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FiveShotMaterialError):
                load_fiveshot_material(root, maximum_uses_per_shot=40)

    def test_a_native_slice_shorter_than_its_row_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release(root)
            native_imu, timeline, trajectory = self._material(root)
            shard_imu = np.arange(24, dtype=np.float32).reshape(4, 6)
            rows = [
                self._row(
                    root, native_imu, timeline, trajectory, shard_imu, ordinal
                )
                for ordinal in range(5)
            ]
            rows[0]["native_end_sample"] = 3
            (root / "material_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(FiveShotMaterialError):
                load_fiveshot_material(root, maximum_uses_per_shot=40)

    def test_a_touch_row_still_binds_its_shard_imu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release(root)
            shard_imu = np.arange(24, dtype=np.float32).reshape(4, 6)
            trajectory = np.arange(36, dtype=np.float32).reshape(4, 9)
            self._shard(root, shard_imu, trajectory)
            (root / "material_manifest.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "split": "train",
                            "user_id": "u0",
                            "action": "tap",
                            "shot_ordinal": ordinal,
                            "event_id": f"e{ordinal}",
                            "source_cluster_id": f"c{ordinal}",
                            "shard_source": str(root / "shard.npz"),
                            "shard_source_sha256": "0" * 64,
                            "shard_index": 0,
                            "samples": 4,
                            "duration_ms": 40.0,
                            "imu_sha256": sha256_array(shard_imu),
                            "trajectory_sha256": sha256_array(trajectory),
                        }
                    )
                    + "\n"
                    for ordinal in range(5)
                ),
                encoding="utf-8",
            )
            pool = load_fiveshot_material(root, maximum_uses_per_shot=40)
            shot = pool.shots(user_id="u0", action="tap")[0]
            self.assertFalse(shot.imu_is_native_slice)
            self.assertIsNone(shot.imu_timestamp_ns)
            np.testing.assert_array_equal(shot.imu, shard_imu)


class RecoveredDonorRenderingTest(unittest.TestCase):
    """The attacker's five recordings and the genuine class share a rendering."""

    def _material(self, root: Path) -> np.ndarray:
        (root / "release.json").write_text(
            json.dumps({"schema_version": SCHEMA}), encoding="utf-8"
        )
        imu = np.arange(24, dtype=np.float32).reshape(4, 6)
        trajectory = np.arange(36, dtype=np.float32).reshape(4, 9)
        np.savez(
            root / "shard.npz",
            schema_version=np.asarray(INPUT_SHARD_SCHEMA),
            coordinate_schema=np.asarray("screen_relative"),
            time_schema=np.asarray("elapsed_seconds"),
            scope=np.asarray("balanced_small"),
            split=np.asarray("train"),
            imu_flat=imu,
            trajectory_flat=trajectory,
            offsets=np.asarray([0, len(imu)], dtype=np.int64),
            label=np.asarray([0], dtype=np.int64),
            user_id=np.asarray(["u0"]),
            session_id=np.asarray(["u0_s0"]),
            event_id=np.asarray(["e0"]),
            action=np.asarray(["pinch"]),
            source_cluster_id=np.asarray(["c0"]),
            sample_idx=np.asarray([0], dtype=np.int64),
            cross_modal_pair_id=np.asarray(["p0"]),
        )
        (root / "material_manifest.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "split": "train",
                        "user_id": "u0",
                        "action": "pinch",
                        "shot_ordinal": ordinal,
                        "event_id": f"e{ordinal}",
                        "source_cluster_id": f"c{ordinal}",
                        "shard_source": str(root / "shard.npz"),
                        "shard_source_sha256": "0" * 64,
                        "shard_index": 0,
                        "samples": 4,
                        "duration_ms": 40.0,
                        "imu_sha256": sha256_array(imu),
                        "trajectory_sha256": sha256_array(trajectory),
                    }
                )
                + "\n"
                for ordinal in range(5)
            ),
            encoding="utf-8",
        )
        return trajectory

    def test_without_bindings_the_frozen_bytes_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = self._material(root)
            pool = load_fiveshot_material(root, maximum_uses_per_shot=40)
            for shot in pool.shots(user_id="u0", action="pinch"):
                np.testing.assert_array_equal(shot.trajectory, trajectory)

    def test_a_donor_without_a_binding_is_refused(self) -> None:
        """A donor that cannot be recovered must stop the load, not fall back."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._material(root)
            with self.assertRaises(FiveShotMaterialError) as caught:
                load_fiveshot_material(
                    root, maximum_uses_per_shot=40, genuine_touch_bindings={}
                )
            self.assertIn("binding", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
