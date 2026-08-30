from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.android_touch_observation import observe_android_rows
from pipeline.keystroke_replay import (
    HmogKeystrokeChordBank,
    KeystrokeReplayAllocator,
    KeystrokeReplayError,
    TimingBounds,
    _integer_chord_time_warp,
    build_genuine_event_id_lookup,
    donor_output_split,
)


class KeystrokeReplayTest(unittest.TestCase):
    def test_short_hold_never_collapses_distinct_move_timestamps(self) -> None:
        source = np.asarray([0, 1, 2, 2, 100], dtype=np.int64)
        warped = _integer_chord_time_warp(source, target_hold_ms=53)
        self.assertEqual(int(warped[0]), 0)
        self.assertEqual(int(warped[-1]), 53)
        # Only source rows that were already simultaneous may remain equal.
        for index in range(1, len(source)):
            if source[index] == source[index - 1]:
                self.assertEqual(int(warped[index]), int(warped[index - 1]))
            else:
                self.assertGreater(int(warped[index]), int(warped[index - 1]))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.corpus = root / "keystroke.npz"
        self.split = root / "split.json"
        self.split.write_text(
            json.dumps({"train_users": [1], "val_users": [], "test_users": [2]}),
            encoding="utf-8",
        )
        # Event 111/user 1 supplies twelve ``a`` contacts plus two ``b``
        # contacts.  The latter make exact-key exhaustion and the
        # same-orientation fallback directly testable.  Event 222/user 2 has a
        # conspicuously different x coordinate, making train-only selection
        # directly observable in the test.
        exact_keys = 12
        fallback_keys = 2
        train_keys = exact_keys + fallback_keys
        key_count = train_keys + 1
        train_x = np.concatenate(
            [
                np.asarray([100 + index, 103 + index, 102 + index])
                for index in range(train_keys)
            ]
        )
        np.savez_compressed(
            self.corpus,
            event_id=np.asarray([111, 222], dtype=np.int64),
            user_id=np.asarray([1, 2], dtype=np.int32),
            event_key_offsets=np.asarray([0, train_keys, key_count], dtype=np.int64),
            keycode=np.concatenate(
                (
                    np.full(exact_keys, 97, dtype=np.int32),
                    np.full(fallback_keys, 98, dtype=np.int32),
                    np.asarray([97], dtype=np.int32),
                )
            ),
            key_orientation_id=np.zeros(key_count, dtype=np.int8),
            key_flight_from_previous_ms=np.zeros(key_count, dtype=np.int32),
            key_touch_found=np.ones(key_count, dtype=np.uint8),
            key_touch_offsets=np.arange(0, 3 * key_count + 1, 3, dtype=np.int64),
            flat_t_rel_ms=np.tile([0, 40, 100], key_count).astype(np.int64),
            flat_x=np.concatenate([train_x, [900, 903, 902]]).astype(np.float32),
            flat_y=np.full(3 * key_count, 1600, dtype=np.float32),
            flat_pressure=np.concatenate(
                [
                    np.tile([0.5, 0.7, 0.4], train_keys),
                    [0.2, 0.3, 0.2],
                ]
            ).astype(np.float32),
            flat_size=np.full(3 * key_count, 0.02, dtype=np.float32),
            flat_pointer_count=np.ones(3 * key_count, dtype=np.int8),
            flat_pointer_id=np.zeros(3 * key_count, dtype=np.int16),
            flat_action_code=np.tile([0, 2, 1], key_count).astype(np.int8),
            flat_frame_index=np.tile([0, 1, 2], key_count).astype(np.int32),
        )
        self.bank = HmogKeystrokeChordBank.from_hmog(
            self.corpus,
            split_path=self.split,
            source_registry_path=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_exact_composition(self, keys: int, duration_ms: int) -> None:
        replay = self.bank.compose(
            keycodes=[97] * keys,
            total_duration_ms=duration_ms,
            orientation_id=0,
            seed=42,
        )
        self.assertEqual(int(replay.t_ms[0]), 0)
        self.assertEqual(int(replay.t_ms[-1]), duration_ms)
        self.assertEqual(int(replay.key_up_ms[-1]), duration_ms)
        self.assertEqual(int(replay.hold_ms.sum() + replay.flight_ms.sum()), duration_ms)
        self.assertEqual(len(replay.hold_ms), keys)
        self.assertEqual(len(replay.flight_ms), keys - 1)
        np.testing.assert_array_equal(replay.source_event_ids, 111)
        self.assertTrue(np.all(replay.x_px < 200.0))
        # Every replayed chord keeps the genuine DOWN/MOVE/UP and pressure
        # rows.  Only their timestamps and target key index may change.
        for key_index in range(keys):
            selected = replay.key_index == key_index
            np.testing.assert_array_equal(replay.android_action[selected], [0, 2, 1])
            np.testing.assert_allclose(replay.pressure[selected], [0.5, 0.7, 0.4])
            self.assertEqual(int(replay.t_ms[selected][0]), int(replay.key_down_ms[key_index]))
            self.assertEqual(int(replay.t_ms[selected][-1]), int(replay.key_up_ms[key_index]))

        observed = observe_android_rows(
            target_samples=duration_ms // 10 + 1,
            **replay.observation_kwargs(),
        )
        self.assertAlmostEqual(float(observed.trajectory[-1, 7]), duration_ms / 1000.0)
        # Strict interiors of every UP-to-next-DOWN flight contain no contact.
        for key_index, flight in enumerate(replay.flight_ms):
            if flight <= 10:
                continue
            grid = np.arange(len(observed.touch), dtype=np.int64) * 10
            interior = (grid > replay.key_up_ms[key_index]) & (
                grid < replay.key_down_ms[key_index + 1]
            )
            self.assertTrue(np.any(interior))
            self.assertTrue(np.all(observed.touch[interior, 0] == 0.0))

    def test_ten_keys_end_exactly_at_ten_seconds(self) -> None:
        self._assert_exact_composition(10, 10_000)

    def test_twelve_keys_end_exactly_at_twelve_seconds(self) -> None:
        self._assert_exact_composition(12, 12_000)

    def test_missing_keycode_falls_back_to_same_orientation_real_chord(self) -> None:
        replay = self.bank.compose(
            keycodes=[999],
            total_duration_ms=100,
            orientation_id=0,
            seed=5,
            bounds=TimingBounds(100, 100, 0, 0),
        )
        self.assertEqual(replay.keycodes.tolist(), [999])
        self.assertIn(int(replay.source_keycodes[0]), (97, 98))
        np.testing.assert_array_equal(replay.keycode, 999)

    def test_target_anchor_uses_closest_oob_safe_common_translation(self) -> None:
        replay = self.bank.compose(
            keycodes=[97],
            total_duration_ms=100,
            orientation_id=0,
            seed=4,
            bounds=TimingBounds(100, 100, 0, 0),
            target_xy_px=[(1080.0, 0.0)],
        )
        # The source DOWN is three pixels left of the chord's rightmost MOVE.
        # Exact DOWN alignment at x=1080 would therefore be out of bounds, so
        # the closest legal common translation leaves the DOWN at x=1077 and
        # the MOVE at the physical edge.  No row was individually clipped.
        selected_source = int(replay.source_key_indices[0])
        source_down_x = 100.0 + selected_source
        self.assertAlmostEqual(float(replay.translations_px[0, 0]), 977.0 - selected_source)
        self.assertAlmostEqual(float(replay.translations_px[0, 1]), -1600.0)
        self.assertAlmostEqual(float(replay.x_px[0]), source_down_x + float(replay.translations_px[0, 0]))
        self.assertAlmostEqual(float(np.max(replay.x_px)), 1080.0)
        self.assertAlmostEqual(float(replay.y_px[0]), 0.0)
        np.testing.assert_allclose(np.diff(replay.x_px), [3.0, -1.0])
        self.assertTrue(np.all(replay.x_px >= 0.0))
        self.assertTrue(np.all(replay.x_px <= 1080.0))

    def test_raw_typing_duration_is_separate_from_detector_window(self) -> None:
        replay = self.bank.compose(
            keycodes=[97] * 10,
            total_duration_ms=10_000,
            orientation_id=0,
            seed=42,
        )
        observed = observe_android_rows(
            target_samples=501,
            **replay.observation_kwargs(detector_duration_ms=5_000),
        )
        self.assertEqual(int(replay.t_ms[-1]), 10_000)
        self.assertAlmostEqual(float(observed.trajectory[-1, 7]), 5.0)

    def test_infeasible_duration_is_rejected(self) -> None:
        with self.assertRaises(KeystrokeReplayError):
            self.bank.compose(
                keycodes=[97] * 10,
                total_duration_ms=30_000,
                orientation_id=0,
                seed=1,
                bounds=TimingBounds(
                    hold_min_ms=50,
                    hold_max_ms=200,
                    flight_min_ms=0,
                    flight_max_ms=1500,
                ),
            )

    def test_genuine_event_id_lookup_can_be_user_scoped(self) -> None:
        self.assertEqual(build_genuine_event_id_lookup(self.corpus), {111: 0, 222: 1})
        self.assertEqual(
            build_genuine_event_id_lookup(self.corpus, allowed_user_ids=[1]),
            {111: 0},
        )

    def test_donor_event_family_is_confined_to_one_output_split(self) -> None:
        assigned = donor_output_split(111, seed=7)
        replay = self.bank.compose(
            keycodes=[97, 97],
            total_duration_ms=1000,
            orientation_id=0,
            seed=42,
            output_split=assigned,
            split_seed=7,
        )
        np.testing.assert_array_equal(replay.source_event_ids, 111)
        wrong = next(split for split in ("train", "development", "test") if split != assigned)
        with self.assertRaises(KeystrokeReplayError):
            self.bank.compose(
                keycodes=[97],
                total_duration_ms=100,
                orientation_id=0,
                seed=42,
                output_split=wrong,
                split_seed=7,
            )
        wrong_allocator = KeystrokeReplayAllocator(
            self.bank,
            output_split=wrong,
            split_seed=7,
            maximum_uses_per_contact=2,
        )
        with self.assertRaises(KeystrokeReplayError):
            wrong_allocator.compose(
                keycodes=[999],
                total_duration_ms=100,
                orientation_id=0,
                seed=9,
                bounds=TimingBounds(100, 100, 0, 0),
            )

    def test_selected_genuine_event_can_be_excluded_from_attack_bank(self) -> None:
        with self.assertRaises(KeystrokeReplayError):
            HmogKeystrokeChordBank.from_hmog(
                self.corpus,
                split_path=self.split,
                excluded_source_event_ids=[111],
            )

    def test_bank_can_be_limited_to_quality_accepted_event_ids(self) -> None:
        accepted = HmogKeystrokeChordBank.from_hmog(
            self.corpus,
            split_path=self.split,
            allowed_source_event_ids=[111],
        )
        self.assertEqual(accepted.source_key_count, 14)
        with self.assertRaises(KeystrokeReplayError):
            HmogKeystrokeChordBank.from_hmog(
                self.corpus,
                split_path=self.split,
                allowed_source_event_ids=[222],
            )

    def test_allocator_never_reuses_and_falls_back_after_exact_exhaustion(self) -> None:
        assigned = donor_output_split(111, seed=7)
        allocator = KeystrokeReplayAllocator(
            self.bank,
            output_split=assigned,
            split_seed=7,
        )
        first = allocator.compose(
            keycodes=[97] * 6,
            total_duration_ms=600,
            orientation_id=0,
            seed=1,
            bounds=TimingBounds(100, 100, 0, 0),
        )
        second = allocator.compose(
            keycodes=[97] * 6,
            total_duration_ms=600,
            orientation_id=0,
            seed=2,
            bounds=TimingBounds(100, 100, 0, 0),
        )
        self.assertFalse(
            set(first.source_key_indices) & set(second.source_key_indices)
        )
        np.testing.assert_array_equal(first.source_use_ordinals, 1)
        np.testing.assert_array_equal(second.source_use_ordinals, 1)
        self.assertEqual(len(allocator.used_source_key_indices), 12)
        fallback = allocator.compose(
            keycodes=[97, 97],
            total_duration_ms=200,
            orientation_id=0,
            seed=3,
            bounds=TimingBounds(100, 100, 0, 0),
        )
        np.testing.assert_array_equal(fallback.source_keycodes, 98)
        self.assertFalse(
            (set(first.source_key_indices) | set(second.source_key_indices))
            & set(fallback.source_key_indices)
        )
        self.assertEqual(len(allocator.used_source_key_indices), 14)
        with self.assertRaises(KeystrokeReplayError):
            allocator.compose(
                keycodes=[97],
                total_duration_ms=100,
                orientation_id=0,
                seed=4,
                bounds=TimingBounds(100, 100, 0, 0),
            )

    def test_max_two_uses_is_balanced_then_fails_closed_on_third_use(self) -> None:
        assigned = donor_output_split(111, seed=7)
        allocator = KeystrokeReplayAllocator(
            self.bank,
            output_split=assigned,
            split_seed=7,
            maximum_uses_per_contact=2,
        )
        first_cycle = []
        second_cycle = []
        for index in range(14):
            first_cycle.append(
                allocator.compose(
                    keycodes=[97],
                    total_duration_ms=100,
                    orientation_id=0,
                    seed=index,
                    bounds=TimingBounds(100, 100, 0, 0),
                    target_xy_px=[(300.0, 1600.0)],
                )
            )
        self.assertEqual(
            len({int(rows.source_key_indices[0]) for rows in first_cycle}),
            14,
        )
        self.assertTrue(
            all(int(rows.source_use_ordinals[0]) == 1 for rows in first_cycle)
        )
        for index in range(14):
            second_cycle.append(
                allocator.compose(
                    keycodes=[97],
                    total_duration_ms=100,
                    orientation_id=0,
                    seed=100 + index,
                    bounds=TimingBounds(100, 100, 0, 0),
                    target_xy_px=[(600.0, 1600.0)],
                )
            )
        np.testing.assert_array_equal(
            [int(rows.source_key_indices[0]) for rows in second_cycle],
            [int(rows.source_key_indices[0]) for rows in first_cycle],
        )
        self.assertTrue(
            all(int(rows.source_use_ordinals[0]) == 2 for rows in second_cycle)
        )
        # Reuse preserves the real chord but independently applies the new
        # target anchor rather than copying an earlier transformed output.
        for first, second in zip(first_cycle, second_cycle):
            self.assertAlmostEqual(
                float(second.translations_px[0, 0])
                - float(first.translations_px[0, 0]),
                300.0,
            )
        self.assertEqual(set(allocator.source_key_use_counts.values()), {2})
        with self.assertRaises(KeystrokeReplayError):
            allocator.compose(
                keycodes=[97],
                total_duration_ms=100,
                orientation_id=0,
                seed=999,
                bounds=TimingBounds(100, 100, 0, 0),
            )

    def test_failed_composition_does_not_commit_source_usage(self) -> None:
        assigned = donor_output_split(111, seed=7)
        allocator = KeystrokeReplayAllocator(
            self.bank,
            output_split=assigned,
            split_seed=7,
            maximum_uses_per_contact=2,
        )
        with self.assertRaises(KeystrokeReplayError):
            allocator.compose(
                keycodes=[999],
                total_duration_ms=101,
                orientation_id=0,
                seed=1,
                bounds=TimingBounds(100, 100, 0, 0),
            )
        self.assertEqual(allocator.source_key_use_counts, {})
        replay = allocator.compose(
            keycodes=[999],
            total_duration_ms=100,
            orientation_id=0,
            seed=2,
            bounds=TimingBounds(100, 100, 0, 0),
        )
        self.assertEqual(int(replay.source_use_ordinals[0]), 1)

    def test_use_cap_must_be_a_positive_integer(self) -> None:
        assigned = donor_output_split(111, seed=7)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(KeystrokeReplayError):
                KeystrokeReplayAllocator(
                    self.bank,
                    output_split=assigned,
                    split_seed=7,
                    maximum_uses_per_contact=invalid,
                )


if __name__ == "__main__":
    unittest.main()
