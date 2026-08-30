from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import unittest

import numpy as np

from pipeline.fiveshot_gesture_timing import (
    FiveShotGestureTimingError,
    carrier_window_imu,
    contact_travel_px,
    duration_law_from_pairs,
    law_from_material,
    window_slice_bounds,
)

SCREEN = (1080.0, 1920.0)


@dataclass(frozen=True)
class _Shot:
    trajectory: np.ndarray
    row: Mapping[str, Any]
    event_id: str
    duration_ms: float = 0.0


def _contact(points_px, *, contact=True) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    rows = np.zeros((len(points), 9), dtype=np.float32)
    rows[:, 0] = 1.0 if contact else 0.0
    rows[:, 1] = points[:, 0] / SCREEN[0]
    rows[:, 2] = points[:, 1] / SCREEN[1]
    return rows


def _shot(travel_px: float, duration_ms: float, event_id: str) -> _Shot:
    trajectory = _contact([(100.0, 100.0), (100.0 + travel_px, 100.0)])
    return _Shot(
        trajectory=trajectory,
        row={"raw_duration_ms": duration_ms},
        event_id=event_id,
    )


class DurationLawTest(unittest.TestCase):
    def test_a_travel_the_victim_recorded_returns_that_recording_duration(
        self,
    ) -> None:
        law = duration_law_from_pairs(
            [(50.0, 800.0), (200.0, 500.0), (500.0, 450.0)],
            source_event_ids=("a", "b", "c"),
        )
        for travel, duration in ((50.0, 800.0), (200.0, 500.0), (500.0, 450.0)):
            with self.subTest(travel=travel):
                self.assertAlmostEqual(law.duration_ms(travel), duration, places=6)

    def test_a_travel_between_two_recordings_lands_between_their_durations(
        self,
    ) -> None:
        law = duration_law_from_pairs(
            [(50.0, 800.0), (200.0, 500.0)], source_event_ids=("a", "b")
        )
        reading = law.duration_ms(100.0)
        self.assertLess(reading, 800.0)
        self.assertGreater(reading, 500.0)

    def test_travel_outside_the_material_holds_the_nearest_recording(self) -> None:
        """Five points say nothing about travel they never covered."""

        law = duration_law_from_pairs(
            [(50.0, 800.0), (200.0, 500.0)], source_event_ids=("a", "b")
        )
        self.assertAlmostEqual(law.duration_ms(5.0), 800.0, places=6)
        self.assertAlmostEqual(law.duration_ms(5000.0), 500.0, places=6)

    def test_two_recordings_of_one_travel_are_averaged_not_dropped(self) -> None:
        law = duration_law_from_pairs(
            [(100.0, 400.0), (100.0, 900.0), (400.0, 450.0)],
            source_event_ids=("a", "b", "c"),
        )
        self.assertAlmostEqual(law.duration_ms(100.0), 600.0, places=6)
        self.assertEqual(law.knots, 2)

    def test_material_covering_one_travel_is_refused(self) -> None:
        with self.assertRaises(FiveShotGestureTimingError):
            duration_law_from_pairs(
                [(100.0, 400.0), (100.0, 900.0)], source_event_ids=("a", "b")
            )

    def test_a_single_recording_is_refused(self) -> None:
        with self.assertRaises(FiveShotGestureTimingError):
            duration_law_from_pairs([(100.0, 400.0)], source_event_ids=("a",))

    def test_a_stationary_recording_is_not_a_timing_source(self) -> None:
        law = duration_law_from_pairs(
            [(0.0, 400.0), (100.0, 500.0), (400.0, 450.0)],
            source_event_ids=("a", "b", "c"),
        )
        self.assertEqual(law.knots, 2)

    def test_the_law_reads_the_victims_own_material(self) -> None:
        shots = [
            _shot(40.0, 900.0, "s0"),
            _shot(160.0, 560.0, "s1"),
            _shot(600.0, 460.0, "s2"),
        ]
        law = law_from_material(
            shots, width_px=SCREEN[0], height_px=SCREEN[1]
        )
        self.assertEqual(law.source_event_ids, ("s0", "s1", "s2"))
        self.assertAlmostEqual(law.duration_ms(40.0), 900.0, places=3)
        self.assertAlmostEqual(law.duration_ms(600.0), 460.0, places=3)

    def test_the_law_keeps_the_direction_the_material_shows(self) -> None:
        """The population's shape must be inherited, never assumed."""

        rising = duration_law_from_pairs(
            [(50.0, 200.0), (500.0, 800.0)], source_event_ids=("a", "b")
        )
        self.assertGreater(rising.duration_ms(500.0), rising.duration_ms(50.0))


class LeaveOneOutSpreadTest(unittest.TestCase):
    """The curve alone is a function; the victim is not."""

    def test_the_law_carries_the_victims_own_departures_from_it(self) -> None:
        law = duration_law_from_pairs(
            [(40.0, 900.0), (90.0, 400.0), (160.0, 560.0), (300.0, 700.0),
             (600.0, 460.0)],
            source_event_ids=tuple("abcde"),
        )
        self.assertEqual(len(law.log_residuals), 5)
        self.assertGreater(law.residual_spread, 0.0)

    def test_a_departure_moves_the_reading_off_the_curve(self) -> None:
        law = duration_law_from_pairs(
            [(40.0, 900.0), (90.0, 400.0), (160.0, 560.0), (300.0, 700.0),
             (600.0, 460.0)],
            source_event_ids=tuple("abcde"),
        )
        on_curve = law.duration_ms(160.0)
        moved = {law.duration_ms(160.0, log_offset=law.residual(d)) for d in range(5)}
        self.assertGreater(len(moved), 1)
        for value in moved:
            self.assertGreater(value, 0.0)
        self.assertAlmostEqual(
            law.duration_ms(160.0, log_offset=0.0), on_curve, places=9
        )

    def test_the_same_draw_always_returns_the_same_departure(self) -> None:
        law = duration_law_from_pairs(
            [(40.0, 900.0), (90.0, 400.0), (160.0, 560.0), (300.0, 700.0)],
            source_event_ids=tuple("abcd"),
        )
        for draw in (0, 3, 17, 1 << 40):
            with self.subTest(draw=draw):
                self.assertEqual(law.residual(draw), law.residual(draw))

    def test_material_too_thin_to_hold_one_out_carries_no_spread(self) -> None:
        """Two recordings cannot say how far a third would land."""

        law = duration_law_from_pairs(
            [(50.0, 800.0), (200.0, 500.0)], source_event_ids=("a", "b")
        )
        self.assertEqual(len(law.log_residuals), 0)
        self.assertEqual(law.residual_spread, 0.0)
        self.assertEqual(law.residual(3), 0.0)
        self.assertAlmostEqual(law.duration_ms(100.0, log_offset=0.0),
                               law.duration_ms(100.0), places=9)


class ContactTravelTest(unittest.TestCase):
    def test_travel_is_measured_between_the_first_and_last_contact(self) -> None:
        rows = _contact([(100.0, 100.0), (500.0, 100.0), (400.0, 400.0)])
        self.assertAlmostEqual(
            contact_travel_px(rows, width_px=SCREEN[0], height_px=SCREEN[1]),
            float(np.hypot(300.0, 300.0)),
            places=3,
        )

    def test_rows_without_contact_are_not_travel(self) -> None:
        rows = _contact([(100.0, 100.0), (500.0, 100.0)], contact=False)
        self.assertEqual(
            contact_travel_px(rows, width_px=SCREEN[0], height_px=SCREEN[1]), 0.0
        )

    def test_malformed_rows_are_refused(self) -> None:
        with self.assertRaises(FiveShotGestureTimingError):
            contact_travel_px(
                np.zeros((0, 9)), width_px=SCREEN[0], height_px=SCREEN[1]
            )


class WindowSliceTest(unittest.TestCase):
    def test_a_longer_cut_keeps_the_action_inside_it(self) -> None:
        start, stop = window_slice_bounds(
            active_start=80, active_stop=96, window_samples=179, samples=90
        )
        self.assertEqual(stop - start, 90)
        self.assertLessEqual(start, 80)
        self.assertGreaterEqual(stop, 96)

    def test_a_cut_running_off_the_head_slides_back_inside(self) -> None:
        start, stop = window_slice_bounds(
            active_start=4, active_stop=20, window_samples=179, samples=90
        )
        self.assertEqual((start, stop), (0, 90))

    def test_a_cut_running_off_the_tail_slides_back_inside(self) -> None:
        start, stop = window_slice_bounds(
            active_start=160, active_stop=176, window_samples=179, samples=90
        )
        self.assertEqual((start, stop), (89, 179))

    def test_a_cut_shorter_than_the_action_sits_inside_it(self) -> None:
        start, stop = window_slice_bounds(
            active_start=40, active_stop=140, window_samples=179, samples=20
        )
        self.assertEqual(stop - start, 20)
        self.assertGreaterEqual(start, 40)
        self.assertLessEqual(stop, 140)

    def test_a_cut_longer_than_the_window_is_refused(self) -> None:
        with self.assertRaises(FiveShotGestureTimingError):
            window_slice_bounds(
                active_start=0, active_stop=16, window_samples=179, samples=200
            )


class CarrierWindowImuTest(unittest.TestCase):
    def _window(
        self, samples: int = 179, active: tuple[int, int] = (80, 96)
    ) -> tuple[np.ndarray, np.ndarray]:
        window = np.arange(samples * 6, dtype=np.float32).reshape(samples, 6)
        mask = np.zeros(samples, dtype=np.uint8)
        mask[active[0] : active[1]] = 1
        return window, mask

    def test_the_cut_comes_from_the_carriers_own_window(self) -> None:
        window, mask = self._window()
        imu, audit = carrier_window_imu(window=window, mask=mask, samples=90)
        self.assertEqual(imu.shape, (90, 6))
        start, stop = audit["cut_span"]
        np.testing.assert_array_equal(imu, window[start:stop])
        self.assertFalse(audit["capped_to_window"])
        self.assertEqual(audit["carrier_active_span"], [80, 96])

    def test_a_duration_past_the_window_is_capped_and_says_so(self) -> None:
        window, mask = self._window(samples=60, active=(20, 36))
        imu, audit = carrier_window_imu(window=window, mask=mask, samples=200)
        self.assertEqual(len(imu), 60)
        self.assertTrue(audit["capped_to_window"])
        self.assertEqual(audit["requested_samples"], 200)

    def test_a_window_marking_no_action_is_refused(self) -> None:
        window, mask = self._window()
        with self.assertRaises(FiveShotGestureTimingError):
            carrier_window_imu(
                window=window, mask=np.zeros_like(mask), samples=90
            )

    def test_a_window_without_six_channels_is_refused(self) -> None:
        _, mask = self._window()
        with self.assertRaises(FiveShotGestureTimingError):
            carrier_window_imu(
                window=np.zeros((179, 3), dtype=np.float32), mask=mask, samples=90
            )


if __name__ == "__main__":
    unittest.main()
