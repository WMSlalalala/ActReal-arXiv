from __future__ import annotations

import unittest

import numpy as np

from pipeline.android_touch_observation import (
    TouchObservationError,
    active_xy_repeat_rate,
    detector_grid_clock,
    observe_android_rows,
)


class DetectorGridClockTest(unittest.TestCase):
    def test_observed_clock_is_the_shared_construction(self) -> None:
        """The observer's own clock column must come from the helper."""

        result = observe_android_rows(
            action="scroll",
            target_samples=7,
            orientation_id=0,
            t_ms=[0.0, 25.0, 60.0],
            x_px=[100.0, 140.0, 200.0],
            y_px=[300.0, 320.0, 360.0],
            pressure=[0.5, 0.5, 0.5],
            pointer_id=[0, 0, 0],
            android_action=[0, 2, 1],
        )
        span_ms = float(result.trajectory[-1, 7] * 1000.0)
        np.testing.assert_array_equal(
            result.trajectory[:, 7], detector_grid_clock(7, span_ms)
        )

    def test_rescaling_a_donor_clock_leaves_a_rounding_signature(self) -> None:
        """Record the arithmetic every producer of this column must avoid."""

        samples, duration_ms = 8, 100.0
        grid = detector_grid_clock(samples, duration_ms)
        # What the five-shot touch path used to write: a donor's own clock,
        # resampled onto the carrier's sample count, renormalised, and stretched
        # to the requested duration.
        donor = detector_grid_clock(samples, 113.0).astype(np.float64)
        progress = (donor - donor[0]) / (donor[-1] - donor[0])
        progress[0], progress[-1] = 0.0, 1.0
        rescaled = ((progress * duration_ms) / 1000.0).astype(np.float32)
        self.assertFalse(np.array_equal(grid, rescaled))
        np.testing.assert_allclose(grid, rescaled, rtol=0.0, atol=1.0e-7)
        # The departure is not symmetric noise: the per-step spread of the two
        # constructions differs, and that is what a detector separates on.
        self.assertNotEqual(
            float(np.std(np.diff(grid.astype(np.float64)))),
            float(np.std(np.diff(rescaled.astype(np.float64)))),
        )

    def test_clock_rejects_degenerate_requests(self) -> None:
        with self.assertRaises(TouchObservationError):
            detector_grid_clock(1, 80.0)
        with self.assertRaises(TouchObservationError):
            detector_grid_clock(9, 0.0)


class AndroidTouchObservationTest(unittest.TestCase):
    def test_single_pointer_uses_last_observation_hold(self) -> None:
        result = observe_android_rows(
            action="scroll",
            target_samples=7,
            orientation_id=0,
            t_ms=[0.0, 25.0, 60.0],
            x_px=[100.0, 300.0, 500.0],
            y_px=[960.0, 960.0, 960.0],
            pressure=[0.5, 0.6, 0.7],
            pointer_id=[0, 0, 0],
            android_action=[0, 2, 1],
            frame_end=[1, 1, 1],
            source_duration_ms=60.0,
        )
        expected_x = np.asarray(
            [100.0, 100.0, 100.0, 300.0, 300.0, 300.0, 500.0]
        ) / 1080.0
        np.testing.assert_allclose(result.touch[:, 1], expected_x)
        self.assertTrue(np.all(result.touch[:, 0] == 1.0))
        self.assertEqual(result.source_updates, 3)
        # Linear interpolation would make every increment non-zero.  Here four
        # of six active increments are true Android holds.
        self.assertAlmostEqual(active_xy_repeat_rate(result.trajectory), 4.0 / 6.0)

    def test_pinch_never_emits_trailing_single_pointer_centroid(self) -> None:
        result = observe_android_rows(
            action="pinch",
            target_samples=5,
            orientation_id=0,
            t_ms=[0, 0, 20, 20, 30, 30, 40],
            x_px=[100, 300, 90, 310, 80, 320, 900],
            y_px=[900, 900, 900, 900, 900, 900, 900],
            pressure=[1, 1, 1, 1, 1, 1, 1],
            pointer_id=[0, 1, 0, 1, 0, 1, 0],
            android_action=[0, 261, 2, 2, 2, 262, 1],
            frame_end=[0, 1, 0, 1, 0, 1, 1],
            source_duration_ms=40.0,
        )
        self.assertTrue(np.all(result.touch[:, 0] == 1.0))
        self.assertTrue(np.all(result.touch[:, 4] == 2.0))
        # All three two-pointer frames have center x=200.  The final x=900
        # primary ACTION_UP is deliberately outside the pinch observation.
        np.testing.assert_allclose(result.touch[:, 1], 200.0 / 1080.0)

    def test_keystroke_preserves_flights_and_never_connects_keys(self) -> None:
        result = observe_android_rows(
            action="keystroke",
            target_samples=7,
            orientation_id=0,
            t_ms=[0, 20, 40, 60],
            x_px=[100, 102, 500, 501],
            y_px=[1600, 1600, 1600, 1600],
            pressure=[1, 1, 1, 1],
            pointer_id=[0, 0, 0, 0],
            android_action=[0, 1, 0, 1],
            key_index=[0, 0, 1, 1],
            frame_end=[1, 1, 1, 1],
            source_duration_ms=60.0,
        )
        np.testing.assert_array_equal(
            result.touch[:, 0],
            np.asarray([1, 1, 0, 0, 1, 1, 0], dtype=np.float32),
        )
        # No dx/dy jump is emitted from the first key to the second key.
        np.testing.assert_array_equal(result.touch[:, 5:7], 0.0)
        self.assertTrue(result.touch_observed)
        self.assertTrue(np.all(result.trajectory[:, 8] == 1.0))

    def test_out_of_screen_coordinates_are_rejected_not_clipped(self) -> None:
        for out_of_bounds_x in (1081.0, 1080.0 + 1.0e-7):
            with self.subTest(out_of_bounds_x=out_of_bounds_x):
                with self.assertRaises(TouchObservationError):
                    observe_android_rows(
                        action="tap",
                        target_samples=2,
                        orientation_id=0,
                        t_ms=[0, 10],
                        x_px=[100, out_of_bounds_x],
                        y_px=[100, 100],
                        pressure=[1, 1],
                        pointer_id=[0, 0],
                        android_action=[0, 1],
                        frame_end=[1, 1],
                        source_duration_ms=10,
                    )

    def test_capped_token_grid_keeps_full_elapsed_duration(self) -> None:
        result = observe_android_rows(
            action="scroll",
            target_samples=256,
            target_duration_ms=12_000.0,
            orientation_id=0,
            t_ms=[0.0, 6_000.0, 12_000.0],
            x_px=[100.0, 200.0, 300.0],
            y_px=[900.0, 800.0, 700.0],
            pressure=[1.0, 1.0, 1.0],
            pointer_id=[0, 0, 0],
            android_action=[0, 2, 1],
            frame_end=[1, 1, 1],
            source_duration_ms=12_000.0,
        )
        self.assertEqual(len(result.trajectory), 256)
        self.assertAlmostEqual(float(result.trajectory[-1, 7]), 12.0)
        self.assertGreater(active_xy_repeat_rate(result.trajectory), 0.98)

    def test_fractional_target_duration_keeps_exact_final_android_row(self) -> None:
        # Regression for an endpoint that used to land 2.27e-13 ms after the
        # final detector tick when scaling multiplied by target/source first.
        # That numerical ordering made ZOH retain the penultimate coordinate.
        target_duration_ms = 1679.9999475479126
        result = observe_android_rows(
            action="scroll",
            target_samples=169,
            target_duration_ms=target_duration_ms,
            orientation_id=0,
            t_ms=[0.0, 3279.0, 3280.0],
            x_px=[166.0, 589.69217062, 589.6795654296875],
            y_px=[1161.0, 1101.08894348, 1101.0],
            pressure=[1.0, 1.0, 1.0],
            pointer_id=[0, 0, 0],
            android_action=[0, 2, 1],
            frame_end=[1, 1, 1],
            source_duration_ms=3280.0,
        )
        self.assertAlmostEqual(
            float(result.trajectory[-1, 1]) * 1080.0,
            589.6795654296875,
            delta=5.0e-4,
        )
        self.assertAlmostEqual(
            float(result.trajectory[-1, 2]) * 1920.0,
            1101.0,
            delta=5.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
