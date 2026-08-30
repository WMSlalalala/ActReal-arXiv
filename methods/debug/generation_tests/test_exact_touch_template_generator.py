from __future__ import annotations

import time
import unittest

import numpy as np

from pipeline.exact_touch_template_generator import (
    ExactTouchTemplateError,
    generate_exact_touch_template,
)


class ExactTouchTemplateGeneratorTest(unittest.TestCase):
    def _generate(self, **overrides: object):
        values: dict[str, object] = {
            "action": "tap",
            "start_xy_px": (300.0, 400.0),
            "end_xy_px": (305.0, 403.0),
            "direction": "down_right",
            "duration_ms": 40.0,
            "template_t_ms": np.asarray((10.0, 20.0, 50.0)),
            "template_x_px": np.asarray((100.0, 102.0, 105.0)),
            "template_y_px": np.asarray((200.0, 199.0, 203.0)),
            "template_pressure": np.asarray((0.4, 0.6, 0.5)),
            "screen_width_px": 1080.0,
            "screen_height_px": 1920.0,
        }
        values.update(overrides)
        return generate_exact_touch_template(**values)

    def test_tap_same_chord_is_pure_translation(self) -> None:
        result = self._generate()
        self.assertEqual(result.mode, "tap_translation")
        self.assertEqual(result.residual_scale, 1.0)
        np.testing.assert_array_equal(result.x_px, (300.0, 302.0, 305.0))
        np.testing.assert_array_equal(result.y_px, (400.0, 399.0, 403.0))
        np.testing.assert_array_equal(result.t_ms, (0.0, 10.0, 40.0))
        np.testing.assert_array_equal(result.pressure, (0.4, 0.6, 0.5))

    def test_donor_chord_reaches_translation_through_pixel_rounding(
        self,
    ) -> None:
        """A caller asking for the donor chord must not fall off the branch."""

        chord = np.asarray((5.0, 3.0))
        start = np.asarray((300.0, 400.0))
        result = self._generate(
            start_xy_px=tuple(start),
            end_xy_px=tuple(start + chord + 5.0e-10),
        )
        self.assertEqual(result.mode, "tap_translation")
        # The donor's middle sample transports without deformation, and the
        # requested endpoints still own the first and last rows exactly.
        np.testing.assert_array_equal(result.x_px[1], 302.0)
        np.testing.assert_array_equal(
            (result.x_px[0], result.y_px[0]), tuple(start)
        )
        np.testing.assert_array_equal(
            (result.x_px[-1], result.y_px[-1]), tuple(start + chord + 5.0e-10)
        )
        # A request an order of magnitude outside the tolerance is a different
        # chord and still has to take the bridge.
        drifted = self._generate(
            start_xy_px=tuple(start),
            end_xy_px=tuple(start + chord + 1.0e-8),
        )
        self.assertEqual(drifted.mode, "tap_residual_bridge")

    def test_stationary_tap_request_ramps_where_the_donor_held_still(
        self,
    ) -> None:
        """Record why a stationary tap request is the defect being avoided."""

        held = self._generate(
            start_xy_px=(400.0, 500.0),
            end_xy_px=(400.0, 500.0),
            direction="stationary",
            template_t_ms=(0.0, 10.0, 20.0, 30.0),
            template_x_px=(100.0, 100.0, 100.0, 112.0),
            template_y_px=(200.0, 200.0, 200.0, 200.0),
            template_pressure=(0.5, 0.5, 0.5, 0.5),
        )
        self.assertEqual(held.mode, "tap_residual_bridge")
        # The donor held one coordinate for three samples and moved once.  The
        # bridge subtracts a straight-line ramp that nothing adds back, so the
        # output moves at every step instead of holding.
        donor_steps = np.diff((100.0, 100.0, 100.0, 112.0))
        self.assertEqual(int(np.count_nonzero(donor_steps)), 1)
        self.assertTrue(np.all(np.diff(held.x_px) != 0.0))
        np.testing.assert_allclose(held.x_px, (400.0, 396.0, 392.0, 400.0))

    def test_identity_request_preserves_complete_template_geometry(self) -> None:
        source_t = np.asarray((10.0, 20.0, 50.0), dtype=np.float64)
        source_x = np.asarray((100.0, 102.0, 105.0), dtype=np.float64)
        source_y = np.asarray((200.0, 199.0, 203.0), dtype=np.float64)
        result = self._generate(
            start_xy_px=(100.0, 200.0),
            end_xy_px=(105.0, 203.0),
            duration_ms=40.0,
            template_t_ms=source_t,
            template_x_px=source_x,
            template_y_px=source_y,
        )
        self.assertTrue(result.identity_transform)
        self.assertEqual(result.mode, "identity_template")
        np.testing.assert_array_equal(result.x_px, source_x)
        np.testing.assert_array_equal(result.y_px, source_y)
        np.testing.assert_array_equal(result.t_ms, source_t - source_t[0])

    def test_tap_different_chord_uses_zero_endpoint_residual_bridge(self) -> None:
        result = self._generate(
            start_xy_px=(200.0, 300.0),
            end_xy_px=(200.0, 320.0),
            direction="down",
            duration_ms=80.0,
            template_t_ms=(0.0, 25.0, 100.0),
            template_x_px=(100.0, 107.0, 120.0),
            template_y_px=(100.0, 104.0, 100.0),
        )
        self.assertEqual(result.mode, "tap_residual_bridge")
        self.assertEqual(result.residual_scale, 1.0)
        np.testing.assert_allclose(result.x_px, (200.0, 202.0, 200.0))
        np.testing.assert_allclose(result.y_px, (300.0, 309.0, 320.0))
        np.testing.assert_array_equal(result.t_ms, (0.0, 20.0, 80.0))
        np.testing.assert_array_equal(
            (result.x_px[0], result.y_px[0]), (200.0, 300.0)
        )
        np.testing.assert_array_equal(
            (result.x_px[-1], result.y_px[-1]), (200.0, 320.0)
        )

    def test_swipe_maps_longitudinal_and_lateral_chord_frame(self) -> None:
        result = self._generate(
            action="swipe",
            start_xy_px=(500.0, 500.0),
            end_xy_px=(500.0, 700.0),
            direction="down",
            duration_ms=75.5,
            template_t_ms=(0.0, 50.0, 100.0),
            template_x_px=(100.0, 140.0, 200.0),
            template_y_px=(100.0, 110.0, 100.0),
        )
        self.assertEqual(result.mode, "swipe_chord_frame")
        self.assertEqual(result.residual_scale, 1.0)
        np.testing.assert_allclose(result.x_px, (500.0, 480.0, 500.0))
        np.testing.assert_allclose(result.y_px, (500.0, 580.0, 700.0))
        self.assertEqual(result.t_ms[0], 0.0)
        self.assertEqual(result.t_ms[-1], 75.5)

    def test_one_global_residual_scale_prevents_edge_clipping(self) -> None:
        result = self._generate(
            start_xy_px=(0.0, 100.0),
            end_xy_px=(0.0, 200.0),
            direction="down",
            template_t_ms=(0.0, 50.0, 100.0),
            template_x_px=(100.0, 90.0, 110.0),
            template_y_px=(100.0, 150.0, 200.0),
        )
        self.assertEqual(result.mode, "tap_residual_bridge")
        self.assertEqual(result.residual_scale, 0.0)
        np.testing.assert_array_equal(result.x_px, (0.0, 0.0, 0.0))
        self.assertTrue(np.all(result.x_px >= 0.0))

    def test_same_chord_translation_never_silently_clips(self) -> None:
        with self.assertRaisesRegex(
            ExactTouchTemplateError, "pure tap translation"
        ):
            self._generate(
                start_xy_px=(0.0, 400.0),
                end_xy_px=(5.0, 403.0),
                template_x_px=(100.0, 90.0, 105.0),
            )

    def test_direction_is_explicit_and_must_match_endpoints(self) -> None:
        with self.assertRaisesRegex(ExactTouchTemplateError, "explicit"):
            self._generate(direction=None)
        with self.assertRaisesRegex(ExactTouchTemplateError, "conflicts"):
            self._generate(direction="left")
        stationary = self._generate(
            start_xy_px=(350.0, 450.0),
            end_xy_px=(350.0, 450.0),
            direction="stationary",
        )
        self.assertEqual(stationary.direction, "stationary")
        with self.assertRaisesRegex(ExactTouchTemplateError, "non-stationary"):
            self._generate(
                action="swipe",
                start_xy_px=(350.0, 450.0),
                end_xy_px=(350.0, 450.0),
                direction="stationary",
            )

    def test_fractional_duration_and_pauses_are_exact(self) -> None:
        result = self._generate(
            duration_ms=37.5,
            template_t_ms=(5.0, 5.0, 15.0, 25.0),
            template_x_px=(100.0, 100.0, 103.0, 105.0),
            template_y_px=(200.0, 200.0, 202.0, 203.0),
            template_pressure=(0.4, 0.4, 0.6, 0.5),
        )
        np.testing.assert_array_equal(result.t_ms, (0.0, 0.0, 18.75, 37.5))
        self.assertEqual(result.t_ms[-1], result.duration_ms)

    def test_rejects_invalid_template_without_mutating_inputs(self) -> None:
        source_x = np.asarray((100.0, 102.0, 105.0))
        frozen = source_x.copy()
        self._generate(template_x_px=source_x)
        np.testing.assert_array_equal(source_x, frozen)
        with self.assertRaisesRegex(ExactTouchTemplateError, "equal lengths"):
            self._generate(template_pressure=(0.4, 0.5))
        with self.assertRaisesRegex(ExactTouchTemplateError, "pressure"):
            self._generate(template_pressure=(0.4, 1.1, 0.5))
        with self.assertRaisesRegex(ExactTouchTemplateError, "coordinate"):
            self._generate(template_x_px=(100.0, 1200.0, 105.0))
        with self.assertRaisesRegex(ExactTouchTemplateError, "nondecreasing"):
            self._generate(template_t_ms=(0.0, 50.0, 40.0))

    def test_large_template_is_generated_under_100ms(self) -> None:
        count = 20_000
        t_ms = np.linspace(0.0, 1000.0, count)
        x_px = np.linspace(100.0, 900.0, count)
        y_px = 700.0 + 8.0 * np.sin(np.linspace(0.0, 10.0, count))
        pressure = np.full(count, 0.5)
        started = time.perf_counter()
        result = self._generate(
            action="swipe",
            start_xy_px=(80.0, 800.0),
            end_xy_px=(980.0, 800.0),
            direction="right",
            duration_ms=900.0,
            template_t_ms=t_ms,
            template_x_px=x_px,
            template_y_px=y_px,
            template_pressure=pressure,
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.1)
        self.assertEqual(len(result.t_ms), count)
        self.assertEqual(result.x_px[0], 80.0)
        self.assertEqual(result.x_px[-1], 980.0)


if __name__ == "__main__":
    unittest.main()
