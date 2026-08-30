from __future__ import annotations

import unittest

import numpy as np

from pipeline.pinch_endpoint_control import (
    PinchEndpointControlError,
    apply_pinch_endpoint_control,
    endpoint_geometry_from_pairs,
    extract_live_two_pointer_endpoints,
    fit_pinch_endpoint_control,
)


def _pinch_rows(*, middle_y: float = 500.0, trailing_xy=(230.0, 510.0)):
    return {
        "t_ms": np.asarray([0, 0, 0, 50, 50, 100, 100, 120], dtype=np.float64),
        "frame_index": np.asarray([0, 0, 0, 1, 1, 2, 2, 3], dtype=np.int64),
        "pointer_id": np.asarray([0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64),
        "android_action": np.asarray([0, 5, 5, 2, 2, 6, 6, 1], dtype=np.int64),
        "x_px": np.asarray(
            [100.0, 100.0, 300.0, 160.0, 390.0, 230.0, 530.0, trailing_xy[0]],
            dtype=np.float64,
        ),
        "y_px": np.asarray(
            [
                500.0,
                500.0,
                500.0,
                middle_y - 20.0,
                middle_y + 20.0,
                510.0,
                550.0,
                trailing_xy[1],
            ],
            dtype=np.float64,
        ),
    }


def _target_for(
    source,
    *,
    start_center=(300.0, 600.0),
    end_center=(330.0, 780.0),
    end_scale=1.10,
):
    end_span = source.end_span_px * end_scale
    return endpoint_geometry_from_pairs(
        start_points_px=((200.0, 600.0), (400.0, 600.0)),
        end_points_px=(
            (end_center[0], end_center[1] - end_span / 2.0),
            (end_center[0], end_center[1] + end_span / 2.0),
        ),
    )


class PinchEndpointControlTest(unittest.TestCase):
    def test_extract_ignores_trailing_primary_up_coordinate(self) -> None:
        rows = _pinch_rows(trailing_xy=(1000.0, 1800.0))
        geometry = extract_live_two_pointer_endpoints(**rows)
        self.assertEqual(geometry.pointer_ids, (0, 1))
        self.assertEqual(
            geometry.start_points_px,
            ((100.0, 500.0), (300.0, 500.0)),
        )
        self.assertEqual(
            geometry.end_points_px,
            ((230.0, 510.0), (530.0, 550.0)),
        )
        self.assertEqual(geometry.end_frame_index, 2)
        self.assertEqual(geometry.end_t_ms, 100.0)

    def test_exact_four_endpoints_and_inputs_remain_unchanged(self) -> None:
        rows = _pinch_rows()
        snapshots = {name: value.copy() for name, value in rows.items()}
        source = extract_live_two_pointer_endpoints(**rows)
        target = _target_for(source)
        fit = fit_pinch_endpoint_control(
            source,
            target,
            screen_width_px=1080.0,
            screen_height_px=1920.0,
        )
        result = apply_pinch_endpoint_control(
            **rows,
            fit=fit,
            screen_width_px=1080.0,
            screen_height_px=1920.0,
        )
        output = extract_live_two_pointer_endpoints(
            t_ms=rows["t_ms"],
            frame_index=rows["frame_index"],
            pointer_id=rows["pointer_id"],
            android_action=rows["android_action"],
            x_px=result.x_px,
            y_px=result.y_px,
        )
        np.testing.assert_allclose(
            output.start_points_px,
            target.start_points_px,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            output.end_points_px,
            target.end_points_px,
            atol=1.0e-9,
        )
        self.assertLess(result.maximum_endpoint_error_px, 1.0e-9)
        self.assertFalse(result.coordinate_clipping_used)
        self.assertAlmostEqual(fit.center_scale, 1.0)
        self.assertAlmostEqual(fit.start_span_scale, 1.0)
        self.assertAlmostEqual(fit.end_span_scale, 1.10)
        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(rows[name], snapshot)

    def test_rejects_span_or_center_scale_outside_gate(self) -> None:
        rows = _pinch_rows()
        source = extract_live_two_pointer_endpoints(**rows)
        oversized_span = _target_for(source, end_scale=1.50)
        with self.assertRaisesRegex(PinchEndpointControlError, "scale gate"):
            fit_pinch_endpoint_control(
                source,
                oversized_span,
                screen_width_px=1080.0,
                screen_height_px=1920.0,
            )
        oversized_center = _target_for(
            source,
            end_center=(360.0, 960.0),
        )
        with self.assertRaisesRegex(PinchEndpointControlError, "scale gate"):
            fit_pinch_endpoint_control(
                source,
                oversized_center,
                screen_width_px=1080.0,
                screen_height_px=1920.0,
            )

    def test_rejects_in_out_change(self) -> None:
        rows = _pinch_rows()
        source = extract_live_two_pointer_endpoints(**rows)
        target = endpoint_geometry_from_pairs(
            start_points_px=((150.0, 600.0), (450.0, 600.0)),
            end_points_px=((230.0, 700.0), (430.0, 700.0)),
        )
        with self.assertRaisesRegex(PinchEndpointControlError, "in/out"):
            fit_pinch_endpoint_control(
                source,
                target,
                screen_width_px=1080.0,
                screen_height_px=1920.0,
            )

    def test_rejects_middle_path_that_would_leave_screen(self) -> None:
        rows = _pinch_rows(middle_y=100.0)
        source = extract_live_two_pointer_endpoints(**rows)
        end_span = source.end_span_px * 1.10
        target = endpoint_geometry_from_pairs(
            start_points_px=((200.0, 100.0), (400.0, 100.0)),
            end_points_px=(
                (480.0 - end_span / 2.0, 130.0),
                (480.0 + end_span / 2.0, 130.0),
            ),
        )
        fit = fit_pinch_endpoint_control(
            source,
            target,
            screen_width_px=1080.0,
            screen_height_px=1920.0,
        )
        with self.assertRaisesRegex(PinchEndpointControlError, "leave the screen"):
            apply_pinch_endpoint_control(
                **rows,
                fit=fit,
                screen_width_px=1080.0,
                screen_height_px=1920.0,
            )

    def test_stationary_and_moving_centers_cannot_be_crossed(self) -> None:
        rows = _pinch_rows()
        source = extract_live_two_pointer_endpoints(**rows)
        end_span = source.end_span_px * 1.05
        target = endpoint_geometry_from_pairs(
            start_points_px=((200.0, 600.0), (400.0, 600.0)),
            end_points_px=(
                (300.0 - end_span / 2.0, 610.0),
                (300.0 + end_span / 2.0, 610.0),
            ),
        )
        with self.assertRaisesRegex(PinchEndpointControlError, "stationarity"):
            fit_pinch_endpoint_control(
                source,
                target,
                screen_width_px=1080.0,
                screen_height_px=1920.0,
            )


if __name__ == "__main__":
    unittest.main()
