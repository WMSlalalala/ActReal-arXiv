from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import unittest

import numpy as np

from pipeline.android_touch_observation import (
    screen_dimensions_for_orientation,
)
from pipeline.pinch_endpoint_control import endpoint_geometry_from_pairs
from pipeline.replay_dataset_builder import (
    PINCH_AREA_SIMILARITY_GENERATION_MODE,
    ReplayDatasetBuildError,
    _observe_pinch_area_similarity,
    _pinch_material_widest_span,
)

SCREEN = np.asarray(screen_dimensions_for_orientation(0), dtype=np.float64)


@dataclass(frozen=True)
class _Material:
    """The fields the placement reads off a frozen material event."""

    trajectory: np.ndarray
    row: dict[str, Any]
    event_id: str = "material-0"
    shot_ordinal: int = 0
    source_cluster_id: str = "cluster-0"
    samples: int = 0


@dataclass(frozen=True)
class _Target:
    action: str
    orientation_id: int


def _pinch_material(
    *,
    orientation_id: int = 0,
    samples: int = 24,
    start_span: float = 300.0,
    end_span: float = 600.0,
    axis_rad: float = 0.0,
    centre: tuple[float, float] = (500.0, 900.0),
    drift_px: float = 40.0,
) -> _Material:
    """Build a coherent two-finger recording and its observed centroid."""

    width, height = screen_dimensions_for_orientation(orientation_id)
    axis = np.asarray((math.cos(axis_rad), math.sin(axis_rad)), dtype=np.float64)
    fraction = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    centroid = np.asarray(centre, dtype=np.float64)[None, :] + np.stack(
        (drift_px * fraction, 0.5 * drift_px * fraction), axis=1
    )
    half = 0.5 * (start_span + (end_span - start_span) * fraction)
    first = centroid - half[:, None] * axis[None, :]
    second = centroid + half[:, None] * axis[None, :]

    trajectory = np.zeros((samples, 9), dtype=np.float32)
    trajectory[:, 0] = 1.0
    trajectory[:, 1] = (centroid[:, 0] / width).astype(np.float32)
    trajectory[:, 2] = (centroid[:, 1] / height).astype(np.float32)
    trajectory[:, 3] = np.linspace(0.4, 0.6, samples, dtype=np.float32)
    trajectory[:, 4] = 2.0
    trajectory[:, 7] = np.linspace(0.0, 0.5, samples, dtype=np.float32)
    trajectory[:, 8] = 1.0
    deltas = np.zeros((samples, 2), dtype=np.float32)
    deltas[1:] = np.diff(trajectory[:, 1:3], axis=0)
    trajectory[:, 5:7] = deltas
    return _Material(
        trajectory=trajectory,
        samples=samples,
        row={
            "orientation_id": orientation_id,
            "pinch_start_points_px": [list(first[0]), list(second[0])],
            "pinch_end_points_px": [list(first[-1]), list(second[-1])],
            "pinch_start_span_px": float(start_span),
            "pinch_end_span_px": float(end_span),
            "pinch_scale_direction": "out" if end_span > start_span else "in",
        },
    )


def _request(
    *,
    centre: tuple[float, float] = (500.0, 900.0),
    span: float = 450.0,
    axis_rad: float = 0.4,
    percent: float = 1.5,
) -> Any:
    """Build a carrier request whose widest moment is its end pair."""

    axis = np.asarray((math.cos(axis_rad), math.sin(axis_rad)), dtype=np.float64)
    middle = np.asarray(centre, dtype=np.float64)
    start = 0.5 * span * axis
    end = 0.5 * span * percent * axis
    return endpoint_geometry_from_pairs(
        start_points_px=(tuple(middle - start), tuple(middle + start)),
        end_points_px=(tuple(middle - end), tuple(middle + end)),
    )


class PinchAreaSimilarityTest(unittest.TestCase):
    def _place(
        self,
        *,
        material: _Material | None = None,
        request: Any | None = None,
        orientation_id: int = 0,
        target_samples: int = 24,
        target_duration_ms: float = 300.0,
        pointer_order: str = "as_requested",
    ):
        material = material or _pinch_material()
        return _observe_pinch_area_similarity(
            material,
            _Target(action="pinch", orientation_id=orientation_id),
            target_samples=target_samples,
            target_duration_ms=target_duration_ms,
            requested=request if request is not None else _request(),
            pointer_order=pointer_order,
        )

    def test_the_widest_moment_lands_on_the_requested_extent(self) -> None:
        request = _request(centre=(700.0, 1200.0), span=400.0, percent=1.5)
        _, generation = self._place(request=request)
        self.assertLess(generation["delivered_area_extent_error_px"], 5.0e-4)
        np.testing.assert_allclose(
            np.asarray(generation["delivered_area_points_px"]),
            np.asarray(request.end_points_px),
            rtol=0.0,
            atol=5.0e-4,
        )
        self.assertEqual(generation["requested_area_moment"], "end")
        self.assertEqual(generation["source_widest_moment"], "end")
        self.assertEqual(
            generation["generation_mode"], PINCH_AREA_SIMILARITY_GENERATION_MODE
        )

    def test_a_closing_recording_anchors_on_its_own_widest_moment(self) -> None:
        material = _pinch_material(start_span=800.0, end_span=200.0)
        _, generation = self._place(material=material)
        self.assertEqual(generation["source_widest_moment"], "start")
        self.assertEqual(generation["source_scale_direction"], "in")
        self.assertAlmostEqual(
            generation["source_widest_span_px"], 800.0, places=9
        )

    def test_scale_is_the_requested_extent_over_the_recorded_extent(self) -> None:
        material = _pinch_material(start_span=300.0, end_span=600.0)
        request = _request(span=400.0, percent=1.5)
        _, generation = self._place(material=material, request=request)
        self.assertAlmostEqual(
            generation["similarity_scale"], 600.0 / 600.0, places=12
        )
        self.assertAlmostEqual(
            _pinch_material_widest_span(material), 600.0, places=9
        )

    def test_rotation_aligns_the_two_widest_axes(self) -> None:
        material = _pinch_material(axis_rad=0.0)
        _, generation = self._place(material=material, request=_request(axis_rad=0.4))
        self.assertAlmostEqual(
            generation["similarity_rotation_rad"], 0.4, places=12
        )

    def test_the_shape_is_preserved_up_to_the_float32_grid(self) -> None:
        material = _pinch_material()
        observation, generation = self._place(material=material)
        scale = float(generation["similarity_scale"])
        source = material.trajectory[:, 1:3].astype(np.float64) * SCREEN
        placed = observation.trajectory[:, 1:3].astype(np.float64) * SCREEN
        source_steps = np.linalg.norm(np.diff(source, axis=0), axis=1)
        placed_steps = np.linalg.norm(np.diff(placed, axis=0), axis=1)
        np.testing.assert_allclose(
            placed_steps, scale * source_steps, rtol=0.0, atol=1.0e-3
        )

    def test_pressure_pointer_count_and_contact_are_untouched(self) -> None:
        material = _pinch_material()
        observation, _ = self._place(material=material)
        for column in (0, 3, 4, 8):
            with self.subTest(column=column):
                np.testing.assert_array_equal(
                    observation.trajectory[:, column],
                    material.trajectory[:, column],
                )

    def test_the_carrier_duration_is_delivered_exactly(self) -> None:
        observation, generation = self._place(target_duration_ms=137.5)
        delivered = float(
            (observation.trajectory[-1, 7] - observation.trajectory[0, 7]) * 1000.0
        )
        self.assertAlmostEqual(delivered, 137.5, places=3)
        self.assertAlmostEqual(
            generation["generated_raw_duration_ms"], 137.5, places=3
        )

    def test_magnitude_and_direction_come_from_the_recording(self) -> None:
        material = _pinch_material(start_span=400.0, end_span=200.0)
        _, generation = self._place(material=material)
        self.assertEqual(generation["source_scale_direction"], "in")
        self.assertAlmostEqual(generation["source_percent"], 0.5, places=12)
        # The carrier asked for an opening pinch; the attacker's own recording
        # decides, so the carrier's direction is recorded but never imposed.
        self.assertEqual(generation["requested_carrier_scale_direction"], "out")

    def test_swapping_the_pointer_order_turns_the_placement_around(self) -> None:
        request = _request()
        first, _ = self._place(request=request, pointer_order="as_requested")
        second, generation = self._place(request=request, pointer_order="swapped")
        centre = np.asarray(generation["requested_area_center_px"], dtype=np.float64)
        mirrored = 2.0 * centre - (
            first.trajectory[:, 1:3].astype(np.float64) * SCREEN
        )
        np.testing.assert_allclose(
            second.trajectory[:, 1:3].astype(np.float64) * SCREEN,
            mirrored,
            rtol=0.0,
            atol=1.0e-3,
        )
        self.assertEqual(generation["pinch_pointer_order"], "swapped")

    def test_a_centroid_that_leaves_the_screen_is_refused(self) -> None:
        material = _pinch_material(drift_px=400.0)
        with self.assertRaises(ReplayDatasetBuildError) as caught:
            self._place(
                material=material,
                request=_request(centre=(120.0, 120.0), span=900.0, percent=1.05),
            )
        self.assertIn("leaves the screen", str(caught.exception))

    def test_a_finger_that_leaves_the_screen_is_refused(self) -> None:
        # The widest pair is the request itself, so only the narrow pair can be
        # pushed out: a recording whose centroid runs far while its fingers stay
        # close does exactly that.
        material = _pinch_material(
            start_span=40.0,
            end_span=800.0,
            drift_px=700.0,
            centre=(200.0, 300.0),
            axis_rad=0.0,
        )
        with self.assertRaises(ReplayDatasetBuildError) as caught:
            self._place(
                material=material,
                request=_request(
                    centre=(540.0, 960.0),
                    span=800.0 / 1.4,
                    percent=1.4,
                    axis_rad=0.0,
                ),
            )
        self.assertIn("finger endpoints", str(caught.exception))

    def test_material_without_frozen_geometry_is_refused(self) -> None:
        material = _pinch_material()
        stripped = _Material(
            trajectory=material.trajectory,
            row={"orientation_id": 0},
            samples=material.samples,
        )
        with self.assertRaises(ReplayDatasetBuildError):
            self._place(material=stripped)

    def test_an_unknown_pointer_order_is_refused(self) -> None:
        with self.assertRaises(ReplayDatasetBuildError):
            self._place(pointer_order="sideways")

    def test_a_non_pinch_target_is_refused(self) -> None:
        with self.assertRaises(ReplayDatasetBuildError):
            _observe_pinch_area_similarity(
                _pinch_material(),
                _Target(action="swipe", orientation_id=0),
                target_samples=24,
                target_duration_ms=300.0,
                requested=_request(),
            )


if __name__ == "__main__":
    unittest.main()
