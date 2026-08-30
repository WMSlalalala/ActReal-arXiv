from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.conditional_touch_generator import (
    ConditionalTouchGenerator,
    ConditionalTouchGeneratorError,
    IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256,
    IMPORT_GENERATOR_SOURCE_SHA256,
    IMPORT_SOURCE_FINGERPRINT_SHA256,
    _evaluate_empirical_quantiles,
    _interpolate_last_value,
    _pause_preserving_support_curve,
    _sample_joint_shape_targets,
)


def _synthetic_rows(event_count: int = 8) -> dict[str, np.ndarray]:
    rows: dict[str, list[object]] = {
        "event_id": [],
        "action": [],
        "orientation_id": [],
        "t_ms": [],
        "x_px": [],
        "y_px": [],
        "pressure": [],
        "android_action": [],
    }
    relative_t = np.asarray((-10.0, 0.0, 25.0, 50.0, 75.0, 100.0))
    lifecycle = np.asarray((2, 0, 2, 2, 2, 1), dtype=np.int64)
    for action in ("tap", "scroll", "swipe"):
        for event_index in range(event_count):
            event_id = f"secret-{action}-{event_index}"
            anchor_x = 200.0 + 7.0 * event_index
            anchor_y = 700.0 + 3.0 * event_index
            u = np.asarray((0.0, 0.0, 0.25, 0.50, 0.75, 1.0))
            if action == "tap":
                if event_index < event_count // 2:
                    x = np.full(len(u), anchor_x)
                    y = np.full(len(u), anchor_y)
                else:
                    amplitude = 3.0 + event_index
                    # Real moving taps have a small DOWN->UP displacement;
                    # train data contains no moving exact-endpoint tap loops.
                    x = anchor_x + amplitude * (
                        (0.75 + 0.05 * event_index) * np.sin(np.pi * u)
                        + 0.8 * u
                    )
                    y = anchor_y - (
                        0.35 + 0.04 * event_index
                    ) * amplitude * np.sin(2.0 * np.pi * u)
                    # Mix early and middle exact holds so the fitted tap
                    # increment model has a nontrivial pause schedule.
                    pause_row = 2 if event_index % 2 == 0 else 3
                    x[pause_row] = x[pause_row - 1]
                    y[pause_row] = y[pause_row - 1]
            else:
                length = (180.0 if action == "scroll" else 360.0) + 8.0 * event_index
                lateral = -(0.10 + 0.01 * event_index) * length * np.sin(np.pi * u)
                longitudinal = length * u + (2.0 + event_index) * np.sin(
                    2.0 * np.pi * u
                )
                x = anchor_x + longitudinal
                y = anchor_y + lateral
            pressure = 0.35 + 0.04 * np.sin(np.pi * u) + 0.005 * event_index
            # The leading MOVE deliberately has a wild coordinate.  Fitting
            # must locate ACTION_DOWN instead of treating row zero as start.
            x[0] = 1000.0
            y[0] = 1800.0
            for row_index in range(len(relative_t)):
                rows["event_id"].append(event_id)
                rows["action"].append(action)
                rows["orientation_id"].append(0)
                rows["t_ms"].append(relative_t[row_index] + 5.0 * event_index)
                rows["x_px"].append(x[row_index])
                rows["y_px"].append(y[row_index])
                rows["pressure"].append(pressure[row_index])
                rows["android_action"].append(lifecycle[row_index])
    return {name: np.asarray(values) for name, values in rows.items()}


class ConditionalTouchGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_rows()
        cls.generator = ConditionalTouchGenerator.fit_from_raw_rows(
            cls.rows,
            grid_size=25,
            max_rank=5,
            training_source_sha256s=("a" * 64,),
        )

    def test_scroll_exact_endpoints_and_timeline(self) -> None:
        timeline = np.asarray((4000.0, 4000.0, 4020.0, 4075.0, 4130.0))
        generated = self.generator.generate(
            action="scroll",
            orientation_id=0,
            start_xy_px=(120.25, 300.75),
            end_xy_px=(520.5, 701.0),
            direction="down_right",
            seed=91,
            t_ms=timeline,
        )
        np.testing.assert_array_equal(generated.t_ms, timeline)
        np.testing.assert_array_equal(generated.x_px[[0, -1]], (120.25, 520.5))
        np.testing.assert_array_equal(generated.y_px[[0, -1]], (300.75, 701.0))
        self.assertTrue(np.all((generated.pressure >= 0.0) & (generated.pressure <= 1.0)))
        np.testing.assert_array_equal(generated.android_action, (0, 2, 2, 2, 1))
        np.testing.assert_array_equal(generated.frame_index, (0, 0, 1, 2, 3))
        np.testing.assert_array_equal(generated.frame_end, (False, True, True, True, True))

    def test_seed_is_deterministic_and_changes_stochastic_path(self) -> None:
        arguments = dict(
            action="swipe",
            orientation_id=0,
            start_xy_px=(100.0, 900.0),
            end_xy_px=(700.0, 900.0),
            direction="right",
            duration_ms=330.0,
            sample_count=31,
        )
        first = self.generator.generate(seed=123, **arguments)
        repeated = self.generator.generate(seed=123, **arguments)
        changed = self.generator.generate(seed=124, **arguments)
        np.testing.assert_array_equal(first.x_px, repeated.x_px)
        np.testing.assert_array_equal(first.y_px, repeated.y_px)
        np.testing.assert_array_equal(first.pressure, repeated.pressure)
        self.assertGreater(
            float(np.max(np.abs(first.y_px - changed.y_px))),
            1.0e-9,
        )

    def test_direction_must_match_endpoint_sector(self) -> None:
        with self.assertRaisesRegex(
            ConditionalTouchGeneratorError, "does not match endpoint sector"
        ):
            self.generator.generate(
                action="scroll",
                orientation_id=0,
                start_xy_px=(100.0, 100.0),
                end_xy_px=(600.0, 100.0),
                direction="left",
                seed=0,
                duration_ms=300.0,
                sample_count=20,
            )

    def test_fit_rejects_multiple_down_lifecycle(self) -> None:
        rows = {name: value.copy() for name, value in self.rows.items()}
        first_event = rows["event_id"] == "secret-tap-0"
        first_row = int(np.flatnonzero(first_event)[0])
        rows["android_action"][first_row] = 0
        generator = ConditionalTouchGenerator.fit_from_raw_rows(
            rows,
            grid_size=9,
            max_rank=1,
        )
        self.assertEqual(generator.metadata["rejected_event_count"], 1)
        self.assertEqual(
            generator.metadata["rejected_event_reasons"],
            {"event must have exactly one ACTION_DOWN": 1},
        )

    def test_fit_duplicate_timestamps_use_last_value(self) -> None:
        observed = _interpolate_last_value(
            np.asarray((0.0, 0.0, 10.0, 20.0)),
            np.asarray((1.0, 7.0, 11.0, 20.0)),
            np.asarray((0.0, 0.5, 1.0)),
        )
        np.testing.assert_array_equal(observed, (7.0, 11.0, 20.0))

    def test_tap_equal_endpoint_contract_selects_stationary_support(self) -> None:
        probability = self.generator.metadata["tap_stationary_probabilities"][
            "tap|0"
        ]
        self.assertAlmostEqual(probability, 0.5)
        support = self.generator.metadata["tap_exact_endpoint_support"]["tap|0"]
        self.assertEqual(support["equal_endpoint_moved_count"], 0)
        self.assertEqual(support["equal_endpoint_stationary_count"], 4)
        for seed in range(20):
            generated = self.generator.generate(
                action="tap",
                orientation_id=0,
                start_xy_px=(410.0, 820.0),
                end_xy_px=(410.0, 820.0),
                direction="stationary",
                seed=seed,
                duration_ms=100.0,
                sample_count=11,
            )
            self.assertTrue(generated.tap_stationary_branch)
            np.testing.assert_array_equal(generated.x_px, np.full(11, 410.0))
            np.testing.assert_array_equal(generated.y_px, np.full(11, 820.0))

    def test_tap_unequal_endpoints_use_moving_support_exactly(self) -> None:
        arguments = dict(
            action="tap",
            orientation_id=0,
            start_xy_px=(410.25, 820.5),
            end_xy_px=(417.75, 816.0),
            direction=None,
            duration_ms=100.0,
            sample_count=11,
        )
        generated = self.generator.generate(seed=23, **arguments)
        repeated = self.generator.generate(seed=23, **arguments)
        changed = self.generator.generate(seed=24, **arguments)
        self.assertFalse(generated.tap_stationary_branch)
        np.testing.assert_array_equal(
            generated.x_px[[0, -1]], (410.25, 417.75)
        )
        np.testing.assert_array_equal(generated.y_px[[0, -1]], (820.5, 816.0))
        np.testing.assert_array_equal(generated.x_px, repeated.x_px)
        np.testing.assert_array_equal(generated.y_px, repeated.y_px)
        self.assertGreater(
            float(
                np.max(
                    np.abs(generated.x_px[1:-1] - changed.x_px[1:-1])
                )
            ),
            1.0e-6,
        )
        self.assertTrue(np.all((generated.x_px >= 0.0) & (generated.x_px <= 1080.0)))
        self.assertTrue(np.all((generated.y_px >= 0.0) & (generated.y_px <= 1920.0)))

    def test_moving_tap_smooth_residual_preserves_exact_endpoints(self) -> None:
        arguments = dict(
            action="tap",
            orientation_id=0,
            start_xy_px=(410.25, 820.5),
            end_xy_px=(417.75, 816.0),
            direction=None,
            t_ms=np.asarray((0.0, 25.0, 50.0, 75.0, 100.0)),
        )
        expected = np.asarray(
            (arguments["start_xy_px"], arguments["end_xy_px"]),
            dtype=np.float64,
        )
        distinct_paths: set[bytes] = set()
        for seed in range(40):
            generated = self.generator.generate(seed=seed, **arguments)
            points = np.column_stack((generated.x_px, generated.y_px))
            np.testing.assert_array_equal(points[[0, -1]], expected)
            self.assertTrue(np.isfinite(points).all())
            self.assertTrue(np.all((points[:, 0] >= 0.0) & (points[:, 0] <= 1080.0)))
            self.assertTrue(np.all((points[:, 1] >= 0.0) & (points[:, 1] <= 1920.0)))
            distinct_paths.add(points.tobytes())
        self.assertGreater(len(distinct_paths), 1)

    def test_moving_tap_boundary_fit_is_exact_finite_and_in_screen(self) -> None:
        expected = np.asarray(((0.0, 100.0), (0.0, 112.0)))
        for seed in range(40):
            generated = self.generator.generate(
                action="tap",
                orientation_id=0,
                start_xy_px=expected[0],
                end_xy_px=expected[1],
                direction=None,
                seed=seed,
                duration_ms=100.0,
                sample_count=11,
            )
            points = np.column_stack((generated.x_px, generated.y_px))
            np.testing.assert_array_equal(points[[0, -1]], expected)
            self.assertTrue(np.isfinite(points).all())
            self.assertTrue(np.all(points[:, 0] >= 0.0))
            self.assertTrue(np.all(points[:, 0] <= 1080.0))
            self.assertTrue(np.all(points[:, 1] >= 0.0))
            self.assertTrue(np.all(points[:, 1] <= 1920.0))

    def test_moving_tap_rejects_mismatched_direction(self) -> None:
        with self.assertRaisesRegex(
            ConditionalTouchGeneratorError, "moving tap direction"
        ):
            self.generator.generate(
                action="tap",
                orientation_id=0,
                start_xy_px=(100.0, 200.0),
                end_xy_px=(110.0, 200.0),
                direction="left",
                seed=0,
                duration_ms=100.0,
                sample_count=11,
            )

    def test_pressure_exact_one_point_mass_is_preserved(self) -> None:
        rows = {name: value.copy() for name, value in self.rows.items()}
        rows["pressure"][:] = 1.0
        generator = ConditionalTouchGenerator.fit_from_raw_rows(
            rows,
            grid_size=9,
            max_rank=1,
        )
        for action, end, direction in (
            ("tap", (100.0, 200.0), "stationary"),
            ("scroll", (500.0, 200.0), "right"),
            ("swipe", (500.0, 200.0), "right"),
        ):
            generated = generator.generate(
                action=action,
                orientation_id=0,
                start_xy_px=(100.0, 200.0),
                end_xy_px=end,
                direction=direction,
                seed=7,
                duration_ms=100.0,
                sample_count=11,
            )
            np.testing.assert_array_equal(generated.pressure, np.ones(11))

    def test_screen_fit_uses_no_coordinate_clipping(self) -> None:
        common = dict(
            action="scroll",
            orientation_id=0,
            end_xy_px=(600.0, 0.5),
            direction="right",
            seed=17,
            duration_ms=400.0,
            sample_count=41,
            minimum_residual_scale=0.0,
        )
        boundary = self.generator.generate(
            start_xy_px=(100.0, 0.5),
            **common,
        )
        center = self.generator.generate(
            start_xy_px=(100.0, 900.0),
            end_xy_px=(600.0, 900.0),
            action=common["action"],
            orientation_id=common["orientation_id"],
            direction=common["direction"],
            seed=common["seed"],
            duration_ms=common["duration_ms"],
            sample_count=common["sample_count"],
            minimum_residual_scale=common["minimum_residual_scale"],
        )
        self.assertLess(boundary.residual_scale, 1.0)
        self.assertAlmostEqual(center.residual_scale, 1.0)
        self.assertLessEqual(boundary.residual_scale, center.residual_scale)
        np.testing.assert_array_equal(boundary.x_px[[0, -1]], (100.0, 600.0))
        np.testing.assert_array_equal(boundary.y_px[[0, -1]], (0.5, 0.5))
        self.assertTrue(np.all(boundary.y_px >= 0.0))

    def test_increment_shape_and_lattice_parameters_are_train_fitted_and_audited(self) -> None:
        metadata = self.generator.metadata
        for action in ("scroll", "swipe"):
            key = f"{action}|0"
            increment = metadata["increment_parameter_audit"][key]
            shape = metadata["shape_parameter_audit"][key]
            self.assertGreater(increment["training_interval_count"], 0)
            self.assertGreater(len(increment["autoregression"]), 0)
            self.assertEqual(shape["hard_support_quantile"], 0.9)
            self.assertGreater(shape["coordinate_lattice_quantum_px"], 0.0)
            self.assertGreaterEqual(shape["coordinate_lattice_probability"], 0.0)
            self.assertLessEqual(shape["coordinate_lattice_probability"], 1.0)
        tap_increment = metadata["increment_parameter_audit"]["tap|0"]
        tap_shape = metadata["shape_parameter_audit"]["tap|0"]
        self.assertEqual(tap_increment["training_event_count"], 4)
        self.assertEqual(len(tap_increment["amplitude_feature_center"]), 3)
        self.assertFalse(tap_increment["coordinate_quantization_applied"])
        self.assertEqual(tap_shape["training_event_count"], 4)
        self.assertIn(
            "lateral_max_normalized_constant_scale",
            tap_shape["calibrated_metrics"],
        )
        self.assertFalse(metadata["runtime_donor_lookup_used"])

    def test_joint_shape_sampling_preserves_lower_cdf_and_winsorizes_q90(self) -> None:
        shape = self.generator._models[("scroll", 0)].shape_model
        self.assertIsNotNone(shape)
        rng = np.random.default_rng(123)
        sampled = np.stack(
            [_sample_joint_shape_targets(shape, rng) for _ in range(20_000)]
        )
        quantile_arrays = (
            shape.path_chord_quantiles,
            shape.lateral_excursion_quantiles,
            shape.lateral_rms_quantiles,
            shape.longitudinal_excursion_quantiles,
            shape.longitudinal_reversal_quantiles,
            shape.half_pixel_fraction_quantiles,
        )
        for channel, quantiles in enumerate(quantile_arrays):
            expected_median = _evaluate_empirical_quantiles(quantiles, 0.5)
            expected_q90 = _evaluate_empirical_quantiles(quantiles, 0.9)
            scale = max(1.0e-6, float(np.ptp(quantiles)))
            self.assertAlmostEqual(
                float(np.median(sampled[:, channel])),
                expected_median,
                delta=0.03 * scale,
            )
            self.assertLessEqual(
                float(np.max(sampled[:, channel])),
                expected_q90 + 1.0e-12,
            )

    def test_explicit_detector_sample_count_is_supported(self) -> None:
        generated = self.generator.generate(
            action="scroll",
            orientation_id=0,
            start_xy_px=(100.0, 300.0),
            end_xy_px=(600.0, 300.0),
            direction="right",
            seed=19,
            duration_ms=500.0,
            sample_count=31,
            detector_sample_count=169,
        )
        np.testing.assert_array_equal(generated.x_px[[0, -1]], (100.0, 600.0))
        with self.assertRaisesRegex(
            ConditionalTouchGeneratorError, "detector_sample_count"
        ):
            self.generator.generate(
                action="scroll",
                orientation_id=0,
                start_xy_px=(100.0, 300.0),
                end_xy_px=(600.0, 300.0),
                direction="right",
                seed=19,
                duration_ms=500.0,
                sample_count=31,
                detector_sample_count=1,
            )

    def test_legal_boundary_endpoints_can_reduce_residual_to_zero(self) -> None:
        generated = self.generator.generate(
            action="scroll",
            orientation_id=0,
            start_xy_px=(100.0, 0.0),
            end_xy_px=(600.0, 0.0),
            direction="right",
            seed=17,
            duration_ms=400.0,
            sample_count=41,
        )
        self.assertEqual(generated.residual_scale, 0.0)
        np.testing.assert_array_equal(generated.x_px[[0, -1]], (100.0, 600.0))
        np.testing.assert_array_equal(generated.y_px, np.zeros(41))
        self.assertTrue(np.all((generated.x_px >= 0.0) & (generated.x_px <= 1080.0)))

    def test_diagonal_boundary_seed27_is_strictly_in_screen(self) -> None:
        generated = self.generator.generate(
            action="scroll",
            orientation_id=0,
            start_xy_px=(0.0, 0.0),
            end_xy_px=(1080.0, 1920.0),
            direction="down_right",
            seed=27,
            duration_ms=3729.0,
            sample_count=147,
        )
        self.assertTrue(np.all(generated.x_px >= 0.0))
        self.assertTrue(np.all(generated.x_px <= 1080.0))
        self.assertTrue(np.all(generated.y_px >= 0.0))
        self.assertTrue(np.all(generated.y_px <= 1920.0))
        np.testing.assert_array_equal(generated.x_px[[0, -1]], (0.0, 1080.0))
        np.testing.assert_array_equal(generated.y_px[[0, -1]], (0.0, 1920.0))

    def test_pause_support_progress_is_canonical_monotone_unit_interval(self) -> None:
        rng = np.random.default_rng(27)
        weights = np.exp(rng.normal(size=146))
        weights[[5, 17, 90]] = 0.0
        progress = np.cumsum(weights) / np.sum(weights)
        local_curve = np.column_stack(
            (
                np.concatenate(([0.0], progress)),
                np.zeros(147, dtype=np.float64),
            )
        )
        support = _pause_preserving_support_curve(local_curve)
        self.assertTrue(np.all(support[:, 0] >= 0.0))
        self.assertTrue(np.all(support[:, 0] <= 1.0))
        self.assertTrue(np.all(np.diff(support[:, 0]) >= 0.0))
        self.assertEqual(support[0, 0], 0.0)
        self.assertEqual(support[-1, 0], 1.0)

    def test_compact_round_trip_contains_no_raw_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "touch_generator.npz"
            digest = self.generator.save(path)
            self.assertEqual(len(digest), 64)
            self.assertLess(path.stat().st_size, 500_000)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files) & {"event_id", "donor_id"}, set())
                manifest = str(np.asarray(archive["manifest_json"]).item())
                self.assertNotIn("secret-tap-0", manifest)
                parsed = json.loads(manifest)
                self.assertFalse(parsed["metadata"]["raw_event_ids_retained"])
                self.assertEqual(
                    parsed["generator_source_sha256"],
                    IMPORT_GENERATOR_SOURCE_SHA256,
                )
                self.assertEqual(
                    parsed["android_touch_observation_source_sha256"],
                    IMPORT_ANDROID_TOUCH_OBSERVATION_SOURCE_SHA256,
                )
                self.assertEqual(
                    parsed["source_fingerprint_sha256"],
                    IMPORT_SOURCE_FINGERPRINT_SHA256,
                )
                for field in (
                    "generator_source_sha256",
                    "android_touch_observation_source_sha256",
                    "source_fingerprint_sha256",
                ):
                    self.assertEqual(parsed["metadata"][field], parsed[field])
            loaded = ConditionalTouchGenerator.load(path)
            self.assertEqual(loaded.artifact_sha256, digest)
            self.assertEqual(loaded.supported_conditions, self.generator.supported_conditions)
            arguments = dict(
                action="swipe",
                orientation_id=0,
                start_xy_px=(100.0, 500.0),
                end_xy_px=(600.0, 500.0),
                direction="right",
                seed=77,
                duration_ms=300.0,
                sample_count=20,
            )
            original = self.generator.generate(**arguments)
            restored = loaded.generate(**arguments)
            np.testing.assert_array_equal(original.x_px, restored.x_px)
            np.testing.assert_array_equal(original.y_px, restored.y_px)
            np.testing.assert_array_equal(original.pressure, restored.pressure)
            tap_arguments = dict(
                action="tap",
                orientation_id=0,
                start_xy_px=(410.25, 820.5),
                end_xy_px=(417.75, 816.0),
                direction=None,
                seed=31,
                duration_ms=100.0,
                sample_count=11,
            )
            original_tap = self.generator.generate(**tap_arguments)
            restored_tap = loaded.generate(**tap_arguments)
            np.testing.assert_array_equal(original_tap.x_px, restored_tap.x_px)
            np.testing.assert_array_equal(original_tap.y_px, restored_tap.y_px)
            np.testing.assert_array_equal(
                original_tap.pressure,
                restored_tap.pressure,
            )

    def test_load_rejects_any_source_binding_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "touch_generator.npz"
            self.generator.save(path)
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
            original_manifest = json.loads(str(payload["manifest_json"].item()))
            fields = (
                "generator_source_sha256",
                "android_touch_observation_source_sha256",
                "source_fingerprint_sha256",
            )
            for location in ("manifest", "metadata"):
                for field in fields:
                    with self.subTest(location=location, field=field):
                        manifest = json.loads(json.dumps(original_manifest))
                        if location == "manifest":
                            manifest[field] = "0" * 64
                        else:
                            manifest["metadata"][field] = "0" * 64
                        tampered_payload = dict(payload)
                        tampered_payload["manifest_json"] = np.asarray(
                            json.dumps(
                                manifest, sort_keys=True, separators=(",", ":")
                            )
                        )
                        tampered = Path(directory) / f"{location}-{field}.npz"
                        np.savez_compressed(tampered, **tampered_payload)
                        with self.assertRaisesRegex(
                            ConditionalTouchGeneratorError,
                            field,
                        ):
                            ConditionalTouchGenerator.load(tampered)


if __name__ == "__main__":
    unittest.main()
