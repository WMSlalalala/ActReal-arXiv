from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.android_touch_observation import (
    ACTION_DOWN,
    ACTION_MOVE,
    ACTION_UP,
    screen_dimensions_for_orientation,
)
from pipeline.conditional_touch_request_generator import (
    ConditionalTouchRequestGenerator,
    ConditionalTouchRequestGeneratorError,
    DIRECTION8,
    RESIDUAL_QUANTILE_COUNT,
    _normal_cdf,
    _normal_inverse,
)


def _wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _direction8(dx: float, dy: float) -> str:
    angle = math.atan2(dy, dx)
    index = int(math.floor((angle + math.pi / 8.0) / (math.pi / 4.0))) % 8
    return DIRECTION8[index]


def _synthetic_rows(events_per_condition: int = 18) -> dict[str, np.ndarray]:
    rows: dict[str, list[object]] = {
        "event_id": [],
        "action": [],
        "orientation_id": [],
        "t_ms": [],
        "x_px": [],
        "y_px": [],
        "android_action": [],
    }
    lifecycle = (ACTION_DOWN, ACTION_MOVE, ACTION_UP)
    for orientation_id in (0, 1):
        width_px, height_px = screen_dimensions_for_orientation(orientation_id)
        for direction_index, direction in enumerate(DIRECTION8):
            center = direction_index * math.pi / 4.0
            for event_index in range(events_per_condition):
                phase = (
                    0.73 * event_index
                    + 0.31 * direction_index
                    + 0.17 * orientation_id
                )
                start = np.asarray(
                    (
                        width_px * (0.50 + 0.13 * math.sin(1.7 * phase)),
                        height_px * (0.50 + 0.13 * math.cos(1.3 * phase)),
                    ),
                    dtype=np.float64,
                )
                duration_ms = float(
                    85 + ((event_index * 47 + direction_index * 13) % 230)
                )
                distance_px = float(
                    8 + ((event_index * 61 + direction_index * 17) % 150)
                )
                angle = center + 0.24 * math.sin(2.1 * phase)
                endpoint = start + distance_px * np.asarray(
                    (math.cos(angle), math.sin(angle)), dtype=np.float64
                )
                event_id = (
                    f"secret-raw-event-{orientation_id}-{direction}-"
                    f"{event_index}"
                )
                for point_index, fraction in enumerate((0.0, 0.43, 1.0)):
                    point = start + fraction * (endpoint - start)
                    rows["event_id"].append(event_id)
                    rows["action"].append("scroll")
                    rows["orientation_id"].append(orientation_id)
                    rows["t_ms"].append(fraction * duration_ms)
                    rows["x_px"].append(float(point[0]))
                    rows["y_px"].append(float(point[1]))
                    rows["android_action"].append(lifecycle[point_index])
    return {name: np.asarray(values) for name, values in rows.items()}


def _synthetic_tap_rows() -> dict[str, np.ndarray]:
    rows: dict[str, list[object]] = {
        "event_id": [],
        "action": [],
        "orientation_id": [],
        "t_ms": [],
        "x_px": [],
        "y_px": [],
        "android_action": [],
    }
    vectors = (
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
    )

    def append_event(
        event_id: str,
        start: tuple[float, float],
        end: tuple[float, float],
        duration_ms: float,
    ) -> None:
        for index, fraction in enumerate((0.0, 0.5, 1.0)):
            rows["event_id"].append(event_id)
            rows["action"].append("tap")
            rows["orientation_id"].append(0)
            rows["t_ms"].append(fraction * duration_ms)
            rows["x_px"].append(start[0] + fraction * (end[0] - start[0]))
            rows["y_px"].append(start[1] + fraction * (end[1] - start[1]))
            rows["android_action"].append(
                (ACTION_DOWN, ACTION_MOVE, ACTION_UP)[index]
            )

    for index in range(32):
        start = (float(200 + 17 * index), float(500 + 19 * index))
        append_event(
            f"secret-tap-stationary-{index}",
            start,
            start,
            float(50 + (index * 13) % 100),
        )
    for direction_index, vector in enumerate(vectors):
        for index in range(4):
            start = (
                float(300 + 23 * index + 5 * direction_index),
                float(800 + 17 * index + 3 * direction_index),
            )
            scale = float(3 + index)
            end = (
                start[0] + scale * vector[0],
                start[1] + scale * vector[1],
            )
            append_event(
                f"secret-tap-moving-{direction_index}-{index}",
                start,
                end,
                float(60 + 11 * index + direction_index),
            )
    return {name: np.asarray(values) for name, values in rows.items()}


class ConditionalTouchRequestGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_rows()
        cls.generator = ConditionalTouchRequestGenerator.fit_from_raw_rows(
            cls.rows,
            ridge=1.0e-3,
            minimum_events_per_condition=10,
            training_source_sha256s=("b" * 64,),
        )

    def test_all_orientation_direction_conditions_are_exact_and_deterministic(
        self,
    ) -> None:
        expected_conditions = {
            ("scroll", orientation_id, direction)
            for orientation_id in (0, 1)
            for direction in DIRECTION8
        }
        self.assertEqual(
            set(self.generator.supported_conditions), expected_conditions
        )
        changed_seed_count = 0
        for orientation_id in (0, 1):
            width_px, height_px = screen_dimensions_for_orientation(
                orientation_id
            )
            for direction_index, direction in enumerate(DIRECTION8):
                arguments = dict(
                    action="scroll",
                    orientation_id=orientation_id,
                    direction=direction,
                    start_xy_px=(0.50 * width_px, 0.50 * height_px),
                    duration_ms=170.0,
                )
                generated = self.generator.generate(seed=9103, **arguments)
                repeated = self.generator.generate(seed=9103, **arguments)
                changed = self.generator.generate(seed=9104, **arguments)
                self.assertEqual(generated, repeated)
                if changed.end_xy_px != generated.end_xy_px:
                    changed_seed_count += 1

                start = np.asarray(generated.start_xy_px)
                endpoint = np.asarray(generated.end_xy_px)
                chord = endpoint - start
                self.assertTrue(np.isfinite(endpoint).all())
                self.assertGreaterEqual(endpoint[0], 0.0)
                self.assertLessEqual(endpoint[0], width_px)
                self.assertGreaterEqual(endpoint[1], 0.0)
                self.assertLessEqual(endpoint[1], height_px)
                self.assertEqual(_direction8(float(chord[0]), float(chord[1])), direction)
                self.assertLess(
                    abs(
                        _wrap_pi(
                            math.atan2(float(chord[1]), float(chord[0]))
                            - direction_index * math.pi / 4.0
                        )
                    ),
                    math.pi / 8.0,
                )
                self.assertAlmostEqual(
                    float(np.linalg.norm(chord)), generated.distance_px, places=10
                )
                self.assertLess(
                    generated.distance_px, generated.available_distance_px
                )
        self.assertEqual(changed_seed_count, len(expected_conditions))

    def test_near_boundary_sampling_is_conditioned_not_clipped(self) -> None:
        support_probabilities: list[float] = []
        for seed in range(40):
            generated = self.generator.generate(
                action="scroll",
                orientation_id=0,
                direction="right",
                start_xy_px=(900.0, 960.0),
                duration_ms=100.0,
                seed=seed,
            )
            self.assertEqual(
                _direction8(
                    generated.end_xy_px[0] - generated.start_xy_px[0],
                    generated.end_xy_px[1] - generated.start_xy_px[1],
                ),
                "right",
            )
            self.assertGreaterEqual(generated.end_xy_px[0], 0.0)
            self.assertLessEqual(generated.end_xy_px[0], 1080.0)
            self.assertGreaterEqual(generated.end_xy_px[1], 0.0)
            self.assertLessEqual(generated.end_xy_px[1], 1920.0)
            self.assertLess(
                generated.distance_px, generated.available_distance_px
            )
            support_probabilities.append(
                generated.conditional_support_probability
            )
        self.assertLess(max(support_probabilities), 1.0)
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "no positive screen ray"
        ):
            self.generator.generate(
                action="scroll",
                orientation_id=0,
                direction="right",
                start_xy_px=(1080.0, 960.0),
                duration_ms=170.0,
                seed=0,
            )

    def test_compact_save_load_roundtrip_contains_no_raw_ids_or_donors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "request_model.npz"
            artifact_sha256 = self.generator.save(artifact)
            self.assertLess(artifact.stat().st_size, 500 * 1024)
            loaded = ConditionalTouchRequestGenerator.load(artifact)
            self.assertEqual(loaded.artifact_sha256, artifact_sha256)
            arguments = dict(
                action="scroll",
                orientation_id=1,
                direction="up_left",
                start_xy_px=(900.0, 540.0),
                duration_ms=204.0,
                seed=4242,
            )
            self.assertEqual(
                self.generator.generate(**arguments), loaded.generate(**arguments)
            )
            metadata = loaded.metadata
            self.assertFalse(metadata["raw_event_ids_retained"])
            self.assertFalse(metadata["raw_trajectories_retained"])
            self.assertFalse(metadata["runtime_donor_lookup_used"])
            self.assertFalse(metadata["coordinate_clipping_used"])
            with np.load(artifact, allow_pickle=False) as archive:
                manifest_text = str(np.asarray(archive["manifest_json"]).item())
                self.assertNotIn("secret-raw-event", manifest_text)
                for name in archive.files:
                    self.assertNotIn("secret-raw-event", name)
                    self.assertNotEqual(np.asarray(archive[name]).dtype, object)
                    if name.endswith("_residual_quantiles"):
                        self.assertEqual(
                            np.asarray(archive[name]).shape,
                            (RESIDUAL_QUANTILE_COUNT,),
                        )

    def test_load_fails_closed_when_source_binding_is_forged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "request_model.npz"
            forged = Path(directory) / "forged.npz"
            self.generator.save(artifact)
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name]) for name in archive.files
                }
            manifest = json.loads(str(arrays["manifest_json"].item()))
            manifest["request_generator_source_sha256"] = "0" * 64
            arrays["manifest_json"] = np.asarray(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with forged.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            with self.assertRaisesRegex(
                ConditionalTouchRequestGeneratorError,
                "does not match imported code",
            ):
                ConditionalTouchRequestGenerator.load(forged)

    def test_input_contracts_fail_explicitly(self) -> None:
        common = dict(
            action="scroll",
            orientation_id=0,
            direction="right",
            start_xy_px=(540.0, 960.0),
            duration_ms=170.0,
            seed=1,
        )
        invalid = dict(common)
        invalid["start_xy_px"] = (-1.0, 100.0)
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "leaves the physical screen"
        ):
            self.generator.generate(**invalid)
        invalid = dict(common)
        invalid["duration_ms"] = 0.0
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "finite and positive"
        ):
            self.generator.generate(**invalid)
        invalid = dict(common)
        invalid["direction"] = "stationary"
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "no fitted endpoint model"
        ):
            self.generator.generate(**invalid)

    def test_fit_rejects_missing_fields_and_sparse_conditions(self) -> None:
        missing = dict(self.rows)
        del missing["t_ms"]
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "missing required fields"
        ):
            ConditionalTouchRequestGenerator.fit_from_raw_rows(missing)
        sparse = _synthetic_rows(events_per_condition=2)
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError,
            "below minimum_events_per_condition",
        ):
            ConditionalTouchRequestGenerator.fit_from_raw_rows(
                sparse, minimum_events_per_condition=3
            )

    def test_python37_normal_fallback_accuracy_and_no_scipy_dependency(self) -> None:
        for probability in (1.0e-8, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1.0e-8):
            self.assertAlmostEqual(
                _normal_cdf(_normal_inverse(probability)),
                probability,
                places=7,
            )
        module_path = Path(
            __import__(
                "pipeline.conditional_touch_request_generator",
                fromlist=["__file__"],
            ).__file__
        )
        self.assertNotIn("scipy", module_path.read_text().lower())

    def test_tap_full_pair_preserves_stationary_mass_and_never_binds_start(
        self,
    ) -> None:
        tap = ConditionalTouchRequestGenerator.fit_from_raw_rows(
            _synthetic_tap_rows(), minimum_events_per_condition=3
        )
        self.assertEqual(tap.supported_conditions, (("tap", 0, "full_pair"),))
        outputs = [
            tap.generate(
                action="tap", orientation_id=0, duration_ms=90.0, seed=seed
            )
            for seed in range(80)
        ]
        self.assertTrue(any(output.stationary for output in outputs))
        self.assertTrue(any(not output.stationary for output in outputs))
        for output in outputs:
            self.assertGreaterEqual(output.start_xy_px[0], 0.0)
            self.assertLessEqual(output.start_xy_px[0], 1080.0)
            self.assertGreaterEqual(output.start_xy_px[1], 0.0)
            self.assertLessEqual(output.start_xy_px[1], 1920.0)
            self.assertGreaterEqual(output.end_xy_px[0], 0.0)
            self.assertLessEqual(output.end_xy_px[0], 1080.0)
            self.assertGreaterEqual(output.end_xy_px[1], 0.0)
            self.assertLessEqual(output.end_xy_px[1], 1920.0)
            if output.stationary:
                self.assertEqual(output.start_xy_px, output.end_xy_px)
                self.assertEqual(output.distance_px, 0.0)
                self.assertEqual(output.direction, "stationary")
            else:
                self.assertNotEqual(output.start_xy_px, output.end_xy_px)
                self.assertEqual(
                    _direction8(
                        output.end_xy_px[0] - output.start_xy_px[0],
                        output.end_xy_px[1] - output.start_xy_px[1],
                    ),
                    output.direction,
                )
        self.assertEqual(outputs[7], tap.generate(
            action="tap", orientation_id=0, duration_ms=90.0, seed=7
        ))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "tap_request.npz"
            tap.save(artifact)
            self.assertLess(artifact.stat().st_size, 500 * 1024)
            loaded = ConditionalTouchRequestGenerator.load(artifact)
            self.assertEqual(
                outputs[7],
                loaded.generate(
                    action="tap", orientation_id=0, duration_ms=90.0, seed=7
                ),
            )
            with np.load(artifact, allow_pickle=False) as archive:
                self.assertNotIn(
                    "secret-tap", str(np.asarray(archive["manifest_json"]).item())
                )
        with self.assertRaisesRegex(
            ConditionalTouchRequestGeneratorError, "do not accept a bound start"
        ):
            tap.generate(
                action="tap",
                orientation_id=0,
                duration_ms=90.0,
                seed=1,
                start_xy_px=(100.0, 100.0),
            )

    def test_swipe_reuses_exact_conditioned_endpoint_contract(self) -> None:
        rows = _synthetic_rows()
        rows["action"] = np.full(len(rows["action"]), "swipe")
        swipe = ConditionalTouchRequestGenerator.fit_from_raw_rows(
            rows, minimum_events_per_condition=10
        )
        generated = swipe.generate(
            action="swipe",
            orientation_id=0,
            direction="down_left",
            start_xy_px=(540.0, 960.0),
            duration_ms=170.0,
            seed=55,
        )
        self.assertEqual(
            _direction8(
                generated.end_xy_px[0] - generated.start_xy_px[0],
                generated.end_xy_px[1] - generated.start_xy_px[1],
            ),
            "down_left",
        )
        self.assertFalse(generated.stationary)


if __name__ == "__main__":
    unittest.main()
