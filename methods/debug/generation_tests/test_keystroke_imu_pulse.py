from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.audit import sha256_array, sha256_file
from pipeline.keystroke_imu_pulse import (
    PulseSourceEvent,
    fit_keystroke_imu_pulse_model,
    generate_keystroke_imu,
    load_user_pulse_sources,
    pulse_source_from_material,
)


class ResidualPoolTest(unittest.TestCase):
    """The pool is pasted anywhere, so it must not carry an impact."""

    def _source(self) -> PulseSourceEvent:
        imu = np.full((200, 6), 0.05, dtype=np.float32)
        # One unmistakable impact per press and nothing at all between them.
        for press in (300.0, 900.0, 1500.0):
            index = int(press / 10.0)
            imu[index:index + 3, 0] = np.asarray([2.0, 6.0, 2.0], dtype=np.float32)
        return PulseSourceEvent(
            event_id="source",
            imu=imu,
            key_down_ms=np.asarray([300.0, 900.0, 1500.0]),
            is_letter=np.asarray([True, True, True]),
        )

    def test_no_pooled_block_carries_an_impact(self) -> None:
        model = fit_keystroke_imu_pulse_model(
            [self._source()], user_id="hmog_u000", pre_ms=20.0, post_ms=60.0,
            residual_block_ms=100.0,
        )
        self.assertTrue(model.residual_blocks)
        impact = float(np.max(np.abs(self._source().imu)))
        for index, block in enumerate(model.residual_blocks):
            with self.subTest(block=index):
                self.assertLess(float(np.max(np.abs(block))), impact / 2.0)

    def test_a_source_with_no_gap_still_builds(self) -> None:
        """Dense enough typing leaves no quiet stretch; refusing to build is worse."""

        presses = np.arange(50.0, 550.0, 50.0)
        imu = np.full((60, 6), 0.05, dtype=np.float32)
        for press in presses:
            imu[int(press / 10.0), 0] = 5.0
        model = fit_keystroke_imu_pulse_model(
            [
                PulseSourceEvent(
                    event_id="dense", imu=imu, key_down_ms=presses,
                    is_letter=np.ones(len(presses), dtype=bool),
                )
            ],
            user_id="hmog_u001", pre_ms=20.0, post_ms=60.0,
            residual_block_ms=100.0,
        )
        self.assertTrue(model.residual_blocks)


class KeystrokeImuPulseTest(unittest.TestCase):
    def test_fit_uses_explicit_absolute_source_timeline(self) -> None:
        timeline = 1_000.0 + np.arange(100, dtype=np.float64) * 10.0
        imu = np.zeros((100, 6), dtype=np.float32)
        imu[49:52, 0] = np.asarray([1.0, 3.0, 1.0], dtype=np.float32)
        model = fit_keystroke_imu_pulse_model(
            [
                PulseSourceEvent(
                    event_id="source",
                    imu=imu,
                    key_down_ms=np.asarray([1_500.0]),
                    is_letter=np.asarray([True]),
                    imu_t_ms=timeline,
                )
            ],
            user_id="hmog_u000",
            pre_ms=10.0,
            post_ms=20.0,
        )
        self.assertEqual(model.explicit_timeline_sources, 1)
        self.assertEqual(model.letter_observations, 1)
        generated = generate_keystroke_imu(
            model,
            key_down_ms=[100.0, 300.0],
            is_letter=[True, False],
            duration_ms=500.0,
            seed=7,
        )
        self.assertEqual(generated.imu.shape, (50, 6))
        self.assertTrue(np.isfinite(generated.imu).all())

    def test_material_loader_reads_native_slice_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = root / "native.npz"
            timestamp_ns = np.arange(20, dtype=np.int64) * 10_000_000
            imu = np.arange(120, dtype=np.float32).reshape(20, 6)
            np.savez_compressed(
                stream,
                timestamp_ns=timestamp_ns,
                imu=imu,
                sample_rate_hz=np.asarray(100),
                source_session_id=np.asarray("hmog_u000_s01"),
            )
            sliced_imu = np.ascontiguousarray(imu[2:12])
            sliced_t = np.ascontiguousarray(timestamp_ns[2:12])
            base = {
                "schema_version": "hmog_fiveshot_attack_material_v2",
                "action": "keystroke",
                "user_id": "hmog_u000",
                "imu_source_kind": "native_continuous_100hz_slice",
                "native_stream": str(stream),
                "native_stream_sha256": sha256_file(stream),
                "native_start_sample": 2,
                "native_end_sample": 12,
                "imu_sha256": sha256_array(sliced_imu),
                "imu_timestamp_ns_sha256": sha256_array(sliced_t),
                "key_down_ms": [30, 70],
                "key_is_letter": [1, 0],
            }
            source = pulse_source_from_material(
                {**base, "event_id": "event-0", "shot_ordinal": 0}
            )
            np.testing.assert_array_equal(source.imu, sliced_imu)
            np.testing.assert_allclose(source.imu_t_ms, sliced_t / 1_000_000.0)

            manifest = root / "material.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(
                        {**base, "event_id": f"event-{index}", "shot_ordinal": index}
                    )
                    + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            sources = load_user_pulse_sources(
                manifest, user_id="hmog_u000", verify_hashes=True
            )
            self.assertEqual(len(sources), 5)


if __name__ == "__main__":
    unittest.main()
