from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import torch

METHOD_ROOT = Path(__file__).resolve().parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from detection.core.event_detectors import (
    ACTION_WINDOW_SAMPLES,
    DEEP_INPUT_CHANNELS,
    _masked_window,
    action_window_samples,
    build_deep_detector,
)


class MaskedWindowTest(unittest.TestCase):
    def test_long_event_is_truncated_without_interpolation(self) -> None:
        values = torch.arange(6 * 2, dtype=torch.float32).reshape(1, 6, 2)
        windowed, valid = _masked_window(
            values, torch.ones((1, 6), dtype=torch.bool), 4
        )
        self.assertEqual(tuple(windowed.shape), (1, 4, 2))
        # Retained rows are the original rows, byte for byte.
        np.testing.assert_array_equal(
            windowed[0].numpy(), values[0, :4].numpy()
        )
        self.assertTrue(bool(valid.all()))

    def test_short_event_is_padded_and_masked(self) -> None:
        values = torch.arange(3 * 2, dtype=torch.float32).reshape(1, 3, 2)
        windowed, valid = _masked_window(
            values, torch.ones((1, 3), dtype=torch.bool), 6
        )
        np.testing.assert_array_equal(
            windowed[0, :3].numpy(), values[0].numpy()
        )
        np.testing.assert_array_equal(
            windowed[0, 3:].numpy(), np.zeros((3, 2), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            valid[0].numpy(), [True, True, True, False, False, False]
        )

    def test_input_mask_selects_real_rows_only(self) -> None:
        values = torch.zeros((1, 6, 2), dtype=torch.float32)
        values[0, :3] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        values[0, 3:] = 999.0  # padding garbage that must never be read
        windowed, valid = _masked_window(
            values,
            torch.tensor([[True, True, True, False, False, False]]),
            4,
        )
        np.testing.assert_array_equal(
            windowed[0, :3].numpy(), values[0, :3].numpy()
        )
        self.assertEqual(float(windowed[0, 3].abs().sum()), 0.0)
        np.testing.assert_array_equal(
            valid[0].numpy(), [True, True, True, False]
        )

    def test_event_shorter_than_two_samples_is_rejected(self) -> None:
        values = torch.zeros((1, 3, 2), dtype=torch.float32)
        with self.assertRaises(Exception):
            _masked_window(
                values,
                torch.tensor([[True, False, False]]),
                4,
            )


class ActionWindowTest(unittest.TestCase):
    def test_every_action_window_divides_the_attention_heads(self) -> None:
        self.assertEqual(
            set(ACTION_WINDOW_SAMPLES),
            {"tap", "scroll", "swipe", "pinch", "keystroke"},
        )
        for action, window in ACTION_WINDOW_SAMPLES.items():
            with self.subTest(action=action):
                self.assertEqual(window % 4, 0)
                self.assertGreaterEqual(window, 4)
                self.assertEqual(action_window_samples(action), window)

    def test_unknown_action_has_no_window(self) -> None:
        with self.assertRaises(Exception):
            action_window_samples("swype")


class DeepDetectorWindowTest(unittest.TestCase):
    def test_models_accept_all_modalities_and_use_the_action_window(
        self,
    ) -> None:
        for detector in ("authconformer", "behaveformer_stdat"):
            for modality, channels in DEEP_INPUT_CHANNELS.items():
                for action in ("tap", "keystroke"):
                    with self.subTest(
                        detector=detector, modality=modality, action=action
                    ):
                        model = build_deep_detector(
                            detector, modality, action
                        ).eval()
                        self.assertEqual(
                            model.target_samples,
                            ACTION_WINDOW_SAMPLES[action],
                        )
                        output = model(
                            torch.zeros(
                                (2, 8, channels), dtype=torch.float32
                            ),
                            torch.ones((2, 8), dtype=torch.bool),
                        )
                        self.assertEqual(tuple(output.shape), (2,))
                        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_pure_imu_modality_has_six_channels(self) -> None:
        self.assertEqual(DEEP_INPUT_CHANNELS["imu_only"], 6)

    def test_padding_does_not_change_the_score(self) -> None:
        torch.manual_seed(0)
        model = build_deep_detector(
            "authconformer", "imu_only", "tap"
        ).eval()
        real = torch.randn((1, 5, 6), dtype=torch.float32)
        padded = torch.cat(
            [real, torch.full((1, 4, 6), 7.0, dtype=torch.float32)], dim=1
        )
        with torch.no_grad():
            tight = model(real, torch.ones((1, 5), dtype=torch.bool))
            loose = model(
                padded,
                torch.tensor([[True] * 5 + [False] * 4]),
            )
        np.testing.assert_allclose(
            tight.numpy(), loose.numpy(), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
