from __future__ import annotations

import unittest

import numpy as np

from pipeline.android_touch_observation import (
    TouchObservation,
    trajectory_from_touch,
)
from pipeline.replay_leakage_smoke import (
    FEATURE_NAMES,
    compare_feature_groups,
    empirical_ks_distance,
    extract_obvious_shortcut_features,
    orientation_free_univariate_auc,
)


class ReplayLeakageSmokeTests(unittest.TestCase):
    def test_extract_features_measures_held_coordinates_and_deltas(self) -> None:
        touch = np.zeros((5, 7), dtype=np.float32)
        touch[:, 0] = 1.0
        touch[:, 1] = np.asarray([0.1, 0.1, 0.2, 0.2, 0.3])
        touch[:, 2] = 0.4
        touch[:, 3] = 0.5
        touch[:, 4] = 1.0
        touch[2, 5] = 0.1
        touch[4, 5] = 0.1
        observation = TouchObservation(
            touch=touch,
            trajectory=trajectory_from_touch(
                touch, touch_observed=True, duration_ms=40.0
            ),
            touch_observed=True,
            source_updates=3,
        )
        features = extract_obvious_shortcut_features(observation)
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertAlmostEqual(features["active_xy_repeat_rate"], 0.5)
        self.assertAlmostEqual(features["active_dxdy_nonzero_fraction"], 0.5)
        self.assertAlmostEqual(features["active_xy_unique_ratio"], 0.6)
        self.assertAlmostEqual(features["duration_seconds"], 0.04)
        self.assertAlmostEqual(features["source_update_rate"], 0.6)
        self.assertAlmostEqual(features["active_x_start"], 0.1)
        self.assertAlmostEqual(features["active_x_end"], 0.3)
        self.assertAlmostEqual(features["active_x_mean"], 0.18)
        self.assertAlmostEqual(features["active_bbox_width"], 0.2)
        self.assertAlmostEqual(features["active_bbox_height"], 0.0)
        self.assertAlmostEqual(features["within_contact_net_displacement"], 0.2)
        self.assertAlmostEqual(features["within_contact_path_length"], 0.2)
        self.assertAlmostEqual(features["within_contact_straightness"], 1.0)
        self.assertAlmostEqual(features["active_pressure_mean"], 0.5)
        self.assertAlmostEqual(features["active_pressure_std"], 0.0)

    def test_shape_features_do_not_bridge_separate_key_contacts(self) -> None:
        touch = np.zeros((5, 7), dtype=np.float32)
        touch[:, 0] = np.asarray([1, 1, 0, 1, 1])
        touch[:, 1] = np.asarray([0.1, 0.2, 0.0, 0.8, 0.9])
        touch[:, 2] = 0.4
        touch[:, 3] = np.asarray([0.2, 0.4, 0.0, 0.6, 0.8])
        touch[:, 4] = touch[:, 0]
        observation = TouchObservation(
            touch=touch,
            trajectory=trajectory_from_touch(
                touch, touch_observed=True, duration_ms=40.0
            ),
            touch_observed=True,
            source_updates=5,
        )
        features = extract_obvious_shortcut_features(observation)
        # The 0.2 -> 0.8 flight is not a contact path segment.
        self.assertAlmostEqual(features["within_contact_path_length"], 0.2)
        self.assertAlmostEqual(features["within_contact_net_displacement"], 0.2)
        self.assertAlmostEqual(features["within_contact_straightness"], 1.0)
        # Layout summaries still describe both touched keys.
        self.assertAlmostEqual(features["active_x_start"], 0.1)
        self.assertAlmostEqual(features["active_x_end"], 0.9)
        self.assertAlmostEqual(features["active_bbox_width"], 0.8)
        self.assertAlmostEqual(features["active_pressure_mean"], 0.5)

    def test_distribution_metrics_handle_ties_and_complete_separation(self) -> None:
        self.assertAlmostEqual(empirical_ks_distance([0, 0], [0, 0]), 0.0)
        self.assertAlmostEqual(empirical_ks_distance([0, 0], [1, 1]), 1.0)
        self.assertAlmostEqual(
            orientation_free_univariate_auc([0, 0], [0, 0]), 0.5
        )
        self.assertAlmostEqual(
            orientation_free_univariate_auc([0, 0], [1, 1]), 1.0
        )

    def test_compare_calls_result_non_formal_diagnostic(self) -> None:
        baseline = {name: 0.25 for name in FEATURE_NAMES}
        equal = compare_feature_groups([baseline] * 4, [baseline] * 4)
        self.assertEqual(
            equal["diagnostic_status"], "no_obvious_shortcut_detected"
        )
        changed = dict(baseline)
        changed["contact_fraction"] = 1.0
        separated = compare_feature_groups([baseline] * 4, [changed] * 4)
        self.assertEqual(
            separated["diagnostic_status"], "obvious_shortcut_candidates_found"
        )
        self.assertIn("contact_fraction", separated["flagged_features"])


if __name__ == "__main__":
    unittest.main()
