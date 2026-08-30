from __future__ import annotations

import unittest

from pipeline.genuine_touch_recovery import (
    native_event_slot,
    native_source_cluster_id,
)


class GenuineTouchRecoveryTest(unittest.TestCase):
    def test_source_cluster_identity_matches_existing_direct100k_release(self) -> None:
        cluster = native_source_cluster_id(
            user_id="hmog_u086",
            session_id="hmog_u086_s18",
            source_event_id="87001800000063",
            source_event_ordinal=100,
            start_sample=15247,
            end_sample=15254,
        )
        self.assertEqual(
            cluster,
            "hmog-genuine-event-9db9e47f7e826b9fc6968187",
        )

    def test_native_event_slot_uses_existing_half_open_100hz_contract(self) -> None:
        self.assertEqual(
            native_event_slot(
                {
                    "relative_start_ns": 15_247 * 10_000_000,
                    "relative_end_ns": 15_254 * 10_000_000,
                },
                20_000,
            ),
            (15_247, 15_254),
        )


if __name__ == "__main__":
    unittest.main()
