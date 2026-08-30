from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from scripts.fit_hmog_conditional_touch_generator import (
    _materialize_training_rows,
    _plan_archive,
    main,
)
from pipeline.conditional_touch_generator import ConditionalTouchGenerator


class ConditionalTouchFitCliTests(unittest.TestCase):
    @staticmethod
    def _write_archive(path: Path, action: str) -> None:
        if action == "tap":
            x = np.asarray((100.0, 101.0, 100.0))
        else:
            x = np.asarray((100.0, 111.0, 120.0))
        np.savez_compressed(
            path,
            action_name=np.asarray(action),
            event_id=np.asarray((10,), dtype=np.int64),
            user_id=np.asarray((1,), dtype=np.int32),
            orientation_id=np.asarray((0,), dtype=np.int8),
            event_offsets=np.asarray((0, 3), dtype=np.int64),
            flat_t_rel_ms=np.asarray((0.0, 10.0, 20.0)),
            flat_x=x,
            flat_y=np.asarray((200.0, 202.0, 200.0)),
            flat_pressure=np.asarray((0.4, 0.6, 0.5)),
            flat_action_code=np.asarray((0, 2, 1), dtype=np.int64),
            flat_valid_mask=np.ones(3, dtype=np.int8),
        )

    def test_archive_plan_filters_whole_events_before_materializing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hmog_trajectory_tap.npz"
            offsets = np.asarray((0, 3, 6, 9, 12), dtype=np.int64)
            np.savez_compressed(
                source,
                action_name=np.asarray("tap"),
                event_id=np.asarray((10, 11, 12, 13), dtype=np.int64),
                user_id=np.asarray((1, 2, 1, 1), dtype=np.int32),
                orientation_id=np.asarray((0, 0, 2, 1), dtype=np.int8),
                event_offsets=offsets,
                flat_t_rel_ms=np.tile(
                    np.asarray((0.0, 10.0, 20.0)), 4
                ),
                flat_x=np.arange(12, dtype=np.float64),
                flat_y=np.arange(12, dtype=np.float64) + 20.0,
                flat_pressure=np.full(12, 0.5, dtype=np.float64),
                flat_action_code=np.tile(
                    np.asarray((0, 2, 1), dtype=np.int64), 4
                ),
                flat_valid_mask=np.asarray(
                    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
                    dtype=np.int8,
                ),
            )
            plan = _plan_archive(
                action="tap",
                path=source,
                train_users=(1,),
                allowed_ids={"10", "11", "12", "13"},
            )
            self.assertEqual(plan.selected_event_count, 1)
            self.assertEqual(plan.selected_row_count, 3)
            self.assertEqual(
                plan.audit["events_excluded"],
                {
                    "non_train_user": 1,
                    "not_accepted_by_optional_filter": 0,
                    "unsupported_orientation": 1,
                    "contains_invalid_raw_row": 1,
                },
            )
            rows = _materialize_training_rows((plan,))
            self.assertEqual(rows["event_id"].tolist(), [0, 0, 0])
            self.assertEqual(rows["action"].tolist(), ["tap", "tap", "tap"])
            self.assertEqual(rows["orientation_id"].tolist(), [0, 0, 0])
            self.assertEqual(rows["android_action"].tolist(), [0, 2, 1])

    def test_main_writes_compact_parameter_only_artifact_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            raw_root.mkdir()
            for action in ("tap", "scroll", "swipe"):
                self._write_archive(
                    raw_root / f"hmog_trajectory_{action}.npz", action
                )
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "train_users": [1],
                        "val_users": [2],
                        "test_users": [3],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "conditional_touch.npz"
            result = main(
                [
                    "--raw-root",
                    str(raw_root),
                    "--split-json",
                    str(split),
                    "--output-model",
                    str(output),
                    "--grid-size",
                    "5",
                    "--max-rank",
                    "0",
                    "--minimum-events",
                    "1",
                ]
            )
            self.assertTrue(output.is_file())
            self.assertLess(output.stat().st_size, 100_000)
            audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
            self.assertFalse(audit["artifact"]["stores_raw_rows_or_donors"])
            self.assertEqual(
                audit["training_summary"]["accepted_event_count"], 3
            )
            self.assertFalse(
                audit["training_summary"]["raw_event_ids_retained"]
            )
            self.assertEqual(
                audit["generation_verification"]["generated_events_checked"],
                30,
            )
            self.assertEqual(
                audit["generation_verification"]["exact_endpoint_failures"],
                0,
            )
            model = ConditionalTouchGenerator.load(output)
            generated = model.generate(
                action="scroll",
                orientation_id=0,
                start_xy_px=(10.0, 20.0),
                end_xy_px=(110.0, 20.0),
                direction="right",
                seed=7,
                duration_ms=20.0,
                sample_count=3,
                minimum_residual_scale=0.0,
            )
            self.assertEqual(generated.x_px[[0, -1]].tolist(), [10.0, 110.0])
            self.assertEqual(generated.y_px[[0, -1]].tolist(), [20.0, 20.0])


if __name__ == "__main__":
    unittest.main()
