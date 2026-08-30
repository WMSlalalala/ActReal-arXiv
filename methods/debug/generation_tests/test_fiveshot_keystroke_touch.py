from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import unittest

import numpy as np

from pipeline.fiveshot_keystroke_touch import (
    KEYSTROKE_PRESS_SLOP_PX,
    FiveShotKeystrokeTouchError,
    KeystrokeRhythm,
    compose_keystroke_touch,
    rhythm_from_material,
)

SCREEN = (1080.0, 1920.0)
# Synthetic portrait keyboard anchors used only by this unit test.
KEYS = {
    "e": (283.0, 100.0),
    "t": (491.0, 104.0),
    "a": (108.0, 245.0),
    "h": (653.0, 253.0),
    "space": (576.0, 545.0),
    "n": (751.0, 401.0),
}


@dataclass(frozen=True)
class _Shot:
    row: dict[str, Any]
    event_id: str = "shot-0"


def _rhythm(holds=(70, 80, 65, 90), flights=(450, 380, 520)) -> KeystrokeRhythm:
    return KeystrokeRhythm(
        holds_ms=tuple(holds), flights_ms=tuple(flights), source_event_ids=("e0",)
    )


def _segments(contact: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(contact) > 0.5
    spans: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


class ComposeKeystrokeTouchTest(unittest.TestCase):
    # Four keys support [4*53+3*66, 4*140+3*1130] = [410, 3950] ms, so the
    # default window sits inside that band; the scope filter keeps carriers
    # outside their own band out of the evaluation.
    def _compose(self, keys=("e", "t", "a", "h"), samples=300, duration=3000.0, **kw):
        return compose_keystroke_touch(
            anchors_px=[KEYS[name] for name in keys],
            rhythm=kw.pop("rhythm", _rhythm()),
            target_samples=samples,
            target_duration_ms=duration,
            screen_width_px=SCREEN[0],
            screen_height_px=SCREEN[1],
            **kw,
        )

    def test_one_contact_per_requested_key(self) -> None:
        composed = self._compose(keys=("e", "t", "a", "h", "n"), duration=3600.0)
        self.assertEqual(composed.contact_segments, 5)
        self.assertEqual(len(_segments(composed.trajectory[:, 0])), 5)

    def test_each_contact_sits_on_its_requested_key(self) -> None:
        keys = ("e", "t", "a", "h")
        composed = self._compose(keys=keys)
        spans = _segments(composed.trajectory[:, 0])
        self.assertEqual(len(spans), len(keys))
        for (start, stop), name in zip(spans, keys):
            with self.subTest(key=name):
                x = composed.trajectory[start:stop, 1] * SCREEN[0]
                y = composed.trajectory[start:stop, 2] * SCREEN[1]
                np.testing.assert_allclose(x, KEYS[name][0], atol=0.05)
                np.testing.assert_allclose(y, KEYS[name][1], atol=0.05)

    def test_a_contact_holds_still_when_the_victim_recorded_no_travel(
        self,
    ) -> None:
        """With no recorded excursion there is nothing to reproduce."""

        composed = self._compose()
        for start, stop in _segments(composed.trajectory[:, 0]):
            with self.subTest(span=(start, stop)):
                self.assertEqual(len(set(composed.trajectory[start:stop, 1])), 1)
                self.assertEqual(len(set(composed.trajectory[start:stop, 2])), 1)

    def test_a_contact_carries_the_victim_own_travel(self) -> None:
        """A finger on a key is not a fixed point, and the material says so."""

        path = np.zeros((6, 2), dtype=np.float64)
        path[:, 0] = np.asarray([0.0, 0.0, 0.001, 0.001, 0.002, 0.002])
        rhythm = _rhythm()
        rhythm = type(rhythm)(
            holds_ms=rhythm.holds_ms, flights_ms=rhythm.flights_ms,
            source_event_ids=rhythm.source_event_ids, press_paths=(path,),
        )
        composed = self._compose(rhythm=rhythm, seed=7)
        moved = 0
        for start, stop in _segments(composed.trajectory[:, 0]):
            if stop - start >= 2 and len(set(composed.trajectory[start:stop, 1])) > 1:
                moved += 1
        self.assertGreater(moved, 0)

    def test_travel_never_leaves_the_touch_slop(self) -> None:
        """A press that travels past the slop stops being a press."""

        path = np.zeros((6, 2), dtype=np.float64)
        path[:, 0] = np.linspace(0.0, 0.5, 6)          # half the screen wide
        rhythm = _rhythm()
        rhythm = type(rhythm)(
            holds_ms=rhythm.holds_ms, flights_ms=rhythm.flights_ms,
            source_event_ids=rhythm.source_event_ids, press_paths=(path,),
        )
        composed = self._compose(rhythm=rhythm, seed=3)
        for start, stop in _segments(composed.trajectory[:, 0]):
            if stop - start < 2:
                continue
            span = composed.trajectory[start:stop, 1:3].astype(np.float64) * SCREEN
            reach = float(np.max(np.linalg.norm(span - span[0], axis=1)))
            with self.subTest(span=(start, stop)):
                self.assertLessEqual(reach, KEYSTROKE_PRESS_SLOP_PX + 1e-6)

    def test_the_victim_holds_are_kept_when_flights_can_absorb_the_window(
        self,
    ) -> None:
        rhythm = _rhythm(holds=(70, 80, 65, 90))
        composed = self._compose(rhythm=rhythm, duration=3000.0)
        np.testing.assert_array_equal(composed.holds_ms, np.asarray([70, 80, 65, 90]))

    def test_the_allocation_fills_the_carrier_window_exactly(self) -> None:
        composed = self._compose(duration=3000.0)
        total = int(composed.holds_ms.sum() + composed.flights_ms.sum())
        self.assertEqual(total, 3000)

    def test_a_fractional_window_still_ends_with_the_finger_up(self) -> None:
        """Every genuine typing window closes on the lift that ended it."""

        for duration in (2999.4, 2999.5, 2999.6, 3000.0, 3000.5, 3000.9):
            with self.subTest(duration=duration):
                composed = self._compose(duration=duration)
                contact = composed.trajectory[:, 0]
                self.assertEqual(float(contact[0]), 1.0)
                self.assertEqual(float(contact[-1]), 0.0)
                total = int(
                    composed.holds_ms.sum() + composed.flights_ms.sum()
                )
                self.assertLessEqual(float(total), duration)

    def test_timing_stays_inside_the_declared_support(self) -> None:
        composed = self._compose(duration=3000.0)
        self.assertTrue(np.all(composed.holds_ms >= 53))
        self.assertTrue(np.all(composed.holds_ms <= 140))
        self.assertTrue(np.all(composed.flights_ms >= 66))
        self.assertTrue(np.all(composed.flights_ms <= 1130))

    def test_pressure_and_availability_match_the_genuine_constants(self) -> None:
        composed = self._compose()
        contact = composed.trajectory[:, 0] > 0.5
        self.assertEqual(set(composed.trajectory[contact, 3].tolist()), {1.0})
        self.assertEqual(set(composed.trajectory[:, 8].tolist()), {1.0})
        self.assertEqual(set(composed.trajectory[contact, 4].tolist()), {1.0})
        self.assertEqual(set(composed.trajectory[~contact, 4].tolist()), {0.0})

    def test_a_request_longer_than_the_rhythm_reuses_it_in_order(self) -> None:
        rhythm = _rhythm(holds=(70, 80), flights=(450,))
        composed = self._compose(
            keys=("e", "t", "a", "h", "n"), rhythm=rhythm, duration=3600.0
        )
        self.assertEqual(composed.contact_segments, 5)
        np.testing.assert_array_equal(
            composed.holds_ms, np.asarray([70, 80, 70, 80, 70])
        )

    def test_a_window_the_rhythm_cannot_fill_is_refused(self) -> None:
        with self.assertRaises(FiveShotKeystrokeTouchError):
            self._compose(keys=("e",), samples=64, duration=445.0)

    def test_an_off_screen_key_anchor_is_refused(self) -> None:
        with self.assertRaises(FiveShotKeystrokeTouchError):
            compose_keystroke_touch(
                anchors_px=[(283.0, 100.0), (1200.0, 104.0)],
                rhythm=_rhythm(),
                target_samples=300,
                target_duration_ms=3000.0,
                screen_width_px=SCREEN[0],
                screen_height_px=SCREEN[1],
            )

    def test_an_empty_rhythm_is_refused(self) -> None:
        with self.assertRaises(FiveShotKeystrokeTouchError):
            KeystrokeRhythm(holds_ms=(), flights_ms=(), source_event_ids=())

    def test_a_multi_key_request_without_a_flight_is_refused(self) -> None:
        rhythm = KeystrokeRhythm(
            holds_ms=(70,), flights_ms=(), source_event_ids=("e0",)
        )
        with self.assertRaises(FiveShotKeystrokeTouchError):
            self._compose(keys=("e", "t"), rhythm=rhythm)


class RhythmFromMaterialTest(unittest.TestCase):
    def test_holds_and_flights_come_from_the_frozen_schedule(self) -> None:
        shot = _Shot(
            row={
                "key_down_ms": [1000, 1200, 1500],
                "key_up_ms": [1070, 1280, 1560],
            }
        )
        rhythm = rhythm_from_material([shot])
        self.assertEqual(rhythm.holds_ms, (70, 80, 60))
        self.assertEqual(rhythm.flights_ms, (130, 220))

    def test_flights_do_not_span_two_events(self) -> None:
        shots = [
            _Shot(row={"key_down_ms": [0, 200], "key_up_ms": [70, 260]}, event_id="a"),
            _Shot(
                row={"key_down_ms": [9000, 9300], "key_up_ms": [9080, 9360]},
                event_id="b",
            ),
        ]
        rhythm = rhythm_from_material(shots)
        self.assertEqual(rhythm.holds_ms, (70, 60, 80, 60))
        self.assertEqual(rhythm.flights_ms, (130, 220))
        self.assertEqual(rhythm.source_event_ids, ("a", "b"))

    def test_material_without_a_schedule_is_refused(self) -> None:
        with self.assertRaises(FiveShotKeystrokeTouchError):
            rhythm_from_material([_Shot(row={})])

    def test_disagreeing_down_and_up_lengths_are_refused(self) -> None:
        with self.assertRaises(FiveShotKeystrokeTouchError):
            rhythm_from_material(
                [_Shot(row={"key_down_ms": [0, 200], "key_up_ms": [70]})]
            )


if __name__ == "__main__":
    unittest.main()
