"""How long an action lasts: the victim's centre, the training population's scatter.

The protocol is the point of most of these.  Seventy users train the model and
may be measured; twenty are held out and one of them is the victim, of whom
five recordings are held and nothing else.  A duration that read anything else
would be an attack with knowledge the threat model does not grant.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal.duration_law import (
    SPREAD_FRACTION,
    TRAIN_LOG_SPREAD,
    ActionTiming,
    DurationLawError,
    TimingBook,
    build_timing,
    window_cap_ms,
)

# One victim's five recorded scrolls, in milliseconds.
FIVE_SCROLLS = [180.0, 210.0, 260.0, 340.0, 460.0]


def _timing(action: str = "scroll", durations=None) -> ActionTiming:
    return build_timing(durations or FIVE_SCROLLS, action=action, victim="unit")


# -- what the material provides ----------------------------------------------


def test_a_timing_needs_a_usable_recording():
    with pytest.raises(DurationLawError, match="no usable duration"):
        build_timing([0.0, -5.0, float("nan")], action="scroll", victim="unit")


def test_unusable_recordings_are_dropped_not_repaired():
    timing = build_timing([210.0, 0.0, 610.0], action="scroll", victim="unit")
    assert timing.durations_ms == (210.0, 610.0)


def test_the_centre_is_the_victims_own_median():
    assert _timing().median_ms == 260.0


def test_draws_sit_around_the_victims_centre():
    timing = _timing()
    draws = [timing.duration_ms(i) for i in range(400)]
    assert statistics.median(draws) == pytest.approx(timing.median_ms, rel=0.12)


# -- where the scatter comes from --------------------------------------------


def test_the_scatter_is_the_training_populations_not_the_five_shots():
    """Five points measure a spread badly; the population measures it well.

    So the location is the victim's and the scale is the training users' --
    and never the held-out population's, which the attacker cannot see.
    """

    timing = _timing()
    assert timing.spread_source.startswith("train_population")
    assert timing.log_spread == pytest.approx(TRAIN_LOG_SPREAD["scroll"] * SPREAD_FRACTION)
    # The five shots' own scatter is reported but deliberately not used.
    assert timing.material_log_spread != pytest.approx(timing.log_spread)


def test_the_realised_spread_matches_what_was_asked_for():
    timing = _timing()
    logs = [math.log(timing.duration_ms(i)) for i in range(600)]
    assert statistics.stdev(logs) == pytest.approx(timing.log_spread, rel=0.15)


def test_the_spread_is_not_flat():
    """A pool with no spread is a line, and a tree splits a line in one cut."""

    timing = _timing()
    logs = [math.log(timing.duration_ms(i)) for i in range(200)]
    assert statistics.stdev(logs) > 0.1


def test_every_action_has_a_training_scale():
    for action in ("tap", "scroll", "swipe", "pinch", "keystroke"):
        assert TRAIN_LOG_SPREAD[action] > 0


# -- what the geometry decides -----------------------------------------------


def test_speed_is_what_the_geometry_decides_not_the_duration():
    """Distance sets speed, not duration.

    On the 70 training users, travel and duration correlate at -0.055 for
    scroll (39,094 events) and -0.086 within a user: people cover more ground
    by moving faster.
    """

    timing = _timing()
    slow = timing.speed_px_per_ms(300.0, draw=2)
    fast = timing.speed_px_per_ms(1200.0, draw=2)
    assert fast == pytest.approx(4 * slow)


def test_the_same_draw_label_always_gives_the_same_duration():
    timing = _timing()
    assert timing.duration_ms(7) == timing.duration_ms(7)
    assert timing.duration_ms("action-7") == timing.duration_ms("action-7")


def test_consecutive_draws_do_not_walk_a_pattern():
    timing = _timing()
    draws = [timing.duration_ms(i) for i in range(60)]
    steps = [b - a for a, b in zip(draws, draws[1:])]
    # A sequential generator would give a monotone or oscillating run; a hashed
    # one changes sign about half the time.
    ups = sum(1 for s in steps if s > 0)
    assert 0.3 < ups / len(steps) < 0.7


# -- the window --------------------------------------------------------------


def test_window_caps_match_the_generators_windows():
    assert window_cap_ms("tap") == 350.0
    assert window_cap_ms("scroll") == 1790.0
    assert window_cap_ms("swipe") == 1670.0
    assert window_cap_ms("nonsense") is None


def test_a_draw_never_exceeds_the_window_it_has_to_fit_in():
    timing = build_timing([300.0], action="tap", victim="unit")
    draws = [timing.duration_ms(i) for i in range(500)]
    assert max(draws) <= window_cap_ms("tap")
    assert min(draws) > 0


def test_a_timing_knows_when_its_material_outruns_its_window():
    assert _timing().fits_window()
    long_typing = build_timing([841.0, 3483.0], action="keystroke", victim="unit")
    assert not long_typing.fits_window()


# -- serialisation -----------------------------------------------------------


def test_a_timing_survives_the_trip_to_the_phone():
    timing = _timing()
    back = ActionTiming.from_dict(json.loads(json.dumps(timing.as_dict())))
    assert back.durations_ms == timing.durations_ms
    assert back.log_spread == pytest.approx(timing.log_spread)
    assert back.duration_ms(3) == pytest.approx(timing.duration_ms(3))


def test_a_book_answers_per_action_and_admits_what_it_lacks():
    book = TimingBook({"scroll": _timing()}, victim="unit")
    assert book.actions == ["scroll"]
    assert book.duration_ms("scroll", 1) > 0
    assert book.duration_ms("swipe", 1) is None


def test_a_book_lists_the_actions_it_cannot_serve():
    book = TimingBook(
        {
            "scroll": _timing(),
            "keystroke": build_timing([3483.0, 900.0], action="keystroke", victim="unit"),
        },
        victim="unit",
    )
    assert book.unfit_actions() == ["keystroke"]


def test_the_synthetic_profile_obeys_the_runtime_contract(synthetic_bundle_dir):
    profile = synthetic_bundle_dir / "timing_profile.json"
    data = json.loads(profile.read_text())
    assert data["schema_version"] == "actreal_timing_profile_v1"
    assert data["split"] == "synthetic_fixture"
    assert data["source"] == "deterministic_pytest_fixture"
    book = TimingBook.from_dict(data["actions"])
    assert len({t.victim for t in book.timings.values()}) == 1
    for action in ("tap", "scroll", "swipe"):
        timing = book.get(action)
        assert timing is not None and timing.count >= 2
        assert timing.fits_window(), f"{action} outruns its window"
        assert timing.spread_source.startswith("train_population")
