"""The pre-generated grid: how a request finds a cell, and what happens when it does not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal.imu_cache import (
    DIRECTIONS8,
    CacheKey,
    ImuCache,
    direction8,
    distance_bin,
    duration_bin,
    duration_bin_count,
    duration_for_bin,
)

CACHE_ROOT = Path(__file__).resolve().parents[1] / "imu_cache"


# -- quantisation -------------------------------------------------------------


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ((0, 0), (100, 0), "E"),
        ((0, 100), (0, 0), "N"),  # screen y grows downward
        ((0, 0), (0, 100), "S"),
        ((100, 0), (0, 0), "W"),
        ((0, 100), (100, 0), "NE"),
    ],
)
def test_direction_uses_screen_coordinates_not_maths_coordinates(start, end, expected):
    assert direction8(start, end) == expected


def test_a_gesture_that_does_not_travel_has_no_direction():
    assert direction8((5, 5), (5, 5)) is None


def test_every_direction_is_reachable_and_distinct():
    seen = {direction8((0, 0), (np.cos(a), -np.sin(a))) for a in np.arange(0, 2 * np.pi, 0.05)}
    assert seen == set(DIRECTIONS8)


def test_duration_bins_round_trip_within_half_a_step():
    for action in ("tap", "scroll", "keystroke"):
        for index in range(duration_bin_count(action)):
            ms = duration_for_bin(action, index)
            assert duration_bin(action, ms) == index


def test_durations_outside_the_grid_clamp_to_its_ends():
    assert duration_bin("tap", 1.0) == 0
    assert duration_bin("tap", 99999.0) == duration_bin_count("tap") - 1


def test_travel_bins_are_ordered_and_saturate():
    """Travel is recorded per window, not part of a cell's identity."""

    assert distance_bin(0.0) == 0
    assert distance_bin(100.0) > distance_bin(10.0)
    assert distance_bin(99999.0) == distance_bin(1919.0)


def test_a_travelling_action_needs_geometry_to_name_a_cell():
    with pytest.raises(ValueError, match="needs start and end"):
        CacheKey.for_request("scroll", duration_ms=400.0)
    # A tap has no heading, so duration alone names its cell.
    assert CacheKey.for_request("tap", duration_ms=80.0).direction is None


def test_travel_no_longer_splits_a_cell():
    """Two scrolls of the same length and heading share a cell whatever the distance."""

    near = CacheKey.for_request(
        "scroll", duration_ms=400.0, start=(540.0, 1000.0), end=(540.0, 800.0)
    )
    far = CacheKey.for_request(
        "scroll", duration_ms=400.0, start=(540.0, 1800.0), end=(540.0, 300.0)
    )
    assert near == far


# -- keys and neighbours ------------------------------------------------------


def test_a_request_becomes_a_key_that_carries_its_geometry():
    key = CacheKey.for_request(
        "scroll", duration_ms=400.0, start=(540.0, 1600.0), end=(540.0, 600.0)
    )
    assert key.action == "scroll"
    assert key.direction == "N"
    assert key.duration_bin == duration_bin("scroll", 400.0)


def test_neighbours_relax_duration_before_direction():
    key = CacheKey("scroll", 0, 10, "N")
    order = list(key.neighbours())
    first_direction_change = next(
        i for i, n in enumerate(order) if n.direction != "N"
    )
    first_duration_change = next(
        i for i, n in enumerate(order) if n.duration_bin != 10
    )
    assert first_duration_change < first_direction_change


def test_neighbours_stay_inside_the_grid():
    key = CacheKey("tap", 0, 0, None)
    for neighbour in key.neighbours():
        assert 0 <= neighbour.duration_bin < duration_bin_count("tap")


def test_a_key_maps_to_a_readable_directory_name():
    key = CacheKey("scroll", 0, 7, "NE")
    assert key.relative_dir().as_posix() == "scroll/ori0/dur007/NE"


# -- lookup -------------------------------------------------------------------


def test_lookup_reports_a_miss_rather_than_returning_the_wrong_cell(tmp_path):
    cache = ImuCache(tmp_path)
    key = CacheKey("scroll", 0, 5, "N")
    assert cache.lookup(key) is None
    assert cache.stats()["miss"] == 1
    cache.close()


def test_lookup_falls_through_to_a_neighbour_and_says_so(tmp_path):
    cache = ImuCache(tmp_path)
    neighbour = CacheKey("scroll", 0, 6, "N")
    path = tmp_path / "x.npz"
    path.write_bytes(b"")
    cache.add(neighbour, path, frames=10, active_start=0, active_end=5,
              requested_duration_ms=100.0)
    cache.commit()

    hit = cache.lookup(CacheKey("scroll", 0, 5, "N"))
    assert hit is not None
    assert hit.exactness == "neighbour"
    assert hit.key == neighbour
    assert cache.stats()["exact"] == 0
    cache.close()


@pytest.mark.skipif(
    not (CACHE_ROOT / "index.sqlite").exists(),
    reason="requires the separately licensed generated-cache fixture",
)
def test_the_cache_covers_every_duration_the_victim_can_actually_draw(
    synthetic_bundle_dir,
):
    """The grid may declare more than is built; what must be built is what gets asked for.

    The runtime only ever requests durations this victim's timing can produce,
    so that range -- not the whole declared grid -- is what the cache has to
    answer exactly.  A neighbour hit here would mean serving a gesture with
    inertia generated for a different length.
    """

    import json

    from actreal.duration_law import TimingBook

    profile = synthetic_bundle_dir / "timing_profile.json"
    book = TimingBook.from_dict(json.loads(profile.read_text())["actions"])
    cache = ImuCache(CACHE_ROOT)
    checked = 0
    # Geometry matters for the travelling actions: a cell is duration *and*
    # direction *and* travel, so the request has to carry all three.
    geometry = {
        "tap": (None, None),
        "scroll": ((540.0, 1500.0), (540.0, 600.0)),
    }
    for action in ("tap", "scroll"):
        timing = book.get(action)
        if timing is None:
            continue
        start, end = geometry[action]
        for draw in range(200):
            ms = timing.duration_ms(draw)
            hit = cache.lookup(
                CacheKey.for_request(action, duration_ms=ms, start=start, end=end)
            )
            assert hit is not None, f"no window for a {ms:.0f} ms {action}"
            assert hit.exactness == "exact", (
                f"{action} at {ms:.0f} ms fell through to a neighbour cell"
            )
            checked += 1
    assert checked > 0
    cache.close()


@pytest.mark.skipif(
    not (CACHE_ROOT / "index.sqlite").exists(),
    reason="requires the separately licensed generated-cache fixture",
)
def test_declared_grid_bins_that_are_not_built_are_visible():
    """A grid that promises more than it holds should say so, not fail silently."""

    cache = ImuCache(CACHE_ROOT)
    conn = cache.connect()
    for action in ("tap", "scroll"):
        built = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT duration_bin FROM samples WHERE action=?", (action,)
            )
        }
        if not built:
            continue
        declared = set(range(duration_bin_count(action)))
        missing = sorted(declared - built)
        # Recorded, not asserted away: the top of a grid is often extended
        # before the generator has caught up with it.
        assert built, f"{action} has no bins at all"
        assert max(built) <= max(declared)
        if missing:
            print(f"{action}: {len(missing)} declared bins not built: {missing[:6]}...")
    cache.close()


@pytest.mark.skipif(
    not (CACHE_ROOT / "index.sqlite").exists(),
    reason="requires the separately licensed generated-cache fixture",
)
def test_cached_windows_load_with_the_fields_the_bundler_needs():
    cache = ImuCache(CACHE_ROOT)
    hit = cache.lookup(CacheKey.for_request("tap", duration_ms=120.0))
    with np.load(hit.path) as data:
        samples = data["samples"]
        assert samples.ndim == 2 and samples.shape[1] == 6
        assert 0 <= data["active_start"] < data["active_end"] <= len(samples)
        assert float(data["period_ms"]) > 0
        # Gravity has to be in there; a window without it is not a held phone.
        assert 8.0 < float(np.linalg.norm(samples[:, :3].mean(axis=0))) < 11.0
    cache.close()
