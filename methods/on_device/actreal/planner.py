"""Turning an agent's semantic action into a playable one.

The agent says "Tap (540, 1200)" or "Swipe (540,1500) to (540,700)".  What has
to come back is a complete action -- a donor gesture re-aimed at those
coordinates, carrying the inertia that was generated with it.

Two decisions live here and both are recorded rather than implied:

* **scroll or swipe.** Mobile-Agent-E has one gesture for both.  The offline
  data does not: they are separate actions with separate checkpoints and
  separate duration behaviour, so the request has to be routed, and the routing
  rule is written down and logged with every action.
* **reachability.** A coordinate outside the mapped rectangle has no source
  coordinate.  Aiming there anyway would put the action somewhere the
  detectors were never fitted, so it is reported, not silently clamped.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .bundle import ActionBundle, from_json_dict, load_bundle
from .duration_law import TimingBook
from .mapping import ScreenMapping

# A vertical drag of at least this fraction of the screen is content scrolling;
# anything shorter is a swipe.  Both numbers are on the source screen, and the
# choice is logged per action so a later analysis can revisit it.
SCROLL_MIN_TRAVEL_FRACTION = 0.35


@dataclass
class PlanNote:
    """Anything about a plan that a later reader would want to know."""

    kind: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class Plan:
    bundle: ActionBundle
    requested_action: str
    resolved_action: str
    start: tuple[float, float]
    end: Optional[tuple[float, float]]
    notes: list[PlanNote] = field(default_factory=list)

    @property
    def reachable(self) -> bool:
        return not any(n.kind == "unreachable" for n in self.notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action,
            "resolved_action": self.resolved_action,
            "start": list(self.start),
            "end": list(self.end) if self.end else None,
            "bundle": self.bundle.summary(),
            "notes": [n.as_dict() for n in self.notes],
        }


BUNDLE_SCHEMA = "actreal_action_bundle_v1"
TIMING_FILENAME = "timing_profile.json"


class BundleLibrary:
    """The playable actions shipped with the local package, and their timing.

    The directory also carries sidecars -- an index, the duration laws -- so
    files are recognised by the schema they declare rather than by being the
    only thing with a .json extension.
    """

    def __init__(self, directory: "str | Path", seed: int = 0, victim: Optional[str] = None):
        self.directory = Path(directory)
        self._by_action: dict[str, list[ActionBundle]] = {}
        self._rng = random.Random(seed)
        self.timing: Optional[TimingBook] = None
        self.victim = victim

        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} is not readable JSON: {error}") from error
            if not isinstance(data, dict):
                continue
            if data.get("schema_version") != BUNDLE_SCHEMA:
                continue
            bundle = from_json_dict(data)
            self._by_action.setdefault(bundle.action, []).append(bundle)
        if not self._by_action:
            raise FileNotFoundError(f"no bundles in {self.directory}")

        timing_path = self.directory / TIMING_FILENAME
        if timing_path.is_file():
            self.timing = self._load_timing(timing_path, victim)

    @staticmethod
    def _load_timing(path: Path, victim: Optional[str]) -> Optional[TimingBook]:
        data = json.loads(path.read_text())
        actions = data.get("actions", {})
        if not actions:
            return None
        stored = data.get("victim")
        if victim is not None and stored and victim != stored:
            raise KeyError(
                f"{path.name} carries {stored!r}, not {victim!r}; one person per package"
            )
        return TimingBook.from_dict(actions)

    @property
    def actions(self) -> list[str]:
        return sorted(self._by_action)

    def count(self, action: str) -> int:
        return len(self._by_action.get(action, []))

    def pick(
        self,
        action: str,
        *,
        avoid: Optional[str] = None,
        duration_ms: Optional[float] = None,
        tolerance: float = 0.15,
        tolerance_floor_ms: float = 45.0,
    ) -> ActionBundle:
        """Choose a donor, preferring not to repeat the previous one.

        Repetition matters more than it looks: the same window delivered twice
        in one session is a duplicate a session-level detector can find without
        knowing anything about gestures.

        With ``duration_ms`` the choice is restricted to bundles whose contact
        lasts about that long, because the inertia in a bundle was generated
        for its own duration -- taking a window built for a 200 ms scroll and
        playing it under an 800 ms one would put somebody else's motion under
        the gesture.
        """

        pool = self._by_action.get(action)
        if not pool:
            raise KeyError(f"no {action!r} bundles; have {self.actions}")

        if duration_ms is not None and duration_ms > 0:
            # A purely relative tolerance is tighter than the grid spacing at
            # short durations: a 50 ms tap has no 15% neighbour on a 20 ms grid.
            allowed = max(tolerance * duration_ms, tolerance_floor_ms)
            near = [b for b in pool if abs(b.gesture_ms - duration_ms) <= allowed]
            if not near:
                # Nothing within tolerance: take the closest rather than a
                # random one, and let the caller record the gap.
                closest = min(pool, key=lambda b: abs(b.gesture_ms - duration_ms))
                near = [
                    b for b in pool if abs(b.gesture_ms - closest.gesture_ms) < 1e-6
                ]
            pool = near

        if len(pool) > 1 and avoid is not None:
            pool = [b for b in pool if b.bundle_id != avoid] or pool
        return pool[self._rng.randrange(len(pool))]


class ActionPlanner:
    def __init__(
        self,
        library: BundleLibrary,
        mapping: ScreenMapping,
        *,
        scroll_min_travel_fraction: float = SCROLL_MIN_TRAVEL_FRACTION,
    ):
        self.library = library
        self.mapping = mapping
        self.scroll_min_travel_fraction = scroll_min_travel_fraction
        self._last_by_action: dict[str, str] = {}
        # Departures are cycled rather than drawn at random so a session uses
        # all of the victim's spread instead of landing on one of them twice.
        self._duration_draw = 0

    def resolve_gesture(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[str, PlanNote]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        travel = (dx * dx + dy * dy) ** 0.5
        source_height = self.mapping.source_h * self.mapping.scale
        fraction = travel / source_height if source_height else 0.0
        vertical = abs(dy) >= abs(dx)
        if vertical and fraction >= self.scroll_min_travel_fraction:
            action = "scroll"
        else:
            action = "swipe"
        note = PlanNote(
            "gesture_routing",
            f"travel {travel:.0f}px = {fraction:.2f} of screen, "
            f"{'vertical' if vertical else 'horizontal'} -> {action}",
        )
        return action, note

    def _duration_for(
        self,
        action: str,
        start: tuple[float, float],
        end: Optional[tuple[float, float]],
        notes: list[PlanNote],
    ) -> Optional[float]:
        """How long this action should last: one of the victim's own timings.

        Not derived from the distance.  Across the recorded population travel
        and duration are uncorrelated (scroll r = -0.077 over 59,937 events,
        -0.107 within a single user at the median): people cover more ground by
        moving faster, and the duration stays where their hand puts it.  So the
        duration is drawn whole from this victim's recordings and the requested
        geometry decides the speed.
        """

        if self.library.timing is None:
            return None
        timing = self.library.timing.get(action)
        if timing is None:
            return None

        draw = self._duration_draw
        self._duration_draw += 1
        duration = timing.duration_ms(draw)

        detail = f"drew {duration:.0f}ms from the victim's {timing.count} recorded {action}s"
        if end is not None:
            source_start = self.mapping.to_source(start)
            source_end = self.mapping.to_source(end)
            travel = (
                (source_end[0] - source_start[0]) ** 2
                + (source_end[1] - source_start[1]) ** 2
            ) ** 0.5
            detail += f"; {travel:.0f}px over {duration:.0f}ms = {travel/duration:.2f} px/ms"
        notes.append(PlanNote("duration_drawn", detail))

        if not timing.fits_window():
            notes.append(
                PlanNote(
                    "duration_unfit",
                    f"the victim's {action}s run up to {timing.span_ms[1]:.0f}ms, "
                    f"past what the window can hold",
                )
            )
        return duration

    def plan(
        self,
        action: str,
        start: tuple[float, float],
        end: Optional[tuple[float, float]] = None,
        *,
        duration_ms: Optional[float] = None,
    ) -> Plan:
        """Compile one requested action into a playable bundle.

        ``duration_ms`` overrides the draw from the victim's own timings.  It
        exists for typing, where how long the action lasts is not a property of
        the hand alone: the framework enters text one character at a time, so
        the interval the inertia has to cover is set by how much text there is.
        Drawing a duration blindly there would put a two-second typing window
        under a four-character word.
        """

        notes: list[PlanNote] = []
        requested = action

        if action == "gesture":
            if end is None:
                raise ValueError("a gesture needs an end point")
            action, note = self.resolve_gesture(start, end)
            notes.append(note)

        points = [start] + ([end] if end else [])
        for label, point in zip(("start", "end"), points):
            on_screen = (
                0.0 <= float(point[0]) <= self.mapping.device_w
                and 0.0 <= float(point[1]) <= self.mapping.device_h
            )
            if not on_screen:
                clamped, moved = self.mapping.clamp_device_point(point)
                notes.append(
                    PlanNote(
                        "unreachable",
                        f"{label} {point} is off the physical screen "
                        f"({self.mapping.device_w:.0f}x{self.mapping.device_h:.0f}); "
                        f"nearest point on it is {clamped} ({moved:.0f}px away)",
                    )
                )
            elif not self.mapping.contains_device_point(point):
                # Outside the letterbox, but on the glass.  This used to be
                # refused, on the assumption -- written into usable_rect's own
                # docstring -- that the target app keeps its controls inside the
                # rectangle that maps back to the donors' 1080x1920 screen. The
                # app does not: the agent found controls at y=223 and y=2282 on
                # a 2424-tall display, in the bands the letterbox excludes, and
                # 24 of 117 actions were refused there.
                #
                # Refusing costs more than serving. A refused action falls back
                # to `adb shell input`, which delivers one synthetic sample with
                # no size, no pressure, no inertia and no correlation to the IMU
                # window -- conspicuous in every dimension. Serving it costs
                # only that the *position* lies outside the donors' screen
                # range, and the human baseline this is measured against was
                # recorded on this same 2424-tall phone, where people tap there
                # too.
                notes.append(
                    PlanNote(
                        "outside_donor_screen",
                        f"{label} {point} is on the glass but outside the "
                        f"{self.mapping.source_w:.0f}x{self.mapping.source_h:.0f} "
                        f"region the donors were recorded in; served anyway, and "
                        f"its position has no counterpart in the donor corpus",
                    )
                )

        if action not in self.library.actions:
            raise KeyError(f"no bundles for {action!r}; have {self.library.actions}")

        if duration_ms is not None and duration_ms > 0:
            target_ms = float(duration_ms)
            notes.append(
                PlanNote("duration_requested", f"caller asked for {target_ms:.0f}ms")
            )
        else:
            target_ms = self._duration_for(action, start, end, notes)
        donor = self.library.pick(
            action,
            avoid=self._last_by_action.get(action),
            duration_ms=target_ms,
        )
        self._last_by_action[action] = donor.bundle_id
        if target_ms is not None:
            gap = abs(donor.gesture_ms - target_ms) / target_ms
            if abs(donor.gesture_ms - target_ms) > max(0.15 * target_ms, 45.0):
                notes.append(
                    PlanNote(
                        "duration_gap",
                        f"drew {target_ms:.0f}ms, nearest bundle lasts "
                        f"{donor.gesture_ms:.0f}ms ({gap:.0%} away)",
                    )
                )
        bundle = donor.reanchored(start, end)

        # Re-aiming can push part of a gesture off screen even when both
        # endpoints are fine, because the donor's shape bulges around the chord.
        off = [
            p
            for p in bundle.touch.points
            if not (0 <= p.x <= self.mapping.device_w and 0 <= p.y <= self.mapping.device_h)
        ]
        if off:
            notes.append(
                PlanNote("off_screen", f"{len(off)} of {len(bundle.touch.points)} points off screen")
            )

        return Plan(
            bundle=bundle,
            requested_action=requested,
            resolved_action=action,
            start=start,
            end=end,
            notes=notes,
        )


def load_default_library(package_root: Optional[Path] = None) -> BundleLibrary:
    root = package_root or Path(__file__).resolve().parents[1]
    for candidate in (root / "bundles", root / "dist" / "bundles"):
        if candidate.is_dir():
            return BundleLibrary(candidate)
    raise FileNotFoundError(f"no bundle directory under {root}")
