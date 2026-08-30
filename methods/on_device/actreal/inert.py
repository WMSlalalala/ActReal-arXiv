"""Where a stray contact changes nothing.

The idle filler has to touch the screen -- a recording in which the hand goes
completely still for twenty seconds between actions says "agent" more loudly
than any single gesture could.  But it must touch without *doing* anything: the
agent chose its next action from a screenshot taken before the gap, and a
filler that opened a menu, submitted a form, or navigated away would not merely
add noise, it would destroy the run.

The earlier filler solved this by never tapping at all and scrolling instead,
always between the same two points in the middle of the screen.  That is safe
in the sense that nothing is activated, but it buys the safety with a signature:
every filler gesture identical to every other, and on one session fifty-seven of
them against seventeen real actions.  A corpus three-quarters composed of one
stereotyped stroke is its own tell.

So this module answers the question that makes a *tap* safe: which points on the
screen right now have nothing behind them that would react?  Android already
knows -- the accessibility tree carries, for every view, whether it responds to
a click -- and ``uiautomator dump`` hands that tree over without the target
being modified or even aware.  A point inside no reactive view's bounds is a
point where a contact is recorded by the digitiser, travels the whole input
stack, and changes nothing.

A tap is also the only filler with *zero* displacement.  A scroll moves the
list, and moving it back does not restore it -- fling carries past the release
point, and a list already at its top absorbs one direction entirely -- so a
gap filled with scrolls leaves the screen somewhere the agent has never seen.
Taps do not have that failure mode at all.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# Views that answer a contact.  ``clickable`` is the obvious one; the others
# matter because a view can react without claiming to be clickable -- a text
# field takes focus and raises the keyboard, a switch toggles on touch-up.
_REACTIVE_ATTRS = ("clickable", "long-clickable", "checkable")
_REACTIVE_CLASSES = (
    "EditText", "Button", "Switch", "CheckBox", "RadioButton",
    "SeekBar", "Spinner", "ToggleButton", "ImageButton",
)

_NODE = re.compile(r"<node\b[^>]*>")
_BOUNDS = re.compile(r'bounds="\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]"')
_CLASS = re.compile(r'class="([^"]*)"')

Rect = tuple[int, int, int, int]


def _reactive(node: str) -> bool:
    if any(f'{attr}="true"' in node for attr in _REACTIVE_ATTRS):
        return True
    match = _CLASS.search(node)
    return bool(match) and any(name in match.group(1) for name in _REACTIVE_CLASSES)


def parse_reactive(xml: str) -> list[Rect]:
    """Bounds of every view on screen that would answer a contact.

    Parent bounds are kept rather than subtracted out.  A ``ViewGroup`` marked
    clickable handles touches anywhere inside it, including over children that
    claim nothing themselves, so the whole rectangle is out of bounds -- and a
    row in a list is exactly that shape.
    """

    found: list[Rect] = []
    for node in _NODE.findall(xml):
        if not _reactive(node):
            continue
        bounds = _BOUNDS.search(node)
        if not bounds:
            continue
        left, top, right, bottom = (int(v) for v in bounds.groups())
        if right > left and bottom > top:
            found.append((left, top, right, bottom))
    return found


@dataclass
class InertMap:
    """The parts of the current screen where a contact is inert.

    ``blocked`` is what must be avoided; ``sample`` finds a point that is not
    in any of it, by rejection rather than by carving the free space into
    rectangles.  Rejection is used because the free region of a real screen is
    not a handful of boxes -- it is whatever is left over -- and because the
    sampler wants a *scattered* point anyway.  Carving would produce a tidy
    region and then a tap in the middle of it, session after session, which is
    the stereotype this module exists to avoid.
    """

    width: int
    height: int
    blocked: list[Rect] = field(default_factory=list)
    # The status bar at the top and the gesture handle at the bottom belong to
    # the system, not the app, and both react to contact.  They never appear in
    # the app's own tree, so they are excluded by geometry instead.
    top_margin: int = 130
    bottom_margin: int = 130
    # How far a filler tap must stay from anything reactive.  A contact is not
    # a point -- the digitiser reports an area, and a touch slop region around
    # a button still hits the button.
    clearance: int = 48
    ime_shown: bool = False

    def is_inert(self, x: float, y: float) -> bool:
        if not (self.clearance <= x <= self.width - self.clearance):
            return False
        if not (self.top_margin <= y <= self.height - self.bottom_margin):
            return False
        c = self.clearance
        for left, top, right, bottom in self.blocked:
            if left - c <= x <= right + c and top - c <= y <= bottom + c:
                return False
        return True

    def sample(self, rng: random.Random, attempts: int = 200) -> Optional[tuple[float, float]]:
        """A random inert point, or ``None`` when the screen has none.

        ``None`` is a real answer, not a failure: a screen that is wall to wall
        buttons has nowhere safe, and the caller is expected to fall back to a
        gesture that does not tap rather than to tap anyway.
        """

        if self.ime_shown:
            # With the keyboard up, most of the lower screen is keys, and a tap
            # in the content area above it takes focus off the field the agent
            # is in the middle of filling.  Neither is worth the gesture.
            return None
        for _ in range(attempts):
            x = rng.uniform(self.clearance, self.width - self.clearance)
            y = rng.uniform(self.top_margin, self.height - self.bottom_margin)
            if self.is_inert(x, y):
                return (x, y)
        return None

    def free_fraction(self, grid: int = 24) -> float:
        """Roughly how much of the screen is inert -- for reports and tests."""

        hits = 0
        total = 0
        for i in range(grid):
            for j in range(grid):
                x = self.width * (i + 0.5) / grid
                y = self.height * (j + 0.5) / grid
                total += 1
                hits += int(self.is_inert(x, y))
        return hits / total if total else 0.0


_REMOTE_XML = "/sdcard/actreal_ui.xml"


def read_inert(
    adb: Any,
    *,
    width: int,
    height: int,
    timeout: float = 20.0,
) -> InertMap:
    """Ask the device what is on screen and where it is safe to touch.

    Everything here is read-only and goes through interfaces the platform
    publishes: the target is not modified, not instrumented, and cannot tell
    it was asked.
    """

    dump = adb.shell(f"uiautomator dump {_REMOTE_XML}", timeout=timeout)
    xml = ""
    if dump.ok and "dumped to" in (dump.stdout or ""):
        got = adb.shell(f"cat {_REMOTE_XML}", timeout=timeout)
        if got.ok:
            xml = got.stdout or ""
        adb.shell(f"rm -f {_REMOTE_XML}", timeout=timeout)

    blocked = parse_reactive(xml) if xml else []
    ime = adb.shell("dumpsys input_method", timeout=timeout)
    shown = "mInputShown=true" in (ime.stdout or "")

    return InertMap(
        width=width,
        height=height,
        blocked=blocked,
        ime_shown=shown,
        # An empty tree means the question was not answered -- a screen mid
        # transition, a window that refuses to be dumped.  Reporting "all of it
        # is safe" would be the one wrong answer, so the map is left with no
        # free space and the caller falls back to not tapping.
    ) if xml else InertMap(
        width=width, height=height,
        blocked=[(0, 0, width, height)], ime_shown=shown,
    )


def describe(inert: InertMap) -> dict[str, Any]:
    return {
        "reactive_views": len(inert.blocked),
        "free_fraction": round(inert.free_fraction(), 3),
        "ime_shown": inert.ime_shown,
    }
