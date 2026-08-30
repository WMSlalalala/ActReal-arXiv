"""Keep the hand busy while the agent is deciding what to do next.

An agent's gap between two actions is not a hand's.  A person taking the next
step spends a second or two; the agent spends a screenshot, a network round
trip, and however long the model takes to answer -- tens of seconds, every
time, on every action.  :class:`~actreal.pacing.DelayPolicy` cannot help here,
because it can only lengthen a gap that came in short.  When the model has
already burned twenty seconds the target is long gone and the policy waits for
zero, which is the honest thing for it to do and no use at all.

So the gap is not shortened; it is *divided*.  While the agent thinks, this
plays gestures drawn from the same victim's corpus at intervals drawn from the
same victim's timing, and a single twenty-second void becomes a handful of
ordinary ones with ordinary human motion between them.  Nothing about the
injected signal changes -- these are the victim's own recorded gestures, played
by the same path as a real action, and the recording cannot tell which was
which because in every measurable respect there is no difference.

What it must never do is change what the app is showing.  The first version of
this filler met that requirement by refusing to tap and scrolling instead,
always between the same two points down the middle of the screen.  Measured
over a campaign that turned out to trade one tell for a worse one: fifty-seven
filler strokes against seventeen real actions in a single session, every one of
them geometrically identical, so three-quarters of the touch corpus was one
stroke repeated.  And the scrolls did not even hold position -- fling carries
past the release point and a list at its top absorbs one direction outright, so
"equal numbers up and down" is not "back where it started", and the agent's
coordinates went stale anyway.

This version taps instead, which fixes both problems at once.  A tap moves
nothing, so the screen the agent looked at is the screen it acts on; and a tap
can go *anywhere*, so the fillers stop being one shape.  What makes it safe is
:mod:`actreal.inert`: Android publishes, for every view, whether it answers a
click, and a point inside no such view is a contact that the digitiser records,
the input stack delivers, and nothing reacts to.

Scrolls were kept at first, on the argument that a hand which only ever taps is
its own pattern.  A measured session took that away: the framework judges an
action by comparing the screen before it against the screen after, it takes the
"after" shot once this filler has already resumed, and a filler scroll inside
that window is read as the *agent's* action having failed.  Three taps on "Add
to Cart" were scored failures that way -- one verdict reads "the screen merely
scrolled downward" -- and the run stopped on its consecutive-failure limit
having done nothing wrong.  So scrolling survives only for the case where the
device cannot be asked which views react, where it is the one shape that cannot
activate what it lands on.  Where tapping is possible, tapping is all there is,
and a gap with nowhere inert to touch is left quiet rather than scrolled.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .inert import InertMap, read_inert


@dataclass
class FillerRecord:
    """One gesture played into a gap, and what it cost."""

    index: int
    kind: str
    detail: str
    waited_s: float
    played: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "detail": self.detail,
            "waited_s": round(self.waited_s, 3),
            "played": self.played,
            "reason": self.reason,
        }


@dataclass
class IdleFiller:
    """Plays safe, varied gestures into the agent's thinking time.

    ``play`` is handed in rather than reached for so that the filler uses
    exactly the path a real action uses -- same configuration, same clock, same
    launch correction.  A filler that went through a shortcut would be a
    different signal wearing the same name, and the recording would show it.

    ``adb`` is optional, and what is lost without it is only the tapping: with
    no way to ask which views react, there is no way to know where a tap is
    inert, and the filler falls back to paired scrolls rather than guessing.
    """

    planner: Any
    play: Callable[[Any], Any]
    pacing: Optional[Any] = None
    adb: Optional[Any] = None
    device_w: Optional[int] = None
    device_h: Optional[int] = None
    # Gaps shorter than this are already human-shaped and left alone.  Starting
    # a gesture inside one would crowd the real action that is about to arrive.
    threshold_s: float = 2.0
    # How often a filler is a tap rather than a scroll.  One, when the device
    # can be asked where a tap is inert -- and the reason is not only that a
    # scroll leaves the agent's coordinates stale.
    #
    # Mobile-Agent-E decides whether an action worked by comparing the screen
    # before it against the screen after, and it takes the "after" shot at the
    # top of the next iteration, by which time this filler is already running.
    # A filler scroll in that window moves the page under the comparison, and
    # the reflector reports the *agent's* action as having failed.  Measured on
    # a real session: three consecutive taps on "Add to Cart" judged failures,
    # one of them with the verdict "the screen merely scrolled downward", and
    # the run stopped on its consecutive-failure limit having done nothing
    # wrong.  A tap on an inert point does not move anything, so the comparison
    # sees exactly what the agent did and nothing else.
    tap_share: float = 1.0
    # A screen reading is good for this long.  Nothing else is moving the app
    # while the agent thinks, so the only thing that invalidates it is one of
    # our own scrolls, which refreshes it explicitly.
    inert_ttl_s: float = 20.0
    # The victim's own inter-action timings are what the interval is drawn
    # from, but drawn without a ceiling they reproduce the very thing the
    # filler is here to break up: a tail sample of eleven seconds leaves an
    # eleven-second hole in the middle of a gap that was already too quiet.
    # So the draw is kept and then clamped, which keeps the shape of the
    # distribution over the body of it and refuses only the tail.
    min_interval_s: float = 1.2
    max_interval_s: float = 6.0
    seed: int = 0
    enabled: bool = True
    log: list[FillerRecord] = field(default_factory=list)

    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _busy: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _rng: random.Random = field(default=None, repr=False)  # type: ignore[assignment]
    _inert: Optional[InertMap] = field(default=None, repr=False)
    _inert_at: float = field(default=0.0, repr=False)
    _inert_reads: int = field(default=0, repr=False)
    _pending_inverse: Optional[tuple[tuple[float, float], tuple[float, float]]] = field(
        default=None, repr=False
    )
    _restores: int = field(default=0, repr=False)
    _taps: int = field(default=0, repr=False)
    _scrolls: int = field(default=0, repr=False)
    _keyboard_rect: Optional[tuple[int, int, int, int]] = field(default=None, repr=False)
    _keyboard_read: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self._rng is None:
            self._rng = random.Random(self.seed)

    # -- what the screen will tolerate ----------------------------------------

    def _reach(self) -> tuple[float, float, float, float]:
        """The rectangle the planner accepts without clamping."""

        mapping = self.planner.mapping
        left, top = mapping.offset_x, mapping.offset_y
        right = left + mapping.source_w * mapping.scale
        bottom = top + mapping.source_h * mapping.scale
        return left, top, right, bottom

    def _screen(self) -> tuple[int, int]:
        mapping = self.planner.mapping
        return (
            int(self.device_w or getattr(mapping, "device_w", 0) or 1080),
            int(self.device_h or getattr(mapping, "device_h", 0) or 1920),
        )

    def _inert_map(self, force: bool = False) -> Optional[InertMap]:
        """The current reading of where a tap is safe, refreshed when stale."""

        if self.adb is None:
            return None
        fresh = (time.monotonic() - self._inert_at) < self.inert_ttl_s
        if self._inert is not None and fresh and not force:
            return self._inert
        width, height = self._screen()
        try:
            self._inert = read_inert(self.adb, width=width, height=height)
            self._inert_reads += 1
        except Exception:
            # A screen that will not be dumped is a screen we will not tap on.
            self._inert = None
        self._inert_at = time.monotonic()
        return self._inert

    # -- the two shapes a filler takes ----------------------------------------

    def _keyboard_up(self) -> bool:
        """Asked fresh every time, never from the cached reading.

        The reading carries both the geometry and whether the IME was on screen,
        and only the first of those keeps.  Cached together for twenty seconds,
        a map taken while the keyboard was down stayed "safe to tap" after the
        agent had tapped a field and raised it -- so a filler landed on what was
        now a key and typed a character nobody asked for.  Measured: a search
        for "usb c hub" went in as "2usb c hu", the 2 being the key above w and
        no part of any string the agent typed.
        """

        if self.adb is None:
            return True
        try:
            shown = self.adb.shell("dumpsys input_method", timeout=8.0)
        except Exception:
            return True   # cannot tell: assume the worst and do not tap
        return "mInputShown=true" in (shown.stdout or "")

    def _keyboard_area(self) -> Optional[tuple[int, int, int, int]]:
        """The rectangle the keys occupy when the IME is up, if it was measured."""

        if not self._keyboard_read:
            self._keyboard_read = True
            try:
                from .keyboard import load_keymap
                keymap = load_keymap()
                self._keyboard_rect = keymap.area if keymap else None
            except Exception:
                self._keyboard_rect = None
        return self._keyboard_rect

    def _tap_point(self) -> Optional[tuple[float, float]]:
        if self._keyboard_up():
            return None
        inert = self._inert_map()
        if inert is None:
            return None
        point = inert.sample(self._rng)
        if point is None:
            return None
        # The planner will refuse anything outside the mapped rectangle, so a
        # point that is inert but unreachable is no use.
        left, top, right, bottom = self._reach()
        x, y = point
        if not (left <= x <= right and top <= y <= bottom):
            return None
        # Second line, because the first one is a question asked over adb and
        # the answer can be stale by the time the contact goes out.  Where the
        # keys sit was measured once; nothing inside that rectangle is ever an
        # idle tap, keyboard up or not.
        area = self._keyboard_area()
        if area is not None:
            kl, kt, kr, kb = area
            if kl <= x <= kr and kt <= y <= kb:
                return None
        return point

    def _scroll_run(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """A vertical run, randomised, or the inverse of the last one.

        Returning the inverse when one is owed is what keeps position: the pair
        is the same stroke forwards and backwards rather than two unrelated
        strokes that happen to point opposite ways, so whatever the first one
        did to the list the second one asks to be undone.
        """

        if self._pending_inverse is not None:
            start, end = self._pending_inverse
            self._pending_inverse = None
            return start, end
        left, top, right, bottom = self._reach()
        span = bottom - top
        # Away from the middle line every time, and away from the edges, where
        # a stroke risks the system's own back and home gestures.
        x = self._rng.uniform(left + (right - left) * 0.25, right - (right - left) * 0.25)
        length = self._rng.uniform(span * 0.15, span * 0.33)
        margin = span * 0.12
        if self._rng.random() < 0.5:
            near = self._rng.uniform(top + margin + length, bottom - margin)
            far = near - length
        else:
            near = self._rng.uniform(top + margin, bottom - margin - length)
            far = near + length
        start, end = (x, near), (x, far)
        self._pending_inverse = (end, start)
        return start, end

    # -- the loop --------------------------------------------------------------

    def _next_wait(self, index: int) -> float:
        """How long to leave the hand still before the next filler.

        Drawn per gesture rather than fixed, and drawn for a randomly chosen
        action type, so the intervals carry the victim's spread instead of one
        action's rhythm repeated.  Then clamped at both ends: too short crowds
        the next gesture, too long is the silence this exists to remove.
        """

        if self.pacing is not None:
            kind = "tap" if self._rng.random() < self.tap_share else "scroll"
            try:
                base = float(self.pacing.target_for(kind, index))
            except Exception:
                base = self._rng.uniform(self.min_interval_s, self.max_interval_s)
        else:
            base = self._rng.uniform(self.min_interval_s, self.max_interval_s)
        base *= self._rng.uniform(0.75, 1.25)
        return max(self.min_interval_s, min(base, self.max_interval_s))

    def _loop(self) -> None:
        while not self._stop.is_set():
            index = len(self.log)
            wait = self._next_wait(index)
            # First interval also serves as the threshold: a gap that ends
            # before it elapses never sees a filler at all.
            if self._stop.wait(max(wait, self.threshold_s)):
                return
            if not self._busy.acquire(blocking=False):
                return
            try:
                if self._stop.is_set():
                    return
                self._play_one(wait)
            finally:
                self._busy.release()

    def _play_one(self, waited_s: float) -> bool:
        """One filler gesture; returns whether it actually played."""

        index = len(self.log)
        kind = "tap" if self._rng.random() < self.tap_share else "scroll"
        point = self._tap_point() if kind == "tap" else None
        if kind == "tap" and point is None:
            if self.adb is not None:
                # The device *can* be asked and the answer was "nowhere" -- the
                # keyboard is up, or the screen is wall to wall controls.  The
                # gap simply stays quiet on the touch channel; the background
                # inertia stream is still running, so the recording is of a
                # hand holding a phone and reading, which is what the agent is
                # doing.  Scrolling here to avoid a silence would move the page
                # under the framework's own before/after comparison and cost a
                # real action, which is a far worse trade.
                self.log.append(
                    FillerRecord(index, "tap", "", waited_s, False, "no inert point"))
                return False
            # With no way to ask, a scroll is the only safe shape left: it is
            # the one gesture that cannot activate whatever it lands on.
            kind = "scroll"

        if kind == "tap":
            assert point is not None
            start, end = point, None
            detail = "%.0f,%.0f" % point
        else:
            start, end = self._scroll_run()
            detail = "%.0f,%.0f->%.0f,%.0f" % (start[0], start[1], end[0], end[1])

        try:
            plan = self.planner.plan(kind, start, end)
            self.play(plan.bundle)
        except Exception as error:
            # A filler that fails is not a run that fails.  It is recorded and
            # the gap simply stays long, which is the situation the filler
            # existed to improve, not to require.
            self.log.append(
                FillerRecord(index, kind, detail, waited_s, False,
                             f"{type(error).__name__}: {error}")
            )
            if kind == "scroll":
                # The stroke never happened, so nothing is owed back.
                self._pending_inverse = None
            return False

        self.log.append(FillerRecord(index, kind, detail, waited_s, True))
        if kind == "tap":
            self._taps += 1
        else:
            self._scrolls += 1
            # The list moved, so what was read about the screen no longer holds.
            self._inert_at = 0.0
        return True

    # -- the surface the controller uses ---------------------------------------

    def begin(self) -> None:
        """Start filling; safe to call when already filling."""

        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="actreal-idle", daemon=True)
        self._thread.start()

    def end(self, timeout_s: float = 20.0) -> None:
        """Stop filling and wait for any gesture already in flight.

        Waiting is the point.  A filler halfway through its stream while the
        real action starts would interleave two gestures on one screen, and the
        recording would show a contact that no single hand made.
        """

        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        # Belt and braces: if the thread outlived its join it is still inside a
        # gesture, and taking the lock waits for exactly that and nothing else.
        with self._busy:
            # The agent chose its coordinates from a screenshot taken before
            # this gap began.  A scroll whose inverse was never played leaves
            # the list somewhere else, and the action about to go out would
            # land on whatever slid into that position -- the filler would have
            # caused the agent to misclick, which is far worse than a long gap.
            if self._pending_inverse is not None:
                self._restores += 1
                self._play_one(0.0)

    def summary(self) -> dict[str, Any]:
        played = [r for r in self.log if r.played]
        return {
            "enabled": self.enabled,
            "gestures": len(played),
            "failed": len(self.log) - len(played),
            "taps": self._taps,
            "scrolls": self._scrolls,
            "restores": self._restores,
            "screen_reads": self._inert_reads,
            "can_tap": self.adb is not None,
            "tap_share": self.tap_share,
            "threshold_s": self.threshold_s,
            "log": [r.as_dict() for r in self.log],
        }
