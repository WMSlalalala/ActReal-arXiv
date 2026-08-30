"""Take over a mobile agent's physical actions without touching the agent.

Mobile-Agent-E reaches Android through five functions in
``MobileAgentE.controller``.  Replacing them replaces the whole physical layer:
the planner, the prompts, the reflection loop and the task set all stay exactly
as they are, and every action the agent decides on is realised as a trajectory
with its own inertia instead of ``adb shell input``.

One detail decides whether the patch works at all: ``MobileAgentE.agents`` does
``from MobileAgentE.controller import tap, swipe, type, ...``, so those names
live in the *agents* module's namespace. Patching only the controller module
would leave the agent calling the originals. Both are rebound here, and
:meth:`ActRealController.install` reports what it actually rebound rather than
assuming.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import background
from .bundle import ActionBundle
from .control import ControlClient
from .frameworks import FrameworkSpec, get as get_framework
from .idle import IdleFiller
from .typing_bundle import compose_typing
from .mapping import ScreenMapping
from .pacing import DelayPolicy
from .inject.imu import BusImuBackend
from .planner import ActionPlanner, BundleLibrary, Plan
from .session import ActionReceipt, PlayReceipt, play_action, play_bundle


@dataclass
class ActionRecord:
    index: int
    api: str
    arguments: dict[str, Any]
    served: bool
    plan: Optional[dict[str, Any]] = None
    receipt: Optional[dict[str, Any]] = None
    fallback_reason: str = ""
    wall_ms: float = 0.0
    pacing: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "api": self.api,
            "arguments": self.arguments,
            "served": self.served,
            "fallback_reason": self.fallback_reason,
            "wall_ms": round(self.wall_ms, 1),
            "plan": self.plan,
            "receipt": self.receipt,
            "pacing": self.pacing,
        }


@dataclass
class ActRealController:
    """The replacement for the framework's five physical primitives."""

    # Optional, because configuration two has no control channel to the target:
    # the app there is one we did not write and did not modify, so the clock and
    # the inertial scheduling both come from the in-process hook instead. The
    # two remaining uses below sit inside `config is None` branches, which is
    # the in-app dispatch path that needs it.
    client: Optional[ControlClient]
    planner: ActionPlanner
    fallback: Optional[Any] = None
    spec: Optional[FrameworkSpec] = None
    pacing: Optional[DelayPolicy] = None
    # Which pair of backends realises an action.  None keeps the original
    # behaviour -- the app dispatching its own MotionEvents -- so a caller that
    # has no probe, no adb or no open device still runs, and says so.
    config: Optional[Any] = None
    # Plays the victim's own gestures while the model is thinking, so a
    # twenty-second decision does not appear in the recording as a
    # twenty-second gap.  None leaves the gaps as the agent makes them.
    idle: Optional[IdleFiller] = None
    lead_ms: float = 250.0
    settle_ms: float = 150.0
    # What one character costs the framework, in wall time.  It enters text one
    # `adb shell input text` per character, so this sets how long a typing
    # action lasts and therefore how long a window has to cover.  Seeded with a
    # plausible round trip and then measured, per session, from what actually
    # happened.
    per_character_ms: float = 60.0
    # Typing by contact needs three things the device and the victim supply
    # separately: where this phone's keys are, how fast this person moves
    # between keys, and how long they hold one down.  Any of them missing and
    # typing falls back to inserting the string with inertia over it, which is
    # what every run did before these existed.
    keymap: Optional[Any] = None
    rhythm: Optional[Any] = None
    press_pool: list[float] = field(default_factory=list)
    seed: int = 0
    # Needed to check the keyboard is actually up before typing by contact.
    # The calibrated key rectangles describe the screen *with* the IME on it;
    # dispatched while it is still animating in, the same coordinates land on
    # whatever the app is showing down there, which is how a typing action
    # would turn into a stray tap on the page behind it.
    adb: Optional[Any] = None
    ime_settle_s: float = 0.35
    ime_wait_s: float = 2.5
    # What the idle stream was seeded with, carried into the session report
    # so a run says whether it had one at all.
    background: dict[str, Any] = field(default_factory=dict)
    log: list[ActionRecord] = field(default_factory=list)
    _patched: dict[Any, Any] = field(default_factory=dict, repr=False)
    _last_action_end: Optional[float] = field(default=None, repr=False)
    _typing_rng: Optional[Any] = field(default=None, repr=False)
    patch_failures: dict[str, str] = field(default_factory=dict)

    # -- the framework-facing surface -----------------------------------------

    def tap(self, adb_path: str, x: float, y: float) -> None:
        self._run("tap", {"x": x, "y": y}, lambda: self.planner.plan("tap", (float(x), float(y))),
                  lambda: self._fallback_call("tap", adb_path, x, y))

    def swipe(self, adb_path: str, x1: float, y1: float, x2: float, y2: float) -> None:
        self._run(
            "swipe",
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            lambda: self.planner.plan("gesture", (float(x1), float(y1)), (float(x2), float(y2))),
            lambda: self._fallback_call("swipe", adb_path, x1, y1, x2, y2),
        )

    def pinch(self, adb_path: str, x: float, y: float, direction: str = "out") -> None:
        """Two contacts about (x, y), moving apart or together.

        This is the one primitive with no fallback worth taking.  Every other
        action degrades to `adb shell input`, which is worse but is still the
        action; `input` has no multi-touch verb, so a pinch that falls back
        does not become a poor pinch, it becomes nothing.  The framework's own
        stub says as much and the record keeps the reason.
        """

        wanted = "in" if str(direction).strip().lower().startswith("in") else "out"
        self._run(
            "pinch",
            {"x": x, "y": y, "direction": wanted},
            lambda: self.planner.plan("pinch", (float(x), float(y))),
            lambda: self._fallback_call("pinch", adb_path, x, y, wanted),
        )

    def type(self, adb_path: str, text: str) -> None:
        """Supply the inertia of typing, and let the framework enter the text.

        **No touch is dispatched for typing, on purpose.** A keystroke donor is
        a run of presses on somebody else's soft keyboard; replaying its
        coordinates here would land them wherever this keyboard's keys are not.
        Worse, it landed them *before* the text went in: the framework types
        with ``input text``, which goes to whatever view holds focus, and a
        stray contact in the lower third of the screen is an excellent way to
        move focus off the field the agent just selected. Typing that "always
        failed to find the position" was this shim taking the focus away from
        it.

        What survives is the half that is real. The hand's motion while typing
        is what the donor actually recorded, and it is delivered on the
        inertial channel across the interval the characters are entered.
        """

        if self._type_by_contact(text):
            return
        self._run_typing(adb_path, text)

    def _keyboard_is_up(self) -> bool:
        """Whether the IME is on screen, waited for briefly if it is not.

        Declining is the safe answer: with no adb there is no way to ask, and
        typing at calibrated coordinates on the chance the keyboard is there
        would put contacts on the page underneath it.
        """

        if self.adb is None:
            return False
        deadline = time.perf_counter() + self.ime_wait_s
        while True:
            try:
                shown = self.adb.shell("dumpsys input_method", timeout=8.0)
            except Exception:
                return False
            if "mInputShown=true" in (shown.stdout or ""):
                # Up, but possibly still sliding; the keys are not where they
                # will settle until the animation ends.
                time.sleep(self.ime_settle_s)
                return True
            if time.perf_counter() >= deadline:
                return False
            time.sleep(0.15)

    def _type_by_contact(self, text: str) -> bool:
        """Type by pressing this phone's keys; report whether it happened.

        One contact per character, at the key the character is actually on,
        spaced by this victim's own measured inter-key gaps and held for one of
        their own measured press durations.  The text arrives *because* of the
        contacts, exactly as it would from a hand -- nothing is inserted
        afterwards, so there is no second copy to undo.

        False means the attempt was declined before anything was dispatched:
        no layout, no rhythm, or a character this keyboard cannot reach.  It is
        never returned half way through, so a declined request cannot leave a
        partially typed field.
        """

        if not text or self.keymap is None or self.rhythm is None:
            return False
        if not self._keyboard_is_up():
            return False
        if self._typing_rng is None:
            import random as _random
            self._typing_rng = _random.Random(self.seed ^ 0x7959)
        rng = self._typing_rng
        try:
            sequence = self.keymap.sequence(text, rng)
        except KeyError:
            return False
        if not sequence:
            return False

        gaps = self.rhythm.intervals(len(sequence), rng)
        if len(gaps) < len(sequence):
            return False
        pool = self.press_pool or [70.0]

        self._take_screen()
        started = time.perf_counter()
        record = ActionRecord(
            index=len(self.log),
            api="type",
            arguments={"text_length": len(text), "contacts": len(sequence)},
            served=False,
        )
        holds = [pool[rng.randrange(len(pool))] for _ in sequence]
        # Every press is scheduled inside one gesture rather than dispatched in
        # a Python loop: the injection path already carries a millisecond clock
        # down to the kernel, so the intervals come out as measured instead of
        # as measured-plus-whatever-the-interpreter-cost.
        donors = [
            self.planner.plan("tap", point, duration_ms=held).bundle
            for (_label, point), held in zip(sequence[:6], holds[:6])
        ]
        composed = compose_typing(
            sequence=sequence, gaps_ms=gaps, holds_ms=holds,
            press_bundles=donors, rng=rng, mapping=self.planner.mapping,
        )
        if composed is None:
            self._release_screen()
            return False

        try:
            receipt = self._play(composed)
            # `_play` answers with an ActionReceipt, and the report is written
            # with json.dumps: stored raw it takes the whole session's report
            # down at the last line, after the run has already happened.
            record.receipt = receipt.as_dict() if hasattr(receipt, "as_dict") else receipt
            record.served = True
        except Exception as error:
            record.fallback_reason = f"{type(error).__name__}: {error}"

        record.arguments["touch_realised"] = record.served
        record.arguments["contacts_dispatched"] = len(sequence)
        record.arguments["typing_ms"] = composed.provenance.get("typing_ms")
        record.arguments["imu_frames"] = composed.imu_frames
        receipts = [
            {"key": label, "held_ms": round(held, 1),
             "x": round(point[0], 1), "y": round(point[1], 1)}
            for (label, point), held in list(zip(sequence, holds))[:40]
        ]
        record.arguments["median_gap_ms"] = round(sorted(gaps)[len(gaps) // 2], 1)
        record.arguments["keys"] = receipts[:40]
        record.wall_ms = (time.perf_counter() - started) * 1000.0
        paced = self._pace("type", started)
        if paced is not None:
            record.pacing = paced.as_dict()
        self.log.append(record)
        self._release_screen()
        # A run that threw part way has already put contacts on the screen, so
        # the string is partly entered and inserting it again would double it.
        # Reporting success here is wrong, but so is falling back; the agent's
        # own reflector sees the field and corrects, which is what a person
        # doing the same thing would do.
        return True

    def _run_typing(self, adb_path: str, text: str) -> None:
        self._take_screen()
        started = time.perf_counter()
        record = ActionRecord(
            index=len(self.log),
            api="type",
            arguments={"text_length": len(text)},
            served=False,
        )
        try:
            # The framework enters one character per adb round trip, so the
            # interval to cover is set by how much text there is, not by a draw
            # from the victim's distribution.  The per-character cost is
            # measured below and fed back, so the estimate stops being a guess
            # after the first typing action of a session.
            target_ms = max(1.0, len(text)) * self.per_character_ms
            plan = self.planner.plan(
                "keystroke", self._keyboard_anchor(), duration_ms=target_ms
            )
            record.plan = plan.as_dict()
            receipt = self._play_inertia_only(plan.bundle)
            record.receipt = receipt
            record.served = True
        except Exception as error:
            record.fallback_reason = f"{type(error).__name__}: {error}"

        typed_from = time.perf_counter()
        self._fallback_call("type", adb_path, text)
        typed_ms = (time.perf_counter() - typed_from) * 1000.0
        if text and self.fallback is not None and typed_ms > 0.0:
            # An exponential average, so one stalled adb call does not move the
            # estimate far and a device that is simply slow is tracked.
            #
            # Guarded on there being a fallback that actually ran: with none --
            # the simulator, or a framework whose `type` is absent -- the
            # measurement is zero, and feeding zeroes back would walk the
            # estimate to nothing and then pick the shortest donor in the
            # library for every message regardless of length.  The floor is the
            # same protection against a device that answers implausibly fast.
            observed = typed_ms / len(text)
            self.per_character_ms = max(
                10.0, 0.7 * self.per_character_ms + 0.3 * observed
            )
        record.arguments["typed_ms"] = round(typed_ms, 1)
        record.arguments["per_character_ms"] = round(self.per_character_ms, 2)
        record.arguments["touch_realised"] = False
        record.arguments["touch_skipped_reason"] = (
            "a keystroke donor's contacts belong to another keyboard's keys; "
            "dispatching them would land them on nothing and could take focus "
            "off the field being typed into"
        )
        record.wall_ms = (time.perf_counter() - started) * 1000.0
        paced = self._pace("type", started)
        if paced is not None:
            record.pacing = paced.as_dict()
        self.log.append(record)
        self._release_screen()

    def enter(self, adb_path: str) -> None:
        self._passthrough("enter", {}, lambda: self._fallback_call("enter", adb_path))

    def back(self, adb_path: str) -> None:
        self._passthrough("back", {}, lambda: self._fallback_call("back", adb_path))

    def home(self, adb_path: str) -> None:
        self._passthrough("home", {}, lambda: self._fallback_call("home", adb_path))

    def switch_app(self, adb_path: str) -> None:
        self._passthrough("switch_app", {}, lambda: self._fallback_call("switch_app", adb_path))

    # -- machinery ------------------------------------------------------------

    def _keyboard_anchor(self) -> tuple[float, float]:
        """Where a soft keyboard sits, in device pixels: the lower third."""

        left, top, right, bottom = self.planner.mapping.usable_rect
        return ((left + right) / 2.0, top + (bottom - top) * 0.78)

    def _fallback_call(self, name: str, *args) -> None:
        """Call the framework's own implementation of a primitive.

        **It has to be the saved original, not the live attribute.** Installing
        this shim replaces ``MobileAgentE.controller.type`` with this object's
        ``type``; reading the name back off the module afterwards therefore
        returns *us*, and calling it recurses until the stack ends. Every
        primitive that falls back was affected -- ``type``, which always does,
        and ``enter``, ``back``, ``home`` and ``switch_app``, which are pure
        passthroughs -- so five of the seven died the first time the agent
        reached for them, and the sixth and seventh died whenever ActReal could
        not serve them.

        ``install`` already keeps the originals so it can undo itself; this
        reads from the same place.
        """

        if self.fallback is None:
            return
        function = self._framework_original(name)
        if function is not None:
            function(*args)

    def _framework_original(self, name: str) -> Optional[Callable[..., Any]]:
        module_name = getattr(self.fallback, "__name__", "")
        saved = self._patched.get((module_name, name))
        if saved is not None:
            return saved
        # Nothing rebound yet, so whatever is on the module is still the
        # framework's own.  Reading it here keeps the shim usable before
        # install() and in tests that never install at all.
        return getattr(self.fallback, name, None)

    def _passthrough(self, api: str, arguments: dict[str, Any], call: Callable[[], None]) -> None:
        """A key event has no touch, so there is nothing for ActReal to realise.

        Recorded anyway: an action with no inertia is a hole in the session,
        and the session-level view needs to know where the holes are.
        """

        self._take_screen()
        started = time.perf_counter()
        call()
        record = ActionRecord(
            index=len(self.log),
            api=api,
            arguments=arguments,
            served=False,
            fallback_reason="key event, no touch to realise",
            wall_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._pace("key", started)
        self.log.append(record)
        self._release_screen()

    def _run(
        self,
        api: str,
        arguments: dict[str, Any],
        make_plan: Callable[[], Plan],
        fallback: Callable[[], None],
        *,
        always_fallback: bool = False,
    ) -> None:
        self._take_screen()
        started = time.perf_counter()
        record = ActionRecord(index=len(self.log), api=api, arguments=arguments, served=False)
        try:
            plan = make_plan()
            record.plan = plan.as_dict()
            if not plan.reachable:
                raise ValueError("target outside the mapped rectangle")
            receipt = self._play(plan.bundle)
            record.receipt = receipt.as_dict()
            record.served = True
            if always_fallback:
                fallback()
        except Exception as error:
            # Falling back keeps the agent's task moving, but the reason is
            # kept: an action served by `adb shell input` has none of the
            # physical realisation this project is about, and a run where that
            # happened silently would be reported as something it was not.
            record.fallback_reason = f"{type(error).__name__}: {error}"
            fallback()
        record.wall_ms = (time.perf_counter() - started) * 1000.0
        paced = self._pace(
            "gesture" if api == "swipe" else api, started
        )
        if paced is not None:
            record.pacing = paced.as_dict()
        self.log.append(record)
        self._release_screen()

    def _uptime_ms(self) -> int:
        """The device clock, from whichever half of the configuration has it.

        Configuration one reads it through the target app's control channel.
        Configuration two has no such channel -- the target is an app we did not
        write and did not modify -- so the reading comes from inside the process
        through the hook instead. Both are device-side reads; routing every
        caller through here is what lets the same shim drive either.
        """

        if self.config is not None and hasattr(self.config.imu, "read_clock"):
            return int(self.config.imu.read_clock().uptime_ms)
        if self.client is None:
            raise RuntimeError("no control channel and no clock-capable IMU backend")
        return int(self.client.ping()["uptime_ms"])

    def _take_screen(self) -> None:
        """Stop the filler and wait for it, before a real action starts.

        Every path that puts something on the screen calls this first.  Two
        gestures overlapping would write a contact into the recording that no
        single hand made, and the filler also restores its scroll position
        here, so the coordinates the agent chose from its screenshot still
        point at what it saw.
        """

        if self.idle is not None:
            self.idle.end()

    def _release_screen(self) -> None:
        """Hand the screen to the filler while the agent decides what is next.

        The gap that follows is a screenshot plus a model call -- tens of
        seconds, on every action.  `_pace` cannot help with that: it shapes a
        gap by waiting, so it can only make a short one longer.  This makes a
        long one into several ordinary ones.
        """

        if self.idle is not None:
            self.idle.begin()

    def _pace(self, kind: str, action_started: float):
        """Shape the gap the app sees, counting time already spent.

        The interval that matters runs from the end of the previous action to
        the end of this one: it contains the agent's own thinking, the
        screenshot, and the playback.  Measuring from there is what turns the
        policy into a target for the whole gap rather than something added on
        top of it.
        """

        now = time.perf_counter()
        if self.pacing is None:
            self._last_action_end = now
            return None
        spent = now - (self._last_action_end if self._last_action_end is not None else action_started)
        gap = self.pacing.wait(kind, already_spent_s=spent)
        self._last_action_end = time.perf_counter()
        return gap

    def _play_inertia_only(self, bundle: ActionBundle) -> dict[str, Any]:
        """Schedule a window's inertia with no touch at all.

        Used by typing.  The IMU half of a configuration is addressed directly
        rather than through :func:`play_action`, because that function's whole
        job is putting a touch and a window on one timeline and here there is no
        touch to put anywhere.
        """

        now = self._uptime_ms()
        t0 = int(now + round(self.lead_ms))
        if self.config is None:
            reply = self.client.schedule_imu(
                bundle.imu_rows(),
                start_uptime_ms=t0,
                period_ms=bundle.imu_period_ms,
                bundle_id=bundle.bundle_id,
            )
            schedule = {
                "start_elapsed_ns": int(reply["start_elapsed_ns"]),
                "frames": bundle.imu_frames,
                "period_ms": bundle.imu_period_ms,
                "bundle_id": bundle.bundle_id,
                "converted_by": "device",
            }
        else:
            schedule = self.config.imu.schedule(
                bundle, t0_uptime_ms=t0, timebase=self.config.timebase
            ).as_dict()
        # The pre-roll is the hand approaching the keyboard; the characters go
        # in after it, which is why the caller types once this returns.
        time.sleep((self.lead_ms + bundle.touch_offset_ms) / 1000.0)
        return {
            "bundle_id": bundle.bundle_id,
            "action": bundle.action,
            "configuration": getattr(self.config, "name", "none"),
            "t0_uptime_ms": t0,
            "imu_frames": bundle.imu_frames,
            "touch_points": 0,
            "imu": schedule,
            "imu_duration_ms": round(bundle.imu_duration_ms, 1),
        }

    def _play(self, bundle: ActionBundle):
        """Realise one action, through a configuration when there is one.

        Without a configuration this falls back to the app's own dispatch, which
        is what this shim did before the two configurations existed.  That path
        reaches only our target app and carries none of the input pipeline's
        provenance, so which one served an action is recorded per action rather
        than assumed for the run: an agent driven through the in-app path and
        one driven through the input pipeline are not the same experiment.
        """

        if self.config is None:
            receipt = play_bundle(self.client, bundle, lead_ms=self.lead_ms)
        else:
            receipt = play_action(
                self.config,
                bundle,
                read_uptime_ms=self._uptime_ms,
                lead_ms=self.lead_ms,
                # The mode is set once when the controller is built; asking
                # again per action would add a round trip to every gesture.
                set_mode=False,
            )
        # The touch may already have played out -- a paced backend blocks for
        # the gesture -- but the inertial window runs to the end of its
        # post-roll either way, and the next action must not start inside it.
        time.sleep((self.lead_ms + bundle.imu_duration_ms + self.settle_ms) / 1000.0)
        return receipt

    # -- installation ---------------------------------------------------------

    def install(self, spec: Optional[FrameworkSpec] = None) -> dict[str, list[str]]:
        """Rebind a framework's primitives onto this controller.

        Returns what was rebound where, so a caller can assert the patch landed
        instead of trusting that it did.  Nothing rebound is an error, not a
        warning: a silent no-op here means every action goes out as
        ``adb shell input`` and the run looks normal.
        """

        spec = spec or self.spec or get_framework("mobile-agent-e")
        self.spec = spec
        rebound: dict[str, list[str]] = {}
        failures: dict[str, str] = {}

        for module_name in spec.rebind_modules:
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                # Not only ImportError: a framework module can fail on anything
                # its own imports do.  Either way the names in it stay
                # unpatched, which is the failure this whole layer exists to
                # avoid, so the reason is kept rather than swallowed.
                failures[module_name] = f"{type(error).__name__}: {error}"
                continue
            hit = []
            for target in spec.framework_names():
                if not hasattr(module, target):
                    continue
                replacement = self._replacement_for(spec, target)
                if replacement is None:
                    continue
                self._patched.setdefault((module_name, target), getattr(module, target))
                setattr(module, target, replacement)
                hit.append(target)
            if hit:
                rebound[module_name] = hit

        for reference in spec.rebind_classes:
            module_name, _, class_name = reference.partition(":")
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            hit = []
            for target in spec.framework_names():
                if not hasattr(cls, target):
                    continue
                replacement = self._replacement_for(spec, target, bound=True)
                if replacement is None:
                    continue
                self._patched.setdefault((reference, target), getattr(cls, target))
                setattr(cls, target, replacement)
                hit.append(target)
            if hit:
                rebound[reference] = hit

        missed = [name for name in spec.required_modules if name not in rebound]
        if missed:
            detail = "; ".join(f"{name}: {failures.get(name, 'no names matched')}" for name in missed)
            raise ImportError(
                f"{spec.name}: required module(s) not patched -- {detail}. "
                "The agent holds its own references to these names, so leaving "
                "them alone means every action goes out as `adb shell input` "
                "while the run looks normal."
            )
        if not rebound:
            raise ImportError(
                f"no {spec.name} module was patched; is it importable? "
                f"tried {spec.rebind_modules + spec.rebind_classes}. {failures}"
            )
        self.patch_failures = failures
        return rebound

    def _replacement_for(self, spec: FrameworkSpec, name: str, bound: bool = False):
        """The function to put in the framework's namespace under ``name``.

        Frameworks disagree about what a gesture is called and about whether
        the first argument is an adb path or ``self``; both differences are
        absorbed here so the physical layer stays one implementation.
        """

        primitive = spec.primitive_for(name)
        if primitive is None:
            return None
        surface = {
            "tap": self.tap,
            "gesture": self.swipe,
            "pinch": self.pinch,
            "type": self.type,
            "key": getattr(self, name, None) or self.back,
        }.get(primitive)
        if surface is None:
            return None
        if not bound:
            return surface
        # A class method is called as method(self, ...); the receiver is the
        # framework's own controller, which carries the adb path we hand back
        # to the fallback.
        def as_method(receiver, *args, _surface=surface, **kwargs):
            return _surface(getattr(receiver, "adb_path", receiver), *args, **kwargs)

        return as_method

    def uninstall(self) -> None:
        if self.idle is not None:
            self.idle.end()
        for (where, target), original in self._patched.items():
            module_name, _, class_name = str(where).partition(":")
            module = importlib.import_module(module_name)
            holder = getattr(module, class_name) if class_name else module
            setattr(holder, target, original)
        self._patched.clear()

    # -- reporting ------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        served = sum(1 for r in self.log if r.served)
        return {
            "schema_version": "actreal_agent_session_v1",
            "actions": len(self.log),
            "served_by_actreal": served,
            "fell_back": len(self.log) - served,
            "by_api": {
                api: sum(1 for r in self.log if r.api == api)
                for api in sorted({r.api for r in self.log})
            },
            "framework": self.spec.name if self.spec else None,
            "pacing": self.pacing.summary() if self.pacing else None,
            "background": self.background or {"frames": 0},
            # Filler gestures are indistinguishable from agent ones in the
            # phone's recording -- that is the point of them -- so the count
            # lives here or nowhere, and a run that used them says so.
            "idle_filler": self.idle.summary() if self.idle else {"enabled": False},
            # Without this a session report says how many actions ActReal served
            # and not what serving them meant.  The two configurations reproduce
            # different things and reach different applications, so a run is not
            # readable without knowing which one was underneath it.
            "configuration": (
                self.config.describe()
                if self.config is not None
                else {
                    "configuration": "none",
                    "detail": "no configuration was supplied; actions were "
                    "dispatched inside the target app, which reaches that app "
                    "only and carries no input-pipeline provenance",
                }
            ),
            "records": [r.as_dict() for r in self.log],
        }

    def write_report(self, path: "str | Path") -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.report(), indent=2))


def build_controller(
    client: Optional[ControlClient],
    bundles_dir: "str | Path",
    *,
    device_w: Optional[int] = None,
    device_h: Optional[int] = None,
    fallback: Optional[Any] = None,
    seed: int = 0,
    spec: Optional[FrameworkSpec] = None,
    pacing: Optional[DelayPolicy] = None,
    config: Optional[Any] = None,
    background_period_ms: float = 10.0,
    fill_idle: bool = True,
    adb: Optional[Any] = None,
) -> ActRealController:
    # The screen size comes from the app when there is a channel to ask, and
    # from the caller when there is not. Configuration two has no channel: the
    # target is unmodified, so nobody there answers questions, and the runner
    # passes the size it read from the device instead.
    if client is not None:
        info = client.hello()
        device_w = device_w or info.display_w
        device_h = device_h or info.display_h
    if not (device_w and device_h):
        raise ValueError(
            "no control channel to ask for the screen size, and none was given; "
            "pass device_w and device_h when driving an unmodified target"
        )
    mapping = ScreenMapping.isotropic(device_w=device_w, device_h=device_h)
    planner = ActionPlanner(BundleLibrary(bundles_dir, seed=seed), mapping)

    # One mode change for the run.  Through a configuration it goes to whichever
    # IMU backend that configuration chose, which for the hook is not the same
    # object as the control channel -- so it cannot be shortcut to the client.
    if config is not None:
        config.imu.set_mode("injected")
    elif client is not None:
        client.set_imu_mode("injected")
    else:
        raise ValueError("neither a configuration nor a control channel to inject through")

    # Between actions the target must still be receiving something.  Injected
    # mode has already taken the phone's own sensors away, and an agent spends
    # most of a session thinking, so without this the recording is mostly a
    # phone that nobody was holding -- one measured gap ran to 257 seconds.
    imu_backend = config.imu if config is not None else BusImuBackend(client)
    background_report = background.install(
        imu_backend, planner.library, period_ms=background_period_ms
    )

    # Typing by contact needs the phone's key positions (measured once by
    # runners/calibrate_keyboard.py, because dumping the tree mid-action costs
    # seconds and can come back empty) and this victim's own typing rhythm.
    # Either missing and typing remains on the framework-native route, which is
    # reported rather than silently counted as an ActReal-served action.
    from .keyboard import load_keymap
    from .rhythm import load_rhythm, press_durations

    victim = Path(bundles_dir).name
    keymap = load_keymap()
    rhythm = load_rhythm(victim)

    controller = ActRealController(
        client=client,
        planner=planner,
        fallback=fallback,
        spec=spec,
        pacing=pacing,
        config=config,
        keymap=keymap,
        rhythm=rhythm,
        press_pool=press_durations(victim),
        seed=seed,
        adb=adb,
    )
    controller.background = background_report
    if fill_idle:
        # Given the controller's own play path rather than a private one: a
        # filler that took a shortcut would be a different signal wearing the
        # same name, and the recording would show the difference.
        # ``adb`` is what lets the filler tap: without a way to ask which
        # views react to a click there is no way to know where a tap is inert,
        # and it falls back to paired scrolls rather than guessing.
        controller.idle = IdleFiller(
            planner=planner, play=controller._play, pacing=pacing,
            adb=adb, device_w=device_w, device_h=device_h, seed=seed,
        )
    return controller
