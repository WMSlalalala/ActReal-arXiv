"""How long to wait between actions.

Mobile-Agent-E waits a fixed amount after every action -- five seconds after a
tap or a swipe, three after typing, ten after Enter -- and the loop adds its
own sleep on top.  Two things are wrong with that, and only one of them is the
length.

A *constant* gap is the bigger problem.  Whether it is five seconds or half a
second, every action separated by the same interval is a rhythm no hand
produces, and a session-level detector reads it without knowing anything about
gestures.  Shortening the constants alone would swap one constant for another.

The second thing is that the wait is not the whole gap.  The agent has already
spent seconds thinking, and ActReal has spent more playing the action out.
What an app sees is the sum, so the sum is what has to be shaped: this samples
a target gap, subtracts what has already been spent, and sleeps for whatever is
left -- which is often nothing at all.

There is still a floor, because an interface needs time to settle before the
next action can land on the right thing; below it the agent stops completing
tasks, which is its own kind of failure.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Median inter-action gap and log spread to aim for, per action, in seconds.
# The wait after an action is mostly the interface's: a list has to redraw, an
# app has to open.  These are targets for the whole gap, not additions to it.
DEFAULT_TARGETS: dict[str, tuple[float, float]] = {
    "tap": (1.1, 0.55),
    "gesture": (0.9, 0.55),
    "type": (1.4, 0.5),
    "key": (1.0, 0.5),
    "open_app": (4.0, 0.35),
    "default": (1.2, 0.55),
}

# Never go below this after an action, whatever the draw says: the interface
# needs it, and an action that lands on a screen that has not finished drawing
# fails the task rather than the detector.
FLOOR_S: dict[str, float] = {
    "tap": 0.6,
    "gesture": 0.5,
    "type": 0.5,
    "key": 0.5,
    "open_app": 2.5,
    "default": 0.5,
}

# A gap longer than this is a stall, not a pause.
CEILING_S = 12.0


@dataclass
class PacedGap:
    action: str
    target_s: float
    already_spent_s: float
    slept_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_s": round(self.target_s, 3),
            "already_spent_s": round(self.already_spent_s, 3),
            "slept_s": round(self.slept_s, 3),
            "gap_s": round(max(self.target_s, self.already_spent_s), 3),
        }


@dataclass
class DelayPolicy:
    """Shapes the gap an app sees between one action and the next."""

    targets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_TARGETS)
    )
    floors: dict[str, float] = field(default_factory=lambda: dict(FLOOR_S))
    ceiling_s: float = CEILING_S
    seed: str = "actreal"
    enabled: bool = True
    log: list[PacedGap] = field(default_factory=list)

    def _draw(self, action: str, index: int) -> float:
        median, spread = self.targets.get(action, self.targets["default"])
        digest = hashlib.sha256(f"{self.seed}|{action}|{index}".encode()).digest()
        a = int.from_bytes(digest[0:8], "big") / float(1 << 64)
        b = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        a = min(max(a, 1e-12), 1.0 - 1e-12)
        z = math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)
        return float(median * math.exp(spread * z))

    def target_for(self, action: str, index: int) -> float:
        floor = self.floors.get(action, self.floors["default"])
        return min(max(self._draw(action, index), floor), self.ceiling_s)

    def wait(self, action: str, already_spent_s: float) -> PacedGap:
        """Sleep for whatever is left of this action's target gap.

        ``already_spent_s`` is everything the app has already seen pass since
        the action landed -- the agent's own thinking, the playback, the
        screenshot.  Subtracting it is the difference between shaping the gap
        and adding to it.
        """

        index = len(self.log)
        target = self.target_for(action, index)
        remaining = target - float(already_spent_s)
        slept = 0.0
        if self.enabled and remaining > 0:
            time.sleep(remaining)
            slept = remaining
        gap = PacedGap(action, target, float(already_spent_s), slept)
        self.log.append(gap)
        return gap

    def summary(self) -> dict[str, Any]:
        if not self.log:
            return {"gaps": 0}
        gaps = [max(g.target_s, g.already_spent_s) for g in self.log]
        slept = [g.slept_s for g in self.log]
        ordered = sorted(gaps)
        return {
            "gaps": len(gaps),
            "median_gap_s": round(ordered[len(ordered) // 2], 3),
            "min_gap_s": round(ordered[0], 3),
            "max_gap_s": round(ordered[-1], 3),
            "total_slept_s": round(sum(slept), 2),
            # How often the agent was already slower than the target, so the
            # policy added nothing: the model, not the policy, set the rhythm.
            # Counted from the times themselves, not from "did not sleep",
            # which is also true when the policy is switched off.
            "already_over_target": sum(
                1 for g in self.log if g.already_spent_s >= g.target_s
            ),
        }


def silence_framework_sleeps(module_names: tuple[str, ...]) -> dict[str, Any]:
    """Neutralise a framework's own fixed waits so the policy owns the gap.

    The framework's constants and the policy would otherwise add up, and the
    sum would be longer and just as regular.  Every suppressed sleep is
    counted, so a run can say how much of its pacing was its own.
    """

    import importlib
    import time as time_module

    suppressed = {"modules": [], "seconds_suppressed": 0.0}

    class _CountingSleep:
        def __init__(self, original):
            self.original = original
            self.total = 0.0

        def __call__(self, seconds):
            self.total += float(seconds)
            suppressed["seconds_suppressed"] += float(seconds)

    for name in module_names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        target = getattr(module, "time", None)
        if target is time_module:
            # The module did `import time`; replacing the attribute on the
            # shared module would affect everything, so wrap per module.
            module.time = _ModuleTimeProxy(time_module, _CountingSleep(time_module.sleep))
            suppressed["modules"].append(name)
        elif callable(getattr(module, "sleep", None)):
            module.sleep = _CountingSleep(time_module.sleep)
            suppressed["modules"].append(name)
    return suppressed


class _ModuleTimeProxy:
    """A stand-in for the ``time`` module with only ``sleep`` replaced."""

    def __init__(self, real, sleep):
        self._real = real
        self.sleep = sleep

    def __getattr__(self, item):
        return getattr(self._real, item)
