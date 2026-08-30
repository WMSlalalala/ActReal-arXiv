"""How long one action lasts.

The split matters here, so it is stated first.  Seventy users are the model's
training population; twenty are held out and one of them is the victim.  The
attacker may know what the training population does -- that is what the model
was fitted on -- and may hold five recordings of the victim.  Nothing else.  No
statistic over the held-out population, and none over the victim's genuine
behaviour beyond those five.

**What the training population says.** The obvious model is a travel-to-duration
curve: the agent names two points, and the distance says how long the gesture
takes.  Measured on the 70 training users alone, that relationship is not
there.  Across 39,094 training scrolls the log correlation between travel and
duration is -0.055, and within a single training user it is -0.086 at the
median; swipes give +0.102 over 47,852 events and -0.107 within a user.  People
cover more ground by moving faster, not by taking longer.

So distance carries no usable information about duration, and fitting a curve
through five points of it fits noise -- the same 1,200 px request read off one
five-shot curve gave 160 ms and off another 3,250 ms.

**What is used instead.** The centre comes from the victim's own five
recordings; the scatter around it comes from how much a training user varies
within themselves (median log spread 0.680 for scroll, 0.760 for swipe).  The
victim's five give a median that is theirs; five points give a spread that is
mostly sampling noise, so the scale is the population's and the location is the
victim's.

Evaluation is pure Python so the packaged local build keeps needing nothing
installed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

# Each action's inertia window is a fixed number of 100 Hz frames, so a gesture
# longer than its window cannot be generated at all.
WINDOW_FRAMES = {"tap": 35, "scroll": 179, "swipe": 167, "pinch": 116, "keystroke": 256}
GRID_PERIOD_MS = 10.0

# Median within-user log-duration spread, measured on the 70 training users
# only.  This is population structure the attacker is allowed to know; it sets
# how far a drawn duration wanders from the victim's own centre.
TRAIN_LOG_SPREAD = {
    "tap": 0.62,
    "scroll": 0.680,
    "swipe": 0.760,
    "pinch": 0.62,
    "keystroke": 0.80,
}

# How much of that spread to actually use.  A full draw at the population's
# spread would make one in twenty gestures three times the victim's median,
# which the window cannot hold and a person rarely does twice in a row.
SPREAD_FRACTION = 0.5


class DurationLawError(ValueError):
    pass


def window_cap_ms(action: str) -> Optional[float]:
    frames = WINDOW_FRAMES.get(action)
    return None if frames is None else frames * GRID_PERIOD_MS


def _unit_normal(seed_text: str) -> float:
    """A deterministic standard normal from a label.

    Deterministic so a session can be replayed exactly from its log, and
    hashed rather than sequential so consecutive actions do not walk a pattern.
    """

    digest = hashlib.sha256(seed_text.encode()).digest()
    # Two independent uniforms out of the digest, then Box-Muller.
    a = int.from_bytes(digest[0:8], "big") / float(1 << 64)
    b = int.from_bytes(digest[8:16], "big") / float(1 << 64)
    a = min(max(a, 1e-12), 1.0 - 1e-12)
    return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)


@dataclass(frozen=True)
class ActionTiming:
    """One victim's timing for one action: their centre, the population's scatter."""

    action: str
    victim: str
    durations_ms: tuple[float, ...]
    source_event_ids: tuple[str, ...] = ()
    log_spread: float = 0.0
    spread_source: str = "train_population"

    def __post_init__(self) -> None:
        if not self.durations_ms:
            raise DurationLawError(f"{self.victim}/{self.action}: no recorded durations")
        if any(not math.isfinite(d) or d <= 0 for d in self.durations_ms):
            raise DurationLawError(f"{self.victim}/{self.action}: a duration is not positive")
        if self.log_spread < 0:
            raise DurationLawError(f"{self.victim}/{self.action}: spread must not be negative")

    @property
    def count(self) -> int:
        return len(self.durations_ms)

    @property
    def median_ms(self) -> float:
        ordered = sorted(self.durations_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @property
    def span_ms(self) -> tuple[float, float]:
        return (min(self.durations_ms), max(self.durations_ms))

    @property
    def material_log_spread(self) -> float:
        """The scatter of the five recordings themselves, for reporting only.

        Five points measure a spread badly, which is why the draw does not use
        this number -- but it is worth being able to see it.
        """

        if self.count < 2:
            return 0.0
        logs = [math.log(d) for d in self.durations_ms]
        mean = sum(logs) / len(logs)
        return math.sqrt(sum((v - mean) ** 2 for v in logs) / (len(logs) - 1))

    def fits_window(self) -> bool:
        cap = window_cap_ms(self.action)
        return cap is None or max(self.durations_ms) <= cap

    def duration_ms(self, draw: int | str) -> float:
        """The victim's centre, moved by a draw at the population's scale.

        Clamped to the action's window: a duration the generator cannot make a
        window for is not a duration this system can deliver.
        """

        z = _unit_normal(f"{self.victim}|{self.action}|{draw}")
        value = self.median_ms * math.exp(self.log_spread * z)
        cap = window_cap_ms(self.action)
        floor = GRID_PERIOD_MS * 2
        if cap is not None:
            value = min(value, cap)
        return float(max(value, floor))

    def speed_px_per_ms(self, travel_px: float, draw: int | str) -> float:
        return float(travel_px) / self.duration_ms(draw)

    def as_dict(self) -> dict[str, Any]:
        low, high = self.span_ms
        return {
            "action": self.action,
            "victim": self.victim,
            "durations_ms": [round(d, 3) for d in self.durations_ms],
            "source_event_ids": list(self.source_event_ids),
            "count": self.count,
            "median_ms": round(self.median_ms, 2),
            "span_ms": [round(low, 2), round(high, 2)],
            "log_spread": round(self.log_spread, 4),
            "spread_source": self.spread_source,
            "material_log_spread": round(self.material_log_spread, 4),
            "window_cap_ms": window_cap_ms(self.action),
            "fits_window": self.fits_window(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionTiming":
        return cls(
            action=str(data["action"]),
            victim=str(data["victim"]),
            durations_ms=tuple(float(v) for v in data["durations_ms"]),
            source_event_ids=tuple(str(v) for v in data.get("source_event_ids", ())),
            log_spread=float(data.get("log_spread", 0.0)),
            spread_source=str(data.get("spread_source", "train_population")),
        )


def build_timing(
    durations: Iterable[float],
    *,
    action: str,
    victim: str,
    source_event_ids: Sequence[str] = (),
    spread_fraction: float = SPREAD_FRACTION,
    train_log_spread: Optional[dict[str, float]] = None,
) -> ActionTiming:
    usable = [float(d) for d in durations if math.isfinite(float(d)) and float(d) > 0]
    if not usable:
        raise DurationLawError(f"{victim}/{action}: the material carries no usable duration")
    table = train_log_spread or TRAIN_LOG_SPREAD
    spread = table.get(action, 0.6) * float(spread_fraction)
    return ActionTiming(
        action=action,
        victim=victim,
        durations_ms=tuple(usable),
        source_event_ids=tuple(str(v) for v in source_event_ids),
        log_spread=spread,
        spread_source=f"train_population x{spread_fraction:g}",
    )


class TimingBook:
    """One victim's timings, one per action."""

    def __init__(self, timings: dict[str, ActionTiming], victim: str = ""):
        self.timings = timings
        self.victim = victim or (next(iter(timings.values())).victim if timings else "")

    @property
    def actions(self) -> list[str]:
        return sorted(self.timings)

    def get(self, action: str) -> Optional[ActionTiming]:
        return self.timings.get(action)

    def duration_ms(self, action: str, draw: int | str) -> Optional[float]:
        timing = self.timings.get(action)
        return None if timing is None else timing.duration_ms(draw)

    def unfit_actions(self) -> list[str]:
        return sorted(a for a, t in self.timings.items() if not t.fits_window())

    def as_dict(self) -> dict[str, Any]:
        return {action: timing.as_dict() for action, timing in self.timings.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingBook":
        return cls({a: ActionTiming.from_dict(v) for a, v in data.items()})
