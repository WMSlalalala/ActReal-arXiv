"""Model a participant's measured typing speed without retaining entered text.

The donors are stored as one timing track per typing burst. Spatial keyboard
coordinates are not used here; only the intervals between successive contacts
are retained. The framework replays a contiguous slice of these intervals so
that generated typing follows the enrolled participant's pace.

What is drawn here is a *run* rather than a bag of samples.  Consecutive keys
in one burst are correlated -- people speed up mid-word and hesitate before
punctuation -- so taking a contiguous slice of one real burst keeps that
structure, where independent draws from the pooled histogram would flatten it
into noise with the right mean and the wrong autocorrelation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TypingRhythm:
    """One victim's measured inter-key intervals, kept as whole bursts."""

    victim: str
    bursts: list[list[float]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(b) for b in self.bursts)

    def intervals(self, n: int, rng: random.Random) -> list[float]:
        """``n`` consecutive gaps, taken from real bursts.

        Long strings need more gaps than any single burst holds, so bursts are
        chained -- each chosen at random, each entered at a random offset --
        rather than one burst being stretched or repeated.
        """

        if n <= 0 or not self.bursts:
            return []
        out: list[float] = []
        usable = [b for b in self.bursts if b]
        while len(out) < n:
            burst = usable[rng.randrange(len(usable))]
            start = rng.randrange(len(burst))
            out.extend(burst[start:])
        return out[:n]

    def summary(self) -> dict[str, Any]:
        flat = [v for b in self.bursts for v in b]
        flat.sort()
        if not flat:
            return {"victim": self.victim, "intervals": 0}
        def q(p: float) -> float:
            return flat[min(len(flat) - 1, int(len(flat) * p))]
        return {
            "victim": self.victim,
            "bursts": len(self.bursts),
            "intervals": len(flat),
            "median_ms": round(q(0.5), 1),
            "p25_ms": round(q(0.25), 1),
            "p75_ms": round(q(0.75), 1),
        }


def load_rhythm(
    victim: str,
    *,
    root: Optional[Path] = None,
    min_ms: float = 60.0,
    max_ms: float = 2000.0,
) -> Optional[TypingRhythm]:
    """Read a victim's typing rhythm, or ``None`` when it was not shipped.

    The clamp is deliberately wide: it exists to drop the gap that spans a
    pause for thought at one end and a decoding artefact at the other, not to
    tidy the distribution into something prettier than the person.
    """

    base = Path(root) if root else Path(__file__).resolve().parents[1] / "victims"
    path = base / victim / "donors" / "keystroke.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("action") != "keystroke":
        return None

    bursts: list[list[float]] = []
    for track in data.get("tracks", []):
        points = track.get("points") or []
        # The final point is the release of the last key, not a key of its own,
        # so the gaps run between successive presses only.
        gaps = [
            float(points[i + 1]["t_ms"]) - float(points[i]["t_ms"])
            for i in range(max(0, len(points) - 2))
        ]
        gaps = [g for g in gaps if min_ms <= g <= max_ms]
        if gaps:
            bursts.append(gaps)
    if not bursts:
        return None
    return TypingRhythm(victim=victim, bursts=bursts)


def press_durations(victim: str, *, root: Optional[Path] = None) -> list[float]:
    """This victim's measured press durations, for how long each key is held.

    Taken from the tap donors' recorded lengths rather than their retimed ones:
    the bundles were rebuilt onto a 40-240 ms grid to give the planner
    something to choose from, and the grid is not the person.
    """

    base = Path(root) if root else Path(__file__).resolve().parents[1] / "victims"
    path = base / victim / "donors" / "tap.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[float] = []
    for track in data.get("tracks", []):
        points = track.get("points") or []
        if len(points) >= 2:
            held = float(points[-1]["t_ms"]) - float(points[0]["t_ms"])
            if 10.0 <= held <= 500.0:
                out.append(held)
    return out
