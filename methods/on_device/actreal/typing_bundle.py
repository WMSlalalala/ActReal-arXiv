"""Compose a whole typed string into one playable action.

Dispatching a string key by key works -- each press is a real tap bundle and
the gaps are the victim's own -- but it leaves the schedule to Python: fourteen
``sleep`` calls, each carrying whatever the interpreter and the adb round trip
happened to cost. That jitter is small against a 460 ms gap, and it is also
avoidable, because the injection path already accepts a whole gesture with its
own millisecond clock and hands it to the kernel.

So typing is compiled instead. One bundle carries every press: contact *i* goes
down at the sum of the victim's measured gaps before it and lifts after one of
their measured hold times, and the inertia underneath is their own recorded
press windows laid so that each window's active part sits on the press it
belongs to.

That last part is the reason this is not simply a concatenation. A recording
where the accelerometer jumps at moments the digitiser reports nothing -- or
worse, reports a contact 200 ms later -- fails the one check that costs a
detector nothing: line the two channels up and see whether they agree. Laying
the windows end to end would produce exactly that, because a press window is
350 ms long and this person's keys are 460 ms apart. Here each window is placed
by its onset, and the uncovered stretches between presses are filled from the
quiet tails of the same windows -- a hand between keys, which is what those
frames were recorded from.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional, Sequence

from .bundle import ActionBundle
from .touch_track import TouchPoint, TouchTrack

# Where a tap window's pre-roll ends and the press begins.  Every tap bundle in
# the library is baked with this lead-in, and it is asserted rather than assumed
# so a rebaked library cannot silently misalign the two channels.
PRE_ROLL_MS = 100.0


def _quiet_rows(bundle: ActionBundle, active: int) -> list[list[float]]:
    """The frames of a press window where the finger was not on the glass."""

    rows = [list(map(float, row)) for row in bundle.imu]
    lead = int(round(PRE_ROLL_MS / bundle.imu_period_ms))
    tail = rows[lead + active:]
    return tail or rows[:lead] or rows


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def _best_entry(previous: Optional[Sequence[float]], pool: Sequence[Sequence[float]]) -> int:
    """Where to start reading a segment so the seam is as small as it can be.

    Splicing two recordings end to end puts a step at the join that no hand
    produces: measured on the keystroke windows this project already shipped,
    the frame-to-frame change reaches 11.4 where a continuous recording of the
    same person never exceeds about 2.2.  One threshold separates them, so the
    seam is the artefact, not the content.

    Nothing here rewrites a sample.  The only freedom taken is *where to cut* --
    the entry point whose first frame is nearest to the frame already written --
    so every value stays exactly as it was recorded and the join lands where the
    material happens to line up.
    """

    if not pool:
        return 0
    if previous is None:
        return 0
    best, best_at = None, 0
    for index, row in enumerate(pool):
        gap = _distance(previous, row)
        if best is None or gap < best:
            best, best_at = gap, index
    return best_at


def compose_typing(
    *,
    sequence: Sequence[tuple[str, tuple[float, float]]],
    gaps_ms: Sequence[float],
    holds_ms: Sequence[float],
    press_bundles: Sequence[ActionBundle],
    rng: random.Random,
    mapping: Any,
) -> Optional[ActionBundle]:
    """One bundle for the whole string, or ``None`` if it cannot be built.

    ``sequence`` is one entry per contact -- a shift counts as its own -- and
    ``gaps_ms`` is the interval *before* each contact after the first.
    """

    if not sequence or not press_bundles:
        return None
    if len(gaps_ms) < max(0, len(sequence) - 1) or len(holds_ms) < len(sequence):
        return None

    period = float(press_bundles[0].imu_period_ms)
    if period <= 0:
        return None

    # -- when each key goes down and comes up ------------------------------
    points: list[TouchPoint] = []
    presses: list[tuple[float, ActionBundle]] = []
    t = 0.0
    for index, (_label, (x, y)) in enumerate(sequence):
        if index:
            t += max(0.0, float(gaps_ms[index - 1]))
        held = max(10.0, float(holds_ms[index]))
        donor = press_bundles[rng.randrange(len(press_bundles))]
        presses.append((t, donor))
        points.append(TouchPoint(t_ms=t, x=float(x), y=float(y), pressure=1.0,
                                 size=0.0, pointer_id=0, action="DOWN"))
        points.append(TouchPoint(t_ms=t + held, x=float(x), y=float(y), pressure=1.0,
                                 size=0.0, pointer_id=0, action="UP"))
        t += held

    typing_ms = t
    lead = int(round(PRE_ROLL_MS / period))
    total_frames = lead + int(round(typing_ms / period)) + lead
    if total_frames <= 0:
        return None

    # -- inertia, aligned press by press -----------------------------------
    # Kept as separate runs rather than one concatenated pool.  Joining the
    # tails of different windows into a single array and then slicing it looks
    # like reading continuous material, but every donor boundary inside it is a
    # seam nobody chose -- which is how a fixed 9.7 step survived the first
    # attempt at seam selection: the step was inside the pool, not at its edges.
    quiet_runs: list[list[list[float]]] = []
    for donor in press_bundles:
        active = int(donor.provenance.get("imu_active_frames") or 0)
        run = _quiet_rows(donor, active)
        if run:
            quiet_runs.append([list(map(float, row)) for row in run])
    if not quiet_runs:
        return None

    # The window is written in time order rather than overlaid, so every join
    # is made once, against the frame that actually precedes it.  Press windows
    # are placed by their onset -- a press bump has to sit on the contact it
    # came from, or the two channels disagree about when the finger landed --
    # and the stretches between them are filled from the quiet frames, entered
    # wherever they continue most smoothly from what is already there.
    frames: list[list[float]] = []

    def fill(count: int, follows: Optional[Sequence[float]]) -> None:
        """Add ``count`` quiet frames, joined at both ends as well as possible.

        A press window cannot be moved -- its bump has to land on its contact --
        so the only seams that can be placed are the ones on either side of the
        quiet stretch between two presses.  Both are scored: the entry against
        the frame already written, and the exit against the frame the next press
        will start with.  Every sample is still exactly as recorded; what is
        chosen is where in the material to cut.
        """

        while count > 0:
            previous = frames[-1] if frames else None
            best: Optional[tuple[float, list[list[float]]]] = None
            for run in quiet_runs:
                take = min(count, len(run))
                # Slices stay inside one recording, so the only joins are the
                # two this loop is choosing.
                for at in range(len(run) - take + 1):
                    piece = run[at:at + take]
                    cost = 0.0
                    if previous is not None:
                        cost += _distance(previous, piece[0])
                    if follows is not None and take == count:
                        cost += _distance(piece[-1], follows)
                    if best is None or cost < best[0]:
                        best = (cost, piece)
            if best is None:
                return
            frames.extend([list(row) for row in best[1]])
            count -= len(best[1])

    for index, (start_ms, _chosen) in enumerate(presses):
        onset = lead + int(round(start_ms / period))
        begin = onset - int(round(PRE_ROLL_MS / period))
        # Which recorded press to use is itself a choice, so it is made on the
        # same criterion: the one whose window opens closest to where the
        # signal already is.
        tail = frames[-1] if frames else None
        donor = min(
            press_bundles,
            key=lambda b: _distance(tail, b.imu[0]) if tail is not None else 0.0,
        )
        rows = [list(map(float, row)) for row in donor.imu]
        if begin > len(frames):
            fill(begin - len(frames), rows[0])
        elif begin < len(frames):
            # The previous window still had frames left; this press starts
            # inside it, so the overlap is dropped rather than written twice.
            rows = rows[len(frames) - begin:]
        for row in rows:
            if len(frames) >= total_frames:
                break
            frames.append(row)
    fill(total_frames - len(frames), None)
    frames = frames[:total_frames]

    track = TouchTrack(
        action="keystroke",
        points=points,
        orientation_id=0,
        screen_w=mapping.device_w,
        screen_h=mapping.device_h,
        source="composed:per-key contacts on this device's keys",
    )

    digest = hashlib.sha1(
        ("|".join(f"{p.t_ms:.1f},{p.x:.1f},{p.y:.1f},{p.action}" for p in points)).encode()
    ).hexdigest()[:16]

    return ActionBundle(
        bundle_id=digest,
        action="keystroke",
        touch=track,
        imu=frames,
        imu_period_ms=period,
        touch_offset_ms=PRE_ROLL_MS,
        mapping=mapping,
        provenance={
            "composed": True,
            "touch_source": "device keymap + victim inter-key intervals",
            "imu_source": f"{len(press_bundles)} of this victim's tap windows, "
                          "one per press, aligned to its contact",
            "contacts": len(sequence),
            "typing_ms": round(typing_ms, 1),
            "orientation_id": 0,
            "note": "key coordinates are this phone's; timing, hold and inertia "
                    "are the victim's own measurements",
        },
    )


def describe(bundle: Optional[ActionBundle]) -> dict[str, Any]:
    if bundle is None:
        return {"composed": False}
    return {
        "composed": True,
        "contacts": len(bundle.touch.points) // 2,
        "typing_ms": bundle.provenance.get("typing_ms"),
        "imu_frames": bundle.imu_frames,
    }
