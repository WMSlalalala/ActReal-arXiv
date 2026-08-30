"""The stream that plays when no action is playing.

Switching the bus to INJECTED unregisters the phone's own sensors, so unless
something else supplies frames the target receives nothing at all between
actions.  On a real agent run that was measured as nine gaps over 200 ms and one
of **257 seconds** -- four minutes in which the recording says the phone was not
merely still but absent, against a human session that is continuous at 100 Hz.
An agent spends most of its wall time thinking, so most of its recording is this
stream; getting the gestures right and leaving this empty would be solving the
easy half.

What plays is not synthesised stillness and not zeros.  Every cached window is
longer than the gesture inside it: the frames before and after the active span
are the same hand, recorded, on its way to the glass and away from it.  Those
are stitched together here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .bundle import ActionBundle


def padding_frames(bundle: ActionBundle) -> tuple[list[list[float]], list[list[float]]]:
    """The lead-in and lead-out of one window, without the gesture between."""

    rows = bundle.imu_rows()
    period = bundle.imu_period_ms
    if period <= 0:
        return [], []
    pre_end = max(0, min(len(rows), int(round(bundle.touch_offset_ms / period))))
    active = int(bundle.provenance.get("imu_active_frames") or 0)
    post_start = max(pre_end, min(len(rows), pre_end + active))
    return rows[:pre_end], rows[post_start:]


def build(
    bundles: Sequence[ActionBundle],
    *,
    period_ms: float = 10.0,
    seconds: float = 12.0,
) -> list[list[float]]:
    """Stitch padding from several windows into one loop.

    Several, not one: a two-second loop repeated for a minute is a periodicity
    no hand has, and a session-level check finds it without knowing anything
    about gestures.  Only windows already on the target grid are used -- a
    keystroke window capped to 512 frames carries a 30 ms period, and splicing
    it in would play that hand at a third of its real speed.
    """

    wanted = int(round(seconds * 1000.0 / period_ms))
    out: list[list[float]] = []
    for bundle in bundles:
        if abs(bundle.imu_period_ms - period_ms) > 1e-9:
            continue
        pre, post = padding_frames(bundle)
        # Lead-out then lead-in, so consecutive pieces join hand-leaving to
        # hand-arriving rather than cutting from mid-approach to mid-approach.
        out.extend(post)
        out.extend(pre)
        if len(out) >= wanted:
            break
    return out[:wanted]


def install(
    imu_backend,
    library,
    *,
    period_ms: float = 10.0,
    seconds: float = 12.0,
    actions: Sequence[str] = ("scroll", "swipe", "tap"),
) -> dict[str, Any]:
    """Build a background from a bundle library and hand it to the backend."""

    chosen: list[ActionBundle] = []
    for action in actions:
        for index in range(min(4, library.count(action))):
            try:
                chosen.append(library.pick(action, avoid=None))
            except KeyError:
                break
    frames = build(chosen, period_ms=period_ms, seconds=seconds)
    if not frames:
        return {"frames": 0, "reason": "no window on the target grid had padding"}
    imu_backend.set_background(frames, period_ms)
    return {
        "frames": len(frames),
        "seconds": round(len(frames) * period_ms / 1000.0, 2),
        "period_ms": period_ms,
        "sources": len(chosen),
    }
