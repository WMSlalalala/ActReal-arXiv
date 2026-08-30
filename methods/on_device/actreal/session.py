"""Play one compiled action on the device, then check what arrived.

The order matters and is not arbitrary: the IMU window is scheduled first and
starts before the touch, because the leading frames are the hand already moving
towards the glass.  Both streams are pinned to one instant ``t0`` on the
MotionEvent clock, and the app converts that once, on the device, using an
offset it measured itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import time

from .bundle import ActionBundle
from .clock import HostAnchor, LaunchCalibration, sleep_until_host
from .control import ControlClient


@dataclass
class PlayReceipt:
    bundle_id: str
    action: str
    t0_uptime_ms: int
    imu_frames: int
    touch_points: int
    imu_start_elapsed_ns: int
    lead_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "action": self.action,
            "t0_uptime_ms": self.t0_uptime_ms,
            "imu_frames": self.imu_frames,
            "touch_points": self.touch_points,
            "imu_start_elapsed_ns": self.imu_start_elapsed_ns,
            "lead_ms": self.lead_ms,
        }


def play_bundle(
    client: ControlClient,
    bundle: ActionBundle,
    *,
    lead_ms: float = 300.0,
    set_mode: bool = True,
) -> PlayReceipt:
    """Schedule both streams of one action against a shared start instant."""

    if set_mode:
        client.set_imu_mode("injected")

    now = client.ping()
    t0 = int(now["uptime_ms"] + round(lead_ms))

    imu_reply = client.schedule_imu(
        bundle.imu_rows(),
        start_uptime_ms=t0,
        period_ms=bundle.imu_period_ms,
        bundle_id=bundle.bundle_id,
    )
    points = bundle.touch_points_on_bundle_clock()
    client.play_touch(points, start_uptime_ms=t0)

    return PlayReceipt(
        bundle_id=bundle.bundle_id,
        action=bundle.action,
        t0_uptime_ms=t0,
        imu_frames=bundle.imu_frames,
        touch_points=len(points),
        imu_start_elapsed_ns=int(imu_reply["start_elapsed_ns"]),
        lead_ms=lead_ms,
    )


# Column positions in the app's touch and IMU dump rows.
TOUCH_UPTIME_NS, TOUCH_ACTION, TOUCH_BATCHED = 1, 3, 4
TOUCH_X, TOUCH_Y, TOUCH_PRESSURE = 8, 9, 10
# Which finger a row belongs to.  Without it a two-contact gesture is matched
# as though every row came from one hand, and a planned point on one finger
# pairs with a recorded row on the other -- an error the width of the pinch,
# measured at 467px on a gesture that in fact landed within a pixel.
#
# Read off the recording, not off the Row constructor's field order: the
# declaration lists pointerId immediately before x, which would put it at 6,
# and field 6 is zero on every row of a two-finger gesture while field 7 splits
# 121/102 exactly along the two planned trajectories.
TOUCH_POINTER_ID = 7
IMU_TIMESTAMP_NS, IMU_ORIGIN, IMU_BUNDLE, IMU_FRAME = 1, 8, 9, 10


@dataclass
class StreamCheck:
    planned: int
    matched: int
    missing: int
    extra: int
    max_value_error: float
    max_time_error_ms: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.missing == 0 and self.matched == self.planned


@dataclass
class VerifyReport:
    bundle_id: str
    action: str
    touch: StreamCheck
    imu: StreamCheck
    alignment_error_ms: Optional[float]
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.touch.ok and self.imu.ok and not self.notes

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "action": self.action,
            "ok": self.ok,
            "touch": {
                "planned": self.touch.planned,
                "matched": self.touch.matched,
                "missing": self.touch.missing,
                "extra": self.touch.extra,
                "max_xy_error_px": round(self.touch.max_value_error, 6),
                "max_time_error_ms": round(self.touch.max_time_error_ms, 6),
                **self.touch.detail,
            },
            "imu": {
                "planned": self.imu.planned,
                "matched": self.imu.matched,
                "missing": self.imu.missing,
                "extra": self.imu.extra,
                "max_value_error": round(self.imu.max_value_error, 9),
                "max_time_error_ms": round(self.imu.max_time_error_ms, 6),
                **self.imu.detail,
            },
            "alignment_error_ms": (
                None if self.alignment_error_ms is None else round(self.alignment_error_ms, 6)
            ),
            "notes": self.notes,
        }


def verify(
    dump: dict[str, Any],
    bundle: ActionBundle,
    receipt: PlayReceipt,
    *,
    touch_tolerance_ms: float = 6.0,
    imu_tolerance_ms: float = 6.0,
) -> VerifyReport:
    """Compare what the app recorded against what was planned."""

    notes: list[str] = []
    clock_offset_ns = int(dump.get("clock_offset_ns", 0))
    t0_ns = receipt.t0_uptime_ms * 1_000_000

    # Batched rows are historical samples inside a coalesced MotionEvent, and
    # they count as delivered.  Dropping them was right when every backend
    # dispatched one event per planned point, and wrong the moment a real input
    # pipeline was involved: Android coalesces rapid MOVEs, so on a Pixel 10 a
    # thirteen-point scroll arrived as two or three events carrying eleven
    # historical samples between them.  Discarding those reported five of
    # thirteen points delivered for a gesture that had in fact arrived whole,
    # and then mispaired the survivors, which is where the hundred-pixel
    # coordinate errors came from.
    all_rows = sorted(dump.get("touch", []), key=lambda row: row[TOUCH_UPTIME_NS])
    batched = sum(1 for row in all_rows if row[TOUCH_BATCHED])

    # Android resamples touch toward the display's frame boundaries, so a real
    # input pipeline delivers samples nobody sent: on a Pixel 10 a thirteen-point
    # scroll came back with six extra MOVEs spaced 16.6 ms apart, at interpolated
    # positions.  Those are the framework's, not ours, and pairing a planned
    # point against one of them is what produced the large coordinate errors.
    #
    # Within a coalesced MotionEvent the historical samples are the raw ones and
    # only the current sample can be resampled, so where coalescing happened the
    # comparison uses the historical samples plus the boundaries, which
    # resampling never moves.  Where it did not happen -- the in-app backend
    # delivers one event per point -- every row is raw and all of them are used.
    # Everything recorded is a candidate; which of them is *ours* is decided by
    # the matcher, which searches within the time tolerance and takes the
    # closest sample rather than merely the soonest.  Filtering by the batched
    # flag instead was nearly right and quietly wrong: a raw MOVE that happened
    # to be the current sample of its event got thrown away with the resampled
    # ones, and its planned point was then reported as never delivered.
    touch_rows = all_rows
    resampled = 0
    planned_points = bundle.touch_points_on_bundle_clock()

    # One stream per finger.  A pinch puts two contacts on the glass at the
    # same instants, so a matcher that searches by time alone has two candidate
    # rows for every planned point and no way to tell them apart; it picks the
    # closer in value, which for two fingers travelling in opposite directions
    # is the wrong one about half the time.  Splitting first makes the search
    # unambiguous and costs nothing for the single-contact actions, where every
    # point carries pointer 0 and the split is a no-op.
    def _pointer_of(point: dict[str, Any]) -> int:
        return int(point.get("pointer_id", 0))

    planned_by_pointer: dict[int, list[Any]] = {}
    for point in planned_points:
        planned_by_pointer.setdefault(_pointer_of(point), []).append(
            (point["t_ms"], (float(point["x"]), float(point["y"]), float(point["pressure"])))
        )
    recorded_by_pointer: dict[int, list[Any]] = {}
    for row in touch_rows:
        pointer = int(row[TOUCH_POINTER_ID]) if len(row) > TOUCH_POINTER_ID else 0
        recorded_by_pointer.setdefault(pointer, []).append(
            (
                (row[TOUCH_UPTIME_NS] - t0_ns) / 1e6,
                (float(row[TOUCH_X]), float(row[TOUCH_Y]), float(row[TOUCH_PRESSURE])),
            )
        )

    per_pointer = {
        pointer: _match_stream(
            planned=points,
            recorded=recorded_by_pointer.get(pointer, []),
            tolerance_ms=touch_tolerance_ms,
            # Pressure shares the tuple but is not a pixel; compared separately.
            value_slice=slice(0, 2),
        )
        for pointer, points in sorted(planned_by_pointer.items())
    }
    touch = _combine_matches(per_pointer)
    touch.detail["batched_rows"] = batched
    # Reported, not hidden: it is a property of the path the touch travelled,
    # and a run with none of it did not go through the same pipeline as one with.
    touch.detail["resampled_rows"] = resampled
    touch.detail["max_pressure_error"] = round(
        _max_component_error(planned_points, touch_rows, "pressure", TOUCH_PRESSURE), 6
    )

    # The IMU stream carries the background between actions; only this bundle's
    # frames are the subject of the check.
    imu_rows = [
        row
        for row in dump.get("imu", [])
        if row[IMU_ORIGIN] == "injected" and row[IMU_BUNDLE] == bundle.bundle_id
    ]
    imu_start_ns = receipt.imu_start_elapsed_ns
    imu = _match_stream(
        planned=[
            (index * bundle.imu_period_ms, tuple(frame))
            for index, frame in enumerate(bundle.imu_rows())
        ],
        recorded=[
            (
                (row[IMU_TIMESTAMP_NS] - imu_start_ns) / 1e6,
                tuple(float(v) for v in row[2:8]),
            )
            for row in imu_rows
        ],
        tolerance_ms=imu_tolerance_ms,
        value_slice=slice(0, 6),
    )

    alignment = _alignment_error_ms(dump, bundle, receipt, clock_offset_ns)
    if alignment is None:
        notes.append("no DOWN row recorded, alignment not measurable")

    if dump.get("touch_dropped"):
        notes.append(f"app dropped {dump['touch_dropped']} touch rows")
    if dump.get("imu_dropped"):
        notes.append(f"app dropped {dump['imu_dropped']} imu rows")

    return VerifyReport(
        bundle_id=bundle.bundle_id,
        action=bundle.action,
        touch=touch,
        imu=imu,
        alignment_error_ms=alignment,
        notes=notes,
    )


def _combine_matches(per_pointer: dict[int, StreamCheck]) -> StreamCheck:
    """Fold one check per finger back into one check for the gesture.

    Counts add; errors take the worst, because a gesture is only as well
    delivered as its worse contact. The per-finger numbers are kept in the
    detail rather than averaged away -- a pinch whose first finger landed
    perfectly and whose second never arrived is a specific failure, and a
    single blended figure would describe it as a mediocre success.
    """

    if not per_pointer:
        return StreamCheck(0, 0, 0, 0, 0.0, 0.0, {"pointers": 0})
    if len(per_pointer) == 1:
        only = next(iter(per_pointer.values()))
        only.detail["pointers"] = 1
        return only
    return StreamCheck(
        planned=sum(c.planned for c in per_pointer.values()),
        matched=sum(c.matched for c in per_pointer.values()),
        missing=sum(c.missing for c in per_pointer.values()),
        extra=sum(c.extra for c in per_pointer.values()),
        max_value_error=max(c.max_value_error for c in per_pointer.values()),
        max_time_error_ms=max(c.max_time_error_ms for c in per_pointer.values()),
        detail={
            "pointers": len(per_pointer),
            "by_pointer": {
                str(pointer): {
                    "planned": c.planned,
                    "matched": c.matched,
                    "max_value_error": round(c.max_value_error, 4),
                }
                for pointer, c in sorted(per_pointer.items())
            },
        },
    )


def _match_stream(
    *,
    planned: list[tuple[float, tuple[float, ...]]],
    recorded: list[tuple[float, tuple[float, ...]]],
    tolerance_ms: float,
    value_slice: slice,
) -> StreamCheck:
    """Pair each planned sample with the recorded one it actually is.

    Not by index: a real input pipeline coalesces and reorders, and a positional
    comparison would report a single dropped event as every later event being
    wrong.  Not by time alone either: Android's resampler inserts synthesised
    samples at frame boundaries, and one of those is often nearer in time to a
    planned point than the planned point's own arrival is -- which is how a
    scroll whose every sample arrived to the pixel came back reading 166 px out.

    So the window is temporal and the choice inside it is positional: among the
    unclaimed rows within ``tolerance_ms``, take the closest one in value.  The
    question being asked is "did this sample arrive, and when", and rows that
    are nobody's planned sample stay unclaimed and are counted as extra.
    """

    used: set[int] = set()
    matched = 0
    max_value = 0.0
    max_time = 0.0
    for t_planned, values in planned:
        best: Optional[int] = None
        best_distance = float("inf")
        best_dt = 0.0
        for index, (t_rec, rec_values) in enumerate(recorded):
            if index in used:
                continue
            dt = abs(t_rec - t_planned)
            if dt > tolerance_ms:
                continue
            distance = sum(
                (a - b) ** 2 for a, b in zip(values[value_slice], rec_values[value_slice])
            )
            if distance < best_distance:
                best, best_distance, best_dt = index, distance, dt
        if best is None:
            continue
        used.add(best)
        matched += 1
        max_time = max(max_time, best_dt)
        rec_values = recorded[best][1]
        for a, b in zip(values[value_slice], rec_values[value_slice]):
            max_value = max(max_value, abs(a - b))
    return StreamCheck(
        planned=len(planned),
        matched=matched,
        missing=len(planned) - matched,
        extra=len(recorded) - len(used),
        max_value_error=max_value,
        max_time_error_ms=max_time,
    )


def _max_component_error(
    planned_points: list[dict[str, Any]],
    touch_rows: list[list[Any]],
    key: str,
    column: int,
) -> float:
    if len(planned_points) != len(touch_rows):
        return float("nan")
    return max(
        (abs(float(p[key]) - float(row[column])) for p, row in zip(planned_points, touch_rows)),
        default=0.0,
    )


def _alignment_error_ms(
    dump: dict[str, Any],
    bundle: ActionBundle,
    receipt: PlayReceipt,
    clock_offset_ns: int,
) -> Optional[float]:
    """How far the first contact sits from the frame the plan put it on.

    This is the number the whole design exists to keep small: the detectors
    window the inertia against the touch boundary, so a touch that lands a few
    frames away from its own IMU is a touch with somebody else's inertia.
    """

    downs = [
        row
        for row in dump.get("touch", [])
        if str(row[TOUCH_ACTION]).endswith("DOWN") and not row[TOUCH_BATCHED]
    ]
    if not downs:
        return None
    down_elapsed_ns = downs[0][TOUCH_UPTIME_NS] + clock_offset_ns
    planned_active_ns = receipt.imu_start_elapsed_ns + int(
        round(bundle.touch_offset_ms * 1_000_000)
    )
    return (down_elapsed_ns - planned_active_ns) / 1e6


# ---------------------------------------------------------------------------
# One action, either configuration
# ---------------------------------------------------------------------------


@dataclass
class ActionReceipt:
    """What a configuration actually committed to for one action.

    Field names overlap :class:`PlayReceipt` where they mean the same thing, so
    :func:`verify` reads either without caring which configuration produced it.
    That matters more than it looks: the comparison against the plan is the one
    piece of this system that must not have a per-configuration branch, or the
    two would be graded by different rulers.
    """

    bundle_id: str
    action: str
    configuration: str
    t0_uptime_ms: int
    imu_frames: int
    touch_points: int
    imu_start_elapsed_ns: int
    lead_ms: float
    imu: dict[str, Any] = field(default_factory=dict)
    touch: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "action": self.action,
            "configuration": self.configuration,
            "t0_uptime_ms": self.t0_uptime_ms,
            "imu_frames": self.imu_frames,
            "touch_points": self.touch_points,
            "imu_start_elapsed_ns": self.imu_start_elapsed_ns,
            "lead_ms": self.lead_ms,
            "imu": self.imu,
            "touch": self.touch,
            "notes": self.notes,
        }


def play_action(
    config,
    bundle: ActionBundle,
    *,
    read_uptime_ms,
    lead_ms: float = 400.0,
    set_mode: bool = True,
    launch_timeout: float = 60.0,
) -> ActionReceipt:
    """Schedule both streams of one action against a single start instant.

    The order is the whole point and it is not arbitrary.

    1. Anything expensive happens *before* the clock starts.  Transferring a
       compiled gesture to the device costs a few hundred milliseconds and the
       cost depends on how long the gesture is; leaving that inside the timed
       region would make the touch land later for longer gestures, which is a
       correlation between duration and delay that no hand has.
    2. ``t0`` is chosen on the touch clock and converted once.
    3. The inertia is scheduled first, because it starts first: the pre-roll is
       the hand already moving towards the glass, and delivering the contact
       before the approach inverts the causality the method rests on.
    4. The gesture is launched against a deadline derived from ``t0``, corrected
       by whatever launch latency was measured for this backend, and how far
       past that deadline control actually resumed is recorded rather than
       assumed to be zero.

    ``read_uptime_ms`` is how this configuration asks the device what time it
    is -- the target app's control channel in one, the hook's RPC in the other.
    """

    notes: list[str] = []
    staged = None
    if hasattr(config.touch, "stage"):
        staged = config.touch.stage(bundle)

    if set_mode:
        config.imu.set_mode("injected")

    anchor = HostAnchor.measure(read_uptime_ms)
    t0 = int(round(anchor.device_uptime_now() + lead_ms))

    imu_schedule = config.imu.schedule(
        bundle, t0_uptime_ms=t0, timebase=config.timebase
    )
    late_by = anchor.device_uptime_now() - t0
    if late_by > 0:
        notes.append(
            f"the inertial stream was scheduled {late_by:.1f} ms after its own "
            f"start instant; raise lead_ms"
        )

    if staged is not None:
        # The compiled stream begins at its own first sample, which belongs at
        # the end of the pre-roll rather than at t0.  touch_start_ms rather than
        # touch_offset_ms: the two are equal for every bundle baked so far, but
        # a track that has been shifted carries the difference, and a silent
        # off-by-a-pre-roll is the one error here that still looks like a
        # working run.
        target_uptime_ms = t0 + bundle.touch_start_ms - config.launch.latency_ms
        overshoot_ms = (
            sleep_until_host(anchor.host_deadline_for(target_uptime_ms)) * 1000.0
        )
        touch_reply = config.touch.launch(staged, timeout=launch_timeout)
        touch_reply.update(
            {
                "launch_target_uptime_ms": round(target_uptime_ms, 3),
                "launch_overshoot_ms": round(overshoot_ms, 3),
                "launch_correction_ms": round(config.launch.latency_ms, 3),
                "honoured_start": True,
            }
        )
        if overshoot_ms > 5.0:
            # The deadline had already gone by the time the scheduler reached
            # it, so the gesture did not start when it was aimed to.  Almost
            # always the correction is larger than the lead: subtracting a
            # 600 ms launch latency from a 400 ms lead puts the target in the
            # past before anything is written.
            notes.append(
                f"the gesture was launched {overshoot_ms:.0f} ms after its own "
                f"deadline; lead_ms ({lead_ms:.0f}) has to exceed the launch "
                f"correction ({config.launch.latency_ms:.0f} ms)"
            )
        if not config.launch.usable:
            notes.append(
                "this backend's launch latency has never been measured, so the "
                "gesture was launched with no correction; calibrate_launch first"
            )
        elif config.launch.spread_ms > 10.0:
            notes.append(
                f"launch latency varies by {config.launch.spread_ms:.1f} ms on "
                f"this device, which the pre-roll has to absorb"
            )
    else:
        touch_reply = config.touch.deliver(bundle, start_uptime_ms=t0)
        touch_reply.setdefault("honoured_start", True)

    return ActionReceipt(
        bundle_id=bundle.bundle_id,
        action=bundle.action,
        configuration=getattr(config, "name", "?"),
        t0_uptime_ms=t0,
        imu_frames=bundle.imu_frames,
        touch_points=len(bundle.touch.points),
        imu_start_elapsed_ns=imu_schedule.start_elapsed_ns,
        lead_ms=lead_ms,
        imu=imu_schedule.as_dict(),
        touch=touch_reply,
        notes=notes,
    )


def calibrate_launch(
    config,
    bundle: ActionBundle,
    client,
    *,
    trials: int = 9,
    warmup: int = 2,
    lead_ms: float = 400.0,
    settle_s: float = 1.0,
) -> LaunchCalibration:
    """Measure how long after the host lets go the first event actually lands.

    Only backends the host has to launch need this, and only the target app can
    answer it, because answering it means reading back when the contact was
    received.  So the measurement is made against our own application and then
    carried into whichever configuration uses the same backend -- including one
    aimed at an application we did not write, where nothing can be read back.
    That transfer is an assumption, and it is written into the calibration's
    ``source`` so it travels with the number instead of being forgotten.

    The median is the correction; the spread is what no correction removes and
    what the pre-roll has to be long enough to absorb.

    ``warmup`` trials are played and thrown away first.  The first launches
    after a virtual device is registered are not like the ones after: an early
    five-trial run on a Pixel 10 produced two landings at 68 and 121 ms, and
    seventy-four later trials across six conditions -- keyboard shown and
    hidden, layout freshly changed and settled, tapping inert areas and live
    controls -- never reproduced them, settling instead at 5.7 to 6.6 ms with
    4 to 11 ms of spread.  Whatever that was, it belonged to the beginning, and
    a correction computed from it would have been wrong by a hundred
    milliseconds for the whole run.
    """

    if not hasattr(config.touch, "stage"):
        return LaunchCalibration.none(f"{config.touch.name}: nothing to launch")

    trials_ms: list[float] = []
    discarded = max(0, warmup)
    for attempt in range(max(1, trials) + discarded):
        client.clear(scheduled=True)
        staged = config.touch.stage(bundle)
        anchor = HostAnchor.measure(lambda: client.ping()["uptime_ms"])
        target_uptime_ms = anchor.device_uptime_now() + lead_ms
        sleep_until_host(anchor.host_deadline_for(target_uptime_ms))
        config.touch.launch(staged)
        time.sleep(settle_s)

        rows = [row for row in client.dump().get("touch", []) if not row[TOUCH_BATCHED]]
        if not rows:
            continue
        landed_uptime_ms = min(row[TOUCH_UPTIME_NS] for row in rows) / 1e6
        if attempt < discarded:
            continue
        trials_ms.append(landed_uptime_ms - target_uptime_ms)

    return LaunchCalibration.from_trials(
        trials_ms,
        source=f"{config.touch.name}/{getattr(config.touch, 'mode', '')} "
        f"measured in-app over {len(trials_ms)} trials after {discarded} discarded",
    )
