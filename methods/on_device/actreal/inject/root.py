"""Touch delivered through the real input pipeline.

A virtual touchscreen is registered with ``uinput`` and the gesture is written
to it as multi-touch protocol B; Android reads it the way it reads the physical
digitiser, so the app receives MotionEvents with a real device id, real source
flags and the driver's own batching.  That is what the in-app backend cannot
produce, and it is what lets a third-party app be the target.

**The timing has to happen on the device.** One ``adb shell sendevent`` per
event costs 10-30 ms of round trip, which is longer than the gaps being
reproduced -- a 200 ms scroll would arrive stretched over seconds and every
inter-sample interval would be the shell's, not the hand's. So the whole
gesture is compiled into a single command stream whose waits are executed on
the phone, and the host sends it once.

Everything here is generated and checked offline. Whether the device will
accept it is a question only the phone can answer (``actreal.device.probe``), but
whether what we send *says* what the plan said is answered by
:func:`parse_uinput_stream`, which reads the stream back into a trajectory and
compares.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from ..bundle import ActionBundle
from ..clock import sleep_until_host
from ..touch_track import DOWN, POINTER_DOWN, POINTER_UP, UP, TouchPoint, TouchTrack

# Linux input event types and codes, as the kernel numbers them.
EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
SYN_REPORT = 0
BTN_TOUCH = 0x14A
ABS_MT_SLOT = 0x2F
ABS_MT_TOUCH_MAJOR = 0x30
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
ABS_MT_TRACKING_ID = 0x39
ABS_MT_PRESSURE = 0x3A

# uinput ioctls, as the `uinput` command's configuration entries name them.
UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_ABSBIT = 100, 101, 103
UI_SET_PROPBIT = 110

# Input device properties, by bit index.  DIRECT is 1; 0 is POINTER, and getting
# that backwards would ask for exactly the behaviour this is here to avoid.
INPUT_PROP_POINTER = 0x00
INPUT_PROP_DIRECT = 0x01

# The digitiser's own pressure scale.  Trajectories carry pressure in 0..1, so
# it is scaled here and only here.
PRESSURE_MAX = 255
TOUCH_MAJOR_MAX = 255


@dataclass
class StagedStream:
    """A compiled gesture ready to play: on the device, or in hand for a session."""

    remote: str
    bytes: int
    mode: str
    bundle_id: str
    # Set when the gesture goes to an open UinputSession, where there is nothing
    # to transfer -- the commands are written straight down the session's stdin.
    text: str = ""
    # (instant_ms, command) for the paced path; see UinputSession.play_timed.
    events: list[tuple[float, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "remote": self.remote,
            "bytes": self.bytes,
            "mode": self.mode,
            "bundle_id": self.bundle_id,
        }


class UinputSession:
    """One virtual touchscreen, registered once and held open for the whole run.

    Registering is not free and it is not fast.  Every stream used to begin by
    creating its own device, and the cost of that was measured on a Pixel 10 at
    **600 ms, varying by 103 ms** between gestures -- not the transfer, not the
    shell, but Android's InputReader noticing that a new input device exists and
    building its state for it.  Holding one device open across the run instead
    brings the same measurement to **6 ms, varying by 9 ms**.

    The 600 ms was correctable on average and the 103 ms was not, which is the
    part that mattered: a gesture aimed at a planned instant landed a hundred
    milliseconds either side of it, and the inertia it was supposed to sit under
    is only ten milliseconds wide per frame.

    The device stays alive exactly as long as its stdin does, so the host holds
    the pipe open and writes each gesture into it.  Timing inside a gesture is
    still the phone's -- the stream carries its own ``delay`` commands -- and
    only the starting instant depends on this write.
    """

    def __init__(
        self,
        adb,
        *,
        device_w: int,
        device_h: int,
        device_id: int = 1,
        name: str = "ActReal Touch",
        profile: Optional[Any] = None,
    ):
        self.adb = adb
        self.device_w = device_w
        self.device_h = device_h
        self.device_id = device_id
        self.name = name
        self.profile = profile
        self._proc: Optional[subprocess.Popen] = None
        self.gestures = 0
        self.settle_s = 0.0

    @property
    def open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, *, settle_s: float = 1.5) -> dict[str, Any]:
        """Register the device and wait for the input stack to pick it up.

        The wait is not optional and cannot be replaced by a retry: events
        written before InputReader has finished building its state for the
        device are accepted by the kernel and delivered nowhere.
        """

        if self.open:
            return {"already_open": True}

        argv = [str(self.adb.path)]
        if self.adb.serial:
            argv += ["-s", self.adb.serial]
        argv += ["shell", "uinput -"]

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._write(json.dumps(
            _register(self.device_id, self.device_w, self.device_h, self.name, self.profile)
        ))
        time.sleep(settle_s)
        self.settle_s = settle_s
        if not self.open:
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"uinput exited while registering: {stderr.strip()[:300]}")
        return {"opened": True, "settle_s": settle_s, "name": self.name}

    def _write(self, text: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("the uinput session is not open")
        self._proc.stdin.write((text.rstrip("\n") + "\n").encode())
        self._proc.stdin.flush()

    def play(self, text: str) -> dict[str, Any]:
        """Write commands into the open device without pacing them."""

        if not self.open:
            raise RuntimeError("the uinput session closed; the device is gone")
        started = time.perf_counter()
        self._write(text)
        self.gestures += 1
        return {"write_ms": round((time.perf_counter() - started) * 1000.0, 3)}

    def play_timed(self, events: Sequence[tuple[float, str]]) -> dict[str, Any]:
        """Write each event at its own instant, paced by this host.

        **This is the part of the design that the device would rather own, and
        does not.** ``uinput``'s own ``delay`` command schedules against a time
        base that only tracks wall clock while the process is reading its input
        continuously.  On a session held open across a run that base falls
        behind, every scheduled instant is already in the past, and the whole
        gesture fires at once -- measured on a Pixel 10 as a 300 ms gap arriving
        as 0.4 ms.  Feeding a fresh process per gesture keeps the delays, but
        then the start instant carries the cost of creating a device: 43-79 ms
        of spread, against an inertial frame that is 10 ms wide.

        So the start comes from the open device, which lands within a couple of
        milliseconds, and the gaps come from here.  What that costs is this
        host's scheduling jitter on every sample rather than only on the first,
        and it is not hidden: :meth:`play_timed` returns the worst and mean
        deviation from the plan for the caller to record.
        """

        if not self.open:
            raise RuntimeError("the uinput session closed; the device is gone")
        if not events:
            return {"events": 0}

        anchor = time.perf_counter()
        worst = 0.0
        total = 0.0
        for t_ms, command in events:
            overshoot_s = sleep_until_host(anchor + t_ms / 1000.0)
            self._write(command)
            worst = max(worst, overshoot_s * 1000.0)
            total += overshoot_s * 1000.0
        self.gestures += 1
        return {
            "events": len(events),
            "paced_by": "host",
            "worst_late_ms": round(worst, 3),
            "mean_late_ms": round(total / len(events), 3),
            "span_ms": round((time.perf_counter() - anchor) * 1000.0, 3),
        }

    def stop(self) -> None:
        """Close stdin, which is what destroys the virtual device."""

        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> "UinputSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@dataclass
class UinputStream:
    """The command stream for one gesture, plus what it claims to contain."""

    lines: list[str] = field(default_factory=list)
    device_w: int = 1080
    device_h: int = 2424
    device_id: int = 1
    total_delay_ms: float = 0.0
    events: int = 0

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def summary(self) -> dict[str, Any]:
        return {
            "lines": len(self.lines),
            "events": self.events,
            "total_delay_ms": round(self.total_delay_ms, 3),
            "device": [self.device_w, self.device_h],
        }


# Extra axis codes the real digitiser declares.
ABS_MT_TOUCH_MINOR = 0x31
ABS_MT_ORIENTATION = 0x34
ABS_MT_TOOL_TYPE = 0x37

# How wide a finger is, as this phone reports it.  Measured, not chosen: across
# 2707 samples of a human session on the Pixel 10 the recorded size had a median
# of 0.0574 and ran from 0.034 to 0.069.  Size is the contact's major axis over
# the axis maximum the device declares, so reproducing it means emitting this
# fraction of whatever ABS_MT_TOUCH_MAJOR the device we are mirroring declares --
# not a raw number, which is what made a contact ninety times too wide when the
# maximum was assumed to be 255 instead of the real 24239.
NOMINAL_SIZE = 0.0574


def _register(
    device_id: int,
    width: int,
    height: int,
    name: str,
    profile: Optional[Any] = None,
) -> dict[str, Any]:
    """Describe a virtual device that mirrors the phone's own digitiser.

    Every axis this declares differently from the real one shows up in the
    recording, so the declaration is copied from the real device rather than
    invented.  Three differences were measured before this took the profile:

    * ``ABS_MT_PRESSURE`` does not exist on this phone's ``focal_ts``.  With no
      pressure axis Android synthesises 1.0 for a contact that is down, which is
      why a human session reports pressure as the single value 1.0 with standard
      deviation exactly zero.  Declaring the axis and reporting a donor's varying
      pressure was *less* like the device, not more, and separated the two
      sessions by forty standard deviations.
    * ``ABS_MT_TOUCH_MAJOR`` runs to 24239 there and 255 here, so the same raw
      value meant a contact ninety-five times wider: measured size 0.25 to 1.00
      against a human 0.056.
    * The position axes run to ten times the pixel count minus one, so the real
      device reports at a tenth of a pixel.  Declaring them in whole pixels threw
      that resolution away on every sample of every gesture.
    """

    def maximum(code: int, fallback: int) -> int:
        if profile is None:
            return fallback
        return profile.axes.get(code, (0, fallback))[1]

    has_pressure = bool(profile.has_pressure) if profile is not None else True
    # Ten device units per pixel when there is no profile to copy, because that
    # is what this class of digitiser does; with a profile the real number is
    # used and this is never reached.
    x_max = maximum(ABS_MT_POSITION_X, max(0, width * 10 - 1))
    y_max = maximum(ABS_MT_POSITION_Y, max(0, height * 10 - 1))

    def abs_entry(code: int, minimum: int, maximum_value: int) -> dict[str, Any]:
        return {
            "code": code,
            "info": {
                "value": 0,
                "minimum": minimum,
                "maximum": maximum_value,
                "fuzz": 0,
                "flat": 0,
                "resolution": 0,
            },
        }

    axes = [
        (ABS_MT_SLOT, 0, maximum(ABS_MT_SLOT, 9)),
        (ABS_MT_TOUCH_MAJOR, 0, maximum(ABS_MT_TOUCH_MAJOR, TOUCH_MAJOR_MAX)),
        (ABS_MT_TOUCH_MINOR, 0, maximum(ABS_MT_TOUCH_MINOR, TOUCH_MAJOR_MAX)),
        (ABS_MT_ORIENTATION, -4096, maximum(ABS_MT_ORIENTATION, 4096)),
        (ABS_MT_POSITION_X, 0, x_max),
        (ABS_MT_POSITION_Y, 0, y_max),
        (ABS_MT_TOOL_TYPE, 0, maximum(ABS_MT_TOOL_TYPE, 2)),
        (ABS_MT_TRACKING_ID, 0, maximum(ABS_MT_TRACKING_ID, 0xFFFF)),
    ]
    if has_pressure:
        axes.append((ABS_MT_PRESSURE, 0, maximum(ABS_MT_PRESSURE, PRESSURE_MAX)))

    return {
        "id": device_id,
        "command": "register",
        "name": name,
        # Google's vendor id and an arbitrary product id; the device announces
        # itself as a touchscreen, not as something pretending to be a
        # particular phone's digitiser.
        "vid": 0x18D1,
        "pid": 0x4F4B,
        "bus": "usb",
        "configuration": [
            {"type": UI_SET_EVBIT, "data": [EV_KEY, EV_ABS, EV_SYN]},
            {"type": UI_SET_KEYBIT, "data": [BTN_TOUCH]},
            # Without this the device has multi-touch axes and no statement that
            # touching it means touching the screen, and Android's
            # TouchInputMapper classifies it as a *touchpad*.  A touchpad does
            # not report where it was touched -- it drives a cursor -- so every
            # coordinate arrives as the screen centre with SOURCE_MOUSE, and its
            # gesture recogniser re-synthesises the contact, so a 40 ms tap is
            # re-emitted as a 6 ms one.  Both were observed on a Pixel 10 before
            # this line existed; they are one cause, not two.
            {"type": UI_SET_PROPBIT, "data": [INPUT_PROP_DIRECT]},
            {"type": UI_SET_ABSBIT, "data": [code for code, _, _ in axes]},
        ],
        "abs_info": [abs_entry(code, low, high) for code, low, high in axes],
    }


def _scaled_pressure(value: float, maximum: int = PRESSURE_MAX) -> int:
    return int(round(min(max(float(value), 0.0), 1.0) * maximum))


# What this digitiser reports at.  Measured, not assumed: across a human session
# on the Pixel 10 the interval between touch samples had a median of 4.147 ms in
# taps, 4.151 in scrolls and 4.149 in swipes -- 241 Hz, and the same to three
# decimal places whatever the gesture.  A donor recorded on a 100 Hz grid
# delivered at 100 Hz therefore arrives at a rate no finger on this screen
# produces, and a tap carries three samples where a real one carries twenty-six.
NOMINAL_SAMPLE_MS = 4.147


def resample_track(track: TouchTrack, interval_ms: float) -> TouchTrack:
    """Re-lay a trajectory on the grid the digitiser reports on.

    **Off by default, and it should stay off.** This is the one thing in this
    module that changes the signal rather than the way the signal is delivered.
    The trajectories come from a generator that was measured and frozen; adding
    points between its samples produces a gesture it never produced, and a
    result obtained that way is a result about this function.

    It exists because the difference it addresses is real -- a human sample
    interval of 4.147 ms against a donor's 10 ms -- but that difference lives in
    the donor, which carries three samples where the recording it came from had
    twenty-six. The fix belongs where the trajectory is made, not here.

    **This adds samples; it does not add information.** Between two donor points
    it can only draw a straight line, so what it recovers is the sample *rate*
    and the coarse path, not the motion that happened between two frames of a
    100 Hz recording and was never captured. A check on how many samples a
    gesture has, or on the interval between them, sees a real one; a check on
    the high-frequency content of a finger sees a donor's, smoothed.

    A uniform grid rather than extra points inserted between the donor's own:
    keeping both left short gaps wherever a donor sample fell near a grid
    instant, and the interval came back at 3.2 ms against a real 4.15. The
    boundaries are exempt -- the DOWN and the UP keep the donor's own instants
    and coordinates, because the alignment against the inertial window is
    measured from them.
    """

    points = track.points
    if interval_ms <= 0 or len(points) < 2:
        return track

    first, last = points[0], points[-1]
    span = last.t_ms - first.t_ms
    if span <= interval_ms:
        return track

    def at(t_ms: float) -> TouchPoint:
        # The donor segment this instant falls in, and where inside it.
        index = 0
        while index + 2 < len(points) and points[index + 1].t_ms < t_ms:
            index += 1
        a, b = points[index], points[index + 1]
        gap = b.t_ms - a.t_ms
        ratio = 0.0 if gap <= 0 else min(1.0, max(0.0, (t_ms - a.t_ms) / gap))
        return replace(
            a,
            t_ms=t_ms,
            x=a.x + (b.x - a.x) * ratio,
            y=a.y + (b.y - a.y) * ratio,
            pressure=a.pressure + (b.pressure - a.pressure) * ratio,
            size=a.size + (b.size - a.size) * ratio,
            # Everything between the boundaries is movement: a second DOWN would
            # open a contact that is already open, an early UP would close it.
            action="MOVE",
        )

    filled: list[TouchPoint] = [first]
    steps = int(span // interval_ms)
    for step in range(1, steps + 1):
        t_ms = first.t_ms + step * interval_ms
        if last.t_ms - t_ms < interval_ms * 0.5:
            break
        filled.append(at(t_ms))
    filled.append(last)
    return replace(track, points=filled)


def build_uinput_stream(
    track: TouchTrack,
    *,
    device_w: int,
    device_h: int,
    device_id: int = 1,
    name: str = "ActReal Touch",
    slot: int = 0,
    include_register: bool = True,
    profile: Optional[Any] = None,
    size: float = NOMINAL_SIZE,
    sample_interval_ms: Optional[float] = None,
) -> UinputStream:
    """Compile a gesture into one device-side command stream.

    Waits are ``delay`` commands, so the phone runs the clock; the host sends
    the whole thing once and does not participate in the timing.

    ``include_register`` is off when the gesture is going to an already-open
    virtual device.  Registering costs far more than the gesture does -- see
    :class:`UinputSession` -- so it belongs once at the start of a session
    rather than at the top of every stream.
    """

    if sample_interval_ms:
        track = resample_track(track, sample_interval_ms)

    stream = UinputStream(device_w=device_w, device_h=device_h, device_id=device_id)
    if include_register:
        stream.lines.append(
            json.dumps(_register(device_id, device_w, device_h, name, profile))
        )

    def axis_max(code: int, fallback: int) -> int:
        if profile is None:
            return fallback
        return profile.axes.get(code, (0, fallback))[1]

    # One device unit per (max + 1) / pixels.  With the real profile this is ten.
    x_max = axis_max(ABS_MT_POSITION_X, device_w * 10 - 1)
    y_max = axis_max(ABS_MT_POSITION_Y, device_h * 10 - 1)
    x_scale = (x_max + 1) / float(device_w)
    y_scale = (y_max + 1) / float(device_h)
    major_max = axis_max(ABS_MT_TOUCH_MAJOR, TOUCH_MAJOR_MAX)
    minor_max = axis_max(ABS_MT_TOUCH_MINOR, TOUCH_MAJOR_MAX)
    # Android does not report size as major/max.  It reports the *mean of the
    # major and minor axes*, normalised by the major axis maximum -- which is
    # why emitting the target fraction on both axes landed short by exactly the
    # ratio between them: major 1391 and minor 620 came back as
    # (1391+620)/2/24239 = 0.0415 against a target of 0.0574, measured.
    #
    # So the raw values are solved for rather than assumed.  Keeping each axis
    # at the same fraction of its own range (which is what a round contact on a
    # digitiser whose axes differ in scale looks like), the fraction f that
    # yields the wanted size satisfies f*(major_max + minor_max)/2 = size*major_max.
    span = (major_max + minor_max) / 2.0

    def contact_axes(point_size: float) -> tuple[int, int]:
        """Raw major and minor for a contact of this reported size.

        The donor's own size is used wherever it has one; ``size`` is the
        fallback for donors that carry none, which is every bundle baked so far.
        Supplying something is unavoidable -- a digitiser reports a contact
        width for every sample and there is no way to stay silent about it --
        but inventing one where the donor already said is not.
        """

        wanted = point_size if point_size > 0 else size
        fraction = wanted * major_max / span if span > 0 else wanted
        return (
            max(1, int(round(fraction * major_max))),
            max(1, int(round(fraction * minor_max))),
        )
    pressure_max = (
        axis_max(ABS_MT_PRESSURE, PRESSURE_MAX)
        if (profile is None or profile.has_pressure)
        else None
    )

    previous_ms: Optional[float] = None
    # Per contact, not one global pair.  Protocol B addresses each finger by its
    # own slot, and the emitter used to write every point into slot 0 -- so a
    # two-finger gesture arrived as one finger jumping between two places, and a
    # pinch was not expressible at all no matter what the donor carried.
    slot_of: dict[int, int] = {}
    previous_xy_of: dict[int, tuple[int, int]] = {}
    tracking_of: dict[int, int] = {}
    tracking_id = 0
    open_contacts: set[int] = set()
    max_slot = axis_max(ABS_MT_SLOT, 9)

    for point in track.points:
        if previous_ms is not None:
            gap = point.t_ms - previous_ms
            if gap > 0:
                # Whole milliseconds: the command takes an integer, and a
                # rounded wait on the device beats a fractional one on the host.
                whole = int(round(gap))
                if whole > 0:
                    stream.lines.append(
                        json.dumps({"id": device_id, "command": "delay", "duration": whole})
                    )
                    stream.total_delay_ms += whole
        previous_ms = point.t_ms

        # Device units, not pixels.  The real digitiser reports at a tenth of a
        # pixel, so rounding to whole pixels here discarded up to half a pixel
        # on every sample -- which lands directly in the path length and the
        # speed a detector computes from it.
        x = min(x_max, int(round(min(max(point.x, 0.0), device_w - 1e-6) * x_scale)))
        y = min(y_max, int(round(min(max(point.y, 0.0), device_h - 1e-6) * y_scale)))
        contact_major, contact_minor = contact_axes(point.size)

        pointer = int(point.pointer_id)
        if pointer not in slot_of:
            # First sight of this finger: give it the lowest free slot, the way
            # a driver does.  A donor with more contacts than the device has
            # slots is a donor this screen could not have recorded.
            used = set(slot_of.values())
            free = next((i for i in range(max_slot + 1) if i not in used), None)
            if free is None:
                raise ValueError(
                    f"gesture needs more than {max_slot + 1} contacts, which this "
                    f"digitiser does not report"
                )
            slot_of[pointer] = free
        contact_slot = slot_of[pointer]
        previous_xy = previous_xy_of.get(pointer)

        events: list[int] = [EV_ABS, ABS_MT_SLOT, contact_slot]
        if point.action in (DOWN, POINTER_DOWN):
            tracking_id += 1
            tracking_of[pointer] = tracking_id
            events += [EV_ABS, ABS_MT_TRACKING_ID, tracking_id]
            events += [EV_ABS, ABS_MT_TOOL_TYPE, 0]   # finger
            open_contacts.add(pointer)
        if point.action in (DOWN, POINTER_DOWN, "MOVE"):
            events += [
                EV_ABS, ABS_MT_POSITION_X, x,
                EV_ABS, ABS_MT_POSITION_Y, y,
                EV_ABS, ABS_MT_TOUCH_MAJOR, contact_major,
                EV_ABS, ABS_MT_TOUCH_MINOR, contact_minor,
            ]
            if pressure_max is not None:
                # Only when the device being mirrored actually has the axis.
                # Where it does not, Android synthesises 1.0 for a contact that
                # is down, which is what a human recording on this phone shows.
                events += [EV_ABS, ABS_MT_PRESSURE, _scaled_pressure(point.pressure, pressure_max)]
        if point.action == DOWN or (point.action == POINTER_DOWN and len(open_contacts) == 1):
            # BTN_TOUCH says "something is on the glass", so it is asserted when
            # the first contact lands and released when the last one leaves --
            # not once per finger.
            events += [EV_KEY, BTN_TOUCH, 1]
        if point.action in (UP, POINTER_UP):
            # The lift gets a frame to itself.  Protocol B treats
            # ABS_MT_TRACKING_ID = -1 as "this contact is gone", and a position
            # sent in that same frame is describing a contact that no longer
            # exists -- the kernel reports the lift at the previous position and
            # the move is simply lost.  It only shows on gestures whose last
            # point moves, which is why two-point taps came back exactly 7 px
            # out on a Pixel 10 while longer ones were exact.
            if (x, y) != (previous_xy or (x, y)):
                events += [
                    EV_ABS, ABS_MT_POSITION_X, x,
                    EV_ABS, ABS_MT_POSITION_Y, y,
                    EV_SYN, SYN_REPORT, 0,
                    EV_ABS, ABS_MT_SLOT, contact_slot,
                ]
            events += [EV_ABS, ABS_MT_TRACKING_ID, -1]
            open_contacts.discard(pointer)
            slot_of.pop(pointer, None)
            if not open_contacts:
                events += [EV_KEY, BTN_TOUCH, 0]
        previous_xy_of[pointer] = (x, y)
        events += [EV_SYN, SYN_REPORT, 0]

        stream.lines.append(
            json.dumps({"id": device_id, "command": "inject", "events": events})
        )
        stream.events += len(events) // 3

    if open_contacts:
        raise ValueError(
            f"gesture ends with {len(open_contacts)} contact(s) still down"
        )
    return stream


def build_uinput_events(
    track: TouchTrack,
    *,
    device_w: int,
    device_h: int,
    device_id: int = 1,
    slot: int = 0,
    profile: Optional[Any] = None,
    size: float = NOMINAL_SIZE,
    sample_interval_ms: Optional[float] = None,
) -> list[tuple[float, str]]:
    """The same gesture as (instant, command) pairs, with no delay commands.

    Built from :func:`build_uinput_stream` rather than beside it, so there is
    one encoder for the events and no chance of the paced path and the streamed
    path disagreeing about what a gesture is.
    """

    # The profile has to come through here too.  Without it the registration
    # (which had it) declared touch-major running to 24239 while the events
    # (which did not) scaled against the default 255 -- so a contact meant to be
    # 0.0574 of the axis was emitted as 15 of 24239, and came back as 0.00062.
    # The two halves have to be told the same thing about the device.
    stream = build_uinput_stream(
        track,
        device_w=device_w,
        device_h=device_h,
        device_id=device_id,
        slot=slot,
        include_register=False,
        profile=profile,
        size=size,
        sample_interval_ms=sample_interval_ms,
    )
    events: list[tuple[float, str]] = []
    t_ms = 0.0
    for line in stream.lines:
        command = json.loads(line)
        kind = command.get("command")
        if kind == "delay":
            t_ms += float(command["duration"])
        elif kind == "inject":
            events.append((t_ms, line))
    return events


def parse_uinput_stream(
    text: str, *, device_w: int, device_h: int, profile: Optional[Any] = None
) -> TouchTrack:
    """Read a command stream back into the gesture it describes.

    This is how the stream is checked without a phone: what the host is about
    to send is decoded independently and compared against the plan, so a
    mistake in the event encoding is caught here rather than as a gesture that
    silently lands somewhere else.
    """

    def axis_max(code: int, fallback: int) -> int:
        if profile is None:
            return fallback
        return profile.axes.get(code, (0, fallback))[1]

    # The same scale the builder used, undone.
    x_scale = (axis_max(ABS_MT_POSITION_X, device_w * 10 - 1) + 1) / float(device_w)
    y_scale = (axis_max(ABS_MT_POSITION_Y, device_h * 10 - 1) + 1) / float(device_h)
    pressure_max = axis_max(ABS_MT_PRESSURE, PRESSURE_MAX)

    t_ms = 0.0
    points: list[TouchPoint] = []
    slot_state: dict[str, Any] = {"x": None, "y": None, "pressure": 0, "tracking": -1}
    down_open = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        command = json.loads(line)
        kind = command.get("command")
        if kind == "delay":
            t_ms += float(command["duration"])
            continue
        if kind != "inject":
            continue

        events = command["events"]
        if len(events) % 3:
            raise ValueError("an inject must carry whole (type, code, value) triples")
        action: Optional[str] = None
        for index in range(0, len(events), 3):
            ev_type, code, value = events[index : index + 3]
            if ev_type == EV_ABS and code == ABS_MT_POSITION_X:
                slot_state["x"] = value
            elif ev_type == EV_ABS and code == ABS_MT_POSITION_Y:
                slot_state["y"] = value
            elif ev_type == EV_ABS and code == ABS_MT_PRESSURE:
                slot_state["pressure"] = value
            elif ev_type == EV_ABS and code == ABS_MT_TRACKING_ID:
                slot_state["tracking"] = value
                if value >= 0:
                    action = DOWN
                    down_open = True
                else:
                    action = UP
                    down_open = False
            elif ev_type == EV_SYN and code == SYN_REPORT:
                if slot_state["x"] is None:
                    continue
                points.append(
                    TouchPoint(
                        t_ms=t_ms,
                        x=float(slot_state["x"]) / x_scale,
                        y=float(slot_state["y"]) / y_scale,
                        pressure=float(slot_state["pressure"]) / pressure_max,
                        size=0.0,
                        pointer_id=0,
                        action=action or "MOVE",
                    )
                )
                action = None

    if down_open:
        raise ValueError("the stream never lifts the pointer")
    if not points:
        raise ValueError("the stream carries no reportable event")

    return TouchTrack(
        action="unknown",
        points=points,
        orientation_id=0,
        screen_w=float(device_w),
        screen_h=float(device_h),
        source="uinput_stream",
    )


def build_sendevent_script(
    track: TouchTrack, *, device_node: str, device_w: int, device_h: int
) -> str:
    """The same gesture as a device-side shell script, for when uinput is closed.

    ``sendevent`` needs root, and one shell round trip per event would destroy
    the timing, so this is written as a single script to push and run rather
    than a sequence of adb calls.
    """

    lines = ["#!/system/bin/sh", "# generated by ActReal -- run on the device, not the host"]
    previous_ms: Optional[float] = None
    previous_xy: Optional[tuple[int, int]] = None
    tracking = 0

    def emit(ev_type: int, code: int, value: int) -> None:
        lines.append(f"sendevent {shlex.quote(device_node)} {ev_type} {code} {value}")

    for point in track.points:
        if previous_ms is not None:
            gap_ms = point.t_ms - previous_ms
            if gap_ms > 0:
                lines.append(f"sleep {gap_ms/1000.0:.3f}")
        previous_ms = point.t_ms
        x = int(round(min(max(point.x, 0.0), device_w - 1.0)))
        y = int(round(min(max(point.y, 0.0), device_h - 1.0)))
        pressure = _scaled_pressure(point.pressure)

        emit(EV_ABS, ABS_MT_SLOT, 0)
        if point.action == DOWN:
            tracking += 1
            emit(EV_ABS, ABS_MT_TRACKING_ID, tracking)
        if point.action == UP:
            # Same rule as the uinput builder: the lift gets its own frame.
            # A position sent alongside ABS_MT_TRACKING_ID = -1 describes a
            # contact that the same SYN_REPORT destroys, so it never becomes a
            # reported sample and the gesture ends at its previous position.
            if (x, y) != (previous_xy or (x, y)):
                emit(EV_ABS, ABS_MT_POSITION_X, x)
                emit(EV_ABS, ABS_MT_POSITION_Y, y)
                emit(EV_SYN, SYN_REPORT, 0)
                emit(EV_ABS, ABS_MT_SLOT, 0)
            emit(EV_ABS, ABS_MT_TRACKING_ID, -1)
            emit(EV_KEY, BTN_TOUCH, 0)
        else:
            emit(EV_ABS, ABS_MT_POSITION_X, x)
            emit(EV_ABS, ABS_MT_POSITION_Y, y)
            emit(EV_ABS, ABS_MT_PRESSURE, pressure)
            if point.action == DOWN:
                emit(EV_KEY, BTN_TOUCH, 1)
        previous_xy = (x, y)
        emit(EV_SYN, SYN_REPORT, 0)

    return "\n".join(lines) + "\n"


class RootTouchBackend:
    """Writes the gesture to a virtual input device on the phone."""

    name = "root"

    def __init__(
        self,
        adb,
        *,
        device_w: int,
        device_h: int,
        mode: str = "uinput",
        device_node: str = "/dev/input/event2",
        remote_dir: str = "/data/local/tmp/actreal",
        persistent: bool = True,
        profile: Optional[Any] = None,
    ):
        if mode not in ("uinput", "sendevent"):
            raise ValueError(f"unknown mode {mode!r}")
        self.adb = adb
        self.device_w = device_w
        self.device_h = device_h
        self.mode = mode
        self.device_node = device_node
        self.remote_dir = remote_dir
        # One device for the run rather than one per gesture.  Only uinput can
        # do this; writing to an existing /dev/input node has no session to
        # hold open, so that mode keeps paying per-stream shell costs.
        self.persistent = persistent and mode == "uinput"
        # What the phone's own digitiser says it is.  Without it the virtual
        # device falls back to plausible defaults, which is what produced a
        # contact ninety-five times too wide and a pressure axis the real device
        # does not have.
        self.profile = profile
        self.session: Optional[UinputSession] = None

    # -- the shared virtual device -------------------------------------------

    def open_device(self, *, settle_s: float = 1.5) -> dict[str, Any]:
        if not self.persistent:
            return {"persistent": False}
        if self.session is None:
            self.session = UinputSession(
                self.adb,
                device_w=self.device_w,
                device_h=self.device_h,
                profile=self.profile,
            )
        return {"persistent": True, **self.session.start(settle_s=settle_s)}

    def close_device(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session = None

    def __enter__(self) -> "RootTouchBackend":
        self.open_device()
        return self

    def __exit__(self, *exc) -> None:
        self.close_device()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "mode": self.mode,
            "requires": ["uinput accessible to shell"]
            if self.mode == "uinput"
            else ["root"],
            "reproduces": [
                "coordinates",
                "pressure",
                "pointer lifecycle",
                "input device id",
                "source flags",
                "driver batching",
            ],
            "does_not_reproduce": (
                ["exact event timestamps (the driver stamps them)"]
                + (
                    ["device-side pacing: with the device held open, uinput's own "
                     "delays no longer schedule, so inter-sample gaps are timed "
                     "by the host and carry its jitter"]
                    if self.persistent
                    else []
                )
            ),
            "targets": "any app",
            "third_party_capable": True,
            "persistent_device": self.persistent,
            "device_open": self.session is not None and self.session.open,
            "gestures_played": self.session.gestures if self.session else 0,
        }

    def compile(self, bundle: ActionBundle) -> str:
        if self.mode == "uinput":
            return build_uinput_stream(
                bundle.touch,
                device_w=self.device_w,
                device_h=self.device_h,
                profile=self.profile,
                # A session already has a device; re-registering inside a stream
                # would create a second one and pay the pickup cost again.
                include_register=not self.persistent,
            ).text()
        return build_sendevent_script(
            bundle.touch,
            device_node=self.device_node,
            device_w=self.device_w,
            device_h=self.device_h,
        )

    # Staging and launching are separate on purpose.  Transferring the stream
    # costs a few hundred milliseconds and varies with its size, so a backend
    # that pushed and ran in one call would put that cost *inside* the interval
    # it is trying to control: the gesture would start a variable, size-dependent
    # delay after the inertia it belongs to, which is precisely the quantity
    # these experiments exist to measure.  So the transfer happens ahead of
    # time and the launch is the only thing left on the clock.

    def stage(self, bundle: ActionBundle) -> "StagedStream":
        """Get the compiled stream ready to play, without playing it."""

        payload = self.compile(bundle)
        name = f"{bundle.bundle_id}.{'uinput' if self.mode == 'uinput' else 'sh'}"
        remote = f"{self.remote_dir}/{name}"

        if self.persistent:
            # Nothing to transfer: the commands go down the open session's
            # stdin, so staging is compiling plus working out when each event
            # belongs, because the phone will not be keeping that time.
            return StagedStream(
                remote="", bytes=len(payload), mode=self.mode,
                bundle_id=bundle.bundle_id, text=payload,
                events=build_uinput_events(
                    bundle.touch,
                    device_w=self.device_w,
                    device_h=self.device_h,
                    profile=self.profile,
                ),
            )

        self.adb.shell(f"mkdir -p {shlex.quote(self.remote_dir)}")
        with tempfile.NamedTemporaryFile(
            "w", suffix=f".{name}", delete=False, encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(payload)
            local = handle.name
        try:
            push = self.adb.run("push", local, remote)
            if not push.ok:
                raise RuntimeError(f"push failed: {push.stderr.strip()}")
        finally:
            Path(local).unlink(missing_ok=True)
        return StagedStream(
            remote=remote, bytes=len(payload), mode=self.mode, bundle_id=bundle.bundle_id
        )

    def launch(self, staged: "StagedStream", *, timeout: float = 60.0) -> dict[str, Any]:
        """Play a staged stream.

        Through a session this returns as soon as the commands are written and
        the gesture plays out on the phone against its own ``delay`` commands --
        which is the intent, because it keeps the host out of the timing loop.
        Without a session it starts a shell and blocks until that shell exits.
        """

        if self.persistent:
            if self.session is None or not self.session.open:
                raise RuntimeError("no open uinput session; call open_device() first")
            written = self.session.play_timed(staged.events)
            return {
                "backend": self.name,
                "mode": staged.mode,
                "persistent": True,
                "bytes": staged.bytes,
                "ok": True,
                # Paced from here, so this call lasts as long as the gesture.
                "blocking": True,
                **written,
            }

        if staged.mode == "uinput":
            command = f"uinput - < {shlex.quote(staged.remote)}"
        else:
            command = f"su -c 'sh {shlex.quote(staged.remote)}'"
        started = time.perf_counter()
        result = self.adb.shell(command, timeout=timeout)
        return {
            "backend": self.name,
            "mode": staged.mode,
            "remote": staged.remote,
            "bytes": staged.bytes,
            "ok": result.ok,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "stderr": result.stderr.strip()[:400],
        }

    def deliver(self, bundle: ActionBundle, *, start_uptime_ms: int) -> dict[str, Any]:
        """Stage and launch in one call, with no attempt to hit ``start_uptime_ms``.

        Kept for callers that only want the gesture played.  Anything that
        cares when it lands should stage first and launch against a deadline --
        see :func:`actreal.session.play_action`, which does exactly that and
        reports how far off the landing was.
        """

        staged = self.stage(bundle)
        return {**self.launch(staged), "honoured_start": False}
