"""The one timeline both streams are placed on.

Android does not hand out a single clock.  A ``MotionEvent`` is stamped with
``SystemClock.uptimeMillis`` (CLOCK_MONOTONIC, stops across suspend); a
``SensorEvent`` is stamped with ``SystemClock.elapsedRealtimeNanos``
(CLOCK_BOOTTIME, keeps counting).  Nothing in the framework relates them, and
the difference is not a constant of the model -- it is whatever this device has
slept since it booted, so it changes per phone and per boot and grows while a
session runs.

So the offset is measured rather than assumed, and measured the way a time
protocol does it: the reader cannot sample both clocks at the same instant, so
it samples them repeatedly, records how wide each read window was, and keeps
the reading taken through the narrowest window.  A median is reported too,
because a best-of-N that disagrees with the median means the phone was busy and
the estimate should not be trusted quietly.

Everything downstream -- the IMU schedule, the touch schedule, the alignment
check -- converts through a :class:`Timebase` and never subtracts one clock
from the other by hand.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

# One sample of both clocks, plus how long the reader took to get them.
ClockReader = Callable[[], "ClockSample"]


@dataclass(frozen=True)
class ClockSample:
    uptime_ms: int
    elapsed_ns: int
    read_window_ns: int = 0

    @property
    def offset_ns(self) -> int:
        """elapsed - uptime, the quantity that differs between the clocks."""

        return self.elapsed_ns - self.uptime_ms * 1_000_000


@dataclass(frozen=True)
class Timebase:
    """How to move between the touch clock and the sensor clock on this device."""

    offset_ns: int
    median_ns: int
    spread_ns: int
    read_window_ns: int
    samples: int
    source: str

    # A best-of-N that sits far from the median means the reads were taken
    # while the phone was busy and the two clocks were not caught close
    # together.  A frame is 10 ms, so disagreement above a frame is worth
    # refusing rather than carrying into an alignment number.
    DISAGREEMENT_LIMIT_NS: int = 10_000_000

    @property
    def trustworthy(self) -> bool:
        return abs(self.offset_ns - self.median_ns) <= self.DISAGREEMENT_LIMIT_NS

    def elapsed_ns_at(self, uptime_ms: float) -> int:
        """The sensor-clock instant that the given touch-clock instant is."""

        return int(round(uptime_ms * 1_000_000)) + self.offset_ns

    def uptime_ms_at(self, elapsed_ns: int) -> float:
        """The touch-clock instant that the given sensor-clock instant is."""

        return (elapsed_ns - self.offset_ns) / 1_000_000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_ns": self.offset_ns,
            "median_ns": self.median_ns,
            "spread_ns": self.spread_ns,
            "read_window_ns": self.read_window_ns,
            "samples": self.samples,
            "source": self.source,
            "trustworthy": self.trustworthy,
            "disagreement_ns": abs(self.offset_ns - self.median_ns),
        }


def measure_timebase(
    read: ClockReader,
    *,
    samples: int = 20,
    settle_s: float = 0.02,
    source: str = "unknown",
) -> Timebase:
    """Sample both clocks repeatedly and keep the cleanest reading.

    ``read`` returns a :class:`ClockSample`; it is whatever can see both clocks
    on this configuration -- the target app's control channel in one, the
    instrumentation's own RPC in the other.  Neither is privileged here, which
    is the point: the two configurations get their offset the same way, so the
    alignment numbers they report are the same kind of number.
    """

    if samples < 1:
        raise ValueError("need at least one sample")

    taken: list[ClockSample] = []
    for index in range(samples):
        taken.append(read())
        if settle_s and index + 1 < samples:
            time.sleep(settle_s)

    offsets = [s.offset_ns for s in taken]
    best = min(taken, key=lambda s: (s.read_window_ns, abs(s.offset_ns - statistics.median(offsets))))
    return Timebase(
        offset_ns=best.offset_ns,
        median_ns=int(statistics.median(offsets)),
        spread_ns=int(max(offsets) - min(offsets)),
        read_window_ns=int(best.read_window_ns),
        samples=len(taken),
        source=source,
    )


def timebase_from_control(client, *, samples: int = 20, settle_s: float = 0.02) -> Timebase:
    """Measure through the target app's control channel (configuration one)."""

    def read() -> ClockSample:
        reply = client.ping()
        return ClockSample(
            uptime_ms=int(reply["uptime_ms"]),
            elapsed_ns=int(reply["elapsed_ns"]),
            read_window_ns=int(reply.get("read_window_ns", 0)),
        )

    return measure_timebase(read, samples=samples, settle_s=settle_s, source="control_channel")


@dataclass(frozen=True)
class LaunchCalibration:
    """How long after the host lets go the device actually starts injecting.

    Only the in-app backend can be told to begin at an instant; a stream handed
    to ``uinput`` begins when the process the shell started gets scheduled, and
    that is a device property nobody can compute.  So it is measured: a
    single-contact stream is launched against a known intended instant, the
    landing is read back, and the difference is what the scheduler subtracts
    next time.

    The spread matters more than the median.  A large median just shifts the
    launch earlier; a large spread is jitter that no amount of shifting removes
    and that the pre-roll has to absorb, so it is carried into the receipt
    rather than averaged away.
    """

    latency_ms: float
    spread_ms: float
    samples: int
    source: str
    trials_ms: tuple[float, ...] = ()

    @property
    def usable(self) -> bool:
        return self.samples > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 3),
            "spread_ms": round(self.spread_ms, 3),
            "samples": self.samples,
            "source": self.source,
            "trials_ms": [round(v, 3) for v in self.trials_ms],
        }

    @classmethod
    def none(cls, source: str = "uncalibrated") -> "LaunchCalibration":
        """No correction, stated as such rather than as a zero that looks measured."""

        return cls(latency_ms=0.0, spread_ms=0.0, samples=0, source=source)

    @classmethod
    def from_trials(cls, trials_ms: list[float], *, source: str) -> "LaunchCalibration":
        if not trials_ms:
            return cls.none(source)
        ordered = sorted(trials_ms)
        return cls(
            latency_ms=float(statistics.median(ordered)),
            spread_ms=float(ordered[-1] - ordered[0]),
            samples=len(ordered),
            source=source,
            trials_ms=tuple(trials_ms),
        )


@dataclass
class HostAnchor:
    """Relates this machine's monotonic clock to the phone's touch clock.

    Needed only by the backends the host has to *launch* rather than schedule.
    The in-app path is told an instant and honours it on the device; a stream
    handed to a shell begins when that process gets scheduled, so the host has
    to decide when to let go, and to do that it needs to know what device
    uptime "now" corresponds to.

    The relation is measured the same way as everything else here -- read the
    device clock with the host clock on either side of it, keep the narrowest
    read -- and it carries its own width so a scheduler can tell the difference
    between a launch that was late and an anchor that was never tight.
    """

    host_perf_s: float
    device_uptime_ms: float
    read_window_s: float
    samples: int

    def host_deadline_for(self, device_uptime_ms: float) -> float:
        """The host-clock instant at which the device clock reads that value."""

        return self.host_perf_s + (device_uptime_ms - self.device_uptime_ms) / 1000.0

    def device_uptime_now(self) -> float:
        return self.device_uptime_ms + (time.perf_counter() - self.host_perf_s) * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_uptime_ms": self.device_uptime_ms,
            "read_window_ms": round(self.read_window_s * 1000.0, 4),
            "samples": self.samples,
        }

    @classmethod
    def measure(cls, read_uptime_ms, *, samples: int = 5) -> "HostAnchor":
        """Keep the reading taken through the narrowest host-side window."""

        best: Optional[HostAnchor] = None
        for _ in range(max(1, samples)):
            before = time.perf_counter()
            uptime = float(read_uptime_ms())
            after = time.perf_counter()
            candidate = cls(
                host_perf_s=(before + after) / 2.0,
                device_uptime_ms=uptime,
                read_window_s=after - before,
                samples=1,
            )
            if best is None or candidate.read_window_s < best.read_window_s:
                best = candidate
        assert best is not None
        return cls(
            host_perf_s=best.host_perf_s,
            device_uptime_ms=best.device_uptime_ms,
            read_window_s=best.read_window_s,
            samples=max(1, samples),
        )


def sleep_until_host(deadline_perf_s: float, *, spin_s: float = 0.002) -> float:
    """Wait for a host-clock instant, then hand control back promptly.

    ``time.sleep`` is accurate to a few milliseconds at best on a general
    purpose OS, which is a whole detector frame, so the last couple of
    milliseconds are spun rather than slept.  What is returned is how far past
    the deadline control actually resumed, and it is recorded per action rather
    than assumed to be zero.
    """

    coarse = deadline_perf_s - spin_s
    now = time.perf_counter()
    if coarse > now:
        time.sleep(coarse - now)
    while time.perf_counter() < deadline_perf_s:
        pass
    return time.perf_counter() - deadline_perf_s
