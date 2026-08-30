"""Two ways to put inertia into an app, behind one interface.

The touch half of the system already had two backends and one interface.  The
IMU half did not, because for a long time there was only one way to do it: the
target app is ours, so the frames go in at the app's own intake and nothing
outside has to be touched.  That is configuration one, and it is exact.

Configuration two exists because that trick does not generalise.  An app we did
not write has no intake we control, so the substitution has to happen inside
its process, at the last framework call before its callback.  What lands there
is not chosen by us at the source -- it is a *replacement* of something the
framework was already delivering, at the rate the app itself asked for.

The two are different enough that pretending they are the same would be the
whole mistake:

``bus``
    The app's IMU intake selects our stream instead of the sensors, and the
    real listener is unregistered rather than ignored -- otherwise the phone's
    own motion and ours are summed and the seam is louder than either.  Frames
    land bit-exact on planned timestamps.  Our app only.

``hook``
    A hook inside the target process rewrites the payload of a delivery the
    framework was already making.  Any app can be the target, but the rate is
    the app's, the delivery instants are the driver's, and what we choose is
    *which planned frame belongs at this instant* rather than when the frame
    arrives.

Both are driven by the same scheduler against the same :class:`Timebase`, and
both report what they can and cannot reproduce so a run on one is never
presented as a run on the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from ..bundle import ActionBundle
from ..clock import Timebase


@dataclass
class ImuSchedule:
    """What a backend actually committed to, on the sensor clock."""

    start_elapsed_ns: int
    frames: int
    period_ms: float
    bundle_id: str
    # Who turned the touch-clock instant into a sensor-clock instant.  The app
    # does it on the device with an offset it measured itself; the hook has no
    # such channel, so the host does it with the offset it measured.  Which one
    # happened decides whose clock error an alignment number contains.
    converted_by: str = "device"
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def end_elapsed_ns(self) -> int:
        return self.start_elapsed_ns + int(round(self.frames * self.period_ms * 1_000_000))

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_elapsed_ns": self.start_elapsed_ns,
            "frames": self.frames,
            "period_ms": self.period_ms,
            "bundle_id": self.bundle_id,
            "converted_by": self.converted_by,
            **self.detail,
        }


class ImuBackend(Protocol):
    """Delivers the inertial half of a bundle."""

    name: str

    def describe(self) -> dict[str, Any]:
        """What this backend is and what it can and cannot reproduce."""

    def set_mode(self, mode: str) -> None:
        """``real`` hands the app back its own sensors; ``injected`` takes over."""

    def set_background(self, frames: Sequence[Sequence[float]], period_ms: float) -> None:
        """The stream that plays between actions.

        Not an optimisation.  A hand holding a phone never stops moving, so an
        IMU that is exact during gestures and silent between them is a stronger
        signal than a wrong gesture would be.
        """

    def schedule(
        self, bundle: ActionBundle, *, t0_uptime_ms: int, timebase: Timebase
    ) -> ImuSchedule:
        """Place this bundle's window so its first frame lands at ``t0``."""


class BusImuBackend:
    """Configuration one: the target app's own intake selects our stream.

    The conversion from the touch clock to the sensor clock is left to the
    device.  The app measured its own offset and applies it to the start
    instant we give it, so the number that ends up in the recording was
    computed on the same machine that stamps the events -- the host's estimate
    of the offset is then a check on that rather than an input to it.
    """

    name = "bus"

    def __init__(self, client, target: str = ""):
        self.client = client
        # Which application owns this control channel.  Asked, not assumed: the
        # bus reaches exactly the app that hosts it, and that app is no longer
        # always ours -- a study app carrying the same intake is reached the
        # same way and must be named correctly in the run's reach.
        self._target = target

    @property
    def target(self) -> str:
        if self._target:
            return self._target
        try:
            named = self.client.hello().raw.get("package", "")
        except Exception:
            named = ""
        self._target = named or "the app hosting the control channel"
        return self._target

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "requires": [],
            "reproduces": [
                "sample values (bit-exact)",
                "planned sample timestamps",
                "planned rate",
                "continuity between actions",
            ],
            "does_not_reproduce": [
                "the driver's own delivery jitter",
                "hardware FIFO batching",
            ],
            "targets": self.target,
            "third_party_capable": False,
            "clock_conversion": "device",
        }

    def set_mode(self, mode: str) -> None:
        self.client.set_imu_mode(mode)

    def set_background(self, frames: Sequence[Sequence[float]], period_ms: float) -> None:
        self.client.set_background(frames, period_ms=period_ms)

    def schedule(
        self, bundle: ActionBundle, *, t0_uptime_ms: int, timebase: Timebase
    ) -> ImuSchedule:
        reply = self.client.schedule_imu(
            bundle.imu_rows(),
            start_uptime_ms=int(t0_uptime_ms),
            period_ms=bundle.imu_period_ms,
            bundle_id=bundle.bundle_id,
        )
        start = int(reply["start_elapsed_ns"])
        # The app converted with its offset, we hold another; the gap between
        # them is reported because it bounds every alignment number this
        # configuration produces.
        host_start = timebase.elapsed_ns_at(t0_uptime_ms)
        return ImuSchedule(
            start_elapsed_ns=start,
            frames=bundle.imu_frames,
            period_ms=bundle.imu_period_ms,
            bundle_id=bundle.bundle_id,
            converted_by="device",
            detail={
                "host_start_elapsed_ns": host_start,
                "clock_disagreement_ms": round((start - host_start) / 1e6, 6),
            },
        )
