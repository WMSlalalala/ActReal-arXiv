"""The backends, and the fact that both halves of an action have two of them.

An action is a touch and an inertial window delivered together.  Each half can
be delivered two ways, and the pairing is what :mod:`actreal.configs` calls a
configuration.  Nothing here decides which pair to use; that depends on what the
device turns out to allow, which only the probe can answer.

Touch
    ``inapp``  the app builds MotionEvents and dispatches them into its own
    view hierarchy.  Needs nothing.  Every coordinate, pressure and timestamp is
    the planned one, and what it cannot produce is the input pipeline's own
    metadata -- device id, source flags, driver batching.  Our app only.

    ``root``   something outside the app writes to an input device and Android
    dispatches the result, so the app receives real MotionEvents with real
    provenance and any app can be the target.  Needs ``uinput`` to be open,
    which on some builds it is to the shell user and on others it is not.

IMU
    ``bus``    the target app's own intake selects our stream instead of the
    sensors.  Bit-exact, on planned timestamps, our app only.

    ``hook``   a hook inside the target process replaces the payload of a
    delivery the framework was already making.  Any app can be the target; the
    rate and the delivery instants belong to that app, because a hook edits
    deliveries and cannot invent them.

Every backend answers ``describe()`` with what it reproduces and what it does
not, and no result is reported without it -- the two touch backends in
particular reproduce different things, and pooling their runs would be the
easiest mistake in the system to make and the hardest to see afterwards.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..bundle import ActionBundle


class TouchBackend(Protocol):
    """Delivers the touch half of a bundle."""

    name: str

    def describe(self) -> dict[str, Any]:
        """What this backend is and what it can and cannot reproduce."""

    def deliver(self, bundle: ActionBundle, *, start_uptime_ms: int) -> dict[str, Any]:
        """Put the bundle's touch on the screen, starting at the given instant."""


from .imu import BusImuBackend, ImuBackend, ImuSchedule  # noqa: E402
from .inapp import InAppTouchBackend  # noqa: E402
from .root import (  # noqa: E402
    RootTouchBackend,
    StagedStream,
    UinputStream,
    parse_uinput_stream,
)

BACKENDS = {"inapp": InAppTouchBackend, "root": RootTouchBackend}
IMU_BACKENDS = {"bus": BusImuBackend, "hook": "actreal.inject.frida_imu:FridaImuBackend"}

__all__ = [
    "TouchBackend",
    "InAppTouchBackend",
    "RootTouchBackend",
    "StagedStream",
    "UinputStream",
    "parse_uinput_stream",
    "ImuBackend",
    "ImuSchedule",
    "BusImuBackend",
    "BACKENDS",
    "IMU_BACKENDS",
]
