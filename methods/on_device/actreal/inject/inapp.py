"""Touch delivered by the target app itself.

Nothing outside the app is involved, so nothing has to be unlocked.  The cost
is that the events do not come up the input pipeline, so they carry our
metadata rather than a driver's -- which is recorded in :meth:`describe` rather
than glossed over, because a recording made this way should never be presented
as one made the other way.
"""

from __future__ import annotations

from typing import Any

from ..bundle import ActionBundle
from ..control import ControlClient


class InAppTouchBackend:
    name = "inapp"

    def __init__(self, client: ControlClient):
        self.client = client

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "requires": [],
            "reproduces": [
                "coordinates",
                "pressure",
                "planned event timestamps",
                "pointer lifecycle",
            ],
            "does_not_reproduce": [
                "input device id",
                "source flags",
                "driver batching (historical points)",
            ],
            "targets": "the ActReal target app only",
            "third_party_capable": False,
        }

    def deliver(self, bundle: ActionBundle, *, start_uptime_ms: int) -> dict[str, Any]:
        reply = self.client.play_touch(
            bundle.touch_points_on_bundle_clock(),
            start_uptime_ms=start_uptime_ms,
        )
        return {"backend": self.name, "points": reply.get("points"), "reply": reply}
