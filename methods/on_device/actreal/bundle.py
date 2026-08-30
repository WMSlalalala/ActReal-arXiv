"""One action, compiled: touch and IMU on a single monotonic timeline.

An :class:`ActionBundle` is what leaves the host and reaches the phone.  Its
clock starts at the first IMU sample, not at the touch DOWN, because the
gesture sits inside a padding window whose leading frames are the same hand
already moving before it reaches the glass.  Delivering the touch first and the
inertia afterwards would invert the causality the whole method rests on.

    t=0            DOWN                       UP                    end
    |--- pre-roll ---|------- gesture --------|---- post-roll ------|
    |<---------------------- IMU stream ---------------------------->|
                     |<----- MotionEvents ---->|
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import config
from .mapping import ScreenMapping
from .touch_track import TouchTrack

if False:  # typing only; importing it for real would pull numpy into the local build
    from .imu_source import IMUWindow


@dataclass
class ActionBundle:
    bundle_id: str
    action: str
    touch: TouchTrack
    # Rows of six channels.  Kept as a plain sequence rather than an array so
    # the packaged local build -- which only loads, plays and checks bundles --
    # needs nothing installed beyond the standard library.
    imu: Sequence[Sequence[float]]
    imu_period_ms: float
    touch_offset_ms: float
    mapping: ScreenMapping
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.imu) == 0:
            raise ValueError("imu window is empty")
        widths = {len(row) for row in self.imu}
        if widths != {6}:
            raise ValueError(f"every imu frame needs 6 channels, saw widths {sorted(widths)}")
        if self.touch_offset_ms < -1e-9:
            raise ValueError("touch cannot start before the bundle clock")
        if self.touch_end_ms > self.imu_duration_ms + 1e-6:
            raise ValueError(
                f"touch ends at {self.touch_end_ms:.1f}ms but the IMU stream "
                f"only covers {self.imu_duration_ms:.1f}ms"
            )

    @property
    def imu_frames(self) -> int:
        return int(len(self.imu))

    def imu_rows(self) -> list[list[float]]:
        """The window as plain floats, whichever container it arrived in."""

        return [[float(v) for v in row] for row in self.imu]

    @property
    def imu_duration_ms(self) -> float:
        return self.imu_frames * self.imu_period_ms

    @property
    def touch_start_ms(self) -> float:
        return self.touch_offset_ms + self.touch.points[0].t_ms

    @property
    def touch_end_ms(self) -> float:
        return self.touch_offset_ms + self.touch.points[-1].t_ms

    @property
    def gesture_ms(self) -> float:
        """How long the contact itself lasts -- what the duration law predicts."""

        return self.touch_end_ms - self.touch_start_ms

    def touch_points_on_bundle_clock(self) -> list[dict[str, Any]]:
        return [
            {
                "t_ms": round(p.t_ms + self.touch_offset_ms, 4),
                "x": round(p.x, 3),
                "y": round(p.y, 3),
                "pressure": round(p.pressure, 5),
                "size": round(p.size, 5),
                "pointer_id": p.pointer_id,
                "action": p.action,
            }
            for p in self.touch.points
        ]

    def to_json_dict(self, *, include_imu: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "actreal_action_bundle_v1",
            "bundle_id": self.bundle_id,
            "action": self.action,
            "imu_period_ms": self.imu_period_ms,
            "imu_frames": self.imu_frames,
            "touch_offset_ms": round(self.touch_offset_ms, 4),
            "touch": self.touch_points_on_bundle_clock(),
            "mapping": self.mapping.as_dict(),
            "provenance": self.provenance,
        }
        if include_imu:
            payload["imu"] = [[round(v, 6) for v in row] for row in self.imu_rows()]
        return payload

    def to_json(self, *, include_imu: bool = True) -> str:
        return json.dumps(self.to_json_dict(include_imu=include_imu), sort_keys=True)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def reanchored(
        self,
        start: tuple[float, float],
        end: Optional[tuple[float, float]] = None,
    ) -> "ActionBundle":
        """Move this action onto new device coordinates, inertia unchanged.

        Re-aiming is a translation (or, with an end point, a rotation and a
        uniform scale of the chord).  None of that changes how fast the hand
        moved or how long it was in contact, so the window that was generated
        for this gesture still describes it and is carried over untouched.
        """

        from dataclasses import replace

        moved = self.touch.reanchor(start, end)
        return replace(
            self,
            touch=moved,
            provenance={**self.provenance, "reanchored_to": [list(start), list(end) if end else None]},
        )

    def summary(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "action": self.action,
            "imu_frames": self.imu_frames,
            "imu_duration_ms": round(self.imu_duration_ms, 1),
            "pre_roll_ms": round(self.touch_start_ms, 1),
            "gesture_ms": round(self.touch_end_ms - self.touch_start_ms, 1),
            "post_roll_ms": round(self.imu_duration_ms - self.touch_end_ms, 1),
            "touch_points": len(self.touch.points),
            "down_xy": [round(v, 1) for v in self.touch.down_xy],
            "up_xy": [round(v, 1) for v in self.touch.up_xy],
            "scale": round(self.mapping.scale, 6),
        }


def from_json_dict(data: dict[str, Any]) -> ActionBundle:
    """Rebuild a bundle that was baked on the host and shipped to the phone."""

    from .touch_track import TouchPoint, TouchTrack

    if data.get("schema_version") != "actreal_action_bundle_v1":
        raise ValueError(f"unknown bundle schema {data.get('schema_version')!r}")

    mapping_fields = dict(data["mapping"])
    mapping_fields.pop("letterbox", None)
    mapping_fields.pop("identity", None)
    mapping = ScreenMapping(**mapping_fields)

    offset = float(data["touch_offset_ms"])
    points = [
        TouchPoint(
            t_ms=float(p["t_ms"]) - offset,
            x=float(p["x"]),
            y=float(p["y"]),
            pressure=float(p["pressure"]),
            size=float(p.get("size", 0.0)),
            pointer_id=int(p.get("pointer_id", 0)),
            action=str(p["action"]),
        )
        for p in data["touch"]
    ]
    track = TouchTrack(
        action=str(data["action"]),
        points=points,
        orientation_id=int(data["provenance"].get("orientation_id", 0)),
        screen_w=mapping.device_w,
        screen_h=mapping.device_h,
        source=str(data["provenance"].get("touch_source", "baked")),
    )
    return ActionBundle(
        bundle_id=str(data["bundle_id"]),
        action=str(data["action"]),
        touch=track,
        imu=[[float(v) for v in row] for row in data["imu"]],
        imu_period_ms=float(data["imu_period_ms"]),
        touch_offset_ms=offset,
        mapping=mapping,
        provenance=dict(data["provenance"]),
    )


def load_bundle(path: "str | Path") -> ActionBundle:
    from pathlib import Path as _Path

    return from_json_dict(json.loads(_Path(path).read_text()))


def compile_bundle(
    *,
    action: str,
    touch: TouchTrack,
    imu: "IMUWindow",
    mapping: ScreenMapping,
    bundle_id: Optional[str] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> ActionBundle:
    """Put a trajectory and an IMU window on one clock, in device pixels.

    The touch is placed so its DOWN lands on the window's first active frame:
    the pre-roll is inertia only, the gesture overlaps the active span, and the
    post-roll carries the hand away again.
    """

    if touch.action != imu.action:
        raise ValueError(f"touch is {touch.action!r} but IMU is {imu.action!r}")

    device_touch = mapping.track_to_device(touch)
    offset_ms = imu.active_start * imu.period_ms
    gesture_ms = device_touch.duration_ms

    # A donor gesture longer than the window's active span would run past the
    # inertia that justifies it.  Rather than truncate the touch -- which would
    # break the pointer lifecycle -- borrow from the post-roll, and only fail
    # if even the whole window is too short.
    available_ms = imu.frames * imu.period_ms - offset_ms
    if gesture_ms > available_ms:
        raise ValueError(
            f"{action}: gesture is {gesture_ms:.1f}ms but only {available_ms:.1f}ms "
            f"of IMU remains after a {offset_ms:.1f}ms pre-roll "
            f"(window {imu.frames} frames)"
        )

    prov = {
        "touch_source": touch.source,
        "imu_source": imu.source,
        "imu_active_frames": imu.active_frames,
        "imu_window_frames": imu.frames,
        "requested_duration_ms": imu.requested_duration_ms,
        "orientation_id": imu.orientation_id,
    }
    prov.update(provenance or {})

    digest_seed = f"{action}|{touch.source}|{imu.source}|{device_touch.down_xy}"
    generated_id = bundle_id or hashlib.sha256(digest_seed.encode()).hexdigest()[:16]

    return ActionBundle(
        bundle_id=generated_id,
        action=action,
        touch=device_touch,
        imu=imu.samples,
        imu_period_ms=imu.period_ms,
        touch_offset_ms=offset_ms,
        mapping=mapping,
        provenance=prov,
    )
