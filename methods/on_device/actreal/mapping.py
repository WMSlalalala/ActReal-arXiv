"""Between the trajectory's screen and the phone's screen.

Every trajectory in this project lives in HMOG's Galaxy S4 pixels, and so do
the detectors that score them.  A Pixel is a different size and a different
aspect ratio, so something has to map between the two.

The mapping is deliberately a single isotropic scale plus a centring offset,
never a per-axis stretch.  With one factor, injecting through the map and
observing back through its inverse returns the original trajectory, so speeds,
dx/dy distributions and the detectors' frozen operating points all survive.
Stretching x and y differently would change the kinematics and quietly
invalidate every threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from .touch_track import TouchTrack, screen_for_orientation


@dataclass(frozen=True)
class ScreenMapping:
    source_w: float
    source_h: float
    device_w: float
    device_h: float
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def isotropic(
        cls,
        *,
        device_w: float,
        device_h: float,
        orientation_id: int = 0,
    ) -> "ScreenMapping":
        source_w, source_h = screen_for_orientation(orientation_id)
        scale = min(device_w / source_w, device_h / source_h)
        return cls(
            source_w=source_w,
            source_h=source_h,
            device_w=float(device_w),
            device_h=float(device_h),
            scale=float(scale),
            offset_x=(device_w - source_w * scale) / 2.0,
            offset_y=(device_h - source_h * scale) / 2.0,
        )

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.scale - 1.0) < 1e-12
            and abs(self.offset_x) < 1e-9
            and abs(self.offset_y) < 1e-9
        )

    @property
    def letterbox(self) -> tuple[float, float]:
        """Unused device pixels on each axis -- the bars the mapping leaves."""

        return (
            self.device_w - self.source_w * self.scale,
            self.device_h - self.source_h * self.scale,
        )

    @property
    def usable_rect(self) -> tuple[float, float, float, float]:
        """The device rectangle that maps back into the source screen.

        A 20:9 phone is taller than the 16:9 screen every trajectory was
        recorded on, so an isotropic map leaves bars.  Device points inside
        this rectangle have a source coordinate; points in the bars do not,
        and scoring them would ask the detectors about positions that cannot
        occur on the screen they were fitted on.  The target app therefore
        lays its interactive surface out inside this rectangle.
        """

        return (
            self.offset_x,
            self.offset_y,
            self.offset_x + self.source_w * self.scale,
            self.offset_y + self.source_h * self.scale,
        )

    def contains_device_point(self, xy: Sequence[float], *, margin: float = 0.0) -> bool:
        left, top, right, bottom = self.usable_rect
        return (
            left - margin <= float(xy[0]) <= right + margin
            and top - margin <= float(xy[1]) <= bottom + margin
        )

    def clamp_device_point(self, xy: Sequence[float]) -> tuple[tuple[float, float], float]:
        """Pull a device point into the usable rectangle; report how far.

        The distance is returned rather than swallowed: a clamped target means
        the agent aimed somewhere this mapping cannot express, and the action
        will land somewhere other than where it was aimed.
        """

        left, top, right, bottom = self.usable_rect
        x = min(max(float(xy[0]), left), right)
        y = min(max(float(xy[1]), top), bottom)
        moved = ((x - float(xy[0])) ** 2 + (y - float(xy[1])) ** 2) ** 0.5
        return (x, y), moved

    def to_device(self, xy: Sequence[float]) -> tuple[float, float]:
        return (
            self.scale * float(xy[0]) + self.offset_x,
            self.scale * float(xy[1]) + self.offset_y,
        )

    def to_source(self, xy: Sequence[float]) -> tuple[float, float]:
        return (
            (float(xy[0]) - self.offset_x) / self.scale,
            (float(xy[1]) - self.offset_y) / self.scale,
        )

    def track_to_device(self, track: TouchTrack) -> TouchTrack:
        from dataclasses import replace

        points = []
        for p in track.points:
            x, y = self.to_device((p.x, p.y))
            points.append(replace(p, x=x, y=y))
        return replace(
            track,
            points=points,
            screen_w=self.device_w,
            screen_h=self.device_h,
            source=track.source + "+device_space",
        )

    def track_to_source(self, track: TouchTrack) -> TouchTrack:
        from dataclasses import replace

        points = []
        for p in track.points:
            x, y = self.to_source((p.x, p.y))
            points.append(replace(p, x=x, y=y))
        return replace(
            track,
            points=points,
            screen_w=self.source_w,
            screen_h=self.source_h,
            source=track.source + "+source_space",
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["letterbox"] = list(self.letterbox)
        data["identity"] = self.is_identity
        return data
