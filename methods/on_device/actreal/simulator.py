"""The target app's protocol, reimplemented on the host.

This exists because the phone is somewhere else.  Everything the host does --
choosing a start instant, converting between the two Android clocks, laying the
touch inside the IMU window, reading the recording back and checking it against
the plan -- can be wrong in ways that only show up end to end, and none of that
needs a real device to catch.

What it deliberately does *not* model is timing jitter, driver batching, or the
input pipeline's own metadata.  Those are exactly what the on-device selftest
is for; a simulator that faked them would report a confidence it has no way to
earn.  Here every event lands on its planned instant, so a discrepancy in the
simulator is a bug in the host's arithmetic and nothing else.
"""

from __future__ import annotations

from typing import Any

from .control import Transport

# An offset that is neither zero nor round, so any code that quietly assumes
# the two clocks agree fails here rather than on the phone.
DEFAULT_CLOCK_OFFSET_NS = 987_654_321_000


class AppSimulator(Transport):
    """In-process stand-in for ``com.actreal.target``."""

    def __init__(
        self,
        *,
        display_w: int = 1080,
        display_h: int = 2424,
        density_dpi: int = 420,
        source_w: float = 1080.0,
        source_h: float = 1920.0,
        clock_offset_ns: int = DEFAULT_CLOCK_OFFSET_NS,
        start_uptime_ms: int = 5_000_000,
        model: str = "Simulated Pixel",
    ):
        self.display_w = display_w
        self.display_h = display_h
        self.density_dpi = density_dpi
        self.source_w = source_w
        self.source_h = source_h
        self.clock_offset_ns = clock_offset_ns
        self.uptime_ms = start_uptime_ms
        self.model = model

        scale = min(display_w / source_w, display_h / source_h)
        mapped_w, mapped_h = source_w * scale, source_h * scale
        left = (display_w - mapped_w) / 2.0
        top = (display_h - mapped_h) / 2.0
        self.usable_rect = (left, top, left + mapped_w, top + mapped_h)

        self.mode = "real"
        self.background: list[list[float]] = []
        self.background_period_ms = 10.0
        self.touch_rows: list[list[Any]] = []
        self.imu_rows: list[list[Any]] = []
        self._touch_seq = 0
        self._imu_seq = 0
        self.closed = False

    # -- clocks ---------------------------------------------------------------

    @property
    def elapsed_ns(self) -> int:
        return self.uptime_ms * 1_000_000 + self.clock_offset_ns

    def advance_ms(self, ms: float) -> None:
        self.uptime_ms += int(round(ms))

    # -- Transport ------------------------------------------------------------

    def close(self) -> None:
        self.closed = True

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            return {"ok": False, "error": "closed"}
        cmd = payload.get("cmd", "")
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown cmd {cmd}"}
        try:
            reply = handler(payload)
        except Exception as error:  # surfaced to the caller like the app does
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}
        reply.setdefault("ok", True)
        reply.setdefault("cmd", cmd)
        return reply

    # -- commands -------------------------------------------------------------

    def _cmd_ping(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"uptime_ms": self.uptime_ms, "elapsed_ns": self.elapsed_ns, "read_window_ns": 0}

    def _cmd_hello(self, _: dict[str, Any]) -> dict[str, Any]:
        left, top, right, bottom = self.usable_rect
        return {
            "protocol": "actreal_control_v1",
            "uptime_ms": self.uptime_ms,
            "elapsed_ns": self.elapsed_ns,
            "read_window_ns": 0,
            "clock_offset_ns": self.clock_offset_ns,
            "device": {
                "model": self.model,
                "manufacturer": "simulator",
                "display_w": self.display_w,
                "display_h": self.display_h,
                "density_dpi": self.density_dpi,
                "source_w": self.source_w,
                "source_h": self.source_h,
                "usable_rect": {"left": left, "top": top, "right": right, "bottom": bottom},
                "accel": {"present": True, "min_delay_us": 2000},
                "gyro": {"present": True, "min_delay_us": 2000},
                "real_rate_hz": 400.0,
            },
        }

    def _cmd_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.mode = "injected" if payload.get("imu") == "injected" else "real"
        return {"imu": self.mode}

    def _cmd_background(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.background = [list(map(float, row)) for row in payload.get("frames", [])]
        self.background_period_ms = float(payload.get("period_ms", 10.0))
        return {"period_ms": self.background_period_ms}

    def _cmd_imu(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.mode != "injected":
            raise RuntimeError("imu scheduled while the bus is in real mode")
        frames = [list(map(float, row)) for row in payload.get("frames", [])]
        if not frames:
            raise ValueError("imu needs at least one frame")
        for i, row in enumerate(frames):
            if len(row) != 6:
                raise ValueError(f"frame {i} has {len(row)} channels, expected 6")
        period_ms = float(payload.get("period_ms", 10.0))
        start_elapsed_ns = self._resolve_start_elapsed(payload)
        bundle_id = payload.get("bundle_id", "")

        period_ns = int(round(period_ms * 1_000_000))
        for index, frame in enumerate(frames):
            self.imu_rows.append(
                [
                    self._imu_seq,
                    start_elapsed_ns + index * period_ns,
                    *frame,
                    "injected",
                    bundle_id,
                    index,
                ]
            )
            self._imu_seq += 1
        return {
            "frames": len(frames),
            "start_elapsed_ns": start_elapsed_ns,
            "now_elapsed_ns": self.elapsed_ns,
        }

    def _cmd_touch(self, payload: dict[str, Any]) -> dict[str, Any]:
        points = payload.get("points") or []
        if not points:
            raise ValueError("touch needs at least one point")
        start_uptime_ms = int(
            payload.get(
                "start_uptime_ms",
                self.uptime_ms + round(float(payload.get("start_in_ms", 0.0))),
            )
        )
        for point in points:
            uptime_ns = (start_uptime_ms + round(float(point["t_ms"]))) * 1_000_000
            self.touch_rows.append(
                [
                    self._touch_seq,
                    uptime_ns,
                    uptime_ns + self.clock_offset_ns,
                    point.get("action", "MOVE"),
                    0,
                    1,
                    0,
                    int(point.get("pointer_id", 0)),
                    float(point["x"]),
                    float(point["y"]),
                    float(point.get("pressure", 1.0)),
                    float(point.get("size", 0.0)),
                    1,
                    -1,
                    4098,
                    0,
                ]
            )
            self._touch_seq += 1
        return {
            "points": len(points),
            "start_uptime_ms": start_uptime_ms,
            "now_uptime_ms": self.uptime_ms,
        }

    def _cmd_clear(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.touch_rows.clear()
        self.imu_rows.clear()
        self._touch_seq = 0
        self._imu_seq = 0
        return {}

    def _cmd_stats(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": {
                "imu_mode": self.mode,
                "injected_frames": len(self.imu_rows),
                "injected_max_lateness_ms": 0.0,
                "injected_mean_lateness_ms": 0.0,
                "touch_rows": len(self.touch_rows),
                "imu_rows": len(self.imu_rows),
                "touch_dropped": 0,
                "imu_dropped": 0,
                "touch_dispatched": len(self.touch_rows),
                "touch_max_lateness_ms": 0.0,
                "control_error": "",
            }
        }

    def _cmd_dump(self, payload: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {
            "touch_rows": len(self.touch_rows),
            "imu_rows": len(self.imu_rows),
            "touch_dropped": 0,
            "imu_dropped": 0,
            "clock_offset_ns": self.clock_offset_ns,
        }
        if payload.get("rows", True):
            data["touch"] = [list(row) for row in self.touch_rows]
            data["imu"] = [list(row) for row in self.imu_rows]
        return {"data": data}

    def _resolve_start_elapsed(self, payload: dict[str, Any]) -> int:
        if "start_elapsed_ns" in payload:
            return int(payload["start_elapsed_ns"])
        if "start_uptime_ms" in payload:
            return int(payload["start_uptime_ms"]) * 1_000_000 + self.clock_offset_ns
        return self.elapsed_ns + int(round(float(payload.get("start_in_ms", 0.0)) * 1_000_000))
