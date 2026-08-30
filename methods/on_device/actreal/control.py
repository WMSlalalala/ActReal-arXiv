"""The host's client for the target app's control channel.

Newline-delimited JSON over a loopback socket that ``adb forward`` publishes on
this machine.  The same client talks to :mod:`actreal.simulator`, which
implements the identical protocol in process, so every line of the host side
can be exercised before a phone exists.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


class ControlError(RuntimeError):
    pass


class Transport(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class SocketTransport:
    """A line-JSON channel to the target app, which reopens itself.

    An agent run leaves this socket idle for as long as the model takes to
    answer -- tens of seconds per action, minutes across a task -- and an idle
    forwarded socket does not always survive that. Eight of one campaign's 117
    actions died on WinError 10053, "an established connection was aborted by
    the software in your host machine", and each one fell back to `adb shell
    input`: a gesture with no size, no pressure and no inertia, lost to a
    transport fault that had nothing to do with the injection.

    So a dropped connection is reopened once and the request retried. Retrying
    is safe because every command the channel carries is idempotent -- ping,
    hello, clear, dump, set-mode, and scheduling a window that is keyed by its
    own deadline. Nothing here increments anything.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8129, timeout: float = 10.0):
        self._host, self._port, self._timeout = host, port, timeout
        self._sock = None
        self._reader = None
        self._reconnects = 0
        self._open()

    def _open(self) -> None:
        self._sock = socket.create_connection(
            (self._host, self._port), timeout=self._timeout
        )
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = self._sock.makefile("r", encoding="utf-8", newline="\n")

    def _reopen(self) -> None:
        try:
            self.close()
        except OSError:
            pass
        self._open()
        self._reconnects += 1

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        for attempt in (0, 1):
            try:
                self._sock.sendall(line.encode("utf-8"))
                reply = self._reader.readline()
                if reply:
                    return json.loads(reply)
                if attempt == 0:
                    self._reopen()
                    continue
                raise ControlError("target app closed the control channel")
            except (OSError, ValueError) as error:
                if attempt == 0:
                    self._reopen()
                    continue
                raise ControlError(
                    f"control channel failed twice: {type(error).__name__}: {error}"
                ) from error
        raise ControlError("control channel failed twice")

    @property
    def reconnects(self) -> int:
        return self._reconnects

    def close(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        finally:
            if self._sock is not None:
                self._sock.close()


@dataclass
class DeviceInfo:
    model: str
    display_w: int
    display_h: int
    density_dpi: int
    usable_rect: tuple[float, float, float, float]
    clock_offset_ns: int
    uptime_ms: int
    elapsed_ns: int
    raw: dict[str, Any]

    @property
    def usable_height(self) -> float:
        return self.usable_rect[3] - self.usable_rect[1]


class ControlClient:
    """Typed wrapper over the wire protocol."""

    def __init__(self, transport: Transport):
        self._transport = transport

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 8129, timeout: float = 10.0):
        return cls(SocketTransport(host, port, timeout))

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "ControlClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _call(self, cmd: str, **fields: Any) -> dict[str, Any]:
        reply = self._transport.request({"cmd": cmd, **fields})
        if not reply.get("ok", False):
            raise ControlError(f"{cmd}: {reply.get('error', 'unknown error')}")
        return reply

    def ping(self) -> dict[str, Any]:
        return self._call("ping")

    def hello(self) -> DeviceInfo:
        reply = self._call("hello")
        device = reply["device"]
        rect = device["usable_rect"]
        return DeviceInfo(
            model=device.get("model", "?"),
            display_w=int(device["display_w"]),
            display_h=int(device["display_h"]),
            density_dpi=int(device.get("density_dpi", 0)),
            usable_rect=(
                float(rect["left"]),
                float(rect["top"]),
                float(rect["right"]),
                float(rect["bottom"]),
            ),
            clock_offset_ns=int(reply["clock_offset_ns"]),
            uptime_ms=int(reply["uptime_ms"]),
            elapsed_ns=int(reply["elapsed_ns"]),
            raw=reply,
        )

    def set_imu_mode(self, mode: str) -> None:
        if mode not in ("real", "injected"):
            raise ValueError(f"imu mode must be real or injected, got {mode!r}")
        self._call("mode", imu=mode)

    def set_background(self, frames: Sequence[Sequence[float]], period_ms: float = 10.0) -> None:
        self._call(
            "background",
            period_ms=period_ms,
            frames=[[float(v) for v in row] for row in frames],
        )

    def schedule_imu(
        self,
        frames: Sequence[Sequence[float]],
        *,
        start_uptime_ms: Optional[int] = None,
        start_in_ms: Optional[float] = None,
        period_ms: float = 10.0,
        bundle_id: str = "",
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "period_ms": period_ms,
            "bundle_id": bundle_id,
            "frames": [[float(v) for v in row] for row in frames],
        }
        if start_uptime_ms is not None:
            fields["start_uptime_ms"] = int(start_uptime_ms)
        elif start_in_ms is not None:
            fields["start_in_ms"] = float(start_in_ms)
        return self._call("imu", **fields)

    def play_touch(
        self,
        points: Sequence[dict[str, Any]],
        *,
        start_uptime_ms: Optional[int] = None,
        start_in_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"points": list(points)}
        if start_uptime_ms is not None:
            fields["start_uptime_ms"] = int(start_uptime_ms)
        elif start_in_ms is not None:
            fields["start_in_ms"] = float(start_in_ms)
        return self._call("touch", **fields)

    def clear(self, *, scheduled: bool = False) -> None:
        self._call("clear", scheduled=scheduled)

    def stats(self) -> dict[str, Any]:
        return self._call("stats")["data"]

    def dump(self, *, rows: bool = True) -> dict[str, Any]:
        return self._call("dump", rows=rows)["data"]
