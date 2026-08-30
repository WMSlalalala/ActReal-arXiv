"""Configuration two, the inertial half, from the host's side.

The native hook in ``methods/hooks/frida/imu_native_hook.js`` decides *what* a
delivery carries; this decides *what the hook knows*. The two halves share one
document -- the plan --
whose schema is fixed here and parsed there, and whose only interesting property
is that it is expressed entirely on the sensor clock.  The hook never converts
between clocks and never sees a touch instant; by the time anything reaches it,
the host has already done that conversion with a measured offset, so there is
exactly one place where the two Android clocks are related and it is
:mod:`actreal.clock`.

The plan is a background stream plus a list of windows:

    background   ...............................................  (loops)
    windows                 [--- bundle A ---]      [--- B ---]

Between actions the background plays, because a hand holding a phone never
stops moving and an IMU that goes quiet between gestures is a stronger signal
than a badly shaped gesture.  Inside a window that action's own inertia plays.
Which frame belongs at a given instant is a function of the instant alone, so
the hook needs no cursor and cannot drift out of step with a target app whose
delivery rate is not ours to choose.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..bundle import ActionBundle
from ..clock import ClockSample, Timebase, measure_timebase
from .imu import ImuSchedule

PLAN_SCHEMA = "actreal_imu_plan_v1"
DEFAULT_REMOTE_PLAN = "/data/local/tmp/actreal/imu_plan.json"
_METHODS_DIR = Path(__file__).resolve().parents[3]
_FRIDA_DIR = _METHODS_DIR / "hooks" / "frida"

# The released implementation hooks below ART at
# ``android::SensorEventQueue::read`` and does not bundle a Java bridge.
NATIVE_SCRIPT = _FRIDA_DIR / "imu_native_hook.js"
DEFAULT_SCRIPT = NATIVE_SCRIPT


@dataclass
class PlanWindow:
    bundle_id: str
    start_elapsed_ns: int
    frames: list[list[float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "start_elapsed_ns": int(self.start_elapsed_ns),
            "frames": [[round(float(v), 6) for v in row] for row in self.frames],
        }


@dataclass
class ImuPlan:
    """Everything the hook needs, on the sensor clock, in one document."""

    period_ms: float
    background: list[list[float]] = field(default_factory=list)
    background_start_elapsed_ns: int = 0
    windows: list[PlanWindow] = field(default_factory=list)

    @property
    def period_ns(self) -> int:
        return int(round(self.period_ms * 1_000_000))

    def end_of(self, window: PlanWindow) -> int:
        return window.start_elapsed_ns + len(window.frames) * self.period_ns

    def add_window(self, window: PlanWindow) -> None:
        """Add one action, refusing an overlap rather than resolving it.

        Two windows on top of each other means two actions were scheduled into
        the same span.  Whichever one the hook then chose, the recording would
        carry inertia belonging to the other, so this is a scheduling bug and is
        raised as one.
        """

        new_end = window.start_elapsed_ns + len(window.frames) * self.period_ns
        for existing in self.windows:
            if window.start_elapsed_ns < self.end_of(existing) and existing.start_elapsed_ns < new_end:
                raise ValueError(
                    f"window {window.bundle_id} overlaps {existing.bundle_id} "
                    f"({window.start_elapsed_ns}..{new_end} vs "
                    f"{existing.start_elapsed_ns}..{self.end_of(existing)})"
                )
        self.windows.append(window)
        self.windows.sort(key=lambda w: w.start_elapsed_ns)

    def prune_before(self, elapsed_ns: int) -> int:
        """Drop windows that have already played, so the document stays small."""

        keep = [w for w in self.windows if self.end_of(w) > elapsed_ns]
        dropped = len(self.windows) - len(keep)
        self.windows = keep
        return dropped

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "period_ms": self.period_ms,
            "background_start_elapsed_ns": int(self.background_start_elapsed_ns),
            "background": [[round(float(v), 6) for v in row] for row in self.background],
            "windows": [w.as_dict() for w in self.windows],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"))

    def frame_at(self, elapsed_ns: int) -> Optional[tuple[list[float], str, int]]:
        """The host's copy of the hook's selection rule.

        Kept deliberately: it is what the offline tests check, and a divergence
        between this and the JavaScript is the one bug in configuration two that
        no amount of on-device testing would attribute correctly -- the stream
        would simply be subtly wrong.
        """

        for window in self.windows:
            if elapsed_ns < window.start_elapsed_ns:
                break
            if elapsed_ns < self.end_of(window):
                index = (elapsed_ns - window.start_elapsed_ns) // self.period_ns
                index = max(0, min(int(index), len(window.frames) - 1))
                return window.frames[index], "window", index
        if not self.background:
            return None
        span = len(self.background) * self.period_ns
        delta = (elapsed_ns - self.background_start_elapsed_ns) % span
        index = int(delta // self.period_ns) % len(self.background)
        return self.background[index], "background", index


class FridaUnavailable(RuntimeError):
    """Raised where the reason matters, rather than at an AttributeError later."""


class FridaSession:
    """The attached script, and the two shapes frida's Python API comes in.

    Older frida exposes ``script.exports``; newer ones renamed it to
    ``exports_sync`` and made the old name asynchronous.  Getting this wrong
    fails as a coroutine nobody awaits -- silently, with the hook installed and
    every call a no-op -- so it is resolved once, here, and reported.
    """

    def __init__(self, script, api, flavour: str, session=None, device=None):
        self._script = script
        # Held, not merely used. frida's session owns the attachment, and
        # letting it fall out of scope detaches: the script is torn down and
        # the next RPC raises "script has been destroyed" -- moments after the
        # hook logged that it had installed successfully. Keeping the device
        # too, because a garbage-collected device takes its sessions with it.
        self._session = session
        self._device = device
        self.api = api
        self.flavour = flavour

    @staticmethod
    def _resolve(device, package: str):
        """A pid, because frida indexes processes by label rather than package.

        ``device.attach("com.example.app")`` matches frida's *process name*, and
        on this phone that name is the application's display label -- the
        collector runs as "Sensor Study Collector" while its package is
        com.sensorworldmodel.collector, so attaching by package raised "unable to
        find process with name" for a process that was plainly running.

        The application list carries both, so it is consulted first and the pid
        used directly; the process list is the fallback for anything without an
        application entry. The name is still accepted verbatim if neither knows
        it, so nothing that worked before stops working.
        """

        try:
            for app in device.enumerate_applications():
                if app.identifier == package and app.pid:
                    return app.pid
        except Exception:
            pass
        try:
            for proc in device.enumerate_processes():
                if proc.name == package:
                    return proc.pid
        except Exception:
            pass
        return package

    @classmethod
    def attach(
        cls,
        package: str,
        *,
        script_path: Path = DEFAULT_SCRIPT,
        spawn: bool = False,
        device_id: Optional[str] = None,
    ) -> "FridaSession":
        try:
            import frida  # type: ignore
        except ImportError as error:
            raise FridaUnavailable(
                "the frida Python package is not installed on the host; "
                "pip install frida frida-tools"
            ) from error

        path = Path(script_path)
        if not path.is_file():
            raise FridaUnavailable(f"hook script missing: {path}")

        try:
            device = frida.get_device(device_id) if device_id else frida.get_usb_device(timeout=10)
        except Exception as error:
            raise FridaUnavailable(f"no frida device: {error}") from error

        pid = None
        try:
            if spawn:
                pid = device.spawn([package])
                session = device.attach(pid)
            else:
                session = device.attach(cls._resolve(device, package))
        except Exception as error:
            raise FridaUnavailable(
                f"could not attach to {package}: {error}; "
                "frida-server must be running as root on the device"
            ) from error

        script = session.create_script(path.read_text(encoding="utf-8"))
        script.load()
        api = getattr(script, "exports_sync", None)
        flavour = "exports_sync"
        if api is None:
            api = script.exports
            flavour = "exports"
        if pid is not None:
            device.resume(pid)
        return cls(script, api, flavour, session=session, device=device)

    def __getattr__(self, name: str):
        return getattr(self.api, name)


class FridaImuBackend:
    """Configuration two: a hook inside the target process rewrites deliveries.

    What this reproduces and what it cannot is not a caveat bolted on at the
    end; it is the reason the two configurations are reported separately.  The
    values an app receives are ours exactly.  The instants at which it receives
    them, the rate, and the batching are the device's, because a hook edits
    deliveries and cannot create them.
    """

    name = "hook"

    def __init__(
        self,
        session: Optional[FridaSession],
        *,
        period_ms: float,
        adb=None,
        remote_plan: str = DEFAULT_REMOTE_PLAN,
        target_package: str = "",
    ):
        self.session = session
        self.adb = adb
        self.remote_plan = remote_plan
        self.target_package = target_package
        self.plan = ImuPlan(period_ms=period_ms)
        self.mode = "real"
        self.pushes = 0

    @classmethod
    def connect(
        cls,
        package: str,
        *,
        period_ms: float,
        adb=None,
        script_path: Path = DEFAULT_SCRIPT,
        spawn: bool = False,
        device_id: Optional[str] = None,
    ) -> "FridaImuBackend":
        session = FridaSession.attach(
            package, script_path=script_path, spawn=spawn, device_id=device_id
        )
        backend = cls(session, period_ms=period_ms, adb=adb, target_package=package)
        backend.anchor_background()
        return backend

    def describe(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        if self.session is not None:
            try:
                status = self.session.status()
            except Exception as error:
                status = {"error": str(error)}
        return {
            "backend": self.name,
            "requires": ["root", "frida-server running as root on the device"],
            "reproduces": [
                "sample values (bit-exact)",
                "continuity between actions",
                "the app's own delivery instants and jitter",
            ],
            "does_not_reproduce": [
                "the sample rate (the app's registration decides it)",
                "batching behaviour",
                "any delivery the app did not already ask for",
            ],
            "targets": self.target_package or "any app",
            "third_party_capable": True,
            "clock_conversion": "host",
            "hook_status": status,
        }

    # -- clock ---------------------------------------------------------------

    def read_clock(self) -> ClockSample:
        if self.session is None:
            raise FridaUnavailable("not attached")
        reply = self.session.clock()
        return ClockSample(
            uptime_ms=int(reply["uptime_ms"]),
            elapsed_ns=int(reply["elapsed_ns"]),
            read_window_ns=int(reply.get("read_window_ns", 0)),
        )

    def timebase(self, *, samples: int = 20, settle_s: float = 0.02) -> Timebase:
        """The measurement configuration one makes, taken from inside the target.

        Configuration one reads both clocks through the target app's control
        channel; here there is no such channel, so the hook itself reads them.
        Either way the numbers come from a process on the device rather than
        from the host, which is what makes the two configurations' alignment
        figures comparable at all.
        """

        return measure_timebase(
            self.read_clock, samples=samples, settle_s=settle_s, source="frida_hook"
        )

    def anchor_background(self) -> int:
        """Pin the background loop to a real instant so it never jumps."""

        self.plan.background_start_elapsed_ns = self.read_clock().elapsed_ns
        return self.plan.background_start_elapsed_ns

    # -- ImuBackend ----------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in ("real", "injected"):
            raise ValueError(f"imu mode must be real or injected, got {mode!r}")
        if self.session is None:
            raise FridaUnavailable("not attached")
        self.session.set_mode(mode)
        self.mode = mode

    def set_background(self, frames: Sequence[Sequence[float]], period_ms: float) -> None:
        if abs(period_ms - self.plan.period_ms) > 1e-9:
            raise ValueError(
                f"the plan is on a {self.plan.period_ms} ms grid; "
                f"a {period_ms} ms background would play at the wrong speed"
            )
        self.plan.background = [[float(v) for v in row] for row in frames]
        self.push()

    def schedule(
        self, bundle: ActionBundle, *, t0_uptime_ms: int, timebase: Timebase
    ) -> ImuSchedule:
        if abs(bundle.imu_period_ms - self.plan.period_ms) > 1e-9:
            raise ValueError(
                f"bundle {bundle.bundle_id} is on a {bundle.imu_period_ms} ms grid "
                f"but the plan is on {self.plan.period_ms} ms; the hook plays one grid"
            )
        start_elapsed_ns = timebase.elapsed_ns_at(t0_uptime_ms)
        # Windows that finished more than a few seconds ago cannot be selected
        # again, and carrying them makes every later push bigger for nothing.
        self.plan.prune_before(start_elapsed_ns - 5 * 1_000_000_000)
        self.plan.add_window(
            PlanWindow(
                bundle_id=bundle.bundle_id,
                start_elapsed_ns=start_elapsed_ns,
                frames=bundle.imu_rows(),
            )
        )
        self.push()
        return ImuSchedule(
            start_elapsed_ns=start_elapsed_ns,
            frames=bundle.imu_frames,
            period_ms=bundle.imu_period_ms,
            bundle_id=bundle.bundle_id,
            converted_by="host",
            detail={
                "timebase_offset_ns": timebase.offset_ns,
                "timebase_trustworthy": timebase.trustworthy,
                "plan_windows": len(self.plan.windows),
            },
        )

    # -- delivery ------------------------------------------------------------

    def push(self) -> dict[str, Any]:
        """Hand the plan to the hook, over RPC when attached and by file if not.

        The file path exists because a hook loaded from the frida CLI reads it
        at startup.  When this host is driving, RPC is used instead, so a plan
        is never half-written on disk while the hook is reading it.
        """

        text = self.plan.to_json()
        self.pushes += 1
        if self.session is not None:
            return self.session.load_plan(text)
        if self.adb is None:
            raise FridaUnavailable("no frida session and no adb to push the plan with")
        return self.push_file(text)

    def push_file(self, text: Optional[str] = None) -> dict[str, Any]:
        """Write the plan onto the device as a file, via a local temporary.

        Not through shell redirection: a plan is tens of kilobytes of JSON and
        the shell would have to quote all of it, which is the kind of thing that
        works until one bundle contains a character nobody thought about.
        """

        import tempfile

        payload = text if text is not None else self.plan.to_json()
        remote_dir = str(Path(self.remote_plan).parent).replace("\\", "/")
        self.adb.shell(f"mkdir -p {shlex.quote(remote_dir)}")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(payload)
            local = handle.name
        try:
            result = self.adb.run("push", local, self.remote_plan)
            if not result.ok:
                raise FridaUnavailable(f"pushing the plan failed: {result.stderr.strip()}")
        finally:
            Path(local).unlink(missing_ok=True)
        return {"pushed_to": self.remote_plan, "bytes": len(payload)}

    def status(self) -> dict[str, Any]:
        if self.session is None:
            return {"attached": False}
        return {"attached": True, **self.session.status()}
