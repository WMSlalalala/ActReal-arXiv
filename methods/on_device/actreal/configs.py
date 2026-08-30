"""The two configurations, as objects rather than as a convention.

Everything below this module is a backend: something that puts touches on a
screen, or something that puts inertia into an app.  A *configuration* is a
chosen pair of them plus the one clock they are both scheduled against, and it
exists because the pairing is not free -- each half constrains what the other
half can honestly claim.

configuration one, no elevated privileges
    Touch goes through the real input pipeline if the device will open
    ``uinput`` to the shell user, and through the app's own dispatch if it will
    not.  Inertia goes in at our target app's intake, exactly, on planned
    timestamps.  The reach of this configuration is our own application: the
    IMU half has no way to touch anything else.  What it measures is what an
    unprivileged agent can generate, not what it can generate *anywhere*.

configuration two, rooted and instrumented
    Touch goes through a virtual input device.  Inertia is substituted inside
    the target process at the last framework hop before its callback.  The
    reach is any application, and the price is that the sample rate belongs to
    the target -- a hook edits deliveries and cannot invent them.

Both are scheduled by the same code against the same measured
:class:`~actreal.clock.Timebase`, and both carry a ``describe()`` that says what
they reproduce and what they do not.  A run is never reported without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from .clock import LaunchCalibration, Timebase
from .device import read_touchscreen
from .inject.imu import BusImuBackend, ImuBackend
from .inject.inapp import InAppTouchBackend
from .inject.root import RootTouchBackend

NON_ROOT = "non_root"
ROOTED = "rooted"


@dataclass
class ConfigNote:
    """Something about how this configuration was assembled that changes a claim."""

    kind: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class Configuration:
    """A touch backend, an IMU backend, and the clock they share."""

    name: str
    touch: Any
    imu: ImuBackend
    timebase: Timebase
    launch: LaunchCalibration = field(default_factory=LaunchCalibration.none)
    notes: list[ConfigNote] = field(default_factory=list)

    def note(self, kind: str, detail: str) -> None:
        self.notes.append(ConfigNote(kind, detail))

    @staticmethod
    def _reaches_third_party(described: dict[str, Any]) -> bool:
        """Whether a backend can act on an application we did not write.

        Backends state this outright.  Inferring it from the ``targets`` string
        was wrong in both directions: a hook attached to one package prints that
        package's name and would have been read as unable to leave our own app.
        """

        if "third_party_capable" in described:
            return bool(described["third_party_capable"])
        return described.get("targets") == "any app"

    @property
    def touch_reaches_third_party(self) -> bool:
        return self._reaches_third_party(self.touch.describe())

    @property
    def imu_reaches_third_party(self) -> bool:
        return self._reaches_third_party(self.imu.describe())

    @property
    def reach(self) -> str:
        """The narrower of the two halves, because an action needs both.

        This is the number that gets misreported most easily.  Configuration
        two's touch reaches any app, but if its IMU half is not attached, an
        action delivered to a third-party app carries that app's real inertia
        and ours goes nowhere.  So the reach of a configuration is the reach of
        its weaker half, and it is computed rather than declared.
        """

        if self.touch_reaches_third_party and self.imu_reaches_third_party:
            return "any app"
        # Name the half that is narrower rather than a fixed string.  The
        # sentence used to say "the ActReal target app only" whichever app was
        # actually on the other end, which stopped being true the moment the
        # study app carried the same intake -- a run against it reported the
        # wrong application in its own receipt.
        narrower = (
            self.imu.describe() if not self.imu_reaches_third_party
            else self.touch.describe()
        )
        named = narrower.get("targets", "")
        return f"{named} only" if named else "one application only"

    def describe(self) -> dict[str, Any]:
        return {
            "configuration": self.name,
            "reach": self.reach,
            "touch": self.touch.describe(),
            "imu": self.imu.describe(),
            "timebase": self.timebase.as_dict(),
            "launch_calibration": self.launch.as_dict(),
            "notes": [n.as_dict() for n in self.notes],
        }


def _capability(probe_report, name: str) -> bool:
    """Whether the device said yes, treating a missing probe as a no.

    An absent probe is not permission to assume the door is open.  It is the
    reason the assembly below falls back and writes a note saying it did.
    """

    if probe_report is None:
        return False
    if hasattr(probe_report, "get"):
        cap = probe_report.get(name)
        if cap is None:
            return False
        return bool(getattr(cap, "available", False))
    return False


def non_root(
    client,
    *,
    timebase: Timebase,
    adb=None,
    device_w: int = 0,
    device_h: int = 0,
    probe_report=None,
    force_inapp: bool = False,
) -> Configuration:
    """Assemble configuration one against whatever this device actually allows.

    The touch half has two candidates and the choice is not ours to make in
    advance: whether an unprivileged shell may open ``uinput`` depends on the
    SELinux policy that shipped with the build, not on the Android version.  So
    the probe's live registration test decides, and if it decided against, the
    fallback is recorded as a note -- because a run on the in-app path
    reproduces strictly less than a run on the input pipeline, and the two must
    never be pooled.
    """

    imu = BusImuBackend(client)
    uinput_ok = _capability(probe_report, "uinput_register")

    if force_inapp or not uinput_ok or adb is None or not (device_w and device_h):
        config = Configuration(
            name=NON_ROOT, touch=InAppTouchBackend(client), imu=imu, timebase=timebase
        )
        if force_inapp:
            reason = "asked for the in-app path explicitly"
        elif adb is None or not (device_w and device_h):
            reason = "no adb connection or screen size to build a uinput stream with"
        elif probe_report is None:
            reason = "no probe was run, so uinput is treated as closed"
        else:
            cap = probe_report.get("uinput_register")
            reason = f"uinput refused registration: {getattr(cap, 'detail', '')}"
        config.note(
            "touch_path",
            f"touch is dispatched inside the app rather than through the input "
            f"pipeline ({reason}); device id, source flags and driver batching "
            f"are ours, not a driver's",
        )
        return config

    # Ask the phone what its own digitiser is, and mirror it.  Left to
    # defaults the virtual device declared a pressure axis this phone does not
    # have and a touch-major range ninety-five times too small, both of which
    # were measurable in the recording.
    profile = read_touchscreen(adb)
    config = Configuration(
        name=NON_ROOT,
        touch=RootTouchBackend(
            adb, device_w=device_w, device_h=device_h, mode="uinput", profile=profile
        ),
        imu=imu,
        timebase=timebase,
    )
    if profile is None:
        config.note(
            "touchscreen_profile",
            "the phone's own digitiser could not be read, so the virtual device "
            "uses defaults; size and pressure will not match a human recording",
        )
    else:
        config.note(
            "touchscreen_profile",
            f"virtual device mirrors {profile.name}: "
            f"{'no pressure axis' if not profile.has_pressure else 'pressure axis present'}, "
            f"touch-major max {profile.maximum(0x30, 0)}, "
            f"position max {profile.maximum(0x35, 0)}x{profile.maximum(0x36, 0)}",
        )
    config.note(
        "touch_path",
        "touch goes through the real input pipeline via uinput opened to the "
        "shell user; no root was used or needed",
    )
    config.note(
        "imu_reach",
        "the IMU half is the target app's own intake, so this configuration "
        "reaches our application only however far the touch half could reach",
    )
    return config


def rooted(
    adb,
    *,
    package: str,
    period_ms: float,
    device_w: int,
    device_h: int,
    timebase: Optional[Timebase] = None,
    probe_report=None,
    script_path: Optional[Path] = None,
    spawn: bool = False,
    touch_mode: str = "",
) -> Configuration:
    """Assemble configuration two: virtual input device plus in-process hook.

    The clock is measured from inside the target here, not through the target
    app's control channel, because in this configuration there may be no such
    app -- the point of it is that the target can be one we did not write.  The
    hook reads both Android clocks in the target's own process and the host
    keeps the narrowest sample, which is the same procedure configuration one
    runs through a different door.
    """

    from .inject.frida_imu import DEFAULT_SCRIPT, FridaImuBackend

    imu = FridaImuBackend.connect(
        package,
        period_ms=period_ms,
        adb=adb,
        script_path=Path(script_path) if script_path else DEFAULT_SCRIPT,
        spawn=spawn,
    )
    measured = timebase if timebase is not None else imu.timebase()

    mode = touch_mode
    if not mode:
        # uinput first even here: it needs no elevated privilege when the policy
        # allows it, and its timing is better.  The /dev/input path is the
        # fallback, and it costs a shell round trip per event unless the whole
        # gesture is compiled first, which is why root.py compiles it.
        mode = "uinput" if _capability(probe_report, "uinput_register") else "sendevent"

    # Mirror the phone's own digitiser here too.  Configuration one reads this
    # profile and configuration two did not, which would have quietly undone
    # every touch fix that path earned: without it the virtual device declares
    # no INPUT_PROP_DIRECT and Android classifies it as a touchpad, so every
    # coordinate lands at the screen centre and arrives as SOURCE_MOUSE (8194)
    # rather than SOURCE_TOUCHSCREEN (4098); the touch-major range is out by a
    # factor of ninety-five; and a pressure axis this phone does not have gets
    # advertised. The two configurations have to differ in their inertial half
    # and nowhere else, or their numbers cannot be compared.
    profile = read_touchscreen(adb)
    config = Configuration(
        name=ROOTED,
        touch=RootTouchBackend(
            adb, device_w=device_w, device_h=device_h, mode=mode, profile=profile
        ),
        imu=imu,
        timebase=measured,
    )
    # The launch delay, if it has been measured. Configuration one reads the
    # landing back through the app's control channel; configuration two has no
    # channel, so the packaged latency probe reads it from inside the process
    # instead -- MotionEvent::initialize, observed read-only -- and leaves the
    # number here. Without it the correction is zero and every touch trails its
    # own inertial window by about ten milliseconds, which is a constant offset
    # between the two halves and exactly what a correlation-based detector looks
    # at.
    delay_file = Path(__file__).resolve().parents[1] / "runs" / "launch_delay.json"
    if delay_file.is_file():
        try:
            # Not `measured`: that name already holds this function's Timebase,
            # and shadowing it replaced a Timebase with a dict several lines
            # before `measured.trustworthy` is read.
            delay = json.loads(delay_file.read_text(encoding="utf-8"))
            config.launch = LaunchCalibration.from_trials(
                [float(v) for v in delay.get("samples_ms", [])],
                source=f"native probe ({delay.get('trials', 0)} trials)",
            )
            config.note(
                "launch_calibration",
                f"touch launches {config.launch.latency_ms:.2f} ms early to "
                f"cancel the measured delay; spread {config.launch.spread_ms:.2f} ms",
            )
        except (json.JSONDecodeError, OSError, ValueError) as error:
            config.note("launch_calibration",
                        f"{delay_file.name} unreadable ({error}); no correction applied")
    else:
        config.note(
            "launch_calibration",
            "not measured: touch trails its own inertial window by the "
            "uncorrected launch delay; run the packaged latency probe",
        )

    config.note(
        "imu_path",
        "inertia is substituted inside the target process at "
        "SensorEventQueue.dispatchSensorEvent; the elevated privilege buys "
        "process injection, not a write interface to the sensors, which Android "
        "does not have",
    )
    config.note(
        "imu_rate",
        "the delivery rate is whatever the target registered; the hook reports "
        "it and cannot change it",
    )
    if mode == "sendevent":
        config.note(
            "touch_path",
            "uinput was unavailable, so the gesture is written to the input "
            "device node instead, which does need the elevated privilege",
        )
    if not measured.trustworthy:
        config.note(
            "clock",
            f"the clock reads disagreed by "
            f"{abs(measured.offset_ns - measured.median_ns) / 1e6:.2f} ms; "
            f"alignment numbers from this session carry that uncertainty",
        )
    return config
