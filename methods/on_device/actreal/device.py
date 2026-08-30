"""Talking to the phone through adb.

Only the parts this project needs: install the target app, publish its control
socket on this machine, start it, and ask the device what it will and will not
let us do.  Nothing here requires root; whether root is available is one of the
things :func:`probe` reports rather than assumes.
"""

from __future__ import annotations

import json
import re
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from . import config

PACKAGE = "com.actreal.target"
ACTIVITY = f"{PACKAGE}/.TargetActivity"
CONTROL_PORT = 8129


class AdbError(RuntimeError):
    pass


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        return self.stdout.strip()


class Adb:
    def __init__(self, adb_path: Optional[Path] = None, serial: Optional[str] = None):
        path = Path(adb_path) if adb_path else config.ADB
        if not path.exists():
            found = shutil.which("adb")
            if not found:
                raise AdbError(
                    f"adb not found at {path} and not on PATH; "
                    "unpack platform-tools into tools/"
                )
            path = Path(found)
        self.path = path
        self.serial = serial

    def _args(self, args: Sequence[str]) -> list[str]:
        base = [str(self.path)]
        if self.serial:
            base += ["-s", self.serial]
        return base + list(args)

    def run(self, *args: str, timeout: float = 60.0, check: bool = False) -> CommandResult:
        argv = self._args(args)
        # UTF-8 explicitly, and never fail on a byte.  ``text=True`` alone
        # decodes with the *locale* encoding, which on this machine is GBK: a
        # full ``dumpsys input_method`` carries bytes GBK cannot represent, so
        # the call raised UnicodeDecodeError, the result was built with a None
        # stdout, and the failure surfaced three frames later as
        # "'NoneType' object has no attribute 'strip'". Filtering on the device
        # with grep had hidden this by only ever returning a short ASCII slice.
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        result = CommandResult(argv, proc.returncode, proc.stdout or "", proc.stderr or "")
        if check and not result.ok:
            raise AdbError(f"{' '.join(args)} failed: {result.stderr.strip() or result.stdout}")
        return result

    def shell(self, command: str, timeout: float = 60.0, check: bool = False) -> CommandResult:
        return self.run("shell", command, timeout=timeout, check=check)

    def devices(self) -> list[tuple[str, str]]:
        out = self.run("devices").stdout.splitlines()[1:]
        found = []
        for line in out:
            parts = line.split()
            if len(parts) >= 2:
                found.append((parts[0], parts[1]))
        return found

    def wait_for_device(self, timeout: float = 60.0) -> None:
        self.run("wait-for-device", timeout=timeout, check=True)

    def getprop(self, name: str) -> str:
        return self.shell(f"getprop {name}").text

    def install(self, apk: Path) -> CommandResult:
        return self.run("install", "-r", "-g", str(apk), timeout=300.0)

    def uninstall(self) -> CommandResult:
        return self.run("uninstall", PACKAGE, timeout=120.0)

    def forward(self, port: int = CONTROL_PORT) -> CommandResult:
        return self.run("forward", f"tcp:{port}", f"tcp:{port}", check=True)

    def remove_forward(self, port: int = CONTROL_PORT) -> CommandResult:
        return self.run("forward", "--remove", f"tcp:{port}")

    def start_target(self) -> CommandResult:
        return self.run("shell", "am", "start", "-n", ACTIVITY, timeout=60.0)

    def stop_target(self) -> CommandResult:
        return self.run("shell", "am", "force-stop", PACKAGE)

    def is_installed(self) -> bool:
        return PACKAGE in self.shell(f"pm list packages {PACKAGE}").text

    def screen_size(self) -> Optional[tuple[int, int]]:
        text = self.shell("wm size").text
        # "Physical size: 1080x2424" and possibly an "Override size:" line.
        best = None
        for line in text.splitlines():
            if "size:" in line:
                value = line.split("size:")[-1].strip()
                if "x" in value:
                    w, _, h = value.partition("x")
                    try:
                        best = (int(w), int(h))
                    except ValueError:
                        continue
        return best


@dataclass
class Capability:
    name: str
    available: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available, "detail": self.detail}


@dataclass
class ProbeReport:
    serial: str
    model: str
    android_release: str
    sdk_int: str
    screen: Optional[tuple[int, int]]
    capabilities: list[Capability] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[Capability]:
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "actreal_probe_v1",
            "serial": self.serial,
            "model": self.model,
            "android_release": self.android_release,
            "sdk_int": self.sdk_int,
            "screen": list(self.screen) if self.screen else None,
            "capabilities": [c.as_dict() for c in self.capabilities],
            "notes": self.notes,
        }


def probe(adb: Adb) -> ProbeReport:
    """Ask the device what input and sensor paths are actually open to us.

    Every answer here is measured, not inferred from the Android version: the
    same release behaves differently depending on how the build was signed and
    what SELinux policy shipped with it, and guessing wrong means writing an
    injector against a door that turns out to be locked.
    """

    devices = adb.devices()
    serial = adb.serial or (devices[0][0] if devices else "")
    report = ProbeReport(
        serial=serial,
        model=adb.getprop("ro.product.model"),
        android_release=adb.getprop("ro.build.version.release"),
        sdk_int=adb.getprop("ro.build.version.sdk"),
        screen=adb.screen_size(),
    )
    if not devices:
        report.notes.append("no device visible to adb")
        return report
    if any(state != "device" for _, state in devices):
        report.notes.append(f"device states: {devices}")

    # Root: not required by the design, but it decides whether the
    # higher-fidelity touch path is on the table at all.
    su = adb.shell("id")
    report.capabilities.append(
        Capability("adb_shell_uid", True, su.text)
    )
    root = adb.shell("su -c id", timeout=10.0)
    report.capabilities.append(
        Capability("root", root.ok and "uid=0" in root.stdout, root.text or root.stderr.strip())
    )

    # uinput: the non-root way to deliver a full trajectory through the real
    # input pipeline, with pressure and our own timestamps.
    uinput = adb.shell("command -v uinput", timeout=10.0)
    uinput_dev = adb.shell("ls -l /dev/uinput", timeout=10.0)
    report.capabilities.append(
        Capability(
            "uinput_command",
            bool(uinput.text),
            uinput.text or "not on PATH",
        )
    )
    report.capabilities.append(
        Capability(
            "uinput_device",
            "Permission denied" not in uinput_dev.stdout + uinput_dev.stderr
            and "No such file" not in uinput_dev.stdout + uinput_dev.stderr,
            (uinput_dev.text or uinput_dev.stderr.strip()),
        )
    )

    sendevent = adb.shell("ls /dev/input/", timeout=10.0)
    report.capabilities.append(
        Capability(
            "input_devices_readable",
            sendevent.ok and bool(sendevent.text) and "Permission denied" not in sendevent.stderr,
            (sendevent.text or sendevent.stderr.strip()).replace("\n", " ")[:200],
        )
    )

    # The always-available fallback, and the baseline the paper compares against.
    report.capabilities.append(
        Capability("input_command", bool(adb.shell("command -v input").text), "adb shell input")
    )

    # Whether uinput actually accepts a device is not the same question as
    # whether the binary exists: SELinux can allow the command and still deny
    # the ioctl.  So a real (empty) device is registered and torn down, and the
    # answer is what the kernel said, not what the PATH said.
    probe_uinput_registration(adb, report)

    report.capabilities.append(Capability("target_app_installed", adb.is_installed()))
    return report


def probe_uinput_registration(adb: Adb, report: ProbeReport) -> None:
    """Try to register a virtual touchscreen, and report whether it took.

    A minimal ``register`` with no events is pushed and fed to ``uinput``; if
    the kernel refuses the ioctl the command errors, and that error is the
    answer.  Nothing is injected, so nothing lands on the screen.
    """

    import json as _json

    register = _json.dumps(
        {
            "id": 1,
            "command": "register",
            "name": "ActReal Probe",
            "vid": 0x18D1,
            "pid": 0x4F4B,
            "bus": "usb",
            "configuration": [
                {"type": 100, "data": [1, 3, 0]},
                {"type": 101, "data": [0x14A]},
                {"type": 103, "data": [0x35, 0x36, 0x39]},
            ],
            "abs_info": [
                {"code": 0x35, "info": {"value": 0, "minimum": 0, "maximum": 1079,
                                        "fuzz": 0, "flat": 0, "resolution": 0}},
                {"code": 0x36, "info": {"value": 0, "minimum": 0, "maximum": 2399,
                                        "fuzz": 0, "flat": 0, "resolution": 0}},
                {"code": 0x39, "info": {"value": 0, "minimum": 0, "maximum": 65535,
                                        "fuzz": 0, "flat": 0, "resolution": 0}},
            ],
        }
    )
    remote = "/data/local/tmp/actreal_uinput_probe.json"
    adb.shell(f"printf '%s' {shlex.quote(register)} > {remote}", timeout=15.0)
    # `uinput -` reads one command and holds the device open until stdin ends;
    # a short timeout closes it, which is the teardown.
    result = adb.shell(f"timeout 2 uinput - < {remote}", timeout=15.0)
    adb.shell(f"rm -f {remote}")
    text = (result.stdout + result.stderr).strip()
    accepted = "Registered" in text or (result.returncode in (0, 124) and "not permitted" not in text
                                        and "Permission denied" not in text and "denied" not in text)
    report.capabilities.append(
        Capability("uinput_register", accepted, text[:200] or f"rc={result.returncode}")
    )


def probe_to_json(report: ProbeReport, path: Path) -> None:
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


def wait_for_control(
    adb: Adb,
    port: int = CONTROL_PORT,
    timeout: float = 30.0,
    *,
    start_target: bool = True,
):
    """Publish the control channel and return a connected client.

    ``start_target`` launches our own target app first.  It is off when the
    application carrying the control channel is one somebody else has to open --
    the study app publishes its channel from the capture service, which starts
    when a run starts, so launching it from here would either do nothing or
    interrupt the very session being measured.
    """

    from .control import ControlClient

    adb.forward(port)
    if start_target:
        adb.start_target()
    deadline = time.time() + timeout
    last: Optional[Exception] = None
    while time.time() < deadline:
        try:
            client = ControlClient.connect(port=port)
            client.ping()
            return client
        except Exception as error:  # socket not up yet
            last = error
            time.sleep(0.5)
    raise AdbError(f"target app control channel never came up on {port}: {last}")


@dataclass
class TouchscreenProfile:
    """What the real digitiser on this phone says it is.

    The virtual device should not be invented; it should be a copy of the one it
    stands in for.  Every place the two declarations differed showed up in the
    recording:

    * The Pixel 10's ``focal_ts`` has **no ``ABS_MT_PRESSURE`` axis at all**, so
      Android synthesises pressure 1.0 for a contact that is down -- which is why
      a human session reports pressure as the single value 1.0 with a standard
      deviation of exactly zero.  Declaring the axis and reporting a donor's
      varying pressure was less like this device, not more.
    * Its ``ABS_MT_TOUCH_MAJOR`` runs to 24239, not the 255 we picked, so the
      same raw number meant a contact ninety times too wide: measured size 0.25
      to 1.00 against a human 0.056.
    * Its position axes run to ten times the pixel count minus one, so it reports
      at a tenth of a pixel.  Rounding trajectories to whole pixels threw that
      resolution away on every sample.
    """

    name: str
    node: str
    axes: dict[int, tuple[int, int]] = field(default_factory=dict)
    keys: list[int] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)

    def maximum(self, code: int, fallback: int) -> int:
        return self.axes.get(code, (0, fallback))[1]

    @property
    def has_pressure(self) -> bool:
        return 0x3A in self.axes

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node": self.node,
            "axes": {hex(code): list(span) for code, span in sorted(self.axes.items())},
            "properties": self.properties,
            "has_pressure": self.has_pressure,
        }


# Word boundary, spelled out: the label has to match as a whole word so that
# ABS_MT_TOUCH_MAJOR is not found inside ABS_MT_TOUCH_MINOR.
WORD = chr(92) + "b"

# Axis names as `getevent -pl` prints them, for the codes this project uses.
_AXIS_CODES = {
    "ABS_MT_SLOT": 0x2F,
    "ABS_MT_TOUCH_MAJOR": 0x30,
    "ABS_MT_TOUCH_MINOR": 0x31,
    "ABS_MT_ORIENTATION": 0x34,
    "ABS_MT_POSITION_X": 0x35,
    "ABS_MT_POSITION_Y": 0x36,
    "ABS_MT_TOOL_TYPE": 0x37,
    "ABS_MT_TRACKING_ID": 0x39,
    "ABS_MT_PRESSURE": 0x3A,
}


def read_touchscreen(adb: Adb) -> Optional[TouchscreenProfile]:
    """Parse ``getevent -pl`` and return the direct touch device, if there is one.

    Chosen by ``INPUT_PROP_DIRECT`` rather than by name: that property is what
    makes Android treat a device as a screen rather than a touchpad, so it is
    also the right test for which device we are standing in for.
    """

    try:
        text = adb.shell("getevent -pl", timeout=20.0).stdout
    except Exception:
        # No device, no permission, or no adb at all.  Returning None says the
        # profile is unknown, which the caller turns into a note; guessing one
        # would put an invented digitiser into a run that reported it as read.
        return None
    profile: Optional[TouchscreenProfile] = None
    best: Optional[TouchscreenProfile] = None
    node = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("add device"):
            if profile is not None and "INPUT_PROP_DIRECT" in profile.properties:
                best = profile
            node = stripped.rsplit(":", 1)[-1].strip()
            profile = TouchscreenProfile(name="", node=node)
        elif profile is None:
            continue
        elif stripped.startswith("name:"):
            profile.name = stripped.split(":", 1)[1].strip().strip('"')
        elif stripped.startswith("INPUT_PROP_"):
            profile.properties.append(stripped)
        else:
            for label, code in _AXIS_CODES.items():
                # getevent prints a device's first axis on the same line as its
                # "ABS (0003):" heading, so the label is not always at the start.
                # Matched on word boundaries, so ABS_MT_TOUCH_MAJOR is not found
                # inside ABS_MT_TOUCH_MINOR or the other way round.
                if re.search(WORD + label + WORD, stripped):
                    low = high = 0
                    for part in stripped.split(","):
                        part = part.strip()
                        if part.startswith("min "):
                            low = int(part.split()[1])
                        elif part.startswith("max "):
                            high = int(part.split()[1])
                    profile.axes[code] = (low, high)
                    break
    if profile is not None and "INPUT_PROP_DIRECT" in profile.properties:
        best = profile
    return best
