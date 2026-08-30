"""Driving the study app from outside it: waking, unlocking, opening a run.

These were inside the campaign script until typing moved onto the touch channel
and the ordering had to change.  The recording starts the moment a capture
session opens, and the campaign used to open it *before* handing control to the
agent -- which then spent sixty to ninety seconds loading GroundingDINO and two
OCR models.  Every session therefore began with a minute and a half of a phone
nobody was holding, and that minute is not the behaviour being modelled; it is
this project's own startup cost sitting inside the measurement.

So the agent opens the run itself, after its models are resident, and these
helpers had to be reachable from both scripts.  Nothing about them changed in
the move.

Everything here goes through interfaces the platform publishes -- input events,
activity and window dumps -- so the study app stays an app we did not write and
did not modify.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from .device import Adb

PACKAGE = "com.sensorworldmodel.collector"
MAIN = f"{PACKAGE}/.MainActivity"
TASK_ACTIVITY = ".SimulatedTaskActivity"


def resumed(adb: Adb) -> str:
    for line in adb.shell("dumpsys activity activities").text.splitlines():
        if "topResumedActivity" in line:
            for piece in line.replace("/", " ").split():
                if piece.startswith(".") and piece.endswith("Activity"):
                    return piece
    return "<unknown>"


def wake_and_unlock(adb: Adb, pin: str = "1223") -> bool:
    """Get the screen on and past the keyguard, or say it failed.

    A locked screen does not refuse touches, it *consumes* them: the keyguard
    window is what the system calls NotificationShade, it takes focus, and every
    tap lands there instead of on the app. Nothing errors. The run just records
    an empty session, and a whole measurement round was thrown away before the
    cause turned out to be a screen that had gone to sleep.
    """

    def locked() -> bool:
        for line in adb.shell("dumpsys window").text.splitlines():
            if "mDreamingLockscreen=" in line:
                return "mDreamingLockscreen=true" in line
        return False

    def screen_on() -> bool:
        return any("mScreenState=ON" in line
                   for line in adb.shell("dumpsys display").text.splitlines())

    for _ in range(3):
        if not screen_on():
            adb.shell("input keyevent 224")          # WAKEUP, never a toggle
            time.sleep(1.5)
        if not locked():
            return True
        adb.shell("input swipe 540 2000 540 900 200")
        time.sleep(1.0)
        if pin:
            adb.shell(f"input text {pin}")
            adb.shell("input keyevent 66")
            time.sleep(2.5)
    return screen_on() and not locked()


def keyboard_down(adb: Adb) -> bool:
    for _ in range(3):
        if not any("mInputShown=true" in line
                   for line in adb.shell("dumpsys input_method").text.splitlines()):
            return True
        adb.shell("input keyevent 111")
        time.sleep(1.5)
    return False


def ui_dump(adb: Adb) -> str:
    adb.shell("uiautomator dump /sdcard/ui.xml")
    return adb.shell("cat /sdcard/ui.xml").text


def ui_center(adb: Adb, *labels: str) -> Optional[tuple[int, int]]:
    import re

    dump = ui_dump(adb)
    for label in labels:
        for match in re.finditer(r"<node[^>]*>", dump):
            node = match.group(0)
            text = re.search(r'text="([^"]*)"', node)
            desc = re.search(r'content-desc="([^"]*)"', node)
            if (text and text.group(1) == label) or (desc and desc.group(1) == label):
                bounds = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', node)
                if bounds:
                    l, t, r, b = (int(v) for v in bounds.groups())
                    return ((l + r) // 2, (t + b) // 2)
    return None


def session_id(adb: Adb) -> str:
    import re

    adb.shell("uiautomator dump /sdcard/ui.xml")
    found = re.search(r'text="(\d{8}T\d{6}\.\d+Z_[0-9a-f]+)"',
                      adb.shell("cat /sdcard/ui.xml").text)
    return found.group(1) if found else ""


def open_session(adb: Adb, victim: str, used: set[str], attempts: int = 3) -> str:
    """A session of this victim's own; returns its id, or "" if none opened.

    Retried, because every step here is a UI tap read back through a uiautomator
    dump and any one of them can come back empty on a busy device -- one run
    failed at the first task and succeeded at the second with nothing changed
    between them.

    The success test is that the session on screen is one no previous victim
    recorded into, rather than merely that the identifier changed. Those differ
    when the dump before the tap came back empty: the identifier then "changes"
    from nothing to whatever was already open, and the run would be appended to
    the previous victim's session -- which is the exact failure this check
    exists to prevent.
    """

    for attempt in range(attempts):
        field = ui_center(adb, "Participant ID")
        if field:
            adb.shell(f"input tap {field[0]} {field[1] + 98}")
            time.sleep(1.5)
            adb.shell("input keyevent 123")
            for _ in range(40):
                adb.shell("input keyevent 67")
            adb.shell(f"input text {victim}")
            time.sleep(1.0)
            keyboard_down(adb)

        button = ui_center(adb, "New capture session")
        if button:
            adb.shell(f"input tap {button[0]} {button[1]}")
            time.sleep(3.0)

        current = session_id(adb)
        if current and current not in used:
            return current
        if attempt + 1 < attempts:
            print(f"  session did not open (saw {current or 'nothing'}); retrying")
            time.sleep(2.0)
    return ""


def open_task(adb: Adb, entry: tuple[int, int], settle_s: float = 7.0) -> str:
    """Tap a task card and report which activity ended up in front.

    Separated from opening the session so the caller can put the whole of its
    own startup between them: the session is what begins recording, the task is
    what the agent then acts on, and anything slow belongs before the first of
    those rather than between them.
    """

    adb.shell(f"input tap {entry[0]} {entry[1]}")
    time.sleep(settle_s)
    where = resumed(adb)
    keyboard_down(adb)
    return where
