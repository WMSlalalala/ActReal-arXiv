"""Where this phone's keys are, so typing can be typed rather than inserted.

Until now the shim entered text through the framework's ``input text``, which
hands the string to whatever view holds focus at the IME layer and dispatches
no contacts at all.  The inertia of the victim typing was played across the
interval, so the sensor channel looked right, but the touch channel recorded
fourteen characters arriving with zero presses.  A recording where text appears
and no finger moved is not a recording of a person typing, and the victim's own
corpus does contain key presses -- so the gesture mix does not match either.

The reason it was built that way is still true: a keystroke donor's coordinates
belong to the keyboard on the phone HMOG was collected on, and replaying them
here lands them between this keyboard's keys, or below it, or on the field
itself where the contact takes focus away from what is being typed into.

What that argument rules out is the donor's *geometry*.  It says nothing about
the donor's *timing*, which is measured, real, and transfers perfectly -- and
it says nothing about where this phone's keys actually are, which Android will
say if asked.  So the split is the same one used everywhere else in this
project: the victim supplies the dynamics, the device supplies the geometry.

    key positions   this phone's IME, read from the accessibility tree
    when each key   the victim's own inter-key intervals, measured
    how long held   the victim's own press durations, measured
    inertia         the victim's own recorded presses, already stitched

Only the scatter *within* a key is invented, because no donor coordinate can
cross keyboards, and hitting the exact centre every time would be its own tell.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

Rect = tuple[int, int, int, int]

_NODE = re.compile(r"<node\b[^>]*/?>")
_ATTR = re.compile(r'(\S+?)="([^"]*)"')

# The IME package is not hard-coded: whichever one is in front owns the keys,
# and this is asked of the device rather than assumed.
_KEY_ALIASES: dict[str, str] = {
    "space": " ",
    "spacebar": " ",
    "space bar": " ",
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation mark": "!",
    "apostrophe": "'",
    "hyphen": "-",
    "at": "@",
    "underscore": "_",
    "slash": "/",
    "colon": ":",
    "semicolon": ";",
}

# Keys that do something other than insert a character.
_SHIFT_NAMES = {"shift", "caps lock", "shift key"}
_DELETE_NAMES = {"delete", "backspace"}
_SYMBOL_NAMES = {"symbol keyboard", "symbols", "more symbols", "digit keyboard"}


def _attrs(node: str) -> dict[str, str]:
    return {k: v for k, v in _ATTR.findall(node)}


def _bounds(text: str) -> Optional[Rect]:
    m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", text or "")
    if not m:
        return None
    left, top, right, bottom = (int(v) for v in m.groups())
    return (left, top, right, bottom) if right > left and bottom > top else None


@dataclass
class KeyMap:
    """The characters this keyboard can enter, and where each one lives."""

    package: str
    chars: dict[str, Rect] = field(default_factory=dict)
    shift: Optional[Rect] = None
    delete: Optional[Rect] = None
    symbols: Optional[Rect] = None
    area: Optional[Rect] = None

    def covers(self, text: str) -> tuple[bool, list[str]]:
        """Whether every character can be typed, and which cannot.

        Reported rather than worked around.  A caller that cannot type the
        whole string should fall back to inserting it, not type half of it.
        """

        missing: list[str] = []
        for ch in text:
            if ch.lower() in self.chars:
                if ch.isupper() and self.shift is None:
                    missing.append(ch)
            else:
                missing.append(ch)
        return (not missing), sorted(set(missing))

    # How far from a key's centre a contact may fall.  Tapping exact centres
    # would be its own tell -- no hand produces a lattice -- but the first
    # version scattered by a fifth of the key and clamped only at the rectangle
    # edge, and that was measured to be too wide: the centres are right (a-z
    # typed through them comes back a-z, in two different fields), while
    # contacts near the top edge landed on the row above.  Every one of those
    # errors was the key directly overhead -- o read as 9, p as 0, t as 5 --
    # because the rectangle is a measurement of the key's drawn size and the
    # part that actually belongs to it is smaller.
    #
    # So the spread is kept and the reach is cut: a contact stays inside the
    # middle of the key, where it cannot be claimed by a neighbour.
    SPREAD = 0.16
    REACH = 0.30

    def point(self, rect: Rect, rng: random.Random) -> tuple[float, float]:
        """A contact inside a key, scattered the way a thumb scatters."""

        left, top, right, bottom = rect
        width, height = right - left, bottom - top
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        x = rng.gauss(cx, width * self.SPREAD)
        y = rng.gauss(cy, height * self.SPREAD)
        x = min(cx + width * self.REACH, max(cx - width * self.REACH, x))
        y = min(cy + height * self.REACH, max(cy - height * self.REACH, y))
        return (x, y)

    def sequence(self, text: str, rng: random.Random) -> list[tuple[str, tuple[float, float]]]:
        """One contact per character, in order, with shifts inserted.

        A capital costs two presses here exactly as it costs two on a real
        keyboard, so the count of contacts matches what a person would have
        produced rather than the count of characters.
        """

        ok, missing = self.covers(text)
        if not ok:
            # Skipping the characters this keyboard cannot reach would type a
            # different string than the agent asked for, and the agent would
            # act on the belief that it typed the right one.  Half-typed text
            # is a worse failure than not typing by contact at all, so this
            # refuses and the caller falls back to inserting the whole string.
            raise KeyError(f"keyboard cannot reach: {''.join(missing)!r}")
        out: list[tuple[str, tuple[float, float]]] = []
        for ch in text:
            rect = self.chars[ch.lower()]
            if ch.isupper() and self.shift is not None:
                out.append(("shift", self.point(self.shift, rng)))
            out.append((ch, self.point(rect, rng)))
        return out


def parse_keymap(xml: str, package_hint: Optional[str] = None) -> Optional[KeyMap]:
    """Build a key map out of one accessibility dump, or ``None``.

    ``None`` means the keyboard did not describe itself -- some IMEs do not,
    and a caller must then fall back rather than guess coordinates.
    """

    nodes = _NODE.findall(xml or "")
    if not nodes:
        return None

    # Whichever package contributes the most content-described nodes low on the
    # screen is the keyboard; asking the device beats hard-coding Gboard.
    counts: dict[str, int] = {}
    for node in nodes:
        a = _attrs(node)
        desc = (a.get("content-desc") or "").strip()
        pkg = a.get("package") or ""
        if desc and pkg and len(desc) <= 24:
            counts[pkg] = counts.get(pkg, 0) + 1
    if package_hint and package_hint in counts:
        package = package_hint
    elif counts:
        package = max(counts, key=lambda k: counts[k])
    else:
        return None

    keymap = KeyMap(package=package)
    left = top = 10**9
    right = bottom = -(10**9)
    for node in nodes:
        a = _attrs(node)
        if (a.get("package") or "") != package:
            continue
        desc = (a.get("content-desc") or "").strip()
        rect = _bounds(a.get("bounds", ""))
        if not desc or rect is None:
            continue
        low = desc.lower()
        if len(desc) == 1:
            keymap.chars.setdefault(low, rect)
        elif low in _KEY_ALIASES:
            keymap.chars.setdefault(_KEY_ALIASES[low], rect)
        elif low in _SHIFT_NAMES and keymap.shift is None:
            keymap.shift = rect
        elif low in _DELETE_NAMES and keymap.delete is None:
            keymap.delete = rect
        elif low in _SYMBOL_NAMES and keymap.symbols is None:
            keymap.symbols = rect
        else:
            continue
        left, top = min(left, rect[0]), min(top, rect[1])
        right, bottom = max(right, rect[2]), max(bottom, rect[3])

    if not keymap.chars:
        return None
    keymap.area = (left, top, right, bottom)
    return keymap


_REMOTE_XML = "/sdcard/actreal_ime.xml"


def read_keymap(adb: Any, *, timeout: float = 20.0) -> Optional[KeyMap]:
    """Ask the device where its keys are, without touching the target app."""

    ime = adb.shell("settings get secure default_input_method", timeout=timeout)
    package = (ime.stdout or "").strip().split("/")[0] or None

    dump = adb.shell(f"uiautomator dump {_REMOTE_XML}", timeout=timeout)
    if not dump.ok or "dumped to" not in (dump.stdout or ""):
        return None
    got = adb.shell(f"cat {_REMOTE_XML}", timeout=timeout)
    adb.shell(f"rm -f {_REMOTE_XML}", timeout=timeout)
    if not got.ok:
        return None
    return parse_keymap(got.stdout or "", package_hint=package)


DEFAULT_KEYMAP = "config/keymap.json"


def to_dict(keymap: KeyMap) -> dict[str, Any]:
    return {
        "schema_version": "actreal_keymap_v1",
        "package": keymap.package,
        "chars": {ch: list(rect) for ch, rect in sorted(keymap.chars.items())},
        "shift": list(keymap.shift) if keymap.shift else None,
        "delete": list(keymap.delete) if keymap.delete else None,
        "symbols": list(keymap.symbols) if keymap.symbols else None,
        "area": list(keymap.area) if keymap.area else None,
    }


def from_dict(data: dict[str, Any]) -> Optional[KeyMap]:
    if data.get("schema_version") != "actreal_keymap_v1":
        return None
    def rect(value: Any) -> Optional[Rect]:
        return tuple(int(v) for v in value) if value else None  # type: ignore[return-value]
    chars = {ch: rect(v) for ch, v in (data.get("chars") or {}).items()}
    chars = {ch: r for ch, r in chars.items() if r}
    if not chars:
        return None
    return KeyMap(
        package=data.get("package", ""),
        chars=chars,  # type: ignore[arg-type]
        shift=rect(data.get("shift")),
        delete=rect(data.get("delete")),
        symbols=rect(data.get("symbols")),
        area=rect(data.get("area")),
    )


def load_keymap(path: Any = None) -> Optional[KeyMap]:
    """The layout measured once by the calibration step.

    Reading it beats dumping the tree per typing action: the dump costs two and
    a half seconds, it can fail while the keyboard is animating, and the layout
    it would return is the same one every time -- one phone, one IME, one
    orientation.  Measuring once and reusing is both faster and steadier.
    """

    from pathlib import Path as _Path
    import json as _json

    target = _Path(path) if path else _Path(__file__).resolve().parents[1] / DEFAULT_KEYMAP
    if not target.exists():
        return None
    try:
        return from_dict(_json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def describe(keymap: Optional[KeyMap]) -> dict[str, Any]:
    if keymap is None:
        return {"readable": False}
    return {
        "readable": True,
        "package": keymap.package,
        "characters": len(keymap.chars),
        "has_shift": keymap.shift is not None,
        "has_delete": keymap.delete is not None,
        "area": keymap.area,
    }
