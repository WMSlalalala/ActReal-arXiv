"""Which functions to replace, in which module, for each agent framework.

Pure data.  The physical layer is the same whichever framework is driving it;
what differs is only where the framework keeps its five primitives and what it
calls them.

The one thing worth knowing about Mobile-Agent-E is in ``rebind_modules``:
``agents.py`` does ``from MobileAgentE.controller import tap, swipe, ...``, so
those names live in the *agents* module's namespace too.  Patching only the
controller leaves the agent calling the originals -- the run looks fine and
every action goes out as ``adb shell input``.  ``tests/test_agent_shim.py``
asserts that re-import still exists in the vendored source, so an upstream
change breaks a test rather than a night of experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Our own primitives, which every framework's names map onto.
TAP = "tap"
GESTURE = "gesture"
TYPE = "type"
KEY = "key"
PINCH = "pinch"


@dataclass(frozen=True)
class FrameworkSpec:
    """Where one framework keeps its physical primitives."""

    name: str
    #: Modules whose namespace holds the names -- including any that re-import.
    rebind_modules: tuple[str, ...]
    #: Of those, the ones that must be patched for the takeover to be real.
    #: A module that re-imports the primitives is required: if it cannot be
    #: imported, the agent will call the originals and the run will look
    #: normal while every action goes out as ``adb shell input``.
    required_modules: tuple[str, ...] = ()
    #: "module:ClassName" for frameworks whose primitives are methods.
    rebind_classes: tuple[str, ...] = ()
    #: The framework's function name -> our primitive.
    bindings: dict[str, str] = field(default_factory=dict)
    #: Whether the first argument is the adb path or a bound ``self``.
    receiver: str = "adb"
    #: Where the framework's own entry point lives, for the launcher.
    entry: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def framework_names(self) -> tuple[str, ...]:
        return tuple(self.bindings)

    def primitive_for(self, name: str) -> Optional[str]:
        return self.bindings.get(name)


MOBILE_AGENT_E = FrameworkSpec(
    name="mobile-agent-e",
    rebind_modules=("MobileAgentE.controller", "MobileAgentE.agents"),
    required_modules=("MobileAgentE.controller", "MobileAgentE.agents"),
    rebind_classes=(),
    bindings={
        "tap": TAP,
        "swipe": GESTURE,
        "pinch": PINCH,
        "type": TYPE,
        "enter": KEY,
        "back": KEY,
        "home": KEY,
        "switch_app": KEY,
    },
    receiver="adb",
    entry="inference_agent_E",
    notes=(
        "agents.py re-imports the names, so both modules must be rebound",
        "pinch is ours: the framework shipped nine atomic actions and none of "
        "them had two contacts, so it could not zoom at all and any task "
        "requiring it was unreachable rather than merely slow",
        "Open_App calls tap() internally, so it is covered without special handling",
        "shortcuts expand through execute_atomic_action, so they are covered too",
    ),
)

FRAMEWORKS: dict[str, FrameworkSpec] = {MOBILE_AGENT_E.name: MOBILE_AGENT_E}


def get(name: str) -> FrameworkSpec:
    if name not in FRAMEWORKS:
        raise KeyError(f"unknown framework {name!r}; have {sorted(FRAMEWORKS)}")
    return FRAMEWORKS[name]
