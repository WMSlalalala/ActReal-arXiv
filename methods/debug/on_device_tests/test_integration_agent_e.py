"""A session driven through the framework's own modules, not through ActReal.

Calling the controller directly proves the controller works.  What has to be
proved is that the *framework* reaches it: the agent calls the names in its own
namespace, and if the patch missed one of those the run still looks normal
while that action goes out as ``adb shell input``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAMEWORK = ROOT / "third_party" / "Mobile-Agent-E"

from actreal.agent_shim import build_controller
from actreal.control import ControlClient
from actreal.frameworks import get as get_framework
from actreal.pacing import DelayPolicy, silence_framework_sleeps
from actreal.simulator import AppSimulator

pytestmark = pytest.mark.skipif(
    not (FRAMEWORK / "MobileAgentE" / "controller.py").is_file(),
    reason="Mobile-Agent-E not vendored here",
)


@pytest.fixture
def wired(synthetic_bundle_dir):
    """A controller installed over the real vendored framework."""

    if str(FRAMEWORK) not in sys.path:
        sys.path.insert(0, str(FRAMEWORK))
    for name in list(sys.modules):
        if name.startswith("MobileAgentE"):
            del sys.modules[name]

    upstream = importlib.import_module("MobileAgentE.controller")
    agents = importlib.import_module("MobileAgentE.agents")

    client = ControlClient(AppSimulator())
    controller = build_controller(
        client,
        synthetic_bundle_dir,
        fallback=_Recorder(),
        spec=get_framework("mobile-agent-e"),
        pacing=DelayPolicy(enabled=False),
    )
    controller.lead_ms = 0.0
    controller.settle_ms = 0.0
    controller._play = _instant(controller)
    controller.install()
    silence_framework_sleeps(("MobileAgentE.agents", "MobileAgentE.controller"))
    yield controller, upstream, agents
    controller.uninstall()
    for name in list(sys.modules):
        if name.startswith("MobileAgentE"):
            del sys.modules[name]


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))

        return call


def _instant(controller):
    from actreal.session import play_bundle

    def play(bundle):
        return play_bundle(controller.client, bundle, lead_ms=0.0)

    return play


def test_the_agents_namespace_reaches_actreal_not_adb(wired):
    """``agents.py`` holds its own references; this is the one that matters."""

    controller, _, agents = wired
    agents.tap("adb", 540, 1200)
    assert controller.log[-1].api == "tap"
    assert controller.log[-1].served, controller.log[-1].fallback_reason


def test_the_controller_namespace_reaches_actreal_too(wired):
    controller, upstream, _ = wired
    upstream.swipe("adb", 540, 1900, 540, 700)
    assert controller.log[-1].api == "swipe"
    assert controller.log[-1].served, controller.log[-1].fallback_reason


def test_a_whole_action_sequence_is_served_and_accounted_for(wired):
    controller, _, agents = wired
    agents.tap("adb", 540, 1200)
    agents.swipe("adb", 540, 1900, 540, 700)      # long vertical -> scroll
    agents.swipe("adb", 200, 1200, 880, 1200)     # horizontal    -> swipe
    agents.type("adb", "hello world")
    agents.back("adb")

    report = controller.report()
    assert report["actions"] == 5
    assert report["framework"] == "mobile-agent-e"
    # Four of the five carry inertia; a key event has no touch to realise.
    assert report["served_by_actreal"] == 4
    assert report["fell_back"] == 1

    routed = [
        r["plan"]["resolved_action"]
        for r in report["records"]
        if r["plan"] is not None
    ]
    assert routed == ["tap", "scroll", "swipe", "keystroke"]


def test_every_served_action_carries_a_pre_roll_and_a_matching_window(wired):
    """The inertia has to start before the touch, or the causality is inverted."""

    controller, _, agents = wired
    agents.tap("adb", 540, 1200)
    agents.swipe("adb", 540, 1900, 540, 700)
    for record in controller.log:
        if not record.served:
            continue
        bundle = record.plan["bundle"]
        if record.plan["resolved_action"] == "keystroke":
            continue  # released donors start at first contact
        assert bundle["pre_roll_ms"] > 0, record.plan


def test_uninstall_gives_the_framework_its_own_functions_back(wired):
    controller, upstream, agents = wired
    controller.uninstall()
    assert not hasattr(agents.tap, "__self__")
    assert not hasattr(upstream.tap, "__self__")


def test_the_frameworks_fixed_sleeps_no_longer_run(wired):
    """Five seconds after every tap is a rhythm; the policy owns the gap now."""

    _, _, agents = wired
    started = __import__("time").perf_counter()
    agents.back("adb")
    elapsed = __import__("time").perf_counter() - started
    # Upstream sleeps 3 s after Back inside execute_atomic_action; the
    # controller-level call should not sleep at all.
    assert elapsed < 1.0
