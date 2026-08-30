"""Taking over the agent's physical layer, and the routing decisions it makes."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal.agent_shim import ActRealController, build_controller
from actreal.control import ControlClient
from actreal.mapping import ScreenMapping
from actreal.planner import ActionPlanner, BundleLibrary
from actreal.simulator import AppSimulator

PIXEL_W, PIXEL_H = 1080, 2424


@pytest.fixture
def controller(synthetic_bundle_dir):
    client = ControlClient(AppSimulator(display_w=PIXEL_W, display_h=PIXEL_H))
    ctrl = build_controller(client, synthetic_bundle_dir, fallback=_FakeFallback())
    ctrl.lead_ms = 0.0
    ctrl.settle_ms = 0.0
    # The simulator delivers instantly; sleeping through real playback time
    # would make the suite take minutes and prove nothing extra.
    ctrl._play = _instant_play(ctrl)
    return ctrl


def _instant_play(ctrl):
    from actreal.session import play_bundle

    def play(bundle):
        return play_bundle(ctrl.client, bundle, lead_ms=0.0)

    return play


class _FakeFallback:
    """Stands in for MobileAgentE.controller when ActReal cannot serve."""

    def __init__(self):
        self.calls = []

    def tap(self, adb, x, y):
        self.calls.append(("tap", x, y))

    def swipe(self, adb, x1, y1, x2, y2):
        self.calls.append(("swipe", x1, y1, x2, y2))

    def type(self, adb, text):
        self.calls.append(("type", text))

    def enter(self, adb):
        self.calls.append(("enter",))

    def back(self, adb):
        self.calls.append(("back",))

    def home(self, adb):
        self.calls.append(("home",))

    def switch_app(self, adb):
        self.calls.append(("switch_app",))


# -- gesture routing ----------------------------------------------------------


def _planner(bundle_dir: Path) -> ActionPlanner:
    mapping = ScreenMapping.isotropic(device_w=PIXEL_W, device_h=PIXEL_H)
    return ActionPlanner(BundleLibrary(bundle_dir), mapping)


def test_a_long_vertical_drag_routes_to_scroll(synthetic_bundle_dir):
    action, note = _planner(synthetic_bundle_dir).resolve_gesture(
        (540.0, 1900.0), (540.0, 500.0)
    )
    assert action == "scroll"
    assert "vertical" in note.detail


def test_a_short_drag_routes_to_swipe(synthetic_bundle_dir):
    action, _ = _planner(synthetic_bundle_dir).resolve_gesture(
        (540.0, 1200.0), (540.0, 1000.0)
    )
    assert action == "swipe"


def test_a_long_horizontal_drag_routes_to_swipe_not_scroll(synthetic_bundle_dir):
    action, note = _planner(synthetic_bundle_dir).resolve_gesture(
        (150.0, 1200.0), (950.0, 1200.0)
    )
    assert action == "swipe"
    assert "horizontal" in note.detail


def test_a_target_in_the_letterbox_is_reported_but_served_without_clamping(
    synthetic_bundle_dir,
):
    plan = _planner(synthetic_bundle_dir).plan("tap", (540.0, 60.0))
    assert plan.reachable
    kinds = [n.kind for n in plan.notes]
    assert "outside_donor_screen" in kinds
    # The bundle still aims where the agent asked; nothing was quietly moved.
    assert plan.bundle.touch.down_xy == pytest.approx((540.0, 60.0))


def test_consecutive_taps_prefer_different_donors(synthetic_bundle_dir):
    planner = _planner(synthetic_bundle_dir)
    first = planner.plan("tap", (540.0, 1200.0)).bundle
    second = planner.plan("tap", (540.0, 1200.0)).bundle
    if planner.library.count("tap") > 1:
        assert first.bundle_id != second.bundle_id


# -- the controller surface ---------------------------------------------------


def test_tap_is_served_by_actreal_and_lands_where_the_agent_aimed(controller):
    controller.tap("adb", 540, 1200)
    record = controller.log[-1]
    assert record.served, record.fallback_reason
    assert record.plan["bundle"]["down_xy"] == [540.0, 1200.0]
    assert not controller.fallback.calls


def test_swipe_is_routed_and_served(controller):
    controller.swipe("adb", 540, 1900, 540, 500)
    record = controller.log[-1]
    assert record.served, record.fallback_reason
    assert record.plan["resolved_action"] == "scroll"


def test_an_unreachable_tap_falls_back_and_says_why(controller):
    controller.tap("adb", 540, -30)
    record = controller.log[-1]
    assert not record.served
    assert "outside the mapped rectangle" in record.fallback_reason
    assert controller.fallback.calls == [("tap", 540, -30)]


def test_typing_still_goes_through_the_framework_and_gets_inertia(controller):
    controller.type("adb", "hello")
    record = controller.log[-1]
    assert record.served
    # The text itself is the framework's job; ActReal supplies the motion.
    assert controller.fallback.calls == [("type", "hello")]


def test_key_events_are_recorded_as_holes_with_no_inertia(controller):
    controller.back("adb")
    controller.home("adb")
    for record in controller.log:
        assert not record.served
        assert "key event" in record.fallback_reason
    assert [c[0] for c in controller.fallback.calls] == ["back", "home"]


def test_the_report_separates_served_actions_from_fallbacks(controller):
    controller.tap("adb", 540, 1200)
    controller.tap("adb", 540, -30)
    controller.back("adb")
    report = controller.report()
    assert report["actions"] == 3
    assert report["served_by_actreal"] == 1
    assert report["fell_back"] == 2


# -- patching -----------------------------------------------------------------


def _fake_agent_framework():
    """Mirror Mobile-Agent-E's import shape: agents.py imports the names."""

    controller = types.ModuleType("MobileAgentE.controller")
    for name in ("tap", "swipe", "type", "enter", "back", "home", "switch_app"):
        setattr(controller, name, lambda *a, _n=name: ("original", _n))
    agents = types.ModuleType("MobileAgentE.agents")
    for name in ("tap", "swipe", "type", "enter", "back", "home", "switch_app"):
        setattr(agents, name, getattr(controller, name))
    package = types.ModuleType("MobileAgentE")
    sys.modules["MobileAgentE"] = package
    sys.modules["MobileAgentE.controller"] = controller
    sys.modules["MobileAgentE.agents"] = agents
    return controller, agents


def test_install_rebinds_both_the_controller_and_the_agents_namespace(controller):
    module_controller, module_agents = _fake_agent_framework()
    try:
        rebound = controller.install()
        assert set(rebound) == {"MobileAgentE.controller", "MobileAgentE.agents"}
        # The one that actually matters: agents.py holds its own reference.
        assert module_agents.tap == controller.tap
        assert module_controller.tap == controller.tap
        controller.uninstall()
        assert module_agents.tap != controller.tap
    finally:
        for name in ("MobileAgentE", "MobileAgentE.controller", "MobileAgentE.agents"):
            sys.modules.pop(name, None)


def test_install_fails_loudly_when_the_framework_is_not_importable(controller):
    """A patch that lands nowhere must not look like success.

    Silently rebinding nothing leaves every action going out as
    ``adb shell input`` while the run reports itself as normal.
    """

    from dataclasses import replace as _replace

    from actreal.frameworks import MOBILE_AGENT_E

    for name in ("MobileAgentE", "MobileAgentE.controller", "MobileAgentE.agents"):
        sys.modules.pop(name, None)
    absent = _replace(
        MOBILE_AGENT_E,
        rebind_modules=("definitely_not_a_module",),
        required_modules=("definitely_not_a_module",),
    )
    with pytest.raises(ImportError, match="required module"):
        controller.install(absent)


def test_a_required_module_that_fails_to_import_is_not_a_partial_success(
    synthetic_bundle_dir,
):
    """The failure this layer exists to prevent.

    ``agents.py`` re-imports the primitives, so if it cannot be imported --
    a missing ``cv2``, say -- patching only ``controller`` leaves the agent
    calling the originals.  The run looks entirely normal and every action
    goes out as ``adb shell input``.  Half a patch has to be an error.
    """

    from dataclasses import replace as _replace

    from actreal.control import ControlClient
    from actreal.frameworks import MOBILE_AGENT_E
    from actreal.simulator import AppSimulator

    controller_module, agents_module = _fake_agent_framework()
    # The one that matters raises on import, the way a missing dependency does.
    sys.modules["MobileAgentE.agents"] = None
    ctrl = build_controller(
        ControlClient(AppSimulator()), synthetic_bundle_dir, fallback=_FakeFallback()
    )
    try:
        with pytest.raises(ImportError, match="adb shell input"):
            ctrl.install(MOBILE_AGENT_E)
    finally:
        for name in ("MobileAgentE", "MobileAgentE.controller", "MobileAgentE.agents"):
            sys.modules.pop(name, None)


def test_a_class_based_framework_is_patched_on_the_class(controller):
    """Some frameworks keep the primitives as methods, not module functions.

    Patching the class covers every instance whenever it was constructed,
    which module patching cannot do for a controller built inside a function.
    """

    from dataclasses import replace as _replace

    from actreal.frameworks import MOBILE_AGENT_E

    module = types.ModuleType("fake_framework")

    class FakeController:
        def __init__(self):
            self.adb_path = "adb"

        def tap(self, x, y):
            return ("original", x, y)

    module.FakeController = FakeController
    sys.modules["fake_framework"] = module
    try:
        spec = _replace(
            MOBILE_AGENT_E,
            rebind_modules=(),
            required_modules=(),
            rebind_classes=("fake_framework:FakeController",),
        )
        rebound = controller.install(spec)
        assert rebound == {"fake_framework:FakeController": ["tap"]}
        instance = FakeController()
        instance.tap(540, 1200)
        assert controller.log[-1].api == "tap"
        assert controller.log[-1].served
        controller.uninstall()
        assert FakeController().tap(1, 2) == ("original", 1, 2)
    finally:
        sys.modules.pop("fake_framework", None)


UPSTREAM = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "Mobile-Agent-E"
    / "MobileAgentE"
)


def test_upstream_still_imports_the_names_into_agents():
    """The patch is built on this; if upstream changes it, fail here.

    ``agents.py`` does ``from MobileAgentE.controller import tap, swipe, ...``,
    so those names live in the agents module's own namespace and patching only
    the controller would leave the agent on ``adb shell input``.  A framework
    update that stops re-importing should break this test, not the run.
    """

    if not (UPSTREAM / "agents.py").is_file():
        pytest.skip("Mobile-Agent-E upstream source is not redistributed")
    source = (UPSTREAM / "agents.py").read_text()
    assert re.search(
        r"from\s+MobileAgentE\.controller\s+import\s+[^\n]*\btap\b", source
    ), "upstream no longer imports tap into agents; revisit the patch targets"


def test_upstream_primitives_still_have_the_signatures_we_replace():
    if not (UPSTREAM / "controller.py").is_file():
        pytest.skip("Mobile-Agent-E upstream source is not redistributed")
    source = (UPSTREAM / "controller.py").read_text()
    for pattern in (
        r"def tap\(adb_path, x, y\)",
        r"def swipe\(adb_path, x1, y1, x2, y2\)",
        r"def type\(adb_path, text\)",
        r"def back\(adb_path\)",
        r"def home\(adb_path\)",
    ):
        assert re.search(pattern, source), f"upstream signature changed: {pattern}"


def test_shortcuts_still_route_through_the_primitives_we_patch():
    """Compound actions expand into atomic ones, so patching five functions covers them.

    ``Tap_Type_and_Enter`` and any shortcut the agent evolves for itself run
    through ``execute_atomic_action``; if that stopped being true, a whole
    class of actions would slip past ActReal unnoticed.
    """

    if not (UPSTREAM / "agents.py").is_file():
        pytest.skip("Mobile-Agent-E upstream source is not redistributed")
    source = (UPSTREAM / "agents.py").read_text()
    assert "atomic_action_sequence" in source
    assert source.count("self.execute_atomic_action(") >= 2
