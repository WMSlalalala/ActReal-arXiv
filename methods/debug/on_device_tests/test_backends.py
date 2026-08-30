"""The two touch backends, and what the root one is about to send.

The in-app backend is checked end to end against the simulator.  The root one
cannot be, because only a phone can say whether it will accept the stream --
so what is checked here is that the stream *says* what the plan said, by
decoding it independently and comparing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actreal.bundle import load_bundle
from actreal.control import ControlClient
from actreal.inject import InAppTouchBackend, RootTouchBackend
from actreal.inject.frida_imu import DEFAULT_SCRIPT
from actreal.inject.root import (
    ABS_MT_POSITION_X,
    ABS_MT_TRACKING_ID,
    BTN_TOUCH,
    EV_ABS,
    EV_KEY,
    EV_SYN,
    PRESSURE_MAX,
    SYN_REPORT,
    build_sendevent_script,
    build_uinput_stream,
    parse_uinput_stream,
)
from actreal.session import play_bundle, verify
from actreal.simulator import AppSimulator
from actreal.touch_track import DOWN, UP, TouchPoint, TouchTrack

PIXEL_W, PIXEL_H = 1080, 2424


def test_default_native_imu_hook_resolves_inside_the_release():
    assert DEFAULT_SCRIPT.is_file(), DEFAULT_SCRIPT
    assert DEFAULT_SCRIPT.name == "imu_native_hook.js"
    assert DEFAULT_SCRIPT.parent.name == "frida"
    assert DEFAULT_SCRIPT.parent.parent.name == "hooks"


def _bundle(bundle_dir: Path, action: str = "tap"):
    files = sorted(p for p in bundle_dir.glob(f"synthetic_{action}_*.json"))
    if not files:
        pytest.skip(f"no {action} bundles baked")
    return load_bundle(files[0])


def _track() -> TouchTrack:
    return TouchTrack(
        action="scroll",
        points=[
            TouchPoint(0.0, 540.0, 1600.0, 0.4, 0.0, 0, DOWN),
            TouchPoint(40.0, 545.0, 1400.0, 0.6, 0.0, 0, "MOVE"),
            TouchPoint(90.0, 550.0, 1100.0, 0.5, 0.0, 0, "MOVE"),
            TouchPoint(140.0, 552.0, 900.0, 0.3, 0.0, 0, UP),
        ],
        orientation_id=0,
        screen_w=float(PIXEL_W),
        screen_h=float(PIXEL_H),
    )


# -- what each backend claims -------------------------------------------------


def test_each_backend_states_what_it_cannot_reproduce():
    """A recording made one way must never be presented as the other."""

    inapp = InAppTouchBackend(ControlClient(AppSimulator())).describe()
    assert "input device id" in inapp["does_not_reproduce"]
    assert inapp["requires"] == []
    assert "target app only" in inapp["targets"]

    root = RootTouchBackend(None, device_w=PIXEL_W, device_h=PIXEL_H).describe()
    assert "input device id" in root["reproduces"]
    assert root["targets"] == "any app"
    assert root["requires"]


def test_the_root_backend_rejects_a_mode_it_cannot_compile():
    with pytest.raises(ValueError, match="unknown mode"):
        RootTouchBackend(None, device_w=PIXEL_W, device_h=PIXEL_H, mode="magic")


# -- the in-app backend -------------------------------------------------------


def test_the_inapp_backend_delivers_every_planned_point(synthetic_bundle_dir):
    simulator = AppSimulator(display_w=PIXEL_W, display_h=PIXEL_H)
    client = ControlClient(simulator)
    client.set_imu_mode("injected")
    bundle = _bundle(synthetic_bundle_dir, "tap")
    client.clear()
    receipt = play_bundle(client, bundle, lead_ms=0.0)
    report = verify(client.dump(), bundle, receipt)
    assert report.ok, report.as_dict()


# -- the root backend's stream ------------------------------------------------


def test_the_stream_registers_a_touchscreen_before_it_writes_to_one():
    stream = build_uinput_stream(_track(), device_w=PIXEL_W, device_h=PIXEL_H)
    first = json.loads(stream.lines[0])
    assert first["command"] == "register"
    codes = {entry["code"] for entry in first["abs_info"]}
    # Without pressure and tracking id the device is not a multi-touch
    # digitiser and Android will not read protocol B from it.
    assert ABS_MT_POSITION_X in codes and ABS_MT_TRACKING_ID in codes
    assert any(e["info"]["maximum"] == PIXEL_W * 10 - 1 for e in first["abs_info"])


def test_the_waits_are_device_side_commands_not_host_sleeps():
    """One adb round trip per event costs more than the gaps being reproduced."""

    track = _track()
    stream = build_uinput_stream(track, device_w=PIXEL_W, device_h=PIXEL_H)
    delays = [json.loads(l) for l in stream.lines if '"delay"' in l]
    assert delays, "a multi-point gesture must carry waits"
    assert sum(d["duration"] for d in delays) == pytest.approx(track.duration_ms, abs=1.0)


def test_the_stream_decodes_back_into_the_gesture_it_was_built_from():
    track = _track()
    stream = build_uinput_stream(track, device_w=PIXEL_W, device_h=PIXEL_H)
    back = parse_uinput_stream(stream.text(), device_w=PIXEL_W, device_h=PIXEL_H)

    # A moved lift is encoded as a final MOVE frame followed by the protocol-B
    # lift frame, so the decoded stream may contain one more point than the
    # semantic track while preserving its endpoints and duration.
    assert back.down_xy == pytest.approx(track.down_xy, abs=1.0)
    assert back.up_xy == pytest.approx(track.up_xy, abs=1.0)
    assert back.duration_ms == pytest.approx(track.duration_ms, abs=1.0)
    assert back.points[0].action == DOWN
    assert back.points[-1].action == UP


def test_pressure_survives_the_scale_to_the_digitisers_range():
    track = _track()
    stream = build_uinput_stream(track, device_w=PIXEL_W, device_h=PIXEL_H)
    back = parse_uinput_stream(stream.text(), device_w=PIXEL_W, device_h=PIXEL_H)
    for planned, decoded in zip(track.points[:-1], back.points[:-1]):
        assert decoded.pressure == pytest.approx(planned.pressure, abs=1.0 / PRESSURE_MAX)


def test_the_pointer_is_always_lifted():
    stream = build_uinput_stream(_track(), device_w=PIXEL_W, device_h=PIXEL_H)
    events = [
        json.loads(line)["events"] for line in stream.lines if '"inject"' in line
    ]
    last = events[-1]
    assert EV_KEY in last and BTN_TOUCH in last
    # A stream that leaves the finger down wedges the touchscreen.
    assert last[last.index(BTN_TOUCH) + 1] == 0


def test_a_gesture_that_never_lifts_cannot_be_built_at_all():
    """A stream that leaves the finger down wedges the touchscreen.

    The rule lives in TouchTrack rather than in the backend, so no backend can
    be handed a gesture that ends mid-contact.
    """

    with pytest.raises(ValueError, match="last point must be an UP"):
        TouchTrack(
            action="scroll",
            points=[
                TouchPoint(0.0, 540.0, 1600.0, 0.4, 0.0, 0, DOWN),
                TouchPoint(40.0, 545.0, 1400.0, 0.6, 0.0, 0, "MOVE"),
            ],
            orientation_id=0,
            screen_w=float(PIXEL_W),
            screen_h=float(PIXEL_H),
        )


def test_the_parser_refuses_a_stream_that_never_lifts():
    """The decoder is the check on the encoder, so it must not forgive this."""

    from actreal.inject.root import EV_ABS as _ABS

    half = (
        json.dumps({"id": 1, "command": "register", "name": "x", "vid": 1, "pid": 1,
                    "bus": "usb", "configuration": [], "abs_info": []})
        + "\n"
        + json.dumps({"id": 1, "command": "inject", "events": [
            _ABS, ABS_MT_TRACKING_ID, 1,
            _ABS, ABS_MT_POSITION_X, 540,
            _ABS, 0x36, 1200,
            EV_SYN, SYN_REPORT, 0]})
        + "\n"
    )
    with pytest.raises(ValueError, match="never lifts"):
        parse_uinput_stream(half, device_w=PIXEL_W, device_h=PIXEL_H)


def test_the_parser_refuses_a_ragged_event_list():
    bad = json.dumps({"id": 1, "command": "inject", "events": [3, 53]}) + "\n"
    with pytest.raises(ValueError, match="whole"):
        parse_uinput_stream(bad, device_w=PIXEL_W, device_h=PIXEL_H)


def test_coordinates_are_clamped_to_the_panel():
    off_screen = TouchTrack(
        action="tap",
        points=[
            TouchPoint(0.0, -50.0, 99999.0, 1.0, 0.0, 0, DOWN),
            TouchPoint(30.0, -50.0, 99999.0, 1.0, 0.0, 0, UP),
        ],
        orientation_id=0,
        screen_w=float(PIXEL_W),
        screen_h=float(PIXEL_H),
    )
    stream = build_uinput_stream(off_screen, device_w=PIXEL_W, device_h=PIXEL_H)
    back = parse_uinput_stream(stream.text(), device_w=PIXEL_W, device_h=PIXEL_H)
    for point in back.points:
        assert 0 <= point.x < PIXEL_W
        assert 0 <= point.y < PIXEL_H


def test_a_synthetic_bundle_compiles_and_decodes_to_where_it_was_aimed(
    synthetic_bundle_dir,
):
    bundle = _bundle(synthetic_bundle_dir, "scroll")
    backend = RootTouchBackend(None, device_w=PIXEL_W, device_h=PIXEL_H)
    back = parse_uinput_stream(backend.compile(bundle), device_w=PIXEL_W, device_h=PIXEL_H)
    assert back.down_xy == pytest.approx(bundle.touch.down_xy, abs=1.0)
    assert back.up_xy == pytest.approx(bundle.touch.up_xy, abs=1.0)
    assert back.duration_ms == pytest.approx(bundle.touch.duration_ms, abs=2.0)


# -- the sendevent fallback ---------------------------------------------------


def test_the_sendevent_script_runs_on_the_device_not_the_host():
    script = build_sendevent_script(
        _track(), device_node="/dev/input/event2", device_w=PIXEL_W, device_h=PIXEL_H
    )
    assert script.startswith("#!/system/bin/sh")
    # Sleeps inside the script, so the phone keeps the clock.
    assert "sleep " in script
    assert script.count("sendevent") > 4
    assert script.rstrip().endswith("0 0 0")


def test_the_sendevent_script_quotes_the_device_node():
    script = build_sendevent_script(
        _track(), device_node="/dev/input/ev 2", device_w=PIXEL_W, device_h=PIXEL_H
    )
    assert "'/dev/input/ev 2'" in script
