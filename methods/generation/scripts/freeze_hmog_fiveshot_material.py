#!/usr/bin/env python3
"""Freeze the deterministic five-shot attack material for every user/action.

The attacker is assumed to hold exactly five real events per user and action.
Selection is a pure function of the event identity, so the frozen set cannot be
re-picked after seeing any detector score.  The manifest binds both signal
digests so a later build can prove it consumed the same material bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audit import sha256_array, sha256_file  # noqa: E402
from pipeline.pinch_endpoint_control import (  # noqa: E402
    PinchEndpointControlError,
    extract_live_two_pointer_endpoints,
)
from pipeline.replay_dataset_builder import (  # noqa: E402
    ACTIONS,
    SPLITS,
    InputShard,
    ReplayDatasetBuildError,
    _detector_window_duration_ms,
    _load_shard,
    load_input_dataset,
)

# v2 fixes a structural cross-pipeline join bug in keystroke material.  The
# detector/native and raw-trajectory pipelines both used ordinal event IDs,
SCHEMA = "hmog_fiveshot_attack_material_v3"
SELECTION_DOMAIN = "hmog-fiveshot-attack-material-v1"
SHOTS_PER_GROUP = 5
PERIOD_NS = 10_000_000
TOUCH_SLOP_PX = 24.0
ALIGNMENT_TOLERANCE_MS = 20.0
# Pinch material is drawn along its own span quantiles instead of by hash, so
# five recordings cover a range of pinch sizes rather than clustering at one.
PINCH_SPAN_LADDER_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
PINCH_SELECTION_RULE = (
    "widest-span quantiles 0.1/0.3/0.5/0.7/0.9 over physically unique pinch "
    "candidates, hash tie-break, then one closest-span substitution per "
    "missing scale direction; live two-pointer endpoints define span and "
    "direction"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINDINGS = (
    REPO_ROOT / "licensed_input" / "hmog" / "genuine_bindings.jsonl"
)
DEFAULT_NATIVE_MANIFEST = (
    REPO_ROOT / "licensed_input" / "hmog" / "native_genuine_manifest.jsonl"
)


class FiveShotMaterialError(RuntimeError):
    pass


def _selection_rank(event_id: str, *, seed: int) -> bytes:
    """Return the deterministic ordering key for one candidate event."""

    return hashlib.sha256(
        f"{SELECTION_DOMAIN}|{int(seed)}|{event_id}".encode("utf-8")
    ).digest()


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FiveShotMaterialError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            yield value


def _numeric_user_id(value: str) -> int:
    text = str(value)
    if not text.startswith("hmog_u"):
        raise FiveShotMaterialError(f"invalid HMOG user ID: {text}")
    return int(text.removeprefix("hmog_u"))


def _numeric_session_id(value: str) -> int:
    text = str(value)
    if "_s" not in text:
        raise FiveShotMaterialError(f"invalid HMOG session ID: {text}")
    return int(text.rsplit("_s", 1)[1])


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native_slot(event: dict[str, Any], samples: int) -> tuple[int, int]:
    start = (int(event["relative_start_ns"]) + PERIOD_NS - 1) // PERIOD_NS
    end = (int(event["relative_end_ns"]) + PERIOD_NS - 1) // PERIOD_NS
    start = max(0, min(int(start), int(samples)))
    end = max(0, min(int(end), int(samples)))
    if end <= start:
        raise FiveShotMaterialError("native keystroke interval is empty")
    return start, end


def _load_aligned_keystrokes(
    *, binding_cache: Path, native_manifest: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Resolve usable keystrokes by physical identity, never ordinal ID."""

    if not binding_cache.is_file() or not native_manifest.is_file():
        raise FiveShotMaterialError(
            "keystroke binding cache or native manifest is missing"
        )
    bindings = {
        str(row["source_cluster_id"]): row
        for row in _jsonl(binding_cache)
        if str(row.get("action", "")) == "keystroke"
    }
    if not bindings:
        raise FiveShotMaterialError("binding cache has no keystroke rows")

    native: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for session in _jsonl(native_manifest):
        audit_path = Path(str(session.get("native_session_audit", ""))).resolve()
        stream_path = Path(str(session.get("source", ""))).resolve()
        if not audit_path.is_file() or not stream_path.is_file():
            raise FiveShotMaterialError("native session audit/stream is missing")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        samples = int(audit.get("samples", -1))
        events = audit.get("event_plan", {}).get("events", [])
        if samples < 2 or not isinstance(events, list):
            raise FiveShotMaterialError(f"{audit_path}: invalid event plan")
        for ordinal, event in enumerate(events):
            if str(event.get("action", "")) != "keystroke":
                continue
            start, end = _native_slot(event, samples)
            start_ns = int(event["start_timestamp_ns"])
            end_ns = int(event["end_timestamp_ns"])
            if start_ns % 1_000_000 or end_ns % 1_000_000:
                raise FiveShotMaterialError(
                    f"{audit_path}: keystroke is not on the millisecond clock"
                )
            identity = (
                str(session["user_id"]),
                str(session["session_id"]),
                str(event["event_id"]),
                int(ordinal),
            )
            if identity in native:
                raise FiveShotMaterialError("duplicate native keystroke identity")
            native[identity] = {
                "activity_id": int(event["activity_id"]),
                "orientation_id": int(event["orientation_id"]),
                "label_start_ms": start_ns // 1_000_000,
                "label_end_ms": end_ns // 1_000_000,
                "native_start_sample": int(start),
                "native_end_sample": int(end),
                "native_stream": str(stream_path),
                "native_stream_sha256": str(session["source_sha256"]),
                "native_session_audit": str(audit_path),
                "native_session_audit_sha256": str(
                    session["native_session_audit_sha256"]
                ),
            }

    raw_sources = {Path(str(row["raw_trajectory_source"])).resolve() for row in bindings.values()}
    if len(raw_sources) != 1:
        raise FiveShotMaterialError("keystroke bindings use multiple raw archives")
    raw_source = next(iter(raw_sources))
    raw_source_sha256 = sha256_file(raw_source)
    declared_hashes = {
        str(row["raw_trajectory_source_sha256"]) for row in bindings.values()
    }
    if declared_hashes != {raw_source_sha256}:
        raise FiveShotMaterialError("raw keystroke archive hash changed")

    required = {
        "event_id",
        "user_id",
        "session_id",
        "activity_id",
        "orientation_id",
        "label_start_ms",
        "label_end_ms",
        "event_offsets",
        "event_key_offsets",
        "flat_key_index",
        "flat_action_code",
        "flat_x",
        "flat_y",
        "keycode",
        "key_is_letter",
        "key_down_ms",
        "key_up_ms",
    }
    with np.load(raw_source, allow_pickle=False) as archive:
        if required - set(archive.files):
            raise FiveShotMaterialError("raw keystroke archive is incomplete")
        values = {name: np.asarray(archive[name]) for name in required}

    raw_by_physical: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list)
    for index in range(len(values["event_id"])):
        raw_by_physical[
            (
                int(values["user_id"][index]),
                int(values["session_id"][index]),
                int(values["activity_id"][index]),
                int(values["orientation_id"][index]),
                int(values["label_start_ms"][index]),
                int(values["label_end_ms"][index]),
            )
        ].append(index)

    exclusions: Counter[str] = Counter()
    eligible: dict[str, dict[str, Any]] = {}
    remapped = 0
    for cluster_id, binding in bindings.items():
        identity = (
            str(binding["user_id"]),
            str(binding["session_id"]),
            str(binding["source_event_id"]),
            int(binding["source_event_ordinal"]),
        )
        event = native.get(identity)
        if event is None:
            exclusions["missing_native_identity"] += 1
            continue
        physical = (
            _numeric_user_id(str(binding["user_id"])),
            _numeric_session_id(str(binding["session_id"])),
            int(event["activity_id"]),
            int(event["orientation_id"]),
            int(event["label_start_ms"]),
            int(event["label_end_ms"]),
        )
        candidates = raw_by_physical.get(physical, [])
        if len(candidates) != 1:
            exclusions[
                "no_physical_raw_match" if not candidates else "ambiguous_physical_raw_match"
            ] += 1
            continue
        raw_index = int(candidates[0])
        remapped += int(raw_index != int(binding["raw_trajectory_event_index"]))

        raw_duration_ms = int(values["label_end_ms"][raw_index]) - int(
            values["label_start_ms"][raw_index]
        )
        native_samples = int(event["native_end_sample"]) - int(
            event["native_start_sample"]
        )
        native_grid_duration_ms = float((native_samples - 1) * 10)
        if abs(float(raw_duration_ms) - native_grid_duration_ms) > ALIGNMENT_TOLERANCE_MS:
            exclusions["raw_native_duration_not_grid_aligned"] += 1
            continue

        key_left, key_right = (
            int(value)
            for value in values["event_key_offsets"][raw_index : raw_index + 2]
        )
        key_down = np.asarray(
            values["key_down_ms"][key_left:key_right], dtype=np.int64
        )
        key_up = np.asarray(values["key_up_ms"][key_left:key_right], dtype=np.int64)
        keycode = np.asarray(values["keycode"][key_left:key_right], dtype=np.int64)
        is_letter = np.asarray(
            values["key_is_letter"][key_left:key_right], dtype=np.uint8
        )
        if (
            not len(key_down)
            or not (
                len(key_down) == len(key_up) == len(keycode) == len(is_letter)
            )
            or np.any(key_up <= key_down)
            or np.any(np.diff(key_down) < 0)
            or int(key_down[0]) < int(event["label_start_ms"])
            or int(key_up[-1]) > int(event["label_end_ms"])
        ):
            exclusions["invalid_key_schedule"] += 1
            continue

        flat_left, flat_right = (
            int(value)
            for value in values["event_offsets"][raw_index : raw_index + 2]
        )
        flat_key = np.asarray(
            values["flat_key_index"][flat_left:flat_right], dtype=np.int64
        )
        flat_action = np.asarray(
            values["flat_action_code"][flat_left:flat_right], dtype=np.int64
        ) & 0xFF
        flat_x = np.asarray(values["flat_x"][flat_left:flat_right], dtype=np.float64)
        flat_y = np.asarray(values["flat_y"][flat_left:flat_right], dtype=np.float64)
        max_chord = 0.0
        max_displacement = 0.0
        valid_contacts = True
        observed_keys = np.unique(flat_key[flat_key >= 0])
        if len(observed_keys) != len(key_down):
            valid_contacts = False
        for key in observed_keys:
            positions = np.flatnonzero(flat_key == int(key))
            down = positions[flat_action[positions] == 0]
            up = positions[flat_action[positions] == 1]
            if len(down) != 1 or len(up) != 1:
                valid_contacts = False
                break
            max_chord = max(
                max_chord,
                float(
                    np.hypot(
                        flat_x[int(up[-1])] - flat_x[int(down[0])],
                        flat_y[int(up[-1])] - flat_y[int(down[0])],
                    )
                ),
            )
            max_displacement = max(
                max_displacement,
                float(
                    np.max(
                        np.hypot(
                            flat_x[positions] - flat_x[int(down[0])],
                            flat_y[positions] - flat_y[int(down[0])],
                        )
                    )
                ),
            )
        if not valid_contacts:
            exclusions["invalid_key_touch_contacts"] += 1
            continue
        if max_chord > TOUCH_SLOP_PX + 1.0e-6:
            exclusions["correct_raw_endpoint_over_24px"] += 1
            continue
        if max_displacement > TOUCH_SLOP_PX + 1.0e-6:
            exclusions["correct_raw_move_over_24px"] += 1
            continue

        schedule_payload = {
            "key_down_ms": key_down.astype(int).tolist(),
            "key_up_ms": key_up.astype(int).tolist(),
            "keycode": keycode.astype(int).tolist(),
            "key_is_letter": is_letter.astype(int).tolist(),
        }
        eligible[cluster_id] = {
            **event,
            "user_id": str(binding["user_id"]),
            "session_id": str(binding["session_id"]),
            "split": str(binding["split"]),
            "native_source_event_id": str(binding["source_event_id"]),
            "raw_trajectory_event_id": str(values["event_id"][raw_index]),
            "raw_trajectory_source": str(raw_source),
            "raw_trajectory_source_sha256": raw_source_sha256,
            "raw_trajectory_event_index": raw_index,
            "raw_duration_ms": raw_duration_ms,
            "native_grid_duration_ms": native_grid_duration_ms,
            "max_key_endpoint_displacement_px": max_chord,
            "max_key_displacement_from_down_px": max_displacement,
            "key_count": int(len(key_down)),
            "letter_count": int(np.count_nonzero(is_letter)),
            "key_schedule_sha256": _canonical_sha256(schedule_payload),
            **schedule_payload,
            "binding_resolution": (
                "exact_user_session_activity_orientation_start_end_v1"
            ),
            "binding_was_remapped": bool(
                raw_index != int(binding["raw_trajectory_event_index"])
            ),
            # Same audit trail as the touch actions: the ordinal index this
            # binding carried before any physical correction.
            "ordinal_raw_trajectory_event_index": int(
                binding.get(
                    "ordinal_raw_trajectory_event_index",
                    binding["raw_trajectory_event_index"],
                )
            ),
        }

    audit = {
        "cached_keystroke_bindings": len(bindings),
        "eligible_keystroke_bindings": len(eligible),
        "raw_indices_corrected": remapped,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "touch_slop_px": TOUCH_SLOP_PX,
        "alignment_tolerance_ms": ALIGNMENT_TOLERANCE_MS,
        "physical_identity_fields": [
            "user_id",
            "session_id",
            "activity_id",
            "orientation_id",
            "label_start_ms",
            "label_end_ms",
        ],
        "raw_trajectory_source": str(raw_source),
        "raw_trajectory_source_sha256": raw_source_sha256,
    }
    return eligible, audit


def _native_keystroke_signal(
    metadata: dict[str, Any],
    cache: dict[Path, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(metadata["native_stream"])).resolve()
    if path not in cache:
        with np.load(path, allow_pickle=False) as archive:
            required = {"timestamp_ns", "imu", "sample_rate_hz", "source_session_id"}
            if required - set(archive.files):
                raise FiveShotMaterialError(f"{path}: incomplete native stream")
            if int(np.asarray(archive["sample_rate_hz"]).item()) != 100:
                raise FiveShotMaterialError(f"{path}: native stream is not 100 Hz")
            cache[path] = {
                "timestamp_ns": np.asarray(archive["timestamp_ns"], dtype=np.int64),
                "imu": np.asarray(archive["imu"], dtype=np.float32),
                "source_session_id": np.asarray(archive["source_session_id"]),
            }
    values = cache[path]
    left = int(metadata["native_start_sample"])
    right = int(metadata["native_end_sample"])
    imu = np.asarray(values["imu"][left:right], dtype=np.float32)
    timeline = np.asarray(values["timestamp_ns"][left:right], dtype=np.int64)
    if (
        len(imu) < 2
        or len(imu) != len(timeline)
        or not np.isfinite(imu).all()
        or np.any(np.diff(timeline) != PERIOD_NS)
    ):
        raise FiveShotMaterialError("native keystroke slice is not complete 100 Hz IMU")
    return np.ascontiguousarray(imu), np.ascontiguousarray(timeline)


def _pinch_geometry_by_cluster(
    *, binding_cache: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """Measure every usable pinch binding's live two-pointer geometry."""

    rows = [
        row
        for row in _jsonl(binding_cache)
        if str(row.get("action", "")) == "pinch"
        and str(row.get("physical_match", "")) == "unique"
    ]
    if not rows:
        raise FiveShotMaterialError("binding cache has no usable pinch rows")
    sources = {Path(str(row["raw_trajectory_source"])).resolve() for row in rows}
    if len(sources) != 1:
        raise FiveShotMaterialError("pinch bindings span multiple raw archives")
    required = (
        "event_offsets",
        "flat_t_rel_ms",
        "flat_frame_index",
        "flat_pointer_id",
        "flat_action_code",
        "flat_x",
        "flat_y",
    )
    with np.load(next(iter(sources)), allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise FiveShotMaterialError(
                f"pinch archive lacks pointer fields {sorted(missing)}"
            )
        arrays = {key: np.asarray(archive[key]) for key in required}
    offsets = np.asarray(arrays["event_offsets"], dtype=np.int64)
    geometry: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        index = int(row["raw_trajectory_event_index"])
        left, right = int(offsets[index]), int(offsets[index + 1])
        try:
            live = extract_live_two_pointer_endpoints(
                t_ms=arrays["flat_t_rel_ms"][left:right],
                frame_index=arrays["flat_frame_index"][left:right],
                pointer_id=arrays["flat_pointer_id"][left:right],
                android_action=arrays["flat_action_code"][left:right],
                x_px=arrays["flat_x"][left:right],
                y_px=arrays["flat_y"][left:right],
            )
        except PinchEndpointControlError as error:
            raise FiveShotMaterialError(
                f"pinch binding {row['source_cluster_id']} has no live "
                f"two-pointer geometry: {error}"
            ) from error
        axis = np.asarray(live.start_points_px[1], dtype=np.float64) - np.asarray(
            live.start_points_px[0], dtype=np.float64
        )
        geometry[str(row["user_id"])][str(row["source_cluster_id"])] = {
            "pinch_scale_direction": live.scale_direction,
            "pinch_start_span_px": float(live.start_span_px),
            "pinch_end_span_px": float(live.end_span_px),
            "pinch_start_axis_rad": float(np.arctan2(axis[1], axis[0])),
            "pinch_start_points_px": [list(value) for value in live.start_points_px],
            "pinch_end_points_px": [list(value) for value in live.end_points_px],
            "pinch_start_center_px": list(live.start_center_px),
            "pinch_end_center_px": list(live.end_center_px),
        }
    return dict(geometry)


def _select_pinch_span_ladder(
    candidates: list[int],
    *,
    clusters: np.ndarray,
    event_ids: np.ndarray,
    pinch_geometry: Mapping[str, Mapping[str, Any]],
    group: str,
    seed: int,
) -> list[int]:
    """Pick five pinches spanning this user's own range of pinch sizes."""

    labelled: list[tuple[float, bytes, int, str]] = []
    for index in candidates:
        entry = pinch_geometry.get(str(clusters[index]))
        if entry is None:
            raise FiveShotMaterialError(
                f"{group}/pinch candidate has no live two-pointer geometry"
            )
        labelled.append(
            (
                max(
                    float(entry["pinch_start_span_px"]),
                    float(entry["pinch_end_span_px"]),
                ),
                _selection_rank(str(event_ids[index]), seed=seed),
                int(index),
                str(entry["pinch_scale_direction"]),
            )
        )
    labelled.sort(key=lambda item: (item[0], item[1]))
    if len(labelled) < SHOTS_PER_GROUP:
        raise FiveShotMaterialError(f"{group}/pinch has fewer than five candidates")

    last = len(labelled) - 1
    positions: list[int] = []
    for quantile in PINCH_SPAN_LADDER_QUANTILES:
        wanted = int(round(float(quantile) * last))
        offset = 0
        while True:
            for position in (wanted + offset, wanted - offset):
                if 0 <= position <= last and position not in positions:
                    positions.append(position)
                    break
            else:
                offset += 1
                continue
            break
    selected = [labelled[position] for position in sorted(positions)]

    for wanted_direction in ("out", "in"):
        present = Counter(entry[3] for entry in selected)
        if present[wanted_direction]:
            continue
        pool = [
            entry
            for entry in labelled
            if entry[3] == wanted_direction and entry not in selected
        ]
        if not pool:
            raise FiveShotMaterialError(
                f"{group}/pinch has no {wanted_direction!r} candidate at all"
            )
        # Give up one copy of whichever direction is over-represented, and take
        # the replacement whose span sits closest to it.
        victim_direction = max(present, key=lambda key: (present[key], key))
        victims = [entry for entry in selected if entry[3] == victim_direction]
        replacement, victim = min(
            (
                (candidate, victim)
                for victim in victims
                for candidate in pool
            ),
            key=lambda pair: (
                abs(pair[0][0] - pair[1][0]),
                pair[0][1],
                pair[1][1],
            ),
        )
        selected[selected.index(victim)] = replacement
        selected.sort(key=lambda item: (item[0], item[1]))

    return [entry[2] for entry in selected]


def _load_aligned_touch(
    *, binding_cache: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Keep only touch bindings whose physical event was uniquely recovered."""

    if not binding_cache.is_file():
        raise FiveShotMaterialError("genuine binding cache is missing")
    aligned: dict[str, dict[str, Any]] = {}
    states: dict[str, Counter] = {}
    for row in _jsonl(binding_cache):
        action = str(row.get("action", ""))
        if action == "keystroke" or action not in ACTIONS:
            continue
        state = str(row.get("physical_match", "unknown"))
        states.setdefault(action, Counter())[state] += 1
        if state != "unique":
            continue
        aligned[str(row["source_cluster_id"])] = {
            "user_id": str(row["user_id"]),
            "orientation_id": int(row["orientation_id"]),
            "activity_id": None,
            "binding_resolution": (
                "exact_user_session_activity_orientation_start_end_v1"
            ),
            "binding_was_remapped": bool(int(row.get("index_shift", 0)) != 0),
            "raw_trajectory_event_index": int(row["raw_trajectory_event_index"]),
            "ordinal_raw_trajectory_event_index": int(
                row["ordinal_raw_trajectory_event_index"]
            ),
            "raw_trajectory_source": str(row["raw_trajectory_source"]),
            "raw_trajectory_source_sha256": str(
                row["raw_trajectory_source_sha256"]
            ),
            "native_start_sample": int(row["start_sample"]),
            "native_end_sample": int(row["end_sample"]),
            "raw_duration_ms": float(row["duration_ms"]),
        }
    if not aligned:
        raise FiveShotMaterialError("binding cache has no aligned touch rows")
    audit = {
        "binding_cache": str(binding_cache),
        "binding_cache_sha256": sha256_file(binding_cache),
        "physical_identity_fields": [
            "activity_id",
            "orientation_id",
            "label_start_ms",
        ],
        "cross_user_assertion": "user_id and session_id re-checked per match",
        "excluded_states_are_not_deleted_from_cache": True,
        "by_action": {
            action: {key: int(value) for key, value in sorted(counter.items())}
            for action, counter in sorted(states.items())
        },
        "aligned_touch_bindings": len(aligned),
    }
    return aligned, audit


def _shard_rows(
    payload: tuple[
        str,
        dict[str, Any],
        int,
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    (
        split,
        shard_row,
        seed,
        aligned_keystrokes,
        aligned_touch,
        pinch_geometry,
    ) = payload
    source = InputShard(
        split=split,
        user_id=str(shard_row["user_id"]),
        path=Path(shard_row["source"]).resolve(),
        sha256=str(shard_row["source_sha256"]),
        manifest_row=dict(shard_row),
    )
    shard = _load_shard(source, signals=True)
    actions = np.asarray(shard.arrays["action"]).astype(str)
    labels = np.asarray(shard.arrays["label"], dtype=np.int64)
    event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
    clusters = np.asarray(shard.arrays["source_cluster_id"]).astype(str)
    sessions = np.asarray(shard.arrays["session_id"]).astype(str)
    sample_idx = np.asarray(shard.arrays["sample_idx"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    audit: dict[str, int] = {}
    native_cache: dict[Path, dict[str, np.ndarray]] = {}
    for action in ACTIONS:
        candidates = np.flatnonzero((actions == action) & (labels == 0))
        if action == "keystroke":
            audit["keystroke_candidates_before_alignment"] = int(len(candidates))
            candidates = np.asarray(
                [
                    int(index)
                    for index in candidates
                    if str(clusters[int(index)]) in aligned_keystrokes
                ],
                dtype=np.int64,
            )
            audit["keystroke_candidates_after_alignment"] = int(len(candidates))
        else:
            audit[f"{action}_candidates_before_alignment"] = int(len(candidates))
            candidates = np.asarray(
                [
                    int(index)
                    for index in candidates
                    if str(clusters[int(index)]) in aligned_touch
                ],
                dtype=np.int64,
            )
            audit[f"{action}_candidates_after_alignment"] = int(len(candidates))
        # A few genuine carriers report an all-zero elapsed channel, so their
        # duration is only recoverable from a raw binding fallback.  Material
        usable: list[int] = []
        for index in candidates.tolist():
            _, candidate_trajectory = shard.event_signal(int(index))
            try:
                _detector_window_duration_ms(candidate_trajectory)
            except ReplayDatasetBuildError:
                continue
            usable.append(int(index))
        audit[f"{action}_candidates_with_usable_timeline"] = int(len(usable))
        candidates = np.asarray(usable, dtype=np.int64)
        if len(candidates) < SHOTS_PER_GROUP:
            raise FiveShotMaterialError(
                f"{split}/{source.user_id}/{action} has only {len(candidates)} "
                f"genuine events; five-shot material needs {SHOTS_PER_GROUP}"
            )
        ordered = sorted(
            candidates.tolist(),
            key=lambda index: _selection_rank(str(event_ids[index]), seed=seed),
        )
        if action == "pinch":
            ordered = _select_pinch_span_ladder(
                candidates.tolist(),
                clusters=clusters,
                event_ids=event_ids,
                pinch_geometry=pinch_geometry,
                group=f"{split}/{source.user_id}",
                seed=seed,
            )
        for ordinal, index in enumerate(ordered[:SHOTS_PER_GROUP]):
            shard_imu, trajectory = shard.event_signal(int(index))
            cluster_id = str(clusters[index])
            row: dict[str, Any] = {
                "schema_version": SCHEMA,
                "split": split,
                "user_id": source.user_id,
                "action": action,
                "shot_ordinal": ordinal,
                "event_id": str(event_ids[index]),
                "source_cluster_id": cluster_id,
                "session_id": str(sessions[index]),
                "sample_idx": int(sample_idx[index]),
                "shard_index": int(index),
                "shard_source": str(source.path),
                "shard_source_sha256": source.sha256,
                "samples": int(len(shard_imu)),
                "duration_ms": float(_detector_window_duration_ms(trajectory)),
                "imu_sha256": sha256_array(shard_imu),
                "trajectory_sha256": sha256_array(trajectory),
                "imu_source_kind": "detector_shard_event",
            }
            if action == "pinch":
                # The builder derives its similarity transform from these, so
                # it never has to reopen the raw archive to place a pinch.
                row.update(pinch_geometry[cluster_id])
            if action != "keystroke":
                touch = dict(aligned_touch[cluster_id])
                touch.pop("activity_id", None)
                touch.pop("user_id", None)
                row.update(touch)
            if action == "keystroke":
                metadata = aligned_keystrokes[cluster_id]
                native_imu, native_t_ns = _native_keystroke_signal(
                    metadata, native_cache
                )
                detector_duration_ms = float(
                    _detector_window_duration_ms(trajectory)
                )
                native_duration_ms = float(
                    (int(native_t_ns[-1]) - int(native_t_ns[0])) / 1_000_000.0
                )
                if not np.isclose(
                    detector_duration_ms,
                    native_duration_ms,
                    rtol=0.0,
                    atol=ALIGNMENT_TOLERANCE_MS,
                ):
                    raise FiveShotMaterialError(
                        f"{cluster_id}: shard/native elapsed time differs"
                    )
                row.update(metadata)
                row.update(
                    {
                        "samples": int(len(native_imu)),
                        "duration_ms": native_duration_ms,
                        "imu_sha256": sha256_array(native_imu),
                        "imu_timestamp_ns_sha256": sha256_array(native_t_ns),
                        "imu_start_timestamp_ns": int(native_t_ns[0]),
                        "imu_end_timestamp_ns": int(native_t_ns[-1]),
                        "imu_source_kind": "native_continuous_100hz_slice",
                        "shard_samples": int(len(shard_imu)),
                        "shard_duration_ms": detector_duration_ms,
                        "shard_imu_sha256": sha256_array(shard_imu),
                        "shard_full_span_resampled": bool(
                            len(shard_imu) != len(native_imu)
                        ),
                    }
                )
            rows.append(row)
    return rows, audit


def _canonical_selection_digest(rows: list[dict[str, Any]]) -> str:
    payload = [
        [
            row["split"],
            row["user_id"],
            row["action"],
            row["shot_ordinal"],
            row["event_id"],
            row["imu_sha256"],
            row["trajectory_sha256"],
        ]
        for row in rows
    ]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--genuine-binding-cache", type=Path, default=DEFAULT_BINDINGS
    )
    parser.add_argument(
        "--native-genuine-manifest", type=Path, default=DEFAULT_NATIVE_MANIFEST
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FiveShotMaterialError(f"output already exists: {output}")
    shards, input_release = load_input_dataset(args.input_manifest)
    aligned_keystrokes, alignment_audit = _load_aligned_keystrokes(
        binding_cache=args.genuine_binding_cache.resolve(),
        native_manifest=args.native_genuine_manifest.resolve(),
    )
    aligned_touch, touch_alignment_audit = _load_aligned_touch(
        binding_cache=args.genuine_binding_cache.resolve()
    )
    touch_by_user: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cluster_id, metadata in aligned_touch.items():
        touch_by_user[str(metadata["user_id"])][cluster_id] = metadata
    pinch_geometry = _pinch_geometry_by_cluster(
        binding_cache=args.genuine_binding_cache.resolve()
    )
    aligned_by_user: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cluster_id, metadata in aligned_keystrokes.items():
        # The source-cluster namespace is native and therefore unique.  The
        # user is recovered from its physical binding, not from the colliding
        # raw ordinal event ID.
        aligned_by_user[str(metadata["user_id"])][cluster_id] = metadata

    jobs = [
        (
            split,
            shard_row,
            args.seed,
            aligned_by_user.get(str(shard_row["user_id"]), {}),
            touch_by_user.get(str(shard_row["user_id"]), {}),
            pinch_geometry.get(str(shard_row["user_id"]), {}),
        )
        for split in SPLITS
        for shard_row in (source.manifest_row for source in shards[split])
    ]
    rows: list[dict[str, Any]] = []
    candidate_audits: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard_rows, shard_audit in pool.map(_shard_rows, jobs):
            rows.extend(shard_rows)
            candidate_audits.append(shard_audit)

    rows.sort(
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["user_id"],
            ACTIONS.index(row["action"]),
            row["shot_ordinal"],
        )
    )
    users = {split: sorted({r["user_id"] for r in rows if r["split"] == split}) for split in SPLITS}
    expected = sum(len(users[split]) for split in SPLITS) * len(ACTIONS) * SHOTS_PER_GROUP
    if len(rows) != expected:
        raise FiveShotMaterialError(
            f"selected {len(rows)} material events, expected {expected}"
        )
    identities = {row["event_id"] for row in rows}
    if len(identities) != len(rows):
        raise FiveShotMaterialError("one event was selected twice")

    output.mkdir(parents=True)
    manifest_path = output / "material_manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )

    per_group_samples = {}
    for split in SPLITS:
        for action in ACTIONS:
            picked = [
                r for r in rows if r["split"] == split and r["action"] == action
            ]
            per_group_samples[f"{split}/{action}"] = {
                "users": len(users[split]),
                "events": len(picked),
                "median_samples": float(np.median([r["samples"] for r in picked])),
                "median_duration_ms": float(
                    np.median([r["duration_ms"] for r in picked])
                ),
            }

    before_counts = [
        int(row["keystroke_candidates_before_alignment"])
        for row in candidate_audits
    ]
    after_counts = [
        int(row["keystroke_candidates_after_alignment"])
        for row in candidate_audits
    ]
    if not after_counts or min(after_counts) < SHOTS_PER_GROUP:
        raise FiveShotMaterialError(
            "aligned keystroke candidates cannot provide five shots per user"
        )
    alignment_audit["detector_candidate_events_before_alignment"] = int(
        sum(before_counts)
    )
    alignment_audit["detector_candidate_events_after_alignment"] = int(
        sum(after_counts)
    )
    alignment_audit["minimum_aligned_candidates_per_user"] = int(
        min(after_counts)
    )
    alignment_audit["median_aligned_candidates_per_user"] = float(
        np.median(after_counts)
    )

    release = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "selection_rule": (
            "sha256(domain|seed|event_id) ascending, first five eligible genuine "
            "events per split/user/action; keystroke eligibility is resolved by "
            "physical interval plus the correct 24px gate; independent of any "
            "detector score"
        ),
        "selection_domain": SELECTION_DOMAIN,
        "shots_per_group": SHOTS_PER_GROUP,
        "seed": int(args.seed),
        "input_manifest": str(Path(args.input_manifest).resolve()),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "input_release_schema": input_release.get("schema_version"),
        "genuine_binding_cache": str(args.genuine_binding_cache.resolve()),
        "genuine_binding_cache_sha256": sha256_file(args.genuine_binding_cache),
        "native_genuine_manifest": str(args.native_genuine_manifest.resolve()),
        "native_genuine_manifest_sha256": sha256_file(
            args.native_genuine_manifest
        ),
        "users_by_split": users,
        "actions": list(ACTIONS),
        "material_events": len(rows),
        "material_manifest": str(manifest_path),
        "material_manifest_sha256": sha256_file(manifest_path),
        "canonical_selection_sha256": _canonical_selection_digest(rows),
        "keystroke_alignment_audit": alignment_audit,
        "touch_alignment_audit": touch_alignment_audit,
        "per_group": per_group_samples,
        "reuse_note": (
            "each user needs 200 fake events per action, so every frozen shot is "
            "reused 40 times"
        ),
    }
    release_path = output / "release.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in release.items() if k != "per_group"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except FiveShotMaterialError as exc:
        print(f"FIVE-SHOT MATERIAL ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
