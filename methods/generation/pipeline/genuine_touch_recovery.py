from __future__ import annotations

"""Recover event-level genuine touch from the accepted raw HMOG corpus.

The former direct100k adapter cropped the native continuous ``TouchEvent``
stream.  That loses most soft-keyboard contacts even though the frozen quality
ledger already binds each accepted event to a complete raw trajectory (with
the OneFinger fallback).  This module resolves the original event identity and
loads that bound trajectory for the common Android observation operator.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .android_touch_observation import TouchObservation, observe_android_rows


PERIOD_NS = 10_000_000
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


class GenuineTouchRecoveryError(ValueError):
    pass


@dataclass(frozen=True)
class GenuineTouchBinding:
    source_cluster_id: str
    user_id: str
    split: str
    session_id: str
    action: str
    source_event_id: str
    source_event_ordinal: int
    orientation_id: int
    start_sample: int
    end_sample: int
    raw_trajectory_source: Path
    raw_trajectory_source_sha256: str
    raw_trajectory_event_index: int
    raw_event_sha256: str
    # The pre-correction ordinal index, kept so provenance can show the shift.
    ordinal_raw_trajectory_event_index: int = -1
    # Archive id of the event the corrected index names.  Empty when the cache
    # predates the physical rebinding, in which case consumers fall back to the
    # ledger's own claim.
    raw_trajectory_event_id: str = ""
    # False when the raw archive has no event with this physical identity, in
    # which case raw_trajectory_event_index is still the unreliable ordinal.
    physically_resolved: bool = True

    @property
    def duration_ms(self) -> float:
        return float((self.end_sample - self.start_sample - 1) * 10.0)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def native_event_slot(event: Mapping[str, Any], samples: int) -> tuple[int, int]:
    try:
        start = int(math.ceil(int(event["relative_start_ns"]) / PERIOD_NS))
        end = int(math.ceil(int(event["relative_end_ns"]) / PERIOD_NS))
    except (KeyError, TypeError, ValueError) as exc:
        raise GenuineTouchRecoveryError("invalid native event interval") from exc
    start = max(0, min(start, int(samples)))
    end = max(0, min(end, int(samples)))
    if end <= start:
        raise GenuineTouchRecoveryError("native event interval is empty")
    return start, end


def native_source_cluster_id(
    *,
    user_id: str,
    session_id: str,
    source_event_id: str,
    source_event_ordinal: int,
    start_sample: int,
    end_sample: int,
) -> str:
    value = {
        "user_id": str(user_id),
        "session_id": str(session_id),
        "source_event_id": str(source_event_id),
        "ordinal": int(source_event_ordinal),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
    }
    return f"hmog-genuine-event-{canonical_json_sha256(value)[:24]}"


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GenuineTouchRecoveryError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            yield value


def load_genuine_touch_bindings(
    *,
    native_genuine_manifest: str | Path,
    dispatch_quality_ledger: str | Path,
) -> dict[str, GenuineTouchBinding]:
    """Load every quality-included event with its exact raw touch binding."""

    manifest = Path(native_genuine_manifest).resolve()
    ledger = Path(dispatch_quality_ledger).resolve()
    if not manifest.is_file() or not ledger.is_file():
        raise GenuineTouchRecoveryError("native manifest or quality ledger is missing")
    quality: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _jsonl(ledger):
        identity = (
            str(row.get("user_id", "")),
            str(row.get("session_id", "")),
            str(row.get("source_action_label", "")),
            str(row.get("source_event_id", "")),
        )
        if identity in quality:
            raise GenuineTouchRecoveryError("duplicate quality-ledger identity")
        quality[identity] = row

    bindings: dict[str, GenuineTouchBinding] = {}
    audited_identities: set[tuple[str, str, str, str]] = set()
    archive_identity_cache: dict[Path, dict[str, np.ndarray]] = {}
    for session in _jsonl(manifest):
        user_id = str(session.get("user_id", ""))
        split = str(session.get("split", ""))
        session_id = str(session.get("session_id", ""))
        audit_path = Path(str(session.get("native_session_audit", ""))).resolve()
        if split not in {"train", "development", "test"} or not audit_path.is_file():
            raise GenuineTouchRecoveryError("native session manifest binding is invalid")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        samples = int(audit.get("samples", -1))
        events = audit.get("event_plan", {}).get("events")
        if samples < 2 or not isinstance(events, list):
            raise GenuineTouchRecoveryError(f"{audit_path}: invalid event plan")
        for ordinal, event in enumerate(events):
            action = str(event.get("action", ""))
            source_event_id = str(event.get("event_id", ""))
            if action not in ACTIONS or not source_event_id:
                raise GenuineTouchRecoveryError(f"{audit_path}: invalid native event")
            start, end = native_event_slot(event, samples)
            cluster_id = native_source_cluster_id(
                user_id=user_id,
                session_id=session_id,
                source_event_id=source_event_id,
                source_event_ordinal=ordinal,
                start_sample=start,
                end_sample=end,
            )
            identity = (user_id, session_id, action, source_event_id)
            row = quality.get(identity)
            if row is None:
                raise GenuineTouchRecoveryError(
                    f"{cluster_id}: missing quality-ledger row"
                )
            audited_identities.add(identity)
            if str(row.get("decision", "")) != "include":
                continue
            raw_source = Path(str(row.get("raw_trajectory_source", ""))).resolve()
            try:
                raw_index = int(row.get("raw_trajectory_event_index", -1))
                orientation = int(row.get("orientation_id", -99))
            except (TypeError, ValueError) as exc:
                raise GenuineTouchRecoveryError(
                    f"{cluster_id}: invalid raw trajectory binding"
                ) from exc
            if (
                row.get("eligible") is not True
                or row.get("raw_trajectory_extraction_status") != "accepted"
                or not raw_source.is_file()
                or raw_index < 0
                or orientation != int(event.get("orientation_id", -98))
                or int(row.get("source_event_ordinal", -1)) != ordinal
                or len(str(row.get("raw_trajectory_source_sha256", ""))) != 64
                or len(str(row.get("raw_event_sha256", ""))) != 64
            ):
                raise GenuineTouchRecoveryError(
                    f"{cluster_id}: incomplete accepted raw trajectory"
                )
            # Re-point the ordinal index at the physically matching archive row.
            # An event the raw pipeline filtered out has no match at all; its
            # ordinal index still names some other event, so the binding is
            # marked unresolved instead of being silently trusted.
            physical_index = _physical_archive_index(
                archive_identity_cache,
                source=raw_source,
                activity_id=int(event["activity_id"]),
                orientation_id=orientation,
                label_start_ms=int(event["start_timestamp_ns"]) // 1_000_000,
            )
            if physical_index is not None:
                identity_values = archive_identity_cache[raw_source]
                if int(
                    identity_values["user_id"][physical_index]
                ) != _numeric_hmog_id(user_id, prefix="hmog_u") or int(
                    identity_values["session_id"][physical_index]
                ) != _numeric_hmog_id(session_id, prefix="_s", tail=True):
                    raise GenuineTouchRecoveryError(
                        f"{cluster_id}: physical match crossed user/session"
                    )
            binding = GenuineTouchBinding(
                source_cluster_id=cluster_id,
                user_id=user_id,
                split=split,
                session_id=session_id,
                action=action,
                source_event_id=source_event_id,
                source_event_ordinal=ordinal,
                orientation_id=orientation,
                start_sample=start,
                end_sample=end,
                raw_trajectory_source=raw_source,
                raw_trajectory_source_sha256=str(
                    row["raw_trajectory_source_sha256"]
                ),
                raw_trajectory_event_index=(
                    raw_index if physical_index is None else physical_index
                ),
                raw_event_sha256=str(row["raw_event_sha256"]),
                ordinal_raw_trajectory_event_index=raw_index,
                # The ledger's `source_event_id` names the event the ORDINAL
                # index pointed at, which for 35% of bindings is a different,
                # also existing archive row.  Carry the corrected event's own
                # archive id so consumers check the pointer that was fixed.
                raw_trajectory_event_id=(
                    ""
                    if physical_index is None
                    else str(
                        np.asarray(
                            archive_identity_cache[raw_source]["event_id"][
                                physical_index
                            ]
                        ).item()
                    )
                ),
                physically_resolved=physical_index is not None,
            )
            if cluster_id in bindings:
                raise GenuineTouchRecoveryError("duplicate genuine source cluster")
            bindings[cluster_id] = binding
    if audited_identities != set(quality):
        raise GenuineTouchRecoveryError(
            "native manifest and quality-ledger coverage differ"
        )
    return bindings


PHYSICAL_ALIGNMENT_TOLERANCE_MS = 20.0


def _numeric_hmog_id(value: str, *, prefix: str, tail: bool = False) -> int:
    """Parse the numeric part of an HMOG user or session identifier."""

    text = str(value)
    if tail:
        if prefix not in text:
            raise GenuineTouchRecoveryError(f"invalid HMOG session ID: {text}")
        return int(text.rsplit(prefix, 1)[1])
    if not text.startswith(prefix):
        raise GenuineTouchRecoveryError(f"invalid HMOG user ID: {text}")
    return int(text.removeprefix(prefix))


def _physical_archive_index(
    archive_cache: dict[Path, dict[str, np.ndarray]],
    *,
    source: Path,
    activity_id: int,
    orientation_id: int,
    label_start_ms: int,
) -> int | None:
    """Return the archive row whose physical identity matches, or None."""

    if source not in archive_cache:
        with np.load(source, allow_pickle=False) as archive:
            missing = {
                "activity_id",
                "orientation_id",
                "label_start_ms",
                "user_id",
                "session_id",
                "event_id",
            } - set(archive.files)
            if missing:
                raise GenuineTouchRecoveryError(
                    f"{source}: archive lacks physical identity fields {sorted(missing)}"
                )
            archive_cache[source] = {
                "activity_id": np.asarray(archive["activity_id"]),
                "orientation_id": np.asarray(archive["orientation_id"]),
                "label_start_ms": np.asarray(
                    archive["label_start_ms"], dtype=np.float64
                ),
                "user_id": np.asarray(archive["user_id"]),
                "session_id": np.asarray(archive["session_id"]),
                "event_id": np.asarray(archive["event_id"]),
            }
    values = archive_cache[source]
    candidates = np.flatnonzero(
        (values["activity_id"] == int(activity_id))
        & (values["orientation_id"] == int(orientation_id))
        & (
            np.abs(values["label_start_ms"] - float(label_start_ms))
            <= PHYSICAL_ALIGNMENT_TOLERANCE_MS
        )
    )
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def load_raw_trajectory_archive(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).resolve()
    required = {
        "event_id",
        "action_name",
        "orientation_id",
        "touch_duration_ms",
        "event_offsets",
        "flat_t_rel_ms",
        "flat_frame_index",
        "flat_pointer_id",
        "flat_action_code",
        "flat_x",
        "flat_y",
        "flat_pressure",
        "flat_valid_mask",
        "flat_key_index",
    }
    with np.load(source, allow_pickle=False) as archive:
        if required - set(archive.files):
            raise GenuineTouchRecoveryError(f"{source}: incomplete trajectory archive")
        optional = {"flat_keycode"}.intersection(archive.files)
        return {
            name: np.asarray(archive[name])
            for name in required.union(optional)
        }


def observe_genuine_binding(
    binding: GenuineTouchBinding,
    *,
    target_samples: int,
    target_duration_ms: float | None = None,
    archive: Mapping[str, np.ndarray] | None = None,
) -> TouchObservation:
    values = (
        load_raw_trajectory_archive(binding.raw_trajectory_source)
        if archive is None
        else archive
    )
    index = binding.raw_trajectory_event_index
    event_ids = np.asarray(values["event_id"])
    offsets = np.asarray(values["event_offsets"], dtype=np.int64)
    if not 0 <= index < len(event_ids) or len(offsets) != len(event_ids) + 1:
        raise GenuineTouchRecoveryError("raw trajectory event index is invalid")
    observed_event_id = str(np.asarray(event_ids[index]).item())
    expected_event_id = (
        binding.raw_trajectory_event_id or binding.source_event_id
    )
    if observed_event_id != expected_event_id:
        raise GenuineTouchRecoveryError("raw trajectory event identity mismatch")
    action_name = str(np.asarray(values["action_name"]).item())
    if action_name != binding.action:
        raise GenuineTouchRecoveryError("raw trajectory action mismatch")
    if int(np.asarray(values["orientation_id"])[index]) != binding.orientation_id:
        raise GenuineTouchRecoveryError("raw trajectory orientation mismatch")
    left, right = (int(value) for value in offsets[index : index + 2])
    valid = np.asarray(values["flat_valid_mask"][left:right], dtype=np.bool_)
    positions = np.arange(left, right, dtype=np.int64)[valid]
    if len(positions) < 2:
        raise GenuineTouchRecoveryError("raw trajectory has fewer than two valid rows")
    source_duration_ms = float(np.asarray(values["touch_duration_ms"])[index])
    output_duration_ms = (
        binding.duration_ms
        if target_duration_ms is None
        else float(target_duration_ms)
    )
    return observe_android_rows(
        action=binding.action,
        target_samples=target_samples,
        target_duration_ms=output_duration_ms,
        orientation_id=binding.orientation_id,
        t_ms=values["flat_t_rel_ms"][positions],
        x_px=values["flat_x"][positions],
        y_px=values["flat_y"][positions],
        pressure=values["flat_pressure"][positions],
        pointer_id=values["flat_pointer_id"][positions],
        android_action=values["flat_action_code"][positions],
        frame_index=values["flat_frame_index"][positions],
        key_index=values["flat_key_index"][positions],
        source_duration_ms=source_duration_ms,
    )


__all__ = [
    "GenuineTouchBinding",
    "GenuineTouchRecoveryError",
    "canonical_json_sha256",
    "load_genuine_touch_bindings",
    "load_raw_trajectory_archive",
    "native_event_slot",
    "native_source_cluster_id",
    "observe_genuine_binding",
]
