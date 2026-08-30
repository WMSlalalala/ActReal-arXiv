from __future__ import annotations

"""Read-only pre-detector audit for conditional-touch smoke releases.

The audit intentionally separates the physical six-axis array (``imu_flat``)
from detector modalities which also contain touch coordinates.  Distribution
summaries in this module are descriptive only; detector non-regression gates
remain the authority for accepting a generator.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .conditional_touch_request_plan import (
    ConditionalTouchRequestPlan,
    ConditionalTouchRequestPlanError,
)


TOUCH_ACTIONS = ("tap", "scroll", "swipe")
SPLITS = ("train", "development", "test")
SHARD_SCHEMA = "joint_event_pad_ragged_shard_v1"
MANIFEST_SCHEMA = "joint_event_pad_manifest_v2"
RELEASE_SCHEMA = "hmog_direct100k_detector_dataset_v1"
SCREEN_DIMENSIONS = {
    0: (1080.0, 1920.0),
    1: (1920.0, 1080.0),
    2: (1080.0, 1920.0),
    3: (1920.0, 1080.0),
    -1: (1080.0, 1920.0),
}
PIXEL_ZERO_ATOL = 1.0e-4
HALF_PIXEL_ATOL = 2.5e-4
REPORT_SCHEMA = "hmog_conditional_touch_smoke_precheck_v1"


class ConditionalTouchSmokeAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeEvent:
    event_id: str
    split: str
    user_id: str
    action: str
    label: int
    imu: np.ndarray
    trajectory: np.ndarray

    @property
    def binding(self) -> tuple[str, str, str, int]:
        return self.split, self.user_id, self.action, self.label


@dataclass(frozen=True)
class LoadedSmoke:
    manifest: Path
    manifest_sha256: str
    events: Mapping[str, SmokeEvent]
    split_counts: Mapping[str, int]
    action_label_counts: Mapping[str, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConditionalTouchSmokeAuditError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConditionalTouchSmokeAuditError(f"JSON root is not an object: {path}")
    return value


def _resolve_shard(manifest: Path, declared: object) -> Path:
    source = Path(str(declared)).expanduser()
    candidates = [source]
    if not source.is_absolute():
        candidates.insert(0, manifest.parent / source)
    candidates.append(manifest.parent / "shards" / source.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise ConditionalTouchSmokeAuditError(
        f"manifest shard is missing: {declared!r} ({manifest})"
    )


def _scalar_text(archive: Mapping[str, np.ndarray], name: str) -> str:
    try:
        return str(np.asarray(archive[name]).item())
    except (KeyError, ValueError) as exc:
        raise ConditionalTouchSmokeAuditError(
            f"shard scalar {name!r} is missing or malformed"
        ) from exc


def load_smoke_manifest(
    manifest_path: str | Path,
    *,
    release_path: str | Path | None = None,
) -> LoadedSmoke:
    """Load and validate the canonical ragged smoke shard contract."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ConditionalTouchSmokeAuditError(f"manifest is missing: {manifest}")
    manifest_digest = sha256_file(manifest)
    if release_path is not None:
        release = _read_json(Path(release_path).expanduser().resolve())
        if release.get("schema_version") != RELEASE_SCHEMA:
            raise ConditionalTouchSmokeAuditError("release schema changed")
        if release.get("event_manifest_sha256") != manifest_digest:
            raise ConditionalTouchSmokeAuditError(
                "release does not bind the supplied manifest bytes"
            )

    rows: list[dict[str, Any]] = []
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ConditionalTouchSmokeAuditError(
                        f"blank manifest row: {manifest}:{line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ConditionalTouchSmokeAuditError(
                        f"manifest row is not an object: {manifest}:{line_number}"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConditionalTouchSmokeAuditError(
            f"cannot read manifest: {manifest}"
        ) from exc

    seen_splits: set[str] = set()
    seen_shards: set[Path] = set()
    events: dict[str, SmokeEvent] = {}
    split_counts = {split: 0 for split in SPLITS}
    action_label_counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", ""))
        if (
            row.get("schema_version") != MANIFEST_SCHEMA
            or split not in SPLITS
            or split in seen_splits
            or not isinstance(row.get("shards"), list)
        ):
            raise ConditionalTouchSmokeAuditError(
                f"invalid or duplicate manifest split row: {split!r}"
            )
        seen_splits.add(split)
        manifest_users = {str(value) for value in row.get("user_ids", [])}
        row_event_count = 0
        for shard_row in row["shards"]:
            if not isinstance(shard_row, dict):
                raise ConditionalTouchSmokeAuditError("manifest shard row is malformed")
            shard = _resolve_shard(manifest, shard_row.get("source", ""))
            if shard in seen_shards:
                raise ConditionalTouchSmokeAuditError(f"duplicate shard: {shard}")
            seen_shards.add(shard)
            declared_digest = str(shard_row.get("source_sha256", ""))
            if not declared_digest or sha256_file(shard) != declared_digest:
                raise ConditionalTouchSmokeAuditError(
                    f"manifest shard SHA-256 mismatch: {shard}"
                )
            required = {
                "schema_version",
                "coordinate_schema",
                "time_schema",
                "split",
                "imu_flat",
                "trajectory_flat",
                "offsets",
                "label",
                "user_id",
                "event_id",
                "action",
            }
            with np.load(shard, allow_pickle=False) as archive:
                missing = required - set(archive.files)
                if missing:
                    raise ConditionalTouchSmokeAuditError(
                        f"shard arrays are missing in {shard}: {sorted(missing)}"
                    )
                if (
                    _scalar_text(archive, "schema_version") != SHARD_SCHEMA
                    or _scalar_text(archive, "coordinate_schema")
                    != "screen_relative_xy_v1"
                    or _scalar_text(archive, "time_schema")
                    != "elapsed_seconds_since_event_start_v1"
                    or _scalar_text(archive, "split") != split
                ):
                    raise ConditionalTouchSmokeAuditError(
                        f"shard contract changed: {shard}"
                    )
                offsets = np.asarray(archive["offsets"], dtype=np.int64)
                imu_flat = np.asarray(archive["imu_flat"], dtype=np.float32)
                trajectory_flat = np.asarray(
                    archive["trajectory_flat"], dtype=np.float32
                )
                labels = np.asarray(archive["label"], dtype=np.int64)
                users = np.asarray(archive["user_id"]).astype(str)
                event_ids = np.asarray(archive["event_id"]).astype(str)
                actions = np.asarray(archive["action"]).astype(str)
                count = len(event_ids)
                if (
                    offsets.shape != (count + 1,)
                    or not count
                    or int(offsets[0]) != 0
                    or np.any(np.diff(offsets) < 2)
                    or int(offsets[-1]) != len(imu_flat)
                    or trajectory_flat.shape != (len(imu_flat), 9)
                    or imu_flat.shape != (len(imu_flat), 6)
                    or labels.shape != (count,)
                    or users.shape != (count,)
                    or actions.shape != (count,)
                    or not np.isfinite(imu_flat).all()
                    or not np.isfinite(trajectory_flat).all()
                ):
                    raise ConditionalTouchSmokeAuditError(
                        f"ragged signal contract changed: {shard}"
                    )
                declared_user = str(shard_row.get("user_id", ""))
                if (
                    not declared_user
                    or set(users.tolist()) != {declared_user}
                    or (manifest_users and declared_user not in manifest_users)
                ):
                    raise ConditionalTouchSmokeAuditError(
                        f"shard user binding changed: {shard}"
                    )
                for index, event_id in enumerate(event_ids.tolist()):
                    if not event_id or event_id in events:
                        raise ConditionalTouchSmokeAuditError(
                            f"duplicate or empty event_id: {event_id!r}"
                        )
                    label = int(labels[index])
                    action = str(actions[index])
                    if label not in (0, 1) or not action:
                        raise ConditionalTouchSmokeAuditError(
                            f"invalid event binding: {event_id}"
                        )
                    left, right = map(int, offsets[index : index + 2])
                    events[event_id] = SmokeEvent(
                        event_id=event_id,
                        split=split,
                        user_id=str(users[index]),
                        action=action,
                        label=label,
                        imu=imu_flat[left:right].copy(),
                        trajectory=trajectory_flat[left:right].copy(),
                    )
                    key = f"{split}/{action}/{label}"
                    action_label_counts[key] = action_label_counts.get(key, 0) + 1
                row_event_count += count
        declared_events = row.get("events")
        if declared_events is not None and int(declared_events) != row_event_count:
            raise ConditionalTouchSmokeAuditError(
                f"manifest split event count changed: {split}"
            )
        split_counts[split] = row_event_count
    if seen_splits != set(SPLITS):
        raise ConditionalTouchSmokeAuditError("manifest needs three fixed split rows")
    return LoadedSmoke(
        manifest=manifest,
        manifest_sha256=manifest_digest,
        events=events,
        split_counts=split_counts,
        action_label_counts=action_label_counts,
    )


def load_provenance(
    path: str | Path,
    *,
    release_path: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConditionalTouchSmokeAuditError(f"provenance is missing: {source}")
    digest = sha256_file(source)
    if release_path is not None:
        release = _read_json(Path(release_path).expanduser().resolve())
        declared = str(release.get("provenance_sha256", ""))
        if declared and declared != digest:
            raise ConditionalTouchSmokeAuditError(
                "release does not bind the supplied provenance bytes"
            )
    rows: dict[str, dict[str, Any]] = {}
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict) or not str(value.get("event_id", "")):
                    raise ConditionalTouchSmokeAuditError(
                        f"invalid provenance row: {source}:{line_number}"
                    )
                event_id = str(value["event_id"])
                if event_id in rows:
                    raise ConditionalTouchSmokeAuditError(
                        f"duplicate provenance event_id: {event_id}"
                    )
                rows[event_id] = value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConditionalTouchSmokeAuditError(
            f"cannot read provenance: {source}"
        ) from exc
    return rows, digest


def _known_orientation(row: Mapping[str, Any] | None) -> int | None:
    if not row:
        return None
    donor = row.get("donor")
    if not isinstance(donor, Mapping):
        return None
    candidates = (
        donor.get("conditioning_orientation_id"),
        donor.get("orientation_id"),
        (
            donor.get("target_binding", {}).get("orientation_id")
            if isinstance(donor.get("target_binding"), Mapping)
            else None
        ),
    )
    for value in candidates:
        if isinstance(value, (int, np.integer)) and int(value) in SCREEN_DIMENSIONS:
            return int(value)
    return None


def _lattice_fraction(xy_rel: np.ndarray, dimensions: tuple[float, float]) -> float:
    pixels = xy_rel * np.asarray(dimensions, dtype=np.float64)
    distance = np.abs(pixels * 2.0 - np.rint(pixels * 2.0)) / 2.0
    return float(np.mean(np.all(distance <= HALF_PIXEL_ATOL, axis=1)))


def _pixel_coordinates(
    trajectory: np.ndarray,
    provenance: Mapping[str, Any] | None,
) -> tuple[np.ndarray, str, int]:
    mask = (trajectory[:, 0] > 0.5) & (trajectory[:, 8] > 0.5)
    xy_rel = np.asarray(trajectory[mask, 1:3], dtype=np.float64)
    if not len(xy_rel):
        raise ConditionalTouchSmokeAuditError("touch event has no active trajectory row")
    known = _known_orientation(provenance)
    if known is not None:
        orientation = known
        source = "provenance"
    else:
        portrait = _lattice_fraction(xy_rel, SCREEN_DIMENSIONS[0])
        landscape = _lattice_fraction(xy_rel, SCREEN_DIMENSIONS[1])
        orientation = 1 if landscape > portrait else 0
        source = "half_pixel_lattice_inference"
    return (
        xy_rel * np.asarray(SCREEN_DIMENSIONS[orientation], dtype=np.float64),
        source,
        orientation,
    )


def event_shape_metrics(
    event: SmokeEvent,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], str]:
    """Compute detector-grid shape diagnostics in physical-pixel units."""

    xy, orientation_source, _ = _pixel_coordinates(event.trajectory, provenance)
    increments = np.diff(xy, axis=0)
    step = np.linalg.norm(increments, axis=1)
    zero_step_fraction = float(np.mean(step <= PIXEL_ZERO_ATOL))
    path_length = float(np.sum(step))
    chord_vector = xy[-1] - xy[0]
    chord = float(np.linalg.norm(chord_vector))
    if chord > PIXEL_ZERO_ATOL:
        axis = chord_vector / chord
        progress = increments @ axis
        nonzero = step > PIXEL_ZERO_ATOL
        reverse = (
            float(np.mean(progress[nonzero] < -PIXEL_ZERO_ATOL))
            if np.any(nonzero)
            else 0.0
        )
        lateral_steps = increments - np.outer(progress, axis)
        offsets = xy - xy[0]
        lateral_offsets = offsets - np.outer(offsets @ axis, axis)
        path_chord = path_length / chord
        lateral_step_rms = float(
            np.sqrt(np.mean(np.sum(lateral_steps * lateral_steps, axis=1)))
        )
        lateral_residual_rms = float(
            np.sqrt(np.mean(np.sum(lateral_offsets * lateral_offsets, axis=1)))
        )
    else:
        reverse = math.nan
        path_chord = math.nan
        lateral_step_rms = float(np.sqrt(np.mean(step * step)))
        lateral_residual_rms = float(
            np.sqrt(np.mean(np.sum((xy - xy[0]) ** 2, axis=1)))
        )
    half_pixel = np.abs(xy * 2.0 - np.rint(xy * 2.0)) / 2.0
    return {
        "path_length_px": path_length,
        "chord_px": chord,
        "path_chord_ratio": path_chord,
        "reverse_progress_fraction": reverse,
        "lateral_step_rms_px": lateral_step_rms,
        "lateral_residual_rms_px": lateral_residual_rms,
        "half_pixel_gridpoint_fraction": float(
            np.mean(np.all(half_pixel <= HALF_PIXEL_ATOL, axis=1))
        ),
        "zero_step_fraction": zero_step_fraction,
    }, orientation_source


def _quantiles(values: Iterable[float], *, total: int) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    finite = data[np.isfinite(data)]
    result: dict[str, Any] = {"events": int(total), "finite_events": int(len(finite))}
    names = (
        ("min", 0.0),
        ("q05", 0.05),
        ("q10", 0.10),
        ("q25", 0.25),
        ("q50", 0.50),
        ("q75", 0.75),
        ("q90", 0.90),
        ("q95", 0.95),
        ("max", 1.0),
    )
    if not len(finite):
        result.update({name: None for name, _ in names})
        result["mean"] = None
        return result
    result.update(
        {name: float(np.quantile(finite, quantile)) for name, quantile in names}
    )
    result["mean"] = float(np.mean(finite))
    return result


def _distribution_report(
    baseline: LoadedSmoke,
    candidate: LoadedSmoke,
    baseline_provenance: Mapping[str, Mapping[str, Any]],
    candidate_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    datasets = {
        "baseline_genuine": (baseline, 0, baseline_provenance),
        "baseline_fake": (baseline, 1, baseline_provenance),
        "candidate_genuine": (candidate, 0, candidate_provenance),
        "candidate_fake": (candidate, 1, candidate_provenance),
    }
    output: dict[str, Any] = {}
    for action in TOUCH_ACTIONS:
        action_groups: dict[str, Any] = {}
        raw_by_group: dict[str, dict[str, list[float]]] = {}
        for group, (dataset, label, provenance) in datasets.items():
            metric_rows: list[dict[str, float]] = []
            orientation_sources: dict[str, int] = {}
            for event in dataset.events.values():
                if event.action != action or event.label != label:
                    continue
                metrics, source = event_shape_metrics(
                    event, provenance.get(event.event_id)
                )
                metric_rows.append(metrics)
                orientation_sources[source] = orientation_sources.get(source, 0) + 1
            names = sorted({name for row in metric_rows for name in row})
            raw = {name: [row[name] for row in metric_rows] for name in names}
            raw_by_group[group] = raw
            action_groups[group] = {
                "events": len(metric_rows),
                "orientation_sources": orientation_sources,
                "metrics": {
                    name: _quantiles(values, total=len(metric_rows))
                    for name, values in raw.items()
                },
            }
        deltas: dict[str, Any] = {}
        for reference in ("candidate_genuine", "baseline_fake"):
            name = f"candidate_fake_minus_{reference}_q50"
            metric_deltas: dict[str, float | None] = {}
            for metric, candidate_values in raw_by_group["candidate_fake"].items():
                left = np.asarray(candidate_values, dtype=np.float64)
                right = np.asarray(
                    raw_by_group[reference].get(metric, []), dtype=np.float64
                )
                left = left[np.isfinite(left)]
                right = right[np.isfinite(right)]
                metric_deltas[metric] = (
                    float(np.median(left) - np.median(right))
                    if len(left) and len(right)
                    else None
                )
            deltas[name] = metric_deltas
        output[action] = {"groups": action_groups, "median_deltas": deltas}
    return output


def _point_pair(value: object, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConditionalTouchSmokeAuditError(f"invalid {name}") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ConditionalTouchSmokeAuditError(f"invalid {name}")
    return result


def audit_conditional_touch_smoke(
    *,
    baseline_manifest: str | Path,
    candidate_manifest: str | Path,
    expected_events: int = 1500,
    baseline_release: str | Path | None = None,
    candidate_release: str | Path | None = None,
    baseline_provenance_path: str | Path | None = None,
    candidate_provenance_path: str | Path | None = None,
    detector_endpoint_tolerance_px: float = 5.0e-4,
) -> dict[str, Any]:
    if expected_events < 1:
        raise ConditionalTouchSmokeAuditError("expected_events must be positive")
    if not np.isfinite(detector_endpoint_tolerance_px) or detector_endpoint_tolerance_px < 0:
        raise ConditionalTouchSmokeAuditError(
            "detector endpoint tolerance must be finite and non-negative"
        )
    baseline = load_smoke_manifest(
        baseline_manifest, release_path=baseline_release
    )
    candidate = load_smoke_manifest(
        candidate_manifest, release_path=candidate_release
    )
    baseline_provenance: dict[str, dict[str, Any]] = {}
    candidate_provenance: dict[str, dict[str, Any]] = {}
    baseline_provenance_digest: str | None = None
    candidate_provenance_digest: str | None = None
    if baseline_provenance_path is not None:
        baseline_provenance, baseline_provenance_digest = load_provenance(
            baseline_provenance_path, release_path=baseline_release
        )
    if candidate_provenance_path is not None:
        candidate_provenance, candidate_provenance_digest = load_provenance(
            candidate_provenance_path, release_path=candidate_release
        )

    failures: list[dict[str, Any]] = []

    def fail(code: str, count: int, examples: Sequence[str] = ()) -> None:
        failures.append(
            {
                "code": code,
                "count": int(count),
                "examples": list(examples[:10]),
            }
        )

    if len(baseline.events) != expected_events:
        fail("baseline_event_count", abs(len(baseline.events) - expected_events))
    if len(candidate.events) != expected_events:
        fail("candidate_event_count", abs(len(candidate.events) - expected_events))
    baseline_ids = set(baseline.events)
    candidate_ids = set(candidate.events)
    missing = sorted(baseline_ids - candidate_ids)
    extra = sorted(candidate_ids - baseline_ids)
    if missing:
        fail("candidate_missing_event_ids", len(missing), missing)
    if extra:
        fail("candidate_extra_event_ids", len(extra), extra)
    binding_mismatch = sorted(
        event_id
        for event_id in baseline_ids & candidate_ids
        if baseline.events[event_id].binding != candidate.events[event_id].binding
    )
    if binding_mismatch:
        fail("event_binding_mismatch", len(binding_mismatch), binding_mismatch)

    provenance_binding_mismatch: list[str] = []
    if candidate_provenance_path is not None:
        for event_id, event in candidate.events.items():
            row = candidate_provenance.get(event_id)
            try:
                provenance_binding = (
                    str(row.get("split", "")),
                    str(row.get("user_id", "")),
                    str(row.get("action", "")),
                    int(row.get("label", -1)),
                )
            except (AttributeError, TypeError, ValueError):
                provenance_binding = ("", "", "", -1)
            if provenance_binding != event.binding:
                provenance_binding_mismatch.append(event_id)
        if provenance_binding_mismatch:
            fail(
                "candidate_provenance_binding_mismatch",
                len(provenance_binding_mismatch),
                sorted(provenance_binding_mismatch),
            )

    target_ids = sorted(
        event_id
        for event_id, event in candidate.events.items()
        if event.label == 1 and event.action in TOUCH_ACTIONS
    )
    imu_mismatch: list[str] = []
    time_mismatch: list[str] = []
    oob_events: list[str] = []
    pressure_events: list[str] = []
    no_active_events: list[str] = []
    endpoint_failures: list[str] = []
    request_plan_failures: list[str] = []
    raw_oob_events: list[str] = []
    clipping_events: list[str] = []
    max_raw_endpoint_error = 0.0
    max_detector_endpoint_error = 0.0
    active_pressure_samples = 0
    bad_pressure_samples = 0
    endpoint_events_checked = 0
    request_plan_events_checked = 0
    moving_tap_events = 0
    moving_tap_request_plans_verified = 0
    signal_events_compared = 0
    for event_id in target_ids:
        event = candidate.events[event_id]
        reference = baseline.events.get(event_id)
        if reference is not None:
            signal_events_compared += 1
            if not np.array_equal(event.imu, reference.imu):
                imu_mismatch.append(event_id)
            if not np.array_equal(
                event.trajectory[:, 7], reference.trajectory[:, 7]
            ):
                time_mismatch.append(event_id)
        trajectory = event.trajectory
        if np.any(trajectory[:, 1:3] < 0.0) or np.any(
            trajectory[:, 1:3] > 1.0
        ):
            oob_events.append(event_id)
        active = (trajectory[:, 0] > 0.5) & (trajectory[:, 8] > 0.5)
        if not np.any(active):
            no_active_events.append(event_id)
            continue
        pressure = trajectory[active, 3]
        active_pressure_samples += len(pressure)
        mismatched_pressure = int(np.sum(pressure != np.float32(1.0)))
        bad_pressure_samples += mismatched_pressure
        if mismatched_pressure:
            pressure_events.append(event_id)

        if candidate_provenance_path is None:
            continue
        row = candidate_provenance.get(event_id)
        if not isinstance(row, Mapping) or not isinstance(row.get("donor"), Mapping):
            endpoint_failures.append(event_id)
            continue
        donor = row["donor"]
        try:
            requested = np.vstack(
                (
                    _point_pair(donor.get("requested_start_px"), name="requested start"),
                    _point_pair(donor.get("requested_end_px"), name="requested end"),
                )
            )
            raw_output = np.vstack(
                (
                    _point_pair(donor.get("raw_output_start_px"), name="raw start"),
                    _point_pair(donor.get("raw_output_end_px"), name="raw end"),
                )
            )
            declared_detector = np.vstack(
                (
                    _point_pair(
                        donor.get("detector_output_start_px"), name="detector start"
                    ),
                    _point_pair(
                        donor.get("detector_output_end_px"), name="detector end"
                    ),
                )
            )
            orientation = donor.get("conditioning_orientation_id")
            if not isinstance(orientation, (int, np.integer)):
                raise ConditionalTouchSmokeAuditError("missing conditioning orientation")
            dimensions = np.asarray(SCREEN_DIMENSIONS[int(orientation)], dtype=np.float64)
        except (ConditionalTouchSmokeAuditError, KeyError):
            endpoint_failures.append(event_id)
            continue
        actual_detector = trajectory[[0, -1], 1:3].astype(np.float64) * dimensions
        raw_error = np.linalg.norm(raw_output - requested, axis=1)
        detector_error = np.linalg.norm(actual_detector - requested, axis=1)
        max_raw_endpoint_error = max(max_raw_endpoint_error, float(np.max(raw_error)))
        max_detector_endpoint_error = max(
            max_detector_endpoint_error, float(np.max(detector_error))
        )
        endpoint_events_checked += 1
        moving_tap = bool(
            event.action == "tap"
            and not np.array_equal(requested[0], requested[1])
        )
        if moving_tap:
            moving_tap_events += 1
        request_plan_payload = donor.get("request_plan")
        reference_request_valid = bool(
            donor.get("request_source")
            == "frozen_smoke_reference_exact_endpoints"
            and reference is not None
            and np.array_equal(
                reference.trajectory[[0, -1], 1:3].astype(np.float64)
                * dimensions[None, :],
                requested,
            )
        )
        request_plan_valid = not moving_tap or reference_request_valid
        if request_plan_payload is not None:
            request_plan_events_checked += 1
            try:
                request_plan = ConditionalTouchRequestPlan.from_json_dict(
                    request_plan_payload
                )
            except (ConditionalTouchRequestPlanError, TypeError, ValueError):
                request_plan_valid = False
            else:
                donor_direction_present = "conditioning_direction" in donor
                donor_direction = donor.get("conditioning_direction")
                request_plan_valid = bool(
                    request_plan.carrier_event_id == event.event_id
                    and request_plan.action == event.action
                    and donor.get("conditioning_action") == event.action
                    and request_plan.orientation_id == int(orientation)
                    and np.array_equal(
                        np.asarray(
                            request_plan.sampled_start_xy_px,
                            dtype=np.float64,
                        ),
                        requested[0],
                    )
                    and np.array_equal(
                        np.asarray(
                            request_plan.sampled_end_xy_px,
                            dtype=np.float64,
                        ),
                        requested[1],
                    )
                    and donor_direction_present
                    and request_plan.sampled_direction == donor_direction
                )
        if not request_plan_valid:
            request_plan_failures.append(event_id)
        elif moving_tap:
            moving_tap_request_plans_verified += 1
        invalid = (
            not np.array_equal(raw_output, requested)
            or float(np.max(detector_error)) > detector_endpoint_tolerance_px
            or not np.allclose(
                declared_detector, actual_detector, rtol=0.0, atol=1.0e-8
            )
        )
        if invalid:
            endpoint_failures.append(event_id)
        if np.any(raw_output < 0.0) or np.any(raw_output > dimensions):
            raw_oob_events.append(event_id)
        if (
            row.get("coordinate_clipping_used") is not False
            or donor.get("coordinate_clipping_used") is not False
        ):
            clipping_events.append(event_id)

    if imu_mismatch:
        fail("fake_touch_imu_flat_mismatch", len(imu_mismatch), imu_mismatch)
    if time_mismatch:
        fail("fake_touch_time_axis_mismatch", len(time_mismatch), time_mismatch)
    if oob_events:
        fail("candidate_detector_xy_out_of_bounds", len(oob_events), oob_events)
    if no_active_events:
        fail("candidate_touch_has_no_active_rows", len(no_active_events), no_active_events)
    if pressure_events:
        fail("candidate_active_pressure_not_exactly_one", len(pressure_events), pressure_events)
    if endpoint_failures:
        fail("candidate_endpoint_mismatch", len(set(endpoint_failures)), sorted(set(endpoint_failures)))
    if request_plan_failures:
        fail(
            "candidate_request_plan_mismatch",
            len(set(request_plan_failures)),
            sorted(set(request_plan_failures)),
        )
    if raw_oob_events:
        fail("candidate_raw_endpoint_out_of_bounds", len(raw_oob_events), raw_oob_events)
    if clipping_events:
        fail("candidate_coordinate_clipping_used", len(clipping_events), clipping_events)

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "inputs": {
            "baseline_manifest": str(baseline.manifest),
            "baseline_manifest_sha256": baseline.manifest_sha256,
            "candidate_manifest": str(candidate.manifest),
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "baseline_provenance_sha256": baseline_provenance_digest,
            "candidate_provenance_sha256": candidate_provenance_digest,
        },
        "hard_invariants": {
            "status": "pass" if not failures else "fail",
            "expected_events": int(expected_events),
            "baseline_events": len(baseline.events),
            "candidate_events": len(candidate.events),
            "exact_event_id_set": not missing and not extra,
            "exact_split_user_action_label_binding": not binding_mismatch,
            "candidate_fake_touch_events": len(target_ids),
            "fake_touch_events_compared": signal_events_compared,
            "imu_flat_exact_events": signal_events_compared - len(imu_mismatch),
            "time_axis_exact_events": signal_events_compared - len(time_mismatch),
            "candidate_detector_oob_events": len(oob_events),
            "candidate_active_pressure_samples": active_pressure_samples,
            "candidate_active_pressure_mismatch_samples": bad_pressure_samples,
            "candidate_raw_endpoint_check": (
                "checked_from_provenance"
                if candidate_provenance_path is not None
                else "not_available_without_candidate_provenance"
            ),
            "raw_oob_coverage": (
                "raw_endpoints_from_provenance_and_all_detector_grid_rows"
                if candidate_provenance_path is not None
                else "all_detector_grid_rows_only"
            ),
            "endpoint_events_checked": endpoint_events_checked,
            "request_plan_events_checked": request_plan_events_checked,
            "moving_tap_events": moving_tap_events,
            "moving_tap_request_plans_verified": (
                moving_tap_request_plans_verified
            ),
            "maximum_raw_endpoint_error_px": (
                max_raw_endpoint_error if endpoint_events_checked else None
            ),
            "maximum_detector_endpoint_error_px": (
                max_detector_endpoint_error if endpoint_events_checked else None
            ),
            "detector_endpoint_tolerance_px": detector_endpoint_tolerance_px,
            "failures": failures,
        },
        "counts": {
            "baseline_split": dict(baseline.split_counts),
            "candidate_split": dict(candidate.split_counts),
            "baseline_action_label": dict(sorted(baseline.action_label_counts.items())),
            "candidate_action_label": dict(sorted(candidate.action_label_counts.items())),
        },
        "signal_semantics": {
            "pure_six_axis_array": "imu_flat",
            "time_axis": "trajectory_flat[:, 7]",
            "imu_only_is_pure_imu": True,
            "imu_only_channels": [
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            ],
            "consequence": (
                "the imu_only modality reads no touch channel, so a trajectory "
                "change leaves both imu_flat and the imu_only detector input "
                "unchanged"
            ),
        },
        "distribution_diagnostics": {
            "gate_authority": "report_only_not_a_detector_acceptance_gate",
            "metric_definitions": {
                "path_chord_ratio": "detector-grid active path length divided by endpoint chord",
                "reverse_progress_fraction": "fraction of non-zero detector steps opposite the endpoint chord",
                "lateral_step_rms_px": "RMS increment perpendicular to the endpoint chord",
                "lateral_residual_rms_px": "RMS point residual perpendicular to the endpoint chord",
                "half_pixel_gridpoint_fraction": "active points on the inferred/provenance HMOG 0.5 px lattice",
                "zero_step_fraction": "fraction of consecutive active detector rows with unchanged x/y",
            },
            "actions": _distribution_report(
                baseline,
                candidate,
                baseline_provenance,
                candidate_provenance,
            ),
        },
    }
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
