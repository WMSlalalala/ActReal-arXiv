#!/usr/bin/env python3
"""Fit the train-only HMOG conditional tap/scroll/swipe generator.

The saved model contains statistical parameters only.  Raw rows, event IDs,
and donor trajectories are deliberately excluded from the runtime artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.conditional_touch_generator import (  # noqa: E402
    ConditionalTouchGenerator,
)
from pipeline.android_touch_observation import (  # noqa: E402
    screen_dimensions_for_orientation,
)


ACTIONS = ("tap", "scroll", "swipe")
AUDIT_SCHEMA = "hmog_conditional_touch_generator_fit_audit_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "licensed_input" / "hmog" / "processed_trajectories"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "users_seed42.json"
REQUIRED_EVENT_KEYS = (
    "event_id",
    "user_id",
    "orientation_id",
    "event_offsets",
)
REQUIRED_ROW_KEYS = (
    "flat_t_rel_ms",
    "flat_x",
    "flat_y",
    "flat_pressure",
    "flat_action_code",
    "flat_valid_mask",
)


class ConditionalTouchFitCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchivePlan:
    action: str
    path: Path
    sha256: str
    selected_events: np.ndarray
    selected_event_count: int
    selected_row_count: int
    event_count: int
    row_count: int
    audit: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConditionalTouchFitCliError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ConditionalTouchFitCliError(f"expected a JSON object: {path}")
    return value


def _load_train_users(path: Path) -> tuple[int, ...]:
    payload = _json_object(path)
    raw = payload.get("train_users")
    if not isinstance(raw, list) or not raw:
        raise ConditionalTouchFitCliError("split JSON has no train_users")
    try:
        users = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ConditionalTouchFitCliError("train_users contains a non-integer") from exc
    if len(users) != len(set(users)):
        raise ConditionalTouchFitCliError("train_users contains duplicates")
    for key in ("val_users", "test_users"):
        other = payload.get(key, [])
        if not isinstance(other, list):
            raise ConditionalTouchFitCliError(f"split JSON {key} is not a list")
        if set(users) & {int(value) for value in other}:
            raise ConditionalTouchFitCliError(
                f"train_users overlaps split JSON {key}"
            )
    return users


def _coerce_action_id_mapping(
    value: Mapping[str, Any], *, source: Path
) -> dict[str, set[str]]:
    nested = value.get("accepted_source_event_ids_by_action")
    mapping = nested if isinstance(nested, dict) else value
    result: dict[str, set[str]] = {}
    for action in ACTIONS:
        raw = mapping.get(action)
        if not isinstance(raw, list):
            raise ConditionalTouchFitCliError(
                f"{source}: accepted IDs must contain a list for {action}"
            )
        identifiers = {str(item) for item in raw}
        if not identifiers:
            raise ConditionalTouchFitCliError(
                f"{source}: accepted ID list for {action} is empty"
            )
        result[action] = identifiers
    return result


def _accepted_ids_json(path: Path) -> tuple[dict[str, set[str]], Mapping[str, Any]]:
    values = _coerce_action_id_mapping(_json_object(path), source=path)
    return values, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "accepted_ids_by_action": {
            action: len(values[action]) for action in ACTIONS
        },
    }


def _accepted_ids_quality_ledger(
    path: Path,
) -> tuple[dict[str, set[str]], Mapping[str, Any]]:
    accepted = {action: set() for action in ACTIONS}
    rows = 0
    applicable_rows = 0
    accepted_rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConditionalTouchFitCliError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise ConditionalTouchFitCliError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                action = str(row.get("source_action_label", ""))
                if action not in accepted:
                    continue
                applicable_rows += 1
                if not (
                    row.get("decision") == "include"
                    and row.get("eligible") is True
                    and row.get("raw_trajectory_extraction_status") == "accepted"
                ):
                    continue
                source_event_id = str(row.get("source_event_id", ""))
                if not source_event_id:
                    raise ConditionalTouchFitCliError(
                        f"{path}:{line_number}: accepted row has no source_event_id"
                    )
                accepted[action].add(source_event_id)
                accepted_rows += 1
    except OSError as exc:
        raise ConditionalTouchFitCliError(
            f"cannot read quality ledger: {path}"
        ) from exc
    for action in ACTIONS:
        if not accepted[action]:
            raise ConditionalTouchFitCliError(
                f"quality ledger has no accepted {action} source IDs"
            )
    return accepted, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": rows,
        "applicable_rows": applicable_rows,
        "accepted_rows": accepted_rows,
        "unique_accepted_ids_by_action": {
            action: len(accepted[action]) for action in ACTIONS
        },
    }


def _combine_allowlists(
    sources: Iterable[dict[str, set[str]]],
) -> dict[str, set[str]] | None:
    values = list(sources)
    if not values:
        return None
    combined = {action: set(values[0][action]) for action in ACTIONS}
    for value in values[1:]:
        for action in ACTIONS:
            combined[action].intersection_update(value[action])
    for action in ACTIONS:
        if not combined[action]:
            raise ConditionalTouchFitCliError(
                f"accepted-source filters have an empty {action} intersection"
            )
    return combined


def _validate_archive_layout(
    archive: Mapping[str, np.ndarray], *, action: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    files = set(getattr(archive, "files", ()))
    missing = set(REQUIRED_EVENT_KEYS + REQUIRED_ROW_KEYS + ("action_name",)) - files
    if missing:
        raise ConditionalTouchFitCliError(
            f"{path}: missing archive keys {sorted(missing)}"
        )
    observed_action = str(np.asarray(archive["action_name"]).item())
    if observed_action != action:
        raise ConditionalTouchFitCliError(
            f"{path}: action_name is {observed_action!r}, expected {action!r}"
        )
    event_ids = np.asarray(archive["event_id"])
    offsets = np.asarray(archive["event_offsets"], dtype=np.int64)
    if (
        event_ids.ndim != 1
        or offsets.ndim != 1
        or len(offsets) != len(event_ids) + 1
        or len(offsets) < 2
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ConditionalTouchFitCliError(f"{path}: invalid event_offsets")
    event_count = len(event_ids)
    for key in ("user_id", "orientation_id"):
        value = np.asarray(archive[key])
        if value.ndim != 1 or len(value) != event_count:
            raise ConditionalTouchFitCliError(
                f"{path}: {key} does not match event_id"
            )
    row_count = int(offsets[-1])
    for key in REQUIRED_ROW_KEYS:
        value = np.asarray(archive[key])
        if value.ndim != 1 or len(value) != row_count:
            raise ConditionalTouchFitCliError(
                f"{path}: {key} does not match event_offsets"
            )
    return event_ids, offsets


def _plan_archive(
    *,
    action: str,
    path: Path,
    train_users: tuple[int, ...],
    allowed_ids: set[str] | None,
) -> ArchivePlan:
    if not path.is_file():
        raise ConditionalTouchFitCliError(f"raw archive is missing: {path}")
    source_sha256 = _sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        event_ids, offsets = _validate_archive_layout(
            archive, action=action, path=path
        )
        users = np.asarray(archive["user_id"], dtype=np.int64)
        orientations = np.asarray(archive["orientation_id"], dtype=np.int64)
        train_mask = np.isin(users, np.asarray(train_users, dtype=np.int64))
        orientation_mask = np.isin(orientations, np.asarray((0, 1, 3)))
        if allowed_ids is None:
            allow_mask = np.ones(len(event_ids), dtype=np.bool_)
        else:
            allow_mask = np.fromiter(
                (str(np.asarray(value).item()) in allowed_ids for value in event_ids),
                dtype=np.bool_,
                count=len(event_ids),
            )
        valid_rows = np.asarray(archive["flat_valid_mask"], dtype=np.bool_)
        invalid_prefix = np.empty(len(valid_rows) + 1, dtype=np.int64)
        invalid_prefix[0] = 0
        np.cumsum(~valid_rows, dtype=np.int64, out=invalid_prefix[1:])
        valid_event_mask = (
            invalid_prefix[offsets[1:]] - invalid_prefix[offsets[:-1]] == 0
        )
        selected = train_mask & allow_mask & orientation_mask & valid_event_mask
        selected_rows = int(np.sum(np.diff(offsets)[selected], dtype=np.int64))
        remaining = train_mask
        non_train = int(np.count_nonzero(~train_mask))
        not_accepted = int(np.count_nonzero(remaining & ~allow_mask))
        remaining = remaining & allow_mask
        unsupported_orientation = int(
            np.count_nonzero(remaining & ~orientation_mask)
        )
        remaining = remaining & orientation_mask
        invalid_raw_rows = int(np.count_nonzero(remaining & ~valid_event_mask))
        audit = {
            "action": action,
            "path": str(path),
            "sha256": source_sha256,
            "events_total": len(event_ids),
            "rows_total": int(offsets[-1]),
            "events_selected": int(np.count_nonzero(selected)),
            "rows_selected": selected_rows,
            "events_excluded": {
                "non_train_user": non_train,
                "not_accepted_by_optional_filter": not_accepted,
                "unsupported_orientation": unsupported_orientation,
                "contains_invalid_raw_row": invalid_raw_rows,
            },
        }
    if not np.any(selected):
        raise ConditionalTouchFitCliError(f"no eligible {action} training events")
    return ArchivePlan(
        action=action,
        path=path,
        sha256=source_sha256,
        selected_events=selected,
        selected_event_count=int(np.count_nonzero(selected)),
        selected_row_count=selected_rows,
        event_count=len(selected),
        row_count=int(audit["rows_total"]),
        audit=audit,
    )


def _materialize_training_rows(plans: Iterable[ArchivePlan]) -> dict[str, np.ndarray]:
    plan_values = tuple(plans)
    total_rows = sum(plan.selected_row_count for plan in plan_values)
    rows = {
        "event_id": np.empty(total_rows, dtype=np.int64),
        "action": np.empty(total_rows, dtype="U6"),
        "orientation_id": np.empty(total_rows, dtype=np.int8),
        "t_ms": np.empty(total_rows, dtype=np.float64),
        "x_px": np.empty(total_rows, dtype=np.float64),
        "y_px": np.empty(total_rows, dtype=np.float64),
        "pressure": np.empty(total_rows, dtype=np.float64),
        "android_action": np.empty(total_rows, dtype=np.int64),
    }
    cursor = 0
    event_base = 0
    source_keys = {
        "t_ms": "flat_t_rel_ms",
        "x_px": "flat_x",
        "y_px": "flat_y",
        "pressure": "flat_pressure",
        "android_action": "flat_action_code",
    }
    for plan in plan_values:
        with np.load(plan.path, allow_pickle=False) as archive:
            _, offsets = _validate_archive_layout(
                archive, action=plan.action, path=plan.path
            )
            lengths = np.diff(offsets)
            row_events = np.repeat(
                np.arange(plan.event_count, dtype=np.int64), lengths
            )
            positions = np.flatnonzero(plan.selected_events[row_events])
            count = len(positions)
            if count != plan.selected_row_count:
                raise ConditionalTouchFitCliError(
                    f"{plan.path}: selected row count changed between passes"
                )
            output = slice(cursor, cursor + count)
            selected_event_indexes = row_events[positions]
            rows["event_id"][output] = event_base + selected_event_indexes
            rows["action"][output] = plan.action
            orientations = np.asarray(archive["orientation_id"], dtype=np.int8)
            rows["orientation_id"][output] = orientations[selected_event_indexes]
            for output_key, source_key in source_keys.items():
                rows[output_key][output] = np.asarray(archive[source_key])[positions]
            cursor += count
            event_base += plan.event_count
    if cursor != total_rows:
        raise ConditionalTouchFitCliError("training row materialization was incomplete")
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _verify_generated_endpoints(
    model: ConditionalTouchGenerator,
) -> Mapping[str, Any]:
    """Exercise every fitted condition without consulting any source row."""

    residual_scales: list[float] = []
    generated_events = 0
    for action, orientation_id in model.supported_conditions:
        width, height = screen_dimensions_for_orientation(orientation_id)
        start = np.asarray((0.30 * width, 0.50 * height), dtype=np.float64)
        end = (
            start.copy()
            if action == "tap"
            else np.asarray((0.70 * width, 0.50 * height), dtype=np.float64)
        )
        expected = np.vstack((start, end))
        for seed in range(10):
            generated = model.generate(
                action=action,
                orientation_id=orientation_id,
                start_xy_px=start,
                end_xy_px=end,
                direction=None if action == "tap" else "right",
                seed=seed,
                duration_ms=300.0,
                sample_count=31,
            )
            endpoints = np.asarray(
                (
                    (generated.x_px[0], generated.y_px[0]),
                    (generated.x_px[-1], generated.y_px[-1]),
                ),
                dtype=np.float64,
            )
            if not np.array_equal(endpoints, expected):
                raise ConditionalTouchFitCliError(
                    f"saved model missed exact endpoints for {action}|{orientation_id}"
                )
            if (
                np.any(generated.x_px < 0.0)
                or np.any(generated.x_px > width)
                or np.any(generated.y_px < 0.0)
                or np.any(generated.y_px > height)
            ):
                raise ConditionalTouchFitCliError(
                    f"saved model generated off-screen {action}|{orientation_id} rows"
                )
            residual_scales.append(float(generated.residual_scale))
            generated_events += 1
    return {
        "runtime_source_lookup": False,
        "conditions_checked": len(model.supported_conditions),
        "generated_events_checked": generated_events,
        "seeds_per_condition": 10,
        "exact_endpoint_failures": 0,
        "off_screen_failures": 0,
        "minimum_residual_scale_observed": min(residual_scales),
        "maximum_residual_scale_observed": max(residual_scales),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a compact train-user-only conditional touch generator for "
            "tap, scroll, and swipe."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--output-audit",
        type=Path,
        help="Default: OUTPUT_MODEL with its suffix replaced by .audit.json.",
    )
    parser.add_argument(
        "--accepted-source-ids-json",
        type=Path,
        help=(
            "Optional JSON mapping tap/scroll/swipe to accepted event-ID lists."
        ),
    )
    parser.add_argument(
        "--quality-ledger",
        type=Path,
        help=(
            "Optional dispatch quality JSONL; only eligible/include/accepted "
            "source events are fitted."
        ),
    )
    parser.add_argument("--grid-size", type=int, default=33)
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--minimum-events", type=int, default=3)
    parser.add_argument("--tap-stationary-tolerance-px", type=float, default=1.0e-6)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model/audit outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    raw_root = args.raw_root.resolve()
    split_path = args.split_json.resolve()
    output_model = args.output_model.resolve()
    output_audit = (
        args.output_audit.resolve()
        if args.output_audit is not None
        else output_model.with_suffix(".audit.json")
    )
    if output_model.suffix != ".npz":
        raise ConditionalTouchFitCliError("--output-model must end in .npz")
    if output_model == output_audit:
        raise ConditionalTouchFitCliError("model and audit outputs must differ")
    if not args.overwrite:
        existing = [path for path in (output_model, output_audit) if path.exists()]
        if existing:
            raise ConditionalTouchFitCliError(
                "output exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)

    train_users = _load_train_users(split_path)
    allowlist_values: list[dict[str, set[str]]] = []
    filter_audits: dict[str, Any] = {}
    if args.accepted_source_ids_json is not None:
        accepted_path = args.accepted_source_ids_json.resolve()
        value, audit = _accepted_ids_json(accepted_path)
        allowlist_values.append(value)
        filter_audits["accepted_source_ids_json"] = audit
    if args.quality_ledger is not None:
        ledger_path = args.quality_ledger.resolve()
        value, audit = _accepted_ids_quality_ledger(ledger_path)
        allowlist_values.append(value)
        filter_audits["quality_ledger"] = audit
    allowlists = _combine_allowlists(allowlist_values)
    if allowlists is not None:
        filter_audits["combined_unique_ids_by_action"] = {
            action: len(allowlists[action]) for action in ACTIONS
        }

    plans = tuple(
        _plan_archive(
            action=action,
            path=(raw_root / f"hmog_trajectory_{action}.npz").resolve(),
            train_users=train_users,
            allowed_ids=None if allowlists is None else allowlists[action],
        )
        for action in ACTIONS
    )
    training_rows = _materialize_training_rows(plans)
    model = ConditionalTouchGenerator.fit_from_raw_rows(
        training_rows,
        grid_size=int(args.grid_size),
        max_rank=int(args.max_rank),
        ridge=float(args.ridge),
        minimum_events=int(args.minimum_events),
        tap_stationary_tolerance_px=float(args.tap_stationary_tolerance_px),
        training_source_sha256s=tuple(plan.sha256 for plan in plans),
    )
    model.save(output_model)
    loaded = ConditionalTouchGenerator.load(output_model)
    if loaded.artifact_sha256 != model.artifact_sha256:
        raise ConditionalTouchFitCliError(
            "saved model failed artifact-digest round-trip verification"
        )
    generation_verification = _verify_generated_endpoints(loaded)

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "artifact": {
            "path": str(output_model),
            "file_sha256": _sha256_file(output_model),
            "artifact_sha256": str(model.artifact_sha256),
            "schema_version": str(model.schema_version),
            "stores_raw_rows_or_donors": False,
        },
        "inputs": {
            "raw_root": str(raw_root),
            "split_json": str(split_path),
            "split_json_sha256": _sha256_file(split_path),
            "train_users": list(train_users),
            "train_user_count": len(train_users),
            "archives": [plan.audit for plan in plans],
            "optional_filters": filter_audits,
        },
        "fit_configuration": {
            "grid_size": int(args.grid_size),
            "max_rank": int(args.max_rank),
            "ridge": float(args.ridge),
            "minimum_events": int(args.minimum_events),
            "tap_stationary_tolerance_px": float(
                args.tap_stationary_tolerance_px
            ),
        },
        "training_summary": model.training_summary,
        "model_metadata": model.metadata,
        "generation_verification": generation_verification,
    }
    _write_json(output_audit, audit)
    result = {
        "model": str(output_model),
        "audit": str(output_audit),
        "artifact_sha256": str(model.artifact_sha256),
        "model_file_sha256": audit["artifact"]["file_sha256"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except (ConditionalTouchFitCliError, ValueError, OSError) as exc:
        print(f"CONDITIONAL TOUCH FIT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
