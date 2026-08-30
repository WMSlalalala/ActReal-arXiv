#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "personal_android_imu_touch_v2"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
FORMAL_GEOMETRY = {
    "tap": (35, 10, 100.0),
    "scroll": (179, 10, 100.0),
    "swipe": (167, 10, 100.0),
    "pinch": (116, 10, 100.0),
    "keystroke": (256, 0, 100.0),
}
HMOG_PORTRAIT_WIDTH_PX = 1080.0
HMOG_PORTRAIT_HEIGHT_PX = 1920.0
REQUIRED_FILES = (
    "manifest.json",
    "imu.csv",
    "touch.csv",
    "events.csv",
    "task_events.csv",
    "keystroke.csv",
    "capture_health.csv",
    "export_audit.json",
)


@dataclass(frozen=True)
class Session:
    root: Path
    manifest: Mapping[str, Any]
    events: List[Dict[str, str]]
    imu: List[Dict[str, str]]
    task_events: List[Dict[str, str]]
    capture_health: List[Dict[str, str]]
    export_audit: Mapping[str, Any]
    source_path: Path
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_csv_file(path: Path, session_id: str) -> int:
    count = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(not name for name in reader.fieldnames):
            raise ValueError("%s has an invalid header" % path.name)
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("%s has duplicate header fields" % path.name)
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "%s row %d does not match its header column count"
                    % (path.name, line_number)
                )
            if row.get("schema") != SCHEMA:
                raise ValueError(
                    "%s row %d has schema %r"
                    % (path.name, line_number, row.get("schema"))
                )
            if row.get("session_id") != session_id:
                raise ValueError(
                    "%s row %d belongs to another session"
                    % (path.name, line_number)
                )
            count += 1
    return count


def _session_directory(path: Path) -> Path:
    if (path / "manifest.json").is_file():
        return path
    candidates = sorted(p.parent for p in path.rglob("manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            "export must contain exactly one session manifest; found %d" % len(candidates)
        )
    return candidates[0]


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("unsafe ZIP member: %s" % info.filename)
        handle.extractall(destination)


@contextlib.contextmanager
def open_session(source: Path) -> Iterator[Session]:
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    temporary: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if source.is_file():
            temporary = tempfile.TemporaryDirectory(prefix="phone_study_")
            unpacked = Path(temporary.name)
            _safe_extract(source, unpacked)
            root = _session_directory(unpacked)
            source_hash = sha256_file(source)
        else:
            root = _session_directory(source)
            digest = hashlib.sha256()
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                digest.update(str(path.relative_to(root)).encode("utf-8"))
                digest.update(bytes.fromhex(sha256_file(path)))
            source_hash = digest.hexdigest()

        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError("session is missing files: %s" % ", ".join(missing))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unexpected manifest schema %r" % manifest.get("schema"))
        session_id = str(manifest.get("session_id", ""))
        if not session_id:
            raise ValueError("manifest has no session_id")
        for csv_path in sorted(root.glob("*.csv")):
            validate_csv_file(csv_path, session_id)
        events = read_csv(root / "events.csv")
        imu = read_csv(root / "imu.csv")
        task_events = read_csv(root / "task_events.csv")
        capture_health = read_csv(root / "capture_health.csv")
        export_audit = json.loads(
            (root / "export_audit.json").read_text(encoding="utf-8")
        )
        if export_audit.get("schema") != SCHEMA:
            raise ValueError("unexpected export audit schema")
        if int(export_audit.get("csv_pending_rows_after_flush", -1)) != 0:
            raise ValueError("export occurred with pending CSV rows")
        if int(export_audit.get("csv_write_errors", -1)) != 0:
            raise ValueError("Android CSV writer reported errors")
        if int(export_audit.get("csv_dropped_rows", -1)) != 0:
            raise ValueError("Android CSV writer dropped rows")
        yield Session(
            root=root,
            manifest=manifest,
            events=events,
            imu=imu,
            task_events=task_events,
            capture_health=capture_health,
            export_audit=export_audit,
            source_path=source,
            source_sha256=source_hash,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def formal_geometry(action: str) -> Tuple[int, int, float]:
    try:
        return FORMAL_GEOMETRY[action]
    except KeyError as exc:
        raise ValueError(f"unknown action {action!r}") from exc


def optional_float(row: Mapping[str, str], key: str) -> float:
    text = row.get(key, "")
    if text is None or not str(text).strip():
        return float("nan")
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def optional_int(row: Mapping[str, str], key: str, default: int = -1) -> int:
    text = row.get(key, "")
    if text is None or not str(text).strip():
        return int(default)
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return int(default)


def _sensor_arrays(session: Session, sensor: str) -> Tuple[np.ndarray, np.ndarray]:
    selected = [row for row in session.imu if row.get("sensor") == sensor]
    if not selected:
        raise ValueError("session contains no %s rows" % sensor)
    time_ns = np.asarray(
        [int(row["event_elapsed_ns"]) for row in selected], dtype=np.int64
    )
    values = np.asarray(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in selected],
        dtype=np.float64,
    )
    finite = np.isfinite(values).all(axis=1)
    order = np.argsort(time_ns[finite], kind="stable")
    time_ns = time_ns[finite][order]
    values = values[finite][order]
    unique = np.r_[True, np.diff(time_ns) > 0]
    return time_ns[unique], values[unique]


def sensor_index(session: Session) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {
        "accelerometer": _sensor_arrays(session, "accelerometer"),
        "gyroscope": _sensor_arrays(session, "gyroscope"),
    }


def _interpolate(
    time_ns: np.ndarray,
    values: np.ndarray,
    target_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = (target_ns >= time_ns[0]) & (target_ns <= time_ns[-1])
    result = np.stack(
        [np.interp(target_ns, time_ns, values[:, channel]) for channel in range(3)],
        axis=1,
    ).astype(np.float32)
    return result, valid


def event_window(
    event: Mapping[str, str],
    sensors: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    action = str(event["action"])
    if action not in ACTIONS:
        raise ValueError("unknown action %s" % action)
    T, pad_pre, hz = formal_geometry(action)
    period_ns = int(round(1_000_000_000.0 / hz))
    start_ns = int(event["start_elapsed_ns"])
    end_ns = int(event["end_elapsed_ns"])
    if end_ns <= start_ns:
        raise ValueError("non-positive event duration for %s" % event.get("event_id"))
    grid_start = start_ns if action == "keystroke" else start_ns - pad_pre * period_ns
    target_ns = grid_start + np.arange(T, dtype=np.int64) * period_ns
    acc, valid_acc = _interpolate(*sensors["accelerometer"], target_ns)
    gyro, valid_gyro = _interpolate(*sensors["gyroscope"], target_ns)
    window = np.concatenate([acc, gyro], axis=1).astype(np.float32)
    valid = (valid_acc & valid_gyro).astype(np.uint8)
    logical_frames = (
        int(math.ceil((end_ns - start_ns) / period_ns))
        if action == "keystroke"
        else int(round((end_ns - start_ns) / period_ns))
    )
    logical_frames = max(1, logical_frames)
    max_active = T if action == "keystroke" else T - pad_pre
    active_len = min(logical_frames, max_active)
    mask = np.zeros(T, dtype=np.uint8)
    begin = 0 if action == "keystroke" else pad_pre
    mask[begin : begin + active_len] = 1
    if np.any(mask > valid):
        raise ValueError(
            "event %s lacks sensor coverage in its active interval"
            % event.get("event_id")
        )
    return {
        "window": window,
        "mask": mask,
        "valid_mask": valid,
        "active_len": np.asarray(active_len, dtype=np.int64),
        "logical_active_len": np.asarray(logical_frames, dtype=np.int64),
        "target_elapsed_ns": target_ns,
        "duration_ms": np.asarray(
            (end_ns - start_ns) / 1_000_000.0, dtype=np.float32
        ),
    }


def event_conditions(event: Mapping[str, str]) -> Dict[str, Any]:
    action = str(event["action"])
    result: Dict[str, Any] = {
        "event_id": event["event_id"],
        "action": action,
        "duration_ms": float(event["duration_ms"]),
        "orientation_id": optional_int(event, "orientation_id", -1),
        "source": event.get("source", ""),
        "condition_quality": event.get("condition_quality", ""),
        "xy_source": event.get("xy_source", ""),
        "observable_fields": event.get("observable_fields", ""),
        "n_letters": optional_int(event, "n_letters", -1),
        "n_keys": optional_int(event, "n_keys", -1),
    }
    xy_raw = np.asarray(
        [
            optional_float(event, "x_start"),
            optional_float(event, "y_start"),
            optional_float(event, "x_end"),
            optional_float(event, "y_end"),
        ],
        dtype=np.float32,
    )
    result["xy_raw"] = xy_raw
    width = optional_float(event, "display_width_px")
    height = optional_float(event, "display_height_px")
    result["display_width_px"] = width
    result["display_height_px"] = height
    if np.isfinite(xy_raw).all():
        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError(
                "event %s has XY but no valid display dimensions"
                % event.get("event_id")
            )
        orientation = int(result["orientation_id"])
        if orientation in (1, 3):
            target_width = HMOG_PORTRAIT_HEIGHT_PX
            target_height = HMOG_PORTRAIT_WIDTH_PX
        else:
            target_width = HMOG_PORTRAIT_WIDTH_PX
            target_height = HMOG_PORTRAIT_HEIGHT_PX
        xy = xy_raw.copy()
        xy[[0, 2]] *= float(target_width / width)
        xy[[1, 3]] *= float(target_height / height)
        result["xy"] = xy.astype(np.float32)
        result["xy_transform"] = (
            "display_pixels_to_hmog_%dx%d"
            % (int(target_width), int(target_height))
        )
    else:
        result["xy"] = xy_raw
        result["xy_transform"] = "not_applied_missing_xy"
    return result


def selected_events(
    session: Session,
    *,
    task: Optional[str] = None,
    source: Optional[str] = None,
    actions: Sequence[str] = ACTIONS,
    run_id: Optional[str] = None,
    completed_only: bool = False,
) -> List[Dict[str, str]]:
    completed = completed_run_ids(session) if completed_only else set()
    result = []
    for row in session.events:
        if row.get("action") not in actions:
            continue
        if task is not None and row.get("task") != task:
            continue
        if source is not None and row.get("source") != source:
            continue
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if completed_only and row.get("run_id") not in completed:
            continue
        result.append(row)
    result.sort(key=lambda row: int(row["start_elapsed_ns"]))
    return result


def completed_run_ids(session: Session) -> set:
    return {
        row.get("run_id", "")
        for row in session.task_events
        if row.get("run_id")
        and row.get("event") in ("calibration_complete", "task_complete")
    }


def validate_run_health(
    session: Session,
    run_ids: Sequence[str],
    *,
    minimum_effective_hz: float = 75.0,
    maximum_gap_ms: float = 100.0,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for run_id in sorted(set(run_ids)):
        rows = [row for row in session.capture_health if row.get("run_id") == run_id]
        if len(rows) != 1:
            raise ValueError(
                "completed run %s must have exactly one capture_health row; got %d"
                % (run_id, len(rows))
            )
        row = rows[0]
        acc_hz = float(row["accelerometer_effective_hz"])
        gyro_hz = float(row["gyroscope_effective_hz"])
        acc_gap = float(row["accelerometer_max_gap_ms"])
        gyro_gap = float(row["gyroscope_max_gap_ms"])
        if int(row["accelerometer_rows"]) <= 1 or int(row["gyroscope_rows"]) <= 1:
            raise ValueError("completed run %s has insufficient IMU rows" % run_id)
        if acc_hz < minimum_effective_hz or gyro_hz < minimum_effective_hz:
            raise ValueError(
                "completed run %s sampling rate too low: acc=%.3f gyro=%.3f"
                % (run_id, acc_hz, gyro_hz)
            )
        if acc_gap > maximum_gap_ms or gyro_gap > maximum_gap_ms:
            raise ValueError(
                "completed run %s sensor gap too large: acc=%.3fms gyro=%.3fms"
                % (run_id, acc_gap, gyro_gap)
            )
        if int(row["sensor_write_errors"]) != 0:
            raise ValueError("completed run %s has sensor write errors" % run_id)
        event_postures = {
            event.get("posture", "")
            for event in session.events
            if event.get("run_id") == run_id
        }
        if len(event_postures) != 1 or next(iter(event_postures)) not in (
            "sitting",
            "walking",
        ):
            raise ValueError(
                "completed run %s has invalid or mixed event posture: %s"
                % (run_id, sorted(event_postures))
            )
        posture = next(iter(event_postures))
        completion_postures = {
            event.get("posture", "")
            for event in session.task_events
            if event.get("run_id") == run_id
            and event.get("event") in ("calibration_complete", "task_complete")
        }
        if completion_postures != {posture}:
            raise ValueError(
                "completed run %s completion posture differs: %s"
                % (run_id, sorted(completion_postures))
            )
        if row.get("posture", "") != posture:
            raise ValueError("completed run %s health posture differs" % run_id)
        result[run_id] = {
            "accelerometer_effective_hz": acc_hz,
            "gyroscope_effective_hz": gyro_hz,
            "accelerometer_max_gap_ms": acc_gap,
            "gyroscope_max_gap_ms": gyro_gap,
            "posture": posture,
        }
    return result
