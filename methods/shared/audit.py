from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, decimals: int = 7) -> str:
    value = np.asarray(array)
    if np.issubdtype(value.dtype, np.floating):
        value = np.round(value.astype(np.float64), decimals=decimals)
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def create_immutable_run_dir(root: str | Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("run_id must be a single safe path component")
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_new(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def assert_disjoint_user_splits(
    train_users: Iterable[int],
    development_users: Iterable[int],
    evaluation_users: Iterable[int],
) -> dict[str, int | bool]:
    train = set(map(int, train_users))
    development = set(map(int, development_users))
    evaluation = set(map(int, evaluation_users))
    overlap = (train & development) | (train & evaluation) | (development & evaluation)
    if overlap:
        raise ValueError(f"User split overlap: {sorted(overlap)}")
    return {
        "passed": True,
        "n_train_users": len(train),
        "n_development_users": len(development),
        "n_evaluation_users": len(evaluation),
    }


def reference_exclusion_audit(
    *,
    reference_event_ids: np.ndarray,
    test_event_ids: np.ndarray,
    reference_session_ids: np.ndarray,
    test_session_ids: np.ndarray,
    reference_user_ids: np.ndarray,
    test_user_ids: np.ndarray,
    reference_raw_start: np.ndarray,
    reference_raw_end: np.ndarray,
    test_raw_start: np.ndarray,
    test_raw_end: np.ndarray,
    reference_signals: np.ndarray | None = None,
    test_signals: np.ndarray | None = None,
    near_duplicate_rms: float = 1e-6,
    derived_sample_matches: int | None = None,
) -> dict[str, int | bool | float | None]:
    ref_event = np.asarray(reference_event_ids).reshape(-1)
    tst_event = np.asarray(test_event_ids).reshape(-1)
    ref_session = np.asarray(reference_session_ids).reshape(-1)
    tst_session = np.asarray(test_session_ids).reshape(-1)
    ref_user = np.asarray(reference_user_ids).reshape(-1)
    tst_user = np.asarray(test_user_ids).reshape(-1)
    ref_start = np.asarray(reference_raw_start).reshape(-1)
    ref_end = np.asarray(reference_raw_end).reshape(-1)
    tst_start = np.asarray(test_raw_start).reshape(-1)
    tst_end = np.asarray(test_raw_end).reshape(-1)
    event_overlap = len(set(ref_event.tolist()) & set(tst_event.tolist()))
    session_overlap = 0
    raw_overlap = 0
    for user in np.intersect1d(np.unique(ref_user), np.unique(tst_user)):
        r = np.flatnonzero(ref_user == user)
        q = np.flatnonzero(tst_user == user)
        session_overlap += len(
            set(ref_session[r].tolist()) & set(tst_session[q].tolist())
        )
        for ri in r:
            raw_overlap += int(
                np.sum((tst_start[q] < ref_end[ri]) & (tst_end[q] > ref_start[ri]))
            )
    exact = 0
    near = 0
    if (reference_signals is None) != (test_signals is None):
        raise ValueError("Both reference_signals and test_signals are required together")
    if reference_signals is not None:
        ref_signal = np.asarray(reference_signals)
        tst_signal = np.asarray(test_signals)
        ref_hashes = {sha256_array(x) for x in ref_signal}
        exact = sum(sha256_array(x) in ref_hashes for x in tst_signal)
        for query in tst_signal:
            distances = np.sqrt(
                np.mean((ref_signal.astype(np.float64) - query.astype(np.float64)) ** 2,
                        axis=tuple(range(1, ref_signal.ndim)))
            )
            near += int(np.any(distances <= near_duplicate_rms))
    if derived_sample_matches is not None and derived_sample_matches < 0:
        raise ValueError("derived_sample_matches cannot be negative")
    derived_checked = derived_sample_matches is not None
    passed = bool(
        derived_checked
        and all(
            v == 0
            for v in (
                event_overlap,
                session_overlap,
                raw_overlap,
                exact,
                near,
                int(derived_sample_matches),
            )
        )
    )
    return {
        "passed": passed,
        "event_id_matches": int(event_overlap),
        "session_id_matches": int(session_overlap),
        "overlapping_raw_ranges": int(raw_overlap),
        "duplicate_hash_matches": int(exact),
        "near_duplicate_matches": int(near),
        "derived_check_performed": derived_checked,
        "derived_sample_matches": (
            int(derived_sample_matches) if derived_sample_matches is not None else None
        ),
        "near_duplicate_rms_threshold": float(near_duplicate_rms),
    }
