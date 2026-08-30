#!/usr/bin/env python3
"""Generate the canonical per-user Android IMU cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FIX_ROOT = Path(
    os.environ.get(
        "ACTREAL_OFFLINE_WORK_ROOT",
        REPOSITORY_ROOT / ".actreal-work" / "offline_generation",
    )
)
ANDROID_ROOT = REPOSITORY_ROOT / "methods" / "generation" / "imu"
BASE_PAD_ROOT = Path(
    os.environ.get(
        "ACTREAL_DETECTOR_SET_ROOT",
        REPOSITORY_ROOT / "licensed_input" / "event_level",
    )
)
PROCESSED_ROOT = Path(
    os.environ.get(
        "ACTREAL_GENERATOR_DATA_ROOT",
        REPOSITORY_ROOT / "data" / "generator_training" / "processed",
    )
)
if str(ANDROID_ROOT) not in sys.path:
    sys.path.insert(0, str(ANDROID_ROOT))

from android_imu_layer import AndroidIMUDiffusionLayer  # noqa: E402


ACTIONS = ("tap", "scroll", "swipe", "pinch")
POOLS = ("fake_train", "fake_val", "fake_test")
SPLIT_TO_POOL = {"train": "fake_train", "val": "fake_val", "test": "fake_test"}
CACHE_SCHEMA_REVISION = "time_schema_v2_division_first_refstable_r3"
EXPECTED_REF_BANK_SEED = 42 + 303


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def stable_job_seed(
    base_seed: int,
    split: str,
    user_id: int,
    action: str,
    sample_idx: int,
    target_active_len: Optional[int],
    bin_rep: int,
) -> int:
    payload = "%d|%s|%d|%s|%d|%s|%d" % (
        int(base_seed),
        split,
        int(user_id),
        action,
        int(sample_idx),
        "none" if target_active_len is None else str(int(target_active_len)),
        int(bin_rep),
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


def existing_sample_is_valid(
    path: Path,
    action: str,
    user_id: int,
    split: str,
    expected_job: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        with np.load(str(path), allow_pickle=False) as z:
            required = {"action", "hz", "window", "active_imu", "mask", "valid_mask", "metadata_json"}
            if not required.issubset(set(z.files)):
                return False
            meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
            hz = float(np.asarray(z["hz"]).item())
            window = np.asarray(z["window"])
            active = np.asarray(z["active_imu"])
            mask = np.asarray(z["mask"])
            valid = np.asarray(z["valid_mask"])
            if str(np.asarray(z["action"]).item()) != action:
                return False
            if str(meta.get("action", action)) != action:
                return False
            if int(meta.get("user_id", -1)) != int(user_id) or str(meta.get("split", "")) != split:
                return False
            if int(meta.get("time_schema_version", -1)) != 2:
                return False
            if str(meta.get("cache_schema_revision", "")) != CACHE_SCHEMA_REVISION:
                return False
            if expected_job is not None:
                for field, expected in expected_job.items():
                    actual = meta.get(field)
                    if isinstance(expected, int):
                        try:
                            actual = int(actual)
                        except Exception:
                            return False
                    if actual != expected:
                        return False
            if int(meta.get("ref_count", -1)) != 5:
                return False
            if int(meta.get("ref_bank_seed", -1)) != EXPECTED_REF_BANK_SEED:
                return False
            refs = np.asarray(meta.get("used_ref_indices", []), dtype=np.int64)
            if refs.shape != (5,) or np.any(refs < 0) or len(np.unique(refs)) != 5:
                return False
            if not math.isfinite(hz) or hz <= 0:
                return False
            if window.ndim != 2 or window.shape[1] != 6 or not np.all(np.isfinite(window)):
                return False
            if active.ndim != 2 or active.shape[1] != 6 or not np.all(np.isfinite(active)):
                return False
            if int(meta.get("active_len", -1)) != int(active.shape[0]):
                return False
            if (
                mask.ndim != 1
                or valid.ndim != 1
                or mask.shape != valid.shape
                or mask.shape[0] != window.shape[0]
                or np.any((mask != 0) & (mask != 1))
                or np.any((valid != 0) & (valid != 1))
                or np.any((mask > 0) & (valid == 0))
            ):
                return False
            selector = mask > 0
            if int(selector.sum()) != int(active.shape[0]):
                return False
            if not np.array_equal(window[selector], active):
                return False
            prior = meta.get("prior_audit") if isinstance(meta.get("prior_audit"), dict) else {}
            request = meta.get("request") if isinstance(meta.get("request"), dict) else {}
            if str(prior.get("prior_source", "")) != "train_users_real_enroll":
                return False
            for value in (
                meta.get("requested_active_len"),
                request.get("active_len"),
                prior.get("source_active_len"),
            ):
                if int(value) != int(active.shape[0]):
                    return False
            logical_ms = float(meta.get("logical_sample_duration_ms", float("nan")))
            if not math.isfinite(logical_ms) or logical_ms <= 0:
                return False
            for value in (
                meta.get("requested_duration_ms"),
                request.get("duration_ms"),
                prior.get("source_duration_ms"),
            ):
                if not math.isclose(float(value), logical_ms, rel_tol=0.0, abs_tol=1e-6):
                    return False
            return True
    except Exception:
        return False


def finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def split_users_from_base(base: np.lib.npyio.NpzFile, split: str) -> np.ndarray:
    key = {"train": "train_fake_users", "val": "val_fake_users", "test": "test_fake_users"}[split]
    return base[key].astype(np.int64)


def load_action_base(action: str) -> np.lib.npyio.NpzFile:
    return np.load(str(BASE_PAD_ROOT / "detector_sets" / ("%s_detector_sets.npz" % action)), allow_pickle=True)


def load_processed_action(action: str) -> np.lib.npyio.NpzFile:
    return np.load(str(PROCESSED_ROOT / ("hmog_%s.npz" % action)), allow_pickle=True)


def train_prior_indices(base: np.lib.npyio.NpzFile) -> np.ndarray:
    uid = base["real_enroll_user_id"].astype(np.int64)
    train_users = base["train_fake_users"].astype(np.int64)
    idx = np.flatnonzero(np.isin(uid, train_users)).astype(np.int64)
    if len(idx) == 0:
        raise RuntimeError("metadata prior has no train-user real enrollment rows")
    return idx


def observed_active_len_support(base: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, Dict[int, float]]:
    idx = train_prior_indices(base)
    active = base["real_enroll_active_len"].astype(np.int64)[idx]
    values, counts = np.unique(active, return_counts=True)
    probabilities = counts.astype(np.float64) / float(counts.sum())
    return values.astype(np.int64), {int(v): float(p) for v, p in zip(values, probabilities)}


def sample_realish_request(
    base: np.lib.npyio.NpzFile,
    action: str,
    user_id: int,
    rng: np.random.Generator,
    target_active_len: Optional[int] = None,
    processed: Optional[np.lib.npyio.NpzFile] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Match the formal generator protocol: metadata conditions come only from
    # train-user real enrollment rows.  The explicit target user is represented
    # exclusively by the five few-shot waveform refs inside the Android layer.
    # Using all metadata rows of a val/test target user would exceed five-shot
    # scope and leak test-user information.
    pool = "real_enroll"
    uid = base["%s_user_id" % pool].astype(np.int64)
    idx = train_prior_indices(base)
    if target_active_len is not None:
        source_active = base["%s_active_len" % pool].astype(np.int64)
        idx = idx[source_active[idx] == int(target_active_len)]
        if len(idx) == 0:
            raise RuntimeError("requested active_len is outside observed train support: %s" % target_active_len)
    i = int(rng.choice(idx))
    duration_ms = float(base["%s_duration_ms" % pool][i])
    source_active_len = int(base["%s_active_len" % pool][i])
    # Preserve both original logical milliseconds and the preprocessing frame
    # count.  The model may have a fixed maximum active window, but that must be
    # recorded as a partial event rather than silently rewriting logical time.
    request_duration_ms = duration_ms
    orientation_key = "%s_orientation_id" % pool
    orientation_id = int(base[orientation_key][i]) if orientation_key in base.files else -1
    req: Dict[str, Any] = {
        "duration_ms": request_duration_ms,
        "active_len": source_active_len,
        "orientation_id": orientation_id,
    }
    audit: Dict[str, Any] = {
        "prior_source": "train_users_real_enroll",
        "prior_source_index": i,
        "prior_source_user_id": int(uid[i]),
        "source_duration_ms": duration_ms,
        "source_active_len": source_active_len,
        "request_duration_ms": request_duration_ms,
        "target_active_len": None if target_active_len is None else int(target_active_len),
    }
    if "real_enroll_source_index" in base.files:
        source_processed_index = int(base["real_enroll_source_index"][i])
        audit["source_processed_index"] = source_processed_index
        if processed is not None and 0 <= source_processed_index < len(processed["duration_ms"]):
            audit["source_parent_event_duration_ms"] = float(
                processed["event_duration_ms"][source_processed_index]
                if "event_duration_ms" in processed.files
                else processed["duration_ms"][source_processed_index]
            )
            if "event_id" in processed.files:
                audit["source_event_id"] = int(processed["event_id"][source_processed_index])

    def getf(name: str) -> Optional[float]:
        key = "%s_%s" % (pool, name)
        if key not in base.files:
            return None
        return finite_or_none(base[key][i])

    if action == "tap":
        x = getf("xy_start_x")
        y = getf("xy_start_y")
        if x is None:
            x = getf("tap_x")
        if y is None:
            y = getf("tap_y")
        if x is None or y is None:
            x = float(rng.uniform(80, 1000))
            y = float(rng.uniform(160, 1760))
            audit["xy_fallback"] = True
        req.update({"x": float(x), "y": float(y)})
    elif action in ("scroll", "swipe"):
        x0 = getf("xy_start_x")
        y0 = getf("xy_start_y")
        x1 = getf("xy_end_x")
        y1 = getf("xy_end_y")
        if x0 is None:
            x0 = getf("%s_start_x" % action)
        if y0 is None:
            y0 = getf("%s_start_y" % action)
        if x1 is None:
            x1 = getf("%s_end_x" % action)
        if y1 is None:
            y1 = getf("%s_end_y" % action)
        if any(v is None for v in (x0, y0, x1, y1)):
            x0 = float(rng.uniform(80, 1000))
            y0 = float(rng.uniform(160, 1760))
            dist = float(rng.uniform(30, 700))
            if action == "scroll":
                x1 = float(np.clip(x0 + rng.normal(0, 20), 40, 1040))
                y1 = float(np.clip(y0 + rng.choice([-1, 1]) * dist, 80, 1840))
            else:
                th = float(rng.uniform(0, 2 * math.pi))
                x1 = float(np.clip(x0 + dist * math.cos(th), 40, 1040))
                y1 = float(np.clip(y0 + dist * math.sin(th), 80, 1840))
            audit["xy_fallback"] = True
        req.update({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)})
    elif action == "pinch":
        x0 = getf("xy_start_x")
        y0 = getf("xy_start_y")
        x1 = getf("xy_end_x")
        y1 = getf("xy_end_y")
        if x0 is None:
            x0 = getf("pinch_start_x")
        if y0 is None:
            y0 = getf("pinch_start_y")
        if x1 is None:
            x1 = getf("pinch_end_x")
        if y1 is None:
            y1 = getf("pinch_end_y")
        s0 = getf("pinch_start_span")
        s1 = getf("pinch_end_span")
        if any(v is None for v in (x0, y0, x1, y1)):
            x0 = float(rng.uniform(80, 1000))
            y0 = float(rng.uniform(160, 1760))
            th = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(20, 800))
            x1 = float(np.clip(x0 + dist * math.cos(th), 0, 1800))
            y1 = float(np.clip(y0 + dist * math.sin(th), 0, 2000))
            audit["xy_fallback"] = True
        if s0 is None:
            s0 = float(rng.uniform(160, 620))
        if s1 is None:
            s1 = float(np.clip(s0 + rng.normal(0, 350), 40, 1200))
        req.update({
            "center_start": (float(x0), float(y0)),
            "center_end": (float(x1), float(y1)),
            "start_span": float(s0),
            "end_span": float(s1),
        })
    else:
        raise ValueError(action)
    return req, audit


def call_layer(
    layer: AndroidIMUDiffusionLayer,
    action: str,
    user_id: int,
    req: Dict[str, Any],
    sample_steps: Optional[int],
    noise_seed: Optional[int] = None,
) -> Dict[str, Any]:
    common = {
        "user_id": int(user_id),
        "sample_steps": sample_steps,
        "orientation_id": int(req["orientation_id"]),
        "active_len": int(req["active_len"]),
        "noise_seed": noise_seed,
    }
    if action == "tap":
        return layer.tap(req["x"], req["y"], duration_ms=req["duration_ms"], **common)
    if action == "scroll":
        return layer.scroll(req["x0"], req["y0"], req["x1"], req["y1"], duration_ms=req["duration_ms"], **common)
    if action == "swipe":
        return layer.swipe(req["x0"], req["y0"], req["x1"], req["y1"], duration_ms=req["duration_ms"], **common)
    if action == "pinch":
        return layer.pinch(
            req["center_start"],
            req["start_span"],
            req["end_span"],
            end_center=req["center_end"],
            duration_ms=req["duration_ms"],
            **common,
        )
    raise ValueError(action)


def save_sample(path: Path, out: Dict[str, Any], meta_extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(out["metadata"])
    meta.update(meta_extra)
    tmp = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                action=np.asarray(out["action"]),
                hz=np.asarray(float(out["hz"]), dtype=np.float32),
                window=out["window"].astype(np.float32),
                active_imu=out["active_imu"].astype(np.float32),
                mask=out["mask"].astype(np.uint8),
                valid_mask=out["valid_mask"].astype(np.uint8),
                metadata_json=np.asarray(json.dumps(meta, default=json_default, ensure_ascii=False)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def quarantine_invalid_sample(path: Path, cache_root: Path, run_tag: str) -> Path:
    """Move an invalid resume artifact outside the protocol cache tree."""
    path = Path(path)
    cache_root = Path(cache_root)
    relative = path.relative_to(cache_root)
    destination = (
        FIX_ROOT
        / "quarantine"
        / str(run_tag)
        / relative.parent
        / ("%s.corrupt.%d" % (path.name, int(time.time_ns())))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(path), str(destination))
    return destination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=FIX_ROOT / "user_cache_eval_200")
    ap.add_argument("--actions", nargs="+", choices=ACTIONS, default=list(ACTIONS))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--samples-per-user-action", type=int, default=200)
    ap.add_argument(
        "--coverage-mode",
        choices=("random", "active_len_stratified"),
        default="random",
        help="random=distributional evaluation cache; active_len_stratified=two (configurable) records for every observed train active_len",
    )
    ap.add_argument("--samples-per-active-len", type=int, default=2)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-users-per-split", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    layer = AndroidIMUDiffusionLayer(seed=args.seed + args.shard_index * 100003, device="cuda")
    bases = {a: load_action_base(a) for a in args.actions}
    processed = {a: load_processed_action(a) for a in args.actions}
    support = {a: observed_active_len_support(bases[a]) for a in args.actions}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.out_dir.name.replace("/", "_")
    manifest_path = FIX_ROOT / "results" / (
        "generation_manifest_%s_shard%d.jsonl" % (run_tag, int(args.shard_index))
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    users_by_split = {}
    first_base = next(iter(bases.values()))
    for split in args.splits:
        users = split_users_from_base(first_base, split)
        if args.max_users_per_split is not None:
            users = users[: int(args.max_users_per_split)]
        users = users[np.arange(len(users)) % int(args.num_shards) == int(args.shard_index)]
        users_by_split[split] = users.astype(np.int64)

    started = time.time()
    n_done = 0
    n_skip = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for split, users in users_by_split.items():
            for user_id in users:
                for action in args.actions:
                    base = bases[action]
                    if args.coverage_mode == "active_len_stratified":
                        values, probabilities = support[action]
                        jobs = [
                            (int(active_len), int(rep), int(seq))
                            for seq, (active_len, rep) in enumerate(
                                (item for value in values for item in ((value, r) for r in range(int(args.samples_per_active_len))))
                            )
                        ]
                    else:
                        probabilities = {}
                        jobs = [(None, -1, int(i)) for i in range(int(args.samples_per_user_action))]
                    for target_active_len, bin_rep, sample_idx in jobs:
                        if target_active_len is None:
                            filename = "sample_%04d.npz" % sample_idx
                        else:
                            filename = "len_%03d_rep_%02d.npz" % (int(target_active_len), int(bin_rep))
                        out_path = args.out_dir / ("user_%03d" % int(user_id)) / action / split / filename
                        job_seed = stable_job_seed(
                            args.seed,
                            split,
                            int(user_id),
                            action,
                            sample_idx,
                            target_active_len,
                            bin_rep,
                        )
                        expected_job = {
                            "job_seed": int(job_seed),
                            "noise_seed": int(job_seed + 4),
                            "coverage_mode": args.coverage_mode,
                            "sample_idx": int(sample_idx),
                            "duration_bin_id": -1 if target_active_len is None else int(target_active_len),
                            "bin_rep": int(bin_rep),
                            "cache_protocol": (
                                "time_schema_v2_active_len_stratified_train_prior"
                                if args.coverage_mode == "active_len_stratified"
                                else "time_schema_v2_full_200_per_user_action_train_prior"
                            ),
                        }
                        if out_path.exists() and not args.overwrite:
                            if existing_sample_is_valid(
                                out_path, action, int(user_id), split, expected_job=expected_job
                            ):
                                n_skip += 1
                                continue
                            quarantine_invalid_sample(out_path, args.out_dir, run_tag)
                        job_rng = np.random.default_rng(job_seed)
                        req, prior_audit = sample_realish_request(
                            base,
                            action,
                            int(user_id),
                            job_rng,
                            target_active_len=target_active_len,
                            processed=processed[action],
                        )
                        layer.rng = np.random.default_rng(job_seed + 1)
                        np.random.seed(job_seed + 2)
                        random.seed(job_seed + 3)
                        t0 = time.perf_counter()
                        out = call_layer(
                            layer,
                            action,
                            int(user_id),
                            req,
                            args.sample_steps,
                            noise_seed=job_seed + 4,
                        )
                        ref_count = int(out["metadata"].get("ref_count", 0))
                        used_refs = np.asarray(out["metadata"].get("used_ref_indices", []), dtype=np.int64)
                        if ref_count != 5 or used_refs.shape != (5,) or np.any(used_refs < 0):
                            raise RuntimeError(
                                "few-shot invariant failed: split=%s user=%d action=%s ref_count=%d used_refs=%s"
                                % (split, int(user_id), action, ref_count, used_refs.tolist())
                            )
                        returned_orientation = int(out["metadata"].get("orientation_id", -999))
                        if returned_orientation != int(req["orientation_id"]):
                            raise RuntimeError(
                                "orientation invariant failed: split=%s user=%d action=%s requested=%d returned=%d"
                                % (split, int(user_id), action, int(req["orientation_id"]), returned_orientation)
                            )
                        call_wall_ms = float((time.perf_counter() - t0) * 1000.0)
                        meta_extra = {
                            "experiment": "android_duration_time_fixed_20260720_%s" % run_tag,
                            "cache_protocol": (
                                "time_schema_v2_active_len_stratified_train_prior"
                                if args.coverage_mode == "active_len_stratified"
                                else "time_schema_v2_full_200_per_user_action_train_prior"
                            ),
                            "time_schema_version": 2,
                            "cache_schema_revision": CACHE_SCHEMA_REVISION,
                            "split": split,
                            "user_id": int(user_id),
                            "action": action,
                            "sample_idx": int(sample_idx),
                            "job_seed": int(job_seed),
                            "coverage_mode": args.coverage_mode,
                            "duration_bin_id": -1 if target_active_len is None else int(target_active_len),
                            "bin_rep": int(bin_rep),
                            "sampling_weight": (
                                None
                                if target_active_len is None
                                else float(probabilities[int(target_active_len)] / float(args.samples_per_active_len))
                            ),
                            "request": req,
                            "prior_audit": prior_audit,
                            "requested_duration_ms": float(req.get("duration_ms", float("nan"))),
                            "requested_active_len": int(req.get("active_len", -1)),
                            "android_call_wall_ms": call_wall_ms,
                            "sample_path": str(out_path),
                        }
                        save_sample(out_path, out, meta_extra)
                        row = {
                            "path": str(out_path),
                            "split": split,
                            "user_id": int(user_id),
                            "action": action,
                            "sample_idx": int(sample_idx),
                            "job_seed": int(job_seed),
                            "coverage_mode": args.coverage_mode,
                            "duration_bin_id": -1 if target_active_len is None else int(target_active_len),
                            "bin_rep": int(bin_rep),
                            "sampling_weight": (
                                None
                                if target_active_len is None
                                else float(probabilities[int(target_active_len)] / float(args.samples_per_active_len))
                            ),
                            "event_duration_ms": float(out["metadata"].get("event_duration_ms", float("nan"))),
                            "logical_event_duration_ms": float(out["metadata"].get("logical_event_duration_ms", float("nan"))),
                            "buffer_duration_ms": float(out["metadata"].get("buffer_duration_ms", float("nan"))),
                            "duration_quantization_error_ms": float(out["metadata"].get("duration_quantization_error_ms", float("nan"))),
                            "emitted_duration_delta_ms": float(out["metadata"].get("emitted_duration_delta_ms", float("nan"))),
                            "clipped_buffer_duration_ms": float(out["metadata"].get("clipped_buffer_duration_ms", float("nan"))),
                            "is_partial_event": bool(out["metadata"].get("is_partial_event", False)),
                            "requested_duration_ms": float(req.get("duration_ms", float("nan"))),
                            "android_call_wall_ms": call_wall_ms,
                            "generation_wall_ms": float(out["metadata"].get("generation_wall_ms", float("nan"))),
                            "active_len": int(out["metadata"].get("active_len", -1)),
                            "sample_steps": out["metadata"].get("sample_steps"),
                        }
                        mf.write(json.dumps(row, default=json_default, ensure_ascii=False) + "\n")
                        mf.flush()
                        n_done += 1
                        if n_done % 100 == 0:
                            print(json.dumps({"done": n_done, "skipped": n_skip, "elapsed_s": time.time() - started}, ensure_ascii=False), flush=True)
    print(json.dumps({"finished": True, "generated": n_done, "skipped": n_skip, "users_by_split": {k: v.astype(int).tolist() for k, v in users_by_split.items()}, "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
