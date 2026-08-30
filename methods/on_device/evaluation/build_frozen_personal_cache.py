#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GENERATION_ROOT = REPOSITORY_ROOT / "methods" / "generation" / "imu"
GENERATION_SCRIPTS = GENERATION_ROOT / "scripts"
ACTIONS = ("tap", "scroll", "swipe", "pinch")
SCHEMA = "phone_personal_frozen_cache_v1"

sys.path.insert(0, str(GENERATION_ROOT))
sys.path.insert(0, str(GENERATION_SCRIPTS))
from android_imu_layer import AndroidIMUDiffusionLayer  # noqa: E402
import generate_user_cache as CACHE_GEN  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def default_rows(array: np.ndarray, count: int) -> np.ndarray:
    shape = (count,) + tuple(array.shape[1:])
    kind = array.dtype.kind
    if kind in ("f", "c"):
        return np.full(shape, np.nan, dtype=array.dtype)
    if kind in ("i", "u"):
        return np.full(shape, -1 if kind == "i" else 0, dtype=array.dtype)
    if kind == "b":
        return np.zeros(shape, dtype=array.dtype)
    if kind in ("U", "S"):
        return np.full(shape, "", dtype=array.dtype)
    if kind == "O":
        result = np.empty(shape, dtype=array.dtype)
        result.fill("")
        return result
    return np.zeros(shape, dtype=array.dtype)


def validate_ref_bundle(refs_root: Path) -> Dict[str, Any]:
    manifest_path = refs_root / "refs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phone_five_shot_refs_v1":
        raise ValueError("unexpected five-shot reference schema")
    if not manifest.get("selected_run_id") or not manifest.get("profile_id"):
        raise ValueError("five-shot manifest has no run/profile identity")
    expected_posture = str(manifest.get("posture", ""))
    expected_run = str(manifest["selected_run_id"])
    expected_profile = str(manifest["profile_id"])
    for action in ACTIONS:
        action_audit = manifest.get("actions", {}).get(action, {})
        if int(action_audit.get("count", -1)) != 5:
            raise ValueError("%s does not contain exactly five references" % action)
        path = refs_root / ("refs_%s.npz" % action)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != action_audit.get("refs_sha256"):
            raise ValueError("%s hash differs from refs_manifest.json" % path.name)
        with np.load(path, allow_pickle=False) as refs:
            if str(np.asarray(refs["schema"]).item()) != "phone_five_shot_refs_v1":
                raise ValueError("unexpected schema in %s" % path.name)
            if str(np.asarray(refs["action"]).item()) != action:
                raise ValueError("action mismatch in %s" % path.name)
            if str(np.asarray(refs["run_id"]).item()) != expected_run:
                raise ValueError("run_id mismatch in %s" % path.name)
            if str(np.asarray(refs["profile_id"]).item()) != expected_profile:
                raise ValueError("profile_id mismatch in %s" % path.name)
            if str(np.asarray(refs["posture"]).item()) != expected_posture:
                raise ValueError("posture mismatch in %s" % path.name)
            if np.asarray(refs["windows"]).shape[0] != 5:
                raise ValueError("reference count mismatch in %s" % path.name)
    return manifest


def inject_phone_refs(
    layer: AndroidIMUDiffusionLayer,
    action: str,
    refs_path: Path,
    user_id: int,
) -> Dict[str, Any]:
    runtime, _ = layer._get_runtime(action)  # Intentional audited adapter.
    if int(runtime.k_refs) != 5:
        raise ValueError("%s checkpoint is not configured for k_refs=5" % action)
    with np.load(refs_path, allow_pickle=False) as refs:
        if str(np.asarray(refs["action"]).item()) != action:
            raise ValueError("reference action mismatch in %s" % refs_path)
        windows = np.asarray(refs["windows"], dtype=np.float32)
        mask = np.asarray(refs["mask"], dtype=np.uint8)
        valid_mask = np.asarray(refs["valid_mask"], dtype=np.uint8)
        active_len = np.asarray(refs["active_len"], dtype=np.int64)
        duration_ms = np.asarray(refs["duration_ms"], dtype=np.float32)
        orientation_id = np.asarray(refs["orientation_id"], dtype=np.int64)
        xy = np.asarray(refs["xy"], dtype=np.float32)
        n_keys = np.asarray(refs["n_keys"], dtype=np.int64)
        n_letters = np.asarray(refs["n_letters"], dtype=np.int64)
        posture = str(np.asarray(refs["posture"]).item())
        run_id = str(np.asarray(refs["run_id"]).item())
    if windows.shape != (5, runtime.data.T, 6):
        raise ValueError(
            "%s references have shape %s, expected (5,%d,6)"
            % (action, windows.shape, runtime.data.T)
        )
    if mask.shape != (5, runtime.data.T) or valid_mask.shape != mask.shape:
        raise ValueError("reference masks do not match windows for %s" % action)
    if not np.all(np.isfinite(windows)):
        raise ValueError("%s references contain non-finite IMU" % action)

    arrays = runtime.data.arrays
    original_n = runtime.data.n
    appended: Dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        if array.ndim == 0 or array.shape[0] != original_n:
            continue
        rows = default_rows(array, 5)
        if key == "windows":
            rows = windows.astype(array.dtype)
        elif key == "mask":
            rows = mask.astype(array.dtype)
        elif key == "valid_mask":
            rows = valid_mask.astype(array.dtype)
        elif key == "active_len":
            rows = active_len.astype(array.dtype)
        elif key == "user_id":
            rows = np.full(5, int(user_id), dtype=array.dtype)
        elif key == "orientation_id":
            rows = orientation_id.astype(array.dtype)
        elif key == "duration_ms":
            rows = duration_ms.astype(array.dtype)
        elif key == "n_keys":
            rows = n_keys.astype(array.dtype)
        elif key in ("n_letters", "event_n_letters"):
            rows = n_letters.astype(array.dtype)
        elif key in ("xy_start_x", "tap_x"):
            rows = xy[:, 0].astype(array.dtype)
        elif key in ("xy_start_y", "tap_y"):
            rows = xy[:, 1].astype(array.dtype)
        elif key == "xy_end_x":
            rows = xy[:, 2].astype(array.dtype)
        elif key == "xy_end_y":
            rows = xy[:, 3].astype(array.dtype)
        appended[key] = np.concatenate([array, rows], axis=0)
    arrays.update(appended)
    indices = np.arange(original_n, original_n + 5, dtype=np.int64)
    runtime.ref_bank = layer._fg["UserRefBank"](
        runtime.data,
        np.asarray([int(user_id)], dtype=np.int64),
        5,
        int(runtime.ref_seed),
        candidate_indices=indices,
    )
    selected = runtime.ref_bank.refs(user_id)
    if not np.array_equal(selected, indices):
        raise RuntimeError(
            "phone reference bank mismatch: %s != %s"
            % (selected.tolist(), indices.tolist())
        )
    return {
        "refs_path": str(refs_path),
        "refs_sha256": sha256_file(refs_path),
        "appended_indices": indices.tolist(),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": sha256_file(Path(runtime.checkpoint_path)),
        "T": int(runtime.data.T),
        "pad_pre_pts": int(runtime.data.pad_pre_pts),
        "ref_bank_seed": int(runtime.ref_seed),
        "posture": posture,
        "run_id": run_id,
    }


def cache_file_valid(
    path: Path,
    action: str,
    user_id: int,
    expected_job: Mapping[str, Any],
    expected_refs: Sequence[int],
) -> bool:
    try:
        if not CACHE_GEN.existing_sample_is_valid(
            path,
            action,
            user_id,
            "personal",
            expected_job=dict(expected_job),
        ):
            return False
        with np.load(path, allow_pickle=False) as store:
            metadata = json.loads(str(np.asarray(store["metadata_json"]).item()))
            return (
                str(np.asarray(store["action"]).item()) == action
                and int(metadata["user_id"]) == int(user_id)
                and metadata.get("phone_ref_source") == "five_shot_app_export"
                and int(metadata.get("ref_count", -1)) == 5
                and np.array_equal(
                    np.sort(
                        np.asarray(
                            metadata.get("used_ref_indices", []), dtype=np.int64
                        )
                    ),
                    np.sort(np.asarray(expected_refs, dtype=np.int64)),
                )
                and np.asarray(store["window"]).ndim == 2
                and np.asarray(store["active_imu"]).shape[1] == 6
            )
    except Exception:
        return False


def freeze_cache(
    out: Path,
    user_id: int,
    ref_manifest: Path,
    expected: Mapping[str, int],
    generation_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    index_path = out / "cache_files.jsonl"
    rows = []
    for path in sorted((out / ("user_%03d" % user_id)).rglob("*.npz")):
        rows.append(
            {
                "relative_path": str(path.relative_to(out)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    actual_by_action = {
        action: sum(
            1
            for row in rows
            if action in Path(row["relative_path"]).parts
        )
        for action in ACTIONS
    }
    if dict(actual_by_action) != dict(expected):
        raise RuntimeError(
            "cache count mismatch expected=%s actual=%s" % (expected, actual_by_action)
        )
    temporary = index_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(index_path)
    frozen = {
        "schema": SCHEMA,
        "frozen": True,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mutation_policy": "No generation, overwrite, or cache expansion is allowed after this file exists.",
        "user_id": int(user_id),
        "expected_files_by_action": dict(expected),
        "total_files": len(rows),
        "total_size_bytes": int(sum(row["size_bytes"] for row in rows)),
        "file_index": str(index_path),
        "file_index_sha256": sha256_file(index_path),
        "refs_manifest": str(ref_manifest),
        "refs_manifest_sha256": sha256_file(ref_manifest),
        "generation_manifest": str(out / "generation_manifest.jsonl"),
        "generation_manifest_sha256": sha256_file(
            out / "generation_manifest.jsonl"
        ),
        "generation_audit": generation_audit,
    }
    path = out / "FROZEN.json"
    temporary_json = path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(path)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a complete active-length cache from five phone refs, then freeze it."
    )
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user-id", type=int, default=1000)
    parser.add_argument("--samples-per-active-len", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Optional immutable label-correction/provenance JSON recorded in FROZEN.json.",
    )
    args = parser.parse_args()
    if (args.out / "FROZEN.json").exists():
        raise RuntimeError("cache is already frozen and cannot be modified: %s" % args.out)
    ref_manifest = args.refs / "refs_manifest.json"
    if not ref_manifest.is_file():
        raise FileNotFoundError(ref_manifest)
    validate_ref_bundle(args.refs)
    if args.samples_per_active_len <= 0:
        raise ValueError("samples-per-active-len must be positive")
    provenance = None
    if args.provenance is not None:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        if provenance.get("raw_data_mutated") is not False:
            raise ValueError("provenance must explicitly state raw_data_mutated=false")
    args.out.mkdir(parents=True, exist_ok=True)

    layer = AndroidIMUDiffusionLayer(seed=args.seed, device=args.device)
    ref_audit: Dict[str, Any] = {}
    expected: Dict[str, int] = {action: 0 for action in ACTIONS}
    generated = 0
    skipped = 0
    manifest_path = args.out / "generation_manifest.jsonl"
    # Rebuild this derived manifest on every invocation. The NPZ files are the
    # resumable source of truth; appending across retries would duplicate rows.
    manifest_path.write_text("", encoding="utf-8")
    for action in ACTIONS:
        refs_path = args.refs / ("refs_%s.npz" % action)
        ref_audit[action] = inject_phone_refs(layer, action, refs_path, args.user_id)
        base = CACHE_GEN.load_action_base(action)
        processed = CACHE_GEN.load_processed_action(action)
        support, probabilities = CACHE_GEN.observed_active_len_support(base)
        expected[action] = int(len(support) * args.samples_per_active_len)
        for sequence, active_len in enumerate(support.astype(int).tolist()):
            for rep in range(args.samples_per_active_len):
                sample_index = sequence * args.samples_per_active_len + rep
                path = (
                    args.out
                    / ("user_%03d" % args.user_id)
                    / action
                    / "personal"
                    / ("len_%03d_rep_%02d.npz" % (active_len, rep))
                )
                job_seed = CACHE_GEN.stable_job_seed(
                    args.seed,
                    "personal",
                    args.user_id,
                    action,
                    sample_index,
                    active_len,
                    rep,
                )
                expected_refs = ref_audit[action]["appended_indices"]
                expected_job = {
                    "job_seed": int(job_seed),
                    "sample_idx": int(sample_index),
                    "duration_bin_id": int(active_len),
                    "bin_rep": int(rep),
                    "phone_ref_source": "five_shot_app_export",
                    "phone_refs_sha256": ref_audit[action]["refs_sha256"],
                }
                if path.exists():
                    if cache_file_valid(
                        path,
                        action,
                        args.user_id,
                        expected_job,
                        expected_refs,
                    ):
                        skipped += 1
                        with manifest_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "path": str(path),
                                        "action": action,
                                        "active_len": int(active_len),
                                        "rep": int(rep),
                                        "job_seed": int(job_seed),
                                        "resumed_existing": True,
                                        "sha256": sha256_file(path),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        continue
                    raise RuntimeError(
                        "invalid partial cache file found; move it aside before resume: %s"
                        % path
                    )
                rng = np.random.default_rng(job_seed)
                request, prior_audit = CACHE_GEN.sample_realish_request(
                    base,
                    action,
                    args.user_id,
                    rng,
                    target_active_len=active_len,
                    processed=processed,
                )
                layer.rng = np.random.default_rng(job_seed + 1)
                np.random.seed(job_seed + 2)
                random.seed(job_seed + 3)
                started = time.perf_counter()
                output = CACHE_GEN.call_layer(
                    layer,
                    action,
                    args.user_id,
                    request,
                    args.sample_steps,
                    noise_seed=job_seed + 4,
                )
                used = np.asarray(
                    output["metadata"].get("used_ref_indices", []), dtype=np.int64
                )
                expected_refs = np.asarray(expected_refs, dtype=np.int64)
                if int(output["metadata"].get("ref_count", -1)) != 5 or not np.array_equal(
                    np.sort(used), expected_refs
                ):
                    raise RuntimeError(
                        "%s generation did not use exactly the five phone refs: %s"
                        % (action, used.tolist())
                    )
                generation_ms = (time.perf_counter() - started) * 1000.0
                metadata = {
                    "experiment": "phone_personal_cache_20260724",
                    "cache_protocol": "active_len_stratified_train_condition_prior_phone_five_shot",
                    "time_schema_version": 2,
                    "cache_schema_revision": CACHE_GEN.CACHE_SCHEMA_REVISION,
                    "split": "personal",
                    "user_id": int(args.user_id),
                    "action": action,
                    "sample_idx": int(sample_index),
                    "job_seed": int(job_seed),
                    "coverage_mode": "active_len_stratified",
                    "duration_bin_id": int(active_len),
                    "bin_rep": int(rep),
                    "sampling_weight": float(
                        probabilities[int(active_len)] / args.samples_per_active_len
                    ),
                    "request": request,
                    "prior_audit": prior_audit,
                    "requested_duration_ms": float(request["duration_ms"]),
                    "requested_active_len": int(request["active_len"]),
                    "android_call_wall_ms": float(generation_ms),
                    "phone_ref_source": "five_shot_app_export",
                    "phone_refs_path": str(refs_path),
                    "phone_refs_sha256": ref_audit[action]["refs_sha256"],
                    "sample_path": str(path),
                }
                CACHE_GEN.save_sample(path, output, metadata)
                row = {
                    "path": str(path),
                    "action": action,
                    "active_len": int(active_len),
                    "rep": int(rep),
                    "job_seed": int(job_seed),
                    "generation_wall_ms": float(generation_ms),
                    "sha256": sha256_file(path),
                }
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                generated += 1
                print(
                    json.dumps(
                        {
                            "generated": generated,
                            "skipped": skipped,
                            "action": action,
                            "active_len": active_len,
                            "rep": rep,
                        }
                    ),
                    flush=True,
                )
        base.close()
        processed.close()
        # Only one action model is needed at a time. Releasing it prevents all
        # five diffusion checkpoints from accumulating on an 8 GB phone-test GPU.
        runtime = layer._runtime.pop(action, None)
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generation_audit = {
        "seed": args.seed,
        "device": args.device,
        "sample_steps_override": args.sample_steps,
        "samples_per_active_len": args.samples_per_active_len,
        "generated_this_run": generated,
        "resumed_valid_files": skipped,
        "phone_refs": ref_audit,
    }
    if args.provenance is not None:
        generation_audit["provenance"] = provenance
        generation_audit["provenance_path"] = str(args.provenance)
        generation_audit["provenance_sha256"] = sha256_file(args.provenance)
    frozen = freeze_cache(
        args.out, args.user_id, ref_manifest, expected, generation_audit
    )
    print(json.dumps(frozen, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
