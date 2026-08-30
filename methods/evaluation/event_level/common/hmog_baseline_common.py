"""Shared plumbing for running third-party generators on the direct100k grid."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
# Frozen detector windows: train-user genuine p95 rounded up to a multiple of 4.
ACTION_WINDOW_SAMPLES = {
    "tap": 16,
    "pinch": 100,
    "swipe": 176,
    "scroll": 208,
    "keystroke": 512,
}
TRAJECTORY_COLUMNS = 9
IMU_CHANNELS = 6


def zoh_resample(values: np.ndarray, target: int) -> np.ndarray:
    """Zero-order-hold resample, the observer the genuine path itself uses."""

    values = np.asarray(values)
    if len(values) == target:
        return values.copy()
    source_positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target, dtype=np.float64)
    held = np.clip(
        np.floor(target_positions * (len(values) - 1) + 0.5).astype(np.int64),
        0,
        len(values) - 1,
    )
    out = values[held].copy()
    if values.ndim == 2 and values.shape[1] == TRAJECTORY_COLUMNS:
        out[:, 7] = np.interp(
            target_positions, source_positions, values[:, 7].astype(np.float64)
        ).astype(values.dtype)
        # Holding columns 5 and 6 would carry the source's per-step deltas onto
        # a different grid, where they no longer describe the stored x/y.  The
        # builder keeps that invariant, so this has to as well.
        recompute_deltas(out)
    return out


def linear_resample(values: np.ndarray, target: int) -> np.ndarray:
    """Resample a continuously sampled physical signal, e.g. an IMU window."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) == target:
        return values.astype(np.float32)
    source_positions = np.linspace(0.0, 1.0, len(values))
    target_positions = np.linspace(0.0, 1.0, target)
    out = np.empty((target, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        out[:, channel] = np.interp(
            target_positions, source_positions, values[:, channel]
        )
    return out


def recompute_deltas(trajectory: np.ndarray) -> None:
    """Rewrite dx/dy in place so they describe the trajectory actually stored."""

    contact = trajectory[:, 0] > 0.0
    deltas = np.zeros((len(trajectory), 2), dtype=np.float32)
    consecutive = contact[1:] & contact[:-1]
    differences = np.diff(trajectory[:, 1:3], axis=0)
    deltas[1:][consecutive] = differences[consecutive]
    trajectory[:, 5:7] = deltas


def apply_carrier_dispatch(
    generated_xy: np.ndarray, carrier_trajectory: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Give a generated path the carrier's update timing and no-touch rows."""

    carrier_xy = np.asarray(carrier_trajectory[:, 1:3], dtype=np.float32)
    placed = np.array(generated_xy, dtype=np.float32, copy=True)
    if len(placed) != len(carrier_xy):
        raise ValueError("generated path and carrier disagree on length")

    # A row the device did not update repeats the previous one.  Forward-fill
    # the generated path through exactly the rows the carrier held.
    held = np.zeros(len(placed), dtype=bool)
    if len(placed) > 1:
        held[1:] = np.all(np.diff(carrier_xy, axis=0) == 0.0, axis=1)
    sources = np.where(~held, np.arange(len(placed)), 0)
    np.maximum.accumulate(sources, out=sources)
    placed = placed[sources]

    # No contact means no coordinate: the dataset writes an exact (0, 0) there.
    off_contact = np.asarray(carrier_trajectory[:, 0], dtype=np.float32) <= 0.0
    placed[off_contact] = carrier_xy[off_contact]
    return placed, int(held.sum()), int(off_contact.sum())


def fit_to_screen(
    path_px: np.ndarray,
    anchor_px: np.ndarray,
    dimensions: np.ndarray,
    considered: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Shrink a path about its anchor until it fits, instead of clipping it."""

    anchor = np.asarray(anchor_px, dtype=np.float64).reshape(1, 2)
    residual = np.asarray(path_px, dtype=np.float64) - anchor
    constrained = residual if considered is None else residual[considered]
    scale = 1.0
    if len(constrained):
        for axis in range(2):
            delta = constrained[:, axis]
            headroom = float(dimensions[axis]) - anchor[0, axis]
            positive, negative = delta > 0.0, delta < 0.0
            if np.any(positive):
                scale = min(scale, float(np.min(headroom / delta[positive])))
            if np.any(negative):
                scale = min(
                    scale, float(np.min(-anchor[0, axis] / delta[negative]))
                )
    scale = max(0.0, min(1.0, scale))
    if scale < 1.0:
        scale = float(np.nextafter(scale, 0.0))
    return anchor + scale * residual, scale


def iter_shards(dataset_dir: Path):
    for path in sorted((dataset_dir / "shards").glob("*.npz")):
        yield path


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def save_shard(path: Path, arrays: dict) -> str:
    np.savez(path, **arrays)
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_genuine_windows(
    dataset_dir: Path, split: str, action: str, kind: str
) -> tuple[np.ndarray, list[str]]:
    """Return every genuine window of one action, on the action's grid."""

    window = ACTION_WINDOW_SAMPLES[action]
    windows: list[np.ndarray] = []
    users: list[str] = []
    for path in iter_shards(dataset_dir):
        arrays = load_shard(path)
        if str(arrays["split"]) != split:
            continue
        offsets = arrays["offsets"]
        selected = np.flatnonzero(
            (arrays["action"] == action) & (arrays["label"] == 0)
        )
        flat = arrays["trajectory_flat" if kind == "trajectory" else "imu_flat"]
        for index in selected:
            start, stop = int(offsets[index]), int(offsets[index + 1])
            segment = flat[start:stop]
            if len(segment) < 2:
                continue
            if kind == "trajectory":
                windows.append(zoh_resample(segment, window).astype(np.float32))
            else:
                windows.append(linear_resample(segment, window))
            users.append(str(arrays["user_id"][index]))
    return np.stack(windows), users


def rewrite_shard_arrays(
    arrays: dict, trajectory_generator=None, imu_generator=None
) -> dict:
    """Swap generated channels on this shard's fake events, in place."""

    counts = {
        "fake": 0,
        "genuine": 0,
        "trajectory_swapped": 0,
        "trajectory_declined": 0,
        "imu_swapped": 0,
        "imu_declined": 0,
        "trajectory_clipped_samples": 0,
        "trajectory_rows_held_by_carrier": 0,
        "trajectory_rows_off_contact": 0,
        "trajectory_rows_total": 0,
    }
    # Per-action bookkeeping, because a declined action leaves the carrier's own
    # fake signal in place and must never be summarised as a baseline result.
    per_action: dict[str, dict[str, int]] = {}
    offsets = arrays["offsets"]
    labels = arrays["label"]
    actions = arrays["action"]
    trajectory = arrays["trajectory_flat"].copy()
    imu = arrays["imu_flat"].copy()
    for index in range(len(labels)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if int(labels[index]) == 0:
            counts["genuine"] += 1
            continue
        counts["fake"] += 1
        action = str(actions[index])
        tally = per_action.setdefault(
            action,
            {
                "fake": 0,
                "trajectory_swapped": 0,
                "trajectory_declined": 0,
                "imu_swapped": 0,
                "imu_declined": 0,
            },
        )
        tally["fake"] += 1
        # The generator sees a copy: the arrays underneath are about to be
        # overwritten, and a view would let a later write change what an
        # earlier one read.
        event = {
            "action": action,
            "user_id": str(arrays["user_id"][index]),
            "event_id": str(arrays["event_id"][index]),
            "split": str(arrays["split"]),
            "samples": stop - start,
            "trajectory": trajectory[start:stop].copy(),
            "imu": imu[start:stop].copy(),
        }
        # The ordinal of the generator-cache sample this event was originally
        # given.  It is what lets an ablation look up the same slot in a
        # different cache and so reproduce the release's own binding rather than
        # inventing a new one.  Absent from older datasets, hence the guard.
        if "sample_idx" in arrays:
            event["sample_idx"] = int(arrays["sample_idx"][index])
        if trajectory_generator is not None:
            generated = trajectory_generator(event)
            if generated is None:
                counts["trajectory_declined"] += 1
                tally["trajectory_declined"] += 1
            else:
                generated = np.asarray(generated, dtype=np.float32)
                if generated.shape != (stop - start, 2):
                    raise ValueError(
                        f"trajectory generator returned {generated.shape}"
                    )
                if not np.isfinite(generated).all():
                    raise ValueError("trajectory generator returned non-finite xy")
                counts["trajectory_clipped_samples"] += int(
                    np.count_nonzero((generated < 0.0) | (generated > 1.0))
                )
                placed, held, off_contact = apply_carrier_dispatch(
                    np.clip(generated, 0.0, 1.0), event["trajectory"]
                )
                trajectory[start:stop, 1:3] = placed
                recompute_deltas(trajectory[start:stop])
                counts["trajectory_swapped"] += 1
                counts["trajectory_rows_held_by_carrier"] += held
                counts["trajectory_rows_off_contact"] += off_contact
                counts["trajectory_rows_total"] += stop - start
                tally["trajectory_swapped"] += 1
        if imu_generator is not None:
            generated = imu_generator(event)
            if generated is None:
                counts["imu_declined"] += 1
                tally["imu_declined"] += 1
            else:
                generated = np.asarray(generated, dtype=np.float32)
                if generated.shape != (stop - start, IMU_CHANNELS):
                    raise ValueError(f"imu generator returned {generated.shape}")
                if not np.isfinite(generated).all():
                    raise ValueError("imu generator returned non-finite values")
                imu[start:stop] = generated
                counts["imu_swapped"] += 1
                tally["imu_swapped"] += 1
    arrays["trajectory_flat"] = trajectory
    arrays["imu_flat"] = imu
    counts["per_action"] = per_action
    return counts


def merge_counts(totals: dict, counts: dict) -> dict:
    """Accumulate one shard's counts, including the nested per-action tally."""

    for key, value in counts.items():
        if isinstance(value, dict):
            nested = totals.setdefault(key, {})
            for action, inner in value.items():
                target = nested.setdefault(action, {})
                for name, amount in inner.items():
                    target[name] = target.get(name, 0) + amount
        else:
            totals[key] = totals.get(key, 0) + value
    return totals


def finalise_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    shard_digests: dict,
    counts: dict,
    method_name: str,
    method_detail: dict,
    swapped: dict,
) -> None:
    """Write the manifest, provenance and release a rewritten dataset needs."""

    _rewrite_manifest(source_dir, output_dir, shard_digests)
    shutil.copy(source_dir / "provenance.jsonl", output_dir / "provenance.jsonl")
    release = json.loads((source_dir / "release.json").read_text())
    # The release points at its own files by absolute path and digest, and the
    # detector runner refuses a release whose manifest is not the one it was
    # handed, so both have to follow the clone.
    release["event_manifest"] = str(output_dir / "event_manifest.jsonl")
    release["event_manifest_sha256"] = sha256_file(
        output_dir / "event_manifest.jsonl"
    )
    release["provenance"] = str(output_dir / "provenance.jsonl")
    release["provenance_sha256"] = sha256_file(output_dir / "provenance.jsonl")
    release["baseline_method"] = method_name
    release["baseline_detail"] = method_detail
    # The comparison is not information-symmetric and the release has to say so.
    release["baseline_threat_model"] = {
        "generator_training_data": (
            "every genuine event of the 70 train-split users, pooled across "
            "users; the generator is user-agnostic"
        ),
        "source_method_training_data": (
            "five real events of the victim, per user, per action"
        ),
        "reading": (
            "the baselines see far more data than the source method and are "
            "not conditioned on a victim; a baseline that still loses is "
            "losing on generation quality, and one that wins may be winning "
            "on data volume"
        ),
    }
    release["baseline_source_dataset"] = str(source_dir)
    release["baseline_swapped"] = swapped
    (output_dir / "release.json").write_text(json.dumps(release, indent=2, sort_keys=True))
    (output_dir / "baseline_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True)
    )


def _rewrite_manifest(source_dir: Path, output_dir: Path, digests: dict) -> None:
    records = []
    named = set()
    for raw in (source_dir / "event_manifest.jsonl").read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        for shard in record.get("shards", []):
            named.add(Path(shard["source"]).name)
        records.append(record)

    # The manifest names every shard the release claims to contain.  If the
    # build did not produce one of them the dataset is incomplete, and the bare
    # KeyError this used to raise said only which name was missing -- from
    # inside the finalise step, after every shard had already been written, so
    # the output directory was left looking almost finished.  Check first, say
    # what is wrong, and leave nothing behind.
    missing = sorted(named - set(digests))
    if missing:
        raise ValueError(
            f"{len(missing)} shard(s) named by {source_dir.name}'s manifest were "
            f"never built: {', '.join(missing[:5])}"
            f"{' ...' if len(missing) > 5 else ''}. The source directory and its "
            "manifest disagree; rebuild from a complete source."
        )

    lines = []
    for record in records:
        for shard in record.get("shards", []):
            name = Path(shard["source"]).name
            shard["source"] = str(output_dir / "shards" / name)
            shard["source_sha256"] = digests[name]
        lines.append(json.dumps(record, sort_keys=True))
    (output_dir / "event_manifest.jsonl").write_text("\n".join(lines) + "\n")


__all__ = [
    "ACTIONS",
    "ACTION_WINDOW_SAMPLES",
    "collect_genuine_windows",
    "finalise_dataset",
    "iter_shards",
    "linear_resample",
    "load_shard",
    "recompute_deltas",
    "rewrite_shard_arrays",
    "save_shard",
    "sha256_file",
    "zoh_resample",
]
