from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    ACTION_NAMES,
    ORIENTATION_NAMES,
    ORIENTATION_VALUES,
    GENERATOR_VERSION,
    orientation_to_index_np,
)


REQUIRED_FIELDS = ("windows", "mask", "valid_mask", "active_len", "user_id", "orientation_id")
COMMON_META_FIELDS = (
    "duration_ms",
    "session_id",
    "motion_id_raw",
    "posture_id",
    "task_id",
    "activity_subtask_id",
    "window_start_ms",
)
XY_COND_FIELDS = ("xy_start_x", "xy_start_y", "xy_end_x", "xy_end_y")
LEGACY_TAP_XY_FIELDS = ("tap_x", "tap_y")
XY_FLOAT_META_FIELDS = (
    *XY_COND_FIELDS,
    "tap_x", "tap_y", "down_x", "down_y", "up_x", "up_y", "dx", "dy",
    "down_pressure", "up_pressure", "down_size", "up_size",
    "scroll_start_x", "scroll_start_y", "scroll_end_x", "scroll_end_y",
    "scroll_distance_x", "scroll_distance_y",
    "swipe_start_x", "swipe_start_y", "swipe_end_x", "swipe_end_y",
    "swipe_speed_x", "swipe_speed_y",
    "pinch_start_x", "pinch_start_y", "pinch_end_x", "pinch_end_y",
    "pinch_start_span", "pinch_end_span",
    "pinch_start_span_x", "pinch_start_span_y",
    "pinch_end_span_x", "pinch_end_span_y",
    "pinch_start_scale", "pinch_end_scale",
)
XY_INT_META_FIELDS = ("raw_tap_id_down", "raw_tap_id_up", "tap_type", "scroll_id", "pinch_id")
XY_META_FIELDS = (*XY_FLOAT_META_FIELDS, *XY_INT_META_FIELDS)
TRAJECTORY_COND_DIM = 10


@dataclass
class ActionData:
    action: str
    path: Path
    arrays: Dict[str, np.ndarray]
    T: int
    C: int
    hz: int
    pad_pre_pts: int

    @property
    def n(self) -> int:
        return int(self.arrays["windows"].shape[0])

    @property
    def users(self) -> np.ndarray:
        return np.unique(self.arrays["user_id"].astype(np.int64))


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    xy_mean: Optional[np.ndarray] = None
    xy_std: Optional[np.ndarray] = None
    traj_mean: Optional[np.ndarray] = None
    traj_std: Optional[np.ndarray] = None

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float32) - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

    def denormalize_np(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32) * self.std.reshape(1, 1, -1) + self.mean.reshape(1, 1, -1)

    @property
    def has_xy(self) -> bool:
        return self.xy_mean is not None and self.xy_std is not None

    @property
    def has_traj(self) -> bool:
        return self.traj_mean is not None and self.traj_std is not None

    def normalize_xy_np(self, xy: np.ndarray) -> np.ndarray:
        if not self.has_xy:
            raise RuntimeError("normalizer has no x/y statistics")
        raw = xy.astype(np.float32)
        mean = self.xy_mean.reshape(1, -1).astype(np.float32)
        std = self.xy_std.reshape(1, -1).astype(np.float32)
        raw = np.where(np.isfinite(raw), raw, mean)
        return ((raw - mean) / std).astype(np.float32)

    def normalize_traj_np(self, traj: np.ndarray) -> np.ndarray:
        if not self.has_traj:
            raise RuntimeError("normalizer has no trajectory statistics")
        raw = traj.astype(np.float32)
        mean = self.traj_mean.reshape(1, -1).astype(np.float32)
        std = self.traj_std.reshape(1, -1).astype(np.float32)
        raw = np.where(np.isfinite(raw), raw, mean)
        return ((raw - mean) / std).astype(np.float32)

    def to_dict(self) -> Dict[str, np.ndarray]:
        out = {"norm_mean": self.mean.astype(np.float32), "norm_std": self.std.astype(np.float32)}
        if self.has_xy:
            out["xy_mean"] = self.xy_mean.astype(np.float32)
            out["xy_std"] = self.xy_std.astype(np.float32)
        if self.has_traj:
            out["traj_mean"] = self.traj_mean.astype(np.float32)
            out["traj_std"] = self.traj_std.astype(np.float32)
        return out


def xy_fields_available(arrays: Dict[str, np.ndarray]) -> bool:
    return all(k in arrays for k in XY_COND_FIELDS) or all(k in arrays for k in LEGACY_TAP_XY_FIELDS)


def xy_raw_np(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    if all(k in arrays for k in XY_COND_FIELDS):
        return np.stack([arrays[k][idx] for k in XY_COND_FIELDS], axis=1).astype(np.float32)
    if all(k in arrays for k in LEGACY_TAP_XY_FIELDS):
        return np.stack([arrays[k][idx] for k in LEGACY_TAP_XY_FIELDS], axis=1).astype(np.float32)
    raise KeyError("no usable XY fields; expected %s or %s" % (XY_COND_FIELDS, LEGACY_TAP_XY_FIELDS))


def trajectory_raw_np(
    arrays: Dict[str, np.ndarray],
    idx: np.ndarray,
    active_len: Optional[np.ndarray] = None,
    hz: int = 100,
) -> np.ndarray:
    """Build a structured gesture-trajectory condition."""
    xy = xy_raw_np(arrays, idx).astype(np.float32)
    if xy.shape[1] == 2:
        sx = xy[:, 0]
        sy = xy[:, 1]
        ex = xy[:, 0]
        ey = xy[:, 1]
    else:
        sx, sy, ex, ey = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
    dx = ex - sx
    dy = ey - sy
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    denom = np.maximum(dist, 1e-6).astype(np.float32)
    dir_x = (dx / denom).astype(np.float32)
    dir_y = (dy / denom).astype(np.float32)
    if active_len is None:
        if "active_len" in arrays:
            active = arrays["active_len"][idx].astype(np.float32)
        else:
            active = np.ones((len(idx),), dtype=np.float32)
    else:
        active = np.asarray(active_len, dtype=np.float32).reshape(-1)
    duration_s = np.maximum(active / float(max(1, int(hz))), 1.0 / float(max(1, int(hz))))
    speed = (dist / duration_s).astype(np.float32)
    return np.stack([sx, sy, ex, ey, dx, dy, dist, dir_x, dir_y, speed], axis=1).astype(np.float32)


def load_action_data(data_dir: str, action: str) -> ActionData:
    if action not in ACTION_NAMES:
        raise ValueError("unknown action %s; valid=%s" % (action, ACTION_NAMES))
    path = Path(data_dir) / ("hmog_%s.npz" % action)
    if not path.exists():
        raise FileNotFoundError(str(path))
    z = np.load(str(path), allow_pickle=True)
    arrays = {k: z[k] for k in z.files}
    missing = [k for k in REQUIRED_FIELDS if k not in arrays]
    if missing:
        raise KeyError("%s missing required fields: %s" % (path, missing))
    w = arrays["windows"]
    if w.ndim != 3 or w.shape[-1] != 6:
        raise ValueError("%s windows must have shape [N,T,6], got %s" % (path, w.shape))
    n, T, C = w.shape
    for k in ("mask", "valid_mask"):
        if arrays[k].shape != (n, T):
            raise ValueError("%s field %s shape mismatch: %s vs (%d,%d)" % (path, k, arrays[k].shape, n, T))
    hz = int(np.asarray(arrays.get("hz", np.array(100))).item())
    pad_pre_pts = int(np.asarray(arrays.get("pad_pre_pts", np.array(0))).item())
    return ActionData(action=action, path=path, arrays=arrays, T=int(T), C=int(C), hz=hz, pad_pre_pts=pad_pre_pts)


def load_split(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    for k in ("train_users", "val_users", "test_users"):
        if k not in s:
            raise KeyError("split file %s missing %s" % (path, k))
    if "seed" not in s:
        s["seed"] = 42
    return s


def users_for(split: Dict[str, Any], name: str) -> np.ndarray:
    key = "%s_users" % name
    if key not in split:
        raise KeyError("split has no %s" % key)
    return np.asarray(split[key], dtype=np.int64)


def loss_mask_np(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return the supervised region for one gesture."""
    valid = valid_mask.astype(np.uint8) > 0
    return valid.astype(np.float32)


def compute_normalizer(data: ActionData, train_users: np.ndarray) -> Normalizer:
    a = data.arrays
    keep = np.isin(a["user_id"].astype(np.int64), train_users)
    lm = loss_mask_np(a["mask"], a["valid_mask"]) > 0
    keep2 = keep[:, None] & lm
    w = a["windows"].astype(np.float32)
    flat = w[keep2]
    if flat.size == 0:
        raise RuntimeError("no samples available to compute normalizer for %s" % data.action)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    xy_mean = None
    xy_std = None
    traj_mean = None
    traj_std = None
    if xy_fields_available(a):
        train_idx = np.flatnonzero(keep)
        xy = xy_raw_np(a, train_idx)
        valid_xy = np.isfinite(xy).all(axis=1)
        if valid_xy.any():
            xy_fit = xy[valid_xy]
            xy_mean = xy_fit.mean(axis=0).astype(np.float32)
            xy_std = np.maximum(xy_fit.std(axis=0), 1e-4).astype(np.float32)
        traj = trajectory_raw_np(a, train_idx, hz=data.hz)
        valid_traj = np.isfinite(traj).all(axis=1)
        if valid_traj.any():
            traj_fit = traj[valid_traj]
            traj_mean = traj_fit.mean(axis=0).astype(np.float32)
            traj_std = np.maximum(traj_fit.std(axis=0), 1e-4).astype(np.float32)
    return Normalizer(mean=mean, std=std, xy_mean=xy_mean, xy_std=xy_std, traj_mean=traj_mean, traj_std=traj_std)


def train_val_indices(
    data: ActionData,
    train_users: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    uid = data.arrays["user_id"].astype(np.int64)
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for u in train_users:
        idx = np.flatnonzero(uid == int(u))
        if len(idx) == 0:
            continue
        perm = rng.permutation(idx)
        n_val = int(round(len(perm) * float(val_ratio)))
        if len(perm) >= 10:
            n_val = max(1, n_val)
        else:
            n_val = 0
        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])
    train_idx = np.concatenate(train_parts).astype(np.int64) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def indices_for_users(data: ActionData, users: np.ndarray) -> np.ndarray:
    uid = data.arrays["user_id"].astype(np.int64)
    return np.flatnonzero(np.isin(uid, users)).astype(np.int64)


class UserRefBank:
    """Deterministic reference-window bank."""

    def __init__(
        self,
        data: ActionData,
        users: Iterable[int],
        k_refs: int,
        seed: int,
        candidate_indices: Optional[np.ndarray] = None,
    ):
        self.data = data
        self.k_refs = int(k_refs)
        self.seed = int(seed)
        uid = data.arrays["user_id"].astype(np.int64)
        allowed = None
        if candidate_indices is not None:
            allowed = np.zeros((len(uid),), dtype=bool)
            allowed[np.asarray(candidate_indices, dtype=np.int64)] = True
        self.refs_by_user: Dict[int, np.ndarray] = {}
        for u in users:
            idx = np.flatnonzero(uid == int(u))
            if allowed is not None:
                idx = idx[allowed[idx]]
            if len(idx) == 0:
                self.refs_by_user[int(u)] = np.array([], dtype=np.int64)
                continue
            rng = np.random.default_rng(self.seed + int(u) * 1009)
            perm = rng.permutation(idx)
            take = min(max(0, self.k_refs), len(perm))
            self.refs_by_user[int(u)] = np.sort(perm[:take]).astype(np.int64)

    def refs(self, user_id: int) -> np.ndarray:
        return self.refs_by_user.get(int(user_id), np.array([], dtype=np.int64))

    def sample_refs(
        self,
        user_id: int,
        exclude_index: Optional[int],
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base = self.refs(user_id)
        if exclude_index is not None and len(base):
            base = base[base != int(exclude_index)]
        if k <= 0:
            return np.array([], dtype=np.int64)
        if len(base) == 0:
            return np.array([], dtype=np.int64)
        replace = len(base) < k
        return rng.choice(base, size=int(k), replace=replace).astype(np.int64)


def _get_with_default(arrays: Dict[str, np.ndarray], key: str, idx: np.ndarray, default: int = -1) -> np.ndarray:
    if key in arrays:
        return arrays[key][idx]
    return np.full((len(idx),), int(default), dtype=np.int64)


def build_target_mask(T: int, pad_pre_pts: int, active_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    active = np.asarray(active_len, dtype=np.int64).copy()
    active = np.clip(active, 1, T)
    n = len(active)
    mask = np.zeros((n, T), dtype=np.uint8)
    valid = np.ones((n, T), dtype=np.uint8)
    start = int(np.clip(pad_pre_pts, 0, T - 1))
    for i, L in enumerate(active):
        end = int(np.clip(start + int(L), start + 1, T))
        mask[i, start:end] = 1
    return mask, valid


class MetadataPrior:
    def __init__(self, data: ActionData, users: np.ndarray, normalizer: Optional[Normalizer] = None):
        self.data = data
        self.users = np.asarray(users, dtype=np.int64)
        self.normalizer = normalizer
        idx = indices_for_users(data, self.users)
        if len(idx) == 0:
            raise RuntimeError("metadata prior has no rows for action %s" % data.action)
        self.idx = idx.astype(np.int64)

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        a = self.data.arrays
        choice = rng.choice(self.idx, size=int(n), replace=len(self.idx) < int(n)).astype(np.int64)
        active_len = a["active_len"][choice].astype(np.int64)
        orientation_id = a["orientation_id"][choice].astype(np.int64)
        mask, valid_mask = build_target_mask(self.data.T, self.data.pad_pre_pts, active_len)
        out: Dict[str, np.ndarray] = {
            "source_index": choice.astype(np.int64),
            "active_len": active_len,
            "orientation_id": orientation_id,
            "orientation_idx": orientation_to_index_np(orientation_id),
            "mask": mask,
            "valid_mask": valid_mask,
        }
        for k in COMMON_META_FIELDS:
            if k in a:
                out[k] = a[k][choice].astype(a[k].dtype)
        if xy_fields_available(a):
            xy = xy_raw_np(a, choice)
            for k in XY_META_FIELDS:
                if k in a and k not in out:
                    out[k] = a[k][choice].astype(a[k].dtype)
            if self.normalizer is not None and self.normalizer.has_xy:
                out["xy_cond"] = self.normalizer.normalize_xy_np(xy)
            if self.normalizer is not None and self.normalizer.has_traj:
                traj = trajectory_raw_np(a, choice, active_len=active_len, hz=self.data.hz)
                out["traj_cond"] = self.normalizer.normalize_traj_np(traj)
        return out


class WindowDataset(Dataset):
    def __init__(
        self,
        data: ActionData,
        indices: np.ndarray,
        normalizer: Normalizer,
        fewshot: bool,
        k_refs: int,
        ref_bank: Optional[UserRefBank],
        seed: int,
    ):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer
        self.fewshot = bool(fewshot)
        self.k_refs = int(k_refs)
        self.ref_bank = ref_bank
        self.seed = int(seed)
        self.max_refs = max(1, self.k_refs)
        self.windows_norm = normalizer.normalize_np(data.arrays["windows"])

    def __len__(self) -> int:
        return int(len(self.indices))

    def _refs_for(self, user_id: int, source_index: int, i: int) -> Tuple[np.ndarray, np.ndarray, int]:
        T, C = self.data.T, self.data.C
        refs = np.zeros((self.max_refs, T, C), dtype=np.float32)
        ref_mask = np.zeros((self.max_refs, T), dtype=np.float32)
        if not self.fewshot or self.k_refs <= 0 or self.ref_bank is None:
            return refs, ref_mask, 0
        rng = np.random.default_rng(self.seed + int(source_index) * 17 + int(i) * 101)
        ref_idx = self.ref_bank.sample_refs(user_id, source_index, self.k_refs, rng)
        if len(ref_idx) == 0:
            return refs, ref_mask, 0
        take = min(len(ref_idx), self.max_refs)
        ref_valid = self.data.arrays["valid_mask"][ref_idx[:take]].astype(np.float32)
        refs[:take] = self.windows_norm[ref_idx[:take]] * ref_valid[:, :, None]
        lm = loss_mask_np(self.data.arrays["mask"][ref_idx[:take]], ref_valid)
        ref_mask[:take] = lm.astype(np.float32)
        return refs, ref_mask, take

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        a = self.data.arrays
        uid = int(a["user_id"][idx])
        refs, ref_mask, ref_count = self._refs_for(uid, idx, i)
        out = {
            "x": torch.from_numpy((self.windows_norm[idx] * a["valid_mask"][idx].astype(np.float32)[:, None]).astype(np.float32)),
            "mask": torch.from_numpy(a["mask"][idx].astype(np.float32)),
            "valid_mask": torch.from_numpy(a["valid_mask"][idx].astype(np.float32)),
            "loss_mask": torch.from_numpy(loss_mask_np(a["mask"][idx:idx + 1], a["valid_mask"][idx:idx + 1])[0]),
            "active_len": torch.tensor(int(a["active_len"][idx]), dtype=torch.long),
            "orientation_idx": torch.tensor(int(orientation_to_index_np(np.array([a["orientation_id"][idx]], dtype=np.int64))[0]), dtype=torch.long),
            "user_id": torch.tensor(uid, dtype=torch.long),
            "source_index": torch.tensor(idx, dtype=torch.long),
            "refs": torch.from_numpy(refs),
            "ref_mask": torch.from_numpy(ref_mask),
            "ref_count": torch.tensor(ref_count, dtype=torch.long),
        }
        if self.normalizer.has_xy and xy_fields_available(a):
            xy = xy_raw_np(a, np.asarray([idx], dtype=np.int64))
            out["xy_cond"] = torch.from_numpy(self.normalizer.normalize_xy_np(xy)[0])
        if self.normalizer.has_traj and xy_fields_available(a):
            traj = trajectory_raw_np(a, np.asarray([idx], dtype=np.int64), hz=self.data.hz)
            out["traj_cond"] = torch.from_numpy(self.normalizer.normalize_traj_np(traj)[0])
        return out


def metadata_to_torch(meta: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in ("mask", "valid_mask"):
        out[k] = torch.from_numpy(meta[k].astype(np.float32)).to(device)
    out["loss_mask"] = out["valid_mask"].float()
    out["active_len"] = torch.from_numpy(meta["active_len"].astype(np.int64)).to(device)
    out["orientation_idx"] = torch.from_numpy(meta["orientation_idx"].astype(np.int64)).to(device)
    if "xy_cond" in meta:
        out["xy_cond"] = torch.from_numpy(meta["xy_cond"].astype(np.float32)).to(device)
    if "traj_cond" in meta:
        out["traj_cond"] = torch.from_numpy(meta["traj_cond"].astype(np.float32)).to(device)
    return out


def refresh_condition_metadata(
    data: ActionData,
    meta: Dict[str, np.ndarray],
    normalizer: Normalizer,
) -> Dict[str, np.ndarray]:
    """Refresh normalized condition vectors after active_len/mask overrides."""
    if not xy_fields_available(meta):
        return meta
    out = dict(meta)
    if normalizer.has_xy:
        out["xy_cond"] = normalizer.normalize_xy_np(xy_raw_np(out, np.arange(len(out["active_len"]), dtype=np.int64)))
    if normalizer.has_traj:
        idx = np.arange(len(out["active_len"]), dtype=np.int64)
        traj = trajectory_raw_np(out, idx, active_len=out["active_len"], hz=data.hz)
        out["traj_cond"] = normalizer.normalize_traj_np(traj)
    return out


def pack_fake_npz(
    data: ActionData,
    windows: np.ndarray,
    meta: Dict[str, np.ndarray],
    user_id: np.ndarray,
    ref_indices: np.ndarray,
    used_ref_indices: np.ndarray,
    normalizer: Normalizer,
    protocol: str,
    run_dir: Path,
    checkpoint_path: Path,
) -> Dict[str, np.ndarray]:
    n = int(windows.shape[0])
    active_len = meta["active_len"].astype(np.int64)
    buffer_duration_ms = active_len.astype(np.float64) * 1000.0 / float(data.hz)
    if "duration_ms" in meta:
        logical_duration_ms = meta["duration_ms"].astype(np.float64)
    else:
        logical_duration_ms = buffer_duration_ms.copy()
    logical_active_len = np.rint(logical_duration_ms / 1000.0 * float(data.hz)).astype(np.int64)
    logical_active_len = np.maximum(logical_active_len, 1)
    logical_buffer_duration_ms = logical_active_len.astype(np.float64) * 1000.0 / float(data.hz)
    model_max_active_len = int(data.T - data.pad_pre_pts)
    expected_effective_active_len = np.minimum(logical_active_len, model_max_active_len)
    clipped_frames = np.maximum(0, logical_active_len - model_max_active_len)
    out: Dict[str, np.ndarray] = {
        "windows": windows.astype(np.float32),
        "mask": meta["mask"].astype(np.uint8),
        "valid_mask": meta["valid_mask"].astype(np.uint8),
        "active_len": active_len,
        "logical_active_len": logical_active_len.astype(np.int64),
        "model_max_active_len": np.full((n,), model_max_active_len, dtype=np.int64),
        "duration_active_len_delta_frames": (active_len - expected_effective_active_len).astype(np.int64),
        "clipped_by_model_window": (clipped_frames > 0).astype(np.uint8),
        "duration_ms": logical_duration_ms.astype(np.float32),
        "logical_sample_duration_ms": logical_duration_ms.astype(np.float32),
        "event_duration_ms": logical_duration_ms.astype(np.float32),
        "parent_event_duration_ms": logical_duration_ms.astype(np.float32),
        "logical_event_duration_ms": logical_duration_ms.astype(np.float32),
        "buffer_duration_ms": buffer_duration_ms.astype(np.float32),
        "logical_buffer_duration_ms": logical_buffer_duration_ms.astype(np.float32),
        "duration_quantization_error_ms": (logical_buffer_duration_ms - logical_duration_ms).astype(np.float32),
        "emitted_duration_delta_ms": (buffer_duration_ms - logical_duration_ms).astype(np.float32),
        "clipped_buffer_duration_ms": (clipped_frames.astype(np.float64) * 1000.0 / float(data.hz)).astype(np.float32),
        "is_partial_event": (clipped_frames > 0).astype(np.uint8),
        "time_schema_version": np.array(2, dtype=np.int64),
        "user_id": user_id.astype(np.int64),
        "ref_indices": ref_indices.astype(np.int64),
        "used_ref_indices": used_ref_indices.astype(np.int64),
        "source_index": meta["source_index"].astype(np.int64),
        "orientation_id": meta["orientation_id"].astype(np.int64),
        "orientation_values": ORIENTATION_VALUES.copy(),
        "orientation_names": ORIENTATION_NAMES.copy(),
        "action": np.array(data.action),
        "split": np.array("generated"),
        "protocol": np.array(protocol),
        "generator_version": np.array(GENERATOR_VERSION),
        "run_dir": np.array(str(run_dir)),
        "checkpoint_path": np.array(str(checkpoint_path)),
        "hz": np.array(data.hz, dtype=np.int64),
        "T": np.array(data.T, dtype=np.int64),
        "pad_pre_pts": np.array(data.pad_pre_pts, dtype=np.int64),
    }
    out.update(normalizer.to_dict())
    for k in COMMON_META_FIELDS:
        if k == "duration_ms":
            # duration_ms is the logical action interval copied above.  It is
            # intentionally distinct from the 100 Hz frame-buffer duration.
            continue
        if k in meta:
            out[k] = meta[k]
        elif k in data.arrays:
            dtype = data.arrays[k].dtype
            out[k] = np.full((n,), -1, dtype=dtype)
    for k in XY_FLOAT_META_FIELDS:
        if k in meta:
            out[k] = meta[k].astype(np.float32)
        elif k in data.arrays:
            out[k] = np.full((n,), np.nan, dtype=np.float32)
    for k in XY_INT_META_FIELDS:
        if k in meta:
            out[k] = meta[k].astype(np.int64)
        elif k in data.arrays:
            out[k] = np.full((n,), -1, dtype=np.int64)
    return out
