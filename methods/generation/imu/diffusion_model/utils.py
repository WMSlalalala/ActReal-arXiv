from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


ACTION_NAMES = ("tap", "scroll", "swipe", "pinch")
GENERATOR_VERSION = "final_action_gen_v2_3_division_first_time_schema"
ORIENTATION_VALUES = np.array([-1, 0, 1, 3], dtype=np.int64)
ORIENTATION_NAMES = np.array(
    ["unknown", "portrait", "landscape_ccw_90", "landscape_cw_90"],
    dtype=object,
)
ORIENTATION_TO_INDEX = {-1: 0, 0: 1, 1: 2, 3: 3}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("YAML config root must be a mapping: %s" % path)
    data["_path"] = str(Path(path).resolve())
    return data


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(obj, str(tmp_path))
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def scalar_npz_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def as_int_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def orientation_to_index_np(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.int64)
    for raw, idx in ORIENTATION_TO_INDEX.items():
        out[values == raw] = idx
    return out


def orientation_to_raw_np(indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    idx = np.clip(idx, 0, len(ORIENTATION_VALUES) - 1)
    return ORIENTATION_VALUES[idx]


def default_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom
