from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .duration_policy import DurationPolicy, xy_distance_px
from .time_contract import (
    TIME_SCHEMA_VERSION,
    active_len_from_duration,
    build_active_timeline,
    build_window_timeline,
    validate_active_len,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
GENERATION_ROOT = REPOSITORY_ROOT / "methods" / "generation" / "imu"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "generator" / "five_shot"
PUBLIC_REFERENCE_ROOT = REPOSITORY_ROOT / "data" / "on_device"
# The selected ACTION_SPECS checkpoints were trained by the diffusion model
# implementation with model.representation=raw_ddpm.  Do not point this import
# at one of the later residual-level experiment copies: those copies subtract a
# per-window level from training data and add it back at sampling time, which is
# incompatible with these raw checkpoints.
DIFFUSION_MODEL_ROOT = GENERATION_ROOT
DEFAULT_POLICY_PATH = str(GENERATION_ROOT / "configs" / "duration_policy.json")


ACTION_SPECS: Dict[str, Dict[str, Any]] = {
    "tap": {
        "config": str(CHECKPOINT_ROOT / "tap" / "effective_config.json"),
        "run_dir": str(CHECKPOINT_ROOT / "tap"),
        "checkpoint": "model.pt",
        "sample_steps": 240,
    },
    "scroll": {
        "config": str(CHECKPOINT_ROOT / "scroll" / "effective_config.json"),
        "run_dir": str(CHECKPOINT_ROOT / "scroll"),
        "checkpoint": "model.pt",
        "sample_steps": 320,
    },
    "swipe": {
        "config": str(CHECKPOINT_ROOT / "swipe" / "effective_config.json"),
        "run_dir": str(CHECKPOINT_ROOT / "swipe"),
        "checkpoint": "model.pt",
        "sample_steps": 240,
    },
    "pinch": {
        "config": str(CHECKPOINT_ROOT / "pinch" / "effective_config.json"),
        "run_dir": str(CHECKPOINT_ROOT / "pinch"),
        "checkpoint": "model.pt",
        "sample_steps": 240,
    },
}


def _import_diffusion_model():
    root = str(DIFFUSION_MODEL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from diffusion_model.checkpoint import resolve_checkpoint  # type: ignore
    from diffusion_model.data import (  # type: ignore
        ActionData,
        MetadataPrior,
        Normalizer,
        UserRefBank,
        build_target_mask,
        load_action_data,
        load_split,
        metadata_to_torch,
        refresh_condition_metadata,
        users_for,
    )
    from diffusion_model.train import (  # type: ignore
        _build_refs_for_sampling,
        apply_action_overrides,
        apply_method_cfg,
        build_model,
        make_schedule,
        protocol_cfg,
    )
    from diffusion_model.utils import default_device, load_yaml  # type: ignore

    return {
        "resolve_checkpoint": resolve_checkpoint,
        "ActionData": ActionData,
        "MetadataPrior": MetadataPrior,
        "Normalizer": Normalizer,
        "UserRefBank": UserRefBank,
        "build_target_mask": build_target_mask,
        "load_action_data": load_action_data,
        "load_split": load_split,
        "metadata_to_torch": metadata_to_torch,
        "refresh_condition_metadata": refresh_condition_metadata,
        "users_for": users_for,
        "_build_refs_for_sampling": _build_refs_for_sampling,
        "apply_action_overrides": apply_action_overrides,
        "apply_method_cfg": apply_method_cfg,
        "build_model": build_model,
        "make_schedule": make_schedule,
        "protocol_cfg": protocol_cfg,
        "default_device": default_device,
        "load_yaml": load_yaml,
    }


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    hi = max(float(lo), float(hi))
    return float(min(max(float(v), float(lo)), hi))


def _validate_noise_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("noise_seed must be a non-negative integer")
    try:
        numeric = int(value)
    except Exception as exc:
        raise ValueError("noise_seed must be a non-negative integer") from exc
    if numeric < 0 or numeric > np.iinfo(np.int64).max or float(value) != float(numeric):
        raise ValueError("noise_seed must be a non-negative int64 integer")
    return numeric


def _load_public_fewshot_action_data(
    action: str,
    root: Path,
    device: str,
    action_data_type: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Build inference data from the released 20-participant references.

    The training loader expects one HMOG action archive.  Those licensed signal
    shards cannot be redistributed, but five-shot inference needs only the
    target reference windows and the normalization state already embedded in
    the checkpoint.  This adapter presents the released phone references using
    the same ``ActionData`` contract without materializing another dataset.
    """

    root = Path(root)
    device = str(device)
    if device not in {"pixel10", "s21"}:
        raise ValueError(
            "reference_device must be 'pixel10' or 's21', got %r" % device
        )
    paths = sorted(root.glob("P[0-9][0-9]/%s/fewshot/%s.npz" % (device, action)))
    expected_ids = {"P%02d" % index for index in range(20)}
    found_ids = {path.parts[-4] for path in paths}
    if found_ids != expected_ids or len(paths) != 20:
        raise FileNotFoundError(
            "released references are incomplete for %s/%s: expected P00--P19 "
            "below %s, found %s"
            % (device, action, root, sorted(found_ids))
        )

    chunks: Dict[str, list[np.ndarray]] = {
        "windows": [],
        "mask": [],
        "valid_mask": [],
        "active_len": [],
        "duration_ms": [],
        "orientation_id": [],
        "user_id": [],
        "xy_start_x": [],
        "xy_start_y": [],
        "xy_end_x": [],
        "xy_end_y": [],
    }
    expected_shape: Optional[Tuple[int, int]] = None
    for path in paths:
        participant = path.parts[-4]
        user_id = int(participant[1:])
        with np.load(path, allow_pickle=False) as archive:
            archive_schema = str(np.asarray(archive["schema"]).item())
            archive_participant = str(np.asarray(archive["participant_id"]).item())
            archive_action = str(np.asarray(archive["action"]).item())
            archive_device = str(np.asarray(archive["device"]).item())
            archive_split = str(np.asarray(archive["split"]).item())
            sample_count = int(np.asarray(archive["sample_count"]).item())
            if (
                archive_schema != "actreal_on_device_processed_v1"
                or archive_participant != participant
                or archive_action != action
                or archive_device != device
                or archive_split != "fewshot"
                or sample_count != 5
            ):
                raise ValueError(
                    "%s is not a five-reference %s/%s archive"
                    % (path, device, action)
                )
            windows = np.asarray(archive["imu"], dtype=np.float32)
            if windows.ndim != 3:
                raise ValueError("%s has invalid IMU shape %s" % (path, windows.shape))
            shape = (int(windows.shape[1]), int(windows.shape[2]))
            if windows.shape[0] != sample_count or shape[1] != 6:
                raise ValueError("%s has invalid IMU shape %s" % (path, windows.shape))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "%s has IMU shape %s, expected (*,%s,%s)"
                    % (path, windows.shape, expected_shape[0], expected_shape[1])
                )
            xy = np.asarray(archive["xy_hmog"], dtype=np.float32)
            if xy.shape != (sample_count, 4) or not np.isfinite(xy).all():
                raise ValueError("%s has invalid xy_hmog shape %s" % (path, xy.shape))
            mask = np.asarray(archive["action_mask"], dtype=np.uint8)
            valid_mask = np.asarray(archive["imu_valid_mask"], dtype=np.uint8)
            active_len = np.asarray(archive["active_len"], dtype=np.int64)
            duration_ms = np.asarray(archive["duration_ms"], dtype=np.float32)
            orientation_id = np.asarray(archive["orientation_id"], dtype=np.int64)
            if mask.shape != windows.shape[:2] or valid_mask.shape != windows.shape[:2]:
                raise ValueError("%s has a mask/IMU shape mismatch" % path)
            for name, values in (
                ("active_len", active_len),
                ("duration_ms", duration_ms),
                ("orientation_id", orientation_id),
            ):
                if values.shape != (sample_count,):
                    raise ValueError("%s has invalid %s shape %s" % (path, name, values.shape))
            chunks["windows"].append(windows)
            chunks["mask"].append(mask)
            chunks["valid_mask"].append(valid_mask)
            chunks["active_len"].append(active_len)
            chunks["duration_ms"].append(duration_ms)
            chunks["orientation_id"].append(orientation_id)
            chunks["user_id"].append(
                np.full((sample_count,), user_id, dtype=np.int64)
            )
            for index, field in enumerate(
                ("xy_start_x", "xy_start_y", "xy_end_x", "xy_end_y")
            ):
                chunks[field].append(xy[:, index])

    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    starts = []
    for mask in arrays["mask"]:
        active = np.flatnonzero(mask)
        if active.size == 0:
            raise ValueError("released %s references contain an empty action mask" % action)
        starts.append(int(active[0]))
    if len(set(starts)) != 1:
        raise ValueError(
            "released %s references disagree on pre-padding: %s"
            % (action, sorted(set(starts)))
        )
    if expected_shape is None:  # defensive: the P00--P19 check makes this unreachable.
        raise RuntimeError("no released references loaded for %s" % action)
    users = sorted(int(value) for value in np.unique(arrays["user_id"]))
    data = action_data_type(
        action=action,
        path=root / ("released_%s_%s_fewshot" % (device, action)),
        arrays=arrays,
        T=expected_shape[0],
        C=expected_shape[1],
        hz=100,
        pad_pre_pts=starts[0],
    )
    split = {
        "seed": 42,
        "train_users": users,
        "val_users": users,
        "test_users": users,
    }
    return data, split


@dataclass
class _Runtime:
    action: str
    spec: Dict[str, Any]
    cfg: Dict[str, Any]
    proto: Dict[str, Any]
    data: Any
    split: Dict[str, Any]
    normalizer: Any
    model: torch.nn.Module
    schedule: Any
    prior: Any
    ref_bank: Any
    ref_seed: int
    k_refs: int
    device: torch.device
    checkpoint_path: Path
    data_source: str
    reference_device: Optional[str]
    load_wall_ms: float


class AndroidIMUDiffusionLayer:
    """Android-facing diffusion generator."""

    def __init__(
        self,
        policy_path: str = DEFAULT_POLICY_PATH,
        seed: int = 42,
        device: Optional[str] = None,
        protocol: str = "fewshot_adv",
        method: str = "diffusion",
        split: str = "test",
        use_ema: bool = True,
        action_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        reference_data_root: Optional[str] = None,
        reference_device: Optional[str] = None,
        tap_jitter_frac: float = 0.10,
        duration_jitter_frac: float = 0.08,
    ):
        self.policy = DurationPolicy.from_json(policy_path)
        self.seed = int(seed)
        self.device_override = device
        self.protocol = str(protocol)
        self.method = str(method)
        self.split = str(split)
        self.use_ema = bool(use_ema)
        self.action_specs = dict(action_specs or ACTION_SPECS)
        self.reference_data_root = Path(
            reference_data_root
            or os.environ.get("ACTREAL_PUBLIC_REFERENCE_ROOT", str(PUBLIC_REFERENCE_ROOT))
        )
        self.reference_device = str(
            reference_device
            or os.environ.get("ACTREAL_REFERENCE_DEVICE", "pixel10")
        )
        self.tap_jitter_frac = float(tap_jitter_frac)
        self.duration_jitter_frac = float(duration_jitter_frac)
        self.rng = np.random.default_rng(self.seed)
        self._fg = _import_diffusion_model()
        self._runtime: Dict[str, _Runtime] = {}

    def tap(
        self,
        x: float,
        y: float,
        user_id: Optional[int] = None,
        duration_ms: Optional[float] = None,
        orientation_id: Optional[int] = None,
        sample_steps: Optional[int] = None,
        active_len: Optional[int] = None,
        start_time_ns: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.generate(
            "tap",
            user_id=user_id,
            xy_start=(x, y),
            xy_end=(x, y),
            duration_ms=duration_ms,
            active_len=active_len,
            start_time_ns=start_time_ns,
            orientation_id=orientation_id,
            sample_steps=sample_steps,
            noise_seed=noise_seed,
        )

    def scroll(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        user_id: Optional[int] = None,
        duration_ms: Optional[float] = None,
        orientation_id: Optional[int] = None,
        sample_steps: Optional[int] = None,
        active_len: Optional[int] = None,
        start_time_ns: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.generate(
            "scroll",
            user_id=user_id,
            xy_start=(x0, y0),
            xy_end=(x1, y1),
            duration_ms=duration_ms,
            active_len=active_len,
            start_time_ns=start_time_ns,
            orientation_id=orientation_id,
            sample_steps=sample_steps,
            noise_seed=noise_seed,
        )

    def swipe(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        user_id: Optional[int] = None,
        duration_ms: Optional[float] = None,
        orientation_id: Optional[int] = None,
        sample_steps: Optional[int] = None,
        active_len: Optional[int] = None,
        start_time_ns: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.generate(
            "swipe",
            user_id=user_id,
            xy_start=(x0, y0),
            xy_end=(x1, y1),
            duration_ms=duration_ms,
            active_len=active_len,
            start_time_ns=start_time_ns,
            orientation_id=orientation_id,
            sample_steps=sample_steps,
            noise_seed=noise_seed,
        )

    def pinch(
        self,
        center: Optional[Sequence[float]] = None,
        start_span: Optional[float] = None,
        end_span: Optional[float] = None,
        user_id: Optional[int] = None,
        duration_ms: Optional[float] = None,
        orientation_id: Optional[int] = None,
        sample_steps: Optional[int] = None,
        end_center: Optional[Sequence[float]] = None,
        active_len: Optional[int] = None,
        start_time_ns: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_xy = tuple(center[:2]) if center is not None and len(center) >= 2 else None
        end_xy = tuple(end_center[:2]) if end_center is not None and len(end_center) >= 2 else start_xy
        return self.generate(
            "pinch",
            user_id=user_id,
            xy_start=start_xy,
            xy_end=end_xy,
            duration_ms=duration_ms,
            active_len=active_len,
            start_time_ns=start_time_ns,
            orientation_id=orientation_id,
            sample_steps=sample_steps,
            noise_seed=noise_seed,
            pinch_start_span=start_span,
            pinch_end_span=end_span,
        )

    def generate(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        if action not in self.action_specs:
            raise KeyError("unknown action %r; known=%s" % (action, sorted(self.action_specs)))
        t0 = time.perf_counter()
        rt, loaded_now = self._get_runtime(action)
        t_after_load = time.perf_counter()
        user_id = self._resolve_user(rt, kwargs.get("user_id"))
        request_kwargs = dict(kwargs)
        request_kwargs.pop("user_id", None)
        meta, duration_info = self._metadata_for_request(rt, user_id=user_id, **request_kwargs)
        t_after_meta = time.perf_counter()
        tbatch = self._fg["metadata_to_torch"](meta, rt.device)
        refs, ref_mask, ref_count, audit_refs, used_refs = self._fg["_build_refs_for_sampling"](
            rt.data,
            rt.normalizer,
            int(user_id),
            fewshot=bool(rt.proto.get("fewshot", False)),
            k_refs=rt.k_refs,
            ref_bank=rt.ref_bank,
            batch_n=1,
        )
        tbatch["refs"] = refs.to(rt.device)
        tbatch["ref_mask"] = ref_mask.to(rt.device)
        tbatch["ref_count"] = ref_count.to(rt.device)
        steps = int(kwargs.get("sample_steps") or rt.spec.get("sample_steps") or rt.cfg["diffusion"].get("sample_steps", 80))
        noise_seed = (
            _validate_noise_seed(kwargs["noise_seed"])
            if kwargs.get("noise_seed") is not None
            else int(self.rng.integers(0, np.iinfo(np.int64).max))
        )
        t_sample0 = time.perf_counter()
        fork_devices = list(range(torch.cuda.device_count())) if rt.device.type == "cuda" else []
        # Keep PyTorch's process-global RNG state unchanged so concurrent
        # callers and unrelated training code are not perturbed by this API.
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(noise_seed)
            with torch.no_grad():
                x = rt.schedule.sample(
                    rt.model,
                    (1, rt.data.T, rt.data.C),
                    tbatch,
                    sample_steps=steps,
                    clamp_std=float(rt.cfg["sample"].get("clamp_std", 8.0)),
                )
        if rt.device.type == "cuda":
            torch.cuda.synchronize(rt.device)
        t_sample1 = time.perf_counter()
        representation = str(rt.cfg.get("model", {}).get("representation", "raw_ddpm"))
        if representation != "raw_ddpm":
            raise ValueError(
                "AndroidIMUDiffusionLayer ACTION_SPECS currently supports raw_ddpm checkpoints only; "
                "got model.representation=%r for action=%s" % (representation, action)
            )
        # The checkpoint normalizer already maps normalized diffusion output to
        # detector-visible raw IMU.  Adding a sampled level here would double the
        # gravity/device baseline and make fake windows trivially detectable.
        raw = rt.normalizer.denormalize_np(x.detach().cpu().numpy())
        window = raw[0].astype(np.float32)
        mask = meta["mask"][0].astype(np.uint8)
        valid_mask = meta["valid_mask"][0].astype(np.uint8)
        active_selector = mask > 0
        active_imu = window[active_selector].astype(np.float32)
        t1 = time.perf_counter()
        active_len = int(meta["active_len"][0])
        logical_duration_ms = float(duration_info["logical_event_duration_ms"])
        active_time = build_active_timeline(
            active_len,
            float(rt.data.hz),
            logical_duration_ms,
            start_time_ns=kwargs.get("start_time_ns"),
            logical_active_len=int(duration_info["logical_active_len"]),
            model_max_active_len=int(max(1, rt.data.T - rt.data.pad_pre_pts)),
        )
        active_indices = np.flatnonzero(active_selector)
        if active_indices.size == 0:
            raise RuntimeError("generated mask contains no active samples")
        window_time = build_window_timeline(
            int(rt.data.T),
            int(active_indices[0]),
            float(rt.data.hz),
            start_time_ns=kwargs.get("start_time_ns"),
        )
        metadata = {
            "generator": "diffusion",
            "action": action,
            "protocol": self.protocol,
            "method": self.method,
            "run_dir": str(rt.spec["run_dir"]),
            "checkpoint": str(rt.checkpoint_path),
            "checkpoint_loaded_this_call": bool(loaded_now),
            "user_id": int(user_id),
            "reference_data_source": rt.data_source,
            "reference_device": rt.reference_device,
            "reference_participant_id": (
                "P%02d" % int(user_id)
                if rt.data_source == "released_on_device_fewshot"
                else None
            ),
            "T": int(rt.data.T),
            "hz": float(rt.data.hz),
            "active_len": active_len,
            "model_max_active_len": int(max(1, rt.data.T - rt.data.pad_pre_pts)),
            "time_schema_version": TIME_SCHEMA_VERSION,
            "event_duration_ms": logical_duration_ms,
            "parent_event_duration_ms": logical_duration_ms,
            "logical_sample_duration_ms": logical_duration_ms,
            "logical_event_duration_ms": logical_duration_ms,
            "buffer_duration_ms": float(active_time["buffer_duration_ms"]),
            "logical_buffer_duration_ms": float(active_time["logical_buffer_duration_ms"]),
            "duration_quantization_error_ms": float(active_time["duration_quantization_error_ms"]),
            "emitted_duration_delta_ms": float(active_time["emitted_duration_delta_ms"]),
            "clipped_buffer_duration_ms": float(active_time["clipped_buffer_duration_ms"]),
            "sample_period_ns": int(active_time["sample_period_ns"]),
            "start_time_ns": active_time["start_time_ns"],
            "end_time_ns": active_time["end_time_ns"],
            "buffer_end_time_ns": active_time["buffer_end_time_ns"],
            "window_start_time_ns": window_time["window_start_time_ns"],
            "is_partial_event": bool(duration_info.get("is_partial_event", False)),
            "window_duration_ms": float(rt.data.T * 1000.0 / float(rt.data.hz)),
            "mask_sum": int(mask.sum()),
            "valid_sum": int(valid_mask.sum()),
            "orientation_id": int(meta["orientation_id"][0]),
            "k_refs": int(rt.k_refs),
            "ref_bank_seed": int(rt.ref_seed),
            "ref_count": int(ref_count.cpu().numpy()[0]) if hasattr(ref_count, "cpu") else int(ref_count[0]),
            "ref_indices": audit_refs[0].astype(int).tolist() if audit_refs.size else [],
            "used_ref_indices": used_refs[0].astype(int).tolist() if used_refs.size else [],
            "sample_steps": int(steps),
            "noise_seed": int(noise_seed),
            "representation": representation,
            "duration_policy": duration_info,
            "model_load_ms": float((t_after_load - t0) * 1000.0) if loaded_now else 0.0,
            "metadata_ms": float((t_after_meta - t_after_load) * 1000.0),
            "sampling_ms": float((t_sample1 - t_sample0) * 1000.0),
            "generation_wall_ms": float((t1 - t0) * 1000.0),
            "xy_start": self._seq_to_list(kwargs.get("xy_start")),
            "xy_end": self._seq_to_list(kwargs.get("xy_end")),
            "xy_condition_note": "recorded for the event plan and for the duration policy; the denoiser is never conditioned on it",
        }
        return {
            "action": action,
            "hz": float(rt.data.hz),
            "window": window,
            "active_imu": active_imu,
            "mask": mask,
            "valid_mask": valid_mask,
            "relative_timestamps_ns": active_time["relative_timestamps_ns"],
            "timestamps_ns": active_time["timestamps_ns"],
            "window_relative_timestamps_ns": window_time["window_relative_timestamps_ns"],
            "window_timestamps_ns": window_time["window_timestamps_ns"],
            "metadata": metadata,
            "sample_period_ns": int(active_time["sample_period_ns"]),
            "start_time_ns": active_time["start_time_ns"],
            "end_time_ns": active_time["end_time_ns"],
            "buffer_end_time_ns": active_time["buffer_end_time_ns"],
            "event_duration_ms": logical_duration_ms,
            "buffer_duration_ms": float(active_time["buffer_duration_ms"]),
        }

    def _get_runtime(self, action: str) -> Tuple[_Runtime, bool]:
        if action in self._runtime:
            return self._runtime[action], False
        t0 = time.perf_counter()
        spec = self.action_specs[action]
        cfg = self._fg["apply_action_overrides"](
            self._fg["apply_method_cfg"](self._fg["load_yaml"](spec["config"]), self.method),
            action,
        )
        data_override = os.environ.get("ACTREAL_GENERATOR_DATA_ROOT")
        split_override = os.environ.get("ACTREAL_SPLIT_FILE")
        if data_override is not None:
            cfg["paths"]["data_dir"] = data_override
        if split_override is not None:
            cfg["paths"]["split_file"] = split_override
        proto = self._fg["protocol_cfg"](cfg, self.protocol)
        try:
            data = self._fg["load_action_data"](cfg["paths"]["data_dir"], action)
        except FileNotFoundError as exc:
            if data_override is not None or split_override is not None:
                raise FileNotFoundError(
                    "explicit ACTREAL_GENERATOR_DATA_ROOT/ACTREAL_SPLIT_FILE inputs "
                    "could not be loaded; public-reference fallback is enabled only "
                    "when the default licensed action data are absent"
                ) from exc
            data, split = _load_public_fewshot_action_data(
                action,
                self.reference_data_root,
                self.reference_device,
                self._fg["ActionData"],
            )
            data_source = "released_on_device_fewshot"
            reference_device = self.reference_device
        else:
            # If the configured action data exist, a missing or invalid split is
            # an input error rather than a reason to change the inference cohort.
            split = self._fg["load_split"](cfg["paths"]["split_file"])
            data_source = "configured_action_data"
            reference_device = None
        device = torch.device(self.device_override) if self.device_override else self._fg["default_device"](str(cfg["runtime"].get("device", "auto")))
        checkpoint_selector = Path(str(spec["checkpoint"]))
        if not checkpoint_selector.is_absolute():
            packaged_candidate = Path(spec["run_dir"]) / checkpoint_selector
            if packaged_candidate.is_file():
                checkpoint_selector = packaged_candidate
        checkpoint_path = self._fg["resolve_checkpoint"](
            Path(spec["run_dir"]), str(checkpoint_selector)
        )
        state = torch.load(str(checkpoint_path), map_location=device)
        normalizer_state = state["normalizer"]
        normalizer = self._fg["Normalizer"](
            mean=np.asarray(normalizer_state["norm_mean"], dtype=np.float32),
            std=np.asarray(normalizer_state["norm_std"], dtype=np.float32),
            xy_mean=np.asarray(normalizer_state["xy_mean"], dtype=np.float32)
            if "xy_mean" in normalizer_state
            else None,
            xy_std=np.asarray(normalizer_state["xy_std"], dtype=np.float32)
            if "xy_std" in normalizer_state
            else None,
            traj_mean=np.asarray(normalizer_state["traj_mean"], dtype=np.float32)
            if "traj_mean" in normalizer_state
            else None,
            traj_std=np.asarray(normalizer_state["traj_std"], dtype=np.float32)
            if "traj_std" in normalizer_state
            else None,
        )
        model = self._fg["build_model"](cfg, data, proto).to(device)
        model.load_state_dict(state["model_state"])
        if self.use_ema and state.get("ema_state") is not None:
            ms = model.state_dict()
            for k, v in state["ema_state"].items():
                if k in ms:
                    ms[k].copy_(v.to(device))
        model.eval()
        schedule = self._fg["make_schedule"](cfg, device)
        prior = self._fg["MetadataPrior"](data, self._fg["users_for"](split, "train"), normalizer=normalizer)
        target_users = self._fg["users_for"](split, self.split)
        k_refs = int(proto.get("k_refs", 0)) if bool(proto.get("fewshot", False)) else 0
        ref_seed = int(cfg["runtime"].get("seed", 42)) + 303
        ref_bank = (
            self._fg["UserRefBank"](data, target_users, k_refs, ref_seed)
            if k_refs > 0
            else None
        )
        rt = _Runtime(
            action=action,
            spec=spec,
            cfg=cfg,
            proto=proto,
            data=data,
            split=split,
            normalizer=normalizer,
            model=model,
            schedule=schedule,
            prior=prior,
            ref_bank=ref_bank,
            ref_seed=ref_seed,
            k_refs=k_refs,
            device=device,
            checkpoint_path=checkpoint_path,
            data_source=data_source,
            reference_device=reference_device,
            load_wall_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        self._runtime[action] = rt
        return rt, True

    def _metadata_for_request(self, rt: _Runtime, user_id: int, **kwargs: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        meta = rt.prior.sample(1, self.rng)
        duration_ms, info = self._duration_ms_for_action(rt.action, rt.data, **kwargs)
        logical_active_len = active_len_from_duration(duration_ms, float(rt.data.hz))
        explicit_active_len = kwargs.get("active_len")
        if explicit_active_len is None:
            requested_active_len = int(logical_active_len)
            active_len_source = "derived_from_logical_duration"
        else:
            requested_active_len = validate_active_len(explicit_active_len)
            active_len_source = "explicit_preprocessed_frame_count"
        max_active = int(max(1, rt.data.T - rt.data.pad_pre_pts))
        expected_effective_active_len = min(int(logical_active_len), max_active)
        # Match preprocessing's division-first floating-point expression
        # exactly.  At half-frame boundaries, reordering the multiplication
        # produces a different IEEE-754 result even though the algebra looks
        # equivalent; accepting +/- one frame here would hide that bug.
        duration_active_len_delta = int(requested_active_len - expected_effective_active_len)
        if explicit_active_len is not None and duration_active_len_delta != 0:
            raise ValueError(
                "active_len=%d contradicts duration_ms=%.9g for action=%s at %.9g Hz; expected %d"
                % (
                    requested_active_len,
                    float(duration_ms),
                    rt.action,
                    float(rt.data.hz),
                    expected_effective_active_len,
                )
            )
        active_len = max(1, min(requested_active_len, max_active))
        buffer_duration_ms = float(active_len * 1000.0 / float(rt.data.hz))
        info["time_schema_version"] = TIME_SCHEMA_VERSION
        info["logical_event_duration_ms"] = float(duration_ms)
        info["logical_active_len"] = int(logical_active_len)
        info["active_len_source"] = active_len_source
        info["requested_active_len"] = requested_active_len
        info["effective_active_len"] = int(active_len)
        info["model_max_active_len"] = int(max_active)
        info["duration_active_len_delta_frames"] = duration_active_len_delta
        info["buffer_duration_ms"] = buffer_duration_ms
        logical_buffer_duration_ms = float(logical_active_len * 1000.0 / float(rt.data.hz))
        info["logical_buffer_duration_ms"] = logical_buffer_duration_ms
        info["duration_quantization_error_ms"] = float(logical_buffer_duration_ms - float(duration_ms))
        info["emitted_duration_delta_ms"] = float(buffer_duration_ms - float(duration_ms))
        # ``clipped`` is reserved for loss caused by the model window.  A
        # one-frame difference between an observed frame count and the nominal
        # duration mapping is sensor-clock quantisation, not a partial event.
        info["clipped_buffer_duration_ms"] = float(
            max(0, int(logical_active_len) - int(max_active)) * 1000.0 / float(rt.data.hz)
        )
        info["clipped_by_model_window"] = bool(
            requested_active_len != active_len or int(logical_active_len) > int(max_active)
        )
        info["active_len_differs_from_duration_mapping"] = bool(requested_active_len != logical_active_len)
        info["is_partial_event"] = bool(int(logical_active_len) > int(max_active))
        mask, valid_mask = self._fg["build_target_mask"](rt.data.T, rt.data.pad_pre_pts, np.asarray([active_len]))
        meta["active_len"] = np.asarray([active_len], dtype=np.int64)
        meta["duration_ms"] = np.asarray([float(duration_ms)], dtype=np.float32)
        meta["mask"] = mask.astype(np.uint8)
        meta["valid_mask"] = valid_mask.astype(np.uint8)
        if kwargs.get("orientation_id") is not None:
            ori = int(kwargs["orientation_id"])
            meta["orientation_id"] = np.asarray([ori], dtype=np.int64)
            from diffusion_model.utils import orientation_to_index_np  # type: ignore

            meta["orientation_idx"] = orientation_to_index_np(meta["orientation_id"])
        self._override_xy(rt.action, meta, kwargs)
        meta = self._fg["refresh_condition_metadata"](rt.data, meta, rt.normalizer)
        return meta, info

    def _duration_ms_for_action(self, action: str, data: Any, **kwargs: Any) -> Tuple[float, Dict[str, Any]]:
        requested = kwargs.get("duration_ms")
        if requested is None and kwargs.get("active_len") is not None:
            length = validate_active_len(kwargs["active_len"])
            duration = float(length * 1000.0 / float(data.hz))
            return duration, {
                "mode": "explicit_active_len_without_logical_duration",
                "requested_duration_ms": None,
                "duration_ms": duration,
                "duration_clipped_by_policy": False,
            }
        cfg = self.policy.actions.get(action, {})
        lo = _finite_float(cfg.get("p05_ms"), cfg.get("min_ms", 10.0))
        hi = _finite_float(cfg.get("p95_ms"), cfg.get("max_ms", max(lo, 100.0)))
        mode = "manual" if requested is not None else "android_policy"
        if requested is not None:
            try:
                duration = float(requested)
            except Exception as exc:
                raise ValueError("duration_ms must be finite and > 0") from exc
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration_ms must be finite and > 0")
            return duration, {
                "mode": mode,
                "requested_duration_ms": float(requested),
                "duration_ms": duration,
                "duration_clipped_by_policy": False,
                "manual_duration_is_exact": True,
            }
        if action == "tap":
            median = _finite_float(cfg.get("median_ms"), 85.0)
            sigma = max(1.0, abs(median) * self.tap_jitter_frac)
            raw = float(self.rng.normal(median, sigma))
            duration = _clamp(raw, lo, hi)
            return duration, {"mode": "tap_median_plus_small_random", "median_ms": median, "jitter_sigma_ms": sigma, "duration_ms": duration}
        if action in {"scroll", "swipe"}:
            dist = xy_distance_px(kwargs.get("xy_start"), kwargs.get("xy_end"))
            if dist is None:
                base = _finite_float(cfg.get("median_ms"), 370.0)
                source = "median_no_xy"
            elif action == "scroll":
                base = _finite_float(cfg.get("intercept_ms"), cfg.get("median_ms", 365.0)) + _finite_float(
                    cfg.get("slope_ms_per_unit"), 0.0
                ) * float(dist)
                source = "linear_xy_distance"
            else:
                base = self._binned_duration(cfg, float(dist), _finite_float(cfg.get("median_ms"), 370.0))
                source = "binned_xy_distance"
            sigma = max(1.0, abs(base) * self.duration_jitter_frac)
            duration = _clamp(float(self.rng.normal(base, sigma)), lo, hi)
            return {
                "scroll": duration,
                "swipe": duration,
            }[action], {"mode": source + "_plus_random", "distance_px": None if dist is None else float(dist), "base_ms": float(base), "jitter_sigma_ms": sigma, "duration_ms": duration}
        if action == "pinch":
            if kwargs.get("pinch_start_span") is not None and kwargs.get("pinch_end_span") is not None:
                span_delta = abs(float(kwargs["pinch_end_span"]) - float(kwargs["pinch_start_span"]))
            else:
                span_delta = None
            base = self._binned_duration(cfg, span_delta, _finite_float(cfg.get("median_ms"), 420.0)) if span_delta is not None else _finite_float(cfg.get("median_ms"), 420.0)
            sigma = max(1.0, abs(base) * self.duration_jitter_frac)
            duration = _clamp(float(self.rng.normal(base, sigma)), lo, hi)
            return duration, {"mode": "pinch_span_delta_binned_plus_random", "span_delta": span_delta, "base_ms": float(base), "jitter_sigma_ms": sigma, "duration_ms": duration}
        median = _finite_float(cfg.get("median_ms"), 100.0)
        return _clamp(median, lo, hi), {"mode": "fallback_median", "duration_ms": _clamp(median, lo, hi)}

    @staticmethod
    def _binned_duration(cfg: Dict[str, Any], value: Optional[float], default: float) -> float:
        if value is None:
            return float(default)
        bins = cfg.get("bins") or []
        for row in bins:
            lo = _finite_float(row.get("lo"), -float("inf"))
            hi = _finite_float(row.get("hi"), float("inf"))
            if float(value) >= lo and float(value) <= hi:
                return _finite_float(row.get("median_ms"), default)
        return _finite_float(bins[-1].get("median_ms"), default) if bins else float(default)

    def _override_xy(self, action: str, meta: Dict[str, np.ndarray], kwargs: Dict[str, Any]) -> None:
        xy_start = kwargs.get("xy_start")
        xy_end = kwargs.get("xy_end")
        if xy_start is not None and xy_end is not None and len(xy_start) >= 2 and len(xy_end) >= 2:
            sx, sy = float(xy_start[0]), float(xy_start[1])
            ex, ey = float(xy_end[0]), float(xy_end[1])
            for k, v in {
                "xy_start_x": sx,
                "xy_start_y": sy,
                "xy_end_x": ex,
                "xy_end_y": ey,
                "tap_x": sx,
                "tap_y": sy,
                "down_x": sx,
                "down_y": sy,
                "up_x": ex,
                "up_y": ey,
                "dx": ex - sx,
                "dy": ey - sy,
                f"{action}_start_x": sx,
                f"{action}_start_y": sy,
                f"{action}_end_x": ex,
                f"{action}_end_y": ey,
            }.items():
                if k in meta:
                    meta[k] = np.asarray([v], dtype=np.float32)
        if action == "pinch":
            if kwargs.get("pinch_start_span") is not None:
                for k in ("pinch_start_span", "pinch_start_scale"):
                    if k in meta:
                        meta[k] = np.asarray([float(kwargs["pinch_start_span"])], dtype=np.float32)
            if kwargs.get("pinch_end_span") is not None:
                for k in ("pinch_end_span", "pinch_end_scale"):
                    if k in meta:
                        meta[k] = np.asarray([float(kwargs["pinch_end_span"])], dtype=np.float32)

    def _resolve_user(self, rt: _Runtime, user_id: Optional[int]) -> int:
        users = self._fg["users_for"](rt.split, self.split).astype(np.int64)
        if user_id is None:
            return int(self.rng.choice(users))
        uid = int(user_id)
        # rt.ref_bank is mutable because explicit callers may request users from
        # train, val and test in one long-lived Android process.  Membership in
        # the configured split does not prove that the *current* bank still
        # contains this user: an earlier out-of-split call may have replaced it.
        available = rt.ref_bank.refs(uid) if rt.ref_bank is not None else np.array([], dtype=np.int64)
        if rt.k_refs > 0 and len(available) == 0:
            # Reference identity is part of the five-shot protocol and must
            # not depend on the caller seed or generation shard.  Use the same
            # config-derived seed as the bank created at model-load time.
            rt.ref_bank = self._fg["UserRefBank"](
                rt.data,
                np.asarray([int(user_id)], dtype=np.int64),
                rt.k_refs,
                rt.ref_seed,
            )
            available = rt.ref_bank.refs(uid)
        if rt.k_refs > 0 and len(available) < rt.k_refs:
            known = sorted(int(value) for value in rt.data.users)
            raise ValueError(
                "user_id=%d has %d/%d required references in %s; available user IDs=%s"
                % (uid, len(available), rt.k_refs, rt.data.path, known)
            )
        return uid

    @staticmethod
    def _seq_to_list(value: Any) -> Optional[list]:
        if value is None:
            return None
        return [float(x) for x in value]

def json_safe_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


__all__ = ["AndroidIMUDiffusionLayer", "ACTION_SPECS", "json_safe_metadata"]
