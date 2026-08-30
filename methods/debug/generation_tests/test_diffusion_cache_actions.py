from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from imu.android_imu_layer.diffusion_generator import (
    ACTION_SPECS,
    AndroidIMUDiffusionLayer,
)
from imu.diffusion_model.model import TemporalDenoiser
from imu.diffusion_model.utils import ACTION_NAMES
from imu.scripts.generate_user_cache import ACTIONS as CACHE_ACTIONS


GESTURE_ACTIONS = {"tap", "scroll", "swipe", "pinch"}


def test_diffusion_layer_and_cache_expose_only_gesture_actions() -> None:
    assert set(ACTION_SPECS) == GESTURE_ACTIONS
    assert set(ACTION_NAMES) == GESTURE_ACTIONS
    assert set(CACHE_ACTIONS) == GESTURE_ACTIONS
    assert not hasattr(AndroidIMUDiffusionLayer, "type_text")


def test_packaged_gesture_checkpoints_strict_load() -> None:
    for spec in ACTION_SPECS.values():
        cfg = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
        model_cfg = cfg["model"]
        model = TemporalDenoiser(
            T=16,
            in_channels=6,
            base_channels=int(model_cfg.get("base_channels", 96)),
            cond_dim=int(model_cfg.get("cond_dim", 192)),
            n_blocks=int(model_cfg.get("n_blocks", 8)),
            dropout=float(model_cfg.get("dropout", 0.05)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            use_ref_context=bool(model_cfg.get("use_ref_context", True)),
            ref_encoder_channels=int(model_cfg.get("ref_encoder_channels", 64)),
        )
        checkpoint = torch.load(spec["run_dir"] + "/" + spec["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)


def test_packaged_tap_checkpoint_runs_with_released_references(monkeypatch) -> None:
    monkeypatch.delenv("ACTREAL_GENERATOR_DATA_ROOT", raising=False)
    monkeypatch.delenv("ACTREAL_SPLIT_FILE", raising=False)
    layer = AndroidIMUDiffusionLayer(
        seed=42,
        device="cpu",
        reference_device="pixel10",
    )
    sample = layer.tap(
        540.0,
        960.0,
        user_id=0,
        duration_ms=100.0,
        sample_steps=1,
        noise_seed=7,
    )
    assert sample["window"].shape == (35, 6)
    assert sample["active_imu"].shape == (10, 6)
    assert sample["metadata"]["ref_count"] == 5
    assert sample["metadata"]["used_ref_indices"] == [0, 1, 2, 3, 4]
    assert sample["metadata"]["reference_data_source"] == "released_on_device_fewshot"
    assert sample["metadata"]["reference_device"] == "pixel10"
    assert sample["metadata"]["reference_participant_id"] == "P00"


@pytest.mark.parametrize(
    "variable",
    ["ACTREAL_GENERATOR_DATA_ROOT", "ACTREAL_SPLIT_FILE"],
)
def test_explicit_missing_generator_input_does_not_fall_back(
    monkeypatch, tmp_path: Path, variable: str
) -> None:
    monkeypatch.delenv("ACTREAL_GENERATOR_DATA_ROOT", raising=False)
    monkeypatch.delenv("ACTREAL_SPLIT_FILE", raising=False)
    monkeypatch.setenv(variable, str(tmp_path / "does-not-exist"))
    layer = AndroidIMUDiffusionLayer(seed=42, device="cpu")
    with pytest.raises(FileNotFoundError, match="explicit ACTREAL_GENERATOR_DATA_ROOT"):
        layer.tap(
            540.0,
            960.0,
            user_id=0,
            duration_ms=100.0,
            sample_steps=1,
            noise_seed=7,
        )
