from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import __version__
from .checkpoint import CheckpointManager, resolve_checkpoint
from .critics import AdvCriticBundle, differentiable_hmog_features
from .data import (
    ActionData,
    MetadataPrior,
    Normalizer,
    UserRefBank,
    WindowDataset,
    build_target_mask,
    compute_normalizer,
    indices_for_users,
    load_action_data,
    load_split,
    loss_mask_np,
    metadata_to_torch,
    pack_fake_npz,
    refresh_condition_metadata,
    train_val_indices,
    users_for,
)
from .diffusion import DDPMSchedule, diffusion_train_step
from .model import EMA, TemporalDenoiser
from .utils import (
    ACTION_NAMES,
    append_jsonl,
    copy_file,
    count_parameters,
    default_device,
    ensure_dir,
    load_yaml,
    now_tag,
    save_json,
    set_seed,
)


def protocol_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    protocols = cfg.get("protocols", {})
    if name not in protocols:
        raise KeyError("unknown protocol %s; valid=%s" % (name, sorted(protocols.keys())))
    out = dict(protocols[name])
    return out


def apply_method_cfg(cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    methods = cfg.get("methods", {})
    if method not in methods:
        raise KeyError("unknown method %s; valid=%s" % (method, sorted(methods.keys())))
    overrides = methods[method].get("loss_overrides", {})
    for k, v in overrides.items():
        cfg.setdefault("loss", {})[k] = v
    cfg["method"] = method
    return cfg


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def apply_action_overrides(cfg: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Apply optional per-action training profile from one unified config."""
    overrides = cfg.get("action_overrides", {})
    if action not in overrides:
        return cfg
    cfg = copy.deepcopy(cfg)
    _deep_update(cfg, overrides[action])
    cfg["applied_action_override"] = action
    return cfg


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def slice_batch(batch: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    """Take one consistent ADV sub-batch across signals, refs and metadata."""
    size = min(int(n), int(batch["x"].shape[0]))
    return {k: v[:size] for k, v in batch.items()}


def adversarial_weight(
    cfg: Dict[str, Any],
    epoch: int,
    step: int,
    steps_per_epoch: int,
    start_epoch: int,
    total_epochs: int,
) -> float:
    """Linearly ramp G's GAN weight after the critic start point."""
    base = float(cfg["adv"].get("g_weight", 0.02))
    ramp_epochs = max(1.0, float(cfg["adv"].get("g_ramp_frac", 0.10)) * float(total_epochs))
    progress = (float(epoch - start_epoch) + float(step) / float(max(1, steps_per_epoch))) / ramp_epochs
    return base * float(np.clip(progress, 0.0, 1.0))


def feature_match_loss(
    fake_x: torch.Tensor,
    real_x: torch.Tensor,
    mask: torch.Tensor,
    adv_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Batch-level differentiable HMOG/rich feature distribution matching."""
    mode = str(adv_cfg.get("feature_match_mode", adv_cfg.get("feature_mode", "legacy"))).lower()
    min_scale = float(adv_cfg.get("feature_match_min_scale", 0.05))
    std_weight = float(adv_cfg.get("feature_match_std_weight", 0.50))
    skew_weight = float(adv_cfg.get("feature_match_skew_weight", 0.0))
    kurt_weight = float(adv_cfg.get("feature_match_kurt_weight", 0.0))
    corr_weight = float(adv_cfg.get("feature_match_corr_weight", 0.0))
    fr = differentiable_hmog_features(real_x.detach(), mask, mode=mode).detach()
    ff = differentiable_hmog_features(fake_x, mask, mode=mode)
    fr_mean = fr.mean(dim=0)
    ff_mean = ff.mean(dim=0)
    fr_std = fr.std(dim=0, unbiased=False).clamp_min(min_scale)
    ff_std = ff.std(dim=0, unbiased=False).clamp_min(min_scale)
    mean_loss = ((ff_mean - fr_mean) / fr_std).abs().mean()
    std_loss = ((ff_std - fr_std) / fr_std).abs().mean()
    loss = mean_loss + std_weight * std_loss

    skew_loss = loss.new_zeros(())
    kurt_loss = loss.new_zeros(())
    corr_loss = loss.new_zeros(())
    if (skew_weight > 0.0 or kurt_weight > 0.0 or corr_weight > 0.0) and int(fr.shape[0]) >= 4:
        z_real = (fr - fr_mean.unsqueeze(0)) / fr_std.unsqueeze(0)
        z_fake = (ff - ff_mean.unsqueeze(0)) / ff_std.unsqueeze(0)
        if skew_weight > 0.0:
            real_skew = z_real.pow(3).mean(dim=0)
            fake_skew = z_fake.pow(3).mean(dim=0)
            skew_loss = (fake_skew - real_skew).abs().mean()
            loss = loss + skew_weight * skew_loss
        if kurt_weight > 0.0:
            real_kurt = z_real.pow(4).mean(dim=0) - 3.0
            fake_kurt = z_fake.pow(4).mean(dim=0) - 3.0
            kurt_loss = (fake_kurt - real_kurt).abs().mean()
            loss = loss + kurt_weight * kurt_loss
        if corr_weight > 0.0:
            # Correlation/covariance structure matters to tree detectors but is
            # not constrained by per-feature mean/std matching.  Use each
            # batch's standardized feature vectors and compare off-diagonal
            # correlations with a small weight because the estimate is noisy
            # for the ADV sub-batch size.
            denom = float(max(1, int(fr.shape[0]) - 1))
            real_corr = torch.matmul(z_real.transpose(0, 1), z_real) / denom
            fake_corr = torch.matmul(z_fake.transpose(0, 1), z_fake) / denom
            eye = torch.eye(real_corr.shape[0], device=real_corr.device, dtype=torch.bool)
            corr_loss = (fake_corr[~eye] - real_corr[~eye]).abs().mean()
            loss = loss + corr_weight * corr_loss

    return loss, {
        "adv_feature_match": float(loss.detach().cpu()),
        "adv_feature_match_mean": float(mean_loss.detach().cpu()),
        "adv_feature_match_std": float(std_loss.detach().cpu()),
        "adv_feature_match_skew": float(skew_loss.detach().cpu()),
        "adv_feature_match_kurt": float(kurt_loss.detach().cpu()),
        "adv_feature_match_corr": float(corr_loss.detach().cpu()),
    }


def backward_with_projected_adversarial_gradients(
    model: torch.nn.Module,
    reconstruction_loss: torch.Tensor,
    adversarial_loss: torch.Tensor,
    adversarial_weight_value: float,
    max_grad_ratio: float,
    project_conflicts: bool = True,
) -> Dict[str, float]:
    """Backpropagate reconstruction first, then safely merge GAN gradients."""
    params = [p for p in model.parameters() if p.requires_grad]
    reconstruction_loss.backward()
    adv_grads = torch.autograd.grad(
        adversarial_loss,
        params,
        allow_unused=True,
    )

    base_sq = reconstruction_loss.new_zeros(())
    adv_sq = reconstruction_loss.new_zeros(())
    dot = reconstruction_loss.new_zeros(())
    for p, adv_grad in zip(params, adv_grads):
        if p.grad is not None:
            base_sq = base_sq + p.grad.detach().float().pow(2).sum()
        if adv_grad is not None:
            adv_detached = adv_grad.detach().float()
            adv_sq = adv_sq + adv_detached.pow(2).sum()
            if p.grad is not None:
                dot = dot + (p.grad.detach().float() * adv_detached).sum()

    eps = torch.finfo(torch.float32).eps
    base_norm = base_sq.sqrt()
    raw_adv_norm = adv_sq.sqrt()
    cosine = dot / (base_norm * raw_adv_norm).clamp_min(eps)
    projection_coeff = reconstruction_loss.new_zeros(())
    if bool(project_conflicts) and float(dot.detach().cpu()) < 0.0 and float(base_sq.detach().cpu()) > 0.0:
        projection_coeff = dot / base_sq.clamp_min(eps)

    projected: List[Optional[torch.Tensor]] = []
    projected_sq = reconstruction_loss.new_zeros(())
    for p, adv_grad in zip(params, adv_grads):
        if adv_grad is None:
            projected.append(None)
            continue
        grad = adv_grad.detach()
        if float(projection_coeff.detach().cpu()) != 0.0 and p.grad is not None:
            grad = grad - projection_coeff.to(dtype=grad.dtype) * p.grad.detach()
        projected.append(grad)
        projected_sq = projected_sq + grad.float().pow(2).sum()

    projected_norm = projected_sq.sqrt()
    weighted_norm = abs(float(adversarial_weight_value)) * projected_norm
    allowed_norm = max(0.0, float(max_grad_ratio)) * base_norm
    if float(weighted_norm.detach().cpu()) > 0.0:
        merge_scale = torch.clamp(allowed_norm / weighted_norm.clamp_min(eps), max=1.0)
    else:
        merge_scale = weighted_norm.new_zeros(())
    applied_weight = float(adversarial_weight_value) * float(merge_scale.detach().cpu())

    for p, adv_grad in zip(params, projected):
        if adv_grad is None or applied_weight == 0.0:
            continue
        contribution = adv_grad.to(dtype=p.dtype) * applied_weight
        if p.grad is None:
            p.grad = contribution.clone()
        else:
            p.grad.add_(contribution)

    applied_norm = abs(applied_weight) * projected_norm
    applied_ratio = applied_norm / base_norm.clamp_min(eps)
    return {
        "adv_grad_base_norm": float(base_norm.detach().cpu()),
        "adv_grad_raw_norm": float(raw_adv_norm.detach().cpu()),
        "adv_grad_projected_norm": float(projected_norm.detach().cpu()),
        "adv_grad_conflict_cosine": float(cosine.detach().cpu()),
        "adv_grad_merge_scale": float(merge_scale.detach().cpu()),
        "adv_grad_applied_ratio": float(applied_ratio.detach().cpu()),
    }


def build_model(cfg: Dict[str, Any], data: ActionData, protocol: Dict[str, Any]) -> TemporalDenoiser:
    m = cfg["model"]
    return TemporalDenoiser(
        T=data.T,
        in_channels=data.C,
        base_channels=int(m.get("base_channels", 96)),
        cond_dim=int(m.get("cond_dim", 192)),
        n_blocks=int(m.get("n_blocks", 8)),
        dropout=float(m.get("dropout", 0.05)),
        time_embed_dim=int(m.get("time_embed_dim", 128)),
        use_ref_context=bool(m.get("use_ref_context", True)),
        ref_encoder_channels=int(m.get("ref_encoder_channels", 64)),
    )


def make_schedule(cfg: Dict[str, Any], device: torch.device) -> DDPMSchedule:
    d = cfg["diffusion"]
    return DDPMSchedule(
        train_steps=int(d.get("train_steps", 1000)),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
        device=device,
    )


def make_loaders(
    cfg: Dict[str, Any],
    data: ActionData,
    split: Dict[str, Any],
    normalizer: Normalizer,
    protocol: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, Optional[UserRefBank]]:
    train_users = users_for(split, "train")
    train_idx, val_idx = train_val_indices(
        data,
        train_users=train_users,
        val_ratio=float(cfg["data"].get("train_val_ratio_within_train_users", 0.08)),
        seed=int(cfg["runtime"].get("seed", 42)) + 11,
    )
    fewshot = bool(protocol.get("fewshot", False))
    k_refs = int(protocol.get("k_refs", 0)) if fewshot else 0
    ref_bank = (
        UserRefBank(
            data,
            train_users,
            k_refs,
            int(cfg["runtime"].get("seed", 42)) + 101,
            candidate_indices=train_idx,
        )
        if fewshot
        else None
    )
    ds_train = WindowDataset(
        data,
        train_idx,
        normalizer,
        fewshot=fewshot,
        k_refs=k_refs,
        ref_bank=ref_bank,
        seed=int(cfg["runtime"].get("seed", 42)) + 1000,
    )
    ds_val = WindowDataset(
        data,
        val_idx,
        normalizer,
        fewshot=fewshot,
        k_refs=k_refs,
        ref_bank=ref_bank,
        seed=int(cfg["runtime"].get("seed", 42)) + 2000,
    )
    loader_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg["runtime"].get("num_workers", 2)),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["runtime"].get("num_workers", 2)),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader_train, loader_val, train_idx, val_idx, ref_bank


@torch.no_grad()
def quick_eval(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    adv_bundle: Optional[AdvCriticBundle] = None,
) -> Dict[str, float]:
    model.eval()
    if adv_bundle is not None:
        adv_bundle.eval()
    losses: List[float] = []
    feature_l1: List[float] = []
    adv_metrics: Dict[str, float] = {}
    max_batches = int(cfg["train"].get("quick_eval_batches", 8))
    sample_steps = min(int(cfg["diffusion"].get("sample_steps", 80)), 24)
    clamp_std = float(cfg["sample"].get("clamp_std", 8.0))
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = move_batch(batch, device)
        loss, metrics, _ = diffusion_train_step(model, schedule, batch, cfg["loss"])
        losses.append(float(loss.detach().cpu()))
        if bi == 0:
            B, T, C = batch["x"].shape
            fake = schedule.sample(model, (B, T, C), batch, sample_steps=sample_steps, clamp_std=clamp_std)
            fr = differentiable_hmog_features(batch["x"], batch["loss_mask"])
            ff = differentiable_hmog_features(fake, batch["loss_mask"])
            feature_l1.append(float((fr.mean(dim=0) - ff.mean(dim=0)).abs().mean().detach().cpu()))
            if adv_bundle is not None:
                adv_metrics = adv_bundle.evaluate(batch["x"], fake, batch, cfg["adv"])
    if not losses:
        return {"val_loss": float("inf"), "feature_l1": float("inf")}
    out = {
        "val_loss": float(np.mean(losses)),
        "feature_l1": float(np.mean(feature_l1)) if feature_l1 else 0.0,
        "quick_metric": float(np.mean(losses) + (np.mean(feature_l1) if feature_l1 else 0.0)),
    }
    out.update(adv_metrics)
    if adv_bundle is not None:
        out["adv_selection_metric"] = float(
            out["quick_metric"]
            + float(cfg["adv"].get("selection_adv_weight", 0.10)) * float(out.get("adv_g_eval", 0.0))
        )
    return out


def train_cmd(args: argparse.Namespace) -> None:
    cfg = apply_action_overrides(apply_method_cfg(load_yaml(args.config), args.method), args.action)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = float(args.weight_decay)
    if args.log_every_steps is not None:
        cfg["train"]["log_every_steps"] = int(args.log_every_steps)
    if args.quick_eval_batches is not None:
        cfg["train"]["quick_eval_batches"] = int(args.quick_eval_batches)
    if args.adv_lr is not None:
        cfg["adv"]["lr"] = float(args.adv_lr)
    if args.adv_g_weight is not None:
        cfg["adv"]["g_weight"] = float(args.adv_g_weight)
    if args.adv_max_grad_ratio is not None:
        cfg["adv"]["max_grad_ratio"] = float(args.adv_max_grad_ratio)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    if args.action not in ACTION_NAMES:
        raise ValueError("unknown action %s" % args.action)
    proto = protocol_cfg(cfg, args.protocol)
    if args.adv is not None:
        proto["adv"] = bool(args.adv)
    cfg["adv"]["enabled"] = bool(proto.get("adv", False))

    data = load_action_data(cfg["paths"]["data_dir"], args.action)
    split = load_split(cfg["paths"]["split_file"])
    train_users = users_for(split, "train")
    normalizer = compute_normalizer(data, train_users)
    loader_train, loader_val, train_idx, val_idx, _ = make_loaders(cfg, data, split, normalizer, proto)

    run_name = args.run_name or ("%s_%s_%s_%s" % (args.action, args.method, args.protocol, now_tag()))
    run_dir = Path(cfg["paths"]["output_root"]) / args.action / args.method / args.protocol / run_name
    ensure_dir(run_dir)
    copy_file(Path(args.config), run_dir / "config.yaml")
    save_json(run_dir / "effective_config.json", cfg)
    save_json(
        run_dir / "run_info.json",
        {
            "version": __version__,
            "action": args.action,
            "method": args.method,
            "protocol": args.protocol,
            "run_dir": str(run_dir),
            "data_path": str(data.path),
            "n_windows": data.n,
            "T": data.T,
            "hz": data.hz,
            "train_idx": int(len(train_idx)),
            "val_idx": int(len(val_idx)),
            "adv_enabled": bool(cfg["adv"]["enabled"]),
            "fewshot": bool(proto.get("fewshot", False)),
            "k_refs": int(proto.get("k_refs", 0)),
        },
    )

    device = default_device(str(cfg["runtime"].get("device", "auto")))
    model = build_model(cfg, data, proto).to(device)
    schedule = make_schedule(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-6)),
    )
    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.999)))
    adv_bundle: Optional[AdvCriticBundle] = None
    adv_optimizer: Optional[torch.optim.Optimizer] = None
    if bool(cfg["adv"].get("enabled", False)):
        adv_bundle = AdvCriticBundle(cfg["adv"]).to(device)
        adv_optimizer = torch.optim.AdamW(
            adv_bundle.parameters(),
            lr=float(cfg["adv"].get("lr", 1e-4)),
            weight_decay=float(cfg["adv"].get("weight_decay", 1e-6)),
        )

    start_epoch = 1
    global_step = 0
    if args.resume:
        ck = torch.load(str(resolve_checkpoint(Path(args.resume_run_dir or run_dir), args.resume)), map_location=device)
        old_version = (ck.get("config") or {}).get("version")
        if (
            adv_bundle is not None
            and old_version is not None
            and str(old_version) != str(cfg.get("version"))
            and not bool(args.skip_adv_state)
        ):
            raise ValueError(
                "cannot resume ADV checkpoint version=%s with rewritten critic version=%s; "
                "start a new ADV run so critic/optimizer states are coherent"
                % (old_version, cfg.get("version"))
            )
        model.load_state_dict(ck["model_state"])
        if not bool(args.reset_optimizer):
            optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("ema_state") is not None:
            ema.load_state_dict(ck["ema_state"])
        if adv_bundle is not None and ck.get("adv_state") is not None and not bool(args.skip_adv_state):
            adv_bundle.load_state_dict(ck["adv_state"])
        if (
            adv_optimizer is not None
            and ck.get("adv_optimizer_state") is not None
            and not bool(args.reset_optimizer)
            and not bool(args.skip_adv_state)
        ):
            adv_optimizer.load_state_dict(ck["adv_optimizer_state"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        global_step = int(ck.get("global_step", 0))

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "action": args.action,
                "method": args.method,
                "protocol": args.protocol,
                "device": str(device),
                "model_params": count_parameters(model),
                "adv_params": count_parameters(adv_bundle) if adv_bundle is not None else 0,
                "train_batches": len(loader_train),
                "val_batches": len(loader_val),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    epochs = int(cfg["train"].get("epochs", 80))
    eval_epochs = sorted(set(max(1, int(round(float(f) * epochs))) for f in cfg["train"].get("quick_eval_fracs", [0.2, 0.4, 0.6, 0.8, 1.0])))
    ckpt = CheckpointManager(run_dir, minimize_metric=True)
    log_path = run_dir / "train_log.jsonl"
    quick_path = run_dir / "quick_eval.jsonl"
    adv_start_epoch = max(1, int(round(float(cfg["adv"].get("start_frac", 0.25)) * epochs)))
    adv_sample_steps = max(2, int(cfg["adv"].get("sample_steps", 8)))
    adv_batch_size = max(1, int(cfg["adv"].get("batch_size", 16)))
    adv_update_every = max(1, int(cfg["adv"].get("update_every", 4)))
    r1_every = max(1, int(cfg["adv"].get("r1_every", 16)))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if adv_bundle is not None:
            adv_bundle.train()
        running: Dict[str, List[float]] = {}
        for step, batch in enumerate(loader_train, start=1):
            batch = move_batch(batch, device)
            adv_epoch_active = adv_bundle is not None and adv_optimizer is not None and epoch >= adv_start_epoch
            adv_update = adv_epoch_active and (global_step % adv_update_every == 0)
            adv_batch: Optional[Dict[str, torch.Tensor]] = None
            fake_adv: Optional[torch.Tensor] = None

            if adv_update and adv_bundle is not None and adv_optimizer is not None:
                adv_batch = slice_batch(batch, adv_batch_size)
                B_adv, T_adv, C_adv = adv_batch["x"].shape
                fake_adv = schedule.sample_differentiable(
                    model,
                    (B_adv, T_adv, C_adv),
                    adv_batch,
                    sample_steps=adv_sample_steps,
                    clamp_std=float(cfg["sample"].get("clamp_std", 8.0)),
                )
                for d_index in range(int(cfg["adv"].get("d_steps", 1))):
                    apply_r1 = (global_step % r1_every == 0) and d_index == 0
                    d_loss, d_logs = adv_bundle.discriminator_loss(
                        adv_batch["x"],
                        fake_adv.detach(),
                        adv_batch,
                        cfg["adv"],
                        apply_r1=apply_r1,
                    )
                    adv_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(adv_bundle.parameters(), float(cfg["adv"].get("grad_clip", 1.0)))
                    adv_optimizer.step()
                    for k, v in d_logs.items():
                        running.setdefault(k, []).append(float(v))

            loss, metrics, _ = diffusion_train_step(model, schedule, batch, cfg["loss"])
            total = loss
            side_loss: Optional[torch.Tensor] = None
            if adv_update and adv_bundle is not None and adv_batch is not None and fake_adv is not None:
                adv_bundle.set_requires_grad(adv_bundle, False)
                g_loss, g_logs = adv_bundle.generator_loss(fake_adv, adv_batch["x"], adv_batch, cfg["adv"])
                adv_bundle.set_requires_grad(adv_bundle, True)
                current_adv_weight = adversarial_weight(cfg, epoch, step, len(loader_train), adv_start_epoch, epochs)
                weighted_g_loss = current_adv_weight * g_loss
                total = total + weighted_g_loss
                side_loss = weighted_g_loss
                running.setdefault("adv_g_weight", []).append(float(current_adv_weight))
                running.setdefault("adv_sample_steps", []).append(float(adv_sample_steps))
                for k, v in g_logs.items():
                    running.setdefault(k, []).append(float(v))
                feature_match_weight = float(cfg["adv"].get("feature_match_weight", 0.0))
                if feature_match_weight > 0.0:
                    fm_loss, fm_logs = feature_match_loss(
                        fake_adv,
                        adv_batch["x"],
                        adv_batch["loss_mask"],
                        cfg["adv"],
                    )
                    weighted_fm_loss = feature_match_weight * fm_loss
                    total = total + weighted_fm_loss
                    side_loss = weighted_fm_loss if side_loss is None else side_loss + weighted_fm_loss
                    running.setdefault("adv_feature_match_weight", []).append(float(feature_match_weight))
                    for k, v in fm_logs.items():
                        running.setdefault(k, []).append(float(v))

            optimizer.zero_grad(set_to_none=True)
            if side_loss is not None:
                grad_logs = backward_with_projected_adversarial_gradients(
                    model=model,
                    reconstruction_loss=loss,
                    adversarial_loss=side_loss,
                    adversarial_weight_value=1.0,
                    max_grad_ratio=float(cfg["adv"].get("max_grad_ratio", 0.25)),
                    project_conflicts=bool(cfg["adv"].get("project_conflicts", True)),
                )
                for k, v in grad_logs.items():
                    running.setdefault(k, []).append(float(v))
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            optimizer.step()
            ema.update(model)
            global_step += 1

            running.setdefault("train_total", []).append(float(total.detach().cpu()))
            for k, v in metrics.items():
                running.setdefault(k, []).append(float(v))
            if step % int(cfg["train"].get("log_every_steps", 100)) == 0:
                row = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "phase": "train",
                }
                row.update({k: float(np.mean(v[-50:])) for k, v in running.items()})
                append_jsonl(log_path, row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            if args.max_steps is not None and step >= int(args.max_steps):
                break

        epoch_metrics = {k: float(np.mean(v)) for k, v in running.items() if v}
        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "phase": "epoch_train",
        }
        epoch_row.update(epoch_metrics)
        append_jsonl(log_path, epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)
        ckpt.save_last(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            ema=ema,
            cfg=cfg,
            action=args.action,
            protocol=args.protocol,
            normalizer=normalizer,
            metrics=epoch_metrics,
            adv=adv_bundle,
            adv_optimizer=adv_optimizer,
        )

        if bool(cfg["train"].get("save_epoch_checkpoints", True)) and epoch in eval_epochs:
            ckpt.save_epoch(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                ema=ema,
                cfg=cfg,
                action=args.action,
                protocol=args.protocol,
                normalizer=normalizer,
                metrics=epoch_metrics,
                adv=adv_bundle,
                adv_optimizer=adv_optimizer,
            )

        if epoch in eval_epochs:
            backup = ema.copy_to(model)
            eval_critics = adv_bundle if adv_bundle is not None and epoch >= adv_start_epoch else None
            q = quick_eval(model, schedule, loader_val, cfg, device, adv_bundle=eval_critics)
            ema.restore(model, backup)
            q.update({"epoch": epoch, "global_step": global_step, "progress": float(epoch) / float(epochs)})
            append_jsonl(quick_path, q)
            best_path = ckpt.save_best_if_needed(
                metric=float(q["quick_metric"]),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                ema=ema,
                cfg=cfg,
                action=args.action,
                protocol=args.protocol,
                normalizer=normalizer,
                metrics=q,
                adv=adv_bundle,
                adv_optimizer=adv_optimizer,
            )
            if best_path is not None:
                q["new_best"] = str(best_path)
            if eval_critics is not None:
                post_adv_path = ckpt.save_best_post_adv_if_needed(
                    metric=float(q["adv_selection_metric"]),
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    cfg=cfg,
                    action=args.action,
                    protocol=args.protocol,
                    normalizer=normalizer,
                    metrics=q,
                    adv=adv_bundle,
                    adv_optimizer=adv_optimizer,
                )
                if post_adv_path is not None:
                    q["new_best_post_adv"] = str(post_adv_path)
            print(json.dumps(q, ensure_ascii=False, sort_keys=True), flush=True)

    print(json.dumps({"done": True, "run_dir": str(run_dir), "last": str(run_dir / "checkpoints" / "last.pt")}, ensure_ascii=False), flush=True)


def _build_refs_for_sampling(
    data: ActionData,
    normalizer: Normalizer,
    user_id: int,
    fewshot: bool,
    k_refs: int,
    ref_bank: Optional[UserRefBank],
    batch_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    T, C = data.T, data.C
    max_refs = max(1, int(k_refs))
    refs = np.zeros((batch_n, max_refs, T, C), dtype=np.float32)
    ref_mask = np.zeros((batch_n, max_refs, T), dtype=np.float32)
    used_width = max_refs if fewshot and k_refs > 0 else 0
    used = np.full((batch_n, used_width), -1, dtype=np.int64)
    audit_width = int(ref_bank.k_refs) if ref_bank is not None else 0
    audit = np.full((batch_n, audit_width), -1, dtype=np.int64)
    count = np.zeros((batch_n,), dtype=np.int64)
    if ref_bank is None:
        return torch.from_numpy(refs), torch.from_numpy(ref_mask), torch.from_numpy(count), audit, used
    available = ref_bank.refs(user_id)
    if len(available) and audit_width > 0:
        audit[:, : min(len(available), audit_width)] = available[:audit_width].reshape(1, -1)
    if fewshot and k_refs > 0 and len(available):
        take = min(int(k_refs), len(available))
        sel = available[:take]
        valid = data.arrays["valid_mask"][sel].astype(np.float32)
        win = normalizer.normalize_np(data.arrays["windows"][sel]) * valid[:, :, None]
        lm = loss_mask_np(data.arrays["mask"][sel], data.arrays["valid_mask"][sel])
        refs[:, :take] = win.reshape(1, take, T, C)
        ref_mask[:, :take] = lm.reshape(1, take, T)
        count[:] = take
        used[:, :take] = sel.reshape(1, -1)
    return torch.from_numpy(refs), torch.from_numpy(ref_mask), torch.from_numpy(count), audit, used


def _max_active_len_for_action(data: ActionData) -> int:
    return max(1, int(data.T) - int(data.pad_pre_pts))


def apply_sample_length_override(
    meta: Dict[str, np.ndarray],
    args: argparse.Namespace,
    data: ActionData,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Optionally override sampled active_len while keeping max-T windows."""
    mode = str(getattr(args, "active_len_mode", "prior") or "prior").lower()
    duration_ms = getattr(args, "duration_ms", None)
    active_len_arg = getattr(args, "active_len", None)
    max_active = _max_active_len_for_action(data)
    if duration_ms is not None:
        duration_value = float(duration_ms)
        if not np.isfinite(duration_value) or duration_value <= 0:
            raise ValueError("--duration-ms must be finite and > 0")
        mode = "fixed"
        frames = duration_value / 1000.0 * float(data.hz)
        derived_active_len = int(round(frames))
        derived_active_len = max(1, derived_active_len)
        if active_len_arg is not None and int(active_len_arg) != min(derived_active_len, max_active):
            raise ValueError(
                "--active-len=%d contradicts --duration-ms=%.9g for action=%s; expected %d"
                % (int(active_len_arg), duration_value, data.action, min(derived_active_len, max_active))
            )
        active_len_arg = derived_active_len
    if mode == "prior" and active_len_arg is None:
        return meta

    n = int(meta["active_len"].shape[0])
    if mode == "fixed":
        if active_len_arg is None:
            raise ValueError("--active-len-mode fixed requires --active-len or --duration-ms")
        if int(active_len_arg) < 1:
            raise ValueError("--active-len must be >= 1")
        active = np.full((n,), int(active_len_arg), dtype=np.int64)
    elif mode == "uniform":
        lo = getattr(args, "min_active_len", None)
        hi = getattr(args, "max_active_len", None)
        lo_i = int(lo) if lo is not None else 1
        hi_i = int(hi) if hi is not None else max_active
        lo_i = max(1, min(lo_i, max_active))
        hi_i = max(lo_i, min(hi_i, max_active))
        active = rng.integers(lo_i, hi_i + 1, size=n, dtype=np.int64)
    else:
        raise ValueError("unknown active_len_mode=%r; expected prior/fixed/uniform" % mode)

    requested_active = active.astype(np.int64).copy()
    active = np.clip(requested_active, 1, max_active)
    mask, valid_mask = build_target_mask(data.T, data.pad_pre_pts, active)
    out = dict(meta)
    out["active_len"] = active.astype(np.int64)
    out["mask"] = mask.astype(np.uint8)
    out["valid_mask"] = valid_mask.astype(np.uint8)
    if duration_ms is not None:
        out["duration_ms"] = np.full((n,), float(duration_ms), dtype=np.float32)
    else:
        # An explicit/uniform frame-count override defines its own logical
        # sample time; retaining the unrelated prior duration would create an
        # internally contradictory schema-v2 row.
        logical_active = requested_active if mode == "fixed" else active
        out["duration_ms"] = (
            logical_active.astype(np.float64) * 1000.0 / float(data.hz)
        ).astype(np.float32)
    return out


@torch.no_grad()
def sample_cmd(args: argparse.Namespace) -> None:
    cfg = apply_action_overrides(apply_method_cfg(load_yaml(args.config), args.method), args.action)
    proto = protocol_cfg(cfg, args.protocol)
    data = load_action_data(cfg["paths"]["data_dir"], args.action)
    split = load_split(cfg["paths"]["split_file"])
    selector = args.checkpoint or str(cfg["sample"].get("checkpoint", "auto"))
    if selector == "auto":
        selector = "best_post_adv" if bool(proto.get("adv", False)) else "best"
    try:
        ckpt_path = resolve_checkpoint(Path(args.run_dir), selector)
    except FileNotFoundError:
        if selector != "best_post_adv":
            raise
        # Compatibility for completed v1.1 ADV runs, which predate the separate
        # post-ADV manifest.  Never silently fall back to the pre-ADV "best".
        ckpt_path = resolve_checkpoint(Path(args.run_dir), "last")
        print("[warning] best_post_adv is unavailable; using last.pt", flush=True)
    device = default_device(str(cfg["runtime"].get("device", "auto")))
    state = torch.load(str(ckpt_path), map_location=device)
    ck_action = state.get("action")
    ck_protocol = state.get("protocol")
    ck_method = (state.get("config") or {}).get("method")
    ck_version = (state.get("config") or {}).get("version")
    if ck_action is not None and str(ck_action) != str(args.action):
        raise ValueError("checkpoint action=%s but requested action=%s" % (ck_action, args.action))
    if ck_protocol is not None and str(ck_protocol) != str(args.protocol):
        raise ValueError("checkpoint protocol=%s but requested protocol=%s" % (ck_protocol, args.protocol))
    if ck_method is not None and str(ck_method) != str(args.method):
        raise ValueError("checkpoint method=%s but requested method=%s" % (ck_method, args.method))
    if ck_version and str(ck_version) != str(cfg.get("version")):
        print(
            "[warning] checkpoint version=%s differs from config version=%s; "
            "old checkpoints should not be mixed with the rewritten training protocol for formal results"
            % (ck_version, cfg.get("version")),
            flush=True,
        )
    normalizer_state = state.get("normalizer")
    if normalizer_state is None:
        raise KeyError("checkpoint has no normalizer")
    normalizer = Normalizer(
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
    model = build_model(cfg, data, proto).to(device)
    model.load_state_dict(state["model_state"])
    if bool(args.use_ema) and state.get("ema_state") is not None:
        ms = model.state_dict()
        for k, v in state["ema_state"].items():
            if k in ms:
                ms[k].copy_(v.to(device))
    model.eval()
    schedule = make_schedule(cfg, device)

    split_name = args.split or str(cfg["sample"].get("split", "test"))
    users = users_for(split, split_name)
    prior_users = users_for(split, "train")
    prior = MetadataPrior(data, prior_users, normalizer=normalizer)
    fewshot = bool(proto.get("fewshot", False))
    k_refs = int(proto.get("k_refs", 0)) if fewshot else 0
    audit_k = int(cfg["data"].get("max_ref_per_user", 8)) if bool(cfg["data"].get("noshot_store_audit_refs", True)) else 0
    ref_k_for_bank = k_refs if fewshot else audit_k
    ref_bank = UserRefBank(data, users, ref_k_for_bank, int(cfg["runtime"].get("seed", 42)) + 303) if ref_k_for_bank > 0 else None

    per_user = int(args.per_user or cfg["sample"].get("per_user", 200))
    batch_size = int(args.batch_size or cfg["sample"].get("batch_size", 128))
    rng = np.random.default_rng(int(cfg["runtime"].get("seed", 42)) + 7000 + {"train": 1, "val": 2, "test": 3}.get(split_name, 4))
    all_windows: List[np.ndarray] = []
    all_user: List[np.ndarray] = []
    all_meta: List[Dict[str, np.ndarray]] = []
    all_ref: List[np.ndarray] = []
    all_used: List[np.ndarray] = []

    for u in users:
        remain = per_user
        while remain > 0:
            n = min(batch_size, remain)
            meta = prior.sample(n, rng)
            meta = apply_sample_length_override(meta, args, data, rng)
            meta = refresh_condition_metadata(data, meta, normalizer)
            tbatch = metadata_to_torch(meta, device)
            refs, ref_mask, ref_count, audit_refs, used_refs = _build_refs_for_sampling(
                data,
                normalizer,
                int(u),
                fewshot=fewshot,
                k_refs=k_refs,
                ref_bank=ref_bank,
                batch_n=n,
            )
            tbatch["refs"] = refs.to(device)
            tbatch["ref_mask"] = ref_mask.to(device)
            tbatch["ref_count"] = ref_count.to(device)
            x = schedule.sample(
                model,
                (n, data.T, data.C),
                tbatch,
                sample_steps=int(args.sample_steps or cfg["diffusion"].get("sample_steps", 80)),
                clamp_std=float(cfg["sample"].get("clamp_std", 8.0)),
            )
            raw = normalizer.denormalize_np(x.detach().cpu().numpy())
            all_windows.append(raw)
            all_user.append(np.full((n,), int(u), dtype=np.int64))
            all_meta.append(meta)
            all_ref.append(audit_refs)
            all_used.append(used_refs)
            remain -= n
        print(json.dumps({"sampled_user": int(u), "n": per_user}, ensure_ascii=False), flush=True)

    windows = np.concatenate(all_windows, axis=0)
    user_id = np.concatenate(all_user, axis=0)
    merged_meta: Dict[str, np.ndarray] = {}
    for k in all_meta[0].keys():
        merged_meta[k] = np.concatenate([m[k] for m in all_meta], axis=0)
    ref_indices = np.concatenate(all_ref, axis=0) if all_ref else np.full((0, 0), -1, dtype=np.int64)
    used_ref_indices = np.concatenate(all_used, axis=0) if all_used else np.full((0, 0), -1, dtype=np.int64)
    out = pack_fake_npz(
        data=data,
        windows=windows,
        meta=merged_meta,
        user_id=user_id,
        ref_indices=ref_indices,
        used_ref_indices=used_ref_indices,
        normalizer=normalizer,
        protocol=args.protocol,
        run_dir=Path(args.run_dir),
        checkpoint_path=ckpt_path,
    )
    out["split"] = np.array(split_name)
    out["method"] = np.array(args.method)
    out_path = Path(args.out) if args.out else Path(args.run_dir) / "samples" / args.action / ("%s_fake.npz" % split_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **out)
    summary = {
        "saved": str(out_path),
        "action": args.action,
        "method": args.method,
        "protocol": args.protocol,
        "split": split_name,
        "n": int(windows.shape[0]),
        "per_user": per_user,
        "users": int(len(users)),
        "checkpoint": str(ckpt_path),
        "fewshot": fewshot,
        "k_refs": k_refs,
        "active_len_mode": str(args.active_len_mode),
        "active_len": None if args.active_len is None else int(args.active_len),
        "duration_ms": None if args.duration_ms is None else float(args.duration_ms),
        "min_active_len": None if args.min_active_len is None else int(args.min_active_len),
        "max_active_len": None if args.max_active_len is None else int(args.max_active_len),
        "active_len_min": int(merged_meta["active_len"].min()),
        "active_len_max": int(merged_meta["active_len"].max()),
        "active_len_unique": int(np.unique(merged_meta["active_len"]).shape[0]),
        "mask_sum_min": int(out["mask"].sum(axis=1).min()),
        "mask_sum_max": int(out["mask"].sum(axis=1).max()),
        "mask_sum_unique": int(np.unique(out["mask"].sum(axis=1)).shape[0]),
    }
    save_json(out_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def inspect_cmd(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    rows = []
    for action in ACTION_NAMES:
        data = load_action_data(cfg["paths"]["data_dir"], action)
        a = data.arrays
        row = {
            "action": action,
            "N": data.n,
            "T": data.T,
            "hz": data.hz,
            "has_orientation": "orientation_id" in a,
            "has_valid_mask": "valid_mask" in a,
            "orientation_values": sorted([int(x) for x in np.unique(a["orientation_id"])]),
            "active_min": int(a["active_len"].min()),
            "active_max": int(a["active_len"].max()),
        }
        rows.append(row)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final standalone HMOG action generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--action", required=True, choices=ACTION_NAMES)
    tr.add_argument("--method", default="new", choices=("diffusion", "new"))
    tr.add_argument("--protocol", required=True, choices=("noshot", "noshot_adv", "fewshot", "fewshot_adv"))
    tr.add_argument("--run-name", default=None)
    tr.add_argument("--resume", default=None, help="checkpoint selector/path, e.g. last")
    tr.add_argument("--resume-run-dir", default=None)
    tr.add_argument("--adv", default=None, type=lambda x: str(x).lower() in ("1", "true", "yes"))
    tr.add_argument("--epochs", default=None, type=int)
    tr.add_argument("--batch-size", default=None, type=int)
    tr.add_argument("--lr", default=None, type=float)
    tr.add_argument("--weight-decay", default=None, type=float)
    tr.add_argument("--log-every-steps", default=None, type=int)
    tr.add_argument("--quick-eval-batches", default=None, type=int)
    tr.add_argument("--adv-lr", default=None, type=float)
    tr.add_argument("--adv-g-weight", default=None, type=float)
    tr.add_argument("--adv-max-grad-ratio", default=None, type=float)
    tr.add_argument("--reset-optimizer", action="store_true", help="resume weights/EMA/critics but start fresh optimizers")
    tr.add_argument("--skip-adv-state", action="store_true", help="resume generator/EMA weights but reinitialize ADV critics and ADV optimizer")
    tr.add_argument("--max-steps", default=None, type=int, help="debug/smoke: maximum train steps per epoch")
    tr.set_defaults(func=train_cmd)

    sp = sub.add_parser("sample")
    sp.add_argument("--config", required=True)
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--action", required=True, choices=ACTION_NAMES)
    sp.add_argument("--method", default="new", choices=("diffusion", "new"))
    sp.add_argument("--protocol", required=True, choices=("noshot", "noshot_adv", "fewshot", "fewshot_adv"))
    sp.add_argument("--split", default=None, choices=("train", "val", "test"))
    sp.add_argument("--per-user", default=None, type=int)
    sp.add_argument("--batch-size", default=None, type=int)
    sp.add_argument("--sample-steps", default=None, type=int)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--use-ema", default=True, type=lambda x: str(x).lower() not in ("0", "false", "no"))
    sp.add_argument(
        "--active-len-mode",
        default="prior",
        choices=("prior", "fixed", "uniform"),
        help="duration/mask source: prior keeps sampled metadata; fixed/uniform override active_len within T_max",
    )
    sp.add_argument("--active-len", default=None, type=int, help="fixed active length in samples")
    sp.add_argument("--duration-ms", default=None, type=float, help="fixed active duration; converted using action hz")
    sp.add_argument("--min-active-len", default=None, type=int, help="uniform active length lower bound")
    sp.add_argument("--max-active-len", default=None, type=int, help="uniform active length upper bound")
    sp.set_defaults(func=sample_cmd)

    ins = sub.add_parser("inspect-data")
    ins.add_argument("--config", required=True)
    ins.set_defaults(func=inspect_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
