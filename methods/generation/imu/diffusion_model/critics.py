from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


HMOG_FEATURE_DIM = 73
RICH_HMOG_FEATURE_DIM = 214
CONDITION_DIM = 8


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().clamp(0.0, 1.0)


def masked_moments(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channel-wise moments over valid time points."""
    m = _safe_mask(mask).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (x * m).sum(dim=1) / denom
    var = ((x - mean.unsqueeze(1)).pow(2) * m).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(1e-8))
    rms = torch.sqrt((x.pow(2) * m).sum(dim=1) / denom).clamp_min(1e-8)
    return mean, std, rms


def _legacy_differentiable_hmog_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compact differentiable local features; no external detector code is imported."""
    _, T, _ = x.shape
    mask = _safe_mask(mask)
    mean, std, rms = masked_moments(x, mask)

    if T >= 2:
        dx = x[:, 1:] - x[:, :-1]
        dm = mask[:, 1:] * mask[:, :-1]
        d_mean, d_std, d_rms = masked_moments(dx, dm)
        d_abs = (dx.abs() * dm.unsqueeze(-1)).sum(dim=1) / dm.sum(dim=1, keepdim=True).clamp_min(1.0)
    else:
        d_mean = torch.zeros_like(mean)
        d_std = torch.zeros_like(std)
        d_rms = torch.zeros_like(rms)
        d_abs = torch.zeros_like(mean)

    if T >= 3:
        ddx = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        ddm = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        dd_abs = (ddx.abs() * ddm.unsqueeze(-1)).sum(dim=1) / ddm.sum(dim=1, keepdim=True).clamp_min(1.0)
    else:
        dd_abs = torch.zeros_like(mean)

    xm = x * mask.unsqueeze(-1)
    spec = torch.fft.rfft(xm, dim=1).abs().pow(2)
    n_freq = spec.shape[1]
    bands: List[torch.Tensor] = []
    for band in range(4):
        lo = int(round(band * n_freq / 4.0))
        hi = int(round((band + 1) * n_freq / 4.0))
        hi = min(max(hi, lo + 1), n_freq)
        bands.append(torch.log1p(spec[:, lo:hi].mean(dim=1)))
    active = mask.mean(dim=1, keepdim=True)
    out = torch.cat([mean, std, rms, d_mean, d_std, d_rms, d_abs, dd_abs] + bands + [active], dim=1)
    if out.shape[1] != HMOG_FEATURE_DIM:
        raise RuntimeError("unexpected HMOG feature dimension %d" % out.shape[1])
    return out


def _masked_quantile(s: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    valid = mask > 0.5
    high = torch.finfo(s.dtype).max / 8.0
    sorted_vals = s.masked_fill(~valid, high).sort(dim=1).values
    count = valid.sum(dim=1).clamp_min(1)
    idx = ((count - 1).float() * float(q)).round().long().clamp(0, s.shape[1] - 1)
    return sorted_vals.gather(1, idx.view(-1, 1)).squeeze(1)


def _rich_signal_stats(s: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    m = _safe_mask(mask)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (s * m).sum(dim=1) / denom
    centered = s - mean.unsqueeze(1)
    var = (centered.pow(2) * m).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(1e-8))
    s_min = s.masked_fill(m <= 0.5, torch.finfo(s.dtype).max / 8.0).min(dim=1).values
    s_max = s.masked_fill(m <= 0.5, -torch.finfo(s.dtype).max / 8.0).max(dim=1).values
    q10 = _masked_quantile(s, m, 0.10)
    q25 = _masked_quantile(s, m, 0.25)
    q50 = _masked_quantile(s, m, 0.50)
    q75 = _masked_quantile(s, m, 0.75)
    q90 = _masked_quantile(s, m, 0.90)
    mad = (centered.abs() * m).sum(dim=1) / denom
    rms = torch.sqrt((s.pow(2) * m).sum(dim=1) / denom).clamp_min(1e-8)
    z = centered / std.unsqueeze(1).clamp_min(1e-6)
    skew = (z.pow(3) * m).sum(dim=1) / denom
    kurt = (z.pow(4) * m).sum(dim=1) / denom - 3.0
    if s.shape[1] >= 2:
        d = s[:, 1:] - s[:, :-1]
        dm = m[:, 1:] * m[:, :-1]
        d_denom = dm.sum(dim=1).clamp_min(1.0)
        mean_abs_diff = (d.abs() * dm).sum(dim=1) / d_denom
        d_mean = (d * dm).sum(dim=1) / d_denom
        d_var = ((d - d_mean.unsqueeze(1)).pow(2) * dm).sum(dim=1) / d_denom
        std_diff = torch.sqrt(d_var.clamp_min(1e-8))
        soft_sign = torch.tanh(3.0 * z)
        zcr = ((soft_sign[:, 1:] - soft_sign[:, :-1]).abs() * dm).sum(dim=1) / (2.0 * d_denom)
    else:
        mean_abs_diff = torch.zeros_like(mean)
        std_diff = torch.zeros_like(mean)
        zcr = torch.zeros_like(mean)
    return [
        mean,
        std,
        s_min,
        s_max,
        s_max - s_min,
        q10,
        q25,
        q50,
        q75,
        q90,
        q75 - q25,
        mad,
        rms,
        skew,
        kurt,
        zcr,
        mean_abs_diff,
        std_diff,
    ]


def _rich_freq_stats(s: torch.Tensor, mask: torch.Tensor, hz: float = 100.0) -> List[torch.Tensor]:
    m = _safe_mask(mask)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (s * m).sum(dim=1) / denom
    centered = (s - mean.unsqueeze(1)) * m
    spec = torch.fft.rfft(centered, dim=1).abs().pow(2)
    total = spec.sum(dim=1).clamp_min(1e-8)
    p = spec / total.unsqueeze(1)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    freqs = torch.fft.rfftfreq(s.shape[1], d=1.0 / float(hz)).to(device=s.device, dtype=s.dtype)
    bands: List[torch.Tensor] = []
    edges = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    for lo, hi in edges:
        sel = (freqs >= lo) & (freqs < hi)
        if bool(sel.any()):
            bands.append(torch.log1p(spec[:, sel].sum(dim=1)))
        else:
            bands.append(torch.zeros_like(total))
    high = freqs >= 20.0
    bands.append(torch.log1p(spec[:, high].sum(dim=1)) if bool(high.any()) else torch.zeros_like(total))
    # Smooth dominant-frequency proxy; avoids non-differentiable argmax while
    # matching the detector's intent of frequency-location sensitivity.
    dom_proxy = (p * freqs.view(1, -1)).sum(dim=1)
    return [torch.log1p(total), entropy] + bands + [dom_proxy]


def _masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _safe_mask(mask)
    denom = m.sum(dim=1).clamp_min(1.0)
    ma = (a * m).sum(dim=1) / denom
    mb = (b * m).sum(dim=1) / denom
    ca = a - ma.unsqueeze(1)
    cb = b - mb.unsqueeze(1)
    cov = (ca * cb * m).sum(dim=1) / denom
    va = (ca.pow(2) * m).sum(dim=1) / denom
    vb = (cb.pow(2) * m).sum(dim=1) / denom
    return cov / torch.sqrt((va * vb).clamp_min(1e-8))


def _rich_differentiable_hmog_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Richer PAD-aligned differentiable HMOG approximation.

    It mirrors the detector-side HMOG feature layout as closely as practical:
    time statistics, physical-frequency bands, and channel correlations.
    """
    mask = _safe_mask(mask)
    acc = x[:, :, :3]
    gyro = x[:, :, 3:6]
    signals = [
        acc[:, :, 0],
        acc[:, :, 1],
        acc[:, :, 2],
        gyro[:, :, 0],
        gyro[:, :, 1],
        gyro[:, :, 2],
        torch.sqrt(acc.pow(2).sum(dim=2) + 1e-8),
        torch.sqrt(gyro.pow(2).sum(dim=2) + 1e-8),
    ]
    feats: List[torch.Tensor] = []
    for s in signals:
        feats.extend(_rich_signal_stats(s, mask))
        feats.extend(_rich_freq_stats(s, mask))
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        feats.append(_masked_corr(x[:, :, a], x[:, :, b], mask))
    out = torch.stack(feats, dim=1)
    if out.shape[1] != RICH_HMOG_FEATURE_DIM:
        raise RuntimeError("unexpected rich HMOG feature dimension %d" % out.shape[1])
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def differentiable_hmog_features(x: torch.Tensor, mask: torch.Tensor, mode: str = "legacy") -> torch.Tensor:
    mode = str(mode or "legacy").lower()
    if mode == "legacy":
        return _legacy_differentiable_hmog_features(x, mask)
    if mode == "rich":
        return _rich_differentiable_hmog_features(x, mask)
    raise ValueError("unknown differentiable_hmog feature mode %r" % mode)


def critic_condition(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compact metadata condition shared by all three critics."""
    B = batch["mask"].shape[0]
    device = batch["mask"].device
    orientation = batch.get("orientation_idx", torch.zeros(B, dtype=torch.long, device=device))
    orientation = orientation.long().clamp(0, 3)
    orientation_1h = F.one_hot(orientation, num_classes=4).float()
    active_fraction = _safe_mask(batch["mask"]).mean(dim=1, keepdim=True)
    valid_fraction = _safe_mask(batch["valid_mask"]).mean(dim=1, keepdim=True)
    # Keep the released critic input width while gesture-only models reserve
    # the final two scalar positions at zero.
    reserved = torch.zeros((B, 2), dtype=torch.float32, device=device)
    return torch.cat([orientation_1h, active_fraction, valid_fraction, reserved], dim=1)


def _sn_linear(in_features: int, out_features: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Linear(in_features, out_features))


def _sn_conv(in_channels: int, out_channels: int, kernel_size: int, **kwargs: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs))


class FeatureCritic(nn.Module):
    """Discriminator over differentiable HMOG-like summary statistics."""

    def __init__(self, hidden: int = 192, feature_mode: str = "legacy"):
        super().__init__()
        self.feature_mode = str(feature_mode or "legacy").lower()
        feature_dim = RICH_HMOG_FEATURE_DIM if self.feature_mode == "rich" else HMOG_FEATURE_DIM
        self.net = nn.Sequential(
            _sn_linear(feature_dim + CONDITION_DIM, hidden),
            nn.LeakyReLU(0.2),
            _sn_linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            _sn_linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = differentiable_hmog_features(x, mask, self.feature_mode)
        return self.net(torch.cat([features, condition], dim=1)).squeeze(-1)


class WaveformCritic(nn.Module):
    """Conditional raw-waveform discriminator with valid-mask-aware pooling."""

    def __init__(self, channels: int = 6, hidden: int = 96):
        super().__init__()
        in_channels = channels + 2 + CONDITION_DIM
        self.net = nn.Sequential(
            _sn_conv(in_channels, hidden, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            _sn_conv(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            _sn_conv(hidden, hidden * 2, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2),
            _sn_conv(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.out = _sn_linear(hidden * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        T = x.shape[1]
        cond_t = condition.unsqueeze(1).expand(-1, T, -1)
        h = torch.cat([x, mask.unsqueeze(-1), valid_mask.unsqueeze(-1), cond_t], dim=-1).permute(0, 2, 1)
        h = self.net(h)
        pooled_valid = F.interpolate(valid_mask.unsqueeze(1), size=h.shape[-1], mode="nearest")
        h = (h * pooled_valid).sum(dim=-1) / pooled_valid.sum(dim=-1).clamp_min(1.0)
        return self.out(h).squeeze(-1)


class SetStyleCritic(nn.Module):
    """Reference-set/candidate discriminator."""

    def __init__(self, hidden: int = 256, feature_mode: str = "legacy"):
        super().__init__()
        self.feature_mode = str(feature_mode or "legacy").lower()
        feature_dim = RICH_HMOG_FEATURE_DIM if self.feature_mode == "rich" else HMOG_FEATURE_DIM
        pair_dim = feature_dim * 5 + CONDITION_DIM
        self.net = nn.Sequential(
            _sn_linear(pair_dim, hidden),
            nn.LeakyReLU(0.2),
            _sn_linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            _sn_linear(hidden, 1),
        )

    def forward(
        self,
        refs: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_count: torch.Tensor,
        cand_x: torch.Tensor,
        cand_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T, C = refs.shape
        ref_features = differentiable_hmog_features(
            refs.reshape(B * K, T, C),
            ref_mask.reshape(B * K, T),
            self.feature_mode,
        )
        ref_features = ref_features.reshape(B, K, -1)
        present = (torch.arange(K, device=refs.device).view(1, K) < ref_count.long().view(B, 1)).float()
        denom = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        centroid = (ref_features * present.unsqueeze(-1)).sum(dim=1) / denom
        variance = ((ref_features - centroid.unsqueeze(1)).pow(2) * present.unsqueeze(-1)).sum(dim=1) / denom
        dispersion = torch.sqrt(variance.clamp_min(1e-8))
        candidate = differentiable_hmog_features(cand_x, cand_mask, self.feature_mode)
        pair = torch.cat(
            [centroid, candidate, (centroid - candidate).abs(), centroid * candidate, dispersion, condition],
            dim=1,
        )
        return self.net(pair).squeeze(-1)


def _hinge_discriminator(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


class AdvCriticBundle(nn.Module):
    """Three local GAN critics used only by the standalone generator."""

    def __init__(self, cfg: Dict[str, object]):
        super().__init__()
        critics_cfg = cfg.get("critics", {}) if isinstance(cfg.get("critics", {}), dict) else {}
        self.use_feature = bool(critics_cfg.get("feature", True))
        self.use_waveform = bool(critics_cfg.get("waveform", critics_cfg.get("deep", True)))
        # Accept the old "verifier" key so old config files fail gracefully.
        self.use_set = bool(critics_cfg.get("set", critics_cfg.get("verifier", True)))
        self.feature_mode = str(cfg.get("feature_mode", critics_cfg.get("feature_mode", "legacy"))).lower()
        self.population_refs = max(1, int(cfg.get("population_refs", 3)))
        if self.use_feature:
            self.feature = FeatureCritic(feature_mode=self.feature_mode)
        if self.use_waveform:
            self.waveform = WaveformCritic()
        if self.use_set:
            self.set_style = SetStyleCritic(feature_mode=self.feature_mode)

    @staticmethod
    def set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _population_reference_set(
        self,
        real_x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        refs = torch.stack(
            [torch.roll(real_x.detach(), shifts=shift, dims=0) for shift in range(1, self.population_refs + 1)],
            dim=1,
        )
        ref_mask = torch.stack(
            [torch.roll(mask.detach(), shifts=shift, dims=0) for shift in range(1, self.population_refs + 1)],
            dim=1,
        )
        count = torch.full(
            (real_x.shape[0],),
            self.population_refs,
            dtype=torch.long,
            device=real_x.device,
        )
        return refs, ref_mask, count

    def _reference_set(
        self,
        real_x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return refs and a flag identifying true same-user enrollment sets."""
        mask = batch["loss_mask"].float()
        population = self._population_reference_set(real_x, mask)
        if "refs" not in batch or "ref_mask" not in batch or "ref_count" not in batch:
            has_enrollment = torch.zeros(real_x.shape[0], dtype=torch.bool, device=real_x.device)
            return population[0], population[1], population[2], has_enrollment

        refs = batch["refs"].detach()
        ref_mask = batch["ref_mask"].detach()
        ref_count = batch["ref_count"].long().detach()
        has_enrollment = ref_count > 0
        if not bool(has_enrollment.any()):
            return population[0], population[1], population[2], has_enrollment
        if bool(has_enrollment.all()):
            return refs, ref_mask, ref_count, has_enrollment

        # Rare fallback for users with no enrollment rows.
        pop_refs, pop_mask, pop_count = population
        K = max(refs.shape[1], pop_refs.shape[1])
        mixed_refs = real_x.new_zeros((real_x.shape[0], K, real_x.shape[1], real_x.shape[2]))
        mixed_mask = mask.new_zeros((real_x.shape[0], K, real_x.shape[1]))
        mixed_refs[:, : refs.shape[1]] = refs
        mixed_mask[:, : ref_mask.shape[1]] = ref_mask
        missing = ~has_enrollment
        mixed_refs[missing] = 0.0
        mixed_mask[missing] = 0.0
        mixed_refs[missing, : pop_refs.shape[1]] = pop_refs[missing]
        mixed_mask[missing, : pop_mask.shape[1]] = pop_mask[missing]
        mixed_count = ref_count.clone()
        mixed_count[missing] = pop_count[missing]
        return mixed_refs, mixed_mask, mixed_count, has_enrollment

    @staticmethod
    def _different_user_indices(user_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uid = user_id.long().view(-1)
        different = uid.view(-1, 1) != uid.view(1, -1)
        has_negative = different.any(dim=1)
        indices = different.float().argmax(dim=1)
        return indices, has_negative

    def discriminator_loss(
        self,
        real_x: torch.Tensor,
        fake_x_detached: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        cfg: Dict[str, float],
        apply_r1: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        logs: Dict[str, float] = {}
        mask = batch["loss_mask"].float()
        valid = batch["valid_mask"].float()
        condition = critic_condition(batch)

        if self.use_feature:
            real_logits = self.feature(real_x, mask, condition)
            fake_logits = self.feature(fake_x_detached, mask, condition)
            loss = _hinge_discriminator(real_logits, fake_logits)
            losses.append(float(cfg.get("feature_weight", 1.0)) * loss)
            logs["adv_d_feature"] = float(loss.detach().cpu())
            logs["adv_acc_feature"] = float(
                (0.5 * ((real_logits > 0).float().mean() + (fake_logits <= 0).float().mean())).detach().cpu()
            )

        if self.use_waveform:
            real_logits = self.waveform(real_x, mask, valid, condition)
            fake_logits = self.waveform(fake_x_detached, mask, valid, condition)
            loss = _hinge_discriminator(real_logits, fake_logits)
            losses.append(float(cfg.get("waveform_weight", cfg.get("deep_weight", 1.0))) * loss)
            logs["adv_d_waveform"] = float(loss.detach().cpu())
            logs["adv_acc_waveform"] = float(
                (0.5 * ((real_logits > 0).float().mean() + (fake_logits <= 0).float().mean())).detach().cpu()
            )

        if self.use_set:
            refs, ref_mask, ref_count, has_enrollment = self._reference_set(real_x, batch)
            positive = self.set_style(refs, ref_mask, ref_count, real_x, mask, condition)
            fake_logits = self.set_style(refs, ref_mask, ref_count, fake_x_detached, mask, condition)
            loss = _hinge_discriminator(positive, fake_logits)

            # Only true enrollment sets have an identity interpretation.
            if bool(has_enrollment.any()) and float(cfg.get("real_negative_weight", 0.0)) > 0:
                neg_idx, has_negative = self._different_user_indices(batch["user_id"])
                use_negative = has_enrollment & has_negative
                if bool(use_negative.any()):
                    negative = self.set_style(
                        refs,
                        ref_mask,
                        ref_count,
                        real_x[neg_idx],
                        mask[neg_idx],
                        condition[neg_idx],
                    )
                    neg_loss = F.relu(1.0 + negative[use_negative]).mean()
                    loss = loss + float(cfg.get("real_negative_weight", 0.0)) * neg_loss
                    logs["adv_d_set_real_negative"] = float(neg_loss.detach().cpu())

            losses.append(float(cfg.get("set_weight", cfg.get("verifier_weight", 1.0))) * loss)
            logs["adv_d_set"] = float(loss.detach().cpu())
            logs["adv_acc_set"] = float(
                (0.5 * ((positive > 0).float().mean() + (fake_logits <= 0).float().mean())).detach().cpu()
            )

        total = sum(losses) if losses else real_x.new_tensor(0.0)
        if apply_r1 and float(cfg.get("r1_gamma", 0.0)) > 0:
            r1 = self.r1_penalty(real_x, batch, cfg)
            lazy_scale = max(1, int(cfg.get("r1_every", 16)))
            r1_term = 0.5 * float(cfg.get("r1_gamma", 1.0)) * float(lazy_scale) * r1
            total = total + r1_term
            logs["adv_r1"] = float(r1.detach().cpu())
            logs["adv_r1_term"] = float(r1_term.detach().cpu())
        logs["adv_d_total"] = float(total.detach().cpu())
        return total, logs

    def r1_penalty(
        self,
        real_x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        cfg: Dict[str, float],
    ) -> torch.Tensor:
        real = real_x.detach().requires_grad_(True)
        mask = batch["loss_mask"].float()
        valid = batch["valid_mask"].float()
        condition = critic_condition(batch).detach()
        scores: List[torch.Tensor] = []
        if self.use_feature:
            scores.append(float(cfg.get("feature_weight", 1.0)) * self.feature(real, mask, condition))
        if self.use_waveform:
            scores.append(
                float(cfg.get("waveform_weight", cfg.get("deep_weight", 1.0)))
                * self.waveform(real, mask, valid, condition)
            )
        if self.use_set:
            refs, ref_mask, ref_count, _ = self._reference_set(real.detach(), batch)
            scores.append(
                float(cfg.get("set_weight", cfg.get("verifier_weight", 1.0)))
                * self.set_style(refs, ref_mask, ref_count, real, mask, condition)
            )
        if not scores:
            return real_x.new_tensor(0.0)
        score = sum(scores)
        gradient = torch.autograd.grad(score.sum(), real, create_graph=True, only_inputs=True)[0]
        m = valid.unsqueeze(-1)
        return (gradient.pow(2) * m).sum() / (m.sum() * real.shape[-1]).clamp_min(1.0)

    def generator_loss(
        self,
        fake_x: torch.Tensor,
        real_x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        cfg: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        logs: Dict[str, float] = {}
        mask = batch["loss_mask"].float()
        valid = batch["valid_mask"].float()
        condition = critic_condition(batch)

        if self.use_feature:
            logits = self.feature(fake_x, mask, condition)
            loss = -logits.mean()
            losses.append(float(cfg.get("feature_weight", 1.0)) * loss)
            logs["adv_g_feature"] = float(loss.detach().cpu())
            logs["adv_fool_feature"] = float((logits > 0).float().mean().detach().cpu())

        if self.use_waveform:
            logits = self.waveform(fake_x, mask, valid, condition)
            loss = -logits.mean()
            losses.append(float(cfg.get("waveform_weight", cfg.get("deep_weight", 1.0))) * loss)
            logs["adv_g_waveform"] = float(loss.detach().cpu())
            logs["adv_fool_waveform"] = float((logits > 0).float().mean().detach().cpu())

        if self.use_set:
            refs, ref_mask, ref_count, _ = self._reference_set(real_x, batch)
            logits = self.set_style(refs, ref_mask, ref_count, fake_x, mask, condition)
            loss = -logits.mean()
            losses.append(float(cfg.get("set_weight", cfg.get("verifier_weight", 1.0))) * loss)
            logs["adv_g_set"] = float(loss.detach().cpu())
            logs["adv_fool_set"] = float((logits > 0).float().mean().detach().cpu())

        total = sum(losses) if losses else fake_x.new_tensor(0.0)
        logs["adv_g_total"] = float(total.detach().cpu())
        return total, logs

    @torch.no_grad()
    def evaluate(
        self,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        cfg: Dict[str, float],
    ) -> Dict[str, float]:
        """Evaluate actual multi-step samples with the current critics."""
        out: Dict[str, float] = {}
        mask = batch["loss_mask"].float()
        valid = batch["valid_mask"].float()
        condition = critic_condition(batch)
        generator_terms: List[torch.Tensor] = []

        if self.use_feature:
            real_logits = self.feature(real_x, mask, condition)
            fake_logits = self.feature(fake_x, mask, condition)
            generator_terms.append(float(cfg.get("feature_weight", 1.0)) * F.softplus(-fake_logits).mean())
            out["adv_eval_acc_feature"] = float(
                (0.5 * ((real_logits > 0).float().mean() + (fake_logits <= 0).float().mean())).cpu()
            )
            out["adv_eval_fool_feature"] = float((fake_logits > 0).float().mean().cpu())

        if self.use_waveform:
            real_logits = self.waveform(real_x, mask, valid, condition)
            fake_logits = self.waveform(fake_x, mask, valid, condition)
            waveform_weight = float(cfg.get("waveform_weight", cfg.get("deep_weight", 1.0)))
            generator_terms.append(waveform_weight * F.softplus(-fake_logits).mean())
            out["adv_eval_acc_waveform"] = float(
                (0.5 * ((real_logits > 0).float().mean() + (fake_logits <= 0).float().mean())).cpu()
            )
            out["adv_eval_fool_waveform"] = float((fake_logits > 0).float().mean().cpu())

        if self.use_set:
            refs, ref_mask, ref_count, _ = self._reference_set(real_x, batch)
            real_logits = self.set_style(refs, ref_mask, ref_count, real_x, mask, condition)
            fake_logits = self.set_style(refs, ref_mask, ref_count, fake_x, mask, condition)
            set_weight = float(cfg.get("set_weight", cfg.get("verifier_weight", 1.0)))
            generator_terms.append(set_weight * F.softplus(-fake_logits).mean())
            out["adv_eval_acc_set"] = float(
                (0.5 * ((real_logits > 0).float().mean() + (fake_logits <= 0).float().mean())).cpu()
            )
            out["adv_eval_fool_set"] = float((fake_logits > 0).float().mean().cpu())

        out["adv_g_eval"] = float(sum(generator_terms).cpu()) if generator_terms else 0.0
        return out
