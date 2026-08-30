from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class DDPMSchedule:
    def __init__(self, train_steps: int, beta_start: float, beta_end: float, device: torch.device):
        self.train_steps = int(train_steps)
        betas = torch.linspace(float(beta_start), float(beta_end), self.train_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)
        self.posterior_var = betas * (1.0 - prev) / (1.0 - alpha_bar)
        self.posterior_var = self.posterior_var.clamp_min(1e-20)

    def gather(self, arr: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return arr.gather(0, t.long()).view(-1, 1, 1).to(x.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return self.gather(self.sqrt_alpha_bar, t, x0) * x0 + self.gather(self.sqrt_one_minus_alpha_bar, t, x0) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x_t - self.gather(self.sqrt_one_minus_alpha_bar, t, x_t) * eps) / self.gather(self.sqrt_alpha_bar, t, x_t)

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        clamp_std: float,
    ) -> torch.Tensor:
        eps = model(x, t, batch)
        beta_t = self.gather(self.betas, t, x)
        sqrt_one_minus = self.gather(self.sqrt_one_minus_alpha_bar, t, x)
        sqrt_recip_alpha = self.gather(self.sqrt_recip_alpha, t, x)
        mean = sqrt_recip_alpha * (x - beta_t * eps / sqrt_one_minus)
        var = self.gather(self.posterior_var, t, x)
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().view(-1, 1, 1)
        out = mean + nonzero * torch.sqrt(var) * noise
        if clamp_std > 0:
            out = out.clamp(-float(clamp_std), float(clamp_std))
        valid = batch["valid_mask"].float().unsqueeze(-1)
        return out * valid

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, int, int],
        batch: Dict[str, torch.Tensor],
        sample_steps: int,
        clamp_std: float,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        x = x * batch["valid_mask"].float().unsqueeze(-1)
        steps = torch.linspace(self.train_steps - 1, 0, int(sample_steps), device=device).long()
        # Deterministic DDIM update on a reduced time grid.  This keeps
        # sample_steps configurable without pretending skipped DDPM transitions
        # are adjacent transitions.
        for i, step in enumerate(steps):
            step_i = int(step.item())
            next_i = int(steps[i + 1].item()) if i + 1 < len(steps) else -1
            t = torch.full((shape[0],), step_i, device=device, dtype=torch.long)
            eps = model(x, t, batch)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if clamp_std > 0:
                x0 = x0.clamp(-float(clamp_std), float(clamp_std))
            if next_i < 0:
                x = x0
            else:
                a_next = self.alpha_bar[next_i].view(1, 1, 1)
                x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps
            x = x * batch["valid_mask"].float().unsqueeze(-1)
        return x

    def sample_differentiable(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, int, int],
        batch: Dict[str, torch.Tensor],
        sample_steps: int,
        clamp_std: float,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reduced-step DDIM sampler that preserves gradients to the generator."""
        if int(sample_steps) < 2:
            raise ValueError("differentiable sampling requires sample_steps >= 2")
        device = next(model.parameters()).device
        valid = batch["valid_mask"].float().unsqueeze(-1)
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        x = x * valid
        steps = torch.linspace(self.train_steps - 1, 0, int(sample_steps), device=device).long()
        for i, step in enumerate(steps):
            step_i = int(step.item())
            next_i = int(steps[i + 1].item()) if i + 1 < len(steps) else -1
            t = torch.full((shape[0],), step_i, device=device, dtype=torch.long)
            eps = model(x, t, batch)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if clamp_std > 0:
                # Exact hard clamp in the forward pass, straight-through
                # derivative in the ADV backward pass.  A plain clamp would
                # zero most gradients at early high-noise DDIM steps.
                clipped = x0.clamp(-float(clamp_std), float(clamp_std))
                x0 = x0 + (clipped - x0).detach()
            if next_i < 0:
                x = x0
            else:
                a_next = self.alpha_bar[next_i].view(1, 1, 1)
                x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps
            x = x * valid
        return x


def masked_mse(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    m = loss_mask.float().unsqueeze(-1)
    return ((pred - target).pow(2) * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


def velocity_loss(x0_pred: torch.Tensor, x0: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if x0.shape[1] < 2:
        return x0_pred.new_tensor(0.0)
    dp = x0_pred[:, 1:] - x0_pred[:, :-1]
    dt = x0[:, 1:] - x0[:, :-1]
    m = (loss_mask[:, 1:] * loss_mask[:, :-1]).float().unsqueeze(-1)
    return ((dp - dt).pow(2) * m).sum() / (m.sum() * x0.shape[-1]).clamp_min(1.0)


def acceleration_loss(x0_pred: torch.Tensor, x0: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if x0.shape[1] < 3:
        return x0_pred.new_tensor(0.0)
    ap = x0_pred[:, 2:] - 2.0 * x0_pred[:, 1:-1] + x0_pred[:, :-2]
    at = x0[:, 2:] - 2.0 * x0[:, 1:-1] + x0[:, :-2]
    m = (loss_mask[:, 2:] * loss_mask[:, 1:-1] * loss_mask[:, :-2]).float().unsqueeze(-1)
    return ((ap - at).pow(2) * m).sum() / (m.sum() * x0.shape[-1]).clamp_min(1.0)


def jerk_loss(x0_pred: torch.Tensor, x0: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if x0.shape[1] < 4:
        return x0_pred.new_tensor(0.0)
    jp = x0_pred[:, 3:] - 3.0 * x0_pred[:, 2:-1] + 3.0 * x0_pred[:, 1:-2] - x0_pred[:, :-3]
    jt = x0[:, 3:] - 3.0 * x0[:, 2:-1] + 3.0 * x0[:, 1:-2] - x0[:, :-3]
    m = (loss_mask[:, 3:] * loss_mask[:, 2:-1] * loss_mask[:, 1:-2] * loss_mask[:, :-3]).float().unsqueeze(-1)
    return ((jp - jt).pow(2) * m).sum() / (m.sum() * x0.shape[-1]).clamp_min(1.0)


def fourier_loss(x0_pred: torch.Tensor, x0: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    m = loss_mask.float().unsqueeze(-1)
    pred = x0_pred * m
    real = x0 * m
    pf = torch.fft.rfft(pred, dim=1)
    rf = torch.fft.rfft(real, dim=1)
    return F.l1_loss(torch.log1p(torch.abs(pf)), torch.log1p(torch.abs(rf)))


def diffusion_train_step(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    valid = batch["valid_mask"].float().unsqueeze(-1)
    x0 = batch["x"] * valid
    B = x0.shape[0]
    t = torch.randint(0, schedule.train_steps, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0) * valid
    x_t = schedule.q_sample(x0, t, noise=noise) * valid
    eps_pred = model(x_t, t, batch)
    loss_mask = batch["loss_mask"].float()
    mse = masked_mse(eps_pred, noise, loss_mask)
    x0_pred = schedule.predict_x0_from_eps(x_t, t, eps_pred) * valid
    aux_x0_pred = x0_pred
    aux_clamp = float(loss_cfg.get("aux_x0_clamp_std", 0.0))
    if aux_clamp > 0:
        aux_x0_pred = x0_pred.clamp(-aux_clamp, aux_clamp)
    v_w = float(loss_cfg.get("velocity_weight", 0.0))
    v = velocity_loss(aux_x0_pred, x0, loss_mask) if v_w > 0 else x0.new_tensor(0.0)
    a_w = float(loss_cfg.get("acceleration_weight", 0.0))
    al = acceleration_loss(aux_x0_pred, x0, loss_mask) if a_w > 0 else x0.new_tensor(0.0)
    j_w = float(loss_cfg.get("jerk_weight", 0.0))
    jl = jerk_loss(aux_x0_pred, x0, loss_mask) if j_w > 0 else x0.new_tensor(0.0)
    f_w = float(loss_cfg.get("fourier_weight", 0.0))
    fl = fourier_loss(aux_x0_pred, x0, loss_mask) if f_w > 0 else x0.new_tensor(0.0)
    total = float(loss_cfg.get("mse_weight", 1.0)) * mse + v_w * v + a_w * al + j_w * jl + f_w * fl
    metrics = {
        "loss_mse": float(mse.detach().cpu()),
        "loss_velocity": float(v.detach().cpu()),
        "loss_acceleration": float(al.detach().cpu()),
        "loss_jerk": float(jl.detach().cpu()),
        "loss_fourier": float(fl.detach().cpu()),
    }
    return total, metrics, x0_pred
