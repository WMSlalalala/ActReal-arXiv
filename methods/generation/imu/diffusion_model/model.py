from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half <= 0:
        return t.float().unsqueeze(-1)
    device = t.device
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualFiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int, dropout: float):
        super().__init__()
        pad = int(dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.cond = nn.Linear(cond_dim, channels * 2)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.cond(cond).unsqueeze(-1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class RefEncoder(nn.Module):
    def __init__(self, channels: int, cond_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels + 1, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, refs: torch.Tensor, ref_mask: torch.Tensor, ref_count: torch.Tensor) -> torch.Tensor:
        # refs: [B,R,T,C], ref_mask: [B,R,T]
        B, R, T, C = refs.shape
        x = refs.reshape(B * R, T, C).permute(0, 2, 1)
        m = ref_mask.reshape(B * R, T).unsqueeze(1)
        h = self.net(torch.cat([x, m], dim=1))
        denom = m.sum(dim=-1).clamp_min(1.0)
        pooled = (h * m).sum(dim=-1) / denom
        pooled = self.proj(pooled).reshape(B, R, -1)
        present = (torch.arange(R, device=refs.device).unsqueeze(0) < ref_count.unsqueeze(1)).float()
        denom2 = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (pooled * present.unsqueeze(-1)).sum(dim=1) / denom2


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        T: int,
        cond_dim: int,
        time_embed_dim: int,
        use_ref_context: bool,
        ref_hidden: int,
    ):
        super().__init__()
        self.T = int(T)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.use_ref_context = bool(use_ref_context)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.orientation = nn.Embedding(4, cond_dim)
        self.active = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        # These reserved embeddings retain the released four-gesture checkpoint
        # shape. Gesture models always use index zero.
        self.reserved_count_a = nn.Embedding(65, cond_dim)
        self.reserved_count_b = nn.Embedding(65, cond_dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.ref_encoder: Optional[RefEncoder]
        if self.use_ref_context:
            self.ref_encoder = RefEncoder(6, cond_dim, ref_hidden)
        else:
            self.ref_encoder = None

    def forward(self, t: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = int(t.shape[0])
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        orient = self.orientation(batch["orientation_idx"].long().clamp(0, 3))
        active = batch["active_len"].float().view(B, 1) / float(max(1, self.T))
        active_emb = self.active(active)
        zero_index = torch.zeros(B, device=t.device, dtype=torch.long)
        reserved_a = self.reserved_count_a(zero_index)
        reserved_b = self.reserved_count_b(zero_index)
        cond = t_emb + orient + active_emb + reserved_a + reserved_b
        if self.ref_encoder is not None and "refs" in batch and "ref_mask" in batch and "ref_count" in batch:
            cond = cond + self.ref_encoder(batch["refs"], batch["ref_mask"], batch["ref_count"])
        return self.mix(cond)


class TemporalDenoiser(nn.Module):
    def __init__(
        self,
        T: int,
        in_channels: int = 6,
        base_channels: int = 96,
        cond_dim: int = 192,
        n_blocks: int = 8,
        dropout: float = 0.05,
        time_embed_dim: int = 128,
        use_ref_context: bool = True,
        ref_encoder_channels: int = 64,
    ):
        super().__init__()
        self.T = int(T)
        self.in_channels = int(in_channels)
        self.cond = ConditionEncoder(
            T=T,
            cond_dim=cond_dim,
            time_embed_dim=time_embed_dim,
            use_ref_context=use_ref_context,
            ref_hidden=ref_encoder_channels,
        )
        self.in_proj = nn.Conv1d(in_channels + 2, base_channels, kernel_size=3, padding=1)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                ResidualFiLMBlock(
                    channels=base_channels,
                    cond_dim=cond_dim,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(int(n_blocks))
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # x_t: [B,T,6]
        mask = batch["mask"].float().unsqueeze(-1)
        valid = batch["valid_mask"].float().unsqueeze(-1)
        x = torch.cat([x_t, mask, valid], dim=-1).permute(0, 2, 1)
        cond = self.cond(t, batch)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).permute(0, 2, 1)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for k, v in backup.items():
            state[k].copy_(v)
