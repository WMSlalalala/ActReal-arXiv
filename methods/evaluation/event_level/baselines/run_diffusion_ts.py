#!/usr/bin/env python3
"""Train and sample Diffusion-TS (ICLR 2024) on HMOG events, per action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
DIFFUSION_TS_ROOT = Path(os.environ.get(
    "DIFFUSION_TS_ROOT", BASE.parents[1] / "third_party" / "DiffusionTS"
))
sys.path.insert(0, str(DIFFUSION_TS_ROOT))

from common.hmog_baseline_common import (  # noqa: E402
    ACTION_WINDOW_SAMPLES,
    collect_genuine_windows,
)
TRAJECTORY_COLUMNS = (1, 2)


class WindowDataset(Dataset):
    """The authors' normalisation, over event windows instead of a sliding one."""

    def __init__(self, windows: np.ndarray):
        flat = windows.reshape(-1, windows.shape[-1])
        self.scaler = MinMaxScaler().fit(flat)
        scaled = self.scaler.transform(flat).reshape(windows.shape)
        self.samples = (scaled * 2.0 - 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.samples[index])

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        zero_one = np.clip((values + 1.0) / 2.0, 0.0, 1.0)
        flat = zero_one.reshape(-1, shape[-1])
        return self.scaler.inverse_transform(flat).reshape(shape)


class _Args:
    """The minimum the authors' Trainer reads off its args object."""

    def __init__(self, name: str, save_dir: str):
        self.name = name
        self.save_dir = save_dir


def build_model(seq_length: int, feature_size: int, sampling_timesteps: int):
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

    # Model hyper-parameters exactly as published for the real-world benchmarks
    # (Config/etth.yaml); only the two shape parameters follow the action being
    # modelled.  The solver settings are the authors' Sines configuration --
    # see solver_config for which two values that changes and why.
    return Diffusion_TS(
        seq_length=seq_length,
        feature_size=feature_size,
        n_layer_enc=3,
        n_layer_dec=2,
        d_model=64,
        timesteps=500,
        sampling_timesteps=sampling_timesteps,
        loss_type="l1",
        beta_schedule="cosine",
        n_heads=4,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=1,
        padding_size=0,
    )


def solver_config(steps: int, results_folder: str) -> dict:
    """The authors' solver block, with the Sines budget rather than the ETTh one."""

    return {
        "solver": {
            "max_epochs": steps,
            "gradient_accumulate_every": 2,
            "save_cycle": max(steps // 2, 1),
            "results_folder": results_folder,
            "base_lr": 1.0e-5,
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 3000,
                    "min_lr": 1.0e-5,
                    "threshold": 1.0e-1,
                    "threshold_mode": "rel",
                    "warmup_lr": 8.0e-4,
                    "warmup": 500,
                    "verbose": False,
                },
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--kind", choices=("trajectory", "imu"), required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--sample-batch", type=int, default=512)
    parser.add_argument("--sampling-timesteps", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--train-events",
        type=int,
        default=0,
        help=(
            "cap the total genuine windows the generator may fit, to match an "
            "attacker's shot budget; 0 means every train-split event"
        ),
    )
    parser.add_argument(
        "--events-per-user",
        type=int,
        default=0,
        help=(
            "cap the windows per train user instead, e.g. 5 for a five-shot "
            "budget spread over everyone the attacker can reach"
        ),
    )
    args = parser.parse_args()

    from engine.solver import Trainer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows, users = collect_genuine_windows(
        args.dataset_dir, "train", args.action, args.kind
    )
    if args.kind == "trajectory":
        windows = windows[:, :, TRAJECTORY_COLUMNS]

    # Shot budget.  The paper's own attack is given five real events of the
    # victim; a generator handed every event of seventy users is a different
    if args.events_per_user > 0 or args.train_events > 0:
        order = sorted(
            range(len(windows)),
            key=lambda index: hashlib.sha256(
                f"{users[index]}|{index}|{args.action}|{args.kind}".encode()
            ).hexdigest(),
        )
        if args.events_per_user > 0:
            taken: dict[str, int] = {}
            keep = []
            for index in order:
                owner = users[index]
                if taken.get(owner, 0) < args.events_per_user:
                    taken[owner] = taken.get(owner, 0) + 1
                    keep.append(index)
        else:
            keep = order[: args.train_events]
        windows = windows[keep]
        users = [users[index] for index in keep]
        print(
            f"  shot budget: kept {len(windows)} windows from "
            f"{len(set(users))} users",
            flush=True,
        )

    seq_length = ACTION_WINDOW_SAMPLES[args.action]
    if windows.shape[1] != seq_length:
        raise SystemExit(f"unexpected window length {windows.shape}")
    print(
        f"{args.action}/{args.kind}: {len(windows)} train-split genuine windows "
        f"of {seq_length}x{windows.shape[-1]} from {len(set(users))} users",
        flush=True,
    )

    dataset = WindowDataset(windows)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        drop_last=len(dataset) > args.batch_size,
        num_workers=2,
        pin_memory=True,
    )
    model = build_model(seq_length, windows.shape[-1], args.sampling_timesteps).cuda()
    tag = f"{args.action}_{args.kind}"
    config = solver_config(args.steps, str(args.output_dir / f"ckpt_{tag}"))
    trainer = Trainer(
        config=config,
        args=_Args(tag, str(args.output_dir)),
        model=model,
        dataloader={"dataloader": loader},
        logger=None,
    )

    started = time.time()
    trainer.train()
    trained = time.time() - started

    started = time.time()
    raw = trainer.sample(
        num=args.samples,
        size_every=args.sample_batch,
        shape=[seq_length, windows.shape[-1]],
    )[: args.samples]
    sampled = time.time() - started
    generated = dataset.unnormalize(raw).astype(np.float32)

    np.save(args.output_dir / f"samples_{tag}.npy", generated)
    (args.output_dir / f"summary_{tag}.json").write_text(
        json.dumps(
            {
                "action": args.action,
                "kind": args.kind,
                "train_windows": int(len(windows)),
                "train_users": len(set(users)),
                "shot_budget_events_total": args.train_events,
                "shot_budget_events_per_user": args.events_per_user,
                "seq_length": seq_length,
                "feature_size": int(windows.shape[-1]),
                "steps": args.steps,
                "samples": int(len(generated)),
                "sampling_timesteps": args.sampling_timesteps,
                "train_seconds": round(trained, 1),
                "sample_seconds": round(sampled, 1),
                "real_mean": np.mean(windows, axis=(0, 1)).tolist(),
                "real_std": np.std(windows, axis=(0, 1)).tolist(),
                "generated_mean": np.mean(generated, axis=(0, 1)).tolist(),
                "generated_std": np.std(generated, axis=(0, 1)).tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"saved {generated.shape} to samples_{tag}.npy", flush=True)


if __name__ == "__main__":
    main()
