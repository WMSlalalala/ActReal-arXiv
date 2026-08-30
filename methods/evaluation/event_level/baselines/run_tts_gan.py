#!/usr/bin/env python3
"""Train and sample TTS-GAN (imics-lab) on HMOG events, per action."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
TTS_GAN_ROOT = Path(os.environ.get(
    "TTS_GAN_ROOT", BASE.parents[1] / "third_party" / "TTSGAN"
))
sys.path.insert(0, str(TTS_GAN_ROOT))

from common.hmog_baseline_common import (  # noqa: E402
    ACTION_WINDOW_SAMPLES,
    collect_genuine_windows,
)

TRAJECTORY_COLUMNS = (1, 2)

# The published UniMiB recipe (RunningGAN_Train.py), verbatim apart from the
# shape and budget flags this driver overrides.
PUBLISHED_ARGV = [
    "-gen_bs", "16", "-dis_bs", "16",
    "--dist-url", "tcp://localhost:4321", "--dist-backend", "nccl",
    "--world-size", "1", "--rank", "0",
    "--dataset", "UniMiB", "--bottom_width", "8",
    "--max_iter", "500000", "--img_size", "32",
    "--gen_model", "my_gen", "--dis_model", "my_dis",
    "--df_dim", "384", "--d_heads", "4", "--d_depth", "3",
    "--g_depth", "5,4,2", "--dropout", "0", "--latent_dim", "100",
    "--gf_dim", "1024", "--num_workers", "4",
    "--g_lr", "0.0001", "--d_lr", "0.0003", "--optimizer", "adam",
    "--loss", "lsgan", "--wd", "1e-3", "--beta1", "0.9", "--beta2", "0.999",
    "--phi", "1", "--batch_size", "16", "--num_eval_imgs", "50000",
    "--init_type", "xavier_uniform", "--n_critic", "1",
    "--val_freq", "20", "--print_freq", "50",
    "--grow_steps", "0", "0", "--fade_in", "0", "--patch_size", "2",
    "--ema_kimg", "500", "--ema_warmup", "0.1", "--ema", "0.9999",
    "--diff_aug", "translation,cutout,color",
    "--exp_name", "hmog",
]


class ChannelFirstWindows(Dataset):
    """Windows as (channels, 1, length), the shape TTS-GAN's models expect."""

    def __init__(self, windows: np.ndarray, seed: int = 12345):
        epsilon = 1.0e-8
        self.means = windows.mean(axis=1)
        self.stds = np.sqrt(windows.var(axis=1)) + epsilon
        scaled = (windows - self.means[:, None, :]) / self.stds[:, None, :]
        self.samples = np.transpose(scaled, (0, 2, 1))[:, :, None, :].astype(
            np.float32
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.samples[index]), 0

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        """Restore physical units on (n, channels, 1, length) generator output."""

        series = np.transpose(values[:, :, 0, :], (0, 2, 1))
        drawn = self._rng.integers(0, len(self.means), size=len(series))
        return series * self.stds[drawn][:, None, :] + self.means[drawn][:, None, :]


def build_args(seq_length: int, max_iter: int, batch_size: int, gpu: int):
    import cfg

    argv = list(PUBLISHED_ARGV)
    for flag, value in (
        ("--max_iter", str(max_iter)),
        ("--batch_size", str(batch_size)),
        ("-gen_bs", str(batch_size)),
        ("-dis_bs", str(batch_size)),
    ):
        argv[argv.index(flag) + 1] = value
    saved, sys.argv = sys.argv, ["run_tts_gan.py"] + argv
    try:
        args = cfg.parse_args()
    finally:
        sys.argv = saved
    args.gpu = gpu
    args.rank = 0
    args.distributed = False
    args.world_size = 1
    args.img_size = seq_length
    args.lr_decay = True
    args.show = False
    args.max_epoch = 1
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--kind", choices=("trajectory", "imu"), required=True)
    parser.add_argument("--max-iter", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--sample-batch", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="epochs between checkpoints; a run is resumable from the last one")
    parser.add_argument("--restart", action="store_true",
                        help="ignore any existing checkpoint and train from scratch")
    args_cli = parser.parse_args()

    from functions import LinearLrDecay, copy_params, load_params, train
    from GANModels import Discriminator, Generator

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.cuda.set_device(args_cli.gpu)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    windows, users = collect_genuine_windows(
        args_cli.dataset_dir, "train", args_cli.action, args_cli.kind
    )
    if args_cli.kind == "trajectory":
        windows = windows[:, :, TRAJECTORY_COLUMNS]
    seq_length = ACTION_WINDOW_SAMPLES[args_cli.action]
    channels = int(windows.shape[-1])
    print(
        f"{args_cli.action}/{args_cli.kind}: {len(windows)} train-split genuine "
        f"windows of {seq_length}x{channels} from {len(set(users))} users",
        flush=True,
    )

    args = build_args(seq_length, args_cli.max_iter, args_cli.batch_size, args_cli.gpu)
    patch_size = int(args.patch_size)
    if seq_length % patch_size:
        raise SystemExit(f"patch size {patch_size} does not divide {seq_length}")

    dataset = ChannelFirstWindows(windows, seed=args_cli.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )
    args.max_epoch = int(np.ceil(args.max_iter * args.n_critic / len(loader)))

    generator = Generator(
        seq_len=seq_length,
        patch_size=patch_size,
        channels=channels,
        latent_dim=int(args.latent_dim),
    ).cuda(args.gpu)
    discriminator = Discriminator(
        in_channels=channels, patch_size=patch_size, seq_length=seq_length
    ).cuda(args.gpu)

    generator_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )
    discriminator_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        args.d_lr,
        (args.beta1, args.beta2),
    )
    schedulers = (
        LinearLrDecay(generator_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic),
        LinearLrDecay(discriminator_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic),
    )
    averaged = copy_params(deepcopy(generator).cpu())
    fixed_z = torch.cuda.FloatTensor(
        np.random.normal(0, 1, (100, args.latent_dim))
    )
    writer_dict = {
        "writer": SummaryWriter(str(args_cli.output_dir / "tb")),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # Resume and checkpoint.  A GAN run here is hours long and the authors' loop
    # keeps nothing between epochs, so an interrupted run previously threw away
    checkpoint_path = args_cli.output_dir / f"ckpt_{args_cli.action}_{args_cli.kind}.pt"
    first_epoch = 0
    trained_before = 0.0
    if checkpoint_path.is_file() and not args_cli.restart:
        state = torch.load(checkpoint_path, map_location=f"cuda:{args.gpu}", weights_only=False)
        # A checkpoint written under a different budget is a different
        # experiment: the learning-rate schedule is indexed by max_iter, so
        # continuing across a change would silently train a model no single
        # configuration describes.  Refuse, and say which flag to pass.
        if int(state["max_iter"]) != int(args.max_iter):
            raise SystemExit(
                f"{checkpoint_path.name} was written at max_iter="
                f"{state['max_iter']}, this run asks for {args.max_iter}. "
                "Pass --restart to train from scratch at the new budget, or "
                "rerun at the old one."
            )
        if state["action"] != args_cli.action or state["kind"] != args_cli.kind:
            raise SystemExit(
                f"{checkpoint_path.name} holds {state['action']}/{state['kind']}, "
                f"this run is {args_cli.action}/{args_cli.kind}"
            )
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        generator_optimizer.load_state_dict(state["generator_optimizer"])
        discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        averaged = [tensor.to("cpu") for tensor in state["averaged"]]
        writer_dict["train_global_steps"] = int(state["global_steps"])
        first_epoch = int(state["epoch"]) + 1
        trained_before = float(state.get("train_seconds", 0.0))
        # Restoring the samplers too means a resumed run continues the same
        # noise sequence rather than replaying the one it already consumed.
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"].cpu())
        if state.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state(state["cuda_random_state"].cpu(), device=args.gpu)
        print(f"resumed from epoch {first_epoch} ({state['global_steps']} steps)", flush=True)

    def save_checkpoint(epoch: int, elapsed: float) -> None:
        # Written to a neighbour first: a crash midway through a multi-hundred
        # megabyte write would otherwise leave a truncated file where the only
        # copy of the run used to be.
        temporary = checkpoint_path.with_suffix(".pt.partial")
        torch.save(
            {
                "epoch": int(epoch),
                "global_steps": int(writer_dict["train_global_steps"]),
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(),
                "averaged": [tensor.detach().cpu() for tensor in averaged],
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state(args.gpu),
                "train_seconds": float(elapsed),
                "action": args_cli.action,
                "kind": args_cli.kind,
                "max_iter": int(args.max_iter),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)

    started = time.time()
    for epoch in range(first_epoch, int(args.max_epoch)):
        train(
            args,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            averaged,
            loader,
            epoch,
            writer_dict,
            fixed_z,
            schedulers,
        )
        if (epoch + 1) % args_cli.checkpoint_every == 0 or epoch + 1 == int(args.max_epoch):
            save_checkpoint(epoch, trained_before + time.time() - started)
    # Training time is cumulative across resumes, so an interrupted run does not
    # report a shorter budget than it actually received.
    trained = trained_before + time.time() - started

    # Sample from the averaged generator, which is the one the authors keep.
    load_params(generator, averaged, args, mode="cpu")
    generator.eval()
    drawn = []
    started = time.time()
    with torch.no_grad():
        while sum(len(chunk) for chunk in drawn) < args_cli.samples:
            noise = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (args_cli.sample_batch, args.latent_dim))
            )
            drawn.append(generator(noise).cpu().numpy())
    sampled = time.time() - started
    raw = np.concatenate(drawn)[: args_cli.samples]
    generated = dataset.unnormalize(raw).astype(np.float32)

    tag = f"{args_cli.action}_{args_cli.kind}"
    np.save(args_cli.output_dir / f"samples_{tag}.npy", generated)
    (args_cli.output_dir / f"summary_{tag}.json").write_text(
        json.dumps(
            {
                "action": args_cli.action,
                "kind": args_cli.kind,
                "train_windows": int(len(windows)),
                "train_users": len(set(users)),
                "seq_length": seq_length,
                "feature_size": channels,
                "patch_size": patch_size,
                "max_iter": int(args.max_iter),
                "max_epoch": int(args.max_epoch),
                "batch_size": int(args.batch_size),
                "samples": int(len(generated)),
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
