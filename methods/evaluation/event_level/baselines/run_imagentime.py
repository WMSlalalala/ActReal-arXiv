#!/usr/bin/env python3
"""ImagenTime (Naiman et al., NeurIPS 2024) as an IMU baseline on HMOG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("IMAGENTIME_ROOT", BASE / "ImagenTime"))
sys.path.insert(0, str(REPO))

# (delay, embedding == img_resolution).  Found by exhaustive search over
# embedding in {8, 16, 32, 48, 64}: these are the pairs whose columns fill the
# square exactly, so the transform round-trips without loss.  tap is the one
# action with no exact fill -- 6 filled columns and 2 zero columns in an 8x8
# square, which still inverts exactly because img_to_ts strips them.
ACTION_GEOMETRY = {
    "tap": {"samples": 16, "delay": 2, "embedding": 8},
    "pinch": {"samples": 100, "delay": 6, "embedding": 16},
    "swipe": {"samples": 176, "delay": 11, "embedding": 16},
    "scroll": {"samples": 208, "delay": 13, "embedding": 16},
    "keystroke": {"samples": 512, "delay": 16, "embedding": 32},
}
IMU_CHANNELS = 6


def build_args(action: str, args_cli) -> argparse.Namespace:
    """Their config, with only the per-action geometry filled in."""

    geometry = ACTION_GEOMETRY[action]
    # The authors' narrower UNet (configs/unconditional/mujoco.yaml): 14.4M
    # parameters against 151.7M, measured 4.5x cheaper per step.  It is used for
    small = not args_cli.large_unet
    namespace = argparse.Namespace(
        seed=args_cli.seed,
        num_workers=2,
        resume=False,
        log_dir=str(args_cli.output_dir / "ckpt"),
        neptune=False,
        tags=["hmog", "unconditional"],
        beta1=1e-5,
        betaT=1e-2,
        deterministic=False,
        epochs=args_cli.epochs,
        batch_size=args_cli.batch_size,
        learning_rate=3e-4,
        weight_decay=1e-5,
        # Any of the six names in that loader branch reaches our parse_datasets;
        # the name itself carries no information beyond routing.
        dataset="fred_md",
        seq_len=geometry["samples"],
        use_stft=False,
        n_fft=None,
        hop_length=None,
        delay=geometry["delay"],
        embedding=geometry["embedding"],
        img_resolution=geometry["embedding"],
        input_channels=IMU_CHANNELS,
        # Their smaller UNet (mujoco.yaml) for the longest action: 14.4M
        # parameters against 151.7M, measured 4.5x cheaper per step, which is
        # what brings keystroke inside a sane budget without touching the model.
        unet_channels=64 if small else 128,
        ch_mult=[1, 2, 2, 2] if small else [1, 2, 4, 4],
        attn_resolution=[8, 4, 2] if small else [32, 16, 8],
        diffusion_steps=args_cli.diffusion_steps,
        ema=True,
        ema_warmup=100,
        logging_iter=10**9,  # their scoring block is never entered; see the docstring
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", required=True, choices=sorted(ACTION_GEOMETRY))
    parser.add_argument("--epochs", type=int, default=1000,
                        help="the authors' own budget; checkpointed every 10 epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--sample-batch", type=int, default=200)
    parser.add_argument("--diffusion-steps", type=int, default=18)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25,
                        help="epochs between convergence checks; 0 disables early stopping")
    parser.add_argument("--patience", type=int, default=3,
                        help="convergence checks without improvement before stopping")
    parser.add_argument("--eval-samples", type=int, default=128,
                        help="windows drawn per convergence check")
    parser.add_argument("--large-unet", action="store_true",
                        help="use the wide UNet for keystroke too")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--restart", action="store_true")
    args_cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    windows_path = args_cli.real_dir / f"real_train_{args_cli.action}_imu.npy"
    stats_path = args_cli.output_dir / f"scaling_{args_cli.action}.json"
    os.environ["IMAGENTIME_HMOG_WINDOWS"] = str(windows_path)
    os.environ["IMAGENTIME_HMOG_STATS"] = str(stats_path)

    torch.random.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    from models.model import ImagenTime
    from models.sampler import DiffusionProcess
    from utils.utils import create_model_name_and_dir
    from utils.utils_data import gen_dataloader

    args = build_args(args_cli.action, args_cli)
    expected = ACTION_GEOMETRY[args_cli.action]["samples"]

    # Rewrites args.log_dir into the checkpoint FILE path; their loader is
    # called afterwards because it overwrites args.seq_len from the data.
    create_model_name_and_dir(args)
    train_loader, test_loader = gen_dataloader(args)
    if int(args.seq_len) != expected:
        raise SystemExit(
            f"{args_cli.action} windows are {args.seq_len} samples, the frozen "
            f"detector window is {expected}"
        )

    model = ImagenTime(args=args, device=args.device).to(args.device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    checkpoint_path = args_cli.output_dir / f"ckpt_{args_cli.action}.pt"
    first_epoch = 0
    trained_before = 0.0
    resumed_convergence = None
    if checkpoint_path.is_file() and not args_cli.restart:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["action"] != args_cli.action:
            raise SystemExit(
                f"{checkpoint_path.name} holds {state['action']}, this run is "
                f"{args_cli.action}"
            )
        # A checkpoint from a different UNet width is a different model.  The
        # authors' `restore_state` loads with strict=False, so without this the
        # mismatched tensors would be skipped in silence and training would
        # continue from a half-initialised network that no configuration
        # describes.
        stored = (state.get("unet_channels"), tuple(state.get("ch_mult") or ()))
        current = (args.unet_channels, tuple(args.ch_mult))
        if stored != (None, ()) and stored != current:
            raise SystemExit(
                f"{checkpoint_path.name} was trained at unet_channels="
                f"{stored[0]} ch_mult={list(stored[1])}, this run asks for "
                f"{current[0]} / {list(current[1])}. Pass --restart to train "
                "from scratch at the new width."
            )
        model.load_state_dict(state["model"], strict=False)
        if args.ema and state.get("ema_model") is not None:
            model.model_ema.load_state_dict(state["ema_model"])
        optimiser.load_state_dict(state["optimiser"])
        first_epoch = int(state["epoch"]) + 1
        trained_before = float(state.get("train_seconds", 0.0))
        resumed_convergence = state.get("convergence")
        print(f"resumed from epoch {first_epoch}", flush=True)

    def save_checkpoint(epoch: int, elapsed: float) -> None:
        temporary = checkpoint_path.with_suffix(".pt.partial")
        torch.save(
            {
                "epoch": int(epoch),
                "model": model.state_dict(),
                "ema_model": model.model_ema.state_dict() if args.ema else None,
                "optimiser": optimiser.state_dict(),
                "train_seconds": float(elapsed),
                # Without this a restart rebuilds the monitor empty: the first
                # check after it becomes the new "best" however poor it is, the
                # patience counter returns to zero, and training that had
                # already converged runs on. That is what happened to scroll.
                "convergence": None if monitor is None else {
                    "best_step": monitor.best_step,
                    "best_gap": monitor.best_gap,
                    "since_best": monitor._since_best,
                    "history": monitor.history,
                },
                "action": args_cli.action,
                "epochs": int(args.epochs),
                "unet_channels": int(args.unet_channels),
                "ch_mult": list(args.ch_mult),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)

    # Early stopping against the real data, not against a fixed epoch count.
    #
    monitor = None
    if args_cli.eval_every > 0:
        from convergence import ConvergenceMonitor

        real_windows = np.load(windows_path).astype(np.float32)
        monitor = ConvergenceMonitor(
            real=real_windows[:400], patience=args_cli.patience,
            minimum_checkpoints=3, tolerance=0.01,
        )

    if monitor is not None and resumed_convergence:
        monitor.best_step = resumed_convergence["best_step"]
        monitor.best_gap = resumed_convergence["best_gap"]
        monitor._since_best = resumed_convergence["since_best"]
        monitor.history = list(resumed_convergence["history"])
        print(f"  restored convergence: best epoch {monitor.best_step} "
              f"gap {monitor.best_gap:.4f}, {monitor._since_best} check(s) since, "
              f"{len(monitor.history)} recorded", flush=True)

    def draw_for_evaluation(count: int) -> np.ndarray:
        """A small unconditional draw, in the same way the final sampling does."""

        model.eval()
        with torch.no_grad(), model.ema_scope():
            primer = next(iter(train_loader))[0].to(args.device)
            model.ts_to_img(primer)          # caches the shape img_to_ts unpads against
            process = DiffusionProcess(
                args, model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            image = process.sampling(sampling_number=count)
            series = model.img_to_ts(image).detach().cpu().numpy()
        model.train()
        low = np.asarray(json.loads(stats_path.read_text())["low"], dtype=np.float32)
        span = np.asarray(json.loads(stats_path.read_text())["span"], dtype=np.float32)
        return (series * span + low).astype(np.float32)

    stopped_early_at = None
    started = time.time()
    for epoch in range(first_epoch, int(args.epochs)):
        model.train()
        model.epoch = epoch
        total, batches = 0.0, 0
        for data in train_loader:
            series = data[0].to(args.device)
            image = model.ts_to_img(series)
            optimiser.zero_grad()
            loss = model.loss_fn(image)
            if isinstance(loss, tuple) and len(loss) == 2:
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            model.on_train_batch_end()
            total += float(loss.item())
            batches += 1
        if (epoch + 1) % args_cli.checkpoint_every == 0 or epoch + 1 == int(args.epochs):
            save_checkpoint(epoch, trained_before + time.time() - started)
        if epoch % 10 == 0 or epoch == int(args.epochs) - 1:
            print(f"epoch {epoch}/{args.epochs} loss {total / max(batches, 1):.5f}",
                  flush=True)

        if monitor is not None and (epoch + 1) % args_cli.eval_every == 0:
            record = monitor.observe(epoch, draw_for_evaluation(args_cli.eval_samples))
            print(f"  convergence epoch {epoch}: gap {record['gap']:.4f} "
                  f"(lag1 {record['lag1']:.3f} vs real {record['real_lag1']:.3f}, "
                  f"dispersion {record['std_ratio_median']:.3f}, "
                  f"energy {record['energy_distance']:.3f}) "
                  f"best {monitor.best_step} since {record['since_best']}", flush=True)
            if monitor.should_stop():
                stopped_early_at = epoch
                save_checkpoint(epoch, trained_before + time.time() - started)
                print(f"  early stop at epoch {epoch}: the gap has not improved for "
                      f"{args_cli.patience} checks; best was epoch {monitor.best_step}",
                      flush=True)
                break
    trained = trained_before + time.time() - started

    # img_to_ts unpads against a shape DelayEmbedder caches the first time
    # ts_to_img runs, so one real batch has to go through before any draw.
    model.eval()
    with torch.no_grad():
        primer = next(iter(train_loader))[0].to(args.device)
        model.ts_to_img(primer)

    drawn = []
    started = time.time()
    with torch.no_grad(), model.ema_scope():
        process = DiffusionProcess(
            args, model.net,
            (args.input_channels, args.img_resolution, args.img_resolution),
        )
        while sum(len(chunk) for chunk in drawn) < args_cli.samples:
            count = min(args_cli.sample_batch,
                        args_cli.samples - sum(len(chunk) for chunk in drawn))
            image = process.sampling(sampling_number=count)
            series = model.img_to_ts(image)
            drawn.append(series.detach().cpu().numpy())
    sampled = time.time() - started

    raw = np.concatenate(drawn)[: args_cli.samples]
    scaling = json.loads(stats_path.read_text())
    low = np.asarray(scaling["low"], dtype=np.float32)
    span = np.asarray(scaling["span"], dtype=np.float32)
    generated = (raw * span + low).astype(np.float32)
    if generated.shape[1:] != (expected, IMU_CHANNELS):
        raise SystemExit(
            f"sampler returned {generated.shape[1:]}, expected "
            f"{(expected, IMU_CHANNELS)} -- the delay geometry did not invert"
        )

    np.save(args_cli.output_dir / f"samples_{args_cli.action}_imu.npy", generated)
    (args_cli.output_dir / f"summary_{args_cli.action}.json").write_text(
        json.dumps(
            {
                "method": "imagentime",
                "citation": "Naiman et al., ImagenTime, NeurIPS 2024",
                "action": args_cli.action,
                "seq_length": int(expected),
                "channels": IMU_CHANNELS,
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "delay": args.delay,
                "embedding": args.embedding,
                "img_resolution": args.img_resolution,
                "unet_channels": args.unet_channels,
                "ch_mult": list(args.ch_mult),
                "diffusion_steps": args.diffusion_steps,
                "epochs": int(args.epochs),
                "epochs_run": (stopped_early_at + 1 if stopped_early_at is not None
                               else int(args.epochs)) - first_epoch,
                "stopped_early_at": stopped_early_at,
                "early_stop_criterion": (
                    None if monitor is None else
                    "gap to real data (lag-1 autocorrelation + per-channel "
                    "dispersion + energy distance over window summaries) did not "
                    f"improve for {args_cli.patience} checks {args_cli.eval_every} "
                    "epochs apart"
                ),
                "convergence_history": None if monitor is None else monitor.history,
                "train_seconds": round(trained, 1),
                "samples": int(len(generated)),
                "sample_seconds": round(sampled, 2),
                "sample_ms_per_event": round(1000.0 * sampled / max(len(generated), 1), 4),
                "normalisation": "per-channel min-max into [0,1], inverted after sampling",
                "generator_modified": False,
                "authors_config": (
                    "configs/unconditional/fred_md.yaml with use_stft off and the "
                    "delay geometry this window length requires; keystroke uses the "
                    "authors' mujoco.yaml UNet width"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"saved {generated.shape} to samples_{args_cli.action}_imu.npy", flush=True)


if __name__ == "__main__":
    main()
