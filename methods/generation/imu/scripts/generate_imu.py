#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from android_imu_layer import AndroidIMUDiffusionLayer  # noqa: E402
from android_imu_layer.diffusion_generator import json_safe_metadata  # noqa: E402


def save_sample(sample, out: Path, summary_json: Optional[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = json_safe_metadata(sample["metadata"])
    np.savez_compressed(
        out,
        action=np.asarray(sample["action"]),
        hz=np.asarray(sample["hz"], dtype=np.float32),
        window=sample["window"].astype(np.float32),
        active_imu=sample["active_imu"].astype(np.float32),
        mask=sample["mask"].astype(np.uint8),
        valid_mask=sample["valid_mask"].astype(np.uint8),
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    summary = {
        "out": str(out),
        "action": sample["action"],
        "hz": float(sample["hz"]),
        "window_shape": list(sample["window"].shape),
        "active_shape": list(sample["active_imu"].shape),
        "event_duration_ms": float(meta["event_duration_ms"]),
        "generation_wall_ms": float(meta["generation_wall_ms"]),
        "metadata": meta,
    }
    if summary_json:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Android gesture IMU with the released diffusion checkpoints.")
    ap.add_argument("action", choices=["tap", "scroll", "swipe", "pinch"])
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--x0", type=float)
    ap.add_argument("--y0", type=float)
    ap.add_argument("--x1", type=float)
    ap.add_argument("--y1", type=float)
    ap.add_argument("--center-x", type=float)
    ap.add_argument("--center-y", type=float)
    ap.add_argument("--start-span", type=float)
    ap.add_argument("--end-span", type=float)
    ap.add_argument("--user-id", type=int, default=None)
    ap.add_argument(
        "--reference-device",
        choices=["pixel10", "s21"],
        default=None,
        help="released five-shot reference phone (default: pixel10)",
    )
    ap.add_argument(
        "--reference-data-root",
        default=None,
        help="override data/on_device for an authorized processed reference export",
    )
    ap.add_argument("--duration-ms", type=float, default=None)
    ap.add_argument("--orientation-id", type=int, default=None)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    layer = AndroidIMUDiffusionLayer(
        seed=args.seed,
        device=args.device,
        reference_device=args.reference_device,
        reference_data_root=args.reference_data_root,
    )
    if args.action == "tap":
        sample = layer.tap(
            x=float(args.x),
            y=float(args.y),
            user_id=args.user_id,
            duration_ms=args.duration_ms,
            orientation_id=args.orientation_id,
            sample_steps=args.sample_steps,
        )
    elif args.action == "scroll":
        sample = layer.scroll(
            x0=float(args.x0),
            y0=float(args.y0),
            x1=float(args.x1),
            y1=float(args.y1),
            user_id=args.user_id,
            duration_ms=args.duration_ms,
            orientation_id=args.orientation_id,
            sample_steps=args.sample_steps,
        )
    elif args.action == "swipe":
        sample = layer.swipe(
            x0=float(args.x0),
            y0=float(args.y0),
            x1=float(args.x1),
            y1=float(args.y1),
            user_id=args.user_id,
            duration_ms=args.duration_ms,
            orientation_id=args.orientation_id,
            sample_steps=args.sample_steps,
        )
    elif args.action == "pinch":
        center = None
        if args.center_x is not None and args.center_y is not None:
            center = (args.center_x, args.center_y)
        sample = layer.pinch(
            center=center,
            start_span=args.start_span,
            end_span=args.end_span,
            user_id=args.user_id,
            duration_ms=args.duration_ms,
            orientation_id=args.orientation_id,
            sample_steps=args.sample_steps,
        )
    else:  # pragma: no cover - argparse restricts the action choices above.
        raise AssertionError(args.action)
    save_sample(sample, Path(args.out), args.summary_json)


if __name__ == "__main__":
    main()
