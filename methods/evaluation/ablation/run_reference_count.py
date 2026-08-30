#!/usr/bin/env python3
"""Generate the canonical IMU reference-count ablation cache.

The ablation has two checkpoint families.  ``k=0`` uses the packaged
zero-shot adversarial checkpoint.  ``k in {1, 3, 5, 8}`` uses the same
packaged five-shot adversarial checkpoint for each action and changes only the
number of references consumed at inference time.  The override is installed
before :class:`AndroidIMUDiffusionLayer` builds an action runtime, which is
when the reference bank and denoiser conditioning shape are selected.

This is a cleaned, repository-relative version of the runner used for the
paper experiment.  It delegates cache construction to the canonical cache
generator and retargets that generator's five-reference resume/assertion
guards to the declared arm.  No model is retrained by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
IMU_ROOT = REPOSITORY_ROOT / "methods" / "generation" / "imu"
CACHE_RUNNER = IMU_ROOT / "scripts" / "generate_user_cache.py"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "generator"
ACTIONS = ("tap", "scroll", "swipe", "pinch")
REFERENCE_COUNTS = (0, 1, 3, 5, 8)
SAMPLE_STEPS = {"tap": 240, "scroll": 320, "swipe": 240, "pinch": 240}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def arm_definition(k_refs: int) -> dict[str, Any]:
    if k_refs not in REFERENCE_COUNTS:
        raise ValueError(f"unsupported reference count: {k_refs}")
    family = "zero_shot" if k_refs == 0 else "five_shot"
    protocol = "noshot_adv" if k_refs == 0 else "fewshot_adv"
    checkpoints: dict[str, dict[str, str]] = {}
    for action in ACTIONS:
        run = CHECKPOINT_ROOT / family / action
        model = run / "model.pt"
        config = run / "effective_config.json"
        for path in (model, config):
            if not path.is_file():
                raise FileNotFoundError(f"missing packaged generator input: {relative(path)}")
        checkpoints[action] = {
            "model": relative(model),
            "model_sha256": sha256(model),
            "config": relative(config),
            "config_sha256": sha256(config),
        }
    return {
        "schema": "actreal_reference_count_run_v1",
        "arm": f"k{k_refs}",
        "inference_k_refs": k_refs,
        "checkpoint_family": family,
        "checkpoint_training_protocol": protocol,
        "runtime_protocol": protocol,
        "override_stage": "before_action_runtime_construction",
        "retrained_for_arm": False,
        "interpretation": (
            "canonical zero-shot checkpoint"
            if k_refs == 0
            else (
                "canonical five-shot checkpoint"
                if k_refs == 5
                else "five-shot checkpoint evaluated with an inference-time reference-count override"
            )
        ),
        "actions": list(ACTIONS),
        "checkpoints": checkpoints,
        "runner": relative(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cache_runner": relative(CACHE_RUNNER),
        "cache_runner_sha256": sha256(CACHE_RUNNER),
    }


def action_specs(definition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {}
    for action, binding in definition["checkpoints"].items():
        run = Path(binding["model"]).parent
        specs[action] = {
            "config": binding["config"],
            "run_dir": run.as_posix(),
            "checkpoint": Path(binding["model"]).name,
            "sample_steps": SAMPLE_STEPS[action],
        }
    return specs


def retarget_cache_invariants(source: str, expected_refs: int) -> str:
    """Retarget only the canonical cache runner's three five-shot guards."""

    rewrites = {
        'if int(meta.get("ref_count", -1)) != 5:':
            f'if int(meta.get("ref_count", -1)) != {expected_refs}:',
        "if refs.shape != (5,) or np.any(refs < 0) or len(np.unique(refs)) != 5:":
            f"if refs.shape != ({expected_refs},) or np.any(refs < 0) "
            f"or len(np.unique(refs)) != {expected_refs}:",
        "if ref_count != 5 or used_refs.shape != (5,) or np.any(used_refs < 0):":
            f"if ref_count != {expected_refs} or used_refs.shape != ({expected_refs},) "
            "or np.any(used_refs < 0):",
    }
    rewritten = source
    for original, replacement in rewrites.items():
        occurrences = rewritten.count(original)
        if occurrences != 1:
            raise RuntimeError(
                "canonical cache invariant moved; refusing an unreviewed rewrite: "
                f"expected one occurrence, found {occurrences}: {original}"
            )
        rewritten = rewritten.replace(original, replacement)
    return rewritten


def run_cache(args: argparse.Namespace, definition: dict[str, Any]) -> None:
    sys.path.insert(0, str(IMU_ROOT))
    from android_imu_layer.diffusion_generator import (  # noqa: PLC0415
        AndroidIMUDiffusionLayer as BaseLayer,
    )

    specs = action_specs(definition)
    k_refs = int(definition["inference_k_refs"])
    protocol = str(definition["runtime_protocol"])

    class ReferenceCountLayer(BaseLayer):
        def __init__(self, *positional: Any, **keyword: Any):
            keyword["protocol"] = protocol
            keyword["action_specs"] = specs
            keyword["device"] = args.device
            super().__init__(*positional, **keyword)
            original_protocol_cfg = self._fg["protocol_cfg"]

            def protocol_cfg(config: dict[str, Any], name: str) -> dict[str, Any]:
                resolved = dict(original_protocol_cfg(config, name))
                resolved["fewshot"] = k_refs > 0
                resolved["k_refs"] = k_refs
                return resolved

            # _get_runtime consults this function before constructing the
            # UserRefBank and model conditioning inputs.
            self._fg["protocol_cfg"] = protocol_cfg

    source = CACHE_RUNNER.read_text(encoding="utf-8")
    source = retarget_cache_invariants(source, k_refs)
    namespace: dict[str, Any] = {
        "__file__": str(CACHE_RUNNER),
        "__name__": "actreal_reference_count_cache",
        "__package__": None,
    }
    exec(compile(source, str(CACHE_RUNNER), "exec"), namespace)  # noqa: S102
    namespace["AndroidIMUDiffusionLayer"] = ReferenceCountLayer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out_dir / "reference_count_protocol.json"
    receipt_path.write_text(
        json.dumps(definition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    forwarded = [
        CACHE_RUNNER.name,
        "--out-dir", str(args.out_dir),
        "--actions", *args.actions,
        "--splits", *args.splits,
        "--samples-per-user-action", str(args.samples_per_user_action),
        "--coverage-mode", args.coverage_mode,
        "--samples-per-active-len", str(args.samples_per_active_len),
        "--seed", str(args.seed),
        "--shard-index", str(args.shard_index),
        "--num-shards", str(args.num_shards),
    ]
    if args.sample_steps is not None:
        forwarded += ["--sample-steps", str(args.sample_steps)]
    if args.max_users_per_split is not None:
        forwarded += ["--max-users-per-split", str(args.max_users_per_split)]
    if args.overwrite:
        forwarded.append("--overwrite")

    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        # Checkpoint/config paths deliberately stay repository-relative so the
        # generated metadata does not capture a private host path.
        os.chdir(REPOSITORY_ROOT)
        sys.argv = forwarded
        namespace["main"]()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-refs", type=int, choices=REFERENCE_COUNTS, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=list(ACTIONS))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--samples-per-user-action", type=int, default=200)
    parser.add_argument(
        "--coverage-mode",
        choices=("random", "active_len_stratified"),
        default="random",
    )
    parser.add_argument("--samples-per-active-len", type=int, default=2)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-users-per-split", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the hash-pinned arm definition without generating samples",
    )
    args = parser.parse_args()
    if not args.describe and args.out_dir is None:
        parser.error("--out-dir is required unless --describe is used")
    return args


def main() -> int:
    args = parse_args()
    definition = arm_definition(args.k_refs)
    if args.describe:
        print(json.dumps(definition, indent=2, sort_keys=True))
        return 0
    run_cache(args, definition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
