#!/usr/bin/env python3
"""Score every attack with the release's own frozen detectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
METHOD_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(METHOD_ROOT))

from release_registry import (  # noqa: E402
    ACTIONS,
    DETECTOR_MODELS,
    DETECTORS,
    MODALITIES,
    action_to_bundle,
    frozen_threshold,
)

DEEP_DETECTORS = frozenset(("behaveformer_stdat", "authconformer"))


def load_threshold(action: str, modality: str, detector: str) -> float:
    """The FRR=5% operating point the release selected on its development split."""
    return float(frozen_threshold(action, modality, detector)["frr5"])


def score_cell(manifest: Path, action: str, modality: str, detector: str,
               device_name: str) -> dict:
    """Run one frozen detector over one attack's test split."""

    import torch
    from detection.core.event_pad import _load_manifest, load_event_partition
    from detection.core.event_detectors import classical_scores
    from detection.core.formal_event_pad import _feature_matrix, _score_deep

    # The manifest is read directly rather than through
    # `_load_manifest_after_pre_test_freeze`.  That guard exists to stop a
    rows = _load_manifest(manifest)
    test = load_event_partition(rows["test"])
    model_dir = DETECTOR_MODELS / f"{action}__{modality}__{detector}"

    if detector in DEEP_DETECTORS:
        from detection.core.event_detectors import build_deep_detector

        device = torch.device(device_name)
        checkpoint = torch.load(model_dir / "checkpoint.pt", map_location=device,
                                weights_only=False)
        model = build_deep_detector(detector, modality, action).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        normalizer = checkpoint["normalizer"]
        scores, _latency, metadata = _score_deep(
            model, test, action=action, modality=modality, device=device,
            mean=np.asarray(normalizer["mean"], dtype=np.float32),
            std=np.asarray(normalizer["std"], dtype=np.float32),
        )
    else:
        import joblib

        model = joblib.load(model_dir / "model.joblib")
        features, metadata = _feature_matrix(
            test, action=action, modality=modality, detector=detector
        )
        scores = classical_scores(model, features)

    cut = load_threshold(action, modality, detector)
    labels = np.asarray(metadata.labels)
    scores = np.asarray(scores, dtype=np.float64)
    fake = labels == 1
    genuine = labels == 0
    if not fake.any() or not genuine.any():
        raise SystemExit(f"{action}/{modality}/{detector}: test split lost a class")
    return {
        "far": float(np.mean(scores[fake] < cut)),
        "frr": float(np.mean(scores[genuine] >= cut)),
        "threshold": cut,
        "fake_events": int(fake.sum()),
        "genuine_events": int(genuine.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-root", type=Path, required=True,
                        help="a built baseline directory containing bundle_manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--actions", nargs="+", default=list(ACTIONS))
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    args = parser.parse_args()

    manifest_path = args.attack_root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{args.attack_root} has no bundle_manifest.json")
    bundle_manifest = json.loads(manifest_path.read_text())
    results, skipped = {}, []
    for action in args.actions:
        for modality in args.modalities:
            bundle = action_to_bundle(modality)[action]
            entry = bundle_manifest["bundles"].get(bundle)
            if entry is None or action not in entry["owned_actions"]:
                skipped.append(f"{action}/{modality}: not owned by {bundle}")
                continue
            tally = entry.get("swapped", {}).get(action, {})
            swapped = tally.get("imu_swapped", 0) + tally.get("trajectory_swapped", 0)
            if tally.get("fake", 0) and not swapped:
                skipped.append(f"{action}/{modality}: declined by this baseline")
                continue
            manifest = Path(entry["output"]) / "event_manifest.jsonl"
            for detector in DETECTORS:
                key = f"{action}__{modality}__{detector}"
                try:
                    results[key] = score_cell(
                        manifest, action, modality, detector, args.device
                    )
                    print(f"  {key}: FAR {results[key]['far']:.3f}", flush=True)
                except Exception as error:  # noqa: BLE001
                    skipped.append(f"{key}: {type(error).__name__}: {error}")
                    print(f"  {key}: SKIPPED ({error})", flush=True)

    values = [cell["far"] for cell in results.values()]
    payload = {
        "attack": args.attack_root.name,
        "protocol": (
            "the release's own fitted detectors and FRR=5% thresholds, applied "
            "without retraining or re-selection"
        ),
        "cells": results,
        "skipped": skipped,
        "median_far": float(np.median(values)) if values else None,
        "cells_at_or_above_0.60": sum(1 for v in values if v >= 0.6),
        "cells_scored": len(values),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(values)} cells, median FAR "
          f"{payload['median_far'] if values else 'n/a'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
