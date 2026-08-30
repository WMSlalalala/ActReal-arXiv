#!/usr/bin/env python3
"""Resolve frozen ActReal models/thresholds and locally rebuilt event scores.

The public artifact keeps fitted detector state under ``checkpoints/`` but does
not package paper-result tables or test-score dumps.  Commands that need the
latter read a regenerated score tree from ``ACTREAL_EVENT_SCORE_ROOT`` (or the
repository-local ``.actreal-work`` default).
"""

from __future__ import annotations

import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DETECTOR_MODELS = REPO / "checkpoints" / "detectors"
BUNDLE_MAP = REPO / "data" / "event_level" / "ACTION_BUNDLE_MAP.json"
CELL_MAP = REPO / "methods" / "evaluation" / "event_level" / "common" / "release_cell_map.json"
EVENT_SCORE_ROOT = Path(os.environ.get(
    "ACTREAL_EVENT_SCORE_ROOT",
    REPO / ".actreal-work" / "event_level" / "actreal" / "cells",
))
if not EVENT_SCORE_ROOT.is_absolute():
    EVENT_SCORE_ROOT = REPO / EVENT_SCORE_ROOT

# HMOG-derived signal shards are not redistributable. An authorised user supplies
# their local bundle root explicitly when rebuilding the event-level experiments.
DATASETS = Path(os.environ.get("ACTREAL_EVENT_DATA_ROOT", REPO / "data" / "event_level"))

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = ("trajectory_xytime", "imu_only", "imu_trajectory_xytime")
DETECTORS = (
    "hmog_style_svm", "hmog_style_rf", "paper_svm", "paper_xgboost",
    "behaveformer_stdat", "authconformer",
)


def _score_file(directory: Path) -> Path | None:
    for name in ("test_scores.jsonl.gz", "test_scores.jsonl"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def cell_dir(action: str, modality: str, detector: str,
             score_root: str | Path | None = None) -> Path:
    """Return one regenerated event-score directory.

    ``score_root`` is intentionally separate from the immutable detector
    checkpoint tree.  This prevents analysis scripts from treating published
    aggregates as executable inputs.
    """
    root = EVENT_SCORE_ROOT if score_root is None else Path(score_root)
    if not root.is_absolute():
        root = REPO / root
    directory = root / f"{action}__{modality}__{detector}"
    if _score_file(directory) is None:
        raise FileNotFoundError(
            f"regenerated event scores are missing for {directory}; run the "
            "event-level scorer first or set ACTREAL_EVENT_SCORE_ROOT"
        )
    return directory


@lru_cache(maxsize=1)
def _cell_registry() -> dict[str, dict]:
    data = json.loads(CELL_MAP.read_text())
    cells = data.get("cells", {})
    if len(cells) != 90:
        raise ValueError(f"expected 90 released cells in {CELL_MAP}, got {len(cells)}")
    return cells


def frozen_threshold_path(action: str, modality: str, detector: str) -> Path:
    """Return the immutable development-selected threshold in checkpoints/."""
    name = f"{action}__{modality}__{detector}"
    try:
        relative = Path(_cell_registry()[name]["thresholds"])
    except KeyError as error:
        raise KeyError(f"no frozen threshold registered for {name}") from error
    path = relative if relative.is_absolute() else REPO / relative
    if not path.is_file():
        raise FileNotFoundError(f"frozen threshold is missing: {path}")
    return path


def frozen_threshold(action: str, modality: str, detector: str) -> dict:
    """Read and validate one frozen threshold receipt."""
    name = f"{action}__{modality}__{detector}"
    path = frozen_threshold_path(action, modality, detector)
    threshold = json.loads(path.read_text())
    registered = _cell_registry()[name]
    if threshold.get("score_direction") != "larger_is_more_fake":
        raise ValueError(f"unexpected score direction in {path}")
    if float(threshold["frr5"]) != float(registered["frr5"]):
        raise ValueError(f"threshold receipt disagrees with {CELL_MAP}: {path}")
    return threshold


@lru_cache(maxsize=1)
def _action_sources() -> dict[str, dict[str, str]]:
    data = json.loads(BUNDLE_MAP.read_text())
    actions = data.get("actions", {})
    if set(actions) != set(ACTIONS):
        raise ValueError(f"unexpected action set in {BUNDLE_MAP}: {sorted(actions)}")
    out = {}
    for action, entry in actions.items():
        if not isinstance(entry, dict) or set(entry) != set(MODALITIES):
            raise ValueError(f"{action}: expected one frozen source per modality")
        out[action] = {str(modality): str(bundle) for modality, bundle in entry.items()}
    return out


@lru_cache(maxsize=1)
def bundle_map() -> dict[str, tuple[str, ...]]:
    """Map each frozen source-bundle label to the actions it supplies."""
    owners: dict[str, set[str]] = {}
    for action, entries in _action_sources().items():
        for bundle in entries.values():
            owners.setdefault(bundle, set()).add(action)
    return {bundle: tuple(sorted(actions)) for bundle, actions in sorted(owners.items())}


def action_to_bundle(modality: str | None = None) -> dict[str, str]:
    """Return the frozen source bundle for each action in one modality."""
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    return {action: entries[modality] for action, entries in _action_sources().items()}


def bundle_dir(name: str) -> Path:
    directory = DATASETS / name
    if not (directory / "shards").is_dir():
        raise SystemExit(
            f"licensed event bundle {name!r} not found under {DATASETS}. "
            "Set ACTREAL_EVENT_DATA_ROOT to an authorised local export; the "
            "HMOG-derived signal shards are intentionally not redistributed."
        )
    return directory


def bundles() -> list[tuple[str, Path, tuple[str, ...]]]:
    return [(name, bundle_dir(name), actions) for name, actions in bundle_map().items()]


@lru_cache(maxsize=1)
def cell_sources() -> dict[tuple[str, str, str], str]:
    """Map each released cell to its frozen source-bundle label."""
    resolved = {}
    for cell, entry in _cell_registry().items():
        action, modality, detector = cell.split("__")
        resolved[(action, modality, detector)] = str(entry["bundle"])
    if len(resolved) != 90:
        raise ValueError(f"expected 90 released cells in {CELL_MAP}, got {len(resolved)}")
    return resolved


def reference_scores(score_root: str | Path | None = None) -> dict[tuple[str, str, str], dict]:
    """Recompute ASR/FRR from local scores at the frozen checkpoint threshold."""
    scores = {}
    for key, source in sorted(cell_sources().items()):
        action, modality, detector = key
        directory = cell_dir(action, modality, detector, score_root)
        path = _score_file(directory)
        assert path is not None
        threshold = frozen_threshold(action, modality, detector)
        cut = float(threshold["frr5"])
        fake, genuine = [], []
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                value = float(row.get("fake_high_score", row.get("score")))
                (fake if int(row["label"]) == 1 else genuine).append(value)
        if not fake or not genuine:
            raise ValueError(f"score cell lost a class: {directory}")
        scores[key] = {
            "far": sum(value < cut for value in fake) / len(fake),
            "frr": sum(value >= cut for value in genuine) / len(genuine),
            "source": source,
            "fake_events": len(fake),
            "genuine_events": len(genuine),
        }
    if set(scores) != set(cell_sources()):
        missing = sorted(set(cell_sources()) - set(scores))
        extra = sorted(set(scores) - set(cell_sources()))
        raise ValueError(f"result cells disagree with cell map: missing={missing}, extra={extra}")
    return scores


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-root", type=Path, default=None,
        help=("regenerated cell-score root; default: ACTREAL_EVENT_SCORE_ROOT "
              "or .actreal-work/event_level/actreal/cells"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    scores = reference_scores(args.score_root)
    print(f"regenerated reference cells: {len(scores)}/90")
    for modality in MODALITIES:
        values = [cell["far"] for key, cell in scores.items() if key[1] == modality]
        print(f"{modality}: mean ASR {sum(values) / len(values):.6f} ({len(values)} cells)")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {f"{a}__{m}__{d}": value for (a, m, d), value in scores.items()},
            indent=2, sort_keys=True,
        ) + "\n")


if __name__ == "__main__":
    main()
