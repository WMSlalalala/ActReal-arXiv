#!/usr/bin/env python3
"""Run selected standard three-input detector cells on physical GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EVENT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[3]
CELL_RUNNER = METHOD_ROOT / "detection" / "scripts" / "run_hmog_direct100k_cell.py"
PYTHON = Path("python3")
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = (
    "trajectory_xytime",
    "imu_only",
    "imu_trajectory_xytime",
)
DETECTORS = (
    "hmog_style_svm",
    "hmog_style_rf",
    "paper_svm",
    "paper_xgboost",
    "behaveformer_stdat",
    "authconformer",
)
# Only these two build a torch model.  The other four are sklearn estimators
# fitted with n_jobs=1 on numpy features, so they ignore the device entirely and
# would merely occupy a GPU queue slot without using the card.
DEEP_DETECTORS = frozenset(("behaveformer_stdat", "authconformer"))
SMOKE_MANIFEST_SCOPE = "balanced_small"
FULL_MANIFEST_SCOPE = "full"
COMPLETION_SCHEMA = "hmog_direct100k_detector_completion_v2"


class DirectDetectorError(RuntimeError):
    pass


def _load_dataset_release(manifest: Path, *, smoke: bool) -> dict[str, Any]:
    release_path = manifest.parent / "release.json"
    if not release_path.is_file():
        raise DirectDetectorError(f"dataset release is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (
        release.get("schema_version") != "hmog_direct100k_detector_dataset_v1"
        or Path(str(release.get("event_manifest", ""))).resolve() != manifest
        or release.get("acceptance_audit_used") is not False
        or release.get("clean_candidate_filter_used") is not False
        or release.get("pass_binding_used") is not False
    ):
        raise DirectDetectorError("dataset is not a direct100k release")
    if smoke:
        if release.get("mode") != "smoke":
            raise DirectDetectorError("--smoke requires a smoke dataset")
    elif (
        release.get("mode") != "full_100k"
        or int(release.get("fake_events", -1)) != 100_000
        or release.get("fake_selection") != "all_100000"
    ):
        raise DirectDetectorError(
            "full training requires the complete, unfiltered 100,000 events"
        )
    expected_scope = SMOKE_MANIFEST_SCOPE if smoke else FULL_MANIFEST_SCOPE
    declared_scope = release.get("manifest_scope")
    if declared_scope is not None and declared_scope != expected_scope:
        raise DirectDetectorError(
            f"dataset release manifest scope {declared_scope!r} does not match "
            f"requested scope {expected_scope!r}"
        )
    try:
        manifest_scopes = {
            str(row.get("scope", ""))
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
            if isinstance(row, dict)
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectDetectorError(
            f"dataset manifest is unreadable: {manifest}"
        ) from exc
    if manifest_scopes != {expected_scope}:
        raise DirectDetectorError(
            f"dataset manifest scopes {sorted(manifest_scopes)!r} do not match "
            f"requested scope {expected_scope!r}"
        )
    return release


def _run_device_queue(
    *,
    device: str,
    cells: "queue.Queue[tuple[str, str, str]]",
    manifest: Path,
    output: Path,
    logs: Path,
    epochs: int,
    bootstrap_replicates: int,
    seed: int,
    scope: str,
) -> list[dict[str, Any]]:
    """Drain a shared cell queue on one device until it is empty.

    Workers pull rather than receiving a fixed stripe, so a queue of slow cells
    cannot straggle behind an already-idle worker.
    """

    results: list[dict[str, Any]] = []
    while True:
        try:
            action, modality, detector = cells.get_nowait()
        except queue.Empty:
            return results
        cell_name = f"{action}__{modality}__{detector}"
        cell_output = output / "cells" / cell_name
        log_path = logs / f"{cell_name}.log"
        command = [
            str(PYTHON),
            str(CELL_RUNNER),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(cell_output),
            "--action",
            action,
            "--modality",
            modality,
            "--detector",
            detector,
            "--scope",
            scope,
            "--device",
            device,
            "--epochs",
            str(epochs),
            "--bootstrap-replicates",
            str(bootstrap_replicates),
            "--seed",
            str(seed),
        ]
        environment = dict(os.environ)
        if device == "cpu":
            # Dozens of concurrent CPU cells that each default to one BLAS
            # thread per core would oversubscribe the machine.  The estimators
            # are already fitted with n_jobs=1, so one thread each loses
            # nothing and keeps the cells from fighting over the same cores.
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = "1"
        started = time.perf_counter()
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=EVENT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=environment,
            )
        elapsed = time.perf_counter() - started
        result = {
            "cell": cell_name,
            "action": action,
            "modality": modality,
            "detector": detector,
            "device": device,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "output_dir": str(cell_output),
            "log": str(log_path),
        }
        results.append(result)
        if completed.returncode != 0:
            raise DirectDetectorError(
                f"{cell_name} failed on {device}; see {log_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train selected actions x (trajectory, IMU+XY+time, "
            "trajectory+IMU) x 2 detectors."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", action="append", default=[], help="Repeat, e.g. cuda:0"
    )
    parser.add_argument(
        "--action",
        action="append",
        choices=ACTIONS,
        default=[],
        help=(
            "Action to run; repeat to select multiple actions. "
            "Defaults to all actions."
        ),
    )
    parser.add_argument(
        "--modality",
        action="append",
        choices=MODALITIES,
        default=[],
        help=(
            "Modality to run; repeat to select multiple. Defaults to all "
            "three. Use this to validate a change on one modality before "
            "spending the full grid."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=1,
        help=(
            "Epochs per cell for --smoke (default: 1). This keeps the cheap "
            "smoke default while allowing a longer small-dataset stress test."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=3,
        help=(
            "Concurrent deep cells per GPU. Each holds a few GB of a 48 GB "
            "card and spends its tail on CPU bootstrap, so more than one per "
            "device keeps the card busy."
        ),
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 8),
        help=(
            "Concurrent sklearn cells. These never touch a GPU and are fitted "
            "with n_jobs=1, so they scale with cores."
        ),
    )
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    devices = args.device or ["cuda:0", "cuda:1"]
    requested_actions = set(args.action)
    selected_actions = tuple(
        action
        for action in ACTIONS
        if not requested_actions or action in requested_actions
    )
    requested_modalities = set(args.modality)
    selected_modalities = tuple(
        modality
        for modality in MODALITIES
        if not requested_modalities or modality in requested_modalities
    )
    expected_cell_keys = {
        (action, modality, detector)
        for action in selected_actions
        for modality in selected_modalities
        for detector in DETECTORS
    }
    expected_cells = len(expected_cell_keys)
    if output.exists():
        raise DirectDetectorError(f"output already exists: {output}")
    if (
        args.epochs < 1
        or args.smoke_epochs < 1
        or args.bootstrap_replicates < 1
    ):
        raise DirectDetectorError("epochs and bootstrap replicates must be positive")
    dataset_release = _load_dataset_release(manifest, smoke=args.smoke)
    scope = SMOKE_MANIFEST_SCOPE if args.smoke else FULL_MANIFEST_SCOPE
    epochs = args.smoke_epochs if args.smoke else args.epochs
    bootstrap = 100 if args.smoke else args.bootstrap_replicates
    output.mkdir(parents=True)
    (output / "cells").mkdir()
    logs = output / "logs"
    logs.mkdir()
    cells = [
        (action, modality, detector)
        for action in selected_actions
        for modality in selected_modalities
        for detector in DETECTORS
    ]
    # Two pools over two disjoint halves of the grid: the deep cells hold a GPU,
    # the sklearn cells hold a core.  Each pool drains one shared queue, so a
    # worker that finishes early takes the next cell instead of idling.
    deep_cells: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
    classical_cells: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
    for cell in cells:
        (deep_cells if cell[2] in DEEP_DETECTORS else classical_cells).put(cell)
    workers: list[tuple[str, "queue.Queue[tuple[str, str, str]]"]] = []
    if deep_cells.qsize():
        workers.extend(
            (device, deep_cells)
            for device in devices
            for _ in range(args.workers_per_device)
        )
    if classical_cells.qsize():
        workers.extend(
            ("cpu", classical_cells)
            for _ in range(min(args.cpu_workers, classical_cells.qsize()))
        )
    print(
        f"{len(cells)} cells: {deep_cells.qsize()} deep over "
        f"{len(devices)}x{args.workers_per_device} GPU workers, "
        f"{classical_cells.qsize()} classical over "
        f"{sum(1 for d, _ in workers if d == 'cpu')} CPU workers",
        flush=True,
    )
    started = time.perf_counter()
    all_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(workers))
    ) as executor:
        futures = [
            executor.submit(
                _run_device_queue,
                device=device,
                cells=pending,
                manifest=manifest,
                output=output,
                logs=logs,
                epochs=epochs,
                bootstrap_replicates=bootstrap,
                seed=args.seed,
                scope=scope,
            )
            for device, pending in workers
        ]
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
    all_results.sort(key=lambda value: str(value["cell"]))
    completed_keys = {
        (
            str(value.get("action", "")),
            str(value.get("modality", "")),
            str(value.get("detector", "")),
        )
        for value in all_results
    }
    if (
        len(all_results) != expected_cells
        or completed_keys != expected_cell_keys
        or any(int(value.get("returncode", -1)) != 0 for value in all_results)
    ):
        raise DirectDetectorError(
            "training did not complete the exact requested action grid"
        )
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full_100k",
        "manifest_scope": scope,
        "manifest": str(manifest),
        "dataset_release": dataset_release,
        "devices": devices,
        "schedule": {
            "gpu_workers_per_device": int(args.workers_per_device),
            "cpu_workers": sum(1 for device, _ in workers if device == "cpu"),
            "deep_detectors": sorted(DEEP_DETECTORS),
        },
        "epochs": epochs,
        "bootstrap_replicates": bootstrap,
        "actions": list(selected_actions),
        "modalities": list(selected_modalities),
        "cells_expected": expected_cells,
        "cells_complete": len(all_results),
        "elapsed_seconds": time.perf_counter() - started,
        "cells": all_results,
    }
    completion_path = output / "completion.json"
    with completion_path.open("x", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except DirectDetectorError as exc:
        print(f"DIRECT DETECTOR RUN ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
