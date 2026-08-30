#!/usr/bin/env python3
"""Freeze the genuine source-cluster -> raw touch binding map once."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audit import sha256_file  # noqa: E402
from pipeline.genuine_touch_recovery import (  # noqa: E402
    load_genuine_touch_bindings,
)

SCHEMA = "hmog_genuine_binding_cache_v2_physical_rebind"
ALIGNMENT_TOLERANCE_MS = 20.0

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT / "licensed_input" / "hmog" / "native_genuine_manifest.jsonl"
)
DEFAULT_LEDGER = (
    REPO_ROOT / "licensed_input" / "hmog" / "native_event_dispatch_quality.jsonl"
)


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _native_physical_identity(
    native_genuine_manifest: Path,
) -> dict[tuple[str, str, str, int], dict[str, int]]:
    """Map each planned native event to its physical identity."""

    identity: dict[tuple[str, str, str, int], dict[str, int]] = {}
    for session in _jsonl(native_genuine_manifest):
        audit = json.loads(
            Path(str(session["native_session_audit"])).read_text(encoding="utf-8")
        )
        for ordinal, event in enumerate(
            audit.get("event_plan", {}).get("events", [])
        ):
            identity[
                (
                    str(session["user_id"]),
                    str(session["session_id"]),
                    str(event["event_id"]),
                    int(ordinal),
                )
            ] = {
                "activity_id": int(event["activity_id"]),
                "orientation_id": int(event["orientation_id"]),
                "label_start_ms": int(event["start_timestamp_ns"]) // 1_000_000,
            }
    return identity


def _archive_index(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return {
        "activity_id": np.asarray(archive["activity_id"]),
        "orientation_id": np.asarray(archive["orientation_id"]),
        "label_start_ms": np.asarray(archive["label_start_ms"], dtype=np.float64),
        "user_id": np.asarray(archive["user_id"]),
        "session_id": np.asarray(archive["session_id"]),
        "event_id": np.asarray(archive["event_id"]),
    }


def _numeric_user_id(value: str) -> int:
    if not value.startswith("hmog_u"):
        raise SystemExit(f"invalid HMOG user ID: {value}")
    return int(value.removeprefix("hmog_u"))


def _numeric_session_id(value: str) -> int:
    if "_s" not in value:
        raise SystemExit(f"invalid HMOG session ID: {value}")
    return int(value.rsplit("_s", 1)[1])


def _repoint(
    bindings: dict[str, Any],
    *,
    native_genuine_manifest: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Re-point every binding at its physically matching archive event."""

    identity = _native_physical_identity(native_genuine_manifest)
    archives: dict[Path, dict[str, np.ndarray]] = {}
    repaired: dict[str, dict[str, Any]] = {}
    stats: dict[str, Counter] = {}
    for cluster_id, binding in bindings.items():
        source = Path(binding.raw_trajectory_source).resolve()
        if source not in archives:
            archives[source] = _archive_index(source)
        archive = archives[source]
        physical = identity.get(
            (
                binding.user_id,
                binding.session_id,
                binding.source_event_id,
                binding.source_event_ordinal,
            )
        )
        counter = stats.setdefault(binding.action, Counter())
        original = int(binding.raw_trajectory_event_index)
        index = original
        if physical is None:
            state = "unplanned"
        else:
            candidates = np.flatnonzero(
                (archive["activity_id"] == physical["activity_id"])
                & (archive["orientation_id"] == physical["orientation_id"])
                & (
                    np.abs(
                        archive["label_start_ms"] - physical["label_start_ms"]
                    )
                    <= ALIGNMENT_TOLERANCE_MS
                )
            )
            if len(candidates) == 1:
                index = int(candidates[0])
                # activity_id was measured to determine (user, session)
                # uniquely in every archive.  Enforce it rather than trust it,
                # so a future re-freeze cannot silently cross users.
                if int(archive["user_id"][index]) != _numeric_user_id(
                    binding.user_id
                ) or int(archive["session_id"][index]) != _numeric_session_id(
                    binding.session_id
                ):
                    raise SystemExit(
                        f"{cluster_id}: physical match crossed user/session"
                    )
                state = "unique"
            elif len(candidates) == 0:
                state = "absent_from_archive"
            else:
                state = "ambiguous"
        counter[state] += 1
        if state == "unique":
            counter["shift_%+d" % (index - original)] += 1
        repaired[cluster_id] = {
            "raw_trajectory_event_index": index,
            "physical_match": state,
            "ordinal_raw_trajectory_event_index": original,
            "index_shift": index - original,
            # The ledger's own `source_event_id` names the event the ordinal
            # index pointed at, which for 35% of bindings is a different, also
            # existing archive row.  Recording the corrected event's own archive
            # id lets a consumer check the corrected pointer instead of
            # re-checking the claim the rebinding just superseded.
            "raw_trajectory_event_id": (
                str(np.asarray(archive["event_id"][index]).item())
                if 0 <= index < len(archive["event_id"])
                else ""
            ),
        }
    summary = {
        action: {key: int(value) for key, value in sorted(counter.items())}
        for action, counter in sorted(stats.items())
    }
    return repaired, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-genuine-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dispatch-quality-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache directory in place.",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output}")
        for child in sorted(output.iterdir()):
            child.unlink()

    started = time.perf_counter()
    bindings = load_genuine_touch_bindings(
        native_genuine_manifest=args.native_genuine_manifest,
        dispatch_quality_ledger=args.dispatch_quality_ledger,
    )
    repaired, rebind_summary = _repoint(
        bindings, native_genuine_manifest=args.native_genuine_manifest
    )
    elapsed = time.perf_counter() - started

    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "genuine_bindings.jsonl"
    by_action: dict[str, int] = {}
    by_split: dict[str, int] = {}
    sources: set[str] = set()
    with rows_path.open("x", encoding="utf-8") as handle:
        for cluster_id in sorted(bindings):
            b = bindings[cluster_id]
            fix = repaired[cluster_id]
            by_action[b.action] = by_action.get(b.action, 0) + 1
            by_split[b.split] = by_split.get(b.split, 0) + 1
            sources.add(str(b.raw_trajectory_source))
            handle.write(
                json.dumps(
                    {
                        "source_cluster_id": b.source_cluster_id,
                        "user_id": b.user_id,
                        "split": b.split,
                        "session_id": b.session_id,
                        "action": b.action,
                        "source_event_id": b.source_event_id,
                        "source_event_ordinal": b.source_event_ordinal,
                        "orientation_id": b.orientation_id,
                        "start_sample": b.start_sample,
                        "end_sample": b.end_sample,
                        "duration_ms": b.duration_ms,
                        "raw_trajectory_source": str(b.raw_trajectory_source),
                        "raw_trajectory_source_sha256": b.raw_trajectory_source_sha256,
                        # Physically re-pointed; see the module docstring.
                        "raw_trajectory_event_index": fix[
                            "raw_trajectory_event_index"
                        ],
                        "physical_match": fix["physical_match"],
                        "ordinal_raw_trajectory_event_index": fix[
                            "ordinal_raw_trajectory_event_index"
                        ],
                        "index_shift": fix["index_shift"],
                        "raw_trajectory_event_id": fix[
                            "raw_trajectory_event_id"
                        ],
                        # The ledger digest describes the ORDINAL index, so it
                        # no longer identifies the re-pointed event.  Kept under
                        # a name that says so rather than silently reused.
                        "ledger_raw_event_sha256": b.raw_event_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    release = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "bindings": len(bindings),
        "by_action": dict(sorted(by_action.items())),
        "by_split": dict(sorted(by_split.items())),
        "physical_rebind": {
            "reason": (
                "the native/detector and raw-trajectory pipelines both name "
                "events by ordinal position in differently filtered lists, so "
                "equal numeric IDs can name different physical events"
            ),
            "match_key": "activity_id + orientation_id + label_start_ms",
            "tolerance_ms": ALIGNMENT_TOLERANCE_MS,
            "nothing_dropped": True,
            "absent_keeps_ordinal_index": True,
            "by_action": rebind_summary,
        },
        "raw_trajectory_sources": sorted(sources),
        "native_genuine_manifest": str(args.native_genuine_manifest.resolve()),
        "native_genuine_manifest_sha256": sha256_file(args.native_genuine_manifest),
        "dispatch_quality_ledger": str(args.dispatch_quality_ledger.resolve()),
        "dispatch_quality_ledger_sha256": sha256_file(args.dispatch_quality_ledger),
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "build_wall_seconds": round(elapsed, 2),
    }
    (output / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {k: v for k, v in release.items() if k != "raw_trajectory_sources"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
