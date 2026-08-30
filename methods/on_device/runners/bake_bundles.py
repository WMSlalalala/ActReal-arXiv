#!/usr/bin/env python3
"""Bake playable actions for the local package.

Each bundle is one generated inertia window plus a human touch donor retimed
onto it.  The window comes from the pre-generated grid, so it carries its own
pre-roll -- the hand already moving before it reaches the glass -- and its
duration is a point on the grid the runtime law reads against.

The pool the donors come from is 3.9 GB and stays on the server; what ships is
these bundles.

    python methods/on_device/runners/bake_bundles.py --actions tap,scroll
    python methods/on_device/runners/bake_bundles.py --released-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from actreal import config
from actreal.bundle import compile_bundle
from actreal.event_store import EventStore
from actreal.imu_cache import CacheKey, ImuCache, duration_for_bin
from actreal.imu_source import load_cached_window
from actreal.mapping import ScreenMapping

# Where each action is aimed while baking.  Runtime re-aims every bundle, so
# this only has to be somewhere legal on the source screen.
BAKE_TARGETS = {
    "tap": ((540.0, 1000.0), None),
    "scroll": ((540.0, 1500.0), (540.0, 600.0)),
    "swipe": ((250.0, 1100.0), (830.0, 1100.0)),
    "keystroke": ((540.0, 1400.0), None),
}

# How far a donor may be stretched or squeezed to land on a window's duration.
# Beyond this the gesture is not that donor's timing any more, so a closer
# donor is found instead of pretending.
MAX_RETIME_FACTOR = 2.5


def donor_pool_from_json(path: Path, action: str):
    """Load the victim's touch tracks from the extracted JSON.

    This is the path the local package takes: the 3.9 GB event pool stays on
    the server, and what ships is one victim's tracks already de-held into the
    events Android delivered.
    """

    from actreal.touch_track import TouchPoint, TouchTrack

    data = json.loads(path.read_text())
    if data.get("schema_version") != "actreal_touch_donors_v1":
        raise ValueError(f"{path.name} is not a donor file")
    if data.get("action") != action:
        raise ValueError(f"{path.name} holds {data.get('action')!r}, not {action!r}")

    tracks = []
    for row in data["tracks"]:
        track = TouchTrack(
            action=action,
            points=[
                TouchPoint(
                    t_ms=float(p["t_ms"]),
                    x=float(p["x"]),
                    y=float(p["y"]),
                    pressure=float(p["pressure"]),
                    size=float(p.get("size", 0.0)),
                    pointer_id=int(p.get("pointer_id", 0)),
                    action=str(p["action"]),
                )
                for p in row["points"]
            ],
            orientation_id=int(row.get("orientation_id", 0)),
            screen_w=float(row["screen_w"]),
            screen_h=float(row["screen_h"]),
            source=f"donor:{row['event_id']}",
        )
        tracks.append((track.duration_ms, track, row["event_id"]))
    tracks.sort(key=lambda r: r[0])
    return tracks


class DonorPool:
    """Human touch tracks for one action, indexed by how long they last.

    Restricted to one victim.  The duration law is that victim's, and a session
    that borrowed somebody else's hand for the trajectory while timing it with
    this victim's curve would belong to neither of them.
    """

    def __init__(
        self,
        action: str,
        rng: np.random.Generator,
        size: int = 240,
        victim: str | None = None,
        donors_dir: "Path | None" = None,
    ):
        self.action = action
        self.victim = victim
        self.tracks = []
        self._rng = rng
        if donors_dir is not None:
            self.tracks = donor_pool_from_json(Path(donors_dir) / f"{action}.json", action)
            if not self.tracks:
                raise LookupError(f"no {action} donors in {donors_dir}")
            return
        store = EventStore(action)
        shards = store.shards
        if victim is not None:
            shards = [p for p in shards if p.stem == victim]
            if not shards:
                raise LookupError(
                    f"no shard for victim {victim!r}; "
                    f"have {[p.stem for p in store.shards[:3]]}..."
                )
        seen = 0
        for shard in shards:
            for event in store.iter_events(limit=200, shards=[shard]):
                try:
                    track = event.touch_track()
                except ValueError:
                    continue
                if track.duration_ms <= 0:
                    continue
                self.tracks.append((track.duration_ms, track, event.event_id))
                seen += 1
            if seen >= size:
                break
        if not self.tracks:
            raise LookupError(f"no usable {action} donors for {victim or 'any victim'}")
        self.tracks.sort(key=lambda row: row[0])
        self._rng = rng

    def nearest(self, duration_ms: float):
        """The donor whose own timing is closest to what is being asked for."""

        best = min(self.tracks, key=lambda row: abs(row[0] - duration_ms))
        return best


def bake_from_cache(args, mapping, rng) -> list[dict]:
    cache = ImuCache(config.IMU_CACHE_ROOT)
    cache.connect()
    conn = cache.connect()
    out = Path(args.out)
    index: list[dict] = []

    for action in args.actions:
        rows = conn.execute(
            "SELECT DISTINCT action, orientation_id, duration_bin, direction "
            "FROM samples WHERE action=? ORDER BY duration_bin, direction",
            (action,),
        ).fetchall()
        if not rows:
            # No grid for this action.  With donors on hand the touch is still
            # the victim's, but the inertia has to come from a whole released
            # event, which starts at first contact and so carries no pre-roll.
            print(f"{action}: no cache cells, falling back to whole released events")
            index.extend(bake_from_released(args, mapping, rng, actions=[action]))
            continue
        donors = DonorPool(
            action, rng, victim=args.victim,
            donors_dir=Path(args.donors) if args.donors else None,
        )
        made = skipped = 0
        for row in rows:
            key = CacheKey(row[0], row[1], row[2], row[3] or None)
            hit = cache.lookup(key)
            if hit is None:
                continue
            window = load_cached_window(hit.path, action=action)
            target_ms = window.active_frames * window.period_ms
            donor_ms, donor, donor_id = donors.nearest(target_ms)
            factor = target_ms / donor_ms if donor_ms > 0 else 0.0
            if not (1 / MAX_RETIME_FACTOR <= factor <= MAX_RETIME_FACTOR):
                skipped += 1
                continue

            start, end = BAKE_TARGETS.get(action, ((540.0, 1000.0), None))
            track = donor.retime(target_ms).reanchor(start, end)
            try:
                bundle = compile_bundle(
                    action=action,
                    touch=track,
                    imu=window,
                    mapping=mapping,
                    provenance={
                        "donor_event_id": donor_id,
                        "donor_duration_ms": round(donor_ms, 2),
                        "retime_factor": round(factor, 4),
                        "imu_cache_cell": key.relative_dir().as_posix(),
                        "duration_bin": key.duration_bin,
                        "grid_duration_ms": duration_for_bin(action, key.duration_bin),
                        "direction": key.direction,

                    },
                )
            except ValueError as error:
                print(f"  skip {action} {key.relative_dir()}: {error}")
                skipped += 1
                continue

            name = f"{action}_{key.duration_bin:03d}_{key.direction or 'none'}_{bundle.bundle_id[:8]}.json"
            (out / name).write_text(json.dumps(bundle.to_json_dict(), indent=1))
            index.append({"file": name, **bundle.summary(), **bundle.provenance})
            made += 1
        print(f"{action}: {made} bundles from cache ({skipped} skipped)")
    cache.close()
    return index


def bake_from_released(args, mapping, rng, actions=None) -> list[dict]:
    """Fallback: whole released events, trajectory and inertia already paired.

    These carry no pre-roll -- the released observation starts at first contact
    -- so they demonstrate delivery but not the timing structure.
    """

    out = Path(args.out)
    index: list[dict] = []
    for action in actions or args.actions:
        store = EventStore(action)
        if args.victim:
            store.shards = [p for p in store.shards if p.stem == args.victim]
            if not store.shards:
                raise LookupError(f"no shard for victim {args.victim!r}")
        start, end = BAKE_TARGETS.get(action, ((540.0, 1000.0), None))
        made = 0
        for attempt in range(args.per_action * 20):
            if made >= args.per_action:
                break
            event = store.sample(rng)
            try:
                bundle = compile_bundle(
                    action=action,
                    touch=event.touch_track().reanchor(start, end),
                    imu=event.imu_window(),
                    mapping=mapping,
                    provenance={"released_event_id": event.event_id, "pre_roll": False},
                )
            except ValueError as error:
                print(f"  skip {action} {event.event_id}: {error}")
                continue
            name = f"{action}_{made:02d}_{bundle.bundle_id[:8]}.json"
            (out / name).write_text(json.dumps(bundle.to_json_dict(), indent=1))
            index.append({"file": name, **bundle.summary()})
            made += 1
        print(f"{action}: {made} bundles from released events")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY_ROOT / ".actreal-work" / "bundles"
    )
    parser.add_argument("--device-w", type=int, default=1080)
    parser.add_argument("--device-h", type=int, default=2424)
    parser.add_argument("--actions", default="tap,scroll")
    parser.add_argument("--per-action", type=int, default=3, help="released mode only")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--donors",
        default="",
        help="use extracted donor JSON instead of the release event pool",
    )
    parser.add_argument(
        "--victim",
        required=True,
        help="whose hand and whose timing; one person per package",
    )
    parser.add_argument(
        "--released-only",
        action="store_true",
        help="skip the cache and bake whole released events instead",
    )
    args = parser.parse_args()
    args.actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Only previous bundles are cleared.  The directory also holds sidecars --
    # the timing profile, the index -- and deleting those by name-guess is how
    # the profile went missing and two tests quietly turned into skips.
    for stale in out.glob("*.json"):
        try:
            schema = json.loads(stale.read_text()).get("schema_version")
        except (json.JSONDecodeError, AttributeError):
            continue
        if schema == "actreal_action_bundle_v1":
            stale.unlink()

    mapping = ScreenMapping.isotropic(device_w=args.device_w, device_h=args.device_h)
    rng = np.random.default_rng(args.seed)

    index = (
        bake_from_released(args, mapping, rng)
        if args.released_only
        else bake_from_cache(args, mapping, rng)
    )
    missing = sorted({a for a in args.actions} - {row.get("action") for row in index})
    if missing:
        print(f"no bundles produced for {missing}")

    (out / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "actreal_bundle_index_v1",
                "device_w": args.device_w,
                "device_h": args.device_h,
                "source": "released" if args.released_only else "imu_cache",
                "victim": args.victim,
                "mapping": mapping.as_dict(),
                "bundles": index,
            },
            indent=1,
        )
    )
    total = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"wrote {len(index)} bundles to {out} ({total/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
