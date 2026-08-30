#!/usr/bin/env python3
"""Resolve the generator checkpoints packaged with the paper artifact.

The registry retains the original run and checkpoint names for provenance. The
public artifact stores inference-only weights under ``checkpoints/generator``;
callers therefore resolve the packaged directory rather than a private training
run tree.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
REGISTRY = Path(__file__).resolve().with_name("released_generators.json")
CHECKPOINT_ROOT = REPO / "checkpoints" / "generator"

ACTIONS = ("tap", "scroll", "swipe", "pinch")
PROTOCOLS = ("noshot_adv", "fewshot_adv")
PACKAGED_PROTOCOL = {
    "noshot_adv": "zero_shot",
    "fewshot_adv": "five_shot",
}


@lru_cache(maxsize=1)
def registry() -> dict:
    if not REGISTRY.is_file():
        raise SystemExit(f"missing released-generator registry: {REGISTRY}")
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return payload["runs"]


def resolve(action: str, protocol: str) -> tuple[Path, Path]:
    """Return ``(packaged run directory, inference checkpoint)``.

    ``effective_config.json`` lives in the returned directory so the ablation
    builders can derive their base configuration exactly as in the release.
    """

    if action not in ACTIONS:
        raise SystemExit(f"unsupported released action {action!r}")
    if protocol not in PACKAGED_PROTOCOL:
        raise SystemExit(f"unsupported released protocol {protocol!r}")
    if registry().get(action, {}).get(protocol) is None:
        raise SystemExit(f"no released run pinned for {action}/{protocol}")

    run = CHECKPOINT_ROOT / PACKAGED_PROTOCOL[protocol] / action
    checkpoint = run / "model.pt"
    config = run / "effective_config.json"
    missing = [path.name for path in (checkpoint, config) if not path.is_file()]
    if missing:
        raise SystemExit(
            f"packaged release for {action}/{protocol} is incomplete: "
            + ", ".join(missing)
        )
    return run, checkpoint


def available(action: str, protocol: str) -> bool:
    return (
        action in ACTIONS
        and protocol in PACKAGED_PROTOCOL
        and registry().get(action, {}).get(protocol) is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="resolve every packaged release checkpoint (the default action)",
    )
    parser.parse_args()

    for action in ACTIONS:
        for protocol in PROTOCOLS:
            if not available(action, protocol):
                print(f"  {action:10s} {protocol:12s} (none)")
                continue
            _run, checkpoint = resolve(action, protocol)
            original = registry()[action][protocol]
            print(
                f"  {action:10s} {protocol:12s} {checkpoint.relative_to(REPO)} "
                f"<- {original['run']}/{original['checkpoint']}"
            )


if __name__ == "__main__":
    main()
