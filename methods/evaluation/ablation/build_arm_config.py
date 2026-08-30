#!/usr/bin/env python3
"""Derive one ablation arm's training config from the released run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_ROOT = REPOSITORY_ROOT / "checkpoints" / "generator" / "five_shot"

RELEASED_MAX_GRAD_RATIO = {"tap": 0.6, "scroll": 0.5, "swipe": 0.5, "pinch": 0.5}


def load_arms() -> dict:
    return json.loads((HERE / "arms.json").read_text(encoding="utf-8"))


def released_run(action: str) -> Path:
    """Return the packaged, immutable five-shot generator for this action."""
    run = GENERATOR_ROOT / action
    if not (run / "effective_config.json").is_file() or not (run / "model.pt").is_file():
        raise SystemExit(f"missing packaged generator for {action}: {run}")
    return run


def get_path(tree: dict, dotted: str):
    node = tree
    for part in dotted.split(".")[:-1]:
        node = node.get(part, {})
    return node.get(dotted.split(".")[-1])


def set_path(tree: dict, dotted: str, value) -> None:
    node = tree
    for part in dotted.split(".")[:-1]:
        node = node.setdefault(part, {})
    node[dotted.split(".")[-1]] = value


def assert_base_is_release(action: str, config: dict) -> None:
    adv = config.get("adv", {})
    critics = adv.get("critics", {})
    problems = []
    for name in ("feature", "waveform", "set"):
        if critics.get(name) is not True:
            problems.append(f"adv.critics.{name}={critics.get(name)!r}, expected True")
    if adv.get("project_conflicts") is not True:
        problems.append(
            f"adv.project_conflicts={adv.get('project_conflicts')!r}, expected True"
        )
    expected_ratio = RELEASED_MAX_GRAD_RATIO[action]
    if adv.get("max_grad_ratio") != expected_ratio:
        problems.append(
            f"adv.max_grad_ratio={adv.get('max_grad_ratio')!r}, expected {expected_ratio}"
        )
    if problems:
        raise SystemExit(
            f"{action}: the base config is not the release's.\n  "
            + "\n  ".join(problems)
            + "\nThis looks like another arm's config. Refusing to build on it, "
            "because the resulting row would differ from the release in more than "
            "the one component it claims."
        )


def is_absent(value) -> bool:
    """Whether the component is already off in the released config."""
    if value is None or value is False:
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def build(arm_id: str, action: str, out_dir: Path) -> dict:
    spec = load_arms()
    arm = next((a for a in spec["arms"] if a["id"] == arm_id), None)
    if arm is None:
        raise SystemExit(f"unknown arm {arm_id!r}; known: "
                         f"{[a['id'] for a in spec['arms']]}")
    if action not in spec["actions"]:
        raise SystemExit(f"unknown action {action!r}")

    source = arm["generator_source"][action]
    if source != "self":
        raise SystemExit(
            f"{arm_id}/{action}: arms.json routes this action's generator to "
            f"'{source}', not to a run of its own, so there is no config to build. "
            f"It is still sampled and scored inside the arm, from that source."
        )

    run = released_run(action)
    config = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    assert_base_is_release(action, config)

    before, after, no_ops = {}, {}, []
    for dotted, value in arm["config_change"].items():
        current = get_path(config, dotted)
        # A multi-key arm may legitimately find one mechanism already disabled.
        # Record that fact, but refuse an arm where nothing changes because such
        # a null comparison must be declared rather than discovered after scoring.
        if is_absent(current):
            no_ops.append(dotted)
        before[dotted] = current
        set_path(config, dotted, value)
        after[dotted] = get_path(config, dotted)
    if len(no_ops) == len(arm["config_change"]):
        raise SystemExit(
            f"{arm_id}/{action}: every key this arm removes is already off in the "
            f"released config ({', '.join(no_ops)}), so the arm would be identical "
            f"to the release. Declare {action}'s generator_source as 'release' for "
            f"this arm in arms.json if that is what you want."
        )
    config.setdefault("adv", {})["enabled"] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"config_{action}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    record = {
        "arm": arm_id,
        "paper_row": arm["paper_row"],
        "action": action,
        "removes": arm["removes"],
        "released_run": str(run),
        "epochs": int(config.get("train", {}).get("epochs", 0)),
        "batch_size": int(config.get("train", {}).get("batch_size", 0)),
        "before": before,
        "after": after,
        "keys_already_off_in_release": no_ops,
        "mechanisms_actually_removed": len(arm["config_change"]) - len(no_ops),
        "base_asserted": {
            "critics": config["adv"]["critics"],
            "project_conflicts_expected": True,
            "max_grad_ratio_expected": RELEASED_MAX_GRAD_RATIO[action],
        },
        "config": str(config_path),
        "run_name": f"{action}_objabl_{arm_id.replace('-', '_')}",
    }
    plan = out_dir / "plan.json"
    existing = json.loads(plan.read_text(encoding="utf-8")) if plan.is_file() else []
    keyed = {(r["arm"], r["action"]): r for r in existing}
    keyed[(arm_id, action)] = record
    plan.write_text(
        json.dumps(sorted(keyed.values(), key=lambda r: (r["arm"], r["action"])), indent=1),
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = build(args.arm, args.action, args.out)
    print(f"{record['arm']}/{record['action']}: {record['before']} -> {record['after']}"
          f"  ({record['epochs']} epochs, base {Path(record['released_run']).name})")
    print(f"  -> {record['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
