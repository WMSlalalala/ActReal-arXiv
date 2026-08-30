#!/usr/bin/env python3
"""Plot the target-reference-count ablation used in the ActReal paper.

The plot reads the released aggregate JSON, validates that every setting uses
the same four actions and six IMU-only detectors, and exports a USENIX
single-column figure as PDF, SVG, and PNG.

The aggregate input and output stem are explicit because numerical results are
maintained with the USENIX paper source rather than this code repository.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

# Keep Matplotlib from writing to a possibly read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actreal-mpl-cache")
)

import matplotlib as mpl
import matplotlib.pyplot as plt


METRIC = "asr_at_frr5"
EXPECTED_ARMS = ("k0", "k1", "k3", "k5", "k8")

LINE_COLOR = "#5F8791"
DEFAULT_FILL = "#D97973"
TEXT_COLOR = "#2F2F2F"
FRAME_COLOR = "#B8B8B8"
GRID_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class Result:
    k: int
    asr: float
    is_default: bool


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(path: Path) -> List[Result]:
    payload = read_json(path)
    require(
        payload.get("schema") == "actreal_imu_reference_count_ablation_v1",
        f"Unexpected schema in {path}",
    )
    require(payload.get("modality") == "imu_only", "Expected IMU-only results")
    require(payload.get("detectors") == 6, "Expected six detectors")
    require(payload.get("cells_per_arm") == 24, "Expected 24 cells per arm")

    actions = payload.get("actions")
    require(
        actions == ["tap", "scroll", "swipe", "pinch"],
        f"Unexpected action coverage: {actions}",
    )
    require(payload.get("reference_arm") == "k5", "Expected k=5 as default")

    rows = payload.get("rows")
    require(isinstance(rows, list), f"Malformed rows block in {path}")
    by_arm = {row["arm"]: row for row in rows}
    require(
        set(by_arm) == set(EXPECTED_ARMS),
        f"Expected arms {EXPECTED_ARMS}, found {tuple(sorted(by_arm))}",
    )

    results: List[Result] = []
    for arm in EXPECTED_ARMS:
        value = float(by_arm[arm][METRIC])
        require(0.0 <= value <= 1.0, f"ASR outside [0, 1] for {arm}: {value}")
        results.append(Result(int(arm[1:]), value, arm == "k5"))
    return results


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Liberation Serif",
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )


def draw(results: Sequence[Result]) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.335, 1.65))

    xs = [result.k for result in results]
    ys = [100.0 * result.asr for result in results]
    ax.plot(
        xs,
        ys,
        color=LINE_COLOR,
        linewidth=1.15,
        zorder=2,
    )

    ordinary = [result for result in results if not result.is_default]
    ax.scatter(
        [result.k for result in ordinary],
        [100.0 * result.asr for result in ordinary],
        s=23,
        marker="o",
        facecolor="white",
        edgecolor=LINE_COLOR,
        linewidth=1.0,
        zorder=3,
    )
    default = next(result for result in results if result.is_default)
    ax.scatter(
        [default.k],
        [100.0 * default.asr],
        s=34,
        marker="D",
        facecolor=DEFAULT_FILL,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    for result in results:
        ax.annotate(
            f"{100.0 * result.asr:.1f}",
            (result.k, 100.0 * result.asr),
            xytext=(0, 5.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=7.0,
            fontweight="bold" if result.is_default else "normal",
            clip_on=False,
        )

    ax.set_xlim(-0.35, 8.35)
    ax.set_ylim(76.0, 85.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0", "1", "3", "5\n(default)", "8"], linespacing=0.9)
    ax.set_yticks([76, 78, 80, 82, 84])
    ax.set_xlabel(r"Number of target-user references, $k$", labelpad=2.0)
    ax.set_ylabel("ASR (%)", labelpad=3.0)

    ax.tick_params(
        axis="x",
        top=False,
        bottom=True,
        length=2.2,
        width=0.5,
        pad=2.0,
        colors=TEXT_COLOR,
    )
    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        length=2.2,
        width=0.5,
        pad=2.0,
        colors=TEXT_COLOR,
    )
    ax.get_xticklabels()[3].set_fontweight("bold")

    ax.set_axisbelow(True)
    for y_value in (78, 80, 82, 84):
        ax.axhline(
            y_value,
            color=GRID_COLOR,
            linewidth=0.45,
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.5)

    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.31, top=0.87)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Reference-count aggregate JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output stem; PDF, SVG, and PNG are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    fig = draw(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.svg')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
