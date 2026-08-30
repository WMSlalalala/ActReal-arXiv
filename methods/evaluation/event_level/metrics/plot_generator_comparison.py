#!/usr/bin/env python3
"""Plot the compact generator comparison used in the ActReal paper.

The script supports the 7.0-inch USENIX two-column layout and a 3.335-inch
single-column layout.  It reads the released aggregate JSON files rather than
duplicating paper values in the plotting code, validates the action/detector
coverage, and writes PDF, SVG, and PNG versions.

The numerical inputs are maintained with the USENIX paper source, so both the
result root and output stem are explicit command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

# Matplotlib tries to create a cache under ~/.config by default.  Keep plotting
# reproducible on read-only build servers without touching the user's home.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actreal-mpl-cache")
)

import matplotlib as mpl
import matplotlib.pyplot as plt


METRIC = "asr_at_frr5"

# A restrained, venue-like palette: one warm highlight for the proposed method,
# one cool variant for the zero-shot ablation, and a neutral external baseline.
# Color never carries meaning alone: every bar is directly labeled.
ACTREAL_FILL = "#D97973"
ZERO_SHOT_FILL = "#86B8C5"
BASELINE_FILL = "#C9CDD2"
TEXT_COLOR = "#2F2F2F"
# Keep the panel boundary visible in print without making three heavy boxes.
FRAME_COLOR = "#B8B8B8"
GRID_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class Point:
    tick_label: str
    value: float
    cells: int
    actreal: bool = False
    zero_shot: bool = False


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_panels(results_root: Path) -> Dict[str, List[Point]]:
    """Load and validate the exact aggregates shown in the comparison."""
    actreal_path = results_root / "event_level/actreal/aggregate_results.json"
    baseline_path = results_root / "event_level/baselines/aggregate_results.json"
    ablation_path = results_root / "ablation/reference_count/aggregate_results.json"

    actreal = read_json(actreal_path)
    baselines = read_json(baseline_path)
    ablation = read_json(ablation_path)

    modality = actreal["modality"]
    require(isinstance(modality, dict), f"Malformed modality block: {actreal_path}")

    baseline_rows = baselines["rows"]
    require(isinstance(baseline_rows, list), f"Malformed rows block: {baseline_path}")
    by_method = {row["method"]: row for row in baseline_rows}

    expected_methods = {
        "diffusion_ts_touch",
        "ghost_cursor_touch",
        "pyclick_touch",
        "diffusion_ts_imu",
        "imagen_time_imu",
        "tts_gan_imu",
        "diffusion_ts_independent_dual_stream",
    }
    require(
        expected_methods.issubset(by_method),
        "Missing baseline methods: "
        + ", ".join(sorted(expected_methods.difference(by_method))),
    )

    for method in expected_methods:
        row = by_method[method]
        require(row["detectors"] == 6, f"{method}: expected six detectors")
        require(
            row["cells"] == row["actions"] * row["detectors"],
            f"{method}: inconsistent action/detector cell count",
        )

    ablation_rows = ablation["rows"]
    require(isinstance(ablation_rows, list), f"Malformed rows block: {ablation_path}")
    by_arm = {row["arm"]: row for row in ablation_rows}
    require({"k5", "k0"}.issubset(by_arm), "Reference-count results lack k5 or k0")
    paired_cells = int(ablation["cells_per_arm"])
    require(paired_cells == 24, "Expected the paired IMU ablation to use 24 cells")
    require(ablation["detectors"] == 6, "Expected six detectors in the IMU ablation")
    require(len(ablation["actions"]) == 4, "Expected four gesture actions in IMU ablation")

    def actreal_point(key: str, label: str) -> Point:
        row = modality[key]
        require(row["cells"] == 30, f"ActReal {key}: expected 30 cells")
        return Point(label, float(row[METRIC]), int(row["cells"]), actreal=True)

    def baseline_point(method: str, label: str) -> Point:
        row = by_method[method]
        return Point(label, float(row[METRIC]), int(row["cells"]))

    panels = {
        "Touch trajectory": [
            actreal_point("trajectory_xytime", "ActReal"),
            baseline_point("diffusion_ts_touch", "Diffusion-\nTS"),
            baseline_point("ghost_cursor_touch", "ghost-\ncursor"),
            baseline_point("pyclick_touch", "pyclick"),
        ],
        "Six-axis IMU": [
            Point(
                "ActReal\n5-shot",
                float(by_arm["k5"][METRIC]),
                paired_cells,
                actreal=True,
            ),
            Point(
                "ActReal\nzero-shot",
                float(by_arm["k0"][METRIC]),
                paired_cells,
                actreal=True,
                zero_shot=True,
            ),
            baseline_point("imagen_time_imu", "ImagenTime"),
            baseline_point("diffusion_ts_imu", "Diffusion-\nTS"),
            baseline_point("tts_gan_imu", "TTS-GAN"),
        ],
        "Joint": [
            actreal_point("imu_trajectory_xytime", "ActReal"),
            baseline_point(
                "diffusion_ts_independent_dual_stream", "Diffusion-\nTS"
            ),
        ],
    }

    # Coverage differs across baselines and remains explicit in the result files.
    for points in panels.values():
        for point in points:
            require(
                point.cells in (24, 30),
                f"Unexpected coverage for {point.tick_label!r}: {point.cells}",
            )
            require(0.0 <= point.value <= 1.0, f"ASR outside [0, 1]: {point}")

    return panels


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
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Preserve the exact USENIX text width declared in figsize.
            "savefig.bbox": None,
        }
    )


def draw_panel(ax: plt.Axes, title: str, points: Sequence[Point]) -> None:
    xs = list(range(len(points)))

    colors = [
        ZERO_SHOT_FILL
        if point.zero_shot
        else ACTREAL_FILL
        if point.actreal
        else BASELINE_FILL
        for point in points
    ]
    bars = ax.bar(
        xs,
        [100.0 * point.value for point in points],
        width=0.62,
        color=colors,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )

    for bar, point in zip(bars, points):
        percent = 100.0 * point.value
        ax.annotate(
            f"{percent:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, percent),
            xytext=(0, 3.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold" if point.actreal and not point.zero_shot else "normal",
            color=TEXT_COLOR,
            clip_on=False,
        )

    ax.set_title(title, fontweight="bold", pad=4.5)
    ax.set_xlim(-0.55, len(points) - 0.45)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([point.tick_label for point in points], linespacing=0.93)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.tick_params(axis="x", length=0, pad=3.5, colors=TEXT_COLOR)
    ax.tick_params(
        axis="y",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        length=2.3,
        width=0.55,
        pad=2.0,
        colors=TEXT_COLOR,
    )
    ax.set_axisbelow(True)
    # Draw only internal grid lines so the 0 and 1 lines do not double the frame.
    for y_value in (20, 40, 60, 80):
        ax.axhline(
            y_value,
            color=GRID_COLOR,
            linewidth=0.45,
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )

    # USENIX papers use both open and boxed axes; when boxed, the recurring
    # convention is a thin, uniform, low-salience frame with no outer rectangle.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.5)

    for tick, point in zip(ax.get_xticklabels(), points):
        tick.set_fontweight("bold" if point.actreal else "normal")
        tick.set_color(TEXT_COLOR)


def plot_double_column(
    panels: Mapping[str, Sequence[Point]],
) -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.15),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 5, 2], "wspace": 0.28},
    )

    for index, (ax, (title, points)) in enumerate(zip(axes, panels.items())):
        panel_title = f"({chr(ord('a') + index)}) {title}"
        draw_panel(ax, panel_title, points)

    axes[0].set_ylabel("ASR (%)")
    fig.subplots_adjust(left=0.066, right=0.985, bottom=0.245, top=0.875)
    return fig


def draw_single_column_panel(
    ax: plt.Axes,
    title: str,
    points: Sequence[Point],
    *,
    show_xlabels: bool,
) -> None:
    ys = list(range(len(points)))
    colors = [
        ZERO_SHOT_FILL
        if point.zero_shot
        else ACTREAL_FILL
        if point.actreal
        else BASELINE_FILL
        for point in points
    ]
    bars = ax.barh(
        ys,
        [100.0 * point.value for point in points],
        height=0.58,
        color=colors,
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )

    for bar, point in zip(bars, points):
        percent = 100.0 * point.value
        ax.annotate(
            f"{percent:.1f}",
            (percent, bar.get_y() + bar.get_height() / 2.0),
            xytext=(3.2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold" if point.actreal and not point.zero_shot else "normal",
            color=TEXT_COLOR,
            clip_on=False,
        )

    ax.set_title(title, loc="left", fontweight="bold", pad=1.5)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(len(points) - 0.35, -0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            point.tick_label.replace("-\n", "-").replace("\n", " ")
            for point in points
        ]
    )
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        labelbottom=show_xlabels,
        length=2.1,
        width=0.5,
        pad=2.0,
        colors=TEXT_COLOR,
    )
    ax.tick_params(
        axis="y",
        left=False,
        right=False,
        length=0,
        pad=3.0,
        colors=TEXT_COLOR,
    )
    ax.set_axisbelow(True)
    for x_value in (20, 40, 60, 80):
        ax.axvline(
            x_value,
            color=GRID_COLOR,
            linewidth=0.45,
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.5)

    for tick, point in zip(ax.get_yticklabels(), points):
        tick.set_fontweight("bold" if point.actreal else "normal")
        tick.set_color(TEXT_COLOR)


def plot_single_column(
    panels: Mapping[str, Sequence[Point]],
) -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(
        3,
        1,
        # Exact USENIX column width, with the three panels packed vertically.
        figsize=(3.335, 2.48),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 5, 2]},
    )

    panel_items = list(panels.items())
    for index, (ax, (title, points)) in enumerate(zip(axes, panel_items)):
        panel_title = f"({chr(ord('a') + index)}) {title}"
        draw_single_column_panel(
            ax,
            panel_title,
            points,
            show_xlabels=index == len(panel_items) - 1,
        )

    axes[-1].set_xlabel("ASR (%)", labelpad=1.5)
    fig.subplots_adjust(
        left=0.300,
        right=0.970,
        bottom=0.125,
        top=0.940,
        hspace=0.35,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing event_level/ and ablation/ results",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Output path without an extension",
    )
    parser.add_argument(
        "--layout",
        choices=("double", "single"),
        default="single",
        help="USENIX double-column or single-column layout",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels = load_panels(args.results_root.resolve())
    figure = (
        plot_single_column(panels)
        if args.layout == "single"
        else plot_double_column(panels)
    )

    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "ActReal generator comparison",
        "Author": "Anonymous",
        "Creator": "plot_generator_comparison.py",
    }
    figure.savefig(output_stem.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(output_stem.with_suffix(".svg"), metadata={"Title": metadata["Title"]})
    figure.savefig(output_stem.with_suffix(".png"), dpi=args.dpi, metadata=metadata)
    plt.close(figure)

    for suffix in (".pdf", ".svg", ".png"):
        print(output_stem.with_suffix(suffix))


if __name__ == "__main__":
    main()
