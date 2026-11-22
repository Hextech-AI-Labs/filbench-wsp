"""Generate publication-quality MMLU Tagalog comparison figures.

Run with: python generate_publication_plots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # Enable reproducible headless rendering.
import matplotlib.pyplot as plt
import numpy as np


# Official scientific-figure-making palette.
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply the repository's minimalist publication style."""
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "text.usetex": style.use_tex,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def annotate_bars(ax: plt.Axes, bars: Iterable, fmt: str = "{:.2f}", fontsize: int = 12) -> None:
    """Print exact values directly above bars."""
    for container in bars:
        ax.bar_label(container, labels=[fmt.format(bar.get_height()) for bar in container],
                     padding=4, fontsize=fontsize, fontweight="medium")


def add_reference_grid(ax: plt.Axes) -> None:
    """Use a restrained horizontal reference grid without competing with data."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=PALETTE["neutral"], linewidth=0.9, alpha=0.65)


def contrast_text_color(hex_color: str) -> str:
    """Choose an annotation color that remains legible over a bar fill."""
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 150 else "black"


def make_grouped_bar(
    ax: plt.Axes,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    ylabel: str = "Value",
    colors: Sequence[str] | None = None,
    annotate: bool = False,
):
    """Render a validated, print-safe grouped bar chart."""
    values = [np.asarray(values, dtype=float) for values in series]
    if len(values) != len(labels) or any(item.ndim != 1 or len(item) != len(categories) for item in values):
        raise ValueError("Each series and label must align with the provided categories.")

    colors = colors or [PALETTE[key] for key in ("blue_main", "green_3", "red_strong", "teal", "violet")]
    if len(colors) < len(values):
        raise ValueError("Provide at least one color for each series.")

    positions = np.arange(len(categories))
    width = 0.23
    offset = (len(values) - 1) * width / 2
    containers = []
    hatches = ("", "//", "..", "xx", "\\\\")
    for index, (data, label, color) in enumerate(zip(values, labels, colors)):
        containers.append(
            ax.bar(
                positions + index * width - offset,
                data,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=1.25,
                hatch=hatches[index % len(hatches)],
            )
        )

    ax.set_xticks(positions, categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.tick_params(axis="both", which="major", width=2, length=6)
    add_reference_grid(ax)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.11), fontsize=13)
    if annotate:
        annotate_bars(ax, containers)
    return containers[-1]


def finalize_figure(
    fig: plt.Figure,
    out_path: str | Path,
    formats: Sequence[str] | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
) -> list[Path]:
    """Tighten layout and save the figure in the requested format(s)."""
    output = Path(out_path)
    requested_formats = list(formats) if formats else [output.suffix.lstrip(".") or "png"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2)
    saved = []
    for extension in requested_formats:
        destination = output.with_suffix(f".{extension}")
        fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
        saved.append(destination)
    if close:
        plt.close(fig)
    return saved


def make_model_comparison(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    make_grouped_bar(
        ax,
        categories=["Qwen3-1.7B", "Llama-3.2-1B", "Gemma-3-1B"],
        series=[[44.74, 21.93, 22.81], [41.23, 23.68, 21.93], [42.98, 24.56, 21.93]],
        labels=["Baseline", "LLM Edited", "Random Control"],
        ylabel="MMLU Tagalog Average (%)",
        colors=[PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["green_3"]],
        annotate=True,
    )
    ax.set_title(
        "Weight Editing vs. Random Control Across Three Models",
        y=1.23,
        pad=0,
        fontweight="bold",
    )
    return finalize_figure(fig, output_dir / "model_comparison.png", formats=["png"], dpi=300)[0]


def make_qwen_control_suite(output_dir: Path) -> Path:
    categories = ["Baseline", "LLM Edited", "Random Control", "Sign Flip", "Layer Shuffle"]
    values = np.asarray([44.74, 41.23, 42.98, 43.86, 43.86])
    colors = [PALETTE[key] for key in ("blue_main", "red_strong", "green_3", "violet", "teal")]
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    hatches = ("", "//", "..", "xx", "\\\\")
    bars = ax.bar(
        np.arange(len(categories)),
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.25,
        hatch=hatches,
    )
    ax.set_xticks(np.arange(len(categories)), categories)
    ax.set_ylabel("MMLU Tagalog Average (%)")
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.tick_params(axis="both", which="major", width=2, length=6)
    add_reference_grid(ax)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=12, fontweight="medium")
    baseline = values[0]
    for bar, value, color in zip(bars[1:], values[1:], colors[1:]):
        delta = value - baseline
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 2.1,
            f"Δ {delta:+.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=contrast_text_color(color),
        )
    ax.set_title("Qwen3-1.7B: The Full Control Suite", pad=22, fontweight="bold")
    return finalize_figure(fig, output_dir / "qwen_control_suite.png", formats=["png"], dpi=300)[0]


def main() -> None:
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.5))
    output_dir = Path("plots")
    for path in (make_model_comparison(output_dir), make_qwen_control_suite(output_dir)):
        print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
