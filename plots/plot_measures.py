"""Plot selected MDKP slack histograms with local styling controls."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.default"] = "rm"
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times",
        "font.size": 10,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts,bm}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

import numpy as np
import pandas as pd

from experiments import measure_definitions as MEASURE_DEFS
from experiments.utils.data_loaders import (
    load_mdkp_instance,
)

DISTRIBUTION_SPECS = MEASURE_DEFS.DISTRIBUTION_SPECS
build_histogram_frame = MEASURE_DEFS.build_histogram_frame
constraint_bar_specs = MEASURE_DEFS.constraint_bar_specs
constraint_center_slack = (
    MEASURE_DEFS.constraint_center_slack
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "brainstorm"
    / "figures"
    / "slack-uniform.pdf"
)
DEFAULT_SEED = 1
DEFAULT_SIZE = 15
DEFAULT_MEASURES = (
    "row_uniform",
    # "slack_uniform",
    # "feasible_q0",
    # "feasible_q1",
    # "feasible_q2",
    # "feasible_q3",
    # "feasible_q4",
)

# Plot configuration constants (edit these to change behavior)
PLOT_SIZE: int = DEFAULT_SIZE
PLOT_SEED: int = DEFAULT_SEED
# Empty list means "all" constraints will be plotted.
PLOT_CONSTRAINTS: list[int] = [2, 6]
# Empty list means use DEFAULT_MEASURES defined above.
PLOT_MEASURES: list[str] = []
# If True, overlay a kernel-smoothed curve on top of the bar
# histogram.
PLOT_SMOOTH: bool = True
# Gaussian kernel width as a multiple of the mean bin width.
PLOT_SMOOTH_SIGMA: float = 0.8
PLOT_OUTPUT_PATH: Path = DEFAULT_OUTPUT_PATH


def _resolve_constraints(
    requested: list[int],
    num_constraints: int,
) -> list[int]:
    """Return validated constraint indices in request order."""
    if not requested:
        return list(range(num_constraints))

    invalid = [
        idx
        for idx in requested
        if idx < 0 or idx >= num_constraints
    ]
    if invalid:
        raise ValueError(
            f"constraint indices out of range: {invalid}"
        )

    seen: set[int] = set()
    ordered: list[int] = []
    for idx in requested:
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def _resolve_measures(
    requested: list[str],
) -> list[tuple[str, str]]:
    """Return selected `(distribution, label)` pairs in requested order."""
    available = {
        distribution: label
        for distribution, label in DISTRIBUTION_SPECS
    }
    names = (
        DEFAULT_MEASURES
        if not requested
        else tuple(requested)
    )

    missing = [
        name for name in names if name not in available
    ]
    if missing:
        raise ValueError(f"unknown measures: {missing}")

    seen: set[str] = set()
    selected: list[tuple[str, str]] = []
    for name in names:
        if name in seen:
            continue
        selected.append((name, available[name]))
        seen.add(name)
    return selected


def _plot_selected_constraints(
    *,
    histogram_frame,
    problem,
    constraints: list[int],
    measure_specs: list[tuple[str, str]],
    output_path: Path,
    instance_name: str = "",
) -> None:
    """Plot the requested MDKP constraints with local styling.

    instance_name, if provided, is rendered in the figure title using a
    monospaced math font (mathtext \mathtt) so names like 'pet3' appear
    in monospace.
    """
    # Approximate Fraunhofer-ish teal/green palette (cohesive, teal-green-ish)
    HIST_COLOR = "#179C7D"

    # Color cycle and transparency for overlapping histograms
    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", [HIST_COLOR])
    )
    alpha_val = 0.75

    # Create a single axis (reduced height) and overlay all selected histograms on it
    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor("none")

    # Set tick label font size
    ax.tick_params(axis="both", labelsize=8)

    # Assign a pastel green to the first constraint and use the color cycle for others
    pastel_green = "#179C7D"
    constraint_colors: dict[int, str] = {}
    for i, constraint_idx in enumerate(constraints):
        if i == 0:
            constraint_colors[constraint_idx] = pastel_green
        else:
            constraint_colors[constraint_idx] = color_cycle[
                (i - 1) % len(color_cycle)
            ]

    # Pre-compute global maximum probability (across all constraints and measures)
    max_prob_global = 0.0
    for constraint_idx in constraints:
        subset = histogram_frame[
            histogram_frame["constraint_idx"]
            == constraint_idx
        ]
        bar_specs = constraint_bar_specs(subset)
        for distribution, _label in measure_specs:
            _, _, probabilities = bar_specs[distribution]
            if probabilities.size:
                max_prob_global = max(
                    max_prob_global,
                    float(np.asarray(probabilities).max()),
                )

    y_limit = max(0.0, 1.05 * max_prob_global)

    # Plot: loop constraints outermost so we can label each constraint once in the legend.
    for constraint_idx in constraints:
        subset = histogram_frame[
            histogram_frame["constraint_idx"]
            == constraint_idx
        ]
        bar_specs = constraint_bar_specs(subset)
        color = constraint_colors[constraint_idx]
        first_for_constraint = True
        for distribution, _label in measure_specs:
            left, widths, probabilities = bar_specs[
                distribution
            ]

            # Convert to numpy and defensively handle zero-width bins.
            widths = np.asarray(widths, dtype=float)
            probabilities = np.asarray(
                probabilities, dtype=float
            )
            widths_safe = np.where(
                widths <= 0.0, 1.0, widths
            )

            label_to_use = (
                f"{constraint_idx}"
                if first_for_constraint
                else None
            )

            ax.bar(
                left,
                probabilities,
                width=widths_safe,
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=0.4,
                alpha=alpha_val,
                label=label_to_use,
            )

            first_for_constraint = False

    # Optional: vertical line at slack center for reference (commented)
    # ax.axvline(0.0, color="0", linestyle="--", linewidth=0.9)

    ax.set_ylim(0.0, y_limit)
    # ax.set_title(
    #     "All selected constraints and measures",
    #     fontsize=10,
    # )
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)

    # Place legend in the top-right inside the axes and label by constraint
    ax.legend(
        title="Constraints",
        loc="upper right",
        fontsize=8,
        framealpha=0.6,
    )

    ax.set_xlabel(
        r"Slack: $a_{j} \cdot \bm{x} - b_{j}$", fontsize=10
    )

    # Build a suptitle; if an instance name is provided render it in
    # monospaced math font so it stands out.
    title = r"Slack Probabilities of Uniform Measure over $\mathbb{F}_{2}^{n}$"
    fig.suptitle(title, fontsize=12, y=0.975, x=0.525)
    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def main() -> None:
    """Run the plotting entry point using constants defined at the top of this file.

    Hardcoded to use data/mdkp_instances/pet3.dat from the repository root.
    """
    data_file = (
        REPO_ROOT / "data" / "mdkp_instances" / "pet3.dat"
    )
    if not data_file.exists():
        raise FileNotFoundError(
            f"expected MDKP data file not found: {data_file}"
        )

    inst = load_mdkp_instance(data_file)
    problem = inst.blp
    print(
        f"[plot_measures] using hardcoded instance: {inst.name} (n={inst.n}, m={inst.m})"
    )

    # Enumerate all states and build exact per-distribution histogram frame
    enum = MEASURE_DEFS.HypercubeSampleEnumerator.full(
        int(problem.n)
    )
    full_slacks = MEASURE_DEFS.all_slacks(enum, problem)
    frames = [
        MEASURE_DEFS.exact_histogram_frame(
            family="mdkp",
            size=int(problem.n),
            distribution=distribution,
            problem=problem,
            all_slacks=full_slacks,
        )
        for distribution, _label in DISTRIBUTION_SPECS
    ]
    histogram_frame = pd.concat(frames, ignore_index=True)

    measure_specs = _resolve_measures(PLOT_MEASURES)
    constraints = _resolve_constraints(
        PLOT_CONSTRAINTS, problem.m
    )
    _plot_selected_constraints(
        histogram_frame=histogram_frame,
        problem=problem,
        constraints=constraints,
        measure_specs=measure_specs,
        output_path=PLOT_OUTPUT_PATH,
        instance_name=inst.name,
    )


if __name__ == "__main__":
    main()
