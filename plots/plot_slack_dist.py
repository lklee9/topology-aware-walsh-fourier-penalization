"""Plot the q2 slack distribution for one inequality."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

if __package__ in (None, ""):
    import sys

    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.rcParams["mathtext.default"] = "rm"
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

from experiments.utils.problems import (
    sample_mdkp_problem,
    sample_mis_problem,
)
from fourier_projection.blp import BLP
from fourier_projection.measures import (
    GaussianSlackTarget,
    ProjectionMeasures,
)
from fourier_projection.projection import (
    HypercubeSampleEnumerator,
)
from fourier_projection.sampling import BinnedSlackTargetLaw

DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "results" / "slack_dist"
)
DEFAULT_PAPER_FIGURE_PDF = (
    REPO_ROOT / "brainstorm" / "figures" / "slack-dist.pdf"
)
PAPER_FIGURE_WIDTH = 4.2
PAPER_FIGURE_HEIGHT = 1.5
DEFAULT_SEED = 1
DEFAULT_CONSTRAINT = 2
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_FAMILY = "mdkp"
DEFAULT_SIZE_BY_FAMILY = {
    "mdkp": 15,
    "mis": 16,
}
FAMILY_CODES = {"mdkp": 3, "mis": 4}


def seed_from_components(
    base_seed: int,
    *components: int,
) -> int:
    """Create one deterministic child seed from integer components."""
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            *[int(value) for value in components],
        ]
    )
    return int(
        sequence.generate_state(1, dtype=np.uint64)[0]
    )


def build_problem(
    family: str,
    size: int,
    seed: int,
) -> BLP:
    """Sample one deterministic problem instance."""
    problem_seed = seed_from_components(
        seed,
        1_000,
        FAMILY_CODES[family],
        size,
    )

    if family == "mdkp":
        return sample_mdkp_problem(size, problem_seed)
    if family == "mis":
        return sample_mis_problem(size, problem_seed)

    raise ValueError(f"unsupported family: {family}")


def all_slacks(
    enum: HypercubeSampleEnumerator,
    problem: BLP,
) -> np.ndarray:
    """Return all inequality slacks on the enumerated states."""
    if problem.m == 0:
        return np.zeros((enum.N, 0), dtype=float)
    return enum.X @ problem.A.T - problem.b


def _slack_support_edges(
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bar intervals around sorted slack support points."""
    support = np.asarray(support, dtype=float)
    if support.ndim != 1:
        raise ValueError("support must be one-dimensional")
    if support.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    support = np.sort(support)
    if support.size == 1:
        center = float(support[0])
        return (
            np.array([center - 0.5], dtype=float),
            np.array([center + 0.5], dtype=float),
        )

    midpoints = 0.5 * (support[:-1] + support[1:])
    lower = np.empty_like(support, dtype=float)
    upper = np.empty_like(support, dtype=float)
    lower[0] = support[0] - 0.5 * (support[1] - support[0])
    lower[1:] = midpoints
    upper[:-1] = midpoints
    upper[-1] = support[-1] + 0.5 * (
        support[-1] - support[-2]
    )
    return lower, upper


def _group_slack_weights(
    slacks: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate normalized mass by exact slack value."""
    slacks = np.asarray(slacks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if slacks.ndim != 1:
        raise ValueError("slacks must be one-dimensional")
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if slacks.shape[0] != weights.shape[0]:
        raise ValueError(
            "slacks and weights must have the same length"
        )
    if slacks.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    order = np.argsort(slacks, kind="mergesort")
    sorted_slacks = slacks[order]
    sorted_weights = weights[order]
    support, start_idx = np.unique(
        sorted_slacks,
        return_index=True,
    )
    probabilities = np.add.reduceat(
        sorted_weights, start_idx
    )
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(
            "weights must have positive total mass"
        )
    return support, probabilities / total


def _support_frame(
    support: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Return a frame for one exact slack distribution."""
    lower, upper = _slack_support_edges(support)
    return pd.DataFrame(
        {
            "slack": support,
            "bin_lower": lower,
            "bin_upper": upper,
            "probability": probabilities,
        }
    )


def _q2_law(
    problem: BLP,
    constraint_idx: int,
) -> BinnedSlackTargetLaw:
    """Build the q2 binned law for one inequality."""
    coeffs = np.asarray(
        problem.A[constraint_idx], dtype=float
    )
    rhs = float(problem.b[constraint_idx])
    proposal_probs = np.full(problem.n, 0.5, dtype=float)
    target = ProjectionMeasures(None, problem).q2(
        constraint_idx
    )
    return BinnedSlackTargetLaw(
        a=coeffs,
        b=rhs,
        p=proposal_probs,
        target=target,
    )


def _perfect_target_support_frame(
    support: np.ndarray,
    law: BinnedSlackTargetLaw,
) -> pd.DataFrame:
    """Return the q2 target on exact slack values via its PDF."""
    target = law.target
    if not isinstance(target, GaussianSlackTarget):
        raise TypeError(
            "q2 support plotting expects a Gaussian target"
        )

    scale = target.resolved_scale(a=law.a, p=law.p)
    probabilities = norm.pdf(
        np.asarray(support, dtype=float),
        loc=float(target.center),
        scale=scale,
    )
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(
            "Gaussian PDF has zero mass on the slack support"
        )
    return _support_frame(support, probabilities / total)


def _support_lookup_indices(
    support: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Map slack values back to indices in the exact slack support."""
    support = np.asarray(support, dtype=float)
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(support, values)
    clipped = np.clip(idx, 0, support.size - 1)
    matches = np.isclose(
        support[clipped], values, rtol=0.0, atol=1e-12
    )
    if np.any(idx >= support.size) or not np.all(matches):
        raise ValueError(
            "encountered slack value outside the exact support"
        )
    return idx


def _plot_overlap_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    first_mass_key: str,
    first_label: str,
    first_color: str,
    second_mass_key: str,
    second_label: str,
    second_color: str,
    title: str,
    ylabel: str,
    first_zorder: float = 2.0,
    second_zorder: float = 1.0,
    first_alpha: float = 0.45,
    second_alpha: float = 0.35,
) -> None:
    """Draw two overlapping bar series on one slack axis."""
    frame = frame.sort_values(
        [left_key, right_key]
    ).reset_index(drop=True)
    left = frame[left_key].to_numpy(dtype=float)
    right = frame[right_key].to_numpy(dtype=float)
    widths = right - left
    first = frame[first_mass_key].to_numpy(dtype=float)
    second = frame[second_mass_key].to_numpy(dtype=float)

    second_bars = ax.bar(
        left,
        second,
        width=widths,
        align="edge",
        color=second_color,
        alpha=second_alpha,
        edgecolor=second_color,
        linewidth=0.35,
        label=second_label,
        zorder=second_zorder,
    )
    first_bars = ax.bar(
        left,
        first,
        width=widths,
        align="edge",
        color=first_color,
        alpha=first_alpha,
        edgecolor=first_color,
        linewidth=0.35,
        label=first_label,
        zorder=first_zorder,
    )
    ax.axvline(
        0.0, color="0.3", linestyle="--", linewidth=0.9
    )
    if title:
        ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.legend(
        [first_bars, second_bars],
        [first_label, second_label],
        loc="upper left",
        framealpha=0.6,
        edgecolor="none",
        handlelength=1.4,
    )


def _style_paper_axis(ax: plt.Axes) -> None:
    """Apply paper-oriented styling to one axis."""
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)


def plot_slack_distributions(
    *,
    family: str,
    size: int,
    seed: int,
    constraint_idx: int,
    sample_size: int,
    output_dir: Path,
    paper_figure_pdf: Path | None,
) -> dict[str, Path]:
    """Build and save the requested figure plus supporting CSVs."""
    problem = build_problem(family, size, seed)
    if problem.m == 0:
        raise ValueError(
            f"{problem.name} has no inequalities; slack plots are undefined"
        )
    if constraint_idx < 0 or constraint_idx >= problem.m:
        raise IndexError(
            f"constraint_idx must be in [0, {problem.m - 1}]"
        )

    enum = HypercubeSampleEnumerator.full(problem.n)
    slacks = all_slacks(enum, problem)[:, constraint_idx]
    support = np.unique(slacks)

    q2_law = _q2_law(problem, constraint_idx)

    q2_perfect_frame = _perfect_target_support_frame(
        support=support,
        law=q2_law,
    )

    proposal_support, proposal_probabilities = (
        _group_slack_weights(
            slacks=slacks,
            weights=np.ones_like(slacks, dtype=float),
        )
    )
    if not np.array_equal(proposal_support, support):
        raise RuntimeError(
            "proposal slack support does not match exact support"
        )

    sample_seed = seed_from_components(
        seed,
        2_000,
        FAMILY_CODES[family],
        size,
        constraint_idx,
        sample_size,
    )
    rng = np.random.default_rng(sample_seed)
    sample_bits = rng.binomial(
        1,
        0.5,
        size=(sample_size, problem.n),
    ).astype(float)
    sample_slacks = q2_law.slack_from_X(sample_bits)

    support_idx = _support_lookup_indices(
        support, sample_slacks
    )
    target_probabilities = q2_perfect_frame[
        "probability"
    ].to_numpy(dtype=float)
    sample_weights = (
        target_probabilities[support_idx]
        / proposal_probabilities[support_idx]
    )
    q2_estimated_support, q2_estimated_probabilities = (
        _group_slack_weights(
            slacks=sample_slacks,
            weights=sample_weights,
        )
    )
    q2_estimated_frame = _support_frame(
        q2_estimated_support,
        q2_estimated_probabilities,
    )

    stem = (
        f"{family}_{size}_constraint_{constraint_idx:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_png_path = output_dir / f"{stem}.png"
    figure_pdf_path = output_dir / f"{stem}.pdf"
    q2_perfect_csv_path = (
        output_dir / f"{stem}_q2_perfect.csv"
    )
    q2_estimated_csv_path = (
        output_dir / f"{stem}_q2_estimated.csv"
    )

    if paper_figure_pdf is not None:
        paper_figure_pdf.parent.mkdir(
            parents=True, exist_ok=True
        )

    q2_perfect_frame.to_csv(
        q2_perfect_csv_path, index=False
    )
    q2_estimated_frame.to_csv(
        q2_estimated_csv_path, index=False
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(PAPER_FIGURE_WIDTH, PAPER_FIGURE_HEIGHT),
    )
    fig.patch.set_facecolor("none")

    # Colors: teal for exact target (consistent with project), coral/red for estimate
    # These have better luminance contrast when overlapping with transparency
    perfect_color = "#1b998b"  # teal - matches projected_full in METHOD_COLORS
    estimated_color = (
        "#e84855"  # red/coral - matches projected_pegasus
    )

    plot_frame = q2_perfect_frame.merge(
        q2_estimated_frame,
        on=["slack", "bin_lower", "bin_upper"],
        how="outer",
        suffixes=("_perfect", "_estimated"),
    ).fillna(0.0)

    _plot_overlap_bars(
        ax,
        plot_frame,
        left_key="bin_lower",
        right_key="bin_upper",
        first_mass_key="probability_perfect",
        first_label=r"Exact $q_2$ target",
        first_color=perfect_color,
        second_mass_key="probability_estimated",
        second_label="IS estimate",
        second_color=estimated_color,
        title="",
        ylabel="Probability mass",
        first_zorder=2.0,
        second_zorder=3.0,
        first_alpha=0.25,
        second_alpha=0.30,
    )

    _style_paper_axis(ax)
    ax.set_xlabel(r"Slack $s = a_j^\top x - b_j$")
    # Move y-label to the top of the plot, horizontally centered
    yl = ax.set_ylabel("Probability mass")
    lab = ax.yaxis.get_label()
    lab.set_rotation(0)
    lab.set_ha("center")
    lab.set_va("bottom")
    # place above the axes (1.02) and centered (0.5)
    ax.yaxis.set_label_coords(0.5, 1.02)

    ax.set_xlim(
        float(plot_frame["bin_lower"].min()),
        float(plot_frame["bin_upper"].max()),
    )
    max_prob = float(
        plot_frame[
            ["probability_perfect", "probability_estimated"]
        ]
        .to_numpy(dtype=float)
        .max()
    )
    if max_prob > 0.0:
        ax.set_ylim(0.0, 1.05 * max_prob)

    fig.tight_layout(pad=0.3)
    fig.savefig(
        figure_pdf_path,
        bbox_inches="tight",
        transparent=True,
    )
    if paper_figure_pdf is not None:
        fig.savefig(
            paper_figure_pdf,
            bbox_inches="tight",
            transparent=True,
        )
    fig.savefig(
        figure_png_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    outputs = {
        "figure_pdf": figure_pdf_path,
        "figure_png": figure_png_path,
        "q2_perfect_csv": q2_perfect_csv_path,
        "q2_estimated_csv": q2_estimated_csv_path,
    }
    if paper_figure_pdf is not None:
        outputs["paper_figure_pdf"] = paper_figure_pdf
    return outputs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot one inequality's q2 slack distribution, comparing "
            "the PDF-based exact-support target against the sampled "
            "importance-weighted estimate."
        )
    )
    parser.add_argument(
        "--family",
        choices=sorted(DEFAULT_SIZE_BY_FAMILY),
        default=DEFAULT_FAMILY,
        help=f"Problem family. Default: {DEFAULT_FAMILY}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Problem size. Defaults to the standard experiment size "
            "for the chosen family."
        ),
    )
    parser.add_argument(
        "--constraint",
        type=int,
        default=DEFAULT_CONSTRAINT,
        help=(
            "Inequality index to plot. "
            f"Default: {DEFAULT_CONSTRAINT}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of Bernoulli(0.5) proposal samples for the "
            f"importance-weighted q2 estimate. Default: {DEFAULT_SAMPLE_SIZE}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--paper-figure-pdf",
        type=Path,
        default=DEFAULT_PAPER_FIGURE_PDF,
        help=(
            "Optional PDF path for the paper figure copy. "
            f"Default: {DEFAULT_PAPER_FIGURE_PDF}"
        ),
    )
    args = parser.parse_args()
    if args.size is None:
        args.size = DEFAULT_SIZE_BY_FAMILY[args.family]
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.constraint < 0:
        raise ValueError("--constraint must be nonnegative")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    return args


def main() -> None:
    """Run the CLI entry point."""
    args = _parse_args()
    outputs = plot_slack_distributions(
        family=args.family,
        size=args.size,
        seed=args.seed,
        constraint_idx=args.constraint,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        paper_figure_pdf=args.paper_figure_pdf,
    )

    for label, path in outputs.items():
        print(f"{label}: {path}", flush=True)


if __name__ == "__main__":
    main()
