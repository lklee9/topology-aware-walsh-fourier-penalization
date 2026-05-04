"""Compare exact per-constraint slack histograms across distributions."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
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

from experiments.utils.problems import (
    sample_mdkp_problem,
    sample_mis_problem,
)
from fourier_projection.blp import BLP
from fourier_projection.measures import ProjectionMeasures
from fourier_projection.projection import (
    HypercubeSampleEnumerator,
)
from fourier_projection.sampling import BinnedSlackTargetLaw

EXACT_RECORD_TYPE = "exact"
HISTOGRAM_COLUMNS = [
    "family",
    "size",
    "distribution",
    "constraint_idx",
    "record_type",
    "slack",
    "bin_center",
    "bin_lower",
    "bin_upper",
    "probability",
]

PROBLEM_SPECS = (
    ("mdkp", 15),
    ("mis", 16),
)
FAMILY_CODES = {"mdkp": 3, "mis": 4}

DISTRIBUTION_SPECS = (
    ("row_uniform", "Row-uniform"),
    ("slack_uniform", "Slack-uniform"),
    ("feasible_q0", "Feasible q0"),
    ("feasible_q1", "Feasible q1"),
    ("feasible_q2", "Feasible q2"),
    ("feasible_q3", "Feasible q3"),
    ("feasible_q4", "Feasible q4"),
)
GAUSSIAN_CENTER_FRACTIONS = {
    "feasible_q0": 0.0,
    "feasible_q1": 0.25,
    "feasible_q2": 0.5,
    "feasible_q3": 0.75,
    "feasible_q4": 1.0,
}


@dataclass(frozen=True)
class ProblemSpec:
    """One requested family/size configuration."""

    family: str
    size: int


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


def build_problem(spec: ProblemSpec, seed: int) -> BLP:
    """Sample one deterministic problem instance."""
    problem_seed = seed_from_components(
        seed,
        1_000,
        FAMILY_CODES[spec.family],
        spec.size,
    )

    if spec.family == "mdkp":
        return sample_mdkp_problem(spec.size, problem_seed)
    if spec.family == "mis":
        return sample_mis_problem(spec.size, problem_seed)
    raise ValueError(f"unsupported family: {spec.family}")


def all_slacks(
    enum: HypercubeSampleEnumerator,
    problem: BLP,
) -> np.ndarray:
    """Return all inequality slack values on the enumerated states."""
    if problem.m == 0:
        return np.zeros((enum.N, 0), dtype=float)
    return enum.X @ problem.A.T - problem.b


def constraint_center_slack(
    problem: BLP,
    constraint_idx: int,
) -> float:
    """Return the slack at x_i = 1/2 for one inequality."""
    coeffs = np.asarray(
        problem.A[constraint_idx], dtype=float
    )
    rhs = float(problem.b[constraint_idx])
    return float(0.5 * coeffs.sum() - rhs)


def _gaussian_target_law(
    problem: BLP,
    constraint_idx: int,
    center_fraction: float,
) -> BinnedSlackTargetLaw:
    """Build the direct binned target law for one inequality."""
    coeffs = np.asarray(
        problem.A[constraint_idx], dtype=float
    )
    rhs = float(problem.b[constraint_idx])
    measures = ProjectionMeasures(None, problem)
    target_builders = {
        0.0: measures.q0,
        0.25: measures.q1,
        0.5: measures.q2,
        0.75: measures.q3,
        1.0: measures.q4,
    }
    try:
        target = target_builders[float(center_fraction)](
            constraint_idx
        )
    except KeyError as exc:
        raise ValueError(
            "center_fraction must be one of 0.0, 0.25, 0.5, 0.75, 1.0"
        ) from exc
    return BinnedSlackTargetLaw(
        a=coeffs,
        b=rhs,
        p=np.full(problem.n, 0.5, dtype=float),
        target=target,
    )


def _uniform_target_law(
    problem: BLP,
    constraint_idx: int,
) -> BinnedSlackTargetLaw:
    """Build the direct binned uniform target law for one inequality."""
    coeffs = np.asarray(
        problem.A[constraint_idx], dtype=float
    )
    rhs = float(problem.b[constraint_idx])
    p = np.full(problem.n, 0.5, dtype=float)
    target = ProjectionMeasures(
        None, problem
    ).uniform_slack(constraint_idx)
    return BinnedSlackTargetLaw(
        a=coeffs,
        b=rhs,
        p=p,
        target=target,
    )


def _slack_support_edges(
    slacks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one bar interval around each sorted slack support point."""
    support = np.asarray(slacks, dtype=float)
    if support.ndim != 1:
        raise ValueError("slacks must be one-dimensional")
    if support.size == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    support = np.sort(support)
    if support.size == 1:
        width = 1.0
        center = float(support[0])
        return (
            np.array([center - 0.5 * width], dtype=float),
            np.array([center + 0.5 * width], dtype=float),
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
    """Sum weights over all rows that share the same slack value."""
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
    unique_slacks, start_idx = np.unique(
        sorted_slacks,
        return_index=True,
    )
    summed_weights = np.add.reduceat(
        sorted_weights, start_idx
    )
    total_weight = float(summed_weights.sum())
    if total_weight <= 0.0:
        raise ValueError(
            "importance weights must have positive total mass"
        )
    return unique_slacks, summed_weights / total_weight


def _weighted_slack_histogram_rows(
    *,
    family: str,
    size: int,
    distribution: str,
    constraint_idx: int,
    slacks: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Return one exact target-mass row per distinct slack value."""
    support, probabilities = _group_slack_weights(
        slacks=slacks,
        weights=weights,
    )
    lower, upper = _slack_support_edges(support)

    rows: list[dict[str, float | int | str]] = []
    for slack, left, right, probability in zip(
        support,
        lower,
        upper,
        probabilities,
    ):
        rows.append(
            {
                "family": family,
                "size": int(size),
                "distribution": distribution,
                "constraint_idx": int(constraint_idx),
                "record_type": EXACT_RECORD_TYPE,
                "slack": float(slack),
                "bin_center": float(slack),
                "bin_lower": float(left),
                "bin_upper": float(right),
                "probability": float(probability),
            }
        )
    return rows


def _constraint_weights(
    *,
    distribution: str,
    problem: BLP,
    constraint_idx: int,
    slacks: np.ndarray,
) -> np.ndarray:
    """Return exact state weights for one distribution/constraint pair."""
    if distribution == "row_uniform":
        return np.ones_like(slacks, dtype=float)
    if distribution == "slack_uniform":
        law = _uniform_target_law(
            problem,
            constraint_idx=constraint_idx,
        )
        return law.importance_weights(S=slacks)
    if distribution in GAUSSIAN_CENTER_FRACTIONS:
        law = _gaussian_target_law(
            problem,
            constraint_idx=constraint_idx,
            center_fraction=GAUSSIAN_CENTER_FRACTIONS[
                distribution
            ],
        )
        return law.importance_weights(S=slacks)
    raise ValueError(
        f"unsupported distribution: {distribution}"
    )


def exact_histogram_frame(
    *,
    family: str,
    size: int,
    distribution: str,
    problem: BLP,
    all_slacks: np.ndarray,
) -> pd.DataFrame:
    """Return exact slack masses induced by one full-hypercube law."""
    rows: list[dict[str, float | int | str]] = []
    for constraint_idx in range(problem.m):
        constraint_slacks = all_slacks[:, constraint_idx]
        weights = _constraint_weights(
            distribution=distribution,
            problem=problem,
            constraint_idx=constraint_idx,
            slacks=constraint_slacks,
        )
        rows.extend(
            _weighted_slack_histogram_rows(
                family=family,
                size=size,
                distribution=distribution,
                constraint_idx=constraint_idx,
                slacks=constraint_slacks,
                weights=weights,
            )
        )
    return pd.DataFrame(rows, columns=HISTOGRAM_COLUMNS)


def build_histogram_frame(
    *,
    family: str,
    size: int,
    seed: int,
) -> tuple[BLP, pd.DataFrame]:
    """Build one exact histogram frame for a deterministic instance."""
    spec = ProblemSpec(family=family, size=size)
    problem = build_problem(spec, seed)
    if problem.m == 0:
        raise ValueError(
            f"{problem.name} has no inequalities; slack histograms are undefined"
        )

    full_enum = HypercubeSampleEnumerator.full(problem.n)
    full_slacks = all_slacks(full_enum, problem)
    frames = [
        exact_histogram_frame(
            family=family,
            size=size,
            distribution=distribution,
            problem=problem,
            all_slacks=full_slacks,
        )
        for distribution, _label in DISTRIBUTION_SPECS
    ]
    return problem, pd.concat(frames, ignore_index=True)


def constraint_bar_specs(
    constraint_frame: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return `(left, width, probability)` bars per distribution."""
    bar_specs: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for distribution, _label in DISTRIBUTION_SPECS:
        ordered = constraint_frame[
            constraint_frame["distribution"] == distribution
        ].sort_values(
            ["bin_lower", "bin_upper", "bin_center"],
            kind="mergesort",
        )
        left = ordered["bin_lower"].to_numpy(dtype=float)
        upper = ordered["bin_upper"].to_numpy(dtype=float)
        probabilities = ordered["probability"].to_numpy(
            dtype=float
        )
        bar_specs[distribution] = (
            left,
            upper - left,
            probabilities,
        )

    return bar_specs


DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "results" / "measure_definitions"
)
DEFAULT_SEED = 1
DEFAULT_PAGE_CONSTRAINTS = 12


def _log(message: str) -> None:
    """Print one stable progress line."""
    print(
        f"[measure_definitions] {message}",
        flush=True,
    )


def _plot_constraint_page(
    *,
    family: str,
    size: int,
    problem: BLP,
    histogram_frame: pd.DataFrame,
    path: Path,
    constraints: list[int],
) -> None:
    """Plot one page of histogram comparisons."""
    cols = len(constraints)
    rows = len(DISTRIBUTION_SPECS)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.0 * cols, 2.8 * rows),
        squeeze=False,
        sharey="row",
    )

    # Prepare colors and labels
    colors = {
        "row_uniform": "#1f77b4",
        "slack_uniform": "#d62728",
        "feasible_q0": "#17becf",
        "feasible_q1": "#2ca02c",
        "feasible_q2": "#ff7f0e",
        "feasible_q3": "#9467bd",
        "feasible_q4": "#8c564b",
    }
    row_max_probs = {
        distribution: 0.0
        for distribution, _label in DISTRIBUTION_SPECS
    }

    for constraint_idx in constraints:
        subset = histogram_frame[
            histogram_frame["constraint_idx"]
            == constraint_idx
        ]
        bar_specs = constraint_bar_specs(subset)
        for distribution, _label in DISTRIBUTION_SPECS:
            probabilities = bar_specs[distribution][2]
            if probabilities.size:
                row_max_probs[distribution] = max(
                    row_max_probs[distribution],
                    float(probabilities.max()),
                )

    for col_idx, constraint_idx in enumerate(constraints):
        subset = histogram_frame[
            histogram_frame["constraint_idx"]
            == constraint_idx
        ]
        bar_specs = constraint_bar_specs(subset)
        center_slack = constraint_center_slack(
            problem, constraint_idx
        )
        for row_idx, (distribution, label) in enumerate(
            DISTRIBUTION_SPECS
        ):
            ax = axes[row_idx][col_idx]
            left, widths, probabilities = bar_specs[
                distribution
            ]
            y_limit = max(
                0.0, 1.1 * row_max_probs[distribution]
            )
            ax.bar(
                left,
                probabilities,
                width=widths,
                align="edge",
                color=colors.get(distribution, "#333333"),
                alpha=0.85,
            )
            ax.axvline(
                0.0,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
            )
            ax.set_ylim(0.0, y_limit)
            ax.set_xlabel("Slack")
            ax.set_ylabel("Bin probability")
            ax.set_title(
                f"constraint {constraint_idx}, {label.lower()}"
            )

    fig.suptitle(
        f"Slack histograms by inequality\nfamily: {family}, size: {size}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(
    *,
    family: str,
    size: int,
    problem: BLP,
    histogram_frame: pd.DataFrame,
    output_dir: Path,
    page_constraints: int,
) -> list[Path]:
    """Save paginated histogram figures for one problem instance."""
    num_constraints = (
        int(histogram_frame["constraint_idx"].max()) + 1
    )
    paths: list[Path] = []
    for page_index, start in enumerate(
        range(0, num_constraints, page_constraints),
        start=1,
    ):
        constraints = list(
            range(
                start,
                min(
                    start + page_constraints,
                    num_constraints,
                ),
            )
        )
        path = (
            output_dir
            / f"{family}_{size}_hist_page_{page_index:02d}.png"
        )
        _plot_constraint_page(
            family=family,
            size=size,
            problem=problem,
            histogram_frame=histogram_frame,
            path=path,
            constraints=constraints,
        )
        paths.append(path)
    return paths


def _summary_row(
    *,
    problem: BLP,
    enum: HypercubeSampleEnumerator,
    slack_vectors: np.ndarray,
) -> dict[str, float | int | str]:
    """Return one compact summary row for one instance."""
    distinct_slack_vectors = (
        int(np.unique(slack_vectors, axis=0).shape[0])
        if problem.m
        else 0
    )
    return {
        "problem_name": problem.name,
        "num_variables": int(problem.n),
        "num_equalities": int(problem.p),
        "num_inequalities": int(problem.m),
        "num_states": int(enum.N),
        "distinct_slack_vectors": distinct_slack_vectors,
    }


def run_experiment(
    *,
    output_dir: Path,
    seed: int,
    page_constraints: int,
) -> None:
    """Run the requested histogram comparison workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    histogram_frames: list[pd.DataFrame] = []

    for family, size in PROBLEM_SPECS:
        spec = ProblemSpec(family=family, size=size)
        _log(f"building {family}({size})")
        problem = build_problem(spec, seed)
        if problem.m == 0:
            raise ValueError(
                f"{problem.name} has no inequalities; slack histograms are undefined"
            )
        full_enum = HypercubeSampleEnumerator.full(
            problem.n
        )
        full_slacks = all_slacks(full_enum, problem)

        problem_dir = output_dir / f"{family}_{size}"
        problem_dir.mkdir(parents=True, exist_ok=True)

        problem_histograms: list[pd.DataFrame] = []
        for distribution, _label in DISTRIBUTION_SPECS:
            _log(
                f"building exact histogram for {family}({size}) from {distribution}"
            )
            frame = exact_histogram_frame(
                family=family,
                size=size,
                distribution=distribution,
                problem=problem,
                all_slacks=full_slacks,
            )

            frame.to_csv(
                problem_dir
                / f"{distribution}_histograms.csv",
                index=False,
            )
            histogram_frames.append(frame)
            problem_histograms.append(frame)

        histogram_frame = pd.concat(
            problem_histograms,
            ignore_index=True,
        )
        histogram_frame.to_csv(
            problem_dir / "histograms.csv",
            index=False,
        )
        _plot_histograms(
            family=family,
            size=size,
            problem=problem,
            histogram_frame=histogram_frame,
            output_dir=problem_dir,
            page_constraints=page_constraints,
        )

        summary_slacks = full_slacks
        summary = _summary_row(
            problem=problem,
            enum=full_enum,
            slack_vectors=summary_slacks,
        )
        summary_rows.append(
            {
                "family": family,
                "size": int(size),
                **summary,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    pd.concat(histogram_frames, ignore_index=True).to_csv(
        output_dir / "all_histograms.csv",
        index=False,
    )
    _log(f"wrote results to {output_dir}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare row-uniform, slack-uniform, and feasible Gaussian "
            "laws via exact per-constraint slack histograms."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSVs and plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--page-constraints",
        type=int,
        default=DEFAULT_PAGE_CONSTRAINTS,
        help=(
            "Maximum number of constraints per figure page. "
            f"Default: {DEFAULT_PAGE_CONSTRAINTS}"
        ),
    )
    args = parser.parse_args()
    if args.page_constraints <= 0:
        raise ValueError(
            "--page-constraints must be positive"
        )
    return args


def main() -> None:
    """Run the CLI entry point."""
    args = _parse_args()
    run_experiment(
        output_dir=args.output_dir,
        seed=args.seed,
        page_constraints=args.page_constraints,
    )


if __name__ == "__main__":
    main()
