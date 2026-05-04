"""Plot topology comparison heatmaps before and after embedding.

This script generates two side-by-side heatmap figures patterned after
``brainstorm/figures/topology_vs_full_cop_heatmap.pdf``:

1. topology vs full projection; and
2. topology vs unbalanced penalization.

Each figure has two panels:

- left panel: before embedding
- right panel: after embedding

Before embedding, each topology-restricted logical projection
(``projected_chimera``, ``projected_pegasus``, ``projected_zephyr``)
is compared against the chosen baseline method on matched instances from
``experiments/results/unbalanced_penalization/cop_instance_summary.csv``.

After embedding, the embedded topology-aware method
(``projected_topology``) is compared against the chosen embedded
baseline method separately for each hardware family on matched
instances from
``experiments/results/compare_embedding/sqa_chain_strength_fraction_1/cop_instance_summary.csv``.

For both figures, each cell shows

``(median(topology CoP) - median(comparator CoP)) / median(comparator CoP) * 100``

computed within one ``(family, size, hardware)`` block after strict
same-instance matching. Positive values favor the topology-constrained
method over the chosen comparator.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PAPER_FIGURES_DIR = REPO_ROOT / "brainstorm" / "figures"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "plots" / "compare_before_after_output"
)
DEFAULT_BASELINE_INPUT_CANDIDATES = (
    EXPERIMENTS_DIR
    / "results"
    / "unbalanced_penalization"
    / "cop_instance_summary.csv",
    REPO_ROOT
    / "results"
    / "unbalanced_penalization"
    / "cop_instance_summary.csv",
)
DEFAULT_EMBEDDING_INPUT_CANDIDATES = (
    EXPERIMENTS_DIR
    / "results"
    / "compare_embedding"
    / "sqa_chain_strength_fraction_1"
    / "cop_instance_summary.csv",
    REPO_ROOT
    / "results"
    / "compare_embedding"
    / "sqa_chain_strength_fraction_1"
    / "cop_instance_summary.csv",
)

if __package__ in (None, ""):
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm

from experiments.experiment_config import (
    FAMILY_LABELS,
    FAMILY_ORDER,
)
from plots.heatmap_layout import (
    HEATMAP_ANNOTATION_COLOR,
    HEATMAP_ANNOTATION_FONTSIZE,
    HEATMAP_FACE_COLOR,
    HEATMAP_PANEL_TITLE_FONTSIZE,
    HEATMAP_TICK_FONTSIZE,
    HEATMAP_TITLE_PAD,
    add_heatmap_grid,
    add_standard_heatmap_colorbar,
    cop_heatmap_cmap,
    heatmap_figsize_for_cells,
    heatmap_text_color_white_default,
    style_heatmap_xaxis,
)

HARDWARE_ORDER = (
    "chimera",
    "pegasus",
    "zephyr",
)
HARDWARE_LABELS = {
    "chimera": "Chimera",
    "pegasus": "Pegasus",
    "zephyr": "Zephyr",
}
BASELINE_TOPOLOGY_METHOD = {
    "chimera": "projected_chimera",
    "pegasus": "projected_pegasus",
    "zephyr": "projected_zephyr",
}
TITLE_PAD = HEATMAP_TITLE_PAD

# Shared constants for other plotting scripts in this directory
ANNOTATION_COLOR = HEATMAP_ANNOTATION_COLOR
PANEL_TITLE_FONTSIZE = HEATMAP_PANEL_TITLE_FONTSIZE
PLOT_FACE_COLOR = HEATMAP_FACE_COLOR
SUPTITLE_FONTSIZE = 11
TICK_FONTSIZE = HEATMAP_TICK_FONTSIZE


@dataclass(frozen=True)
class ComparatorSpec:
    """One baseline comparator configuration."""

    key: str
    label: str
    baseline_method: str
    embedding_method: str
    title: str
    figure_stem: str
    summary_name: str


COMPARATOR_SPECS = {
    "full": ComparatorSpec(
        key="full",
        label="full projection",
        baseline_method="projected_full",
        embedding_method="projected_full",
        title="Relative CoP change vs full projection",
        figure_stem="topology_vs_full_before_after_cop_heatmap",
        summary_name="topology_vs_full_before_after_summary.csv",
    ),
    "unbalanced": ComparatorSpec(
        key="unbalanced",
        label="unbalanced penalization",
        baseline_method="unbalanced",
        embedding_method="unbalanced",
        title="Relative CoP change vs unbalanced penalization",
        figure_stem="topology_vs_unbalanced_before_after_cop_heatmap",
        summary_name="topology_vs_unbalanced_before_after_summary.csv",
    ),
}
COMPARATOR_ORDER = (
    "full",
    "unbalanced",
)
EXCLUDED_FAMILIES_BY_COMPARATOR: dict[
    str, frozenset[str]
] = {}


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=None,
        help="Path to the baseline cop_instance_summary.csv.",
    )
    parser.add_argument(
        "--embedding-input",
        type=Path,
        default=None,
        help="Path to the embedding cop_instance_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and summary CSVs are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser


def resolve_input_path(
    path: Path | None,
    *,
    candidates: tuple[Path, ...],
    label: str,
) -> Path:
    """Resolve one input file path."""
    if path is not None:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"{label} input not found: {resolved}"
            )
        return resolved
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(
        str(candidate) for candidate in candidates
    )
    raise FileNotFoundError(
        f"could not find {label} input in: {tried}"
    )


def required_columns() -> set[str]:
    """Return the core columns needed by this script."""
    return {
        "family",
        "size",
        "instance_index",
        "method",
        "sqa_logical_cop",
    }


def load_instance_csv(path: Path) -> pd.DataFrame:
    """Load one instance-level CoP summary CSV."""
    frame = pd.read_csv(path)
    missing = required_columns() - set(frame.columns)
    if missing:
        raise ValueError(
            f"missing required columns {sorted(missing)} in {path}"
        )
    if frame.empty:
        raise ValueError(f"input CSV is empty: {path}")
    return frame.copy()


def shared_instance_keys(
    lhs_df: pd.DataFrame,
    rhs_df: pd.DataFrame,
) -> list[str]:
    """Return stable merge keys for strict same-instance matching."""
    keys = ["family", "size", "instance_index"]
    if (
        "problem_seed" in lhs_df.columns
        and "problem_seed" in rhs_df.columns
        and lhs_df["problem_seed"].notna().all()
        and rhs_df["problem_seed"].notna().all()
    ):
        keys.append("problem_seed")
    return keys


def family_size_sort_key(
    item: tuple[str, int],
) -> tuple[int, int, str]:
    """Return one stable sort key for heatmap rows."""
    family, size = item
    try:
        family_rank = FAMILY_ORDER.index(family)
    except ValueError:
        family_rank = len(FAMILY_ORDER)
    return (family_rank, int(size), family)


def family_size_label(
    family: str,
    size: int,
) -> str:
    """Return one compact row label."""
    family_label = FAMILY_LABELS.get(family, family)
    return f"{family_label} {int(size)}"


def relative_change_percent(
    topology_median: float,
    comparator_median: float,
) -> float:
    """Return relative percent change of topology vs comparator."""
    if not math.isfinite(topology_median):
        return math.nan
    if not math.isfinite(comparator_median):
        return math.nan
    if topology_median == 0.0 and comparator_median == 0.0:
        return math.nan
    if comparator_median == 0.0:
        if topology_median > 0.0:
            return math.inf
        return math.nan
    return (
        (float(topology_median) - float(comparator_median))
        / float(comparator_median)
        * 100.0
    )


def summarize_stage_pairs(
    topology_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    *,
    stage: str,
    hardware_family: str,
) -> pd.DataFrame:
    """Return one per-block summary for matched topology/comparator rows."""
    merge_keys = shared_instance_keys(
        topology_df, comparator_df
    )
    topology_cols = merge_keys + ["sqa_logical_cop"]
    comparator_cols = merge_keys + ["sqa_logical_cop"]
    merged = (
        topology_df[topology_cols]
        .rename(columns={"sqa_logical_cop": "topology_cop"})
        .merge(
            comparator_df[comparator_cols].rename(
                columns={
                    "sqa_logical_cop": "comparator_cop"
                }
            ),
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "hardware_family",
                "family",
                "size",
                "n_pairs",
                "topology_cop_median",
                "comparator_cop_median",
                "cop_is_zero_zero",
                "relative_change_pct",
            ]
        )

    grouped = merged.groupby(
        ["family", "size"],
        as_index=False,
    ).agg(
        n_pairs=("instance_index", "count"),
        topology_cop_median=("topology_cop", "median"),
        comparator_cop_median=("comparator_cop", "median"),
    )
    grouped["stage"] = stage
    grouped["hardware_family"] = hardware_family
    grouped["cop_is_zero_zero"] = (
        grouped["topology_cop_median"] == 0.0
    ) & (grouped["comparator_cop_median"] == 0.0)
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: relative_change_percent(
            float(row["topology_cop_median"]),
            float(row["comparator_cop_median"]),
        ),
        axis=1,
    )
    return grouped[
        [
            "stage",
            "hardware_family",
            "family",
            "size",
            "n_pairs",
            "topology_cop_median",
            "comparator_cop_median",
            "cop_is_zero_zero",
            "relative_change_pct",
        ]
    ].copy()


def build_before_summary(
    baseline_df: pd.DataFrame,
    spec: ComparatorSpec,
) -> pd.DataFrame:
    """Return the pre-embedding topology summary."""
    comparator_df = baseline_df[
        baseline_df["method"] == spec.baseline_method
    ].copy()
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        topology_method = BASELINE_TOPOLOGY_METHOD[
            hardware_family
        ]
        topology_df = baseline_df[
            baseline_df["method"] == topology_method
        ].copy()
        frames.append(
            summarize_stage_pairs(
                topology_df,
                comparator_df,
                stage="before",
                hardware_family=hardware_family,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_after_summary(
    embedding_df: pd.DataFrame,
    spec: ComparatorSpec,
) -> pd.DataFrame:
    """Return the post-embedding topology summary."""
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        hardware_df = embedding_df[
            embedding_df["hardware_family"]
            == hardware_family
        ].copy()
        topology_df = hardware_df[
            hardware_df["method"] == "projected_topology"
        ].copy()
        comparator_df = hardware_df[
            hardware_df["method"] == spec.embedding_method
        ].copy()
        frames.append(
            summarize_stage_pairs(
                topology_df,
                comparator_df,
                stage="after",
                hardware_family=hardware_family,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ordered_blocks(
    summary_df: pd.DataFrame,
) -> list[tuple[str, int]]:
    """Return one stable family/size row order."""
    blocks = {
        (str(row.family), int(row.size))
        for row in summary_df.itertuples()
    }
    return sorted(blocks, key=family_size_sort_key)


def exclude_families(
    summary_df: pd.DataFrame,
    spec: ComparatorSpec,
) -> pd.DataFrame:
    """Drop comparator-specific families from one summary table."""
    excluded = EXCLUDED_FAMILIES_BY_COMPARATOR.get(
        spec.key, frozenset()
    )
    if not excluded or summary_df.empty:
        return summary_df.reset_index(drop=True)
    filtered = summary_df[
        ~summary_df["family"].isin(excluded)
    ].copy()
    return filtered.reset_index(drop=True)


def format_relative_change(rel: float) -> str:
    """Format one relative change as a compact percent label."""
    if np.isposinf(rel):
        return "∞"
    if not np.isfinite(rel):
        return ""
    pct = float(rel) * 100.0
    if pct == 0.0:
        return "0%"
    abs_pct = abs(pct)
    if abs_pct >= 100.0:
        return f"{pct:+.0f}%"
    if abs_pct >= 10.0:
        return f"{pct:+.1f}%"
    if abs_pct >= 1.0:
        return f"{pct:+.2f}%"
    if abs_pct >= 0.01:
        return f"{pct:+.3f}%"
    return f"{pct:+.1e}%"


def build_stage_matrix(
    summary_df: pd.DataFrame,
    blocks: list[tuple[str, int]],
) -> tuple[np.ndarray, list[list[str]]]:
    """Return one stage matrix and the matching cell annotations."""
    lookup = {
        (
            str(row["family"]),
            int(row["size"]),
            str(row["hardware_family"]),
        ): row
        for row in summary_df.to_dict("records")
    }
    matrix = np.full(
        (len(blocks), len(HARDWARE_ORDER)),
        np.nan,
        dtype=float,
    )
    annotations = [
        ["" for _ in HARDWARE_ORDER]
        for _ in range(len(blocks))
    ]
    for row_index, (family, size) in enumerate(blocks):
        for column_index, hardware_family in enumerate(
            HARDWARE_ORDER
        ):
            row = lookup.get(
                (family, size, hardware_family)
            )
            if row is None:
                continue
            if bool(row.get("cop_is_zero_zero", False)):
                annotations[row_index][column_index] = "0/0"
                continue
            value = float(
                row.get("relative_change_pct", math.nan)
            )
            matrix[row_index, column_index] = value
            if np.isposinf(value):
                annotations[row_index][column_index] = "∞"
            else:
                annotations[row_index][column_index] = (
                    format_relative_change(value / 100.0)
                )
    return matrix, annotations


def delta_log_norm(values: np.ndarray) -> SymLogNorm:
    """Return one symmetric-log colour norm."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return SymLogNorm(
            linthresh=1.0,
            vmin=-1.0,
            vmax=1.0,
        )
    vmax = float(np.max(np.abs(finite)))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    linthresh = max(1e-3, vmax * 0.01)
    return SymLogNorm(
        linthresh=linthresh,
        vmin=-vmax,
        vmax=vmax,
    )


def shared_color_scale(
    summary_dfs: list[pd.DataFrame],
) -> tuple[SymLogNorm, float]:
    """Return one color norm shared across all rendered figures."""
    finite_parts: list[np.ndarray] = []
    for summary_df in summary_dfs:
        values = summary_df["relative_change_pct"].to_numpy(
            dtype=float
        )
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            finite_parts.append(finite)
    if finite_parts:
        finite_values = np.concatenate(finite_parts)
    else:
        finite_values = np.array([0.0], dtype=float)
    vmax_for_inf = float(np.max(np.abs(finite_values)))
    if (
        not math.isfinite(vmax_for_inf)
        or vmax_for_inf <= 0.0
    ):
        vmax_for_inf = 1.0
    return delta_log_norm(finite_values), vmax_for_inf


def paper_pdf_path(output_path: Path) -> Path:
    """Return the paper-ready PDF path for one PNG output."""
    return PAPER_FIGURES_DIR / f"{output_path.stem}.pdf"


def save_figure(
    figure: plt.Figure,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Save one figure as PNG plus matching PDF copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path, dpi=dpi, bbox_inches="tight"
    )
    pdf_path = paper_pdf_path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")


def stage_title(stage: str) -> str:
    """Return the display title for one stage."""
    if stage == "before":
        return "Before Embed"
    return "After Embed"


def plot_before_after_heatmap(
    combined_summary_df: pd.DataFrame,
    spec: ComparatorSpec,
    *,
    output_path: Path,
    dpi: int,
    norm: SymLogNorm,
    vmax_for_inf: float,
    title_prefix: str = "",
    colorbar_label: str = "relative change (%)",
    cell_width_scale: float = 1.0,
) -> None:
    """Plot the two-panel before/after topology heatmap."""
    before_df = combined_summary_df[
        combined_summary_df["stage"] == "before"
    ].copy()
    after_df = combined_summary_df[
        combined_summary_df["stage"] == "after"
    ].copy()
    blocks = ordered_blocks(combined_summary_df)
    row_labels = [
        family_size_label(family, size)
        for family, size in blocks
    ]
    before_matrix, before_annotations = build_stage_matrix(
        before_df,
        blocks,
    )
    after_matrix, after_annotations = build_stage_matrix(
        after_df,
        blocks,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=heatmap_figsize_for_cells(
            len(blocks),
            len(HARDWARE_ORDER),
            num_panels=2,
            cell_width_scale=cell_width_scale,
        ),
        constrained_layout=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes)
    cmap = cop_heatmap_cmap()
    image = None

    for axis, stage, matrix, annotations in zip(
        axes_array,
        ("before", "after"),
        (before_matrix, after_matrix),
        (before_annotations, after_annotations),
    ):
        matrix_for_color = matrix.copy()
        matrix_for_color[np.isposinf(matrix_for_color)] = (
            vmax_for_inf
        )
        matrix_for_color[np.isneginf(matrix_for_color)] = (
            -vmax_for_inf
        )
        masked = np.ma.masked_invalid(matrix_for_color)
        image = axis.imshow(
            masked,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        axis.set_facecolor(HEATMAP_FACE_COLOR)
        axis.set_title(
            f"{title_prefix}{stage_title(stage)}",
            fontsize=HEATMAP_PANEL_TITLE_FONTSIZE,
            pad=TITLE_PAD,
        )
        style_heatmap_xaxis(
            axis,
            [
                HARDWARE_LABELS[item]
                for item in HARDWARE_ORDER
            ],
            fontsize=HEATMAP_TICK_FONTSIZE,
        )
        if axis is axes_array[0]:
            axis.set_yticks(np.arange(len(row_labels)))
            axis.set_yticklabels(
                row_labels,
                fontsize=HEATMAP_TICK_FONTSIZE,
            )
            axis.tick_params(
                axis="y", labelsize=HEATMAP_TICK_FONTSIZE
            )
        else:
            axis.set_yticks(np.arange(len(row_labels)))
            axis.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )
        add_heatmap_grid(axis, matrix.shape)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                text = annotations[row_index][col_index]
                value = matrix[row_index, col_index]
                if not text:
                    continue
                if not np.isfinite(value):
                    if np.isinf(value):
                        text_color = heatmap_text_color_white_default(
                            norm,
                            matrix_for_color[
                                row_index, col_index
                            ],
                        )
                    else:
                        text_color = (
                            HEATMAP_ANNOTATION_COLOR
                        )
                    axis.text(
                        col_index,
                        row_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                        color=text_color,
                    )
                    continue
                text_color = (
                    heatmap_text_color_white_default(
                        norm,
                        matrix_for_color[
                            row_index, col_index
                        ],
                    )
                )
                axis.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                    color=text_color,
                )

    # figure.suptitle(
    #     spec.title,
    #     fontsize=SUPTITLE_FONTSIZE,
    # )
    if image is not None:
        add_standard_heatmap_colorbar(
            figure,
            image,
            ax=list(axes_array),
            label=colorbar_label,
            tick_fontsize=HEATMAP_TICK_FONTSIZE,
        )

    save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def build_summary_df(
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    spec: ComparatorSpec,
) -> pd.DataFrame:
    """Return one sorted summary table for a comparator."""
    before_summary_df = build_before_summary(
        baseline_df, spec
    )
    after_summary_df = build_after_summary(
        embedding_df, spec
    )
    summary_df = pd.concat(
        [before_summary_df, after_summary_df],
        ignore_index=True,
    )
    stage_rank = {"before": 0, "after": 1}
    summary_df["stage_rank"] = summary_df["stage"].map(
        stage_rank
    )
    summary_df = summary_df.sort_values(
        by=[
            "stage_rank",
            "hardware_family",
            "family",
            "size",
        ]
    ).drop(columns=["stage_rank"])
    summary_df = summary_df.reset_index(drop=True)
    summary_df["compare_against"] = spec.key
    summary_df["compare_against_label"] = spec.label
    return exclude_families(summary_df, spec)


def render_comparison(
    summary_df: pd.DataFrame,
    spec: ComparatorSpec,
    *,
    output_dir: Path,
    dpi: int,
    norm: SymLogNorm,
    vmax_for_inf: float,
) -> tuple[Path, Path]:
    """Render one comparator summary CSV and figure."""
    summary_path = output_dir / spec.summary_name
    summary_df.to_csv(summary_path, index=False)

    output_path = output_dir / f"{spec.figure_stem}.png"
    plot_before_after_heatmap(
        summary_df,
        spec,
        output_path=output_path,
        dpi=dpi,
        norm=norm,
        vmax_for_inf=vmax_for_inf,
    )
    return summary_path, output_path


def main() -> None:
    """Run the plotting workflow."""
    parser = build_argument_parser()
    args = parser.parse_args()

    baseline_input = resolve_input_path(
        args.baseline_input,
        candidates=DEFAULT_BASELINE_INPUT_CANDIDATES,
        label="baseline",
    )
    embedding_input = resolve_input_path(
        args.embedding_input,
        candidates=DEFAULT_EMBEDDING_INPUT_CANDIDATES,
        label="embedding",
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_instance_csv(baseline_input)
    embedding_df = load_instance_csv(embedding_input)

    summary_by_key = {
        comparator_key: build_summary_df(
            baseline_df,
            embedding_df,
            COMPARATOR_SPECS[comparator_key],
        )
        for comparator_key in COMPARATOR_ORDER
    }
    shared_norm, shared_vmax_for_inf = shared_color_scale(
        list(summary_by_key.values())
    )

    print(f"baseline input: {baseline_input}")
    print(f"embedding input: {embedding_input}")
    print(f"output dir: {output_dir}")
    print(
        "shared color scale max |relative change| (%): "
        f"{shared_vmax_for_inf:.3g}"
    )
    for comparator_key in COMPARATOR_ORDER:
        spec = COMPARATOR_SPECS[comparator_key]
        summary_path, output_path = render_comparison(
            summary_by_key[comparator_key],
            spec,
            output_dir=output_dir,
            dpi=int(args.dpi),
            norm=shared_norm,
            vmax_for_inf=shared_vmax_for_inf,
        )
        print(f"wrote {summary_path.name}")
        print(f"wrote {output_path.name}")
        print(f"wrote {paper_pdf_path(output_path)}")


if __name__ == "__main__":
    main()
