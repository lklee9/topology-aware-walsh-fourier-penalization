"""Plot before/after heatmaps for tuned unbalanced projections.

This script mirrors the topology-vs-unbalanced before/after heatmap
layout from ``plots/compare_before_after.py`` but uses the tuned
unbalanced projection results from
``experiments/results/compare_unb_pen_up_projection``.

It renders four two-panel heatmaps:

- median CoP relative change
- mean CoP relative change
- feasibility relative improvement
- objective-gap relative improvement

The CoP heatmaps reuse
``compare_before_after.plot_before_after_heatmap()`` so layout,
annotation formatting, row ordering, and colorbar construction stay in
sync with ``plots/compare_before_after.py``.

For feasibility and objective gap, each cell shows a paired block-level
relative improvement after strict same-instance matching, normalized by
the unbalanced baseline:

- feasibility: ``(topology - unbalanced) / unbalanced``
- objective gap: ``(unbalanced - topology) / unbalanced``

Positive values therefore favor the topology-aware tuned-unbalanced
method for all rendered metrics.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from plots.compare_before_after import (
    COMPARATOR_SPECS,
    HARDWARE_LABELS,
    HARDWARE_ORDER,
    exclude_families,
    family_size_label,
    load_instance_csv,
    ordered_blocks,
    paper_pdf_path,
    plot_before_after_heatmap,
    resolve_input_path,
    save_figure,
    shared_color_scale,
    shared_instance_keys,
    stage_title,
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
    cap_infinite_heatmap_values,
    cop_heatmap_cmap,
    heatmap_figsize_for_cells,
    heatmap_text_color_white_default,
    style_heatmap_xaxis,
    style_heatmap_yaxis,
)

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "plots" / "compare_before_after_output"
)
DEFAULT_LOGICAL_INPUT_CANDIDATES = (
    EXPERIMENTS_DIR
    / "results"
    / "compare_unb_pen_up_projection"
    / "logical"
    / "cop_instance_summary.csv",
    REPO_ROOT
    / "results"
    / "compare_unb_pen_up_projection"
    / "logical"
    / "cop_instance_summary.csv",
)
DEFAULT_EMBEDDING_INPUT_CANDIDATES = (
    EXPERIMENTS_DIR
    / "results"
    / "compare_unb_pen_up_projection"
    / "embedding"
    / "sqa_chain_strength_fraction_1"
    / "cop_instance_summary.csv",
    REPO_ROOT
    / "results"
    / "compare_unb_pen_up_projection"
    / "embedding"
    / "sqa_chain_strength_fraction_1"
    / "cop_instance_summary.csv",
)
LOGICAL_TOPOLOGY_METHOD = {
    "chimera": "projected_chimera_unb_pen",
    "pegasus": "projected_pegasus_unb_pen",
    "zephyr": "projected_zephyr_unb_pen",
}
EMBEDDED_TOPOLOGY_METHOD = "projected_topology_unb_pen"
DELTA_COP_HEATMAP_CELL_WIDTH_SCALE = 1.2
FIGURE_SPEC = replace(
    COMPARATOR_SPECS["unbalanced"],
    figure_stem=(
        "topology_vs_unbalanced_before_after_"
        "cop_heatmap_up_projection"
    ),
    summary_name=(
        "topology_vs_unbalanced_before_after_"
        "up_projection_summary.csv"
    ),
)


@dataclass(frozen=True)
class RelativeMetricSpec:
    """One CoP-style heatmap using relative change."""

    key: str
    column: str
    aggregation: str
    figure_stem: str
    summary_name: str
    title_prefix: str
    colorbar_label: str = "relative change (%)"


@dataclass(frozen=True)
class DeltaMetricSpec:
    """One heatmap using signed relative improvement vs unbalanced."""

    key: str
    column: str
    aggregation: str
    maximize: bool
    figure_stem: str
    summary_name: str
    title_prefix: str
    colorbar_label: str


RELATIVE_METRICS = (
    RelativeMetricSpec(
        key="cop_median",
        column="sqa_logical_cop",
        aggregation="median",
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "cop_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "up_projection_summary.csv"
        ),
        title_prefix="$\\Delta$ CoP\nMedian ",
    ),
    RelativeMetricSpec(
        key="cop_mean",
        column="sqa_logical_cop",
        aggregation="mean",
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "cop_mean_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "cop_mean_up_projection_summary.csv"
        ),
        title_prefix="$\\Delta$ CoP ",
    ),
)

DELTA_METRICS = (
    DeltaMetricSpec(
        key="fea",
        column="sqa_fea",
        aggregation="median",
        maximize=True,
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "fea_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "fea_up_projection_summary.csv"
        ),
        title_prefix="Feasibility ",
        colorbar_label="median relative improvement (%)",
    ),
    DeltaMetricSpec(
        key="fea_mean",
        column="sqa_fea",
        aggregation="mean",
        maximize=True,
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "fea_mean_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "fea_mean_up_projection_summary.csv"
        ),
        title_prefix="Mean Feasibility ",
        colorbar_label="mean relative improvement (%)",
    ),
    DeltaMetricSpec(
        key="gap",
        column="sqa_gap",
        aggregation="median",
        maximize=False,
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "gap_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "gap_up_projection_summary.csv"
        ),
        title_prefix="Objective Gap ",
        colorbar_label="median relative improvement (%)",
    ),
    DeltaMetricSpec(
        key="gap_mean",
        column="sqa_gap",
        aggregation="mean",
        maximize=False,
        figure_stem=(
            "topology_vs_unbalanced_before_after_"
            "gap_mean_heatmap_up_projection"
        ),
        summary_name=(
            "topology_vs_unbalanced_before_after_"
            "gap_mean_up_projection_summary.csv"
        ),
        title_prefix="",
        colorbar_label="mean relative improvement (%)",
    ),
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logical-input",
        type=Path,
        default=None,
        help="Path to the logical cop_instance_summary.csv.",
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
        help="Directory where the figures and summary CSVs are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser


def relative_change_percent(
    topology_stat: float,
    comparator_stat: float,
) -> float:
    """Return relative percent change of topology vs comparator."""
    if not math.isfinite(topology_stat):
        return math.nan
    if not math.isfinite(comparator_stat):
        return math.nan
    if topology_stat == 0.0 and comparator_stat == 0.0:
        return math.nan
    if comparator_stat == 0.0:
        if topology_stat > 0.0:
            return math.inf
        return math.nan
    return (
        (float(topology_stat) - float(comparator_stat))
        / float(comparator_stat)
        * 100.0
    )


def paired_delta(
    topology_value: float,
    comparator_value: float,
    *,
    maximize: bool,
) -> float:
    """Return one signed relative improvement where positive favors topology.

    The result is normalized by the unbalanced comparator:

    - maximize=True:  (topology - comparator) / comparator
    - maximize=False: (comparator - topology) / comparator
    """
    if not math.isfinite(topology_value):
        return math.nan
    if not math.isfinite(comparator_value):
        return math.nan
    if topology_value == 0.0 and comparator_value == 0.0:
        return math.nan
    if comparator_value == 0.0:
        if maximize:
            if topology_value > 0.0:
                return math.inf
            return math.nan
        if topology_value > 0.0:
            return -math.inf
        return math.nan
    if maximize:
        return (
            float(topology_value) - float(comparator_value)
        ) / float(comparator_value)
    return (
        float(comparator_value) - float(topology_value)
    ) / float(comparator_value)


def summarize_relative_stage_pairs(
    topology_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    *,
    stage: str,
    hardware_family: str,
    metric: RelativeMetricSpec,
) -> pd.DataFrame:
    """Return one per-block CoP summary for matched rows."""
    merge_keys = shared_instance_keys(
        topology_df, comparator_df
    )
    topology_cols = merge_keys + [metric.column]
    comparator_cols = merge_keys + [metric.column]
    merged = (
        topology_df[topology_cols]
        .rename(columns={metric.column: "topology_value"})
        .merge(
            comparator_df[comparator_cols].rename(
                columns={metric.column: "comparator_value"}
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
                "topology_stat",
                "comparator_stat",
                "cop_is_zero_zero",
                "relative_change_pct",
                "metric_key",
                "metric_column",
                "aggregation",
            ]
        )

    if metric.aggregation not in {"median", "mean"}:
        raise ValueError(
            f"unsupported aggregation: {metric.aggregation}"
        )

    grouped = merged.groupby(
        ["family", "size"],
        as_index=False,
    ).agg(
        n_pairs=("instance_index", "count"),
        topology_stat=(
            "topology_value",
            metric.aggregation,
        ),
        comparator_stat=(
            "comparator_value",
            metric.aggregation,
        ),
    )
    grouped["stage"] = stage
    grouped["hardware_family"] = hardware_family
    grouped["cop_is_zero_zero"] = (
        grouped["topology_stat"] == 0.0
    ) & (grouped["comparator_stat"] == 0.0)
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: relative_change_percent(
            float(row["topology_stat"]),
            float(row["comparator_stat"]),
        ),
        axis=1,
    )
    grouped["metric_key"] = metric.key
    grouped["metric_column"] = metric.column
    grouped["aggregation"] = metric.aggregation
    return grouped[
        [
            "stage",
            "hardware_family",
            "family",
            "size",
            "n_pairs",
            "topology_stat",
            "comparator_stat",
            "cop_is_zero_zero",
            "relative_change_pct",
            "metric_key",
            "metric_column",
            "aggregation",
        ]
    ].copy()


def summarize_delta_stage_pairs(
    topology_df: pd.DataFrame,
    comparator_df: pd.DataFrame,
    *,
    stage: str,
    hardware_family: str,
    metric: DeltaMetricSpec,
) -> pd.DataFrame:
    """Return one per-block relative-improvement summary for matched rows."""
    merge_keys = shared_instance_keys(
        topology_df, comparator_df
    )
    topology_cols = merge_keys + [metric.column]
    comparator_cols = merge_keys + [metric.column]
    merged = (
        topology_df[topology_cols]
        .rename(columns={metric.column: "topology_value"})
        .merge(
            comparator_df[comparator_cols].rename(
                columns={metric.column: "comparator_value"}
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
                "topology_stat",
                "comparator_stat",
                "delta_value",
                "metric_key",
                "metric_column",
                "aggregation",
            ]
        )

    if metric.aggregation not in {"median", "mean"}:
        raise ValueError(
            f"unsupported aggregation: {metric.aggregation}"
        )

    grouped = merged.groupby(
        ["family", "size"],
        as_index=False,
    ).agg(
        n_pairs=("instance_index", "count"),
        topology_stat=(
            "topology_value",
            metric.aggregation,
        ),
        comparator_stat=(
            "comparator_value",
            metric.aggregation,
        ),
    )
    grouped["stage"] = stage
    grouped["hardware_family"] = hardware_family
    grouped["delta_value"] = grouped.apply(
        lambda row: paired_delta(
            float(row["topology_stat"]),
            float(row["comparator_stat"]),
            maximize=metric.maximize,
        ),
        axis=1,
    )
    grouped["metric_key"] = metric.key
    grouped["metric_column"] = metric.column
    grouped["aggregation"] = metric.aggregation
    return grouped[
        [
            "stage",
            "hardware_family",
            "family",
            "size",
            "n_pairs",
            "topology_stat",
            "comparator_stat",
            "delta_value",
            "metric_key",
            "metric_column",
            "aggregation",
        ]
    ].copy()


def build_before_relative_summary(
    logical_df: pd.DataFrame,
    metric: RelativeMetricSpec,
) -> pd.DataFrame:
    """Return the pre-embedding relative summary."""
    comparator_df = logical_df[
        logical_df["method"] == "unbalanced"
    ].copy()
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        topology_df = logical_df[
            logical_df["method"]
            == LOGICAL_TOPOLOGY_METHOD[hardware_family]
        ].copy()
        if (
            "projection_topology_family"
            in topology_df.columns
        ):
            topology_df = topology_df[
                topology_df["projection_topology_family"]
                == hardware_family
            ].copy()
        frames.append(
            summarize_relative_stage_pairs(
                topology_df,
                comparator_df,
                stage="before",
                hardware_family=hardware_family,
                metric=metric,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_after_relative_summary(
    embedding_df: pd.DataFrame,
    metric: RelativeMetricSpec,
) -> pd.DataFrame:
    """Return the post-embedding relative summary."""
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        hardware_df = embedding_df[
            embedding_df["hardware_family"]
            == hardware_family
        ].copy()
        topology_df = hardware_df[
            hardware_df["method"]
            == EMBEDDED_TOPOLOGY_METHOD
        ].copy()
        if (
            "projection_topology_family"
            in topology_df.columns
        ):
            topology_df = topology_df[
                topology_df["projection_topology_family"]
                == hardware_family
            ].copy()
        comparator_df = hardware_df[
            hardware_df["method"] == "unbalanced"
        ].copy()
        frames.append(
            summarize_relative_stage_pairs(
                topology_df,
                comparator_df,
                stage="after",
                hardware_family=hardware_family,
                metric=metric,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_before_delta_summary(
    logical_df: pd.DataFrame,
    metric: DeltaMetricSpec,
) -> pd.DataFrame:
    """Return the pre-embedding delta summary."""
    comparator_df = logical_df[
        logical_df["method"] == "unbalanced"
    ].copy()
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        topology_df = logical_df[
            logical_df["method"]
            == LOGICAL_TOPOLOGY_METHOD[hardware_family]
        ].copy()
        if (
            "projection_topology_family"
            in topology_df.columns
        ):
            topology_df = topology_df[
                topology_df["projection_topology_family"]
                == hardware_family
            ].copy()
        frames.append(
            summarize_delta_stage_pairs(
                topology_df,
                comparator_df,
                stage="before",
                hardware_family=hardware_family,
                metric=metric,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_after_delta_summary(
    embedding_df: pd.DataFrame,
    metric: DeltaMetricSpec,
) -> pd.DataFrame:
    """Return the post-embedding delta summary."""
    frames: list[pd.DataFrame] = []
    for hardware_family in HARDWARE_ORDER:
        hardware_df = embedding_df[
            embedding_df["hardware_family"]
            == hardware_family
        ].copy()
        topology_df = hardware_df[
            hardware_df["method"]
            == EMBEDDED_TOPOLOGY_METHOD
        ].copy()
        if (
            "projection_topology_family"
            in topology_df.columns
        ):
            topology_df = topology_df[
                topology_df["projection_topology_family"]
                == hardware_family
            ].copy()
        comparator_df = hardware_df[
            hardware_df["method"] == "unbalanced"
        ].copy()
        frames.append(
            summarize_delta_stage_pairs(
                topology_df,
                comparator_df,
                stage="after",
                hardware_family=hardware_family,
                metric=metric,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def finalize_summary_df(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply stable sorting and comparator metadata."""
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
    summary_df["compare_against"] = FIGURE_SPEC.key
    summary_df["compare_against_label"] = FIGURE_SPEC.label
    return exclude_families(summary_df, FIGURE_SPEC)


def build_relative_summary_df(
    logical_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    metric: RelativeMetricSpec,
) -> pd.DataFrame:
    """Return one sorted summary table for a relative metric."""
    before_summary_df = build_before_relative_summary(
        logical_df,
        metric,
    )
    after_summary_df = build_after_relative_summary(
        embedding_df,
        metric,
    )
    summary_df = pd.concat(
        [before_summary_df, after_summary_df],
        ignore_index=True,
    )
    return finalize_summary_df(summary_df)


def build_delta_summary_df(
    logical_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    metric: DeltaMetricSpec,
) -> pd.DataFrame:
    """Return one sorted summary table for a relative-improvement metric."""
    before_summary_df = build_before_delta_summary(
        logical_df,
        metric,
    )
    after_summary_df = build_after_delta_summary(
        embedding_df,
        metric,
    )
    summary_df = pd.concat(
        [before_summary_df, after_summary_df],
        ignore_index=True,
    )
    return finalize_summary_df(summary_df)


def delta_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Return one symmetric linear colour norm centered at zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(
            vmin=-1.0, vcenter=0.0, vmax=1.0
        )
    vmax = float(np.max(np.abs(finite)))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def format_delta(value: float, suffix: str = "") -> str:
    """Format one delta annotation compactly.

    Args:
        value: The delta value to format.
        suffix: Optional suffix to append (e.g., "%" for percentages).
    """
    if not np.isfinite(value):
        if value > 0:
            return rf"$+\infty${suffix}"
        if value < 0:
            return rf"$-\infty${suffix}"
        return ""
    abs_value = abs(float(value))
    if abs_value == 0.0:
        return f"0{suffix}"
    if abs_value >= 100.0:
        return f"{value:+.0f}{suffix}"
    if abs_value >= 10.0:
        return f"{value:+.1f}{suffix}"
    if abs_value >= 1.0:
        return f"{value:+.2f}{suffix}"
    if abs_value >= 0.1:
        return f"{value:+.3f}{suffix}"
    if abs_value >= 0.01:
        return f"{value:+.4f}{suffix}"
    return f"{value:+.1e}{suffix}"


def build_delta_stage_matrix(
    summary_df: pd.DataFrame,
    blocks: list[tuple[str, int]],
    *,
    scale_by_100: bool = False,
    suffix: str = "",
) -> tuple[np.ndarray, list[list[str]]]:
    """Return one delta matrix and matching annotations.

    Args:
        summary_df: The summary dataframe with delta values.
        blocks: List of (family, size) tuples defining row order.
        scale_by_100: If True, multiply delta values by 100 so unitless
            relative-improvement ratios render as percentages.
        suffix: Optional suffix to append to formatted values (e.g., "%").
    """
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
            value = float(row.get("delta_value", math.nan))
            if scale_by_100:
                value *= 100.0
            matrix[row_index, column_index] = value
            annotations[row_index][column_index] = (
                format_delta(value, suffix)
            )
    return matrix, annotations


def plot_delta_heatmap(
    summary_df: pd.DataFrame,
    metric: DeltaMetricSpec,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one before/after heatmap for a delta metric."""
    before_df = summary_df[
        summary_df["stage"] == "before"
    ].copy()
    after_df = summary_df[
        summary_df["stage"] == "after"
    ].copy()
    blocks = ordered_blocks(summary_df)
    row_labels = [
        family_size_label(family, size)
        for family, size in blocks
    ]
    # Relative-improvement values are stored as unitless ratios,
    # so multiply by 100 to express them as percentages.
    scale_by_100 = metric.key in (
        "gap",
        "gap_mean",
        "fea",
        "fea_mean",
    )
    suffix = "%" if scale_by_100 else ""
    before_matrix, before_annotations = (
        build_delta_stage_matrix(
            before_df,
            blocks,
            scale_by_100=scale_by_100,
            suffix=suffix,
        )
    )
    after_matrix, after_annotations = (
        build_delta_stage_matrix(
            after_df,
            blocks,
            scale_by_100=scale_by_100,
            suffix=suffix,
        )
    )

    all_values = np.concatenate(
        [before_matrix.ravel(), after_matrix.ravel()]
    )
    norm = delta_norm(all_values)
    cmap = cop_heatmap_cmap()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=heatmap_figsize_for_cells(
            len(blocks),
            len(HARDWARE_ORDER),
            num_panels=2,
        ),
        constrained_layout=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes)
    image = None

    positive_cap = float(
        norm.vmax if norm.vmax is not None else 1.0
    )
    negative_cap = float(
        norm.vmin
        if norm.vmin is not None
        else -positive_cap
    )

    for axis, stage, matrix, annotations in zip(
        axes_array,
        ("before", "after"),
        (before_matrix, after_matrix),
        (before_annotations, after_annotations),
    ):
        matrix_for_color = cap_infinite_heatmap_values(
            matrix,
            positive_cap=positive_cap,
            negative_cap=negative_cap,
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
            f"{metric.title_prefix}{stage_title(stage)}",
            fontsize=HEATMAP_PANEL_TITLE_FONTSIZE,
            pad=HEATMAP_TITLE_PAD,
        )
        style_heatmap_xaxis(
            axis,
            [
                HARDWARE_LABELS[item]
                for item in HARDWARE_ORDER
            ],
            fontsize=HEATMAP_TICK_FONTSIZE,
        )
        style_heatmap_yaxis(
            axis,
            row_labels,
            show_labels=axis is axes_array[0],
            fontsize=HEATMAP_TICK_FONTSIZE,
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

    if image is not None:
        add_standard_heatmap_colorbar(
            figure,
            image,
            ax=list(axes_array),
            label=metric.colorbar_label,
            tick_fontsize=HEATMAP_TICK_FONTSIZE,
        )

    save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    """Run the plotting workflow."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logical_input = resolve_input_path(
        args.logical_input,
        candidates=DEFAULT_LOGICAL_INPUT_CANDIDATES,
        label="logical",
    )
    embedding_input = resolve_input_path(
        args.embedding_input,
        candidates=DEFAULT_EMBEDDING_INPUT_CANDIDATES,
        label="embedding",
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logical_df = load_instance_csv(logical_input)
    embedding_df = load_instance_csv(embedding_input)

    relative_summary_by_key = {
        metric.key: build_relative_summary_df(
            logical_df,
            embedding_df,
            metric,
        )
        for metric in RELATIVE_METRICS
    }
    relative_norm, relative_vmax_for_inf = (
        shared_color_scale(
            list(relative_summary_by_key.values())
        )
    )

    delta_summary_by_key = {
        metric.key: build_delta_summary_df(
            logical_df,
            embedding_df,
            metric,
        )
        for metric in DELTA_METRICS
    }

    included_families: set[str] = set()
    for summary_df in list(
        relative_summary_by_key.values()
    ) + list(delta_summary_by_key.values()):
        included_families.update(
            str(value)
            for value in summary_df["family"]
            .dropna()
            .unique()
        )

    print(f"logical input: {logical_input}")
    print(f"embedding input: {embedding_input}")
    print(f"output dir: {output_dir}")
    print(
        "shared CoP color scale max |relative change| (%): "
        f"{relative_vmax_for_inf:.3g}"
    )
    print(
        "included families: "
        + ", ".join(sorted(included_families))
    )

    for metric in RELATIVE_METRICS:
        summary_df = relative_summary_by_key[metric.key]
        summary_path = output_dir / metric.summary_name
        output_path = (
            output_dir / f"{metric.figure_stem}.png"
        )
        summary_df.to_csv(summary_path, index=False)
        plot_before_after_heatmap(
            summary_df,
            FIGURE_SPEC,
            output_path=output_path,
            dpi=int(args.dpi),
            norm=relative_norm,
            vmax_for_inf=relative_vmax_for_inf,
            title_prefix=metric.title_prefix,
            colorbar_label=metric.colorbar_label,
            cell_width_scale=DELTA_COP_HEATMAP_CELL_WIDTH_SCALE,
        )
        print(f"wrote {summary_path.name}")
        print(f"wrote {output_path.name}")
        print(f"wrote {paper_pdf_path(output_path)}")

    for metric in DELTA_METRICS:
        summary_df = delta_summary_by_key[metric.key]
        summary_path = output_dir / metric.summary_name
        output_path = (
            output_dir / f"{metric.figure_stem}.png"
        )
        summary_df.to_csv(summary_path, index=False)
        plot_delta_heatmap(
            summary_df,
            metric,
            output_path=output_path,
            dpi=int(args.dpi),
        )
        print(f"wrote {summary_path.name}")
        print(f"wrote {output_path.name}")
        print(f"wrote {paper_pdf_path(output_path)}")


if __name__ == "__main__":
    main()
