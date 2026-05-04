"""Plot full-projection vs unbalanced CoP heatmap before and after embedding.

This script creates one heatmap with four columns:

1. before embedding (logical full projection vs logical unbalanced);
2. after embedding on Chimera;
3. after embedding on Pegasus; and
4. after embedding on Zephyr.

Each cell shows

``(median(full CoP) - median(unbalanced CoP)) / median(unbalanced CoP) * 100``

computed from strict same-instance matches within one ``(family, size)``
block. Positive values favor ``projected_full``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.compare_before_after import (
    DEFAULT_BASELINE_INPUT_CANDIDATES,
    DEFAULT_EMBEDDING_INPUT_CANDIDATES,
    delta_log_norm,
    family_size_label,
    format_relative_change,
    load_instance_csv,
    ordered_blocks,
    paper_pdf_path,
    resolve_input_path,
    save_figure,
    shared_instance_keys,
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

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "plots"
    / "compare_full_vs_unbalanced_output"
)
FIGURE_STEM = (
    "full_vs_unbalanced_before_after_embedding_cop_heatmap"
)
SUMMARY_CSV_NAME = (
    "full_vs_unbalanced_before_after_embedding_summary.csv"
)
COLUMN_ORDER = (
    "before_embedding",
    "chimera",
    "pegasus",
    "zephyr",
)
COLUMN_LABELS = {
    "before_embedding": "Before",
    "chimera": "Chimera",
    "pegasus": "Pegasus",
    "zephyr": "Zephyr",
}


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
        help="Directory where the figure and summary CSV are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser


def relative_change_percent(
    full_median: float,
    unbalanced_median: float,
) -> float:
    """Return relative percent change where positive favors full."""
    if not math.isfinite(full_median):
        return math.nan
    if not math.isfinite(unbalanced_median):
        return math.nan
    if full_median == 0.0 and unbalanced_median == 0.0:
        return math.nan
    if unbalanced_median == 0.0:
        if full_median > 0.0:
            return math.inf
        return math.nan
    return (
        (float(full_median) - float(unbalanced_median))
        / float(unbalanced_median)
        * 100.0
    )


def summarize_pairwise_comparison(
    full_df: pd.DataFrame,
    unbalanced_df: pd.DataFrame,
    *,
    column_key: str,
) -> pd.DataFrame:
    """Return one per-block full-vs-unbalanced summary."""
    merge_keys = shared_instance_keys(
        full_df, unbalanced_df
    )
    full_cols = merge_keys + ["sqa_logical_cop"]
    unbalanced_cols = merge_keys + ["sqa_logical_cop"]
    merged = (
        full_df[full_cols]
        .rename(columns={"sqa_logical_cop": "full_cop"})
        .merge(
            unbalanced_df[unbalanced_cols].rename(
                columns={
                    "sqa_logical_cop": "unbalanced_cop"
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
                "column_key",
                "family",
                "size",
                "n_pairs",
                "full_cop_median",
                "unbalanced_cop_median",
                "cop_is_zero_zero",
                "relative_change_pct",
            ]
        )

    grouped = merged.groupby(
        ["family", "size"],
        as_index=False,
    ).agg(
        n_pairs=("instance_index", "count"),
        full_cop_median=("full_cop", "median"),
        unbalanced_cop_median=("unbalanced_cop", "median"),
    )
    grouped["column_key"] = column_key
    grouped["cop_is_zero_zero"] = (
        grouped["full_cop_median"] == 0.0
    ) & (grouped["unbalanced_cop_median"] == 0.0)
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: relative_change_percent(
            float(row["full_cop_median"]),
            float(row["unbalanced_cop_median"]),
        ),
        axis=1,
    )
    grouped["column_label"] = COLUMN_LABELS[column_key]
    return grouped[
        [
            "column_key",
            "column_label",
            "family",
            "size",
            "n_pairs",
            "full_cop_median",
            "unbalanced_cop_median",
            "cop_is_zero_zero",
            "relative_change_pct",
        ]
    ].copy()


def build_summary_df(
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one sorted summary table for the four-column heatmap."""
    frames: list[pd.DataFrame] = []

    before_full_df = baseline_df[
        baseline_df["method"] == "projected_full"
    ].copy()
    before_unbalanced_df = baseline_df[
        baseline_df["method"] == "unbalanced"
    ].copy()
    frames.append(
        summarize_pairwise_comparison(
            before_full_df,
            before_unbalanced_df,
            column_key="before_embedding",
        )
    )

    for hardware_family in ("chimera", "pegasus", "zephyr"):
        hardware_df = embedding_df[
            embedding_df["hardware_family"]
            == hardware_family
        ].copy()
        full_df = hardware_df[
            hardware_df["method"] == "projected_full"
        ].copy()
        unbalanced_df = hardware_df[
            hardware_df["method"] == "unbalanced"
        ].copy()
        frames.append(
            summarize_pairwise_comparison(
                full_df,
                unbalanced_df,
                column_key=hardware_family,
            )
        )

    summary_df = pd.concat(frames, ignore_index=True)
    column_rank = {
        key: index for index, key in enumerate(COLUMN_ORDER)
    }
    summary_df["column_rank"] = summary_df[
        "column_key"
    ].map(column_rank)
    summary_df = summary_df.sort_values(
        by=["family", "size", "column_rank"]
    ).drop(columns=["column_rank"])
    summary_df = summary_df.reset_index(drop=True)
    return summary_df


def build_heatmap_matrix(
    summary_df: pd.DataFrame,
    blocks: list[tuple[str, int]],
) -> tuple[np.ndarray, list[list[str]]]:
    """Return one block-by-column matrix and cell annotations."""
    lookup = {
        (
            str(row["family"]),
            int(row["size"]),
            str(row["column_key"]),
        ): row
        for row in summary_df.to_dict("records")
    }
    matrix = np.full(
        (len(blocks), len(COLUMN_ORDER)),
        np.nan,
        dtype=float,
    )
    annotations = [
        ["" for _ in COLUMN_ORDER] for _ in blocks
    ]
    for row_index, (family, size) in enumerate(blocks):
        for column_index, column_key in enumerate(
            COLUMN_ORDER
        ):
            row = lookup.get((family, size, column_key))
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


def plot_full_vs_unbalanced_heatmap(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    dpi: int,
    title_prefix: str = "",
    colorbar_label: str = "relative change (%)",
) -> None:
    """Plot the 4-column full-vs-unbalanced heatmap."""
    blocks = ordered_blocks(summary_df)
    row_labels = [
        family_size_label(family, size)
        for family, size in blocks
    ]
    matrix, annotations = build_heatmap_matrix(
        summary_df, blocks
    )

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=float)
    vmax_for_inf = float(np.max(np.abs(finite_values)))
    if (
        not math.isfinite(vmax_for_inf)
        or vmax_for_inf <= 0.0
    ):
        vmax_for_inf = 1.0
    norm = delta_log_norm(finite_values)

    matrix_for_color = matrix.copy()
    matrix_for_color[np.isposinf(matrix_for_color)] = (
        vmax_for_inf
    )
    matrix_for_color[np.isneginf(matrix_for_color)] = (
        -vmax_for_inf
    )

    figure, axis = plt.subplots(
        1,
        1,
        figsize=heatmap_figsize_for_cells(
            len(blocks),
            len(COLUMN_ORDER),
            num_panels=1,
        ),
        constrained_layout=True,
    )
    cmap = cop_heatmap_cmap()
    masked = np.ma.masked_invalid(matrix_for_color)
    image = axis.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )
    axis.set_facecolor(HEATMAP_FACE_COLOR)
    axis.set_title(
        # "Relative CoP change: full projection vs unbalanced penalization",
        f"{title_prefix}With Embed",
        fontsize=HEATMAP_PANEL_TITLE_FONTSIZE,
        pad=HEATMAP_TITLE_PAD,
    )
    style_heatmap_xaxis(
        axis,
        [COLUMN_LABELS[key] for key in COLUMN_ORDER],
        fontsize=HEATMAP_TICK_FONTSIZE,
    )
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(
        row_labels, fontsize=HEATMAP_TICK_FONTSIZE
    )
    axis.tick_params(
        axis="y", labelsize=HEATMAP_TICK_FONTSIZE
    )
    add_heatmap_grid(axis, matrix.shape)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            text = annotations[row_index][column_index]
            value = matrix[row_index, column_index]
            if not text:
                continue
            if not np.isfinite(value):
                if np.isinf(value):
                    text_color = (
                        heatmap_text_color_white_default(
                            norm,
                            matrix_for_color[
                                row_index, column_index
                            ],
                        )
                    )
                else:
                    text_color = HEATMAP_ANNOTATION_COLOR
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                    color=text_color,
                )
                continue
            text_color = heatmap_text_color_white_default(
                norm,
                matrix_for_color[row_index, column_index],
            )
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=HEATMAP_ANNOTATION_FONTSIZE,
                color=text_color,
            )

    add_standard_heatmap_colorbar(
        figure,
        image,
        ax=axis,
        label=colorbar_label,
        tick_fontsize=HEATMAP_TICK_FONTSIZE,
    )

    save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


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
    summary_df = build_summary_df(baseline_df, embedding_df)

    summary_path = output_dir / SUMMARY_CSV_NAME
    summary_df.to_csv(summary_path, index=False)

    output_path = output_dir / f"{FIGURE_STEM}.png"
    plot_full_vs_unbalanced_heatmap(
        summary_df,
        output_path=output_path,
        dpi=int(args.dpi),
    )

    print(f"baseline input: {baseline_input}")
    print(f"embedding input: {embedding_input}")
    print(f"output dir: {output_dir}")
    print(f"wrote {summary_path.name}")
    print(f"wrote {output_path.name}")
    print(f"wrote {paper_pdf_path(output_path)}")


if __name__ == "__main__":
    main()
