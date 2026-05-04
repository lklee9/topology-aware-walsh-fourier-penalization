"""Plot mean full-projection vs unbalanced CoP heatmap before/after embedding.

This mirrors ``plots/compare_full_vs_unbalanced.py`` but replaces the
median CoP aggregation with a mean CoP aggregation.

Rendering is delegated to
``compare_full_vs_unbalanced.plot_full_vs_unbalanced_heatmap()``, which
uses the shared heatmap sizing/styling config in ``plots/heatmap_layout.py``.

Each cell shows

``(mean(full CoP) - mean(unbalanced CoP)) / mean(unbalanced CoP) * 100``

computed from strict same-instance matches within one ``(family, size)``
block. Positive values favor ``projected_full``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

import pandas as pd

from plots.compare_before_after import (
    DEFAULT_BASELINE_INPUT_CANDIDATES,
    DEFAULT_EMBEDDING_INPUT_CANDIDATES,
    load_instance_csv,
    paper_pdf_path,
    resolve_input_path,
    shared_instance_keys,
)
from plots.compare_full_vs_unbalanced import (
    COLUMN_LABELS,
    COLUMN_ORDER,
    DEFAULT_OUTPUT_DIR,
    plot_full_vs_unbalanced_heatmap,
)

FIGURE_STEM = "full_vs_unbalanced_before_after_embedding_cop_heatmap_mean"
SUMMARY_CSV_NAME = "full_vs_unbalanced_before_after_embedding_summary_mean.csv"
EXCLUDED_FAMILIES = frozenset()


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
    full_mean: float, unbalanced_mean: float
) -> float:
    """Return relative percent change where positive favors full."""
    if not math.isfinite(full_mean):
        return math.nan
    if not math.isfinite(unbalanced_mean):
        return math.nan
    if full_mean == 0.0 and unbalanced_mean == 0.0:
        return math.nan
    if unbalanced_mean == 0.0:
        if full_mean > 0.0:
            return math.inf
        return math.nan
    return (
        (float(full_mean) - float(unbalanced_mean))
        / float(unbalanced_mean)
        * 100.0
    )


def summarize_pairwise_comparison(
    full_df: pd.DataFrame,
    unbalanced_df: pd.DataFrame,
    *,
    column_key: str,
) -> pd.DataFrame:
    """Return one per-block full-vs-unbalanced mean summary."""
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
                "column_label",
                "family",
                "size",
                "n_pairs",
                "full_cop_mean",
                "unbalanced_cop_mean",
                "cop_is_zero_zero",
                "relative_change_pct",
            ]
        )

    grouped = merged.groupby(
        ["family", "size"], as_index=False
    ).agg(
        n_pairs=("instance_index", "count"),
        full_cop_mean=("full_cop", "mean"),
        unbalanced_cop_mean=("unbalanced_cop", "mean"),
    )
    grouped["column_key"] = column_key
    grouped["column_label"] = COLUMN_LABELS[column_key]
    grouped["cop_is_zero_zero"] = (
        grouped["full_cop_mean"] == 0.0
    ) & (grouped["unbalanced_cop_mean"] == 0.0)
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: relative_change_percent(
            float(row["full_cop_mean"]),
            float(row["unbalanced_cop_mean"]),
        ),
        axis=1,
    )
    return grouped[
        [
            "column_key",
            "column_label",
            "family",
            "size",
            "n_pairs",
            "full_cop_mean",
            "unbalanced_cop_mean",
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

    for hardware_family in COLUMN_ORDER[1:]:
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
    if EXCLUDED_FAMILIES:
        summary_df = summary_df[
            ~summary_df["family"].isin(EXCLUDED_FAMILIES)
        ].copy()
    column_rank = {
        key: index for index, key in enumerate(COLUMN_ORDER)
    }
    summary_df["column_rank"] = summary_df[
        "column_key"
    ].map(column_rank)
    summary_df = summary_df.sort_values(
        by=["family", "size", "column_rank"]
    )
    summary_df = summary_df.drop(columns=["column_rank"])
    return summary_df.reset_index(drop=True)


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
        title_prefix=r"$\Delta$ CoP ",
        colorbar_label="",
    )

    print(f"baseline input: {baseline_input}")
    print(f"embedding input: {embedding_input}")
    print(f"output dir: {output_dir}")
    print(f"wrote {summary_path.name}")
    print(f"wrote {output_path.name}")
    print(f"wrote {paper_pdf_path(output_path)}")


if __name__ == "__main__":
    main()
