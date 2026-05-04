"""Plot mean topology-vs-unbalanced CoP heatmap before and after embedding.

This mirrors ``plots/compare_before_after.py`` for the unbalanced
comparator but replaces the median CoP aggregation with a mean CoP
aggregation.

Rendering is delegated to
``compare_before_after.plot_before_after_heatmap()``, which uses the
shared heatmap sizing/styling config in ``plots/heatmap_layout.py``.

Each cell shows

``(mean(topology CoP) - mean(unbalanced CoP)) / mean(unbalanced CoP) * 100``

computed within one ``(family, size, hardware)`` block after strict
same-instance matching. Positive values favor the topology-constrained
method over unbalanced penalization.
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
    BASELINE_TOPOLOGY_METHOD,
    COMPARATOR_SPECS,
    DEFAULT_BASELINE_INPUT_CANDIDATES,
    DEFAULT_EMBEDDING_INPUT_CANDIDATES,
    HARDWARE_ORDER,
    exclude_families,
    load_instance_csv,
    paper_pdf_path,
    plot_before_after_heatmap,
    resolve_input_path,
    shared_color_scale,
    shared_instance_keys,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "plots"
    / "compare_before_after_output"
)
SPEC = COMPARATOR_SPECS["unbalanced"]
FIGURE_STEM = (
    "topology_vs_unbalanced_before_after_cop_heatmap_mean"
)
SUMMARY_CSV_NAME = (
    "topology_vs_unbalanced_before_after_summary_mean.csv"
)


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
        help="Directory where figure and summary CSV are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser


def relative_change_percent(
    topology_mean: float, comparator_mean: float
) -> float:
    """Return relative percent change of topology vs comparator."""
    if not math.isfinite(topology_mean):
        return math.nan
    if not math.isfinite(comparator_mean):
        return math.nan
    if topology_mean == 0.0 and comparator_mean == 0.0:
        return math.nan
    if comparator_mean == 0.0:
        if topology_mean > 0.0:
            return math.inf
        return math.nan
    return (
        (float(topology_mean) - float(comparator_mean))
        / float(comparator_mean)
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
                "topology_cop_mean",
                "comparator_cop_mean",
                "cop_is_zero_zero",
                "relative_change_pct",
            ]
        )

    grouped = merged.groupby(
        ["family", "size"], as_index=False
    ).agg(
        n_pairs=("instance_index", "count"),
        topology_cop_mean=("topology_cop", "mean"),
        comparator_cop_mean=("comparator_cop", "mean"),
    )
    grouped["stage"] = stage
    grouped["hardware_family"] = hardware_family
    grouped["cop_is_zero_zero"] = (
        grouped["topology_cop_mean"] == 0.0
    ) & (grouped["comparator_cop_mean"] == 0.0)
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: relative_change_percent(
            float(row["topology_cop_mean"]),
            float(row["comparator_cop_mean"]),
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
            "topology_cop_mean",
            "comparator_cop_mean",
            "cop_is_zero_zero",
            "relative_change_pct",
        ]
    ].copy()


def build_before_summary(
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the pre-embedding topology summary."""
    comparator_df = baseline_df[
        baseline_df["method"] == SPEC.baseline_method
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
    return pd.concat(frames, ignore_index=True)


def build_after_summary(
    embedding_df: pd.DataFrame,
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
            hardware_df["method"] == SPEC.embedding_method
        ].copy()
        frames.append(
            summarize_stage_pairs(
                topology_df,
                comparator_df,
                stage="after",
                hardware_family=hardware_family,
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_summary_df(
    baseline_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one sorted summary table for the mean heatmap."""
    before_summary_df = build_before_summary(baseline_df)
    after_summary_df = build_after_summary(embedding_df)
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
    summary_df["compare_against"] = SPEC.key
    summary_df["compare_against_label"] = SPEC.label
    return exclude_families(summary_df, SPEC)


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
    norm, vmax_for_inf = shared_color_scale([summary_df])

    summary_path = output_dir / SUMMARY_CSV_NAME
    output_path = output_dir / f"{FIGURE_STEM}.png"
    summary_df.to_csv(summary_path, index=False)
    plot_before_after_heatmap(
        summary_df,
        SPEC,
        output_path=output_path,
        dpi=int(args.dpi),
        norm=norm,
        vmax_for_inf=vmax_for_inf,
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
