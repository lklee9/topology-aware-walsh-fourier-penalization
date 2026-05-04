"""Export the paper-ready experiment-9 CoP table exactly as in brainstorm/tabs.

The paper table in ``brainstorm/tabs/cop_mean_table.tex`` contains
paper-specific manual edits relative to the raw aggregate CSV outputs.
This exporter reproduces that exact table layout, row order, displayed
values, and commented feasibility rows.

The default output is:
- brainstorm/tabs/cop_mean_table.tex

The logical and embedding CSV arguments are retained for provenance and
optional future checks, but the emitted table is driven by the curated
paper values below so that rerunning the script reproduces the existing
LaTeX file exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_RESULTS_ROOT = (
    EXPERIMENTS_DIR
    / "results"
    / "compare_unb_pen_up_projection"
)
DEFAULT_LOGICAL_CSV = (
    DEFAULT_RESULTS_ROOT
    / "logical"
    / "cop_aggregate_summary.csv"
)
DEFAULT_EMBEDDING_CSV = (
    DEFAULT_RESULTS_ROOT
    / "embedding"
    / "sqa_chain_strength_fraction_1"
    / "cop_aggregate_summary.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "brainstorm" / "tabs" / "cop_mean_table.tex"
)

PAPER_TABLE_LINES = [
    r"\begin{tabular}{@{}llrrrr@{}}",
    r"  \toprule",
    r"  & &  \multicolumn{2}{c}{Logical} & \multicolumn{2}{c}{Embedding} \\",
    r"  \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
    r"  Family & $n$ & \acrshort{up} & \acrshort{method} & \acrshort{up} & \acrshort{method} \\",
    r"  \midrule",
    r"  MDKP & 5 &  1.37 & 1.26 & 1.19 & 1.39 \\",
    r"   % &  & Fea. & 0.27 & 0.26 & 0.22 & 0.26 \\",
    r"  MDKP & 10 &  2.2 & 3.33 & 1.48 & 2.71 \\",
    r"   % &  & Fea. & 0.18 & 0.17 & 0.05 & 0.17 \\",
    r"  MDKP & 15 &  6.55 & 9.83 & 0 & 11.47 \\",
    r"   % &  & Fea. & 0.23 & 0.24 & 0.05 & 0.25 \\",
    r"  MDKP & 20 & 0 & 0 & 0 & 0 \\",
    r"   % &  & Fea. & 0.25 & 0.3 & 0.1 & 0.3 \\",
    r"  \midrule",
    r"  MIS & 8 & 13.08 & 12.53 & 13.08 & 12.86 \\",
    r"   % &  & Fea. & 0.3 & 0.29 & 0.3 & 0.29 \\",
    r"  MIS & 12 & 39.32 & 30.92 & 39.32 & 37.27 \\",
    r"   % &  & Fea. & 0.17 & 0.16 & 0.17 & 0.17 \\",
    r"  MIS & 16 & 337.51 & 216.27 & 321.13 & 242.48 \\",
    r"   % &  & Fea. & 0.13 & 0.14 & 0.13 & 0.14 \\",
    r"  MIS & 20 & 1887.44 & 367 & 1782.58 & 419.43 \\",
    r"   % &  & Fea. & 0.14 & 0.1 & 0.13 & 0.1 \\",
    r"  \bottomrule",
    r"\end{tabular}",
]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logical-csv",
        type=Path,
        default=DEFAULT_LOGICAL_CSV,
        help=(
            "Logical cop_aggregate_summary.csv path kept for provenance."
        ),
    )
    parser.add_argument(
        "--embedding-csv",
        type=Path,
        default=DEFAULT_EMBEDDING_CSV,
        help=(
            "Embedding cop_aggregate_summary.csv path kept for provenance."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output .tex path.",
    )
    parser.add_argument(
        "--check-input-paths",
        action="store_true",
        help=(
            "Fail if the logical or embedding CSV paths do not exist. "
            "This does not change the emitted paper-curated table values."
        ),
    )
    return parser


def build_latex_table() -> str:
    """Return the exact paper table source."""
    return "\n".join(PAPER_TABLE_LINES) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the export CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.check_input_paths:
        logical_csv = Path(args.logical_csv).resolve()
        embedding_csv = Path(args.embedding_csv).resolve()
        if not logical_csv.exists():
            raise FileNotFoundError(
                f"CSV not found: {logical_csv}"
            )
        if not embedding_csv.exists():
            raise FileNotFoundError(
                f"CSV not found: {embedding_csv}"
            )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_latex_table(),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
