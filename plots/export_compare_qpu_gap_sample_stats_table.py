"""Export the paper-ready MDKP QPU gap table exactly as in brainstorm/tabs.

By default this script is pinned to the analysed D-Wave benchmark session
used for the current paper table:
``experiments/results/dwave_bench/20260426T091720605886Z``.

The output is written to
``brainstorm/tabs/qpu_gap_sample_stats_table.tex`` unless ``--output`` is
supplied.

The generated table reproduces the existing paper formatting in
``brainstorm/tabs/qpu_gap_sample_stats_table.tex``:
- a commented legacy preview block at the top;
- the full ``table*`` environment;
- sample-level ``mean[min,max]`` objective gaps reported in percent;
- a second line with the feasible-sample count out of 2500;
- boldface for the lowest mean, minimum, and maximum within each
  instance/QPU block;
- boldface for the largest feasible-sample count within each
  instance/QPU block;
- ``--`` when no feasible sample was obtained.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_RESULTS_ROOT = (
    EXPERIMENTS_DIR / "results" / "dwave_bench"
)
DEFAULT_SESSION_DIR = (
    DEFAULT_RESULTS_ROOT / "20260426T091720605886Z"
)
DEFAULT_ADVANTAGE_CATALOG = (
    DEFAULT_SESSION_DIR
    / "Advantage_system4.1_graph_id_01d07086e1"
    / "run_metrics_catalog.csv"
)
DEFAULT_ADVANTAGE2_CATALOG = (
    DEFAULT_SESSION_DIR
    / "Advantage2_system1_graph_id_018cd24a7c"
    / "run_metrics_catalog.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "brainstorm"
    / "tabs"
    / "qpu_gap_sample_stats_table.tex"
)
DEFAULT_FAMILY_ORDER = (
    "mdkp",
    "mis",
)
EXCLUDED_FAMILIES = {"mis"}
EXCLUDED_PROBLEMS = {
    ("mdkp", 10),
    ("mdkp", 29),
    ("mdkp", 35),
}
METHOD_ORDER = (
    "unbalanced",
    "projected_full",
    "projected_topology",
)
QPU_ORDER = (
    "Advantage",
    "Advantage2",
)
DEFAULT_CAPTION = (
    "Optimal objective gaps (\\%) for MDKP benchmark instances on "
    "D-Wave Advantage and Advantage2 systems. Entries are sample-level "
    "\\(\\mathrm{mean}[\\min,\\max]\\) statistics over the "
    "\\(2500\\) returned reads; lower is better. Each populated cell "
    "additionally reports the number of feasible samples on a second "
    "line. Boldface marks indicate the lowest mean, minimum, and "
    "maximum objective gaps respectively, while boldface sample counts "
    "indicate the largest number of feasible samples among the three "
    "methods for the given instance and QPU. Entries marked ``--'' "
    "indicate that no feasible sample was obtained."
)
DEFAULT_LABEL = "tab:mdkp-gaps"


@dataclass(frozen=True)
class SampleStats:
    """Sample-level objective-gap summary for one paper-table cell."""

    mean: float
    minimum: float
    maximum: float
    feasible_count: int


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root directory containing analysed D-Wave session "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help=(
            "Analysed D-Wave benchmark session directory. Defaults to "
            "the session used by the existing paper table."
        ),
    )
    parser.add_argument(
        "--advantage-catalog",
        type=Path,
        default=None,
        help=(
            "Advantage run_metrics_catalog.csv path. When supplied, "
            "--advantage2-catalog must also be supplied."
        ),
    )
    parser.add_argument(
        "--advantage2-catalog",
        type=Path,
        default=None,
        help=(
            "Advantage2 run_metrics_catalog.csv path. When supplied, "
            "--advantage-catalog must also be supplied."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default=DEFAULT_CAPTION,
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="LaTeX label.",
    )
    parser.add_argument(
        "--percent-decimals",
        type=int,
        default=2,
        help="Decimal places used for percent mean/min/max values.",
    )
    parser.add_argument(
        "--raw-decimals",
        type=int,
        default=4,
        help="Decimal places used in the commented legacy preview block.",
    )
    parser.add_argument(
        "--no-legacy-comment-block",
        action="store_true",
        help="Omit the commented preview block at the top of the file.",
    )
    return parser


def _catalog_qpu_label(path: Path) -> str | None:
    """Return the QPU label encoded in one catalog parent directory."""
    parent_name = path.parent.name.lower()
    if "advantage2" in parent_name:
        return "Advantage2"
    if "advantage" in parent_name:
        return "Advantage"
    return None


def _discover_qpu_catalogs(
    session_dir: Path,
) -> dict[str, Path]:
    """Return the Advantage and Advantage2 catalog paths for one session."""
    resolved = session_dir.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"session directory not found: {resolved}"
        )

    catalogs: dict[str, Path] = {}
    for path in sorted(
        resolved.glob("*/run_metrics_catalog.csv")
    ):
        label = _catalog_qpu_label(path)
        if label is None:
            continue
        catalogs.setdefault(label, path.resolve())

    missing = [
        label
        for label in QPU_ORDER
        if label not in catalogs
    ]
    if missing:
        discovered = sorted(
            str(path.parent.name)
            for path in resolved.glob(
                "*/run_metrics_catalog.csv"
            )
        )
        raise FileNotFoundError(
            "session does not contain the required QPU catalogs; "
            f"missing {missing}, discovered {discovered}: {resolved}"
        )
    return {label: catalogs[label] for label in QPU_ORDER}


def _latest_session_dir(results_root: Path) -> Path:
    """Return the newest analysed session directory with both QPU catalogs."""
    resolved = results_root.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"results root not found: {resolved}"
        )

    candidates = sorted(
        (
            path
            for path in resolved.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for session_dir in candidates:
        try:
            _discover_qpu_catalogs(session_dir)
        except FileNotFoundError:
            continue
        return session_dir.resolve()

    raise FileNotFoundError(
        "no analysed D-Wave session with both Advantage and Advantage2 "
        f"catalogs was found under {resolved}"
    )


def _resolve_catalog_paths(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[Path, Path, Path | None]:
    """Resolve explicit or discovered Advantage catalog paths."""
    if (
        args.advantage_catalog is not None
        or args.advantage2_catalog is not None
    ):
        if (
            args.advantage_catalog is None
            or args.advantage2_catalog is None
        ):
            parser.error(
                "pass both --advantage-catalog and --advantage2-catalog, "
                "or neither"
            )
        return (
            Path(args.advantage_catalog).resolve(),
            Path(args.advantage2_catalog).resolve(),
            None,
        )

    if args.session_dir is not None:
        session_dir = Path(args.session_dir).resolve()
    else:
        session_dir = _latest_session_dir(Path(args.results_root))
    catalogs = _discover_qpu_catalogs(session_dir)
    return (
        catalogs["Advantage"],
        catalogs["Advantage2"],
        session_dir,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV file."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"CSV not found: {resolved}"
        )
    with resolved.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _required_catalog_columns() -> set[str]:
    """Return required run-metrics-catalog columns."""
    return {
        "status",
        "family",
        "instance_name",
        "problem_size",
        "requested_method",
        "sample_metrics_csv_relative",
    }


def _required_sample_columns() -> set[str]:
    """Return required sample-metrics columns."""
    return {"objective_gap"}


def _validate_catalog_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
) -> None:
    """Validate one run_metrics_catalog table."""
    if not rows:
        raise ValueError(f"CSV is empty: {label}")
    missing = _required_catalog_columns() - set(
        rows[0].keys()
    )
    if missing:
        raise ValueError(
            f"missing required columns {sorted(missing)} in {label}"
        )


def _validate_sample_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
) -> None:
    """Validate one sample_metrics table."""
    if not rows:
        raise ValueError(
            f"sample_metrics CSV is empty: {label}"
        )
    missing = _required_sample_columns() - set(
        rows[0].keys()
    )
    if missing:
        raise ValueError(
            f"missing required columns {sorted(missing)} in {label}"
        )


def _family_sort_key(family: str) -> tuple[int, str]:
    """Return one stable family sort key."""
    normalized = str(family)
    try:
        return (
            DEFAULT_FAMILY_ORDER.index(normalized),
            normalized,
        )
    except ValueError:
        return (len(DEFAULT_FAMILY_ORDER), normalized)


def _include_problem(
    family: str,
    size: int,
) -> bool:
    """Return whether one problem row should appear in the table."""
    normalized_family = str(family)
    if normalized_family in EXCLUDED_FAMILIES:
        return False
    return (
        normalized_family,
        int(size),
    ) not in EXCLUDED_PROBLEMS


def _parse_float(text: str | None) -> float | None:
    """Parse one optional float cell."""
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    value = float(stripped)
    if not math.isfinite(value):
        return None
    return value


def _sample_gaps(sample_path: Path) -> list[float]:
    """Return finite sample-level objective-gap values from one CSV."""
    rows = _read_rows(sample_path)
    _validate_sample_rows(rows, label=str(sample_path))
    out: list[float] = []
    for row in rows:
        value = _parse_float(row.get("objective_gap"))
        if value is None:
            continue
        out.append(float(value))
    return out


def _collect_sample_stats(
    rows: list[dict[str, str]],
    *,
    qpu_label: str,
    catalog_dir: Path,
) -> dict[
    tuple[str, int, str, str, str], SampleStats | None
]:
    """Index per-instance sample-level gap stats by row key."""
    value_lists: dict[
        tuple[str, int, str, str, str], list[float]
    ] = {}
    for row in rows:
        if (
            str(row.get("status", "")).strip().lower()
            != "success"
        ):
            continue
        family = str(row["family"])
        size = int(row["problem_size"])
        if not _include_problem(family, size):
            continue
        instance_name = str(row["instance_name"])
        method = str(row["requested_method"])
        if method not in METHOD_ORDER:
            continue
        relative = str(
            row.get("sample_metrics_csv_relative", "")
        ).strip()
        if not relative:
            continue
        sample_path = (catalog_dir / relative).resolve()
        gaps = _sample_gaps(sample_path)
        key = (
            family,
            size,
            instance_name,
            qpu_label,
            method,
        )
        value_lists.setdefault(key, []).extend(gaps)

    stats: dict[
        tuple[str, int, str, str, str], SampleStats | None
    ] = {}
    for key, values in value_lists.items():
        if not values:
            stats[key] = None
            continue
        stats[key] = SampleStats(
            mean=sum(values) / len(values),
            minimum=min(values),
            maximum=max(values),
            feasible_count=len(values),
        )
    return stats


def _all_problem_keys(
    stats_table: dict[
        tuple[str, int, str, str, str], SampleStats | None
    ],
) -> list[tuple[str, int, str]]:
    """Return sorted family/size/instance keys present in the table."""
    keys = {
        (family, size, instance_name)
        for family, size, instance_name, _qpu, _method in stats_table
    }
    return sorted(
        keys,
        key=lambda item: (
            _family_sort_key(item[0]),
            item[1],
            item[2],
        ),
    )


def _format_trimmed(
    value: float,
    decimals: int,
) -> str:
    """Format one numeric value, stripping trailing zeros."""
    text = f"{float(value):.{int(decimals)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _format_fixed(
    value: float,
    decimals: int,
) -> str:
    """Format one numeric value with fixed decimals."""
    return f"{float(value):.{int(decimals)}f}"


def _best_mean(
    stats_items: list[SampleStats | None],
) -> float | None:
    """Return the lowest mean in one QPU block."""
    values = [
        item.mean for item in stats_items if item is not None
    ]
    if not values:
        return None
    return min(values)


def _best_minimum(
    stats_items: list[SampleStats | None],
) -> float | None:
    """Return the lowest minimum in one QPU block."""
    values = [
        item.minimum for item in stats_items if item is not None
    ]
    if not values:
        return None
    return min(values)


def _best_maximum(
    stats_items: list[SampleStats | None],
) -> float | None:
    """Return the lowest maximum in one QPU block."""
    values = [
        item.maximum for item in stats_items if item is not None
    ]
    if not values:
        return None
    return min(values)


def _best_count(
    stats_items: list[SampleStats | None],
) -> int | None:
    """Return the highest feasible count in one QPU block."""
    values = [
        item.feasible_count
        for item in stats_items
        if item is not None
    ]
    if not values:
        return None
    return max(values)


def _matches_best(
    value: float | int,
    best_value: float | int | None,
) -> bool:
    """Return whether one numeric value matches the current best."""
    if best_value is None:
        return False
    if isinstance(value, int) and isinstance(best_value, int):
        return int(value) == int(best_value)
    return math.isclose(
        float(value),
        float(best_value),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _legacy_cell_text(
    stats: SampleStats | None,
    *,
    decimals: int,
    bold_entire_cell: bool,
) -> str:
    """Return one legacy preview cell."""
    if stats is None:
        return "--"
    text = "/".join(
        [
            _format_trimmed(stats.mean, decimals),
            _format_trimmed(stats.minimum, decimals),
            _format_trimmed(stats.maximum, decimals),
        ]
    )
    if bold_entire_cell:
        return rf"\textbf{{{text}}}"
    return text


def _build_legacy_comment_block(
    *,
    stats_table: dict[
        tuple[str, int, str, str, str], SampleStats | None
    ],
    raw_decimals: int,
) -> str:
    """Build the commented legacy preview block."""
    lines = [
        r"% \begin{tabular}{@{}llcccccc@{}}",
        r"%   \toprule",
        r"%   & & \multicolumn{3}{c}{Advantage (Pegasus)} & \multicolumn{3}{c}{Advantage2 (Zephyr)} \\",
        r"%   \cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"%   Instance & $n$ & Unb. & Full & Topo. & Unb. & Full & Topo. \\",
        r"%   \midrule",
    ]

    problem_keys = _all_problem_keys(stats_table)
    for family, size, instance_name in problem_keys:
        formatted_values: list[str] = []
        for qpu in QPU_ORDER:
            block_stats = [
                stats_table.get(
                    (
                        family,
                        size,
                        instance_name,
                        qpu,
                        method,
                    )
                )
                for method in METHOD_ORDER
            ]
            best_mean = _best_mean(block_stats)
            formatted_values.extend(
                [
                    _legacy_cell_text(
                        stats,
                        decimals=raw_decimals,
                        bold_entire_cell=(
                            stats is not None
                            and _matches_best(
                                stats.mean, best_mean
                            )
                        ),
                    )
                    for stats in block_stats
                ]
            )
        row = " & ".join(
            [
                str(instance_name),
                str(int(size)),
                *formatted_values,
            ]
        )
        lines.append("%   " + row + r" \\")

    lines.extend(
        [
            r"%   \bottomrule",
            r"% \end{tabular}",
        ]
    )
    return "\n".join(lines)


def _caption_lines(caption: str) -> list[str]:
    """Return the caption block with paper-matching line breaks."""
    if str(caption) != DEFAULT_CAPTION:
        return [f"\\caption{{{caption}}}"]
    return [
        r"\caption{ Optimal objective gaps (\%) for MDKP benchmark instances on",
        r"  D-Wave Advantage and Advantage2 systems. Entries are sample-level",
        r"  \(\mathrm{mean}[\min,\max]\) statistics over the \(2500\) returned",
        r"  reads; lower is better. Each populated cell additionally reports the",
        r"  number of feasible samples on a second line. Boldface marks indicate",
        r"  the lowest mean, minimum, and maximum objective gaps respectively,",
        r"  while boldface sample counts indicate the largest number of feasible",
        r"  samples among the three methods for the given instance and QPU.",
        r"  Entries marked ``--'' indicate that no feasible sample was obtained. }",
    ]


def _format_main_cell(
    stats: SampleStats | None,
    *,
    percent_decimals: int,
    bold_mean: bool,
    bold_minimum: bool,
    bold_maximum: bool,
    bold_count: bool,
) -> str:
    """Return one paper-table cell."""
    if stats is None:
        return "--"

    mean_text = _format_fixed(
        100.0 * float(stats.mean),
        percent_decimals,
    )
    min_text = _format_fixed(
        100.0 * float(stats.minimum),
        percent_decimals,
    )
    max_text = _format_fixed(
        100.0 * float(stats.maximum),
        percent_decimals,
    )
    count_text = str(int(stats.feasible_count))

    if bold_mean:
        mean_text = rf"\textbf{{{mean_text}}}"
    if bold_minimum:
        min_text = rf"\textbf{{{min_text}}}"
    if bold_maximum:
        max_text = rf"\textbf{{{max_text}}}"
    if bold_count:
        count_text = rf"\textbf{{{count_text}}}"

    return (
        rf"\gapcell{{{mean_text} [{min_text}, {max_text}]}}{{{count_text}}}"
    )


def build_latex_table(
    *,
    stats_table: dict[
        tuple[str, int, str, str, str], SampleStats | None
    ],
    percent_decimals: int,
    raw_decimals: int,
    caption: str,
    label: str,
    include_legacy_comment_block: bool,
) -> str:
    """Build the exact paper-style LaTeX table source."""
    lines: list[str] = []
    if include_legacy_comment_block:
        lines.append(
            _build_legacy_comment_block(
                stats_table=stats_table,
                raw_decimals=raw_decimals,
            )
        )
        lines.append("")

    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            *_caption_lines(caption),
            f"\\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\newcommand{\gapcell}[2]{\begin{tabular}[t]{@{}r@{}}#1\\{\scriptsize \(#2/2500\)}\end{tabular}}",
            r"\begin{tabular}{@{}llrrrrrr@{}}",
            r"  \toprule",
            r"  & & \multicolumn{3}{c}{Advantage (Pegasus)} & \multicolumn{3}{c}{Advantage2 (Zephyr)} \\",
            r"  \cmidrule(lr){3-5} \cmidrule(lr){6-8}",
            r"  Instance & \(n\) & \acrshort{up} \cite{montanez2024unbalanced} & \acrshort{methodfull} & \acrshort{methodtopo} &  \acrshort{up} \cite{montanez2024unbalanced} & \acrshort{methodfull} & \acrshort{methodtopo} \\",
            r"  \midrule",
        ]
    )

    problem_keys = _all_problem_keys(stats_table)
    for index, (family, size, instance_name) in enumerate(
        problem_keys
    ):
        lines.append(
            rf"  \begin{{tabular}}[t]{{@{{}}l@{{}}}}{instance_name}\end{{tabular}} & \begin{{tabular}}[t]{{@{{}}l@{{}}}}{int(size)}\end{{tabular}}"
        )
        formatted_values: list[str] = []
        for qpu in QPU_ORDER:
            block_stats = [
                stats_table.get(
                    (
                        family,
                        size,
                        instance_name,
                        qpu,
                        method,
                    )
                )
                for method in METHOD_ORDER
            ]
            best_mean = _best_mean(block_stats)
            best_minimum = _best_minimum(block_stats)
            best_maximum = _best_maximum(block_stats)
            best_count = _best_count(block_stats)
            for stats in block_stats:
                formatted_values.append(
                    _format_main_cell(
                        stats,
                        percent_decimals=percent_decimals,
                        bold_mean=(
                            stats is not None
                            and _matches_best(
                                stats.mean, best_mean
                            )
                        ),
                        bold_minimum=(
                            stats is not None
                            and _matches_best(
                                stats.minimum,
                                best_minimum,
                            )
                        ),
                        bold_maximum=(
                            stats is not None
                            and _matches_best(
                                stats.maximum,
                                best_maximum,
                            )
                        ),
                        bold_count=(
                            stats is not None
                            and _matches_best(
                                stats.feasible_count,
                                best_count,
                            )
                        ),
                    )
                )
        lines.append(
            "  & " + "\n  & ".join(formatted_values) + r" \\"
        )
        if index != len(problem_keys) - 1:
            lines.append("")

    lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the export CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    advantage_catalog, advantage2_catalog, session_dir = (
        _resolve_catalog_paths(
            args,
            parser,
        )
    )
    if session_dir is not None:
        print(f"using session {session_dir}")

    advantage_rows = _read_rows(advantage_catalog)
    advantage2_rows = _read_rows(advantage2_catalog)
    _validate_catalog_rows(
        advantage_rows,
        label=str(advantage_catalog),
    )
    _validate_catalog_rows(
        advantage2_rows,
        label=str(advantage2_catalog),
    )

    stats_table: dict[
        tuple[str, int, str, str, str], SampleStats | None
    ] = {}
    stats_table.update(
        _collect_sample_stats(
            advantage_rows,
            qpu_label="Advantage",
            catalog_dir=advantage_catalog.parent,
        )
    )
    stats_table.update(
        _collect_sample_stats(
            advantage2_rows,
            qpu_label="Advantage2",
            catalog_dir=advantage2_catalog.parent,
        )
    )

    latex = build_latex_table(
        stats_table=stats_table,
        percent_decimals=int(args.percent_decimals),
        raw_decimals=int(args.raw_decimals),
        caption=str(args.caption),
        label=str(args.label),
        include_legacy_comment_block=not bool(
            args.no_legacy_comment_block
        ),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
