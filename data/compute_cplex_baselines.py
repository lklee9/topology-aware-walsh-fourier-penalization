"""Compute cached CPLEX references for benchmark or synthetic problems."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from experiments.utils.benchmark_data import (
    BenchmarkProblemSpec,
    load_family_problem_specs,
)
from experiments.utils.cplex_reference import (
    DEFAULT_CPLEX_REFERENCE_SOURCE,
    merge_reference_rows,
    repo_relative_path,
    require_cplex_runtime,
    solve_problem_cplex_optimum,
    synthetic_reference_row,
)
from experiments.utils.cplex_reference import (
    write_csv_rows as _write_csv_rows,
)
from experiments.utils.family_cli import parse_size_list
from experiments.utils.synthetic_bench import (
    DEFAULT_SYNTHETIC_FAMILY_SIZES,
    SyntheticProblemInstance,
    load_synthetic_problem_batches,
)

DEFAULT_OUTPUT_CSV = Path(
    "data/classical_baselines/cplex_optima.csv"
)
DEFAULT_SKIPPED_CSV = Path(
    "data/classical_baselines/cplex_optima_skipped.csv"
)
DEFAULT_MIS_DIR = Path("data/mis_instances")
DEFAULT_MDKP_DIR = Path("data/mdkp_instances")
DEFAULT_FAMILIES = ("mdkp", "mis")


def _resolve_default_repo_path(
    requested_path: Path,
    *,
    default_path: Path,
) -> Path:
    """Resolve default CLI paths relative to the repo root."""
    if requested_path == default_path:
        return (REPO_ROOT / default_path).resolve()
    return requested_path.resolve()


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into row dictionaries when it exists."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _solve_cplex_optimum(
    benchmark_problem: (
        BenchmarkProblemSpec | SyntheticProblemInstance
    ),
) -> tuple[float, tuple[int, ...], str, str]:
    """Solve one benchmark or synthetic instance exactly with CPLEX."""
    return solve_problem_cplex_optimum(
        benchmark_problem.blp
    )


def _benchmark_reference_row(
    benchmark_problem: BenchmarkProblemSpec,
    *,
    optimum_objective: float,
    solution_vector: tuple[int, ...],
    solve_status: str,
    solve_details_json: str,
    duration_s: float,
    solved_at: datetime,
) -> dict[str, object]:
    """Return one benchmark-backed cached reference row."""
    objective_check = float(
        benchmark_problem.blp.objective(solution_vector)
    )
    is_feasible = bool(
        benchmark_problem.blp.is_feasible(solution_vector)
    )
    if not is_feasible:
        raise RuntimeError(
            "persisted CPLEX solution failed the local feasibility check"
        )
    if abs(objective_check - optimum_objective) > 1e-6:
        raise RuntimeError(
            "persisted CPLEX solution failed the local objective check"
        )
    return {
        "family": benchmark_problem.family,
        "instance_name": benchmark_problem.instance_name,
        "source_path": str(
            benchmark_problem.source_path.resolve()
        ),
        "source_path_relative": repo_relative_path(
            benchmark_problem.source_path
        ),
        "problem_size": int(benchmark_problem.size),
        "size": int(benchmark_problem.size),
        "problem_seed": None,
        "instance_source": "benchmark_dataset",
        "objective_sense": "min",
        "num_variables": int(
            benchmark_problem.blp.num_variables
        ),
        "num_equalities": int(
            benchmark_problem.blp.num_equalities
        ),
        "num_inequalities": int(
            benchmark_problem.blp.num_inequalities
        ),
        "optimum_objective": float(optimum_objective),
        "optimum_source": DEFAULT_CPLEX_REFERENCE_SOURCE,
        "solution_vector_json": json.dumps(
            [int(bit) for bit in solution_vector]
        ),
        "variable_names_json": json.dumps(
            list(benchmark_problem.blp.variable_names)
        ),
        "num_selected_variables": int(sum(solution_vector)),
        "solve_status": solve_status,
        "solve_details_json": solve_details_json,
        "objective_check": objective_check,
        "is_feasible": is_feasible,
        "solved_at_utc": solved_at.isoformat(),
        "solve_wall_clock_seconds": duration_s,
    }


def _synthetic_reference_row(
    synthetic_problem: SyntheticProblemInstance,
    *,
    optimum_objective: float,
    solution_vector: tuple[int, ...],
    solve_status: str,
    solve_details_json: str,
    duration_s: float,
    solved_at: datetime,
) -> dict[str, object]:
    """Return one generated-instance cached reference row."""
    return synthetic_reference_row(
        synthetic_problem,
        optimum_objective=optimum_objective,
        solution_vector=solution_vector,
        solve_status=solve_status,
        solve_details_json=solve_details_json,
        duration_s=duration_s,
        solved_at=solved_at,
        optimum_source=DEFAULT_CPLEX_REFERENCE_SOURCE,
    )


def _synthetic_family_sizes_from_args(
    args: argparse.Namespace,
) -> dict[str, list[int]]:
    """Return the selected synthetic size grid keyed by family."""
    selected = {
        str(family).strip().lower()
        for family in args.families
    }
    family_sizes: dict[str, list[int]] = {}
    for family in DEFAULT_FAMILIES:
        if family not in selected:
            family_sizes[family] = []
            continue
        argument_name = f"{family.replace('-', '_')}_sizes"
        family_sizes[family] = list(
            getattr(args, argument_name)
        )
    return family_sizes


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the precompute step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=(
            "CSV that will receive one cached optimum row "
            "per benchmark instance"
        ),
    )
    parser.add_argument(
        "--skipped-csv",
        type=Path,
        default=DEFAULT_SKIPPED_CSV,
        help="CSV that will receive skipped or failed rows",
    )
    parser.add_argument(
        "--mis-dir",
        type=Path,
        default=DEFAULT_MIS_DIR,
        help="MIS benchmark directory",
    )
    parser.add_argument(
        "--mdkp-dir",
        type=Path,
        default=DEFAULT_MDKP_DIR,
        help="MDKP benchmark directory",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(DEFAULT_FAMILIES),
        default=list(DEFAULT_FAMILIES),
        help="Problem families to solve",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic instances instead of loading benchmark files",
    )
    parser.add_argument(
        "--max-instances-per-family",
        type=int,
        default=0,
        help=(
            "Maximum number of files loaded per family; "
            "use 0 to load everything"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed used for synthetic instance generation",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of synthetic instances per family/size block",
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=None,
        help="Optional seed manifest used to define synthetic instances",
    )
    parser.add_argument(
        "--mdkp-sizes",
        type=parse_size_list,
        default=list(
            DEFAULT_SYNTHETIC_FAMILY_SIZES["mdkp"]
        ),
        help="Comma-separated MDKP synthetic sizes",
    )
    parser.add_argument(
        "--mis-sizes",
        type=parse_size_list,
        default=list(DEFAULT_SYNTHETIC_FAMILY_SIZES["mis"]),
        help="Comma-separated MIS synthetic sizes",
    )
    return parser


def main() -> None:
    """Compute and persist the exact benchmark optima."""
    parser = build_argument_parser()
    args = parser.parse_args()
    require_cplex_runtime()

    if args.max_instances_per_family < 0:
        raise ValueError(
            "--max-instances-per-family must be non-negative"
        )
    if args.num_instances <= 0:
        raise ValueError("--num-instances must be positive")

    limit = (
        None
        if int(args.max_instances_per_family) == 0
        else int(args.max_instances_per_family)
    )
    benchmark_directories = {
        "mdkp": _resolve_default_repo_path(
            args.mdkp_dir,
            default_path=DEFAULT_MDKP_DIR,
        ),
        "mis": _resolve_default_repo_path(
            args.mis_dir,
            default_path=DEFAULT_MIS_DIR,
        ),
    }
    output_csv = _resolve_default_repo_path(
        args.output_csv,
        default_path=DEFAULT_OUTPUT_CSV,
    )
    skipped_csv = _resolve_default_repo_path(
        args.skipped_csv,
        default_path=DEFAULT_SKIPPED_CSV,
    )

    optimum_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    if args.synthetic:
        family_sizes = _synthetic_family_sizes_from_args(
            args
        )
        synthetic_batches = load_synthetic_problem_batches(
            base_seed=int(args.seed),
            num_instances=int(args.num_instances),
            instance_manifest=args.instance_manifest,
            family_sizes=family_sizes,
        )
        for family in args.families:
            family_name = str(family).strip().lower()
            sizes = family_sizes[family_name]
            total_instances = sum(
                len(
                    synthetic_batches[
                        (family_name, int(size))
                    ]
                )
                for size in sizes
            )
            print(
                f"Loaded {total_instances} synthetic {family_name} instance(s) "
                f"across sizes {sizes}"
            )
            running_index = 0
            for size in sizes:
                problems = synthetic_batches[
                    (family_name, int(size))
                ]
                for synthetic_problem in problems:
                    running_index += 1
                    print(
                        f"[{family_name} {running_index}/{total_instances}] "
                        f"solving {synthetic_problem.instance_name}"
                    )
                    started_counter = perf_counter()
                    try:
                        (
                            optimum_objective,
                            solution_vector,
                            solve_status,
                            solve_details_json,
                        ) = _solve_cplex_optimum(
                            synthetic_problem
                        )
                        duration_s = (
                            perf_counter() - started_counter
                        )
                        solved_at = datetime.now(
                            timezone.utc
                        )
                        optimum_rows.append(
                            _synthetic_reference_row(
                                synthetic_problem,
                                optimum_objective=float(
                                    optimum_objective
                                ),
                                solution_vector=solution_vector,
                                solve_status=solve_status,
                                solve_details_json=solve_details_json,
                                duration_s=duration_s,
                                solved_at=solved_at,
                            )
                        )
                        print(
                            "  solved "
                            f"objective={optimum_objective:.6f} "
                            f"vars={synthetic_problem.blp.num_variables}"
                        )
                    except Exception as exc:
                        skipped_rows.append(
                            {
                                "family": synthetic_problem.family,
                                "instance_name": synthetic_problem.instance_name,
                                "source_path": None,
                                "problem_size": int(
                                    synthetic_problem.size
                                ),
                                "size": int(
                                    synthetic_problem.size
                                ),
                                "problem_seed": int(
                                    synthetic_problem.problem_seed
                                ),
                                "instance_source": str(
                                    synthetic_problem.blp.metadata.get(
                                        "instance_source",
                                        "generated",
                                    )
                                ),
                                "num_variables": int(
                                    synthetic_problem.blp.num_variables
                                ),
                                "error_type": type(
                                    exc
                                ).__name__,
                                "error_message": str(exc),
                            }
                        )
                        print(
                            "  skipped "
                            f"{synthetic_problem.instance_name}: "
                            f"{type(exc).__name__}: {exc}"
                        )
    else:
        for family in args.families:
            problems, family_skips = (
                load_family_problem_specs(
                    family,
                    directory=benchmark_directories[family],
                    limit=limit,
                )
            )
            skipped_rows.extend(family_skips)
            print(
                f"Loaded {len(problems)} {family} benchmark instance(s)"
            )

            for index, benchmark_problem in enumerate(
                problems, start=1
            ):
                print(
                    f"[{family} {index}/{len(problems)}] "
                    f"solving {benchmark_problem.instance_name}"
                )
                started_counter = perf_counter()
                try:
                    (
                        optimum_objective,
                        solution_vector,
                        solve_status,
                        solve_details_json,
                    ) = _solve_cplex_optimum(
                        benchmark_problem
                    )
                    duration_s = (
                        perf_counter() - started_counter
                    )
                    solved_at = datetime.now(timezone.utc)
                    optimum_rows.append(
                        _benchmark_reference_row(
                            benchmark_problem,
                            optimum_objective=float(
                                optimum_objective
                            ),
                            solution_vector=solution_vector,
                            solve_status=solve_status,
                            solve_details_json=solve_details_json,
                            duration_s=duration_s,
                            solved_at=solved_at,
                        )
                    )
                    print(
                        "  solved "
                        f"objective={optimum_objective:.6f} "
                        f"vars={benchmark_problem.blp.num_variables}"
                    )
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "family": benchmark_problem.family,
                            "instance_name": benchmark_problem.instance_name,
                            "source_path": repo_relative_path(
                                benchmark_problem.source_path
                            ),
                            "problem_size": int(
                                benchmark_problem.size
                            ),
                            "size": int(
                                benchmark_problem.size
                            ),
                            "problem_seed": None,
                            "instance_source": "benchmark_dataset",
                            "num_variables": int(
                                benchmark_problem.blp.num_variables
                            ),
                            "error_type": type(
                                exc
                            ).__name__,
                            "error_message": str(exc),
                        }
                    )
                    print(
                        "  skipped "
                        f"{benchmark_problem.instance_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    existing_optimum_rows = [
        dict(row) for row in _load_csv_rows(output_csv)
    ]
    existing_skipped_rows = [
        dict(row) for row in _load_csv_rows(skipped_csv)
    ]
    merged_optimum_rows = merge_reference_rows(
        existing_optimum_rows, optimum_rows
    )
    merged_skipped_rows = merge_reference_rows(
        existing_skipped_rows, skipped_rows
    )
    _write_csv_rows(output_csv, merged_optimum_rows)
    _write_csv_rows(skipped_csv, merged_skipped_rows)
    print(
        f"Wrote {len(optimum_rows)} new cached optimum row(s) "
        f"to {output_csv} "
        f"(merged total={len(merged_optimum_rows)})"
    )
    print(
        f"Wrote {len(skipped_rows)} new skipped row(s) "
        f"to {skipped_csv} "
        f"(merged total={len(merged_skipped_rows)})"
    )


if __name__ == "__main__":
    main()
