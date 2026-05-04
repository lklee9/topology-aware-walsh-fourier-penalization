"""Create one persisted seed manifest for matched evaluation instances."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

from experiments.experiment_config import (
    DEFAULT_NUM_INSTANCES,
    DEFAULT_SEED,
    FAMILY_ORDER,
)
from experiments.utils.cplex_reference import (
    default_manifest_cplex_reference_path,
    require_cplex_runtime,
    solve_synthetic_problem_references,
)
from experiments.utils.cplex_reference import (
    write_csv_rows as _write_csv_rows,
)
from experiments.utils.driver_common import (
    build_instance_seed_rows,
    build_problem_instances,
    write_rows_csv,
)
from experiments.utils.family_cli import (
    add_family_selection_arguments,
    selected_family_sizes_from_args,
)
from experiments.utils.merge_outputs import (
    ensure_run_metadata,
    merge_csv_rows,
)
from experiments.utils.synthetic_bench import (
    SyntheticProblemInstance,
    synthetic_instance_name,
)

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "shared_eval_seed_manifest.csv"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_family_selection_arguments(
        parser, include_sizes=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path for the generated seed manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed used to derive per-instance problem seeds",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=DEFAULT_NUM_INSTANCES,
        help="Number of instances to emit per family/size block",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file when it already exists",
    )
    return parser


def _manifest_metadata(
    *,
    seed: int,
    num_instances: int,
    cplex_reference_csv: Path,
) -> dict[str, object]:
    """Return the setup metadata enforced for one manifest."""
    return {
        "base_seed": int(seed),
        "num_instances": int(num_instances),
        "cplex_reference_csv": str(
            cplex_reference_csv.resolve()
        ),
    }


def _group_manifest_rows(
    rows: list[dict[str, object]],
) -> dict[tuple[str, int], list[int]]:
    """Return ordered seeds grouped by family/size from manifest rows."""
    grouped: dict[
        tuple[str, int], list[tuple[int, int]]
    ] = {}
    for row in rows:
        family = str(row["family"]).strip().lower()
        size = int(row["size"])
        instance_index = int(row["instance_index"])
        problem_seed = int(row["problem_seed"])
        grouped.setdefault((family, size), []).append(
            (instance_index, problem_seed)
        )
    ordered: dict[tuple[str, int], list[int]] = {}
    for key, indexed_rows in grouped.items():
        indexed_rows.sort()
        ordered[key] = [
            problem_seed for _, problem_seed in indexed_rows
        ]
    return ordered


def _synthetic_problems_from_manifest_rows(
    *,
    rows: list[dict[str, object]],
    manifest_path: Path,
    base_seed: int,
) -> list[SyntheticProblemInstance]:
    """Materialize the synthetic problems defined by manifest rows."""
    grouped_rows = _group_manifest_rows(rows)
    problems: list[SyntheticProblemInstance] = []
    for (family, size), problem_seeds in sorted(
        grouped_rows.items()
    ):
        block = build_problem_instances(
            family,
            size,
            base_seed=int(base_seed),
            num_instances=len(problem_seeds),
            instance_seeds=list(problem_seeds),
            instance_manifest=manifest_path,
        )
        for instance_index, (
            problem_seed,
            problem,
        ) in enumerate(
            zip(problem_seeds, block, strict=True)
        ):
            problems.append(
                SyntheticProblemInstance(
                    family=family,
                    size=int(size),
                    instance_index=int(instance_index),
                    problem_seed=int(problem_seed),
                    instance_name=synthetic_instance_name(
                        family,
                        int(size),
                        int(problem_seed),
                    ),
                    blp=problem,
                )
            )
    return problems


def main() -> None:
    """Write the requested seed manifest to disk."""
    parser = build_argument_parser()
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    cplex_reference_path = (
        default_manifest_cplex_reference_path(output_path)
    )
    family_sizes = selected_family_sizes_from_args(args)

    # By default, include size 20 for MDKP and MIS when the user did not
    # explicitly provide --mdkp-sizes or --mis-sizes on the CLI. This keeps
    # existing default sizes while adding the commonly-needed size 20.
    #
    # We import sys locally so this behavior doesn't rely on the module-level
    # conditional sys import above.
    import sys as _sys

    if "mdkp" in family_sizes and family_sizes["mdkp"]:
        if "--mdkp-sizes" not in _sys.argv and 20 not in family_sizes["mdkp"]:
            family_sizes["mdkp"] = sorted(set(family_sizes["mdkp"] + [20]))
    if "mis" in family_sizes and family_sizes["mis"]:
        if "--mis-sizes" not in _sys.argv and 20 not in family_sizes["mis"]:
            family_sizes["mis"] = sorted(set(family_sizes["mis"] + [20]))

    metadata = _manifest_metadata(
        seed=int(args.seed),
        num_instances=int(args.num_instances),
        cplex_reference_csv=cplex_reference_path,
    )

    rows = build_instance_seed_rows(
        base_seed=int(args.seed),
        num_instances=int(args.num_instances),
        family_sizes=family_sizes,
    )
    manifest_rows = rows
    if not args.force:
        manifest_rows = merge_csv_rows(
            output_path,
            rows,
            block_fields=("family", "size"),
        )

    # Keep only known problem families (defined in FAMILY_ORDER).
    # This avoids referencing legacy or external families such as "mis-hub".
    allowed_families = {str(f).strip().lower() for f in FAMILY_ORDER}
    filtered_rows: list[dict[str, object]] = []
    removed_families: set[str] = set()
    for row in manifest_rows:
        fam = str(row.get("family", "")).strip().lower()
        if fam in allowed_families:
            filtered_rows.append(row)
        else:
            removed_families.add(fam)
    if removed_families:
        print(
            f"[create_instance_seed_manifest] warning: removing rows for unknown families: {', '.join(sorted(removed_families))}"
        )
        manifest_rows = filtered_rows

    synthetic_problems = _synthetic_problems_from_manifest_rows(
        rows=manifest_rows,
        manifest_path=output_path,
        base_seed=int(args.seed),
    )
    require_cplex_runtime()
    cplex_rows = solve_synthetic_problem_references(
        synthetic_problems
    )

    ensure_run_metadata(
        output_path,
        metadata,
        force=bool(args.force),
    )
    write_rows_csv(output_path, manifest_rows)
    _write_csv_rows(cplex_reference_path, cplex_rows)

    print(
        f"[create_instance_seed_manifest] wrote {output_path}"
    )
    print(
        f"[create_instance_seed_manifest] wrote {cplex_reference_path}"
    )


if __name__ == "__main__":
    main()
