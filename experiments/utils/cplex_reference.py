"""Shared CPLEX reference loading, validation, and solving helpers."""

from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from experiments.utils.driver_common import write_rows_csv
from experiments.utils.problems import blp_to_docplex_model
from experiments.utils.synthetic_bench import (
    SyntheticProblemInstance,
    synthetic_reference_key,
)
from fourier_projection.blp import BLP

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CPLEX_REFERENCE_CSV = Path(
    "data/classical_baselines/cplex_optima.csv"
)
DEFAULT_CPLEX_REFERENCE_SOURCE = "cplex_cache"
DEFAULT_OBJECTIVE_SENSE = "min"
DEFAULT_REFERENCE_ATOL = 1e-9


@dataclass(frozen=True)
class CplexReference:
    """One cached exact constrained optimum loaded from CSV."""

    family: str
    instance_name: str
    problem_size: int
    problem_seed: int | None
    source_path: str | None
    num_variables: int | None
    num_equalities: int | None
    num_inequalities: int | None
    optimum_objective: float
    optimum_source: str
    solution_vector: tuple[int, ...] | None
    variable_names: tuple[str, ...] | None
    objective_sense: str = DEFAULT_OBJECTIVE_SENSE


@dataclass(frozen=True)
class CplexReferenceIndex:
    """Loaded CPLEX references indexed by benchmark and synthetic keys."""

    path: Path
    benchmark: dict[tuple[str, str], CplexReference]
    synthetic: dict[tuple[str, int, int], CplexReference]


def repo_relative_path(path: str | Path) -> str:
    """Return a repo-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def default_manifest_cplex_reference_path(
    manifest_path: str | Path,
) -> Path:
    """Return the default manifest-local CPLEX sidecar path."""
    path = Path(manifest_path).resolve()
    if path.suffix:
        return path.with_name(
            f"{path.stem}_cplex_optima.csv"
        )
    return path.with_name(f"{path.name}_cplex_optima.csv")


def resolve_cplex_reference_path(
    requested_path: str | Path | None,
    *,
    instance_manifest: str | Path | None = None,
) -> Path:
    """Resolve the reference CSV path with manifest-sidecar fallback."""
    if requested_path is not None:
        return Path(requested_path).resolve()
    if instance_manifest is not None:
        return default_manifest_cplex_reference_path(
            instance_manifest
        )
    return (
        REPO_ROOT / DEFAULT_CPLEX_REFERENCE_CSV
    ).resolve()


def _load_optional_json_list(
    raw_value: object,
) -> tuple[Any, ...] | None:
    """Parse one optional JSON array field."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("expected one JSON array field")
    return tuple(values)


def _parse_optional_int(value: object) -> int | None:
    """Parse one optional integer from a row field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _parse_optional_float(value: object) -> float | None:
    """Parse one optional float from a row field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into row dictionaries when it exists."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_cplex_reference_index(
    path: str | Path,
) -> CplexReferenceIndex:
    """Load one reference CSV into benchmark and synthetic key maps."""
    csv_path = Path(path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CPLEX reference CSV not found: {csv_path}"
        )

    benchmark: dict[tuple[str, str], CplexReference] = {}
    synthetic: dict[
        tuple[str, int, int], CplexReference
    ] = {}
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            family = (
                str(row.get("family", "")).strip().lower()
            )
            instance_name = str(
                row.get("instance_name", "")
            ).strip()
            if not family or not instance_name:
                raise ValueError(
                    f"invalid CPLEX reference row in {csv_path}: "
                    "family and instance_name are required"
                )
            problem_size = _parse_optional_int(
                row.get("size")
            )
            if problem_size is None:
                problem_size = _parse_optional_int(
                    row.get("problem_size")
                )
            if problem_size is None:
                raise ValueError(
                    f"CPLEX reference row is missing size/problem_size: "
                    f"{family}/{instance_name}"
                )
            optimum_objective = _parse_optional_float(
                row.get("optimum_objective")
            )
            if optimum_objective is None:
                raise ValueError(
                    f"CPLEX reference row is missing optimum_objective: "
                    f"{family}/{instance_name}"
                )
            source_value = str(
                row.get(
                    "optimum_source",
                    DEFAULT_CPLEX_REFERENCE_SOURCE,
                )
            ).strip()
            objective_sense = str(
                row.get(
                    "objective_sense",
                    DEFAULT_OBJECTIVE_SENSE,
                )
            ).strip()
            solution_vector_raw = _load_optional_json_list(
                row.get("solution_vector_json")
            )
            variable_names_raw = _load_optional_json_list(
                row.get("variable_names_json")
            )
            reference = CplexReference(
                family=family,
                instance_name=instance_name,
                problem_size=int(problem_size),
                problem_seed=_parse_optional_int(
                    row.get("problem_seed")
                ),
                source_path=(
                    str(
                        row.get("source_path_relative")
                        or row.get("source_path")
                    )
                    if row.get("source_path_relative")
                    or row.get("source_path")
                    else None
                ),
                num_variables=_parse_optional_int(
                    row.get("num_variables")
                ),
                num_equalities=_parse_optional_int(
                    row.get("num_equalities")
                ),
                num_inequalities=_parse_optional_int(
                    row.get("num_inequalities")
                ),
                optimum_objective=float(optimum_objective),
                optimum_source=source_value
                or DEFAULT_CPLEX_REFERENCE_SOURCE,
                solution_vector=(
                    None
                    if solution_vector_raw is None
                    else tuple(
                        int(value)
                        for value in solution_vector_raw
                    )
                ),
                variable_names=(
                    None
                    if variable_names_raw is None
                    else tuple(
                        str(value)
                        for value in variable_names_raw
                    )
                ),
                objective_sense=objective_sense
                or DEFAULT_OBJECTIVE_SENSE,
            )
            benchmark_key = (family, instance_name)
            if benchmark_key in benchmark:
                raise ValueError(
                    "duplicate benchmark CPLEX reference row for "
                    f"{family}/{instance_name}"
                )
            benchmark[benchmark_key] = reference
            if reference.problem_seed is not None:
                synthetic_key = synthetic_reference_key(
                    family=family,
                    size=int(problem_size),
                    problem_seed=int(
                        reference.problem_seed
                    ),
                )
                if synthetic_key in synthetic:
                    raise ValueError(
                        "duplicate synthetic CPLEX reference row for "
                        f"{family}/{problem_size}/{reference.problem_seed}"
                    )
                synthetic[synthetic_key] = reference
    return CplexReferenceIndex(
        path=csv_path,
        benchmark=benchmark,
        synthetic=synthetic,
    )


def _validate_reference_against_problem(
    reference: CplexReference,
    problem: BLP,
    *,
    expected_size: int,
    expected_problem_seed: int | None = None,
    atol: float = DEFAULT_REFERENCE_ATOL,
) -> CplexReference:
    """Validate one matched reference row against one concrete problem."""
    if int(reference.problem_size) != int(expected_size):
        raise ValueError(
            "CPLEX reference size does not match problem: "
            f"{reference.family}/{reference.instance_name}"
        )
    if (
        expected_problem_seed is not None
        and reference.problem_seed is not None
    ):
        if int(reference.problem_seed) != int(
            expected_problem_seed
        ):
            raise ValueError(
                "CPLEX reference problem_seed does not match problem: "
                f"{reference.family}/{reference.instance_name}"
            )
    if reference.num_variables is not None and int(
        reference.num_variables
    ) != int(problem.num_variables):
        raise ValueError(
            "CPLEX reference variable count does not match problem: "
            f"{reference.family}/{reference.instance_name}"
        )
    if reference.variable_names is not None and tuple(
        problem.variable_names
    ) != tuple(reference.variable_names):
        raise ValueError(
            "CPLEX reference variable names do not match problem: "
            f"{reference.family}/{reference.instance_name}"
        )
    if reference.solution_vector is not None:
        solution = np.asarray(
            reference.solution_vector, dtype=float
        )
        if solution.shape[0] != int(problem.num_variables):
            raise ValueError(
                "CPLEX reference solution length does not match problem: "
                f"{reference.family}/{reference.instance_name}"
            )
        if not np.all(np.isin(solution, (0.0, 1.0))):
            raise ValueError(
                "CPLEX reference solution vector is not binary: "
                f"{reference.family}/{reference.instance_name}"
            )
        if not bool(
            problem.is_feasible(solution, atol=atol)
        ):
            raise ValueError(
                "CPLEX reference solution is infeasible for the problem: "
                f"{reference.family}/{reference.instance_name}"
            )
        objective_value = float(problem.objective(solution))
        if not np.isclose(
            objective_value,
            float(reference.optimum_objective),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                "CPLEX reference objective does not match the cached solution: "
                f"{reference.family}/{reference.instance_name}"
            )
    return reference


def cplex_reference_for_synthetic_problem(
    *,
    index: CplexReferenceIndex,
    family: str,
    size: int,
    problem_seed: int,
    problem: BLP,
    atol: float = DEFAULT_REFERENCE_ATOL,
) -> CplexReference:
    """Return the validated reference row for one synthetic problem."""
    key = synthetic_reference_key(
        family=str(family).strip().lower(),
        size=int(size),
        problem_seed=int(problem_seed),
    )
    try:
        reference = index.synthetic[key]
    except KeyError as exc:
        raise KeyError(
            "missing synthetic CPLEX reference for "
            f"family={family}, size={size}, problem_seed={problem_seed} "
            f"in {index.path}"
        ) from exc
    return _validate_reference_against_problem(
        reference,
        problem,
        expected_size=int(size),
        expected_problem_seed=int(problem_seed),
        atol=atol,
    )


def reference_objective_hit_mask(
    objective_values: np.ndarray,
    feasible_mask: np.ndarray,
    *,
    optimum_objective: float,
    objective_sense: str = DEFAULT_OBJECTIVE_SENSE,
    atol: float = DEFAULT_REFERENCE_ATOL,
) -> np.ndarray:
    """Return whether each sample matches or beats the reference optimum."""
    objective = np.asarray(objective_values, dtype=float)
    feasible = np.asarray(feasible_mask, dtype=bool)
    sense = str(objective_sense).strip().lower()
    if sense == "min":
        return feasible & (
            objective
            <= float(optimum_objective) + float(atol)
        )
    if sense == "max":
        return feasible & (
            objective
            >= float(optimum_objective) - float(atol)
        )
    raise ValueError(
        f"unsupported objective sense: {objective_sense}"
    )


def reference_objective_gap_ratio(
    found_objective: np.ndarray | float | None,
    optimum_objective: float,
    *,
    objective_sense: str = DEFAULT_OBJECTIVE_SENSE,
) -> np.ndarray | float:
    """Return the normalized gap to the reference optimum."""
    if found_objective is None:
        return float("nan")
    denom = max(1.0, abs(float(optimum_objective)))
    values = np.asarray(found_objective, dtype=float)
    sense = str(objective_sense).strip().lower()
    if sense == "min":
        result = (values - float(optimum_objective)) / denom
    elif sense == "max":
        result = (float(optimum_objective) - values) / denom
    else:
        raise ValueError(
            f"unsupported objective sense: {objective_sense}"
        )
    if result.ndim == 0:
        return float(result)
    return result


def require_cplex_runtime() -> None:
    """Fail fast when the local CPLEX runtime is unavailable."""
    has_cplex = (
        importlib.util.find_spec("cplex") is not None
    )
    has_docplex = (
        importlib.util.find_spec("docplex") is not None
    )
    if has_cplex and has_docplex:
        return
    raise SystemExit(
        "CPLEX runtime not found. Install both the IBM CPLEX Python package "
        "and DOcplex so models can be solved locally, then rerun the command."
    )


def _solve_details_json(model: Any) -> str:
    """Return a compact JSON encoding of useful solve details."""
    details = getattr(model, "solve_details", None)
    if details is None:
        return "{}"

    payload: dict[str, object] = {}
    for field in (
        "status",
        "status_code",
        "problem_type",
        "columns",
        "nb_iterations",
        "time",
        "gap",
        "best_bound",
    ):
        value = getattr(details, field, None)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[field] = value
        else:
            payload[field] = str(value)
    return json.dumps(payload, sort_keys=True)


def solve_problem_cplex_optimum(
    problem: BLP,
) -> tuple[float, tuple[int, ...], str, str]:
    """Solve one repo-convention BLP exactly with CPLEX."""
    from docplex.mp.utils import DOcplexException

    model = blp_to_docplex_model(
        problem,
        name=f"{problem.name}_cplex_optimum",
    )
    try:
        solution = model.solve(log_output=False)
    except DOcplexException as exc:
        raise RuntimeError(str(exc)) from exc
    if solution is None:
        raise RuntimeError("CPLEX returned no solution")

    bits: list[int] = []
    for variable_name in problem.variable_names:
        variable = model.get_var_by_name(variable_name)
        if variable is None:
            raise RuntimeError(
                f"missing DOcplex variable in solved model: {variable_name}"
            )
        value = float(solution.get_value(variable))
        rounded = int(round(value))
        if (
            rounded not in {0, 1}
            or abs(value - rounded) > 1e-6
        ):
            raise RuntimeError(
                "CPLEX returned a non-binary solution value for "
                f"{variable_name}: {value}"
            )
        bits.append(rounded)

    solve_details = getattr(model, "solve_details", None)
    status = str(
        getattr(solve_details, "status", "unknown")
    )
    return (
        float(solution.objective_value),
        tuple(bits),
        status,
        _solve_details_json(model),
    )


def synthetic_reference_row(
    synthetic_problem: SyntheticProblemInstance,
    *,
    optimum_objective: float,
    solution_vector: tuple[int, ...],
    solve_status: str,
    solve_details_json: str,
    duration_s: float,
    solved_at: datetime,
    optimum_source: str = DEFAULT_CPLEX_REFERENCE_SOURCE,
) -> dict[str, object]:
    """Return one generated-instance cached reference row."""
    objective_check = float(
        synthetic_problem.blp.objective(solution_vector)
    )
    is_feasible = bool(
        synthetic_problem.blp.is_feasible(solution_vector)
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
        "family": synthetic_problem.family,
        "instance_name": synthetic_problem.instance_name,
        "source_path": None,
        "source_path_relative": None,
        "problem_size": int(synthetic_problem.size),
        "size": int(synthetic_problem.size),
        "problem_seed": int(synthetic_problem.problem_seed),
        "instance_source": str(
            synthetic_problem.blp.metadata.get(
                "instance_source", "generated"
            )
        ),
        "objective_sense": DEFAULT_OBJECTIVE_SENSE,
        "num_variables": int(
            synthetic_problem.blp.num_variables
        ),
        "num_equalities": int(
            synthetic_problem.blp.num_equalities
        ),
        "num_inequalities": int(
            synthetic_problem.blp.num_inequalities
        ),
        "optimum_objective": float(optimum_objective),
        "optimum_source": str(optimum_source),
        "solution_vector_json": json.dumps(
            [int(bit) for bit in solution_vector]
        ),
        "variable_names_json": json.dumps(
            list(synthetic_problem.blp.variable_names)
        ),
        "num_selected_variables": int(sum(solution_vector)),
        "solve_status": solve_status,
        "solve_details_json": solve_details_json,
        "objective_check": objective_check,
        "is_feasible": is_feasible,
        "solved_at_utc": solved_at.isoformat(),
        "solve_wall_clock_seconds": float(duration_s),
    }


def output_row_key(
    row: Mapping[str, object],
) -> tuple[object, ...]:
    """Return the mixed benchmark/synthetic merge key for one row."""
    family = str(row.get("family", "")).strip().lower()
    problem_seed = _parse_optional_int(
        row.get("problem_seed")
    )
    if problem_seed is not None:
        size = _parse_optional_int(row.get("size"))
        if size is None:
            size = _parse_optional_int(
                row.get("problem_size")
            )
        if size is None:
            raise ValueError(
                "synthetic reference row is missing size"
            )
        return ("synthetic",) + synthetic_reference_key(
            family=family,
            size=size,
            problem_seed=problem_seed,
        )
    return (
        "benchmark",
        family,
        str(row.get("instance_name", "")).strip(),
    )


def merge_reference_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge mixed benchmark/synthetic rows without dropping old entries."""
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    order: list[tuple[object, ...]] = []
    for row in existing_rows:
        key = output_row_key(row)
        if key in merged:
            continue
        merged[key] = dict(row)
        order.append(key)
    for row in new_rows:
        key = output_row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = dict(row)
    return [merged[key] for key in order]


def write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows to CSV, creating an empty file when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    write_rows_csv(path, rows)


def solve_synthetic_problem_references(
    problems: list[SyntheticProblemInstance],
    *,
    optimum_source: str = DEFAULT_CPLEX_REFERENCE_SOURCE,
) -> list[dict[str, object]]:
    """Solve one synthetic batch and return serialized CPLEX rows."""
    require_cplex_runtime()
    rows: list[dict[str, object]] = []
    for synthetic_problem in problems:
        started_counter = perf_counter()
        (
            optimum_objective,
            solution_vector,
            solve_status,
            solve_details_json,
        ) = solve_problem_cplex_optimum(
            synthetic_problem.blp
        )
        rows.append(
            synthetic_reference_row(
                synthetic_problem,
                optimum_objective=float(optimum_objective),
                solution_vector=solution_vector,
                solve_status=solve_status,
                solve_details_json=solve_details_json,
                duration_s=(
                    perf_counter() - started_counter
                ),
                solved_at=datetime.now(timezone.utc),
                optimum_source=optimum_source,
            )
        )
    return rows


__all__ = [
    "CplexReference",
    "CplexReferenceIndex",
    "DEFAULT_CPLEX_REFERENCE_CSV",
    "DEFAULT_CPLEX_REFERENCE_SOURCE",
    "DEFAULT_OBJECTIVE_SENSE",
    "DEFAULT_REFERENCE_ATOL",
    "cplex_reference_for_synthetic_problem",
    "default_manifest_cplex_reference_path",
    "load_cplex_reference_index",
    "merge_reference_rows",
    "output_row_key",
    "reference_objective_gap_ratio",
    "reference_objective_hit_mask",
    "repo_relative_path",
    "require_cplex_runtime",
    "resolve_cplex_reference_path",
    "solve_problem_cplex_optimum",
    "solve_synthetic_problem_references",
    "synthetic_reference_row",
    "write_csv_rows",
]
