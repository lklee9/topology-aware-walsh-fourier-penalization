"""Shared benchmark conversions and persisted optimum helpers."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from fourier_projection.blp import BLP
except (
    ImportError
):  # pragma: no cover - direct script usage.
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fourier_projection.blp import BLP

try:
    from .data_loaders import (
        MDKPInstance,
        MISInstance,
        load_mdkp_directory,
        load_mis_directory,
    )
    from .problems import (
        _mis_graph_to_blp,
    )
except (
    ImportError
):  # pragma: no cover - direct script usage.
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from data_loaders import MISInstance  # type: ignore
    from data_loaders import (  # type: ignore
        MDKPInstance,
        load_mdkp_directory,
        load_mis_directory,
    )
    from problems import _mis_graph_to_blp  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkProblemSpec:
    """One benchmark instance converted into the repo's BLP form."""

    family: str
    instance_name: str
    source_path: Path
    size: int
    blp: BLP


@dataclass(frozen=True)
class KnownOptimumRecord:
    """One persisted optimum row loaded from CSV."""

    family: str
    instance_name: str
    source_path: str | None
    problem_size: int | None
    num_variables: int | None
    optimum_objective: float
    optimum_source: str
    solution_vector: tuple[int, ...] | None
    variable_names: tuple[str, ...] | None


def repo_relative_path(path: str | Path) -> str:
    """Return a repo-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _deduplicate_non_test_instances(
    instances: list[Any],
) -> list[Any]:
    """Drop duplicated test copies while keeping one file per name."""
    unique: dict[str, Any] = {}
    for instance in instances:
        if "test" in instance.path.parts:
            continue
        unique.setdefault(instance.path.name, instance)
    return [unique[name] for name in sorted(unique)]


def mis_instance_to_blp(instance: MISInstance) -> BLP:
    """Convert one MIS graph into the repo-convention BLP."""
    return _mis_graph_to_blp(
        instance.graph,
        family="mis",
        name=instance.name,
        metadata={
            "instance_name": instance.name,
            "source_family": "mis",
            "source_path": str(instance.path),
        },
    )


def _mis_instance_to_spec(
    instance: MISInstance,
    *,
    family: str,
) -> BenchmarkProblemSpec:
    """Wrap one MIS or MIS-hub loader result as a benchmark problem."""
    if family == "mis":
        instance_name = instance.name
        blp = mis_instance_to_blp(instance)
    else:
        raise ValueError(
            f"unknown MIS-derived family: {family}"
        )
    return BenchmarkProblemSpec(
        family=family,
        instance_name=instance_name,
        source_path=instance.path,
        size=int(instance.n),
        blp=blp,
    )


def _mdkp_instance_to_spec(
    instance: MDKPInstance,
) -> BenchmarkProblemSpec:
    """Wrap one MDKP loader result as a benchmark problem."""
    return BenchmarkProblemSpec(
        family="mdkp",
        instance_name=instance.name,
        source_path=instance.path,
        size=int(instance.n),
        blp=instance.blp,
    )


def load_family_problem_specs(
    family: str,
    *,
    directory: Path,
    limit: int | None = None,
) -> tuple[
    list[BenchmarkProblemSpec], list[dict[str, object]]
]:
    """Load and convert one benchmark family from disk."""
    skipped_rows: list[dict[str, object]] = []

    if family == "mis":
        instances = _deduplicate_non_test_instances(
            load_mis_directory(directory)
        )
        if limit is not None:
            instances = instances[: int(limit)]
        problems = [
            _mis_instance_to_spec(instance, family=family)
            for instance in instances
        ]
        return problems, skipped_rows

    if family == "mdkp":
        instances = _deduplicate_non_test_instances(
            load_mdkp_directory(directory)
        )
        if limit is not None:
            instances = instances[: int(limit)]
        return (
            [
                _mdkp_instance_to_spec(instance)
                for instance in instances
            ],
            skipped_rows,
        )

    raise ValueError(f"unknown family: {family}")


def _load_optional_json_list(
    raw_value: str | None,
) -> tuple[Any, ...] | None:
    """Parse one optional JSON array field from CSV."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("expected one JSON array field")
    return tuple(values)


def load_known_optima_csv(
    path: str | Path,
) -> dict[tuple[str, str], KnownOptimumRecord]:
    """Load the persisted optimum cache keyed by family/name."""
    csv_path = Path(path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"known optima CSV not found: {csv_path}"
        )

    records: dict[tuple[str, str], KnownOptimumRecord] = {}
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            family = str(row["family"]).strip().lower()
            instance_name = str(
                row["instance_name"]
            ).strip()
            objective_value = row.get("optimum_objective")
            if objective_value is None:
                objective_value = row.get(
                    "known_optimum_objective"
                )
            source_value = row.get("optimum_source")
            if source_value is None:
                source_value = row.get(
                    "known_optimum_source"
                )
            if (
                objective_value is None
                or source_value is None
            ):
                raise ValueError(
                    "known optima CSV is missing required columns"
                )

            raw_solution = _load_optional_json_list(
                row.get("solution_vector_json")
            )
            raw_names = _load_optional_json_list(
                row.get("variable_names_json")
            )
            solution_vector = (
                None
                if raw_solution is None
                else tuple(
                    int(value) for value in raw_solution
                )
            )
            variable_names = (
                None
                if raw_names is None
                else tuple(
                    str(value) for value in raw_names
                )
            )
            key = (family, instance_name)
            if key in records:
                raise ValueError(
                    "duplicate known-optimum row for "
                    f"{family}/{instance_name}"
                )

            records[key] = KnownOptimumRecord(
                family=family,
                instance_name=instance_name,
                source_path=row.get("source_path_relative")
                or row.get("source_path"),
                problem_size=(
                    None
                    if not row.get("problem_size")
                    else int(row["problem_size"])
                ),
                num_variables=(
                    None
                    if not row.get("num_variables")
                    else int(row["num_variables"])
                ),
                optimum_objective=float(objective_value),
                optimum_source=str(source_value),
                solution_vector=solution_vector,
                variable_names=variable_names,
            )
    return records


def known_optimum_for_problem(
    problem: BenchmarkProblemSpec,
    known_optima: Mapping[
        tuple[str, str],
        KnownOptimumRecord,
    ],
) -> KnownOptimumRecord | None:
    """Return the cached optimum row for one problem."""
    record = known_optima.get(
        (problem.family, problem.instance_name)
    )
    if record is None:
        return None
    if record.problem_size is not None and int(
        record.problem_size
    ) != int(problem.size):
        raise ValueError(
            "cached optimum size does not match benchmark problem: "
            f"{problem.family}/{problem.instance_name}"
        )
    if record.num_variables is not None and int(
        record.num_variables
    ) != int(problem.blp.num_variables):
        raise ValueError(
            "cached optimum variable count does not match "
            f"{problem.family}/{problem.instance_name}"
        )
    if (
        record.variable_names is not None
        and tuple(problem.blp.variable_names)
        != record.variable_names
    ):
        raise ValueError(
            "cached optimum variable names do not match "
            f"{problem.family}/{problem.instance_name}"
        )
    return record


__all__ = [
    "BenchmarkProblemSpec",
    "KnownOptimumRecord",
    "known_optimum_for_problem",
    "load_family_problem_specs",
    "load_known_optima_csv",
    "mis_instance_to_blp",
    "repo_relative_path",
]
