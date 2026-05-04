"""Analyse raw samples produced by ``experiments.dwave_bench``.

This script reads the raw per-run ``samples.csv`` and ``metadata.csv``
files produced by the D-Wave benchmark runner, recomputes all sample-
level and run-level metrics, and joins any known classical optima stored
in ``data/classical_baselines/cplex_optima.csv``.

For each successful run it writes:

* ``sample_metrics.csv`` with one row per sampled logical solution; and
* ``run_metrics.csv`` with one summary row for the run.

Each analysed session directory also receives:

* ``run_metrics_catalog.csv`` with one row per analysed run;
* ``instance_summary.csv`` with one row per benchmark instance and
  method, averaged across repeated runs;
* ``aggregate_summary.csv`` with robust central summaries across
  analysed instances for the main embedding-style gap/feasibility
  metrics, excluding spread columns;
* ``analysis_skipped.csv`` for runs that could not be analysed; and
* ``analysis_manifest.json`` describing the analysis inputs and outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from experiments.utils.benchmark_data import (
    KnownOptimumRecord,
    known_optimum_for_problem,
    load_family_problem_specs,
    load_known_optima_csv,
    repo_relative_path,
)
from experiments.utils.driver_common import (
    objective_gap_ratio as _objective_gap_ratio,
)
from experiments.utils.driver_common import (
    write_rows_csv as _write_rows_csv,
)
from fourier_projection.blp import BLP

DEFAULT_INPUT_DIR = Path("experiments/results/dwave_bench")
DEFAULT_KNOWN_OPTIMA_CSV = Path(
    "data/classical_baselines/cplex_optima.csv"
)
DEFAULT_MIS_DIR = Path("data/mis_instances")
DEFAULT_MDKP_DIR = Path("data/mdkp_instances")
DEFAULT_FAMILIES = ("mdkp", "mis")
KNOWN_OPTIMUM_TOL = 1e-9
PROJECTED_TOPOLOGY_METHOD = "projected_topology"
ZERO_ATOL = 1e-12


def _configure_csv_field_limit() -> None:
    """Raise the CSV parser field limit for large QPU metadata blobs."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_configure_csv_field_limit()


@dataclass(frozen=True)
class BenchmarkProblem:
    """One benchmark instance converted into the repo's BLP form."""

    family: str
    instance_name: str
    source_path: Path
    size: int
    blp: BLP
    known_optimum_objective: float | None
    known_optimum_source: str | None


def _resolve_default_repo_path(
    requested_path: Path,
    *,
    default_path: Path,
) -> Path:
    """Resolve default CLI paths relative to the repository root."""
    if requested_path == default_path:
        return (REPO_ROOT / default_path).resolve()
    return requested_path.resolve()


def _resolve_optional_repo_path(
    requested_path: Path,
    *,
    default_path: Path,
) -> Path | None:
    """Resolve an optional CSV path, allowing a missing default."""
    if requested_path == default_path:
        resolved = (REPO_ROOT / default_path).resolve()
        if resolved.exists():
            return resolved
        return None
    resolved = requested_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"CSV not found: {resolved}"
        )
    return resolved


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows to CSV, creating an empty file when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    _write_rows_csv(path, rows)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into a list of row dicts."""
    with path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_optional_float(
    value: str | None,
) -> float | None:
    """Return one optional float parsed from CSV text."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _safe_float(value: object) -> float | None:
    """Return one finite float parsed from text-like input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _safe_int(value: object) -> int | None:
    """Return one integer parsed from text-like input."""
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _compare_embedding_method_name(
    requested_method: object,
) -> str:
    """Return the compare-embedding method label for one run."""
    method = str(requested_method).strip().lower()
    if method == "projected_qpu":
        return PROJECTED_TOPOLOGY_METHOD
    return method


def _mean_metric(
    items: list[dict[str, object]],
    *aliases: str,
) -> float | None:
    """Return the mean of the first populated numeric alias per row."""
    values: list[float] = []
    for item in items:
        value: float | None = None
        for alias in aliases:
            value = _safe_float(item.get(alias))
            if value is not None:
                break
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _first_metric_value(
    item: Mapping[str, object],
    aliases: tuple[str, ...],
) -> float:
    """Return the first finite numeric alias from one row."""
    for alias in aliases:
        value = _safe_float(item.get(alias))
        if value is not None:
            return float(value)
    return math.nan


def _metric_values(
    items: list[dict[str, object]],
    aliases: tuple[str, ...],
) -> np.ndarray:
    """Return the finite values for one metric across grouped rows."""
    values = np.asarray(
        [
            _first_metric_value(item, aliases)
            for item in items
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _hardware_size_from_shape_json(
    value: object,
) -> int | None:
    """Infer the D-Wave topology size parameter from a shape JSON field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return int(raw[0])
    except (TypeError, ValueError):
        return None


def _instance_summary_rows(
    run_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate analysed runs into per-instance embedding summary rows."""
    grouped: dict[
        tuple[str, int, str, int, str, str, str],
        list[dict[str, object]],
    ] = {}
    for row in run_rows:
        hardware_size = _safe_int(row.get("hardware_size"))
        if hardware_size is None:
            hardware_size = _hardware_size_from_shape_json(
                row.get("solver_topology_shape_json")
            )
        size = _safe_int(row.get("size"))
        if size is None:
            size = _safe_int(row.get("problem_size"))
        if hardware_size is None or size is None:
            raise ValueError(
                "missing hardware_size or size in analysed run row"
            )
        requested_method = (
            str(row.get("requested_method", ""))
            .strip()
            .lower()
        )
        resolved_method = (
            str(row.get("resolved_method", ""))
            .strip()
            .lower()
        )
        key = (
            str(row["hardware_family"]),
            hardware_size,
            str(row["family"]),
            size,
            str(row["instance_name"]),
            _compare_embedding_method_name(
                requested_method or resolved_method
            ),
            resolved_method,
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key in sorted(grouped):
        (
            hardware_family,
            hardware_size,
            family,
            size,
            instance_name,
            method,
            resolved_method,
        ) = key
        items = grouped[key]
        first = items[0]
        num_variables = _safe_int(
            first.get("num_variables")
        )
        if num_variables is None:
            raise ValueError(
                "missing num_variables in analysed run row"
            )
        row: dict[str, object] = {
            "session_id": first.get("session_id"),
            "device_name": first.get("device_name"),
            "device_slug": first.get("device_slug"),
            "hardware_family": hardware_family,
            "hardware_size": hardware_size,
            "family": family,
            "size": size,
            "instance_name": instance_name,
            "instance_index": _safe_int(
                first.get("instance_index")
            ),
            "method": method,
            "requested_method": first.get(
                "requested_method"
            ),
            "resolved_method": resolved_method,
            "num_runs": len(items),
            "num_variables": num_variables,
            "true_optimum_objective": _safe_float(
                first.get("known_optimum_objective")
            ),
            "known_optimum_objective": _safe_float(
                first.get("known_optimum_objective")
            ),
            "known_optimum_source": first.get(
                "known_optimum_source"
            ),
            "sqa_gap": _mean_metric(
                items, "sqa_gap", "objective_gap"
            ),
            "objective_gap": _mean_metric(
                items,
                "objective_gap",
                "sqa_gap",
            ),
            "mean_sqa_energy_gap": _mean_metric(
                items,
                "mean_sqa_energy_gap",
                "mean_feasible_objective_gap",
            ),
            "optimum_probability": _mean_metric(
                items,
                "optimum_probability",
                "known_optimum_read_fraction",
                "sqa_optimum_rate",
            ),
            "sqa_optimum_rate": _mean_metric(
                items,
                "sqa_optimum_rate",
                "optimum_probability",
                "known_optimum_read_fraction",
            ),
            "sqa_fea": _mean_metric(
                items,
                "sqa_fea",
                "feasible_read_fraction",
            ),
            "feasible_read_fraction": _mean_metric(
                items,
                "feasible_read_fraction",
                "sqa_fea",
            ),
            "best_feasible_objective": _mean_metric(
                items,
                "best_feasible_objective",
            ),
            "num_feasible_reads": _mean_metric(
                items,
                "num_feasible_reads",
                "feasible_reads",
            ),
            "mean_chain_break_fraction": _mean_metric(
                items,
                "mean_chain_break_fraction",
            ),
            "broken_read_rate": _mean_metric(
                items, "broken_read_rate"
            ),
            "physical_qubits": _mean_metric(
                items, "physical_qubits"
            ),
            "mean_chain_length": _mean_metric(
                items, "mean_chain_length"
            ),
            "max_chain_length": _mean_metric(
                items, "max_chain_length"
            ),
            "effective_chain_strength": _mean_metric(
                items,
                "effective_chain_strength",
            ),
            "embedded_chain_to_problem_ratio": _mean_metric(
                items,
                "embedded_chain_to_problem_ratio",
            ),
            "wall_clock_duration_seconds": _mean_metric(
                items,
                "wall_clock_duration_seconds",
            ),
        }
        out.append(row)
    return out


def _aggregate_summary_rows(
    instance_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-instance rows into robust central-summary rows."""
    grouped: dict[
        tuple[str, int, str, int, str],
        list[dict[str, object]],
    ] = {}
    for row in instance_rows:
        hardware_size = _safe_int(row.get("hardware_size"))
        size = _safe_int(row.get("size"))
        num_variables = _safe_int(row.get("num_variables"))
        if (
            hardware_size is None
            or size is None
            or num_variables is None
        ):
            raise ValueError(
                "missing hardware_size, size, or num_variables in instance summary row"
            )
        key = (
            str(row["hardware_family"]),
            hardware_size,
            str(row["family"]),
            size,
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    metric_specs = (
        ("sqa_gap", ("sqa_gap", "objective_gap")),
        ("optimum_probability", ("optimum_probability",)),
        ("sqa_energy_gap", ("mean_sqa_energy_gap",)),
        ("sqa_optimum_rate", ("sqa_optimum_rate",)),
        ("sqa_fea", ("sqa_fea", "feasible_read_fraction")),
        (
            "mean_chain_break_fraction",
            ("mean_chain_break_fraction",),
        ),
        ("broken_read_rate", ("broken_read_rate",)),
    )
    for key in sorted(grouped):
        (
            hardware_family,
            hardware_size,
            family,
            size,
            method,
        ) = key
        items = grouped[key]
        first = items[0]
        num_variables = _safe_int(
            first.get("num_variables")
        )
        if num_variables is None:
            raise ValueError(
                "missing num_variables in instance summary row"
            )
        row: dict[str, object] = {
            "hardware_family": hardware_family,
            "hardware_size": hardware_size,
            "family": family,
            "size": size,
            "method": method,
            "n": num_variables,
            "inst": len(items),
        }
        for output_name, aliases in metric_specs:
            values = _metric_values(items, aliases)
            if values.size == 0:
                row[f"{output_name}_median"] = math.nan
                row[f"{output_name}_mean"] = math.nan
                continue
            row[f"{output_name}_median"] = float(
                np.median(values)
            )
            row[f"{output_name}_mean"] = float(
                np.mean(values)
            )
        sqa_gap_values = _metric_values(
            items, ("sqa_gap", "objective_gap")
        )
        if sqa_gap_values.size == 0:
            row["sqa_gap_zero_rate"] = math.nan
        else:
            row["sqa_gap_zero_rate"] = float(
                np.mean(
                    np.isclose(
                        sqa_gap_values, 0.0, atol=ZERO_ATOL
                    )
                )
            )
        out.append(row)
    return out


def _load_family_problems(
    family: str,
    *,
    directory: Path,
    known_optima: Mapping[
        tuple[str, str],
        KnownOptimumRecord,
    ],
) -> tuple[list[BenchmarkProblem], list[dict[str, object]]]:
    """Load and convert one benchmark family from disk."""
    problem_specs, skipped_rows = load_family_problem_specs(
        family,
        directory=directory,
    )
    problems: list[BenchmarkProblem] = []
    for spec in problem_specs:
        known_record = known_optimum_for_problem(
            spec, known_optima
        )
        problems.append(
            BenchmarkProblem(
                family=spec.family,
                instance_name=spec.instance_name,
                source_path=spec.source_path,
                size=spec.size,
                blp=spec.blp,
                known_optimum_objective=(
                    None
                    if known_record is None
                    else float(
                        known_record.optimum_objective
                    )
                ),
                known_optimum_source=(
                    None
                    if known_record is None
                    else str(known_record.optimum_source)
                ),
            )
        )
    return problems, skipped_rows


def _load_problem_index(
    *,
    known_optima_path: Path | None,
) -> tuple[
    dict[str, Path],
    dict[tuple[str, str], BenchmarkProblem],
    list[dict[str, object]],
    int,
]:
    """Load the benchmark instances used by the raw D-Wave bench."""
    benchmark_directories = {
        "mdkp": _resolve_default_repo_path(
            DEFAULT_MDKP_DIR,
            default_path=DEFAULT_MDKP_DIR,
        ),
        "mis": _resolve_default_repo_path(
            DEFAULT_MIS_DIR,
            default_path=DEFAULT_MIS_DIR,
        ),
    }
    known_optima = (
        {}
        if known_optima_path is None
        else load_known_optima_csv(known_optima_path)
    )
    problems_by_key: dict[
        tuple[str, str], BenchmarkProblem
    ] = {}
    skipped_rows: list[dict[str, object]] = []
    for family in DEFAULT_FAMILIES:
        problems, load_skips = _load_family_problems(
            family,
            directory=benchmark_directories[family],
            known_optima=known_optima,
        )
        for problem in problems:
            key = (problem.family, problem.instance_name)
            if key in problems_by_key:
                raise ValueError(
                    "duplicate benchmark problem key: "
                    f"{problem.family}/{problem.instance_name}"
                )
            problems_by_key[key] = problem
        skipped_rows.extend(load_skips)
    return (
        benchmark_directories,
        problems_by_key,
        skipped_rows,
        len(known_optima),
    )


def _session_dir_from_run_artifact(
    path: Path,
) -> Path | None:
    """Return the session directory that contains one raw run artifact."""
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "runs":
            return parent.parent.resolve()
    return None


def _has_discoverable_runs(session_dir: Path) -> bool:
    """Return whether one session directory contains raw run files."""
    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        return False
    for metadata_path in runs_dir.rglob("metadata.csv"):
        if (metadata_path.parent / "samples.csv").exists():
            return True
    return False


def _discover_session_dirs(input_path: Path) -> list[Path]:
    """Return session directories with either manifests or discoverable runs."""
    resolved_input = input_path.resolve()
    manifest_path = resolved_input / "run_manifest.json"
    if manifest_path.exists() or _has_discoverable_runs(
        resolved_input
    ):
        return [resolved_input]

    session_dirs = {
        path.parent.resolve()
        for path in resolved_input.rglob(
            "run_manifest.json"
        )
    }
    for metadata_path in resolved_input.rglob(
        "metadata.csv"
    ):
        session_dir = _session_dir_from_run_artifact(
            metadata_path
        )
        if session_dir is None:
            continue
        if (metadata_path.parent / "samples.csv").exists():
            session_dirs.add(session_dir)
    discovered = sorted(session_dirs)
    if discovered:
        return discovered

    raise FileNotFoundError(
        "no D-Wave bench session directories found under "
        f"{resolved_input}"
    )


def _discover_run_catalog_rows(
    session_dir: Path,
) -> list[dict[str, str]]:
    """Load the run catalog or reconstruct it from per-run metadata files."""
    run_catalog_path = session_dir / "run_catalog.csv"
    if run_catalog_path.exists():
        return _load_csv_rows(run_catalog_path)

    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"missing run catalog and runs directory: {run_catalog_path}"
        )

    discovered_rows: list[dict[str, str]] = []
    for metadata_path in sorted(
        runs_dir.rglob("metadata.csv")
    ):
        run_dir = metadata_path.parent
        samples_path = run_dir / "samples.csv"
        if not samples_path.exists():
            continue
        metadata_rows = _load_csv_rows(metadata_path)
        if len(metadata_rows) != 1:
            raise ValueError(
                f"expected one metadata row in discovery, found {len(metadata_rows)}: "
                f"{metadata_path}"
            )
        metadata_row = metadata_rows[0]
        run_id = str(metadata_row.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(
                f"missing run_id in metadata: {metadata_path}"
            )
        discovered_rows.append(
            {
                "run_id": run_id,
                "family": str(
                    metadata_row.get("family", "")
                ).strip(),
                "instance_name": str(
                    metadata_row.get("instance_name", "")
                ).strip(),
                "requested_method": str(
                    metadata_row.get("requested_method", "")
                ).strip(),
                "resolved_method": str(
                    metadata_row.get("resolved_method", "")
                ).strip(),
                "samples_csv_relative": str(
                    samples_path.relative_to(session_dir)
                ),
                "metadata_csv_relative": str(
                    metadata_path.relative_to(session_dir)
                ),
            }
        )
    if discovered_rows:
        return discovered_rows

    raise FileNotFoundError(
        "missing run catalog and no complete run directories found under "
        f"{runs_dir}"
    )


def _load_or_synthesize_run_manifest(
    session_dir: Path,
    run_catalog_rows: list[dict[str, str]],
) -> dict[str, object]:
    """Load the run manifest or synthesize a minimal replacement."""
    manifest_path = session_dir / "run_manifest.json"
    if manifest_path.exists():
        return json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    if not run_catalog_rows:
        return {
            "session_id": session_dir.name,
            "device_name": session_dir.name,
            "device_slug": session_dir.name,
            "synthetic_run_manifest": True,
        }

    first_metadata_path = session_dir / str(
        run_catalog_rows[0]["metadata_csv_relative"]
    )
    metadata_rows = _load_csv_rows(first_metadata_path)
    if len(metadata_rows) != 1:
        raise ValueError(
            f"expected one metadata row while synthesizing manifest: {first_metadata_path}"
        )
    metadata_row = metadata_rows[0]
    return {
        "session_id": metadata_row.get(
            "session_id", session_dir.name
        ),
        "device_name": metadata_row.get(
            "device_name", session_dir.name
        ),
        "device_slug": metadata_row.get(
            "device_slug", session_dir.name
        ),
        "hardware_family": metadata_row.get(
            "hardware_family"
        ),
        "solver_id": metadata_row.get("solver_id"),
        "synthetic_run_manifest": True,
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert arbitrary values into JSON-safe data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if (
        isinstance(value, (str, int, float, bool))
        or value is None
    ):
        return value
    return str(value)


def _sample_metrics_rows(
    *,
    problem: BenchmarkProblem,
    sample_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Annotate one raw run with all derived sample and run metrics."""
    if not sample_rows:
        raise ValueError("samples CSV is empty")

    variable_names = list(problem.blp.variable_names)
    samples = np.asarray(
        [
            [int(row[name]) for name in variable_names]
            for row in sample_rows
        ],
        dtype=float,
    )
    penalized_energies = np.asarray(
        [
            float(row["penalized_energy"])
            for row in sample_rows
        ],
        dtype=float,
    )
    objective_values = np.asarray(
        problem.blp.objective_values(samples),
        dtype=float,
    )
    feasible_mask = np.asarray(
        problem.blp.feasible_mask(samples),
        dtype=bool,
    )
    chain_break_values = [
        _parse_optional_float(
            row.get("chain_break_fraction")
        )
        for row in sample_rows
    ]

    out_rows: list[dict[str, object]] = []
    known_optimum_objective = (
        problem.known_optimum_objective
    )
    for index, raw_row in enumerate(sample_rows):
        sample_gap = None
        matches_known_optimum = None
        if known_optimum_objective is not None:
            matches_known_optimum = bool(
                feasible_mask[index]
                and objective_values[index]
                <= known_optimum_objective
                + KNOWN_OPTIMUM_TOL
            )
            if feasible_mask[index]:
                sample_gap = float(
                    _objective_gap_ratio(
                        float(objective_values[index]),
                        known_optimum_objective,
                    )
                )
        row = dict(raw_row)
        row.update(
            {
                "objective_value": float(
                    objective_values[index]
                ),
                "objective_gap": sample_gap,
                "is_feasible": bool(feasible_mask[index]),
                "known_optimum_objective": known_optimum_objective,
                "known_optimum_source": problem.known_optimum_source,
                "matches_known_optimum": matches_known_optimum,
            }
        )
        out_rows.append(row)

    total_reads = len(sample_rows)
    feasible_reads = int(np.sum(feasible_mask))
    feasible_read_fraction = feasible_reads / total_reads
    best_sample_objective = float(np.min(objective_values))
    best_feasible_objective = None
    best_feasible_gap = None
    mean_feasible_objective_gap = None
    if np.any(feasible_mask):
        best_feasible_objective = float(
            np.min(objective_values[feasible_mask])
        )

    objective_gap = None
    known_optimum_reads = None
    known_optimum_read_fraction = None
    known_optimum_hit = None
    if known_optimum_objective is not None:
        if np.any(feasible_mask):
            feasible_gaps = np.asarray(
                [
                    _objective_gap_ratio(
                        float(value),
                        known_optimum_objective,
                    )
                    for value in objective_values[
                        feasible_mask
                    ]
                ],
                dtype=float,
            )
            mean_feasible_objective_gap = float(
                np.mean(feasible_gaps)
            )
        if best_feasible_objective is not None:
            objective_gap = float(
                _objective_gap_ratio(
                    best_feasible_objective,
                    known_optimum_objective,
                )
            )
            best_feasible_gap = float(
                best_feasible_objective
                - known_optimum_objective
            )
            known_optimum_hit = bool(
                best_feasible_objective
                <= known_optimum_objective
                + KNOWN_OPTIMUM_TOL
            )
        known_optimum_mask = feasible_mask & (
            objective_values
            <= known_optimum_objective + KNOWN_OPTIMUM_TOL
        )
        known_optimum_reads = int(
            np.sum(known_optimum_mask)
        )
        known_optimum_read_fraction = (
            known_optimum_reads / total_reads
        )

    finite_chain_breaks = [
        value
        for value in chain_break_values
        if value is not None
    ]
    mean_chain_break_fraction = None
    broken_read_rate = None
    if finite_chain_breaks:
        mean_chain_break_fraction = float(
            np.mean(finite_chain_breaks)
        )
        broken_read_rate = float(
            np.mean(
                np.asarray(finite_chain_breaks, dtype=float)
                > 0.0
            )
        )

    return out_rows, {
        "num_reads_analysed": total_reads,
        "min_penalized_energy": float(
            np.min(penalized_energies)
        ),
        "mean_penalized_energy": float(
            np.mean(penalized_energies)
        ),
        "best_sample_objective": best_sample_objective,
        "best_feasible_objective": best_feasible_objective,
        "best_feasible_gap": best_feasible_gap,
        "objective_gap": objective_gap,
        "sqa_gap": objective_gap,
        "mean_feasible_objective_gap": mean_feasible_objective_gap,
        "mean_sqa_energy_gap": mean_feasible_objective_gap,
        "optimum_probability": known_optimum_read_fraction,
        "sqa_optimum_rate": known_optimum_read_fraction,
        "num_feasible_reads": feasible_reads,
        "feasible_reads": feasible_reads,
        "feasible_read_fraction": feasible_read_fraction,
        "sqa_fea": feasible_read_fraction,
        "known_optimum_objective": known_optimum_objective,
        "known_optimum_source": problem.known_optimum_source,
        "known_optimum_hit": known_optimum_hit,
        "known_optimum_reads": known_optimum_reads,
        "known_optimum_read_fraction": known_optimum_read_fraction,
        "mean_chain_break_fraction": mean_chain_break_fraction,
        "broken_read_rate": broken_read_rate,
    }


def _validate_run_files(
    *,
    catalog_row: Mapping[str, str],
    metadata_row: Mapping[str, str],
    sample_rows: list[dict[str, str]],
) -> None:
    """Reject mismatched or truncated raw run files before analysis."""
    run_id = str(catalog_row["run_id"])
    expected_pairs = (
        ("run_id", run_id),
        ("family", str(catalog_row["family"])),
        (
            "instance_name",
            str(catalog_row["instance_name"]),
        ),
        (
            "resolved_method",
            str(catalog_row["resolved_method"]),
        ),
    )
    for field_name, expected_value in expected_pairs:
        actual_value = str(metadata_row.get(field_name, ""))
        if actual_value != expected_value:
            raise ValueError(
                f"metadata {field_name} mismatch: "
                f"expected {expected_value}, found {actual_value}"
            )

    expected_reads = int(
        str(metadata_row["num_reads_returned"])
    )
    if len(sample_rows) != expected_reads:
        raise ValueError(
            f"expected {expected_reads} sample rows, found {len(sample_rows)}"
        )

    for sample_row in sample_rows:
        if str(sample_row.get("run_id", "")) != run_id:
            raise ValueError(
                "samples.csv contains rows from a different run"
            )


def _analyse_session(
    *,
    session_dir: Path,
    problems_by_key: Mapping[
        tuple[str, str], BenchmarkProblem
    ],
    benchmark_directories: Mapping[str, Path],
    known_optima_path: Path | None,
    loaded_optima_count: int,
    initial_skipped_rows: list[dict[str, object]],
) -> None:
    """Analyse one session directory produced by the raw D-Wave bench."""
    manifest_path = session_dir / "run_manifest.json"
    run_catalog_path = session_dir / "run_catalog.csv"

    run_catalog_rows = _discover_run_catalog_rows(
        session_dir
    )
    run_manifest = _load_or_synthesize_run_manifest(
        session_dir,
        run_catalog_rows,
    )
    analysed_catalog_rows: list[dict[str, object]] = []
    analysed_run_rows: list[dict[str, object]] = []
    skipped_rows = list(initial_skipped_rows)

    for catalog_row in run_catalog_rows:
        run_id = str(catalog_row["run_id"])
        try:
            samples_path = session_dir / str(
                catalog_row["samples_csv_relative"]
            )
            metadata_path = session_dir / str(
                catalog_row["metadata_csv_relative"]
            )
            metadata_rows = _load_csv_rows(metadata_path)
            if len(metadata_rows) != 1:
                raise ValueError(
                    f"expected one metadata row, found {len(metadata_rows)}"
                )
            metadata_row = dict(metadata_rows[0])
            family = (
                str(metadata_row["family"]).strip().lower()
            )
            instance_name = str(
                metadata_row["instance_name"]
            )
            problem_key = (family, instance_name)
            if problem_key not in problems_by_key:
                raise KeyError(
                    "unknown benchmark problem for run: "
                    f"{family}/{instance_name}"
                )

            problem = problems_by_key[problem_key]
            raw_sample_rows = _load_csv_rows(samples_path)
            _validate_run_files(
                catalog_row=catalog_row,
                metadata_row=metadata_row,
                sample_rows=raw_sample_rows,
            )
            sample_metric_rows, run_metrics = (
                _sample_metrics_rows(
                    problem=problem,
                    sample_rows=raw_sample_rows,
                )
            )

            run_dir = samples_path.parent
            sample_metrics_path = (
                run_dir / "sample_metrics.csv"
            )
            run_metrics_path = run_dir / "run_metrics.csv"
            _write_csv_rows(
                sample_metrics_path, sample_metric_rows
            )

            analysis_generated_at = datetime.now(
                timezone.utc
            ).isoformat()
            run_metrics_row = dict(metadata_row)
            run_metrics_row.update(run_metrics)
            run_metrics_row.update(
                {
                    "analysis_status": "success",
                    "analysis_generated_at_utc": analysis_generated_at,
                    "analysis_source_samples_csv": str(
                        samples_path
                    ),
                    "analysis_source_metadata_csv": str(
                        metadata_path
                    ),
                    "sample_metrics_csv": str(
                        sample_metrics_path
                    ),
                    "run_metrics_csv": str(
                        run_metrics_path
                    ),
                    "sample_metrics_csv_relative": str(
                        sample_metrics_path.relative_to(
                            session_dir
                        )
                    ),
                    "run_metrics_csv_relative": str(
                        run_metrics_path.relative_to(
                            session_dir
                        )
                    ),
                }
            )
            _write_csv_rows(
                run_metrics_path, [run_metrics_row]
            )
            analysed_run_rows.append(run_metrics_row)

            analysed_catalog_rows.append(
                {
                    "session_id": metadata_row.get(
                        "session_id"
                    ),
                    "device_name": metadata_row.get(
                        "device_name"
                    ),
                    "device_slug": metadata_row.get(
                        "device_slug"
                    ),
                    "run_id": run_id,
                    "status": "success",
                    "execution_backend": metadata_row.get(
                        "execution_backend"
                    ),
                    "dry_run": metadata_row.get("dry_run"),
                    "requested_method": metadata_row.get(
                        "requested_method"
                    ),
                    "resolved_method": metadata_row.get(
                        "resolved_method"
                    ),
                    "family": family,
                    "instance_name": instance_name,
                    "source_path": metadata_row.get(
                        "source_path"
                    ),
                    "problem_size": metadata_row.get(
                        "problem_size"
                    ),
                    "hardware_family": metadata_row.get(
                        "hardware_family"
                    ),
                    "hardware_size": metadata_row.get(
                        "hardware_size"
                    ),
                    "num_variables": metadata_row.get(
                        "num_variables"
                    ),
                    "num_reads_returned": metadata_row.get(
                        "num_reads_returned"
                    ),
                    "num_reads_analysed": run_metrics[
                        "num_reads_analysed"
                    ],
                    "best_feasible_objective": run_metrics[
                        "best_feasible_objective"
                    ],
                    "objective_gap": run_metrics[
                        "objective_gap"
                    ],
                    "sqa_gap": run_metrics["sqa_gap"],
                    "mean_sqa_energy_gap": run_metrics[
                        "mean_sqa_energy_gap"
                    ],
                    "optimum_probability": run_metrics[
                        "optimum_probability"
                    ],
                    "sqa_optimum_rate": run_metrics[
                        "sqa_optimum_rate"
                    ],
                    "feasible_reads": run_metrics[
                        "feasible_reads"
                    ],
                    "feasible_read_fraction": run_metrics[
                        "feasible_read_fraction"
                    ],
                    "sqa_fea": run_metrics["sqa_fea"],
                    "known_optimum_objective": run_metrics[
                        "known_optimum_objective"
                    ],
                    "known_optimum_source": run_metrics[
                        "known_optimum_source"
                    ],
                    "known_optimum_hit": run_metrics[
                        "known_optimum_hit"
                    ],
                    "known_optimum_reads": run_metrics[
                        "known_optimum_reads"
                    ],
                    "known_optimum_read_fraction": run_metrics[
                        "known_optimum_read_fraction"
                    ],
                    "mean_chain_break_fraction": run_metrics[
                        "mean_chain_break_fraction"
                    ],
                    "broken_read_rate": run_metrics[
                        "broken_read_rate"
                    ],
                    "sample_metrics_csv_relative": str(
                        sample_metrics_path.relative_to(
                            session_dir
                        )
                    ),
                    "run_metrics_csv_relative": str(
                        run_metrics_path.relative_to(
                            session_dir
                        )
                    ),
                }
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "session_id": run_manifest.get(
                        "session_id"
                    ),
                    "device_name": run_manifest.get(
                        "device_name"
                    ),
                    "run_id": run_id,
                    "family": catalog_row.get("family"),
                    "instance_name": catalog_row.get(
                        "instance_name"
                    ),
                    "requested_method": catalog_row.get(
                        "requested_method"
                    ),
                    "resolved_method": catalog_row.get(
                        "resolved_method"
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    instance_rows = _instance_summary_rows(
        analysed_run_rows
    )
    aggregate_rows = _aggregate_summary_rows(instance_rows)

    for legacy_output in (
        session_dir / "cop_instance_summary.csv",
        session_dir / "cop_aggregate_summary.csv",
        session_dir / "cop_aggregate_robust_summary.csv",
    ):
        if legacy_output.exists():
            legacy_output.unlink()

    _write_csv_rows(
        session_dir / "run_metrics_catalog.csv",
        analysed_catalog_rows,
    )
    _write_csv_rows(
        session_dir / "instance_summary.csv",
        instance_rows,
    )
    _write_csv_rows(
        session_dir / "aggregate_summary.csv",
        aggregate_rows,
    )
    _write_csv_rows(
        session_dir / "analysis_skipped.csv",
        skipped_rows,
    )
    analysis_manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "session_dir": str(session_dir),
        "session_dir_repo_relative": repo_relative_path(
            session_dir
        ),
        "run_manifest_json": (
            str(manifest_path)
            if manifest_path.exists()
            else None
        ),
        "run_catalog_csv": (
            str(run_catalog_path)
            if run_catalog_path.exists()
            else None
        ),
        "synthetic_run_manifest": bool(
            run_manifest.get("synthetic_run_manifest")
        ),
        "discovered_run_catalog": not run_catalog_path.exists(),
        "known_optima_csv": (
            None
            if known_optima_path is None
            else str(known_optima_path)
        ),
        "loaded_known_optima": int(loaded_optima_count),
        "benchmark_inputs": {
            family: repo_relative_path(path)
            for family, path in benchmark_directories.items()
        },
        "analysed_runs": len(analysed_catalog_rows),
        "analysed_instances": len(instance_rows),
        "aggregate_groups": len(aggregate_rows),
        "skipped_rows": len(skipped_rows),
        "outputs": {
            "run_metrics_catalog_csv": str(
                session_dir / "run_metrics_catalog.csv"
            ),
            "instance_summary_csv": str(
                session_dir / "instance_summary.csv"
            ),
            "aggregate_summary_csv": str(
                session_dir / "aggregate_summary.csv"
            ),
            "analysis_skipped_csv": str(
                session_dir / "analysis_skipped.csv"
            ),
        },
    }
    (session_dir / "analysis_manifest.json").write_text(
        json.dumps(
            _jsonable(analysis_manifest),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Path to one session directory or to the D-Wave bench root "
            "that contains one or more analysed sessions"
        ),
    )
    parser.add_argument(
        "--known-optima-csv",
        type=Path,
        default=DEFAULT_KNOWN_OPTIMA_CSV,
        help="CSV containing known classical optimum values",
    )
    return parser


def main() -> None:
    """Analyse raw D-Wave benchmark outputs."""
    parser = build_argument_parser()
    args = parser.parse_args()

    input_dir = _resolve_default_repo_path(
        args.input_dir,
        default_path=DEFAULT_INPUT_DIR,
    )
    known_optima_path = _resolve_optional_repo_path(
        args.known_optima_csv,
        default_path=DEFAULT_KNOWN_OPTIMA_CSV,
    )
    (
        benchmark_directories,
        problems_by_key,
        initial_skipped_rows,
        loaded_optima_count,
    ) = _load_problem_index(
        known_optima_path=known_optima_path,
    )
    session_dirs = _discover_session_dirs(input_dir)

    for session_dir in session_dirs:
        print(
            f"Analysing D-Wave bench session: {session_dir}"
        )
        _analyse_session(
            session_dir=session_dir,
            problems_by_key=problems_by_key,
            benchmark_directories=benchmark_directories,
            known_optima_path=known_optima_path,
            loaded_optima_count=loaded_optima_count,
            initial_skipped_rows=initial_skipped_rows,
        )


if __name__ == "__main__":
    main()
