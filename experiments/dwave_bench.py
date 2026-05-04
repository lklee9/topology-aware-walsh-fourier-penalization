"""Run the benchmark datasets on a real D-Wave annealer.

This driver mirrors the penalty-selection protocol used by
``experiments.compare_methods_embedding``:

1. load the family-level tuning summaries from ``experiments/tunings/``;
2. reuse the tuned unbalanced-penalty parameters for every benchmark
   instance in the same family;
3. reuse the topology-specific projected penalty selected during
   tuning for the current QPU family;
4. load or build the current-hardware projected penalty for each
   benchmark instance; and
5. sample those penalized QUBOs on the selected D-Wave solver.

Unlike ``experiments.compare_methods_embedding``, this script does not compute
performance metrics from the returned samples. Instead, each successful
QPU run writes:

* one ``samples.csv`` containing every sampled logical solution; and
* one single-row ``metadata.csv`` describing the QPU run, the selected
  penalty configuration, and where the sampled solutions were saved.

The output directory therefore behaves like a small CSV-backed database:
each run has its own record files, and the session root also stores
``run_catalog.csv``, ``skipped.csv``, and ``run_manifest.json``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import numpy as np
from dwave.system import (
    DWaveSampler,
    FixedEmbeddingComposite,
)

from experiments.experiment_config import (
    DEFAULT_MEASURE_LAM,
    DEFAULT_PROJECTED_STANDARDIZE,
    DEFAULT_PROJECTION_REG,
    DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
    FAMILY_CODES,
    PROGRESS_UI_CHOICES,
)
from experiments.utils.baseline_common import (
    qubo_normalization_scale as _shared_qubo_normalization_scale,
)
from experiments.utils.baseline_common import (
    scale_qubo_coefficients as _shared_scale_qubo_coefficients,
)
from experiments.utils.benchmark_data import (
    load_family_problem_specs,
)
from experiments.utils.driver_common import (
    binary_to_spin_states as _binary_to_spin_states,
)
from experiments.utils.driver_common import (
    build_child_rng as _build_child_rng,
)
from experiments.utils.driver_common import (
    full_pair_edges as _full_pair_edges,
)
from experiments.utils.driver_common import (
    is_complete_pair_edge_set as _is_full_pair_edge_set,
)
from experiments.utils.driver_common import (
    write_rows_csv as _write_rows_csv,
)
from experiments.utils.embedding import _qubo_arrays_to_bqm
from experiments.utils.experiment_progress import (
    ProgressTotals,
    build_progress_reporter,
)
from experiments.utils.projected_method_selection import (
    projected_method_topology,
)
from experiments.utils.projected_pipeline import (
    ProjectedPenaltyComponents,
)
from experiments.utils.projected_pipeline import (
    build_projected_components as _shared_build_projected_components,
)
from experiments.utils.projected_qubo import (
    add_qubo_terms,
    build_unit_equality_constraint_qubos,
    combine_constraint_terms,
    objective_terms,
    projection_sample_size,
    scale_terms,
)
from experiments.utils.projection_measure import (
    build_projection_sampling_catalog,
    sample_projection_states_with_inequality_support,
)
from experiments.utils.qa_simulator import (
    build_dwave_graph,
    default_qpu_solver_properties,
    find_minor_embedding,
    simulate_dwave_annealer_classically,
)
from experiments.utils.tuning_models import (
    SelectedProjectedConfig,
    TunedUnbalancedParameters,
)
from experiments.utils.tuning_summary import (
    load_selected_projected_configs,
    load_tuned_unbalanced_parameters,
)
from experiments.utils.tuning_support import (
    projected_components_cache_key as _projected_components_cache_key,
)
from experiments.utils.unbalanced_pipeline import (
    build_unbalanced_components as _build_unbalanced_components,
)
from experiments.utils.unbalanced_pipeline import (
    build_unbalanced_qubo_from_components as _build_unbalanced_qubo_from_components,
)
from fourier_projection.blp import BLP
from fourier_projection.greedy_mapping import (
    mapped_logical_topology_from_graph,
)
from fourier_projection.penalties import IdealPenalty
from fourier_projection.projection import (
    project_penalty_values_importance,
)
from fourier_projection.topology import HardwareTopology

DEFAULT_TUNING_DIR = Path("experiments/tunings")
PROJECTED_SUMMARY_FILENAME = (
    "projected_penalty_tuning_summary.csv"
)
UNBALANCED_SUMMARY_FILENAME = (
    "unbalanced_penalty_tuning_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/results/dwave_bench")
DEFAULT_PROJECTED_SUMMARY_CSV = (
    DEFAULT_TUNING_DIR / PROJECTED_SUMMARY_FILENAME
)
DEFAULT_UNBALANCED_SUMMARY_CSV = (
    DEFAULT_TUNING_DIR / UNBALANCED_SUMMARY_FILENAME
)
DEFAULT_MIS_DIR = Path("data/mis_instances")
DEFAULT_MDKP_DIR = Path("data/mdkp_instances")
DEFAULT_METHODS = (
    "projected_topology",
    "projected_full",
    "unbalanced",
)
DEFAULT_FAMILIES = ("mdkp", "mis")
DEFAULT_LIVE_QPU_TOPOLOGIES = ("pegasus", "zephyr")
DEFAULT_NUM_READS = 2500
DEFAULT_TEST_NUM_READS = 2
DEFAULT_NUM_RUNS_PER_INSTANCE = 1
DEFAULT_SEED = 1
DEFAULT_DRY_RUN_SAMPLER = "random"
DEFAULT_DRY_RUN_HARDWARE_FAMILY = "pegasus"
DEFAULT_DRY_RUN_HARDWARE_SIZE = 16
DEFAULT_DRY_RUN_QPU_ACCESS_TIME_SECONDS = 0.8
DEFAULT_TOKEN_ENV_VAR = "DWAVE_API_TOKEN"
DEFAULT_PROJECTION_BACKEND = "torch"
DEFAULT_ANNEALING_TIME = 25
DEFAULT_TEST_ANNEALING_TIME = 25
DEFAULT_AUTO_SCALE = False
DEFAULT_PROGRESS_UI = "plain"
TOKEN_ENV_CANDIDATES = (
    "DWAVE_API_TOKEN",
    "DWAVE_API_KEY",
)
ALLOWED_METHODS = {
    "unbalanced",
    "projected_full",
    "projected_topology",
}
RUNTIME_DEPENDENCIES = {
    "env_vars": [
        "DWAVE_API_TOKEN",
        "DWAVE_API_KEY",
        "DWAVE_API_ENDPOINT (optional)",
    ],
    "files": [
        str(DEFAULT_PROJECTED_SUMMARY_CSV),
        str(DEFAULT_UNBALANCED_SUMMARY_CSV),
    ],
    "directories": [
        str(DEFAULT_MIS_DIR),
        str(DEFAULT_MDKP_DIR),
    ],
    "python_modules": [
        "experiments/utils/data_loaders.py",
        "experiments/utils/benchmark_data.py",
        "experiments/utils/driver_common.py",
        "experiments/utils/embedding.py",
        "experiments/utils/projected_method_selection.py",
        "experiments/utils/projected_pipeline.py",
        "experiments/utils/projected_qubo.py",
        "experiments/utils/projection_measure.py",
        "experiments/utils/qa_simulator.py",
        "experiments/utils/tuning_models.py",
        "experiments/utils/tuning_summary.py",
        "experiments/utils/tuning_support.py",
        "experiments/utils/unb_pen.py",
        "fourier_projection/blp.py",
        "fourier_projection/greedy_mapping.py",
        "fourier_projection/penalties.py",
        "fourier_projection/projection.py",
        "fourier_projection/topology.py",
    ],
    "pip_packages": [
        "numpy",
        "networkx",
        "dimod",
        "minorminer",
        "dwave-system",
        "dwave-networkx",
    ],
}


@dataclass(frozen=True)
class BenchmarkProblem:
    """One benchmark instance converted into the repo's BLP form."""

    family: str
    instance_name: str
    source_path: Path
    size: int
    blp: BLP


@dataclass(frozen=True)
class DeviceSummary:
    """Stable subset of the selected QPU metadata."""

    requested_topology: str | None
    requested_device: str | None
    solver_id: str
    chip_id: str | None
    topology_type: str | None
    topology_shape: tuple[int, ...]
    num_qubits: int
    num_couplers: int
    location: str | None
    category: str | None
    avg_load: float | None


@dataclass
class QpuTimeAccumulator:
    """Track cumulative QPU access time across live target sessions."""

    total_qpu_access_time_us: float = 0.0
    live_runs_with_qpu_timing: int = 0
    live_runs_missing_qpu_timing: int = 0

    def record_qpu_access_time(
        self, access_time_us: float | None
    ) -> None:
        """Add one live run's QPU access time, when reported."""
        if access_time_us is None:
            self.live_runs_missing_qpu_timing += 1
            return
        self.total_qpu_access_time_us += float(
            access_time_us
        )
        self.live_runs_with_qpu_timing += 1


def _load_family_problems(
    family: str,
    *,
    directory: Path,
) -> tuple[list[BenchmarkProblem], list[dict[str, object]]]:
    """Load and convert one benchmark family from disk."""
    problem_specs, skipped_rows = load_family_problem_specs(
        family,
        directory=directory,
    )
    problems: list[BenchmarkProblem] = []
    for spec in problem_specs:
        problems.append(
            BenchmarkProblem(
                family=spec.family,
                instance_name=spec.instance_name,
                source_path=spec.source_path,
                size=spec.size,
                blp=spec.blp,
            )
        )
    return problems, skipped_rows


def _resolve_default_repo_path(
    requested_path: Path,
    *,
    default_path: Path,
) -> Path:
    """
    Resolve default CLI paths relative to the repository root.

    User-supplied paths keep their normal CLI behavior and therefore
    resolve relative to the invocation directory when they are not
    absolute.
    """
    if requested_path == default_path:
        return (REPO_ROOT / default_path).resolve()
    return requested_path.resolve()


def _resolve_tuning_summary_paths(
    tuning_dir: Path,
) -> tuple[Path, Path]:
    """Resolve the required tuning summary CSVs from one directory."""
    resolved_tuning_dir = _resolve_default_repo_path(
        tuning_dir,
        default_path=DEFAULT_TUNING_DIR,
    )
    projected_summary_path = (
        resolved_tuning_dir / PROJECTED_SUMMARY_FILENAME
    ).resolve()
    unbalanced_summary_path = (
        resolved_tuning_dir / UNBALANCED_SUMMARY_FILENAME
    ).resolve()
    if not projected_summary_path.exists():
        raise FileNotFoundError(
            "projected tuning summary not found: "
            f"{projected_summary_path}"
        )
    if not unbalanced_summary_path.exists():
        raise FileNotFoundError(
            "unbalanced tuning summary not found: "
            f"{unbalanced_summary_path}"
        )
    return projected_summary_path, unbalanced_summary_path


def _resolve_dwave_token(
    preferred_env_var: str,
    *,
    cli_token: str | None = None,
) -> tuple[str, str, str | None]:
    """Return the D-Wave token plus how it was provided."""
    if cli_token is not None:
        stripped_token = str(cli_token).strip()
        if stripped_token:
            return stripped_token, "cli_arg", None

    candidates = [preferred_env_var, *TOKEN_ENV_CANDIDATES]
    seen: set[str] = set()
    for env_var in candidates:
        if env_var in seen:
            continue
        seen.add(env_var)
        value = os.environ.get(env_var)
        if value:
            return value, "env_var", env_var
    tried = ", ".join(seen)
    raise RuntimeError(
        "missing D-Wave API token; pass --dwave-api-token or set one of "
        f"{tried}"
    )


def _build_sampler(
    *,
    qpu_topology: str,
    token: str,
) -> DWaveSampler:
    """Construct the real D-Wave sampler for the selected QPU."""
    config: dict[str, Any] = {
        "token": token,
        "solver": {
            "topology__type": str(qpu_topology)
            .strip()
            .lower(),
        },
    }
    endpoint = os.environ.get("DWAVE_API_ENDPOINT")
    if endpoint:
        config["endpoint"] = endpoint
    return DWaveSampler(**config)


def _parse_optional_float(value: object) -> float | None:
    """Return one optional floating-point value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _device_summary(
    sampler: DWaveSampler,
    *,
    requested_topology: str | None,
    requested_device: str | None,
) -> DeviceSummary:
    """Extract a stable subset of solver metadata."""
    properties = dict(
        getattr(sampler, "properties", {}) or {}
    )
    topology = properties.get("topology", {})
    raw_shape = topology.get("shape", ())
    if isinstance(raw_shape, (list, tuple)):
        topology_shape = tuple(
            int(value) for value in raw_shape
        )
    else:
        topology_shape = ()

    solver = getattr(sampler, "solver", None)
    if solver is not None and getattr(solver, "id", None):
        solver_id = str(solver.id)
    else:
        fallback_solver_id = properties.get("chip_id")
        if fallback_solver_id is None:
            fallback_solver_id = (
                requested_device or requested_topology
            )
        solver_id = str(fallback_solver_id)

    return DeviceSummary(
        requested_topology=requested_topology,
        requested_device=requested_device,
        solver_id=solver_id,
        chip_id=(
            None
            if properties.get("chip_id") is None
            else str(properties.get("chip_id"))
        ),
        topology_type=(
            None
            if topology.get("type") is None
            else str(topology.get("type")).strip().lower()
        ),
        topology_shape=topology_shape,
        num_qubits=len(sampler.nodelist),
        num_couplers=len(sampler.edgelist),
        location=(
            None
            if properties.get("location") is None
            else str(properties.get("location"))
        ),
        category=(
            None
            if properties.get("category") is None
            else str(properties.get("category"))
        ),
        avg_load=_parse_optional_float(
            properties.get("avg_load")
        ),
    )


def _offline_device_summary(
    *,
    requested_topology: str,
    hardware_family: str,
    hardware_size: int,
    hardware_graph: nx.Graph,
) -> DeviceSummary:
    """Return synthetic device metadata for offline dry runs."""
    solver_id = (
        f"offline_{hardware_family}_{int(hardware_size)}"
    )
    return DeviceSummary(
        requested_topology=requested_topology,
        requested_device=None,
        solver_id=solver_id,
        chip_id=None,
        topology_type=str(hardware_family),
        topology_shape=(int(hardware_size),),
        num_qubits=int(hardware_graph.number_of_nodes()),
        num_couplers=int(hardware_graph.number_of_edges()),
        location="offline",
        category="offline_simulation",
        avg_load=None,
    )


def _normalize_hardware_family(
    topology_type: str | None,
) -> str:
    """Return the canonical QPU family name."""
    if topology_type is None:
        raise RuntimeError(
            "selected solver does not expose topology.type; "
            "cannot resolve the QPU-specific projected method"
        )
    normalized = str(topology_type).strip().lower()
    if normalized not in {"chimera", "pegasus", "zephyr"}:
        raise RuntimeError(
            "unsupported QPU topology family for projected tuning: "
            f"{topology_type}"
        )
    return normalized


def _hardware_projected_summary_method(
    hardware_family: str,
) -> str:
    """Map `projected_topology` to the topology-specific summary id."""
    methods = {
        "chimera": "projected_chimera",
        "pegasus": "projected_pegasus",
        "zephyr": "projected_zephyr",
    }
    try:
        return methods[str(hardware_family)]
    except KeyError as exc:
        raise ValueError(
            "unknown hardware family for projected summary lookup: "
            f"{hardware_family}"
        ) from exc


def _resolve_requested_method(
    method: str,
    *,
    hardware_family: str,
) -> str:
    """Resolve one supported method name to the tuned summary id."""
    normalized = str(method).strip().lower()
    if normalized not in ALLOWED_METHODS:
        raise ValueError(f"unknown method: {method}")
    if normalized == "projected_topology":
        return _hardware_projected_summary_method(
            hardware_family
        )
    return normalized


def _resolve_requested_methods(
    methods: list[str],
    *,
    hardware_family: str,
) -> list[tuple[str, str]]:
    """Resolve the requested methods while rejecting duplicates."""
    resolved: list[tuple[str, str]] = []
    seen_resolved: dict[str, str] = {}
    for requested_method in methods:
        actual_method = _resolve_requested_method(
            requested_method,
            hardware_family=hardware_family,
        )
        if actual_method != "unbalanced":
            expected_topology = projected_method_topology(
                actual_method
            )
            if expected_topology not in {
                hardware_family,
                "fully_connected",
            }:
                raise ValueError(
                    f"method {requested_method} resolves to {actual_method}, "
                    f"which targets {expected_topology}, not "
                    f"{hardware_family}"
                )
        if actual_method in seen_resolved:
            previous = seen_resolved[actual_method]
            raise ValueError(
                f"duplicate effective method selection: {previous} and "
                f"{requested_method} both resolve to {actual_method}"
            )
        seen_resolved[actual_method] = requested_method
        resolved.append((requested_method, actual_method))
    return resolved


def _with_projected_standardize_default(
    config: SelectedProjectedConfig,
) -> SelectedProjectedConfig:
    """Fill in the shared projected-standardize default when absent."""
    if config.projected_standardize is not None:
        return config
    return replace(
        config,
        projected_standardize=DEFAULT_PROJECTED_STANDARDIZE,
    )


def _load_tuning_configs(
    *,
    projected_summary_path: Path,
    unbalanced_summary_path: Path,
    families: tuple[str, ...],
    resolved_methods: list[tuple[str, str]],
) -> tuple[
    dict[str, TunedUnbalancedParameters],
    dict[tuple[str, str], SelectedProjectedConfig],
]:
    """Load the family-level tuning configs used by the benchmark."""
    tuned_unbalanced = load_tuned_unbalanced_parameters(
        unbalanced_summary_path
    )
    projected_configs = {
        key: _with_projected_standardize_default(config)
        for key, config in load_selected_projected_configs(
            projected_summary_path
        ).items()
    }

    for family in families:
        if family not in tuned_unbalanced:
            raise RuntimeError(
                "missing unbalanced tuning summary for "
                f"{family} in {unbalanced_summary_path}; regenerate tuning "
                "artifacts for that family before running this command"
            )

    required_projected_methods = {
        actual_method
        for _, actual_method in resolved_methods
        if actual_method != "unbalanced"
    }

    for family in families:
        for method in required_projected_methods:
            key = (family, method)
            if key not in projected_configs:
                raise RuntimeError(
                    "missing projected tuning summary for "
                    f"{family}/{method} in {projected_summary_path}; regenerate "
                    "tuning artifacts for that family before running this command"
                )
    return tuned_unbalanced, projected_configs


def _qpu_projection_pair_edges(
    problem: BLP,
    hardware_graph: nx.Graph,
) -> list[tuple[int, int]]:
    """Map the logical coupler graph onto the selected QPU topology."""
    placement, topology = (
        mapped_logical_topology_from_graph(
            problem.constraint_matrix,
            hardware_graph,
            logical_vertices=range(problem.num_variables),
        )
    )
    edges = [tuple(edge) for edge in topology.E]
    if all(
        0 <= u < problem.num_variables
        and 0 <= v < problem.num_variables
        for u, v in edges
    ):
        return sorted(
            {tuple(sorted(edge)) for edge in edges}
        )

    hardware_to_logical = {
        hardware_vertex: logical_vertex
        for logical_vertex, hardware_vertex in placement.items()
    }
    canonical_edges: set[tuple[int, int]] = set()
    for u, v in edges:
        if (
            u not in hardware_to_logical
            or v not in hardware_to_logical
        ):
            continue
        logical_u = int(hardware_to_logical[u])
        logical_v = int(hardware_to_logical[v])
        if logical_u == logical_v:
            continue
        canonical_edges.add(
            tuple(sorted((logical_u, logical_v)))
        )

    if not canonical_edges:
        raise ValueError(
            "QPU logical topology could not be canonicalized"
        )
    return sorted(canonical_edges)


def _qpu_projection_topology_details(
    problem: BLP,
    hardware_graph: nx.Graph,
) -> tuple[list[tuple[int, int]], dict[int, list[Any]]]:
    """Return the projected logical couplers plus the injective placement."""
    placement, topology = (
        mapped_logical_topology_from_graph(
            problem.constraint_matrix,
            hardware_graph,
            logical_vertices=range(problem.num_variables),
        )
    )
    edges = [tuple(edge) for edge in topology.E]
    fixed_embedding = {
        int(logical_vertex): [hardware_vertex]
        for logical_vertex, hardware_vertex in placement.items()
    }
    if all(
        0 <= u < problem.num_variables
        and 0 <= v < problem.num_variables
        for u, v in edges
    ):
        return (
            sorted({tuple(sorted(edge)) for edge in edges}),
            fixed_embedding,
        )

    hardware_to_logical = {
        hardware_vertex: logical_vertex
        for logical_vertex, hardware_vertex in placement.items()
    }
    canonical_edges: set[tuple[int, int]] = set()
    for u, v in edges:
        if (
            u not in hardware_to_logical
            or v not in hardware_to_logical
        ):
            continue
        logical_u = int(hardware_to_logical[u])
        logical_v = int(hardware_to_logical[v])
        if logical_u == logical_v:
            continue
        canonical_edges.add(
            tuple(sorted((logical_u, logical_v)))
        )

    if not canonical_edges:
        raise ValueError(
            "QPU logical topology could not be canonicalized"
        )
    return sorted(canonical_edges), fixed_embedding


def _build_projected_components(
    problem: BLP,
    *,
    family: str,
    size: int,
    instance_index: int,
    base_seed: int,
    pair_edges: list[tuple[int, int]],
    measure_name: str,
    template_name: str,
    template_kwargs: dict[str, float] | None,
    projected_standardize: bool,
    sample_cap_log2: int,
    projection_reg: float,
    projection_backend: str,
) -> ProjectedPenaltyComponents:
    """Build the projected equality and inequality penalty pieces."""
    sample_size = projection_sample_size(
        problem,
        sample_cap_log2,
    )
    sample_rng = _build_child_rng(
        base_seed,
        2_000,
        FAMILY_CODES[family],
        size,
        instance_index,
    )
    return _shared_build_projected_components(
        problem,
        pair_edges=pair_edges,
        sample_size=sample_size,
        sample_rng=sample_rng,
        measure_name=measure_name,
        measure_lam=DEFAULT_MEASURE_LAM,
        penalty_template=template_name,
        penalty_template_kwargs=template_kwargs,
        reg=projection_reg,
        standardize=projected_standardize,
        build_projection_sampling_catalog=(
            build_projection_sampling_catalog
        ),
        sample_projection_states_with_inequality_support=(
            sample_projection_states_with_inequality_support
        ),
        build_unit_equality_constraint_qubos=(
            build_unit_equality_constraint_qubos
        ),
        combine_constraint_terms=combine_constraint_terms,
        project_penalty_values_importance=functools.partial(
            project_penalty_values_importance,
            backend=projection_backend,
        ),
        hardware_topology_cls=HardwareTopology,
        ideal_penalty_cls=IdealPenalty,
        binary_to_spin_states=_binary_to_spin_states,
        is_complete_pair_edge_set=_is_full_pair_edge_set,
    )


def _resolve_projected_components(
    benchmark_problem: BenchmarkProblem,
    *,
    run_id: str,
    instance_index: int,
    config: SelectedProjectedConfig,
    hardware_family: str,
    hardware_graph: nx.Graph,
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ],
    sample_cap_log2: int,
    projection_reg: float,
    base_seed: int,
    projection_backend: str,
) -> tuple[
    ProjectedPenaltyComponents,
    str,
    list[tuple[int, int]],
    dict[int, list[Any]] | None,
]:
    """Build one projected-penalty fit for the current hardware."""
    problem = benchmark_problem.blp
    family = benchmark_problem.family
    size = benchmark_problem.size
    projection_method = (
        config.projection_method or config.method
    )
    if projection_method == "projected_full":
        pair_edges = _full_pair_edges(
            int(problem.num_variables)
        )
        fixed_embedding = None
        deployment_topology = "fully_connected"
        deployment_topology_size = int(
            problem.num_variables
        )
    else:
        _log_runtime_event(
            "Resolving QPU logical topology for projected penalty",
            run_id=run_id,
        )
        topology_started = perf_counter()
        pair_edges, fixed_embedding = (
            _qpu_projection_topology_details(
                problem,
                hardware_graph,
            )
        )
        _log_runtime_event(
            "Resolved QPU logical topology "
            f"with {len(pair_edges)} logical pair edges",
            run_id=run_id,
            elapsed_s=perf_counter() - topology_started,
        )
        deployment_topology = hardware_family
        deployment_topology_size = (
            hardware_graph.number_of_nodes()
        )
    cache_key = _projected_components_cache_key(
        projection_method=projection_method,
        family=family,
        size=size,
        instance_index=instance_index,
        measure_name=config.measure_name,
        measure_lam=DEFAULT_MEASURE_LAM,
        penalty_template=config.penalty_template,
        penalty_template_kwargs=config.penalty_template_kwargs,
        standardize=config.projected_standardize,
        deployment_topology=deployment_topology,
        deployment_topology_size=deployment_topology_size,
    )
    if cache_key in components_cache:
        _log_runtime_event(
            "Reusing cached projected penalty components from memory",
            run_id=run_id,
        )
        return (
            components_cache[cache_key],
            projection_method,
            pair_edges,
            fixed_embedding,
        )

    projected_samples = projection_sample_size(
        problem,
        sample_cap_log2,
    )
    _log_runtime_event(
        "Starting projected penalty construction "
        f"method={projection_method} "
        f"measure={config.measure_name} "
        f"samples={projected_samples}",
        run_id=run_id,
    )
    projection_started = perf_counter()
    components = _build_projected_components(
        problem,
        family=family,
        size=size,
        instance_index=instance_index,
        base_seed=base_seed,
        pair_edges=pair_edges,
        measure_name=config.measure_name,
        template_name=config.penalty_template,
        template_kwargs=config.penalty_template_kwargs,
        projected_standardize=bool(
            config.projected_standardize
        ),
        sample_cap_log2=sample_cap_log2,
        projection_reg=projection_reg,
        projection_backend=projection_backend,
    )
    _log_runtime_event(
        "Finished projected penalty construction",
        run_id=run_id,
        elapsed_s=perf_counter() - projection_started,
    )
    components_cache[cache_key] = components
    return (
        components,
        projection_method,
        pair_edges,
        fixed_embedding,
    )


def _build_method_qubo(
    benchmark_problem: BenchmarkProblem,
    *,
    run_id: str,
    instance_index: int,
    requested_method: str,
    resolved_method: str,
    hardware_family: str,
    hardware_graph: nx.Graph,
    tuned_unbalanced: dict[str, TunedUnbalancedParameters],
    projected_configs: dict[
        tuple[str, str], SelectedProjectedConfig
    ],
    projected_summary_path: Path,
    unbalanced_summary_path: Path,
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ],
    sample_cap_log2: int,
    projection_reg: float,
    base_seed: int,
    projection_backend: str,
) -> tuple[Any, dict[str, object]]:
    """Build the method-specific BQM plus metadata for reporting."""
    problem = benchmark_problem.blp
    family = benchmark_problem.family

    def normalized_bqm(
        quadratic: np.ndarray,
        linear: np.ndarray,
        const: float,
    ) -> tuple[Any, float]:
        """Return the normalized logical QUBO used for annealing."""
        if hardware_graph.number_of_nodes() == 0:
            return (
                _qubo_arrays_to_bqm(
                    quadratic, linear, const
                ),
                1.0,
            )

        normalization_scale = (
            _shared_qubo_normalization_scale(
                quadratic,
                linear,
                num_variables=np.asarray(
                    linear, dtype=float
                )
                .reshape(-1)
                .shape[0],
            )
        )
        (
            scaled_quadratic,
            scaled_linear,
            scaled_const,
        ) = _shared_scale_qubo_coefficients(
            quadratic,
            linear,
            const,
            normalization_scale=normalization_scale,
        )
        return (
            _qubo_arrays_to_bqm(
                scaled_quadratic,
                scaled_linear,
                scaled_const,
            ),
            float(normalization_scale),
        )

    if resolved_method == "unbalanced":
        _log_runtime_event(
            "Building unbalanced penalty QUBO",
            run_id=run_id,
        )
        build_started = perf_counter()
        tuned = tuned_unbalanced[family]
        up_components = _build_unbalanced_components(
            problem,
            lambda1_shape=float(tuned.up_lambda1_shape),
            lambda2_shape=float(tuned.up_lambda2_shape),
            standardize=(
                True
                if tuned.per_constraint_standardization
                is None
                else bool(
                    tuned.per_constraint_standardization
                )
            ),
        )
        quadratic, linear, const = (
            _build_unbalanced_qubo_from_components(
                problem,
                up_components,
                equality_multiplier=(
                    0.0
                    if tuned.up_equality_multiplier is None
                    else float(tuned.up_equality_multiplier)
                ),
                inequality_multiplier=float(
                    tuned.up_inequality_multiplier
                ),
            )
        )
        bqm, normalization_scale = normalized_bqm(
            quadratic,
            linear,
            const,
        )
        _log_runtime_event(
            "Built unbalanced penalty QUBO",
            run_id=run_id,
            elapsed_s=perf_counter() - build_started,
        )
        return bqm, {
            "requested_method": requested_method,
            "resolved_method": resolved_method,
            "anchor_size": int(tuned.anchor_size),
            "penalty_config_source": str(
                unbalanced_summary_path
            ),
            "tuning_objective": tuned.tuning_objective,
            "tuning_objective_value": float(
                tuned.objective_value
            ),
            "base_parameter_source": tuned.base_parameter_source,
            "up_global_multiplier": tuned.global_multiplier,
            "up_equality_multiplier": tuned.up_equality_multiplier,
            "up_inequality_multiplier": (
                tuned.up_inequality_multiplier
            ),
            "up_lambda1_shape": tuned.up_lambda1_shape,
            "up_lambda2_shape": tuned.up_lambda2_shape,
            "up_lambda_gauge": tuned.up_lambda_gauge,
            "up_normalization_regime": tuned.normalization_regime,
            "up_per_constraint_standardization": (
                tuned.per_constraint_standardization
            ),
            "up_lambda0": tuned.lambda0,
            "up_lambda1": tuned.lambda1,
            "up_lambda2": tuned.lambda2,
            "qubo_normalization_scale": normalization_scale,
            "projection_method": None,
            "projection_measure": None,
            "projection_penalty_template": None,
            "projection_penalty_template_kwargs_json": None,
            "projection_selection_mode": None,
            "projection_selection_source": None,
            "projection_candidate_rank": None,
            "projected_standardize": None,
            "projected_samples": None,
            "logical_pair_edges": None,
            "projected_equality_multiplier": None,
            "projected_inequality_multiplier": None,
            "fixed_embedding": None,
            "embedding_strategy": "minor_embedding",
        }

    config = projected_configs[(family, resolved_method)]
    components, projection_method, _, fixed_embedding = (
        _resolve_projected_components(
            benchmark_problem,
            run_id=run_id,
            instance_index=instance_index,
            config=config,
            hardware_family=hardware_family,
            hardware_graph=hardware_graph,
            components_cache=components_cache,
            sample_cap_log2=sample_cap_log2,
            projection_reg=projection_reg,
            base_seed=base_seed,
            projection_backend=projection_backend,
        )
    )

    assemble_started = perf_counter()
    terms = [objective_terms(problem)]
    if (
        problem.num_equalities
        and config.tuning.equality_multiplier != 0.0
    ):
        terms.append(
            scale_terms(
                components.equality_terms,
                config.tuning.equality_multiplier,
            )
        )
    if (
        problem.num_inequalities
        and config.tuning.inequality_multiplier != 0.0
    ):
        terms.append(
            scale_terms(
                components.inequality_terms,
                config.tuning.inequality_multiplier,
            )
        )

    qubo = add_qubo_terms(*terms)
    bqm, normalization_scale = normalized_bqm(
        qubo.quadratic,
        qubo.linear,
        qubo.const,
    )
    _log_runtime_event(
        "Assembled projected QUBO",
        run_id=run_id,
        elapsed_s=perf_counter() - assemble_started,
    )
    return bqm, {
        "requested_method": requested_method,
        "resolved_method": resolved_method,
        "anchor_size": int(config.tuning.anchor_size),
        "penalty_config_source": str(
            projected_summary_path
        ),
        "tuning_objective": config.tuning.tuning_objective,
        "tuning_objective_value": float(
            config.tuning.objective_value
        ),
        "projection_method": projection_method,
        "projection_measure": config.measure_name,
        "projection_penalty_template": config.penalty_template,
        "projection_penalty_template_kwargs_json": _json_dumps(
            config.penalty_template_kwargs
        ),
        "projection_selection_mode": config.selection_mode,
        "projection_selection_source": config.selection_source,
        "projection_candidate_rank": int(
            config.candidate_rank
        ),
        "projected_standardize": bool(
            config.projected_standardize
        ),
        "projected_samples": int(components.sample_size),
        "logical_pair_edges": int(
            components.num_quadratic_couplers
        ),
        "projected_equality_multiplier": (
            config.tuning.equality_multiplier
        ),
        "projected_inequality_multiplier": (
            config.tuning.inequality_multiplier
        ),
        "qubo_normalization_scale": normalization_scale,
        "up_global_multiplier": None,
        "up_lambda0": None,
        "up_lambda1": None,
        "up_lambda2": None,
        "base_parameter_source": None,
        "fixed_embedding": fixed_embedding,
        "embedding_strategy": (
            "minor_embedding"
            if fixed_embedding is None
            else "topology_injective_placement"
        ),
    }


def _embedding_stats(
    embedding: dict[Any, list[Any]],
) -> dict[str, float | int]:
    """Return simple chain statistics for one embedding."""
    chain_lengths = [
        len(chain) for chain in embedding.values()
    ]
    return {
        "physical_qubits": int(sum(chain_lengths)),
        "mean_chain_length": float(np.mean(chain_lengths)),
        "max_chain_length": int(max(chain_lengths)),
    }


def _assert_embedding_matches_bqm(
    bqm: Any,
    hardware_graph: nx.Graph,
    embedding: dict[int, list[Any]] | None,
) -> None:
    """Reject fixed embeddings when the BQM adds edges outside the placement."""
    if embedding is None:
        return
    if set(bqm.variables) != set(embedding):
        raise ValueError(
            "fixed embedding does not cover the logical BQM variables"
        )
    singleton_nodes = {
        chain[0]
        for chain in embedding.values()
        if len(chain) == 1
    }
    if len(singleton_nodes) != len(embedding):
        raise ValueError(
            "fixed embedding must contain singleton chains only"
        )
    for u, v in bqm.quadratic:
        physical_u = embedding[int(u)][0]
        physical_v = embedding[int(v)][0]
        if physical_u == physical_v:
            continue
        if not hardware_graph.has_edge(
            physical_u, physical_v
        ):
            raise ValueError(
                "fixed embedding is incompatible with the logical BQM couplers"
            )


def _sampleset_arrays(
    sampleset: Any,
    num_variables: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
]:
    """Extract logical samples, energies, occurrences, and breaks."""
    samples = np.asarray(
        [
            [
                int(sample[index])
                for index in range(num_variables)
            ]
            for sample in sampleset.samples()
        ],
        dtype=int,
    )
    energies = np.asarray(
        sampleset.record.energy, dtype=float
    )
    occurrences = np.asarray(
        sampleset.record.num_occurrences,
        dtype=int,
    )
    chain_break_fraction = None
    if (
        "chain_break_fraction"
        in sampleset.record.dtype.names
    ):
        chain_break_fraction = np.asarray(
            sampleset.record.chain_break_fraction,
            dtype=float,
        )
    return (
        samples,
        energies,
        occurrences,
        chain_break_fraction,
    )


def _slugify(value: str) -> str:
    """Return a filesystem-safe slug."""
    text = re.sub(
        r"[^A-Za-z0-9._-]+", "_", str(value).strip()
    )
    text = text.strip("._")
    return text or "item"


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


def _json_dumps(value: Any) -> str:
    """Return one stable JSON encoding."""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
    )


def _filtered_property_json(
    mapping: Mapping[str, Any],
    *terms: str,
) -> str:
    """Return a JSON subset of keys matching any substring."""
    lowered = tuple(term.lower() for term in terms)
    filtered = {
        str(key): value
        for key, value in mapping.items()
        if any(term in str(key).lower() for term in lowered)
    }
    return _json_dumps(filtered)


def _log_runtime_event(
    message: str,
    *,
    run_id: str | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Print one timestamped runtime log line."""
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    parts = [f"[{timestamp}]"]
    # if run_id is not None:
    #     parts.append(f"[{run_id}]")
    parts.append(message)
    if elapsed_s is not None:
        parts.append(f"(elapsed={elapsed_s:.3f}s)")
    print(" ".join(parts), flush=True)


def _scalar_timing_fields(
    timing: Mapping[str, Any],
) -> dict[str, object]:
    """Flatten scalar timing values into CSV columns."""
    fields: dict[str, object] = {}
    for key, value in timing.items():
        if (
            isinstance(value, (str, int, float, bool))
            or value is None
        ):
            fields[f"timing_{key}"] = value
    return fields


def _optional_float(value: Any) -> float | None:
    """Return a finite float for numeric-like timing values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _qpu_access_time_us(
    timing: Mapping[str, Any],
) -> float | None:
    """Return D-Wave's QPU access time in microseconds, when available."""
    return _optional_float(timing.get("qpu_access_time"))


def _microseconds_to_seconds(
    value_us: float | None,
) -> float | None:
    """Convert microseconds to seconds while preserving missing values."""
    if value_us is None:
        return None
    return float(value_us) / 1_000_000.0


def _format_qpu_seconds(value_us: float | None) -> str:
    """Format a microsecond QPU duration for runtime logs."""
    value_s = _microseconds_to_seconds(value_us)
    if value_s is None:
        return "unknown"
    return f"{value_s:.6f}s"


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


def _sample_rows(
    *,
    run_id: str,
    benchmark_problem: BenchmarkProblem,
    samples: np.ndarray,
    penalized_energies: np.ndarray,
    occurrences: np.ndarray,
    chain_break_fraction: np.ndarray | None,
) -> list[dict[str, object]]:
    """Expand the logical samples into one CSV row per returned read."""
    problem = benchmark_problem.blp
    rows: list[dict[str, object]] = []
    expanded_index = 0
    for row_index in range(samples.shape[0]):
        chain_break_value = (
            None
            if chain_break_fraction is None
            else float(chain_break_fraction[row_index])
        )
        count = int(occurrences[row_index])
        for occurrence_index in range(count):
            row: dict[str, object] = {
                name: int(samples[row_index, var_index])
                for var_index, name in enumerate(
                    problem.variable_names
                )
            }
            row.update(
                {
                    "run_id": run_id,
                    "sample_index": expanded_index,
                    "sampleset_row_index": int(row_index),
                    "occurrence_index_within_row": int(
                        occurrence_index
                    ),
                    "row_num_occurrences": count,
                    "penalized_energy": float(
                        penalized_energies[row_index]
                    ),
                    "chain_break_fraction": chain_break_value,
                }
            )
            rows.append(row)
            expanded_index += 1
    return rows


def _session_id() -> str:
    """Return one UTC timestamp-based session id."""
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-env-var",
        default=DEFAULT_TOKEN_ENV_VAR,
        help="Preferred environment variable for the D-Wave API token",
    )
    parser.add_argument(
        "--dwave-api-token",
        default=None,
        help=(
            "Direct D-Wave API token. When provided, this overrides "
            "--token-env-var and the environment-variable lookup."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive the per-session run database",
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=DEFAULT_TUNING_DIR,
        help=(
            "Directory containing projected_penalty_tuning_summary.csv "
            "and unbalanced_penalty_tuning_summary.csv"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do not submit to the QPU; instead sample the embedded "
            "problem locally with a cheap random baseline sampler"
        ),
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help=(
            "Submit only the smallest loaded benchmark problem to "
            "the live QPU once for Pegasus and once for Zephyr, "
            f"using {DEFAULT_TEST_NUM_READS} reads and a shorter "
            "default annealing time unless --annealing-time "
            "overrides it"
        ),
    )
    parser.add_argument(
        "--annealing-time",
        type=float,
        default=None,
        help=(
            "Annealing time in microseconds. Defaults to "
            f"{DEFAULT_ANNEALING_TIME} for normal runs and "
            f"{DEFAULT_TEST_ANNEALING_TIME} for --test-run."
        ),
    )
    parser.add_argument(
        "--progress-ui",
        choices=PROGRESS_UI_CHOICES,
        default=DEFAULT_PROGRESS_UI,
        help=(
            "Console progress rendering mode. Use 'tui' for a Rich-based "
            "multi-bar UI."
        ),
    )
    return parser


def _load_benchmark_inputs() -> tuple[
    dict[str, Path],
    dict[str, list[BenchmarkProblem]],
    list[dict[str, object]],
]:
    """Resolve benchmark directories and load the default problem sets."""
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
    problems_by_family: dict[
        str, list[BenchmarkProblem]
    ] = {}
    skipped_rows: list[dict[str, object]] = []
    for family in DEFAULT_FAMILIES:
        problems, load_skips = _load_family_problems(
            family,
            directory=benchmark_directories[family],
        )
        problems_by_family[family] = problems
        skipped_rows.extend(load_skips)
    return (
        benchmark_directories,
        problems_by_family,
        skipped_rows,
    )


def _benchmark_problem_sort_key(
    problem: BenchmarkProblem,
) -> tuple[int, int, int, str, str]:
    """Return a deterministic cross-family size key."""
    return (
        int(problem.blp.num_variables),
        int(
            problem.blp.num_equalities
            + problem.blp.num_inequalities
        ),
        int(problem.size),
        str(problem.family),
        str(problem.instance_name),
    )


def _select_smallest_benchmark_problem(
    problems_by_family: Mapping[
        str, list[BenchmarkProblem]
    ],
) -> BenchmarkProblem:
    """Return the smallest loaded benchmark problem."""
    all_problems = [
        problem
        for family in DEFAULT_FAMILIES
        for problem in problems_by_family.get(family, [])
    ]
    if not all_problems:
        raise RuntimeError(
            "no benchmark problems were loaded"
        )
    return min(
        all_problems,
        key=_benchmark_problem_sort_key,
    )


def _run_target_session(
    *,
    session_id: str,
    root_output_dir: Path,
    tuning_dir: Path,
    projected_summary_path: Path,
    unbalanced_summary_path: Path,
    requested_token_env_var: str,
    resolved_token_env_var: str | None,
    token_source: str,
    progress_ui: str,
    dry_run: bool,
    test_run: bool,
    num_reads: int,
    annealing_time: float | None,
    qpu_time_accumulator: QpuTimeAccumulator,
    sampler: DWaveSampler | None,
    device_summary: DeviceSummary,
    hardware_family: str,
    hardware_graph: nx.Graph,
    solver_properties: Mapping[str, Any],
    solver_parameters: Mapping[str, Any],
    benchmark_directories: Mapping[str, Path],
    problems_by_family: Mapping[
        str, list[BenchmarkProblem]
    ],
    initial_skipped_rows: list[dict[str, object]],
) -> None:
    """Run the benchmark for one resolved hardware target."""
    device_name = device_summary.solver_id
    device_slug = _slugify(device_name)
    session_dir = root_output_dir / session_id / device_slug
    session_dir.mkdir(parents=True, exist_ok=False)

    requested_methods = list(DEFAULT_METHODS)
    requested_families = tuple(problems_by_family)
    if not requested_families:
        raise RuntimeError(
            "no benchmark families were selected"
        )
    resolved_methods = _resolve_requested_methods(
        requested_methods,
        hardware_family=hardware_family,
    )
    tuned_unbalanced, projected_configs = (
        _load_tuning_configs(
            projected_summary_path=projected_summary_path,
            unbalanced_summary_path=unbalanced_summary_path,
            families=requested_families,
            resolved_methods=resolved_methods,
        )
    )

    total_instances = sum(
        len(problems_by_family[family])
        for family in requested_families
    )
    total_method_batches = total_instances * len(
        resolved_methods
    )
    total_runs = (
        total_method_batches * DEFAULT_NUM_RUNS_PER_INSTANCE
    )
    reporter_mode = (
        "rich" if progress_ui == "tui" else progress_ui
    )
    reporter = build_progress_reporter(
        mode=str(reporter_mode),
        totals=ProgressTotals(
            instances=int(total_instances),
            topologies=int(total_method_batches),
            measures=int(total_runs),
        ),
        worker_count=1,
        stream=sys.stderr,
    )

    print(
        "Running on selected_solver="
        f"{device_summary.solver_id} "
        "requested_topology="
        f"{device_summary.requested_topology} "
        f"family={hardware_family} "
        f"qubits={device_summary.num_qubits} "
        f"couplers={device_summary.num_couplers}"
    )
    if dry_run:
        print(
            "Dry run enabled: using a cheap random sampler "
            f"on a local {hardware_family}({DEFAULT_DRY_RUN_HARDWARE_SIZE}) "
            "graph instead of QPU submission"
        )
    if test_run:
        print(
            "Test run enabled: submitting only the selected "
            "smallest benchmark problem "
            f"with num_reads={num_reads} and "
            f"annealing_time={annealing_time} microseconds"
        )
    print(
        "Using tuning dir="
        f"{tuning_dir} "
        f"projected_summary={projected_summary_path} "
        f"unbalanced_summary={unbalanced_summary_path}"
    )

    auto_scale = DEFAULT_AUTO_SCALE
    catalog_rows: list[dict[str, object]] = []
    skipped_rows = list(initial_skipped_rows)
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ] = {}
    target_qpu_access_time_us = 0.0
    target_live_runs_with_qpu_timing = 0
    target_live_runs_missing_qpu_timing = 0
    show_qpu_time_in_status = str(progress_ui) == "tui"

    def update_status(**kwargs: object) -> None:
        """Update progress status, adding QPU totals to the TUI detail."""
        if show_qpu_time_in_status:
            detail = kwargs.get("detail")
            qpu_detail = (
                "total_qpu_time="
                f"{_format_qpu_seconds(qpu_time_accumulator.total_qpu_access_time_us)}"
            )
            kwargs["detail"] = (
                qpu_detail
                if detail is None
                else f"{detail}; {qpu_detail}"
            )
        reporter.update_status(**kwargs)

    reporter.start()
    try:
        update_status(
            stage="initializing",
            topology=str(hardware_family),
            detail=(
                f"requested_topology={device_summary.requested_topology}; "
                f"solver={device_summary.solver_id}; "
                f"instances={total_instances}; "
                f"methods={len(resolved_methods)}; "
                f"runs={DEFAULT_NUM_RUNS_PER_INSTANCE}"
            ),
        )

        for family in requested_families:
            problems = problems_by_family[family]
            print(
                f"Loaded {len(problems)} {family} benchmark instance(s)"
            )
            update_status(
                stage="family",
                family=str(family),
                detail=f"loaded {len(problems)} instance(s)",
            )
            for (
                instance_index,
                benchmark_problem,
            ) in enumerate(problems):
                print(
                    f"[{family} {instance_index + 1}/{len(problems)}] "
                    f"{benchmark_problem.instance_name}"
                )
                update_status(
                    stage="instance",
                    family=str(family),
                    size=int(benchmark_problem.size),
                    instance_index=int(instance_index),
                    total_instances=int(len(problems)),
                    detail=str(
                        benchmark_problem.instance_name
                    ),
                )
                for method_offset, (
                    requested_method,
                    resolved_method,
                ) in enumerate(resolved_methods):
                    update_status(
                        stage="method",
                        topology=str(hardware_family),
                        measure=str(resolved_method),
                        detail=(
                            f"method {method_offset + 1}/{len(resolved_methods)}"
                        ),
                    )
                    method_seed = (
                        DEFAULT_SEED
                        + (100_000 * instance_index)
                        + (1_000 * method_offset)
                    )
                    for run_index in range(
                        DEFAULT_NUM_RUNS_PER_INSTANCE
                    ):
                        run_seed = method_seed + run_index
                        run_id = (
                            f"{session_id}__{device_slug}__{family}__"
                            f"{_slugify(benchmark_problem.instance_name)}__"
                            f"{_slugify(requested_method)}__"
                            f"run_{run_index + 1:03d}"
                        )
                        run_dir = (
                            session_dir
                            / "runs"
                            / family
                            / _slugify(
                                benchmark_problem.instance_name
                            )
                            / _slugify(resolved_method)
                            / f"run_{run_index + 1:03d}"
                        )
                        samples_path = (
                            run_dir / "samples.csv"
                        )
                        metadata_path = (
                            run_dir / "metadata.csv"
                        )
                        label = f"dwave-bench|{run_id}|{resolved_method}"
                        started_at = datetime.now(
                            timezone.utc
                        )
                        started_counter = perf_counter()

                        update_status(
                            stage="run",
                            measure=str(resolved_method),
                            detail=(
                                f"run {run_index + 1}/"
                                f"{DEFAULT_NUM_RUNS_PER_INSTANCE}"
                            ),
                        )
                        try:
                            _log_runtime_event(
                                "Starting run "
                                f"family={family} "
                                f"instance={benchmark_problem.instance_name} "
                                f"method={resolved_method} "
                                f"seed={run_seed}",
                                run_id=run_id,
                            )
                            update_status(
                                stage="qubo",
                                detail="building penalized QUBO",
                            )
                            bqm, method_info = (
                                _build_method_qubo(
                                    benchmark_problem,
                                    run_id=run_id,
                                    instance_index=instance_index,
                                    requested_method=requested_method,
                                    resolved_method=resolved_method,
                                    hardware_family=hardware_family,
                                    hardware_graph=hardware_graph,
                                    tuned_unbalanced=tuned_unbalanced,
                                    projected_configs=projected_configs,
                                    projected_summary_path=projected_summary_path,
                                    unbalanced_summary_path=unbalanced_summary_path,
                                    components_cache=components_cache,
                                    sample_cap_log2=DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                                    projection_reg=DEFAULT_PROJECTION_REG,
                                    base_seed=DEFAULT_SEED,
                                    projection_backend=DEFAULT_PROJECTION_BACKEND,
                                )
                            )

                            fixed_embedding = method_info[
                                "fixed_embedding"
                            ]
                            if fixed_embedding is None:
                                update_status(
                                    stage="embedding",
                                    detail="minor-embedding search",
                                )
                                _log_runtime_event(
                                    "Starting minor-embedding search",
                                    run_id=run_id,
                                )
                                embedding_started = (
                                    perf_counter()
                                )
                                embedding = find_minor_embedding(
                                    bqm,
                                    hardware_graph,
                                    random_seed=run_seed,
                                )
                                _log_runtime_event(
                                    "Finished minor-embedding search",
                                    run_id=run_id,
                                    elapsed_s=perf_counter()
                                    - embedding_started,
                                )
                            else:
                                _assert_embedding_matches_bqm(
                                    bqm,
                                    hardware_graph,
                                    fixed_embedding,
                                )
                                update_status(
                                    stage="embedding",
                                    detail="reusing injective projected placement",
                                )
                                _log_runtime_event(
                                    "Reusing injective projected placement as fixed embedding",
                                    run_id=run_id,
                                )
                                embedding = fixed_embedding
                            embedding_stats = (
                                _embedding_stats(embedding)
                            )
                            timing: dict[str, Any] = {}
                            qpu_access_time_us: (
                                float | None
                            ) = None
                            backend_name = "dwave_qpu"
                            effective_chain_strength = None
                            if dry_run:
                                dry_run_rng = (
                                    _build_child_rng(
                                        run_seed,
                                        6_000,
                                    )
                                )
                                update_status(
                                    stage="sampling",
                                    detail=(
                                        "dry-run sampling "
                                        f"({DEFAULT_DRY_RUN_SAMPLER})"
                                    ),
                                )
                                _log_runtime_event(
                                    "Sampling embedded problem with dry-run "
                                    f"{DEFAULT_DRY_RUN_SAMPLER} backend "
                                    f"num_reads={num_reads}",
                                    run_id=run_id,
                                )
                                sampling_started = (
                                    perf_counter()
                                )
                                simulation = simulate_dwave_annealer_classically(
                                    logical_bqm=bqm,
                                    hardware_graph=hardware_graph,
                                    chain_strength_multiplier=None,
                                    effective_chain_strength=None,
                                    num_reads=num_reads,
                                    rng=dry_run_rng,
                                    embedding=embedding,
                                    sampler_name=DEFAULT_DRY_RUN_SAMPLER,
                                    decoder_name="majority_vote",
                                )
                                _log_runtime_event(
                                    "Finished dry-run sampling",
                                    run_id=run_id,
                                    elapsed_s=perf_counter()
                                    - sampling_started,
                                )
                                sampleset = (
                                    simulation.decoded_sampleset
                                )
                                backend_name = (
                                    "simulated_dwave_"
                                    f"{DEFAULT_DRY_RUN_SAMPLER}"
                                )
                                effective_chain_strength = float(
                                    simulation.effective_chain_strength
                                )
                                qpu_access_time_us = (
                                    DEFAULT_DRY_RUN_QPU_ACCESS_TIME_SECONDS
                                    * 1_000_000.0
                                )
                                timing = {
                                    "qpu_access_time": qpu_access_time_us,
                                    "qpu_access_time_source": "dry_run_fake",
                                }
                                qpu_time_accumulator.record_qpu_access_time(
                                    qpu_access_time_us
                                )
                                target_qpu_access_time_us += (
                                    qpu_access_time_us
                                )
                                target_live_runs_with_qpu_timing += (
                                    1
                                )
                                _log_runtime_event(
                                    "Dry-run fake QPU access time this run="
                                    f"{_format_qpu_seconds(qpu_access_time_us)}; "
                                    "target total="
                                    f"{_format_qpu_seconds(target_qpu_access_time_us)}; "
                                    "experiment total="
                                    f"{_format_qpu_seconds(qpu_time_accumulator.total_qpu_access_time_us)}",
                                    run_id=run_id,
                                )
                                update_status(
                                    stage="sampling",
                                    detail="dry-run sampling complete",
                                )
                            else:
                                if sampler is None:
                                    raise RuntimeError(
                                        "live QPU run is missing a D-Wave sampler"
                                    )
                                composite = (
                                    FixedEmbeddingComposite(
                                        sampler,
                                        embedding,
                                    )
                                )
                                sample_kwargs: dict[
                                    str, object
                                ] = {
                                    "num_reads": num_reads,
                                    "auto_scale": auto_scale,
                                    "answer_mode": "raw",
                                    "label": label,
                                }
                                if (
                                    annealing_time
                                    is not None
                                ):
                                    sample_kwargs[
                                        "annealing_time"
                                    ] = annealing_time

                                update_status(
                                    stage="sampling",
                                    detail="QPU sampling",
                                )
                                annealing_time_detail = ""
                                if (
                                    annealing_time
                                    is not None
                                ):
                                    annealing_time_detail = f" annealing_time={annealing_time}us"
                                _log_runtime_event(
                                    "Submitting QPU sampling request "
                                    f"num_reads={num_reads}"
                                    f"{annealing_time_detail}",
                                    run_id=run_id,
                                )
                                sampling_started = (
                                    perf_counter()
                                )
                                sampleset = (
                                    composite.sample(
                                        bqm,
                                        **sample_kwargs,
                                    )
                                )
                                _log_runtime_event(
                                    "Received QPU sampling response",
                                    run_id=run_id,
                                    elapsed_s=perf_counter()
                                    - sampling_started,
                                )
                                timing = dict(
                                    sampleset.info.get(
                                        "timing", {}
                                    )
                                )
                                qpu_access_time_us = (
                                    _qpu_access_time_us(
                                        timing
                                    )
                                )
                                qpu_time_accumulator.record_qpu_access_time(
                                    qpu_access_time_us
                                )
                                if (
                                    qpu_access_time_us
                                    is None
                                ):
                                    target_live_runs_missing_qpu_timing += (
                                        1
                                    )
                                    _log_runtime_event(
                                        "QPU response did not include "
                                        "qpu_access_time; cumulative known "
                                        "QPU access time remains "
                                        f"{_format_qpu_seconds(qpu_time_accumulator.total_qpu_access_time_us)}",
                                        run_id=run_id,
                                    )
                                else:
                                    target_qpu_access_time_us += (
                                        qpu_access_time_us
                                    )
                                    target_live_runs_with_qpu_timing += (
                                        1
                                    )
                                    _log_runtime_event(
                                        "QPU access time this run="
                                        f"{_format_qpu_seconds(qpu_access_time_us)}; "
                                        "target total="
                                        f"{_format_qpu_seconds(target_qpu_access_time_us)}; "
                                        "experiment total="
                                        f"{_format_qpu_seconds(qpu_time_accumulator.total_qpu_access_time_us)}",
                                        run_id=run_id,
                                    )
                                update_status(
                                    stage="sampling",
                                    detail="QPU sampling complete",
                                )
                            finished_at = datetime.now(
                                timezone.utc
                            )
                            duration_s = (
                                perf_counter()
                                - started_counter
                            )

                            update_status(
                                stage="decoding",
                                detail="decoding sampleset",
                            )
                            _log_runtime_event(
                                "Decoding sampleset arrays",
                                run_id=run_id,
                            )
                            decode_started = perf_counter()
                            (
                                samples,
                                penalized_energies,
                                occurrences,
                                chain_break_fraction,
                            ) = _sampleset_arrays(
                                sampleset,
                                benchmark_problem.blp.num_variables,
                            )
                            _log_runtime_event(
                                "Decoded sampleset arrays",
                                run_id=run_id,
                                elapsed_s=perf_counter()
                                - decode_started,
                            )
                            _log_runtime_event(
                                "Materializing per-read sample rows",
                                run_id=run_id,
                            )
                            rows_started = perf_counter()
                            sample_rows = _sample_rows(
                                run_id=run_id,
                                benchmark_problem=benchmark_problem,
                                samples=samples,
                                penalized_energies=penalized_energies,
                                occurrences=occurrences,
                                chain_break_fraction=chain_break_fraction,
                            )
                            _log_runtime_event(
                                "Built sample rows",
                                run_id=run_id,
                                elapsed_s=perf_counter()
                                - rows_started,
                            )
                            update_status(
                                stage="writing",
                                detail="writing CSV outputs",
                            )
                            _log_runtime_event(
                                "Writing sample CSV",
                                run_id=run_id,
                            )
                            write_samples_started = (
                                perf_counter()
                            )
                            _write_csv_rows(
                                samples_path,
                                sample_rows,
                            )
                            _log_runtime_event(
                                f"Wrote sample CSV to {samples_path}",
                                run_id=run_id,
                                elapsed_s=(
                                    perf_counter()
                                    - write_samples_started
                                ),
                            )
                            total_reads = int(
                                np.sum(occurrences)
                            )

                            metadata_row: dict[
                                str, object
                            ] = {
                                "session_id": session_id,
                                "device_name": device_name,
                                "device_slug": device_slug,
                                "run_id": run_id,
                                "status": "success",
                                "execution_backend": backend_name,
                                "dry_run": bool(dry_run),
                                "test_run": bool(test_run),
                                "requested_method": requested_method,
                                "resolved_method": resolved_method,
                                "family": family,
                                "instance_name": benchmark_problem.instance_name,
                                "source_path": str(
                                    benchmark_problem.source_path
                                ),
                                "problem_size": benchmark_problem.size,
                                "num_variables": (
                                    benchmark_problem.blp.num_variables
                                ),
                                "num_equalities": (
                                    benchmark_problem.blp.num_equalities
                                ),
                                "num_inequalities": (
                                    benchmark_problem.blp.num_inequalities
                                ),
                                "hardware_family": hardware_family,
                                "requested_topology": (
                                    device_summary.requested_topology
                                ),
                                "requested_device": (
                                    device_summary.requested_device
                                ),
                                "solver_id": device_summary.solver_id,
                                "chip_id": device_summary.chip_id,
                                "solver_category": device_summary.category,
                                "solver_location": device_summary.location,
                                "solver_avg_load": device_summary.avg_load,
                                "solver_topology_type": (
                                    device_summary.topology_type
                                ),
                                "solver_topology_shape_json": _json_dumps(
                                    device_summary.topology_shape
                                ),
                                "solver_num_qubits": (
                                    device_summary.num_qubits
                                ),
                                "solver_num_couplers": (
                                    device_summary.num_couplers
                                ),
                                "solver_temperature_json": (
                                    _filtered_property_json(
                                        solver_properties,
                                        "temp",
                                        "temperature",
                                    )
                                ),
                                "solver_calibration_json": (
                                    _filtered_property_json(
                                        solver_properties,
                                        "calibration",
                                    )
                                ),
                                "solver_topology_json": _json_dumps(
                                    solver_properties.get(
                                        "topology", {}
                                    )
                                ),
                                "solver_properties_json": (
                                    _json_dumps(
                                        solver_properties
                                    )
                                ),
                                "solver_parameters_json": (
                                    _json_dumps(
                                        solver_parameters
                                    )
                                ),
                                "token_source": token_source,
                                "token_env_var": resolved_token_env_var,
                                "label": label,
                                "answer_mode_requested": "raw",
                                "num_reads_requested": num_reads,
                                "num_reads_returned": total_reads,
                                "num_distinct_sampleset_rows": int(
                                    samples.shape[0]
                                ),
                                "annealing_time_requested": annealing_time,
                                "chain_strength_requested": None,
                                "effective_chain_strength": (
                                    effective_chain_strength
                                ),
                                "auto_scale": auto_scale,
                                "auto_scale_factor": None,
                                "qubo_normalization_scale": (
                                    method_info[
                                        "qubo_normalization_scale"
                                    ]
                                ),
                                "dry_run_num_sweeps": None,
                                "dry_run_num_sweeps_per_beta": None,
                                "dry_run_sqa_schedule_id": None,
                                "dry_run_sqa_schedule_json": None,
                                "dry_run_sqa_beta_scale": None,
                                "projection_summary_csv": str(
                                    projected_summary_path
                                ),
                                "unbalanced_summary_csv": str(
                                    unbalanced_summary_path
                                ),
                                "anchor_size": method_info[
                                    "anchor_size"
                                ],
                                "penalty_config_source": (
                                    method_info[
                                        "penalty_config_source"
                                    ]
                                ),
                                "tuning_objective": (
                                    method_info[
                                        "tuning_objective"
                                    ]
                                ),
                                "tuning_objective_value": (
                                    method_info[
                                        "tuning_objective_value"
                                    ]
                                ),
                                "base_parameter_source": (
                                    method_info[
                                        "base_parameter_source"
                                    ]
                                ),
                                "up_global_multiplier": (
                                    method_info[
                                        "up_global_multiplier"
                                    ]
                                ),
                                "up_lambda0": method_info[
                                    "up_lambda0"
                                ],
                                "up_lambda1": method_info[
                                    "up_lambda1"
                                ],
                                "up_lambda2": method_info[
                                    "up_lambda2"
                                ],
                                "projection_method": method_info[
                                    "projection_method"
                                ],
                                "projection_measure": method_info[
                                    "projection_measure"
                                ],
                                "projection_penalty_template": (
                                    method_info[
                                        "projection_penalty_template"
                                    ]
                                ),
                                "projection_penalty_template_kwargs_json": (
                                    method_info[
                                        "projection_penalty_template_kwargs_json"
                                    ]
                                ),
                                "projection_selection_mode": (
                                    method_info[
                                        "projection_selection_mode"
                                    ]
                                ),
                                "projection_selection_source": (
                                    method_info[
                                        "projection_selection_source"
                                    ]
                                ),
                                "projection_candidate_rank": (
                                    method_info[
                                        "projection_candidate_rank"
                                    ]
                                ),
                                "projected_standardize": (
                                    method_info[
                                        "projected_standardize"
                                    ]
                                ),
                                "projected_samples": (
                                    method_info[
                                        "projected_samples"
                                    ]
                                ),
                                "projected_equality_multiplier": (
                                    method_info[
                                        "projected_equality_multiplier"
                                    ]
                                ),
                                "projected_inequality_multiplier": (
                                    method_info[
                                        "projected_inequality_multiplier"
                                    ]
                                ),
                                "logical_pair_edges": (
                                    method_info[
                                        "logical_pair_edges"
                                    ]
                                ),
                                "embedding_strategy": (
                                    method_info[
                                        "embedding_strategy"
                                    ]
                                ),
                                "physical_qubits": (
                                    embedding_stats[
                                        "physical_qubits"
                                    ]
                                ),
                                "mean_chain_length": (
                                    embedding_stats[
                                        "mean_chain_length"
                                    ]
                                ),
                                "max_chain_length": (
                                    embedding_stats[
                                        "max_chain_length"
                                    ]
                                ),
                                "embedding_json": _json_dumps(
                                    embedding
                                ),
                                "sampleset_info_json": _json_dumps(
                                    sampleset.info
                                ),
                                "embedding_context_json": _json_dumps(
                                    sampleset.info.get(
                                        "embedding_context",
                                        {},
                                    )
                                ),
                                "problem_id": sampleset.info.get(
                                    "problem_id"
                                ),
                                "run_started_at_utc": started_at.isoformat(),
                                "run_finished_at_utc": finished_at.isoformat(),
                                "wall_clock_duration_seconds": duration_s,
                                "qpu_access_time_us": qpu_access_time_us,
                                "qpu_access_time_seconds": (
                                    _microseconds_to_seconds(
                                        qpu_access_time_us
                                    )
                                ),
                                "target_cumulative_qpu_access_time_us": (
                                    target_qpu_access_time_us
                                ),
                                "target_cumulative_qpu_access_time_seconds": (
                                    _microseconds_to_seconds(
                                        target_qpu_access_time_us
                                    )
                                ),
                                "experiment_cumulative_qpu_access_time_us": (
                                    qpu_time_accumulator.total_qpu_access_time_us
                                ),
                                "experiment_cumulative_qpu_access_time_seconds": (
                                    _microseconds_to_seconds(
                                        qpu_time_accumulator.total_qpu_access_time_us
                                    )
                                ),
                                "samples_csv": str(
                                    samples_path
                                ),
                                "metadata_csv": str(
                                    metadata_path
                                ),
                                "samples_csv_relative": str(
                                    samples_path.relative_to(
                                        session_dir
                                    )
                                ),
                                "metadata_csv_relative": str(
                                    metadata_path.relative_to(
                                        session_dir
                                    )
                                ),
                            }
                            metadata_row.update(
                                _scalar_timing_fields(
                                    timing
                                )
                            )
                            metadata_row["timing_json"] = (
                                _json_dumps(timing)
                            )

                            _log_runtime_event(
                                "Writing metadata CSV",
                                run_id=run_id,
                            )
                            write_metadata_started = (
                                perf_counter()
                            )
                            _write_csv_rows(
                                metadata_path,
                                [metadata_row],
                            )
                            _log_runtime_event(
                                f"Wrote metadata CSV to {metadata_path}",
                                run_id=run_id,
                                elapsed_s=(
                                    perf_counter()
                                    - write_metadata_started
                                ),
                            )
                            catalog_rows.append(
                                {
                                    "session_id": session_id,
                                    "device_name": device_name,
                                    "device_slug": device_slug,
                                    "run_id": run_id,
                                    "status": "success",
                                    "execution_backend": backend_name,
                                    "dry_run": bool(
                                        dry_run
                                    ),
                                    "test_run": bool(
                                        test_run
                                    ),
                                    "requested_method": requested_method,
                                    "resolved_method": resolved_method,
                                    "family": family,
                                    "instance_name": (
                                        benchmark_problem.instance_name
                                    ),
                                    "source_path": str(
                                        benchmark_problem.source_path
                                    ),
                                    "problem_size": benchmark_problem.size,
                                    "requested_topology": (
                                        device_summary.requested_topology
                                    ),
                                    "requested_device": (
                                        device_summary.requested_device
                                    ),
                                    "solver_id": device_summary.solver_id,
                                    "hardware_family": hardware_family,
                                    "num_reads_returned": total_reads,
                                    "qpu_access_time_us": qpu_access_time_us,
                                    "target_cumulative_qpu_access_time_us": (
                                        target_qpu_access_time_us
                                    ),
                                    "experiment_cumulative_qpu_access_time_us": (
                                        qpu_time_accumulator.total_qpu_access_time_us
                                    ),
                                    "problem_id": sampleset.info.get(
                                        "problem_id"
                                    ),
                                    "samples_csv_relative": str(
                                        samples_path.relative_to(
                                            session_dir
                                        )
                                    ),
                                    "metadata_csv_relative": str(
                                        metadata_path.relative_to(
                                            session_dir
                                        )
                                    ),
                                }
                            )
                            print(
                                "  wrote "
                                f"{resolved_method} run {run_index + 1}/"
                                f"{DEFAULT_NUM_RUNS_PER_INSTANCE}: "
                                f"{len(sample_rows)} samples"
                            )
                        except Exception as exc:
                            update_status(
                                stage="skipped",
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            skipped_rows.append(
                                {
                                    "session_id": session_id,
                                    "run_id": run_id,
                                    "device_name": device_name,
                                    "hardware_family": hardware_family,
                                    "test_run": bool(
                                        test_run
                                    ),
                                    "family": family,
                                    "instance_name": (
                                        benchmark_problem.instance_name
                                    ),
                                    "source_path": str(
                                        benchmark_problem.source_path
                                    ),
                                    "problem_size": benchmark_problem.size,
                                    "requested_method": requested_method,
                                    "resolved_method": resolved_method,
                                    "run_index": run_index
                                    + 1,
                                    "error_type": type(
                                        exc
                                    ).__name__,
                                    "error_message": str(
                                        exc
                                    ),
                                }
                            )
                            print(
                                f"  skipped {resolved_method} "
                                f"run {run_index + 1}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        finally:
                            reporter.advance_measures(1)

                    reporter.advance_topologies(1)

                reporter.advance_instances(1)

        update_status(
            stage="finalizing",
            detail="writing session catalogs",
        )
        manifest = {
            "session_id": session_id,
            "device_name": device_name,
            "device_slug": device_slug,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "device": device_name,
            "dry_run": bool(dry_run),
            "test_run": bool(test_run),
            "hardware_family": hardware_family,
            "requested_topology": device_summary.requested_topology,
            "token_source": token_source,
            "resolved_token_env_var": resolved_token_env_var,
            "qpu_timing_totals": {
                "target_qpu_access_time_us": target_qpu_access_time_us,
                "target_qpu_access_time_seconds": _microseconds_to_seconds(
                    target_qpu_access_time_us
                ),
                "target_live_runs_with_qpu_timing": (
                    target_live_runs_with_qpu_timing
                ),
                "target_live_runs_missing_qpu_timing": (
                    target_live_runs_missing_qpu_timing
                ),
                "experiment_qpu_access_time_us": (
                    qpu_time_accumulator.total_qpu_access_time_us
                ),
                "experiment_qpu_access_time_seconds": _microseconds_to_seconds(
                    qpu_time_accumulator.total_qpu_access_time_us
                ),
                "experiment_live_runs_with_qpu_timing": (
                    qpu_time_accumulator.live_runs_with_qpu_timing
                ),
                "experiment_live_runs_missing_qpu_timing": (
                    qpu_time_accumulator.live_runs_missing_qpu_timing
                ),
            },
            "solver": {
                "requested_topology": device_summary.requested_topology,
                "requested_device": device_summary.requested_device,
                "solver_id": device_summary.solver_id,
                "chip_id": device_summary.chip_id,
                "category": device_summary.category,
                "location": device_summary.location,
                "avg_load": device_summary.avg_load,
                "topology_type": device_summary.topology_type,
                "topology_shape": device_summary.topology_shape,
                "num_qubits": device_summary.num_qubits,
                "num_couplers": device_summary.num_couplers,
            },
            "args": {
                "progress_ui": str(progress_ui),
                "dry_run": bool(dry_run),
                "test_run": bool(test_run),
                "tuning_dir": str(tuning_dir),
                "output_dir": str(root_output_dir),
                "token_env_var": str(
                    requested_token_env_var
                ),
                "dwave_api_token_provided": token_source
                == "cli_arg",
            },
            "internal_defaults": {
                "families": list(requested_families),
                "methods": list(DEFAULT_METHODS),
                "live_qpu_topologies": list(
                    DEFAULT_LIVE_QPU_TOPOLOGIES
                ),
                "resolved_methods": [
                    {
                        "requested_method": requested,
                        "resolved_method": resolved,
                    }
                    for requested, resolved in resolved_methods
                ],
                "num_runs_per_instance": DEFAULT_NUM_RUNS_PER_INSTANCE,
                "num_reads": int(num_reads),
                "annealing_time": annealing_time,
                "chain_strength": None,
                "seed": DEFAULT_SEED,
                "sample_cap_log2": DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                "projection_reg": DEFAULT_PROJECTION_REG,
                "projection_backend": DEFAULT_PROJECTION_BACKEND,
                "auto_scale": DEFAULT_AUTO_SCALE,
                "dry_run_sampler": DEFAULT_DRY_RUN_SAMPLER,
                "dry_run_hardware_family": (
                    DEFAULT_DRY_RUN_HARDWARE_FAMILY
                ),
                "dry_run_hardware_size": DEFAULT_DRY_RUN_HARDWARE_SIZE,
                "dry_run_num_sweeps": None,
                "dry_run_num_sweeps_per_beta": None,
                "dry_run_sqa_schedule_id": None,
                "dry_run_sqa_schedule_json": None,
                "dry_run_sqa_beta_scale": None,
            },
            "tuning_summaries": {
                "projected": str(projected_summary_path),
                "unbalanced": str(unbalanced_summary_path),
            },
            "benchmark_inputs": {
                family: {
                    "directory": str(
                        benchmark_directories[family]
                    ),
                    "instances_loaded": len(
                        problems_by_family[family]
                    ),
                    "instance_names": [
                        problem.instance_name
                        for problem in problems_by_family[
                            family
                        ]
                    ],
                }
                for family in requested_families
            },
            "runtime_dependencies": RUNTIME_DEPENDENCIES,
        }
        _write_csv_rows(
            session_dir / "run_catalog.csv",
            catalog_rows,
        )
        _write_csv_rows(
            session_dir / "skipped.csv",
            skipped_rows,
        )
        (session_dir / "run_manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        update_status(
            stage="done",
            detail="session complete",
        )
    finally:
        reporter.close()

    print(
        f"Wrote {len(catalog_rows)} successful run record(s) "
        f"and {len(skipped_rows)} skipped row(s) to {session_dir}"
    )


def main() -> None:
    """Run the D-Wave benchmark."""
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.dry_run and args.test_run:
        parser.error(
            "--dry-run and --test-run are mutually exclusive"
        )

    projected_summary_path, unbalanced_summary_path = (
        _resolve_tuning_summary_paths(args.tuning_dir)
    )
    tuning_dir = projected_summary_path.parent
    root_output_dir = _resolve_default_repo_path(
        args.output_dir,
        default_path=DEFAULT_OUTPUT_DIR,
    )
    session_id = _session_id()
    (
        benchmark_directories,
        problems_by_family,
        skipped_rows,
    ) = _load_benchmark_inputs()
    if (
        args.annealing_time is not None
        and args.annealing_time <= 0.0
    ):
        parser.error("--annealing-time must be positive")

    annealing_time = (
        float(args.annealing_time)
        if args.annealing_time is not None
        else float(DEFAULT_ANNEALING_TIME)
    )
    if args.test_run:
        smallest_problem = (
            _select_smallest_benchmark_problem(
                problems_by_family
            )
        )
        problems_by_family = {
            smallest_problem.family: [smallest_problem]
        }
        skipped_rows = []
        if args.annealing_time is None:
            annealing_time = float(
                DEFAULT_TEST_ANNEALING_TIME
            )
        print(
            "Test run selected smallest benchmark problem "
            f"{smallest_problem.family}/"
            f"{smallest_problem.instance_name} "
            f"(size={smallest_problem.size}, "
            f"num_variables={smallest_problem.blp.num_variables}, "
            f"num_equalities={smallest_problem.blp.num_equalities}, "
            f"num_inequalities={smallest_problem.blp.num_inequalities}) "
            f"with num_reads={DEFAULT_TEST_NUM_READS} and "
            f"annealing_time={annealing_time} microseconds"
        )

    qpu_time_accumulator = QpuTimeAccumulator()

    if args.dry_run:
        hardware_family = DEFAULT_DRY_RUN_HARDWARE_FAMILY
        hardware_graph = build_dwave_graph(
            hardware_family,
            DEFAULT_DRY_RUN_HARDWARE_SIZE,
        )
        device_summary = _offline_device_summary(
            requested_topology=hardware_family,
            hardware_family=hardware_family,
            hardware_size=DEFAULT_DRY_RUN_HARDWARE_SIZE,
            hardware_graph=hardware_graph,
        )
        solver_properties: dict[str, Any] = {
            **default_qpu_solver_properties(
                hardware_family,
                hardware_size=DEFAULT_DRY_RUN_HARDWARE_SIZE,
            ),
            "category": "offline_simulation",
            "location": "offline",
        }
        solver_parameters: dict[str, Any] = {}
        _run_target_session(
            session_id=session_id,
            root_output_dir=root_output_dir,
            tuning_dir=tuning_dir,
            projected_summary_path=projected_summary_path,
            unbalanced_summary_path=unbalanced_summary_path,
            requested_token_env_var=str(args.token_env_var),
            resolved_token_env_var=None,
            token_source="dry_run",
            progress_ui=str(args.progress_ui),
            dry_run=True,
            test_run=False,
            num_reads=DEFAULT_NUM_READS,
            annealing_time=annealing_time,
            qpu_time_accumulator=qpu_time_accumulator,
            sampler=None,
            device_summary=device_summary,
            hardware_family=hardware_family,
            hardware_graph=hardware_graph,
            solver_properties=solver_properties,
            solver_parameters=solver_parameters,
            benchmark_directories=benchmark_directories,
            problems_by_family=problems_by_family,
            initial_skipped_rows=skipped_rows,
        )
        return

    token, token_source, resolved_token_env_var = (
        _resolve_dwave_token(
            str(args.token_env_var),
            cli_token=args.dwave_api_token,
        )
    )
    for requested_topology in DEFAULT_LIVE_QPU_TOPOLOGIES:
        sampler: DWaveSampler | None = None
        try:
            sampler = _build_sampler(
                qpu_topology=requested_topology,
                token=token,
            )
            hardware_graph = sampler.to_networkx_graph()
            device_summary = _device_summary(
                sampler,
                requested_topology=requested_topology,
                requested_device=None,
            )
            hardware_family = _normalize_hardware_family(
                device_summary.topology_type
            )
            if hardware_family != requested_topology:
                raise RuntimeError(
                    "selected solver topology does not match the requested "
                    f"topology: requested={requested_topology} "
                    f"selected={hardware_family}"
                )
            solver_properties = dict(
                getattr(sampler, "properties", {}) or {}
            )
            solver_parameters = dict(
                getattr(sampler, "parameters", {}) or {}
            )
            _run_target_session(
                session_id=session_id,
                root_output_dir=root_output_dir,
                tuning_dir=tuning_dir,
                projected_summary_path=projected_summary_path,
                unbalanced_summary_path=unbalanced_summary_path,
                requested_token_env_var=str(
                    args.token_env_var
                ),
                resolved_token_env_var=resolved_token_env_var,
                token_source=token_source,
                progress_ui=str(args.progress_ui),
                dry_run=False,
                test_run=bool(args.test_run),
                num_reads=(
                    DEFAULT_TEST_NUM_READS
                    if args.test_run
                    else DEFAULT_NUM_READS
                ),
                annealing_time=annealing_time,
                qpu_time_accumulator=qpu_time_accumulator,
                sampler=sampler,
                device_summary=device_summary,
                hardware_family=hardware_family,
                hardware_graph=hardware_graph,
                solver_properties=solver_properties,
                solver_parameters=solver_parameters,
                benchmark_directories=benchmark_directories,
                problems_by_family=problems_by_family,
                initial_skipped_rows=skipped_rows,
            )
        finally:
            if sampler is not None:
                close = getattr(sampler, "close", None)
                if callable(close):
                    close()


if __name__ == "__main__":
    main()
