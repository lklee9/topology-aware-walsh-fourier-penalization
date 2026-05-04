"""Embedding-based comparison against unbalanced penalization.

This script follows the logical baseline workflow driven by
``experiments.tune_multipliers`` and
``experiments.compare_methods_baseline`` to generate and tune
penalized QUBOs, then evaluates those QUBOs using D-Wave-style embedded
sampling:

1. compare unbalanced penalization, the projected penalty on the full
   logical topology, and the projected penalty with couplings restricted
   to the currently evaluated hardware-induced logical topology;

2. fit the topology-restricted projected penalties from sampled states
   using the family-level projected combo selected during tuning and
   stored in ``experiments/tunings/projected_penalty_tuning_summary.csv``;
   equality QUBOs remain exact on the full logical topology but are
   projected on sparse logical topologies using the same sampled states
   and importance-fitting pipeline as the inequality constraints;

3. load the projected multipliers and unbalanced-penalty lambdas from
   ``experiments/tunings/projected_penalty_tuning_summary.csv`` and
   ``experiments/tunings/unbalanced_penalty_tuning_summary.csv``, then
   reuse those family-level settings across every evaluated size in this
   script;

4. for each penalized logical QUBO, embed onto Chimera, Pegasus, and
   Zephyr hardware graphs, sample with Ocean SQA on the embedded physical
   QUBO, decode with majority vote, and report CoP plus SA-style energy
   metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import dimod
import networkx as nx
import numpy as np

from experiments.experiment_config import (
    DEFAULT_MDKP_SIZES,
    DEFAULT_MIS_SIZES,
    DEFAULT_PROJECTED_STANDARDIZE,
    DEFAULT_SQA_NUM_READS,
    DEFAULT_SQA_NUM_SWEEPS,
    DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
    DEFAULT_TUNING_DIR,
)
from experiments.utils import (
    tuning_models as _shared_tuning_models,
)
from experiments.utils import (
    tuning_support as _shared_tuning_support,
)
from experiments.utils.baseline_common import (
    qubo_normalization_scale as _shared_qubo_normalization_scale,
)
from experiments.utils.baseline_common import (
    scale_qubo_coefficients as _shared_scale_qubo_coefficients,
)
from experiments.utils.cplex_reference import (
    DEFAULT_REFERENCE_ATOL,
    CplexReference,
    cplex_reference_for_synthetic_problem,
    load_cplex_reference_index,
    reference_objective_gap_ratio,
    reference_objective_hit_mask,
    resolve_cplex_reference_path,
)
from experiments.utils.driver_common import (
    binary_to_spin_states as _binary_to_spin_states,
)
from experiments.utils.driver_common import (
    build_child_rng as _rng_from_seed,
)
from experiments.utils.driver_common import (
    decoded_sampleset_bits as _decoded_sampleset_bits,
)
from experiments.utils.driver_common import (
    full_pair_edges as _full_pair_edges,
)
from experiments.utils.driver_common import (
    is_complete_pair_edge_set as _is_fully_connected_pair_edge_set,
)
from experiments.utils.driver_common import (
    num_inequality_quadratic_terms as _num_inequality_quadratic_terms,
)
from experiments.utils.driver_common import (
    problem_batch_getter as _shared_problem_batch_getter,
)
from experiments.utils.driver_common import (
    problem_provenance_fields as _problem_provenance_fields,
)
from experiments.utils.driver_common import (
    projection_regime_fields as _projection_regime_fields,
)
from experiments.utils.driver_common import (
    write_rows_csv as _write_rows_csv,
)
from experiments.utils.embedding import _qubo_arrays_to_bqm
from experiments.utils.experiment_progress import (
    ExperimentProgressReporter,
    ProgressTotals,
    build_progress_reporter,
)
from experiments.utils.family_cli import (
    add_family_selection_arguments,
    selected_families_from_args,
    selected_family_sizes_from_args,
)
from experiments.utils.fixed_sqa import (
    FixedSqaSchedule,
)
from experiments.utils.fixed_sqa import (
    anneal_schedule_json as _shared_anneal_schedule_json,
)
from experiments.utils.fixed_sqa import (
    fixed_sqa_schedule as _fixed_sqa_schedule,
)
from experiments.utils.merge_outputs import (
    ensure_run_metadata,
    merge_csv_rows,
)
from experiments.utils.problems import iter_state_chunks
from experiments.utils.projected_method_selection import (
    ProjectionComboChoice,
)
from experiments.utils.projected_method_selection import (
    projected_candidate_spec as _shared_projected_candidate_spec,
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
    build_unit_equality_constraint_qubos as _build_unit_equality_constraint_qubos,
)
from experiments.utils.projected_qubo import (
    combine_constraint_terms as _combine_constraint_terms,
)
from experiments.utils.projected_qubo import (
    projection_sample_size as _projection_sample_size,
)
from experiments.utils.projection_measure import (
    build_projection_sampling_catalog,
    canonical_projection_measure_name,
    sample_projection_states_with_inequality_support,
)
from experiments.utils.qa_simulator import (
    ClassicalAnnealerSimulation,
    build_dwave_graph,
    embedded_chain_to_problem_ratio,
    find_minor_embedding,
    qpu_anneal_schedule_to_sqa_fields,
    simulate_dwave_annealer_classically,
    sqa_dwave_annealer_samples,
)
from experiments.utils.tuning_summary import (
    LoadedProjectedPenaltySummary,
    load_projected_penalty_summaries,
    load_selected_projected_configs,
    load_unbalanced_penalty_summaries,
)
from experiments.utils.unb_pen import qubo_energy_values
from experiments.utils.unbalanced_pipeline import (
    UP_LAMBDA_GAUGE as _UP_LAMBDA_GAUGE,
)
from experiments.utils.unbalanced_pipeline import (
    UP_NORMALIZATION_REGIME as _UP_NORMALIZATION_REGIME,
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

DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "results" / "compare_embedding"
)
DEFAULT_PROJECTED_PENALTY_SUMMARY_FALLBACK_PATH = (
    DEFAULT_TUNING_DIR
    / "projected_penalty_tuning_summary.csv"
)
DEFAULT_UNBALANCED_PENALTY_SUMMARY_FALLBACK_PATH = (
    DEFAULT_TUNING_DIR
    / "unbalanced_penalty_tuning_summary.csv"
)

# Result-table method name for the projected penalty restricted to the
# currently evaluated hardware-induced logical topology.
PROJECTED_TOPOLOGY_METHOD = "projected_topology"

DEFAULT_METHODS = (
    "unbalanced",
    "projected_full",
    "projected_up_support",
    PROJECTED_TOPOLOGY_METHOD,
)
DEFAULT_SEED = 1
DEFAULT_NUM_INSTANCES = 20
DEFAULT_MEASURE_LAM = 0.01
DEFAULT_CHUNK_SIZE = 1 << 15
DEFAULT_PEGASUS_SIZE = 16
DEFAULT_PROJECTION_SAMPLE_CAP_LOG2 = 15
DEFAULT_PROJECTION_REG = 1e-8
DEFAULT_CHIMERA_SIZE = 16
DEFAULT_ZEPHYR_SIZE = 16
DEFAULT_TUNING_OBJECTIVE = "gap"
DEFAULT_PROGRESS_UI = "plain"
DEFAULT_WORKERS = 1
DEFAULT_SQA_CHAIN_STRENGTH_FRACTION = 1.0
FAMILY_ORDER = (
    "mdkp",
    "mis",
)
FAMILY_LABELS = {
    "mdkp": "MDKP",
    "mis": "MIS",
}
METHOD_LABELS = {
    "unbalanced": "Unbalanced Penalization",
    "projected_full": "Projected Penalty (Full Pairwise)",
    "projected_up_support": "Projected Penalty (UP Support)",
    PROJECTED_TOPOLOGY_METHOD: "Projected Penalty (Current Hardware Topology)",
}
FAMILY_CODES = {
    "mdkp": 3,
    "mis": 4,
}
METHOD_CODES = {
    "unbalanced": 1,
    "projected_full": 2,
    "projected_up_support": 3,
    PROJECTED_TOPOLOGY_METHOD: 4,
}
HARDWARE_FAMILIES = ("chimera", "pegasus", "zephyr")
HARDWARE_CODES = {"chimera": 1, "pegasus": 2, "zephyr": 3}
PROGRESS_UI_CHOICES = ("plain", "tui", "rich")

_SQA_WARNING_EMITTED = False
_ANNEALER_SEED_DOMAIN = 4_000
_EMBEDDING_SEED_DOMAIN = 4_100


def _stable_seed_component(label: object) -> int:
    """Return one stable integer seed component for a text label."""
    digest = hashlib.blake2b(
        str(label).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(
        digest, byteorder="little", signed=False
    )


def _problem_seed(problem: BLP) -> int:
    """Return the stable manifest/problem seed attached to one instance."""
    metadata = dict(getattr(problem, "metadata", {}) or {})
    if metadata.get("problem_seed") is None:
        raise ValueError(
            "problem metadata is missing 'problem_seed'; "
            "cannot align annealer randomness across experiments"
        )
    return int(metadata["problem_seed"])


def _canonical_annealer_method(
    method: str,
    *,
    hardware_family: str | None = None,
) -> str:
    """Return the cross-experiment method label used for annealer seeding."""
    normalized_method = str(method)
    if normalized_method != PROJECTED_TOPOLOGY_METHOD:
        return normalized_method
    if hardware_family is None:
        raise ValueError(
            "hardware_family is required when canonicalizing projected_topology"
        )
    return _hardware_projected_summary_method(
        hardware_family
    )


def _annealer_schedule_seed_label(
    *,
    schedule: FixedSqaSchedule,
    num_points: int,
    num_sweeps_per_beta: int,
) -> str:
    """Return the canonical schedule label used for annealer seeding."""
    return (
        f"{schedule.schedule_kind}:{schedule.schedule_id}:"
        f"beta={float(schedule.beta_scale):.12g}:"
        f"points={int(num_points)}:spb={int(num_sweeps_per_beta)}"
    )


def _shared_annealer_rng(
    *,
    base_seed: int,
    problem_seed: int,
    family: str,
    size: int,
    canonical_method: str,
    schedule: FixedSqaSchedule,
    num_points: int,
    num_sweeps_per_beta: int,
) -> np.random.Generator:
    """Return the annealer RNG shared with the logical baseline."""
    return _rng_from_seed(
        base_seed,
        _ANNEALER_SEED_DOMAIN,
        int(problem_seed),
        FAMILY_CODES[family],
        int(size),
        _stable_seed_component(canonical_method),
        _stable_seed_component(
            _annealer_schedule_seed_label(
                schedule=schedule,
                num_points=num_points,
                num_sweeps_per_beta=num_sweeps_per_beta,
            )
        ),
    )


def _embedding_search_seed(
    *,
    base_seed: int,
    problem_seed: int,
    family: str,
    size: int,
    canonical_method: str,
    hardware_family: str,
    hardware_size: int,
) -> int:
    """Return one embedding-search seed decoupled from annealer randomness."""
    return int(
        np.random.SeedSequence(
            [
                int(base_seed),
                _EMBEDDING_SEED_DOMAIN,
                int(problem_seed),
                FAMILY_CODES[family],
                int(size),
                _stable_seed_component(canonical_method),
                HARDWARE_CODES[hardware_family],
                int(hardware_size),
            ]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def _is_singleton_embedding(
    embedding: dict[int, list[Any]] | None,
) -> bool:
    """Return whether every logical variable is mapped to one physical qubit."""
    if not embedding:
        return False
    return all(
        len(chain) == 1 for chain in embedding.values()
    )


def _singleton_embedding_simulation(
    *,
    logical_bqm: dimod.BinaryQuadraticModel,
    embedding: dict[int, list[Any]],
    rng: np.random.Generator,
    request: "AnnealerRunRequest",
    hp_field: np.ndarray,
    hd_field: np.ndarray,
) -> ClassicalAnnealerSimulation:
    """Sample one singleton embedding via the logical SQA path."""
    logical_sampleset = sqa_dwave_annealer_samples(
        logical_bqm,
        num_reads=request.num_reads,
        rng=rng,
        num_sweeps=int(hp_field.size),
        num_sweeps_per_beta=request.num_sweeps_per_beta,
        hp_field=hp_field,
        hd_field=hd_field,
    )
    relabel_map = {
        variable: chain[0]
        for variable, chain in embedding.items()
    }
    physical_bqm = logical_bqm.relabel_variables(
        relabel_map,
        inplace=False,
    )
    physical_samples = np.asarray(
        logical_sampleset.record.sample,
        dtype=np.int8,
    )
    physical_variables = [
        relabel_map[variable]
        for variable in logical_sampleset.variables
    ]
    physical_sampleset = dimod.SampleSet.from_samples_bqm(
        (physical_samples, physical_variables),
        physical_bqm,
        num_occurrences=np.asarray(
            logical_sampleset.record.num_occurrences,
            dtype=np.int64,
        ),
        sort_labels=False,
    )
    return ClassicalAnnealerSimulation(
        sampler_name="sqa",
        decoder_name="majority_vote",
        embedding=embedding,
        effective_chain_strength=0.0,
        physical_bqm=physical_bqm,
        physical_sampleset=physical_sampleset,
        decoded_sampleset=logical_sampleset,
        chain_break_fraction=np.zeros(
            len(physical_sampleset), dtype=float
        ),
    )


def _chain_strength_fraction_dirname(
    fraction: float,
) -> str:
    """Return a filesystem-safe result-directory suffix."""
    label = f"{float(fraction):g}".replace(
        "-", "minus_"
    ).replace(".", "p")
    return f"sqa_chain_strength_fraction_{label}"


@dataclass(frozen=True)
class EmbeddedSQAMetrics:
    """Summary metrics for one embedded SQA run on a penalized QUBO."""

    mean_sqa_energy_gap: float
    sqa_optimum_rate: float
    optimum_probability: float
    coefficient_of_performance: float
    best_feasible_objective: float | None
    best_feasible_gap: float | None
    objective_gap: float
    num_feasible_reads: int
    feasible_read_fraction: float
    physical_qubits: int | None
    mean_chain_length: float | None
    max_chain_length: int | None
    mean_chain_break_fraction: float | None
    broken_read_rate: float | None
    effective_chain_strength: float | None
    embedded_chain_to_problem_ratio: float | None


@dataclass(frozen=True)
class ScaledAnnealerInput:
    """Scaled logical-QUBO data used by embedded SQA."""

    quadratic: np.ndarray
    linear: np.ndarray
    const: float


@dataclass(frozen=True)
class AnnealerRunRequest:
    """All solver inputs needed for one embedded D-Wave-style anneal."""

    hardware_graph: Any
    hardware_family: str
    hardware_size: int
    family: str
    size: int
    instance_index: int
    problem_seed: int
    method: str
    base_seed: int
    num_reads: int
    num_points: int
    num_sweeps_per_beta: int
    schedule: FixedSqaSchedule
    chain_strength_fraction: float
    fixed_embedding: dict[int, list[Any]] | None = None


class DWaveAnnealingBackend:
    """Backend interface for embedded D-Wave-style annealing."""

    backend_name = "unknown"

    def sample(
        self,
        *,
        quadratic: np.ndarray,
        linear: np.ndarray,
        const: float,
        request: AnnealerRunRequest,
    ) -> ClassicalAnnealerSimulation:
        """Run one penalized QUBO through the configured backend."""
        raise NotImplementedError


class SimulatedDWaveAnnealingBackend(DWaveAnnealingBackend):
    """Current embedded Ocean-SQA backend used by this experiment."""

    backend_name = "simulated_dwave_sqa"

    def sample(
        self,
        *,
        quadratic: np.ndarray,
        linear: np.ndarray,
        const: float,
        request: AnnealerRunRequest,
    ) -> ClassicalAnnealerSimulation:
        """Run one embedded D-Wave-style anneal via the classical simulator."""
        logical_bqm = _qubo_arrays_to_bqm(
            quadratic, linear, const
        )
        canonical_method = _canonical_annealer_method(
            request.method,
            hardware_family=request.hardware_family,
        )
        annealer_rng = _shared_annealer_rng(
            base_seed=request.base_seed,
            problem_seed=request.problem_seed,
            family=request.family,
            size=request.size,
            canonical_method=canonical_method,
            schedule=request.schedule,
            num_points=request.num_points,
            num_sweeps_per_beta=request.num_sweeps_per_beta,
        )
        embedding = request.fixed_embedding
        if embedding is None:
            embedding_seed = _embedding_search_seed(
                base_seed=request.base_seed,
                problem_seed=request.problem_seed,
                family=request.family,
                size=request.size,
                canonical_method=canonical_method,
                hardware_family=request.hardware_family,
                hardware_size=request.hardware_size,
            )
            _set_progress_status(
                stage="method",
                activity="embedded SQA",
                detail=(
                    f"finding minor embedding on {request.hardware_family}"
                    f"({request.hardware_size})"
                ),
            )
            embedding = find_minor_embedding(
                logical_bqm,
                request.hardware_graph,
                random_seed=embedding_seed,
            )
        else:
            _assert_embedding_matches_bqm(
                logical_bqm,
                request.hardware_graph,
                embedding,
            )
            _set_progress_status(
                stage="method",
                activity="embedded SQA",
                detail=(
                    "reusing topology-induced injective placement as "
                    "fixed embedding"
                ),
            )
        _set_progress_status(
            stage="method",
            activity="embedded SQA",
            detail=(
                f"discretizing anneal schedule {request.schedule.schedule_id}"
            ),
        )
        hp_field, hd_field = (
            qpu_anneal_schedule_to_sqa_fields(
                request.schedule.anneal_schedule,
                request.schedule.beta_scale,
                num_points=request.num_points,
            )
        )
        _set_progress_status(
            stage="method",
            activity="embedded SQA",
            detail=(
                f"running SQA: {request.num_reads} reads, "
                f"{int(hp_field.size)} sweeps"
            ),
        )
        if _is_singleton_embedding(embedding):
            _set_progress_status(
                stage="method",
                activity="embedded SQA",
                detail=(
                    "singleton embedding detected; reusing matched logical "
                    "SQA randomness"
                ),
            )
            return _singleton_embedding_simulation(
                logical_bqm=logical_bqm,
                embedding=embedding,
                rng=annealer_rng,
                request=request,
                hp_field=hp_field,
                hd_field=hd_field,
            )
        return simulate_dwave_annealer_classically(
            logical_bqm=logical_bqm,
            hardware_graph=request.hardware_graph,
            chain_strength_multiplier=request.chain_strength_fraction,
            num_reads=request.num_reads,
            rng=annealer_rng,
            embedding=embedding,
            sampler_name="sqa",
            decoder_name="majority_vote",
            num_sweeps=int(hp_field.size),
            num_sweeps_per_beta=request.num_sweeps_per_beta,
            hp_field=hp_field,
            hd_field=hd_field,
        )


@dataclass(frozen=True)
class TunedProjectedMultipliers:
    """Family-level projected-penalty multipliers selected on the anchor
    size."""

    method: str
    family: str
    anchor_size: int
    equality_multiplier: float
    inequality_multiplier: float
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "family": self.family,
            "anchor_size": self.anchor_size,
            "equality_multiplier": self.equality_multiplier,
            "inequality_multiplier": self.inequality_multiplier,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class SelectedProjectedConfig:
    """One projected-method combo selected for downstream evaluation."""

    method: str
    family: str
    measure_name: str
    penalty_template: str
    selection_mode: str
    selection_source: str
    candidate_rank: int
    projected_standardize: bool
    tuning: TunedProjectedMultipliers


@dataclass(frozen=True)
class TunedUnbalancedParameters:
    """Family-level unbalanced-penalty parameters selected on the anchor
    size."""

    family: str
    anchor_size: int
    lambda0: float | None
    lambda1: float
    lambda2: float
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str

    def as_row(self) -> dict[str, object]:
        return {
            "family": self.family,
            "anchor_size": self.anchor_size,
            "lambda0": self.lambda0,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


_ACTIVE_PROGRESS: ExperimentProgressReporter | None = None


def _resolve_existing_csv_path(*candidates: Path) -> Path:
    """Return the first existing CSV path from a priority-ordered list."""
    for path in candidates:
        resolved = (
            path
            if path.is_absolute()
            else (REPO_ROOT / path)
        )
        if resolved.exists():
            return resolved.resolve()
    default_path = candidates[0]
    if default_path.is_absolute():
        return default_path.resolve()
    return (REPO_ROOT / default_path).resolve()


def _resolve_projected_penalty_summary_path() -> Path:
    """Resolve the projected-penalty summary used as the primary source."""
    return _resolve_existing_csv_path(
        DEFAULT_PROJECTED_PENALTY_SUMMARY_FALLBACK_PATH,
        Path(
            "results/unbalanced_penalization/projected_penalty_tuning_summary.csv"
        ),
    )


def _resolve_unbalanced_penalty_summary_path() -> Path:
    """Resolve the UP summary used as the primary source."""
    return _resolve_existing_csv_path(
        DEFAULT_UNBALANCED_PENALTY_SUMMARY_FALLBACK_PATH,
        Path(
            "results/unbalanced_penalization/unbalanced_penalty_tuning_summary.csv"
        ),
    )


def _build_compare_embedding_progress(
    *,
    mode: str,
    totals: ProgressTotals,
    worker_count: int,
) -> ExperimentProgressReporter | None:
    """Build the shared progress reporter used by this experiment."""
    reporter_mode = "rich" if mode == "tui" else mode
    return build_progress_reporter(
        mode=reporter_mode,
        totals=totals,
        worker_count=worker_count,
        stream=sys.stderr,
    )


def _set_progress_status(**kwargs: object) -> None:
    """Update the active progress reporter, if any."""
    if _ACTIVE_PROGRESS is None:
        return
    activity = kwargs.pop("activity", None)
    method = kwargs.pop("method", None)
    detail = kwargs.pop("detail", None)
    status_detail = detail
    if activity is not None:
        status_detail = (
            str(activity)
            if status_detail is None
            else f"{activity}; {status_detail}"
        )
    _ACTIVE_PROGRESS.update_status(
        **kwargs,
        measure=method,
        detail=status_detail,
    )


def _advance_progress_counter(
    counter: str, amount: int = 1
) -> None:
    """Advance one aggregate progress counter."""
    if _ACTIVE_PROGRESS is None:
        return
    if counter == "instances":
        _ACTIVE_PROGRESS.advance_instances(amount)
        return
    if counter == "topologies":
        _ACTIVE_PROGRESS.advance_topologies(amount)
        return
    if counter == "methods":
        _ACTIVE_PROGRESS.advance_measures(amount)
        return
    raise ValueError(
        f"unknown compare_methods_embedding progress counter: {counter}"
    )


def _log(message: str) -> None:
    """Emit one stable progress log line."""
    print(
        f"[compare_methods_embedding] {message}",
        file=sys.stderr,
        flush=True,
    )


def _state_chunk_total(
    problem: BLP, chunk_size: int
) -> int:
    """Return the number of hypercube chunks for one exact scan."""
    return max(
        1,
        math.ceil(
            int(problem.num_states) / int(chunk_size)
        ),
    )


def _should_report_subprogress(
    completed: int,
    total: int,
    *,
    target_updates: int = 10,
) -> bool:
    """Return whether one inner-loop progress point should be surfaced."""
    if total <= 0:
        return True
    if completed <= 1 or completed >= total:
        return True
    stride = max(
        1, int(math.ceil(total / max(1, target_updates)))
    )
    return completed % stride == 0


def _progress_detail(
    label: str,
    *,
    completed: int,
    total: int,
    unit: str,
    extra: str | None = None,
) -> str:
    """Return one compact human-facing bottleneck progress string."""
    bounded_total = max(1, int(total))
    bounded_completed = min(
        max(0, int(completed)), bounded_total
    )
    percent = (
        100.0
        * float(bounded_completed)
        / float(bounded_total)
    )
    detail = (
        f"{label}: {bounded_completed}/{bounded_total} {unit} "
        f"({percent:0.0f}%)"
    )
    if extra is not None:
        return f"{detail}; {extra}"
    return detail


def _report_state_chunk_progress(
    label: str,
    *,
    problem: BLP,
    stop: int,
    chunk_index: int,
    total_chunks: int,
    extra: str | None = None,
) -> None:
    """Update the active reporter for one exact chunked scan."""
    completed_chunks = int(chunk_index) + 1
    if not _should_report_subprogress(
        completed_chunks, total_chunks
    ):
        return
    _set_progress_status(
        detail=_progress_detail(
            label,
            completed=min(
                int(stop), int(problem.num_states)
            ),
            total=int(problem.num_states),
            unit="states",
            extra=(
                f"chunk {completed_chunks}/{total_chunks}"
                if extra is None
                else (
                    f"chunk {completed_chunks}/{total_chunks}; {extra}"
                )
            ),
        )
    )


# D-Wave SA dependencies imported at module top-level (see above).


def _warn_sqa_unavailable(message: str) -> None:
    """Log one SQA warning while letting the experiment continue."""
    global _SQA_WARNING_EMITTED
    if _SQA_WARNING_EMITTED:
        return
    _log(f"warning: {message}")
    _SQA_WARNING_EMITTED = True


def _hardware_projected_summary_method(
    hardware_family: str,
) -> str:
    """Return the summary-row method used for one hardware family."""
    methods = {
        "chimera": "projected_chimera",
        "pegasus": "projected_pegasus",
        "zephyr": "projected_zephyr",
    }
    try:
        return methods[str(hardware_family)]
    except KeyError as exc:
        raise ValueError(
            f"unknown hardware family for projected summary lookup: {hardware_family}"
        ) from exc


def _proxy_projection_selection_method(method: str) -> str:
    """Return the exact summary label used for one projected method.

    The tuning summary stores one selected combo per downstream method,
    including the quadratic variants, so keep the requested label intact.
    """
    return str(method)


def _projected_components_cache_key(
    *,
    projection_method: str,
    family: str,
    size: int,
    instance_index: int,
    measure_name: str,
    measure_lam: float,
    penalty_template: str,
    penalty_template_kwargs: dict[str, float] | None,
    standardize: bool,
    deployment_topology: str | None = None,
    deployment_topology_size: int | None = None,
) -> tuple[object, ...]:
    """Return the cache key for one sampled projected-penalty fit."""
    return (
        projection_method,
        family,
        int(size),
        int(instance_index),
        canonical_projection_measure_name(
            measure_name,
            legacy_default_lam=measure_lam,
        ),
        penalty_template,
        tuple(
            sorted(
                (
                    str(key),
                    float(value),
                )
                for key, value in (
                    penalty_template_kwargs or {}
                ).items()
            )
        ),
        bool(standardize),
        (
            None
            if deployment_topology is None
            else str(deployment_topology)
        ),
        (
            None
            if deployment_topology_size is None
            else int(deployment_topology_size)
        ),
    )


def _selected_projected_config(
    method: str,
    *,
    family: str,
    candidate: ProjectionComboChoice,
    selection_mode: str,
    tuned: TunedProjectedMultipliers,
    projection_method: str | None = None,
    penalty_template_kwargs: dict[str, float] | None = None,
    projected_standardize: bool = DEFAULT_PROJECTED_STANDARDIZE,
) -> SelectedProjectedConfig:
    """Return the frozen projected config used after outer selection."""
    return _shared_projected_candidate_spec(
        method,
        family=family,
        candidate=candidate,
        selection_mode=selection_mode,
        tuned=tuned,
        projection_method=projection_method,
        penalty_template_kwargs=penalty_template_kwargs,
        projected_standardize=bool(projected_standardize),
    )


def _summary_family_variants(
    family: str,
) -> tuple[str, ...]:
    """Return canonical and legacy family labels used in summary artifacts."""
    if family == "mdkp":
        return ("mdkp", "kp")
    if family == "mis":
        return ("mis", "bpp")
    return (family,)


def _load_projected_combo_from_records(
    summary_path: Path,
    *,
    family: str,
    method: str,
    anchor_size: int,
) -> (
    tuple[str, str, str | None, str | None, int | None]
    | None
):
    """Recover one projected combo from sibling tuning-record CSVs."""
    records_dir = (
        summary_path.parent / "projected_tuning_records"
    )
    if not records_dir.exists():
        return None

    combos: dict[
        tuple[str, str],
        tuple[str | None, str | None, int | None],
    ] = {}
    for family_name in _summary_family_variants(family):
        record_path = (
            records_dir
            / f"{family_name}_{method}_anchor_{anchor_size}.csv"
        )
        if not record_path.exists():
            continue
        with record_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                measure_name = str(
                    row.get("projection_measure", "")
                ).strip()
                penalty_template = str(
                    row.get(
                        "projection_penalty_template", ""
                    )
                ).strip()
                if not measure_name or not penalty_template:
                    continue
                combos[(measure_name, penalty_template)] = (
                    (
                        None
                        if not str(
                            row.get(
                                "projection_selection_mode",
                                "",
                            )
                        ).strip()
                        else str(
                            row["projection_selection_mode"]
                        ).strip()
                    ),
                    (
                        None
                        if not str(
                            row.get(
                                "projection_selection_source",
                                "",
                            )
                        ).strip()
                        else str(
                            row[
                                "projection_selection_source"
                            ]
                        ).strip()
                    ),
                    (
                        None
                        if not str(
                            row.get(
                                "projection_candidate_rank",
                                "",
                            )
                        ).strip()
                        else int(
                            row["projection_candidate_rank"]
                        )
                    ),
                )
    if not combos:
        return None
    if len(combos) != 1:
        raise ValueError(
            "projected tuning records define multiple projection combos for "
            f"{family}/{method} under {summary_path.parent}"
        )
    (measure_name, penalty_template), metadata = next(
        iter(combos.items())
    )
    selection_mode, selection_source, candidate_rank = (
        metadata
    )
    return (
        measure_name,
        penalty_template,
        selection_mode,
        selection_source,
        candidate_rank,
    )


def _summary_projected_combo_candidate(
    summary: LoadedProjectedPenaltySummary,
    *,
    summary_path: Path,
    family: str,
    method: str,
) -> tuple[ProjectionComboChoice, str] | None:
    """Return the globally selected projected combo when the summary stores it."""
    measure_name = summary.projection_measure
    penalty_template = summary.projection_penalty_template
    selection_mode = summary.projection_selection_mode
    selection_source = summary.projection_selection_source
    candidate_rank = summary.projection_candidate_rank
    if measure_name is None or penalty_template is None:
        recovered = _load_projected_combo_from_records(
            summary_path,
            family=family,
            method=method,
            anchor_size=summary.anchor_size,
        )
        if recovered is None:
            return None
        (
            measure_name,
            penalty_template,
            recovered_mode,
            recovered_source,
            recovered_rank,
        ) = recovered
        if selection_mode is None:
            selection_mode = recovered_mode
        if selection_source is None:
            selection_source = recovered_source
        if candidate_rank is None:
            candidate_rank = recovered_rank
    return (
        ProjectionComboChoice(
            family=family,
            topology=projected_method_topology(
                _proxy_projection_selection_method(method)
            ),
            penalty_template=penalty_template,
            measure_name=measure_name,
            source=selection_source or "global_summary",
            candidate_rank=(
                1
                if candidate_rank is None
                else int(candidate_rank)
            ),
        ),
        selection_mode or "global_summary",
    )


def _required_summary_projected_candidate(
    projected_summaries: dict[
        tuple[str, str],
        LoadedProjectedPenaltySummary,
    ],
    *,
    summary_path: Path,
    family: str,
    method: str,
) -> tuple[
    LoadedProjectedPenaltySummary,
    ProjectionComboChoice,
    str,
    str,
]:
    """Load the single projected combo/multiplier source mandated by the summaries."""
    summary_method = _proxy_projection_selection_method(
        method
    )
    try:
        summary = projected_summaries[
            (summary_method, family)
        ]
    except KeyError as exc:
        raise RuntimeError(
            "missing global projected penalty summary for "
            f"{summary_method}/{family} in {summary_path}"
        ) from exc

    summary_combo = _summary_projected_combo_candidate(
        summary,
        summary_path=summary_path,
        family=family,
        method=summary_method,
    )
    if summary_combo is None:
        raise RuntimeError(
            "global projected penalty summary does not record the selected "
            f"projection combo for {summary_method}/{family}: {summary_path}"
        )
    candidate, selection_mode = summary_combo
    return (
        summary,
        candidate,
        selection_mode,
        summary_method,
    )


def _loaded_projected_summary_row(
    config: SelectedProjectedConfig,
    *,
    size: int,
    source_path: Path,
    projection_method_name: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Serialize one size-specific projected config into the summary format."""
    row = config.tuning.as_row()
    row.update(
        {
            "size": int(size),
            "penalty_config_source": str(source_path),
            "projection_summary_method": config.method,
            "projection_method": projection_method_name,
            "projection_selection_mode": config.selection_mode,
            "projection_selection_source": config.selection_source,
            "projection_candidate_rank": int(
                config.candidate_rank
            ),
            "projection_measure": config.measure_name,
            "projection_penalty_template": config.penalty_template,
            "selected_projection_combo": True,
            "projected_standardize": config.projected_standardize,
        }
    )
    for (
        key,
        value,
    ) in config.penalty_template_kwargs.items():
        row[f"projection_penalty_template_{key}"] = float(
            value
        )
    if extra_fields is not None:
        row.update(extra_fields)
    return row


def _load_penalty_configs_from_summaries(
    *,
    projected_summary_path: Path,
    unbalanced_summary_path: Path,
    families: tuple[str, ...],
    family_sizes: dict[str, list[int]],
    projected_methods: tuple[str, ...],
) -> tuple[
    dict[tuple[str, int], TunedUnbalancedParameters],
    dict[tuple[str, int, str], SelectedProjectedConfig],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Load embedding configs from the global summary CSVs.

    The global projected and UP summaries are the only source of truth.
    """
    projected_summaries = load_projected_penalty_summaries(
        projected_summary_path
    )
    selected_projected_configs = (
        load_selected_projected_configs(
            projected_summary_path
        )
    )
    unbalanced_summaries = (
        load_unbalanced_penalty_summaries(
            unbalanced_summary_path
        )
    )

    up_configs: dict[
        tuple[str, int], TunedUnbalancedParameters
    ] = {}
    projected_configs: dict[
        tuple[str, int, str],
        SelectedProjectedConfig,
    ] = {}
    projected_rows: list[dict[str, object]] = []
    up_rows: list[dict[str, object]] = []

    for family in families:
        try:
            up_summary = unbalanced_summaries[family]
        except KeyError as exc:
            raise RuntimeError(
                "missing unbalanced penalty summary for "
                f"{family} in {unbalanced_summary_path}; regenerate tuning "
                "artifacts for that family before running this command"
            ) from exc

        for size in family_sizes[family]:
            tuned_up = _shared_tuning_models.TunedUnbalancedParameters(
                family=family,
                anchor_size=up_summary.anchor_size,
                up_equality_multiplier=up_summary.up_equality_multiplier,
                up_inequality_multiplier=up_summary.up_inequality_multiplier,
                up_lambda1_shape=up_summary.up_lambda1_shape,
                up_lambda2_shape=up_summary.up_lambda2_shape,
                up_lambda_gauge=(
                    up_summary.up_lambda_gauge
                    or _UP_LAMBDA_GAUGE
                ),
                normalization_regime=up_summary.normalization_regime,
                per_constraint_standardization=(
                    up_summary.per_constraint_standardization
                ),
                global_multiplier=up_summary.global_multiplier,
                lambda0=up_summary.lambda0,
                lambda1=up_summary.lambda1,
                lambda2=up_summary.lambda2,
                base_parameter_source=up_summary.base_parameter_source,
                tuning_objective=up_summary.tuning_objective,
                objective_value=up_summary.objective_value,
                success=up_summary.success,
                status=up_summary.status,
                message=f"loaded from {unbalanced_summary_path}: {up_summary.message}",
            )
            up_configs[(family, size)] = tuned_up
            up_row = tuned_up.as_row()
            up_row.update(
                {
                    "size": int(size),
                    "penalty_config_source": str(
                        unbalanced_summary_path
                    ),
                    "unbalanced_penalty_tuning_summary": str(
                        unbalanced_summary_path
                    ),
                }
            )
            up_rows.append(up_row)

        for method in projected_methods:
            try:
                (
                    summary,
                    candidate,
                    selection_mode,
                    projection_method_name,
                ) = _required_summary_projected_candidate(
                    projected_summaries,
                    summary_path=projected_summary_path,
                    family=family,
                    method=method,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "missing projected penalty summary-backed config for "
                    f"{family}/{method} in {projected_summary_path}; regenerate "
                    "tuning artifacts for that family before running this command"
                ) from exc

            selected_config = (
                selected_projected_configs.get(
                    (family, method)
                )
            )
            projection_method = (
                summary.projection_method
                or projection_method_name
            )
            penalty_template_kwargs: dict[str, float] = {}
            if selected_config is not None:
                candidate = ProjectionComboChoice(
                    family=family,
                    topology=projected_method_topology(
                        method
                    ),
                    penalty_template=selected_config.penalty_template,
                    measure_name=selected_config.measure_name,
                    source=selected_config.selection_source,
                    candidate_rank=int(
                        selected_config.candidate_rank
                    ),
                )
                selection_mode = (
                    selected_config.selection_mode
                )
                projection_method = (
                    selected_config.projection_method
                    or projection_method
                )
                penalty_template_kwargs = dict(
                    selected_config.penalty_template_kwargs
                )

            tuned = _shared_tuning_models.TunedProjectedMultipliers(
                method=method,
                family=family,
                anchor_size=summary.anchor_size,
                equality_multiplier=summary.equality_multiplier,
                inequality_multiplier=summary.inequality_multiplier,
                tuning_objective=summary.tuning_objective,
                objective_value=summary.objective_value,
                success=summary.success,
                status=summary.status,
                message=(
                    f"loaded from {projected_summary_path}; "
                    f"source_message={summary.message}"
                ),
            )
            config = _selected_projected_config(
                method,
                family=family,
                candidate=candidate,
                selection_mode=selection_mode,
                tuned=tuned,
                projection_method=projection_method,
                penalty_template_kwargs=penalty_template_kwargs,
                projected_standardize=DEFAULT_PROJECTED_STANDARDIZE,
            )
            for size in family_sizes[family]:
                projected_configs[
                    (family, size, method)
                ] = config
                projected_rows.append(
                    _loaded_projected_summary_row(
                        config,
                        size=size,
                        source_path=projected_summary_path,
                        projection_method_name=projection_method,
                        extra_fields={
                            "projected_penalty_tuning_summary": str(
                                projected_summary_path
                            )
                        },
                    )
                )

    return (
        up_configs,
        projected_configs,
        projected_rows,
        up_rows,
    )


def _projection_pair_edges(
    problem: BLP,
    *,
    projection_method: str,
    pegasus_size: int,
    projection_hardware_graph: nx.Graph | None = None,
) -> list[tuple[int, int]]:
    """Return the admissible quadratic pairs for one projected-method variant.

    ``projected_pegasus`` must use the currently evaluated hardware graph in
    this experiment, otherwise the fitted penalty would target the wrong
    topology.
    """
    pair_edges, _ = _projection_topology_details(
        problem,
        projection_method=projection_method,
        projection_hardware_graph=projection_hardware_graph,
    )
    return pair_edges


def _projection_topology_details(
    problem: BLP,
    *,
    projection_method: str,
    projection_hardware_graph: nx.Graph | None = None,
) -> tuple[
    list[tuple[int, int]], dict[int, list[Any]] | None
]:
    """Return the projected logical couplers plus any fixed injective embedding."""
    normalized_method = (
        str(projection_method).strip().lower()
    )
    if normalized_method == "projected_full":
        return _full_pair_edges(problem.num_variables), None
    if normalized_method in {
        "projected_pegasus",
        "projected_chimera",
        "projected_zephyr",
    }:
        if projection_hardware_graph is None:
            raise ValueError(
                "projection_hardware_graph is required for topology-aware projected methods "
                "in compare_methods_embedding"
            )
        placement, topology = (
            mapped_logical_topology_from_graph(
                problem.constraint_matrix,
                projection_hardware_graph,
                logical_vertices=range(
                    problem.num_variables
                ),
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
                sorted(
                    {tuple(sorted(edge)) for edge in edges}
                ),
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
        if canonical_edges:
            return sorted(canonical_edges), fixed_embedding
        raise ValueError(
            f"{projection_method} logical topology could not be canonicalized"
        )
    raise ValueError(
        f"unknown projected method: {projection_method}"
    )


def _assert_embedding_matches_bqm(
    bqm: Any,
    hardware_graph: nx.Graph,
    embedding: dict[int, list[Any]] | None,
) -> None:
    """Reject fixed embeddings when the BQM adds edges outside the mapped topology."""
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


def _projected_component_energies(
    problem: BLP,
    components: ProjectedPenaltyComponents,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact energy tables for the combined projected penalties."""
    equality = np.zeros(problem.num_states, dtype=float)
    inequality = np.zeros(problem.num_states, dtype=float)
    if problem.num_equalities:
        equality = _qubo_energies(
            problem,
            quadratic=components.equality_terms.quadratic,
            linear=components.equality_terms.linear,
            const=components.equality_terms.const,
            chunk_size=chunk_size,
        )
    if problem.num_inequalities:
        inequality = _qubo_energies(
            problem,
            quadratic=components.inequality_terms.quadratic,
            linear=components.inequality_terms.linear,
            const=components.inequality_terms.const,
            chunk_size=chunk_size,
        )
    return equality, inequality


def _build_projected_components(
    problem: BLP,
    *,
    projection_method: str,
    family: str,
    size: int,
    instance_index: int,
    base_seed: int,
    measure_name: str,
    measure_lam: float,
    penalty_template: str,
    penalty_template_kwargs: dict[str, float] | None,
    pegasus_size: int,
    sample_cap_log2: int,
    chunk_size: int,
    reg: float,
    standardize: bool,
    projection_hardware_graph: nx.Graph | None = None,
    pair_edges: list[tuple[int, int]] | None = None,
) -> ProjectedPenaltyComponents:
    """Construct projected-method pieces with optional per-constraint standardization."""
    del chunk_size

    sample_size = _projection_sample_size(
        problem, sample_cap_log2
    )
    _set_progress_status(
        stage="components",
        activity="building projected components",
        detail=(
            f"{projection_method}: sampling {sample_size} projection states "
            f"with measure={measure_name}"
        ),
    )
    sample_rng = _rng_from_seed(
        base_seed,
        2_000,
        FAMILY_CODES[family],
        size,
        instance_index,
    )
    _set_progress_status(
        stage="components",
        activity="building projected components",
        detail=(
            f"{projection_method}: mapping logical couplers onto deployment "
            "topology"
        ),
    )
    if pair_edges is None:
        pair_edges = _projection_pair_edges(
            problem,
            projection_method=projection_method,
            pegasus_size=pegasus_size,
            projection_hardware_graph=projection_hardware_graph,
        )
    return _shared_build_projected_components(
        problem,
        pair_edges=pair_edges,
        sample_size=sample_size,
        sample_rng=sample_rng,
        measure_name=measure_name,
        measure_lam=measure_lam,
        penalty_template=penalty_template,
        penalty_template_kwargs=penalty_template_kwargs,
        reg=reg,
        standardize=standardize,
        build_projection_sampling_catalog=build_projection_sampling_catalog,
        sample_projection_states_with_inequality_support=(
            sample_projection_states_with_inequality_support
        ),
        build_unit_equality_constraint_qubos=(
            _build_unit_equality_constraint_qubos
        ),
        combine_constraint_terms=_combine_constraint_terms,
        project_penalty_values_importance=(
            project_penalty_values_importance
        ),
        hardware_topology_cls=HardwareTopology,
        ideal_penalty_cls=IdealPenalty,
        binary_to_spin_states=_binary_to_spin_states,
        is_complete_pair_edge_set=_is_fully_connected_pair_edge_set,
    )


def _objective_energies(
    problem: BLP,
    *,
    chunk_size: int,
    progress_label: str | None = None,
) -> np.ndarray:
    """Enumerate the raw constrained objective over the whole hypercube."""
    values = np.empty(problem.num_states, dtype=float)
    total_chunks = _state_chunk_total(problem, chunk_size)
    for chunk_index, (start, bitstrings) in enumerate(
        iter_state_chunks(
            problem.num_variables,
            chunk_size=chunk_size,
            dtype=float,
        )
    ):
        stop = start + bitstrings.shape[0]
        values[start:stop] = problem.objective_values(
            bitstrings
        )
        if progress_label is not None:
            _report_state_chunk_progress(
                progress_label,
                problem=problem,
                stop=stop,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
    return values


def _optimum_states(
    problem: BLP,
    *,
    chunk_size: int,
    atol: float = 1e-9,
    progress_label: str | None = None,
) -> tuple[float, np.ndarray]:
    """Return the constrained optimum objective value and all optimal states."""
    best_value = np.inf
    optimum_chunks: list[np.ndarray] = []
    total_chunks = _state_chunk_total(problem, chunk_size)

    for chunk_index, (start, bitstrings) in enumerate(
        iter_state_chunks(
            problem.num_variables,
            chunk_size=chunk_size,
            dtype=float,
        )
    ):
        feasible = problem.feasible_mask(
            bitstrings, atol=atol
        )
        if np.any(feasible):
            objective_values = problem.objective_values(
                bitstrings
            )
            feasible_indices = np.flatnonzero(feasible)
            feasible_values = objective_values[
                feasible_indices
            ]
            chunk_best = float(np.min(feasible_values))

            if chunk_best < best_value - atol:
                best_value = chunk_best
                local = feasible_indices[
                    np.isclose(
                        feasible_values,
                        chunk_best,
                        atol=atol,
                        rtol=0.0,
                    )
                ]
                optimum_chunks = [
                    start + local.astype(np.int64)
                ]
            elif np.isclose(
                chunk_best, best_value, atol=atol, rtol=0.0
            ):
                local = feasible_indices[
                    np.isclose(
                        feasible_values,
                        best_value,
                        atol=atol,
                        rtol=0.0,
                    )
                ]
                optimum_chunks.append(
                    start + local.astype(np.int64)
                )

        if progress_label is not None:
            _report_state_chunk_progress(
                progress_label,
                problem=problem,
                stop=start + bitstrings.shape[0],
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                extra=(
                    None
                    if not np.isfinite(best_value)
                    else f"best={best_value:0.6g}"
                ),
            )

    if not optimum_chunks:
        raise RuntimeError(
            f"problem {problem.name} has no feasible states"
        )
    return best_value, np.concatenate(optimum_chunks)


def _qubo_energies(
    problem: BLP,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    chunk_size: int,
    progress_label: str | None = None,
) -> np.ndarray:
    """Enumerate one QUBO energy landscape over the full hypercube."""
    energies = np.empty(problem.num_states, dtype=float)
    total_chunks = _state_chunk_total(problem, chunk_size)
    for chunk_index, (start, bitstrings) in enumerate(
        iter_state_chunks(
            problem.num_variables,
            chunk_size=chunk_size,
            dtype=float,
        )
    ):
        stop = start + bitstrings.shape[0]
        energies[start:stop] = qubo_energy_values(
            bitstrings,
            quadratic=quadratic,
            linear=linear,
            const=const,
        )
        if progress_label is not None:
            _report_state_chunk_progress(
                progress_label,
                problem=problem,
                stop=stop,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
    return energies


def _qaoa_normalization_scale(
    quadratic: np.ndarray,
    linear: np.ndarray,
) -> float:
    """Return the shared OpenQAOA-style normalization scale."""
    return _shared_qubo_normalization_scale(
        quadratic,
        linear,
        num_variables=np.asarray(linear, dtype=float)
        .reshape(-1)
        .shape[0],
    )


def _scaled_annealer_input(
    problem: BLP,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    normalization_scale: float,
    chunk_size: int,
) -> ScaledAnnealerInput:
    """Return the normalized logical QUBO used by the SQA baseline."""
    del problem
    del chunk_size
    scaled_quadratic, scaled_linear, scaled_const = (
        _shared_scale_qubo_coefficients(
            quadratic,
            linear,
            const,
            normalization_scale=normalization_scale,
        )
    )
    return ScaledAnnealerInput(
        quadratic=scaled_quadratic,
        linear=scaled_linear,
        const=scaled_const,
    )


def _embedded_sqa_penalized_metrics(
    problem: BLP,
    annealer_input: ScaledAnnealerInput,
    *,
    annealer_backend: DWaveAnnealingBackend,
    reference: CplexReference,
    hardware_graph,
    hardware_family: str,
    hardware_size: int,
    family: str,
    size: int,
    instance_index: int,
    method: str,
    base_seed: int,
    num_reads: int,
    num_points: int,
    num_sweeps_per_beta: int,
    schedule: FixedSqaSchedule,
    chain_strength_fraction: float,
    atol: float,
    fixed_embedding: dict[int, list[Any]] | None = None,
) -> EmbeddedSQAMetrics:
    """Run embedded Ocean SQA and summarize SA-style and CoP metrics."""
    try:
        simulation = annealer_backend.sample(
            quadratic=annealer_input.quadratic,
            linear=annealer_input.linear,
            const=annealer_input.const,
            request=AnnealerRunRequest(
                hardware_graph=hardware_graph,
                hardware_family=hardware_family,
                hardware_size=hardware_size,
                family=family,
                size=size,
                instance_index=instance_index,
                problem_seed=_problem_seed(problem),
                method=method,
                base_seed=base_seed,
                num_reads=num_reads,
                num_points=num_points,
                num_sweeps_per_beta=num_sweeps_per_beta,
                schedule=schedule,
                chain_strength_fraction=chain_strength_fraction,
                fixed_embedding=fixed_embedding,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        _warn_sqa_unavailable(
            f"{annealer_backend.backend_name}: {exc}"
        )
        return EmbeddedSQAMetrics(
            mean_sqa_energy_gap=math.nan,
            sqa_optimum_rate=math.nan,
            optimum_probability=math.nan,
            coefficient_of_performance=math.nan,
            best_feasible_objective=None,
            best_feasible_gap=None,
            objective_gap=math.nan,
            num_feasible_reads=0,
            feasible_read_fraction=math.nan,
            physical_qubits=None,
            mean_chain_length=None,
            max_chain_length=None,
            mean_chain_break_fraction=None,
            broken_read_rate=None,
            effective_chain_strength=None,
            embedded_chain_to_problem_ratio=None,
        )

    _set_progress_status(
        stage="method",
        activity="embedded SQA",
        detail="decoding embedded reads and summarizing metrics",
    )
    sampleset = simulation.decoded_sampleset
    sample_energies = np.asarray(
        sampleset.record.energy, dtype=float
    )
    occurrences = np.asarray(
        sampleset.record.num_occurrences, dtype=float
    )
    total_reads = float(np.sum(occurrences))
    if total_reads <= 0:
        return EmbeddedSQAMetrics(
            mean_sqa_energy_gap=math.nan,
            sqa_optimum_rate=math.nan,
            optimum_probability=math.nan,
            coefficient_of_performance=math.nan,
            best_feasible_objective=None,
            best_feasible_gap=None,
            objective_gap=math.nan,
            num_feasible_reads=0,
            feasible_read_fraction=math.nan,
            physical_qubits=None,
            mean_chain_length=None,
            max_chain_length=None,
            mean_chain_break_fraction=None,
            broken_read_rate=None,
            effective_chain_strength=None,
            embedded_chain_to_problem_ratio=None,
        )

    bitstrings = _decoded_sampleset_bits(
        sampleset,
        num_variables=problem.num_variables,
    )
    bits_float = bitstrings.astype(float, copy=False)
    feasible_mask = problem.feasible_mask(bits_float)
    objective_values = problem.objective_values(bits_float)
    optimum_hits = reference_objective_hit_mask(
        objective_values,
        feasible_mask,
        optimum_objective=float(
            reference.optimum_objective
        ),
        objective_sense=reference.objective_sense,
        atol=atol,
    )
    optimum_probability = float(
        np.sum(occurrences[optimum_hits]) / total_reads
    )
    coefficient_of_performance = (
        optimum_probability
        * float(1 << problem.num_variables)
    )
    num_feasible_reads = int(
        np.sum(occurrences[feasible_mask])
    )
    feasible_read_fraction = (
        float(num_feasible_reads) / total_reads
    )
    if np.any(feasible_mask):
        mean_gap = float(
            np.average(
                reference_objective_gap_ratio(
                    objective_values[feasible_mask],
                    float(reference.optimum_objective),
                    objective_sense=reference.objective_sense,
                ),
                weights=occurrences[feasible_mask],
            )
        )
    else:
        mean_gap = math.nan
    if np.any(feasible_mask):
        feasible_objectives = objective_values[
            feasible_mask
        ]
        if (
            str(reference.objective_sense).strip().lower()
            == "max"
        ):
            best_feasible_objective = float(
                np.max(feasible_objectives)
            )
        else:
            best_feasible_objective = float(
                np.min(feasible_objectives)
            )
        best_feasible_gap = reference_objective_gap_ratio(
            best_feasible_objective,
            float(reference.optimum_objective),
            objective_sense=reference.objective_sense,
        )
    else:
        best_feasible_objective = None
        best_feasible_gap = None

    embedding = simulation.embedding
    used_physical = {
        node
        for chain in embedding.values()
        for node in chain
    }
    chain_lengths = np.array(
        [len(chain) for chain in embedding.values()],
        dtype=float,
    )
    if chain_lengths.size == 0:
        mean_chain_length = math.nan
        max_chain_length = 0
    else:
        mean_chain_length = float(np.mean(chain_lengths))
        max_chain_length = int(np.max(chain_lengths))

    chain_break_fraction = np.asarray(
        simulation.chain_break_fraction, dtype=float
    )
    mean_chain_break_fraction = float(
        np.mean(chain_break_fraction)
    )
    broken_read_rate = float(
        np.mean(chain_break_fraction > 0.0)
    )
    ratio = embedded_chain_to_problem_ratio(
        simulation.physical_bqm,
        embedding,
    )

    return EmbeddedSQAMetrics(
        mean_sqa_energy_gap=mean_gap,
        sqa_optimum_rate=optimum_probability,
        optimum_probability=optimum_probability,
        coefficient_of_performance=coefficient_of_performance,
        best_feasible_objective=best_feasible_objective,
        best_feasible_gap=best_feasible_gap,
        objective_gap=reference_objective_gap_ratio(
            best_feasible_objective,
            float(reference.optimum_objective),
            objective_sense=reference.objective_sense,
        ),
        num_feasible_reads=num_feasible_reads,
        feasible_read_fraction=feasible_read_fraction,
        physical_qubits=len(used_physical),
        mean_chain_length=mean_chain_length,
        max_chain_length=max_chain_length,
        mean_chain_break_fraction=mean_chain_break_fraction,
        broken_read_rate=broken_read_rate,
        effective_chain_strength=float(
            simulation.effective_chain_strength
        ),
        embedded_chain_to_problem_ratio=(
            None if ratio is None else float(ratio)
        ),
    )


# Exact-spectrum CSV export removed


def _embedding_aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate embedded-SQA rows using baseline-comparison columns."""
    grouped: dict[
        tuple[str, str, int, str, float],
        list[dict[str, object]],
    ] = {}
    for row in rows:
        key = (
            str(row["hardware_family"]),
            str(row["family"]),
            int(row["size"]),
            str(row["method"]),
            float(
                row.get(
                    "sqa_chain_strength_fraction",
                    DEFAULT_SQA_CHAIN_STRENGTH_FRACTION,
                )
                or DEFAULT_SQA_CHAIN_STRENGTH_FRACTION
            ),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key in sorted(grouped):
        (
            hardware_family,
            family,
            size,
            method,
            chain_strength_fraction,
        ) = key
        items = grouped[key]
        num_variables = int(items[0]["num_variables"])
        hardware_size = int(items[0]["hardware_size"])
        sqa_cop = np.array(
            [
                float(
                    item.get(
                        "sqa_logical_cop",
                        item.get("cop", math.nan),
                    )
                )
                for item in items
            ],
            dtype=float,
        )
        optimum_probabilities = np.array(
            [
                float(item["optimum_probability"])
                for item in items
            ],
            dtype=float,
        )
        sqa_fea = np.array(
            [
                float(
                    item.get(
                        "sqa_fea",
                        item.get(
                            "feasible_read_fraction",
                            math.nan,
                        ),
                    )
                )
                for item in items
            ],
            dtype=float,
        )
        sqa_gap = np.array(
            [
                float(item.get("sqa_gap", math.nan))
                for item in items
            ],
            dtype=float,
        )
        sqa_gaps = np.array(
            [
                float(
                    item.get(
                        "mean_sqa_energy_gap", math.nan
                    )
                )
                for item in items
            ],
            dtype=float,
        )
        sqa_optimum_rates = np.array(
            [
                float(
                    item.get("sqa_optimum_rate", math.nan)
                )
                for item in items
            ],
            dtype=float,
        )
        chain_breaks = np.array(
            [
                float(
                    item.get(
                        "mean_chain_break_fraction",
                        math.nan,
                    )
                )
                for item in items
            ],
            dtype=float,
        )
        broken_rates = np.array(
            [
                float(
                    item.get("broken_read_rate", math.nan)
                )
                for item in items
            ],
            dtype=float,
        )
        finite_sqa_cop = np.isfinite(sqa_cop)
        finite_sqa_gaps = np.isfinite(sqa_gaps)
        finite_sqa_fea = np.isfinite(sqa_fea)
        finite_sqa_gap = np.isfinite(sqa_gap)
        finite_sqa_optimum_rates = np.isfinite(
            sqa_optimum_rates
        )
        finite_optimum_probabilities = np.isfinite(
            optimum_probabilities
        )
        finite_chain_breaks = np.isfinite(chain_breaks)
        finite_broken_rates = np.isfinite(broken_rates)

        for stat_name in ("mean", "std"):
            row: dict[str, object] = {
                "hardware_family": hardware_family,
                "hardware_size": hardware_size,
                "family": family,
                "size": size,
                "method": method,
                "n": num_variables,
                "stat": stat_name,
                "inst": len(items),
                "sqa_chain_strength_fraction": chain_strength_fraction,
                "sqa_cop": math.nan,
                "sqa_fea": math.nan,
                "sqa_gap": math.nan,
                "optimum_probability": math.nan,
                "sqa_energy_gap": math.nan,
                "sqa_optimum_rate": math.nan,
                "mean_chain_break_fraction": math.nan,
                "broken_read_rate": math.nan,
            }
            summaries = (
                ("sqa_cop", sqa_cop, finite_sqa_cop),
                ("sqa_fea", sqa_fea, finite_sqa_fea),
                ("sqa_gap", sqa_gap, finite_sqa_gap),
                (
                    "optimum_probability",
                    optimum_probabilities,
                    finite_optimum_probabilities,
                ),
                (
                    "sqa_energy_gap",
                    sqa_gaps,
                    finite_sqa_gaps,
                ),
                (
                    "sqa_optimum_rate",
                    sqa_optimum_rates,
                    finite_sqa_optimum_rates,
                ),
                (
                    "mean_chain_break_fraction",
                    chain_breaks,
                    finite_chain_breaks,
                ),
                (
                    "broken_read_rate",
                    broken_rates,
                    finite_broken_rates,
                ),
            )
            for field, values, finite in summaries:
                if not np.any(finite):
                    row[field] = math.nan
                elif stat_name == "mean":
                    row[field] = float(
                        np.mean(values[finite])
                    )
                else:
                    row[field] = float(
                        np.std(values[finite], ddof=0)
                    )
            out.append(row)
    return out


def _evaluate_embedding_instance(
    *,
    family: str,
    size: int,
    instance_index: int,
    problem: BLP,
    reference: CplexReference,
    tuned_up: _shared_tuning_models.TunedUnbalancedParameters,
    summary_projected_config_map: dict[
        str, _shared_tuning_models.SelectedProjectedConfig
    ],
    fixed_projected_methods: tuple[str, ...],
    hardware_aware_projected_methods: tuple[str, ...],
    hardware_families: tuple[str, ...],
    hardware_sizes: dict[str, int],
    base_seed: int,
    measure_lam: float,
    pegasus_size: int,
    sample_cap_log2: int,
    chunk_size: int,
    projection_reg: float,
    sqa_num_reads: int,
    sqa_num_points: int,
    sqa_num_sweeps_per_beta: int,
    sqa_schedule: FixedSqaSchedule,
    sqa_chain_strength_fraction: float,
) -> list[dict[str, object]]:
    """Evaluate one problem instance across all hardware/method pairs."""
    annealer_backend = SimulatedDWaveAnnealingBackend()
    hardware_graphs = {
        hardware_family: build_dwave_graph(
            hardware_family,
            hardware_sizes[hardware_family],
        )
        for hardware_family in hardware_families
    }
    components_cache: dict[
        tuple[object, ...],
        ProjectedPenaltyComponents,
    ] = {}
    fixed_projected_config_map = {
        method: summary_projected_config_map[method]
        for method in fixed_projected_methods
    }
    fixed_projected_multiplier_map = {
        method: fixed_projected_config_map[method].tuning
        for method in fixed_projected_config_map
    }
    projected_components_map: dict[
        str, ProjectedPenaltyComponents
    ] = {}

    for method in fixed_projected_methods:
        selected_config = fixed_projected_config_map[method]
        cache_key = _shared_tuning_support.projected_components_cache_key(
            projection_method=selected_config.projection_method
            or method,
            family=family,
            size=size,
            instance_index=instance_index,
            measure_name=selected_config.measure_name,
            measure_lam=measure_lam,
            penalty_template=selected_config.penalty_template,
            penalty_template_kwargs=selected_config.penalty_template_kwargs,
            standardize=selected_config.projected_standardize,
        )
        if cache_key not in components_cache:
            components_cache[cache_key] = (
                _shared_tuning_support.build_projected_components(
                    problem,
                    projection_method=(
                        selected_config.projection_method
                        or method
                    ),
                    family=family,
                    size=size,
                    instance_index=instance_index,
                    base_seed=base_seed,
                    measure_name=selected_config.measure_name,
                    measure_lam=measure_lam,
                    penalty_template=selected_config.penalty_template,
                    penalty_template_kwargs=(
                        selected_config.penalty_template_kwargs
                    ),
                    pegasus_size=pegasus_size,
                    sample_cap_log2=sample_cap_log2,
                    chunk_size=chunk_size,
                    reg=projection_reg,
                    standardize=selected_config.projected_standardize,
                )
            )
        projected_components_map[method] = components_cache[
            cache_key
        ]

    up_components = _build_unbalanced_components(
        problem,
        lambda1_shape=float(tuned_up.up_lambda1_shape),
        lambda2_shape=float(tuned_up.up_lambda2_shape),
        standardize=(
            True
            if tuned_up.per_constraint_standardization
            is None
            else bool(
                tuned_up.per_constraint_standardization
            )
        ),
    )
    fixed_method_qubos: dict[
        str, tuple[np.ndarray, np.ndarray, float]
    ] = {
        "unbalanced": _build_unbalanced_qubo_from_components(
            problem,
            up_components,
            equality_multiplier=(
                0.0
                if tuned_up.up_equality_multiplier is None
                else float(tuned_up.up_equality_multiplier)
            ),
            inequality_multiplier=float(
                tuned_up.up_inequality_multiplier
            ),
        ),
        "projected_full": _shared_tuning_support.projected_full_qubo(
            problem,
            projected_components_map["projected_full"],
            fixed_projected_multiplier_map[
                "projected_full"
            ],
        ),
        "projected_up_support": _shared_tuning_support.projected_full_qubo(
            problem,
            projected_components_map[
                "projected_up_support"
            ],
            fixed_projected_multiplier_map[
                "projected_up_support"
            ],
        ),
    }

    rows: list[dict[str, object]] = []
    for hardware_family in hardware_families:
        hardware_size = hardware_sizes[hardware_family]
        hardware_graph = hardware_graphs[hardware_family]
        hardware_projected_config_map = {
            **fixed_projected_config_map,
            "projected_pegasus": summary_projected_config_map[
                _hardware_projected_summary_method(
                    hardware_family
                )
            ],
        }
        hardware_projected_multiplier_map = {
            method: hardware_projected_config_map[
                method
            ].tuning
            for method in hardware_projected_config_map
        }
        hardware_projected_components_map = dict(
            projected_components_map
        )
        projected_topology_fixed_embedding: (
            dict[int, list[Any]] | None
        ) = None

        for method in hardware_aware_projected_methods:
            selected_config = hardware_projected_config_map[
                method
            ]
            pair_edges, fixed_embedding = (
                _projection_topology_details(
                    problem,
                    projection_method=(
                        selected_config.projection_method
                        or method
                    ),
                    projection_hardware_graph=hardware_graph,
                )
            )
            cache_key = _shared_tuning_support.projected_components_cache_key(
                projection_method=selected_config.projection_method
                or method,
                family=family,
                size=size,
                instance_index=instance_index,
                measure_name=selected_config.measure_name,
                measure_lam=measure_lam,
                penalty_template=selected_config.penalty_template,
                penalty_template_kwargs=selected_config.penalty_template_kwargs,
                standardize=selected_config.projected_standardize,
                deployment_topology=hardware_family,
                deployment_topology_size=hardware_size,
            )
            if cache_key not in components_cache:
                components_cache[cache_key] = (
                    _build_projected_components(
                        problem,
                        projection_method=(
                            selected_config.projection_method
                            or method
                        ),
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        base_seed=base_seed,
                        measure_name=selected_config.measure_name,
                        measure_lam=measure_lam,
                        penalty_template=selected_config.penalty_template,
                        penalty_template_kwargs=(
                            selected_config.penalty_template_kwargs
                        ),
                        pegasus_size=pegasus_size,
                        sample_cap_log2=sample_cap_log2,
                        chunk_size=chunk_size,
                        reg=projection_reg,
                        standardize=selected_config.projected_standardize,
                        projection_hardware_graph=hardware_graph,
                        pair_edges=pair_edges,
                    )
                )
            hardware_projected_components_map[method] = (
                components_cache[cache_key]
            )
            if method == "projected_pegasus":
                projected_topology_fixed_embedding = (
                    fixed_embedding
                )

        fixed_embeddings: dict[
            str, dict[int, list[Any]] | None
        ] = {
            "unbalanced": None,
            "projected_full": None,
            "projected_up_support": None,
            PROJECTED_TOPOLOGY_METHOD: projected_topology_fixed_embedding,
        }

        method_qubos: dict[
            str, tuple[np.ndarray, np.ndarray, float]
        ] = {
            "unbalanced": fixed_method_qubos["unbalanced"],
            "projected_full": fixed_method_qubos[
                "projected_full"
            ],
            "projected_up_support": fixed_method_qubos[
                "projected_up_support"
            ],
            PROJECTED_TOPOLOGY_METHOD: _shared_tuning_support.projected_full_qubo(
                problem,
                hardware_projected_components_map[
                    "projected_pegasus"
                ],
                hardware_projected_multiplier_map[
                    "projected_pegasus"
                ],
            ),
        }
        num_inequality_quadratic_terms = (
            _num_inequality_quadratic_terms(problem)
        )

        for method in DEFAULT_METHODS:
            quadratic, linear, const = method_qubos[method]
            normalization_scale = _qaoa_normalization_scale(
                quadratic,
                linear,
            )
            annealer_input = _scaled_annealer_input(
                problem,
                quadratic,
                linear,
                const,
                normalization_scale=normalization_scale,
                chunk_size=chunk_size,
            )
            sqa_metrics = _embedded_sqa_penalized_metrics(
                problem,
                annealer_input,
                annealer_backend=annealer_backend,
                reference=reference,
                hardware_graph=hardware_graph,
                hardware_family=hardware_family,
                hardware_size=hardware_size,
                family=family,
                size=size,
                instance_index=instance_index,
                method=method,
                base_seed=base_seed,
                num_reads=sqa_num_reads,
                num_points=sqa_num_points,
                num_sweeps_per_beta=sqa_num_sweeps_per_beta,
                schedule=sqa_schedule,
                chain_strength_fraction=sqa_chain_strength_fraction,
                atol=DEFAULT_REFERENCE_ATOL,
                fixed_embedding=fixed_embeddings.get(
                    method
                ),
            )
            projection_lookup_method = (
                "projected_pegasus"
                if method == PROJECTED_TOPOLOGY_METHOD
                else method
            )
            projected_components = (
                hardware_projected_components_map.get(
                    projection_lookup_method
                )
            )
            selected_config = (
                hardware_projected_config_map.get(
                    projection_lookup_method
                )
            )
            projected_multipliers = (
                hardware_projected_multiplier_map.get(
                    projection_lookup_method
                )
            )
            rows.append(
                {
                    "hardware_family": hardware_family,
                    "hardware_size": hardware_size,
                    "family": family,
                    "size": size,
                    "instance_index": instance_index,
                    "method": method,
                    **_problem_provenance_fields(problem),
                    **_projection_regime_fields(
                        method,
                        hardware_family=hardware_family,
                    ),
                    "num_variables": problem.num_variables,
                    "num_states": problem.num_states,
                    "reference_optimum_objective": (
                        float(reference.optimum_objective)
                    ),
                    "reference_optimum_source": reference.optimum_source,
                    "reference_objective_sense": reference.objective_sense,
                    "reference_match_tolerance": DEFAULT_REFERENCE_ATOL,
                    "num_inequality_quadratic_terms": (
                        num_inequality_quadratic_terms
                    ),
                    "sqa_logical_cop": (
                        sqa_metrics.coefficient_of_performance
                    ),
                    "sqa_fea": sqa_metrics.feasible_read_fraction,
                    "sqa_gap": sqa_metrics.objective_gap,
                    "optimum_probability": (
                        sqa_metrics.optimum_probability
                    ),
                    "cop": sqa_metrics.coefficient_of_performance,
                    "true_optimum_objective": float(
                        reference.optimum_objective
                    ),
                    "best_feasible_objective": (
                        sqa_metrics.best_feasible_objective
                    ),
                    "best_feasible_gap": sqa_metrics.best_feasible_gap,
                    "objective_gap": sqa_metrics.objective_gap,
                    "num_feasible_reads": (
                        sqa_metrics.num_feasible_reads
                    ),
                    "feasible_read_fraction": (
                        sqa_metrics.feasible_read_fraction
                    ),
                    "sqa_num_reads": sqa_num_reads,
                    "sqa_num_sweeps": sqa_num_points,
                    "sqa_num_sweeps_per_beta": (
                        sqa_num_sweeps_per_beta
                    ),
                    "sqa_chain_strength_fraction": (
                        sqa_chain_strength_fraction
                    ),
                    "sqa_schedule_id": sqa_schedule.schedule_id,
                    "sqa_schedule_kind": sqa_schedule.schedule_kind,
                    "sqa_schedule_anchor_size": None,
                    "sqa_beta_scale": sqa_schedule.beta_scale,
                    "sqa_total_schedule_time": (
                        sqa_schedule.total_schedule_time
                    ),
                    "sqa_anneal_schedule": (
                        _shared_anneal_schedule_json(
                            sqa_schedule.anneal_schedule
                        )
                    ),
                    "mean_sqa_energy_gap": (
                        sqa_metrics.mean_sqa_energy_gap
                    ),
                    "sqa_optimum_rate": sqa_metrics.sqa_optimum_rate,
                    "physical_qubits": sqa_metrics.physical_qubits,
                    "mean_chain_length": sqa_metrics.mean_chain_length,
                    "max_chain_length": sqa_metrics.max_chain_length,
                    "mean_chain_break_fraction": (
                        sqa_metrics.mean_chain_break_fraction
                    ),
                    "broken_read_rate": sqa_metrics.broken_read_rate,
                    "effective_chain_strength": (
                        sqa_metrics.effective_chain_strength
                    ),
                    "embedded_chain_to_problem_ratio": (
                        sqa_metrics.embedded_chain_to_problem_ratio
                    ),
                    "projected_sample_size": (
                        projected_components.sample_size
                        if projected_components is not None
                        else None
                    ),
                    "projected_num_quadratic_couplers": (
                        projected_components.num_quadratic_couplers
                        if projected_components is not None
                        else None
                    ),
                    "projection_method": (
                        selected_config.projection_method
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_summary_method": (
                        selected_config.method
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_measure": (
                        selected_config.measure_name
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_measure_lam": (
                        measure_lam
                        if projected_components is not None
                        else None
                    ),
                    "projection_penalty_template": (
                        selected_config.penalty_template
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_selection_mode": (
                        selected_config.selection_mode
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_selection_source": (
                        selected_config.selection_source
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projection_candidate_rank": (
                        int(selected_config.candidate_rank)
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projected_standardize": (
                        selected_config.projected_standardize
                        if projected_components is not None
                        and selected_config is not None
                        else None
                    ),
                    "projected_equality_multiplier": (
                        projected_multipliers.equality_multiplier
                        if projected_multipliers is not None
                        else None
                    ),
                    "projected_inequality_multiplier": (
                        projected_multipliers.inequality_multiplier
                        if projected_multipliers is not None
                        else None
                    ),
                    "normalization_regime": (
                        (
                            tuned_up.normalization_regime
                            or _UP_NORMALIZATION_REGIME
                        )
                        if method == "unbalanced"
                        else _UP_NORMALIZATION_REGIME
                    ),
                    "per_constraint_standardization": (
                        (
                            True
                            if tuned_up.per_constraint_standardization
                            is None
                            else bool(
                                tuned_up.per_constraint_standardization
                            )
                        )
                        if method == "unbalanced"
                        else (
                            None
                            if selected_config is None
                            else bool(
                                selected_config.projected_standardize
                            )
                        )
                    ),
                    "qubo_normalization_scale": normalization_scale,
                    "up_equality_multiplier": (
                        tuned_up.up_equality_multiplier
                        if method == "unbalanced"
                        else None
                    ),
                    "up_inequality_multiplier": (
                        tuned_up.up_inequality_multiplier
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda1_shape": (
                        tuned_up.up_lambda1_shape
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda2_shape": (
                        tuned_up.up_lambda2_shape
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda_gauge": (
                        tuned_up.up_lambda_gauge
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda0": (
                        tuned_up.lambda0
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda1": (
                        tuned_up.lambda1
                        if method == "unbalanced"
                        else None
                    ),
                    "up_lambda2": (
                        tuned_up.lambda2
                        if method == "unbalanced"
                        else None
                    ),
                }
            )

    return rows


def _progress_totals(
    problem_batches: dict[tuple[str, int], list[BLP]],
    *,
    hardware_families: tuple[str, ...],
) -> ProgressTotals:
    """Return the aggregate counts shown in the progress UI."""
    instances = 0
    topologies = 0
    methods = 0
    for batch in problem_batches.values():
        instances += len(batch)
        for problem in batch:
            del problem
            topologies += len(hardware_families)
            methods += len(hardware_families) * len(
                DEFAULT_METHODS
            )
    return ProgressTotals(
        instances=instances,
        topologies=topologies,
        measures=methods,
    )


def _cop_row_sort_key(
    row: dict[str, object],
) -> tuple[int, int, int, int, int, float]:
    """Return a stable row order for instance-level CSV output."""
    return (
        FAMILY_ORDER.index(str(row["family"])),
        int(row["size"]),
        int(row["instance_index"]),
        HARDWARE_CODES[str(row["hardware_family"])],
        METHOD_CODES[str(row["method"])],
        float(
            row.get(
                "sqa_chain_strength_fraction",
                DEFAULT_SQA_CHAIN_STRENGTH_FRACTION,
            )
            or DEFAULT_SQA_CHAIN_STRENGTH_FRACTION
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the intentionally small CLI surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and CSV summaries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed used for instance generation and sampled projection",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=DEFAULT_NUM_INSTANCES,
        help="Number of random instances per family/size",
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=None,
        help=(
            "Optional seed manifest CSV used to construct one matched "
            "instance set for evaluation"
        ),
    )
    parser.add_argument(
        "--cplex-reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional CPLEX reference CSV. Defaults to the manifest-sidecar "
            "reference when --instance-manifest is provided, otherwise "
            "`data/classical_baselines/cplex_optima.csv`."
        ),
    )
    add_family_selection_arguments(
        parser, include_sizes=True
    )
    parser.add_argument(
        "--hardware-families",
        nargs="+",
        choices=list(HARDWARE_FAMILIES),
        default=list(HARDWARE_FAMILIES),
        help="One or more D-Wave hardware graph families to evaluate",
    )
    parser.add_argument(
        "--sqa-num-reads",
        type=int,
        default=DEFAULT_SQA_NUM_READS,
        help="Number of decoded reads drawn from embedded Ocean SQA",
    )
    parser.add_argument(
        "--sqa-chain-strength-fraction",
        type=float,
        default=DEFAULT_SQA_CHAIN_STRENGTH_FRACTION,
        help=(
            "Fraction of Ocean's default chain strength to use in "
            "embedded SQA"
        ),
    )
    parser.add_argument(
        "--chimera-size",
        type=int,
        default=DEFAULT_CHIMERA_SIZE,
        help="Size parameter for dwave_networkx.chimera_graph",
    )
    parser.add_argument(
        "--pegasus-size",
        type=int,
        default=DEFAULT_PEGASUS_SIZE,
        help="Size parameter for dwave_networkx.pegasus_graph",
    )
    parser.add_argument(
        "--zephyr-size",
        type=int,
        default=DEFAULT_ZEPHYR_SIZE,
        help="Size parameter for dwave_networkx.zephyr_graph",
    )
    parser.add_argument(
        "--progress-ui",
        choices=PROGRESS_UI_CHOICES,
        default=DEFAULT_PROGRESS_UI,
        help="Progress renderer to use while the experiment runs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes for per-instance evaluation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSVs and run metadata instead of merging.",
    )
    return parser


def main() -> None:
    """Run the full embedding-based comparison experiment."""
    parser = build_argument_parser()
    args = parser.parse_args()

    active_families = selected_families_from_args(args)
    family_sizes = selected_family_sizes_from_args(args)
    chunk_size = DEFAULT_CHUNK_SIZE
    measure_lam = DEFAULT_MEASURE_LAM
    sample_cap_log2 = DEFAULT_PROJECTION_SAMPLE_CAP_LOG2
    projection_reg = DEFAULT_PROJECTION_REG
    pegasus_size = int(args.pegasus_size)
    sqa_num_reads = int(args.sqa_num_reads)
    sqa_chain_strength_fraction = float(
        args.sqa_chain_strength_fraction
    )
    if not (0.0 < sqa_chain_strength_fraction <= 1.0):
        parser.error(
            "--sqa-chain-strength-fraction must be in (0, 1]"
        )
    sqa_num_points = DEFAULT_SQA_NUM_SWEEPS
    sqa_num_sweeps_per_beta = (
        DEFAULT_SQA_NUM_SWEEPS_PER_BETA
    )
    workers = max(1, int(args.workers))
    output_dir = Path(
        args.output_dir
    ) / _chain_strength_fraction_dirname(
        sqa_chain_strength_fraction
    )
    hardware_families = tuple(
        dict.fromkeys(args.hardware_families)
    )
    # Prefer the manifest-sidecar CPLEX CSV when a manifest is used or
    # when the project-default shared manifest exists and no explicit
    # instance manifest was provided on the CLI.
    instance_manifest_arg = getattr(args, "instance_manifest", None)
    if instance_manifest_arg is None:
        default_manifest = EXPERIMENTS_DIR / "manifests" / "shared_eval_seed_manifest.csv"
        if default_manifest.exists():
            instance_manifest_arg = default_manifest
            _log(
                f"auto-using instance manifest {instance_manifest_arg} (found project default)"
            )
    cplex_reference_path = resolve_cplex_reference_path(
        getattr(args, "cplex_reference_csv", None),
        instance_manifest=instance_manifest_arg,
    )
    # Keep args.instance_manifest in sync with the resolved manifest so the rest
    # of this run uses the same manifest (and the corresponding CPLEX sidecar).
    args.instance_manifest = instance_manifest_arg
    cplex_reference_index = load_cplex_reference_index(
        cplex_reference_path
    )
    projected_summary_path = (
        _resolve_projected_penalty_summary_path()
    )
    unbalanced_summary_path = (
        _resolve_unbalanced_penalty_summary_path()
    )
    fixed_sqa_schedule = _fixed_sqa_schedule()

    hardware_sizes = {
        "chimera": int(args.chimera_size),
        "pegasus": int(args.pegasus_size),
        "zephyr": int(args.zephyr_size),
    }

    summary_projected_methods = (
        "projected_full",
        "projected_up_support",
        "projected_pegasus",
        "projected_chimera",
        "projected_zephyr",
    )
    fixed_projected_methods = (
        "projected_full",
        "projected_up_support",
    )
    hardware_aware_projected_methods = (
        "projected_pegasus",
    )
    get_problem_batch = _shared_problem_batch_getter(
        base_seed=args.seed,
        num_instances=args.num_instances,
        instance_manifest=args.instance_manifest,
        family_sizes=family_sizes,
    )

    (
        size_specific_up_params,
        size_specific_projected_configs,
        projected_tuning_rows,
        up_tuning_rows,
    ) = _load_penalty_configs_from_summaries(
        projected_summary_path=projected_summary_path,
        unbalanced_summary_path=unbalanced_summary_path,
        families=active_families,
        family_sizes=family_sizes,
        projected_methods=summary_projected_methods,
    )
    projected_selection_rows = [
        dict(row) for row in projected_tuning_rows
    ]
    for family in active_families:
        for size in family_sizes[family]:
            if (
                family,
                size,
            ) not in size_specific_up_params:
                raise RuntimeError(
                    "missing unbalanced summary-backed row for "
                    f"family={family}, size={size} using "
                    f"{unbalanced_summary_path}"
                )
            for method in summary_projected_methods:
                if (
                    family,
                    size,
                    method,
                ) not in size_specific_projected_configs:
                    raise RuntimeError(
                        "missing summary-backed projected row for "
                        f"family={family}, size={size}, method={method} in "
                        f"{projected_summary_path}"
                    )

    for family in active_families:
        for size in family_sizes[family]:
            get_problem_batch(family, size)
    problem_batches = {
        (family, size): get_problem_batch(family, size)
        for family in active_families
        for size in family_sizes[family]
    }
    reference_batches = {
        (family, size): [
            cplex_reference_for_synthetic_problem(
                index=cplex_reference_index,
                family=family,
                size=size,
                problem_seed=int(
                    problem.metadata["problem_seed"]
                ),
                problem=problem,
                atol=DEFAULT_REFERENCE_ATOL,
            )
            for problem in problem_batches[(family, size)]
        ]
        for family in active_families
        for size in family_sizes[family]
    }

    progress_totals = _progress_totals(
        problem_batches,
        hardware_families=hardware_families,
    )
    progress = _build_compare_embedding_progress(
        mode=str(args.progress_ui),
        totals=progress_totals,
        worker_count=workers,
    )
    global _ACTIVE_PROGRESS
    _ACTIVE_PROGRESS = progress

    if progress is not None:
        progress.start()
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
    try:
        ensure_run_metadata(
            output_dir,
            {
                "base_seed": int(args.seed),
                "num_instances": int(args.num_instances),
                "instance_manifest": (
                    None
                    if args.instance_manifest is None
                    else str(
                        Path(
                            args.instance_manifest
                        ).resolve()
                    )
                ),
                "cplex_reference_csv": str(
                    cplex_reference_path.resolve()
                ),
                "projected_summary_path": str(
                    projected_summary_path.resolve()
                ),
                "unbalanced_summary_path": str(
                    unbalanced_summary_path.resolve()
                ),
                "hardware_families": list(
                    hardware_families
                ),
                "hardware_sizes": dict(hardware_sizes),
                "sqa_num_reads": int(sqa_num_reads),
                "sqa_num_points": int(sqa_num_points),
                "sqa_num_sweeps_per_beta": int(
                    sqa_num_sweeps_per_beta
                ),
                "sqa_chain_strength_fraction": (
                    float(sqa_chain_strength_fraction)
                ),
                "sqa_schedule_id": fixed_sqa_schedule.schedule_id,
            },
            force=bool(args.force),
        )
        if args.instance_manifest is None:
            _log(
                "using generated evaluation instances from "
                f"seed={args.seed} with num_instances={args.num_instances}"
            )
        else:
            _log(
                "using matched evaluation instances from manifest "
                f"{Path(args.instance_manifest).resolve()}"
            )
        _log(
            f"using CPLEX references from {cplex_reference_path}"
        )
        _log(
            "using projected penalty summaries from "
            f"{projected_summary_path}"
        )
        _log(
            "using unbalanced penalty summaries from "
            f"{unbalanced_summary_path}"
        )
        _log(
            "using the fixed shared SQA schedule "
            f"{fixed_sqa_schedule.schedule_id}"
        )
        _log(
            "using SQA chain strength fraction "
            f"{sqa_chain_strength_fraction:g}"
        )
        _log(f"using {workers} worker process(es)")

        cop_rows: list[dict[str, object]] = []
        methods_per_instance = len(hardware_families) * len(
            DEFAULT_METHODS
        )
        topologies_per_instance = len(hardware_families)
        for family in active_families:
            sqa_schedule = fixed_sqa_schedule
            for size in family_sizes[family]:
                tuned_up = size_specific_up_params[
                    (family, size)
                ]
                summary_projected_config_map = {
                    method: size_specific_projected_configs[
                        (family, size, method)
                    ]
                    for method in summary_projected_methods
                }
                problems = problem_batches[(family, size)]
                references = reference_batches[
                    (family, size)
                ]
                _log(
                    f"evaluating {family} size {size} "
                    f"with {len(problems)} instance(s)"
                )
                if executor is None:
                    for (
                        instance_index,
                        problem,
                    ) in enumerate(problems):
                        _set_progress_status(
                            stage="instance",
                            activity="running instance evaluation",
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            total_instances=len(problems),
                            topology=None,
                            method=None,
                            detail="sequential worker",
                        )
                        instance_rows = _evaluate_embedding_instance(
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            problem=problem,
                            reference=references[
                                instance_index
                            ],
                            tuned_up=tuned_up,
                            summary_projected_config_map=(
                                summary_projected_config_map
                            ),
                            fixed_projected_methods=fixed_projected_methods,
                            hardware_aware_projected_methods=(
                                hardware_aware_projected_methods
                            ),
                            hardware_families=hardware_families,
                            hardware_sizes=hardware_sizes,
                            base_seed=int(args.seed),
                            measure_lam=measure_lam,
                            pegasus_size=pegasus_size,
                            sample_cap_log2=sample_cap_log2,
                            chunk_size=chunk_size,
                            projection_reg=projection_reg,
                            sqa_num_reads=sqa_num_reads,
                            sqa_num_points=sqa_num_points,
                            sqa_num_sweeps_per_beta=(
                                sqa_num_sweeps_per_beta
                            ),
                            sqa_schedule=sqa_schedule,
                            sqa_chain_strength_fraction=(
                                sqa_chain_strength_fraction
                            ),
                        )
                        cop_rows.extend(instance_rows)
                        _advance_progress_counter(
                            "methods",
                            methods_per_instance,
                        )
                        _advance_progress_counter(
                            "topologies",
                            topologies_per_instance,
                        )
                        _advance_progress_counter(
                            "instances", 1
                        )
                    continue

                futures = {
                    executor.submit(
                        _evaluate_embedding_instance,
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        problem=problem,
                        reference=references[
                            instance_index
                        ],
                        tuned_up=tuned_up,
                        summary_projected_config_map=(
                            summary_projected_config_map
                        ),
                        fixed_projected_methods=fixed_projected_methods,
                        hardware_aware_projected_methods=(
                            hardware_aware_projected_methods
                        ),
                        hardware_families=hardware_families,
                        hardware_sizes=hardware_sizes,
                        base_seed=int(args.seed),
                        measure_lam=measure_lam,
                        pegasus_size=pegasus_size,
                        sample_cap_log2=sample_cap_log2,
                        chunk_size=chunk_size,
                        projection_reg=projection_reg,
                        sqa_num_reads=sqa_num_reads,
                        sqa_num_points=sqa_num_points,
                        sqa_num_sweeps_per_beta=(
                            sqa_num_sweeps_per_beta
                        ),
                        sqa_schedule=sqa_schedule,
                        sqa_chain_strength_fraction=(
                            sqa_chain_strength_fraction
                        ),
                    ): instance_index
                    for instance_index, problem in enumerate(
                        problems
                    )
                }
                completed_rows: dict[
                    int, list[dict[str, object]]
                ] = {}
                completed = 0
                for future in as_completed(futures):
                    instance_index = futures[future]
                    try:
                        completed_rows[instance_index] = (
                            future.result()
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "parallel embedded evaluation failed for "
                            f"family={family}, size={size}, "
                            f"instance_index={instance_index}"
                        ) from exc
                    completed += 1
                    _set_progress_status(
                        stage="instance",
                        activity="collecting worker result",
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        total_instances=len(problems),
                        topology=None,
                        method=None,
                        detail=(
                            f"completed {completed}/{len(problems)} "
                            "instance worker(s)"
                        ),
                    )
                    _advance_progress_counter(
                        "methods",
                        methods_per_instance,
                    )
                    _advance_progress_counter(
                        "topologies",
                        topologies_per_instance,
                    )
                    _advance_progress_counter(
                        "instances", 1
                    )
                for instance_index in range(len(problems)):
                    cop_rows.extend(
                        completed_rows[instance_index]
                    )

        _set_progress_status(
            stage="writing",
            activity="writing outputs",
            family=None,
            size=None,
            instance_index=None,
            total_instances=None,
            topology=None,
            method=None,
            detail="writing CSV summaries",
        )

        projected_tuning_path = (
            output_dir
            / "projected_penalty_tuning_summary.csv"
        )
        if args.force:
            merged_projected_tuning_rows = list(
                projected_tuning_rows
            )
            merged_projected_tuning_rows.sort(
                key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    str(row["method"]),
                )
            )
        else:
            merged_projected_tuning_rows = merge_csv_rows(
                projected_tuning_path,
                projected_tuning_rows,
                key_fields=("family", "size", "method"),
                sort_key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    str(row["method"]),
                ),
            )
        _write_rows_csv(
            projected_tuning_path,
            merged_projected_tuning_rows,
        )
        _log(f"wrote {projected_tuning_path}")

        projected_selection_path = (
            output_dir
            / "projected_combo_selection_summary.csv"
        )
        if args.force:
            merged_projected_selection_rows = list(
                projected_selection_rows
            )
            merged_projected_selection_rows.sort(
                key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    str(row["method"]),
                )
            )
        else:
            merged_projected_selection_rows = (
                merge_csv_rows(
                    projected_selection_path,
                    projected_selection_rows,
                    key_fields=("family", "size", "method"),
                    sort_key=lambda row: (
                        FAMILY_ORDER.index(
                            str(row["family"])
                        ),
                        int(row["size"]),
                        str(row["method"]),
                    ),
                )
            )
        _write_rows_csv(
            projected_selection_path,
            merged_projected_selection_rows,
        )
        _log(f"wrote {projected_selection_path}")

        up_tuning_path = (
            output_dir
            / "unbalanced_penalty_tuning_summary.csv"
        )
        if args.force:
            merged_up_tuning_rows = list(up_tuning_rows)
            merged_up_tuning_rows.sort(
                key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                )
            )
        else:
            merged_up_tuning_rows = merge_csv_rows(
                up_tuning_path,
                up_tuning_rows,
                key_fields=("family", "size"),
                sort_key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                ),
            )
        _write_rows_csv(
            up_tuning_path, merged_up_tuning_rows
        )
        _log(f"wrote {up_tuning_path}")

        cop_instance_path = (
            output_dir / "cop_instance_summary.csv"
        )
        cop_aggregate_path = (
            output_dir / "cop_aggregate_summary.csv"
        )
        if args.force:
            merged_cop_rows = list(cop_rows)
            merged_cop_rows.sort(key=_cop_row_sort_key)
        else:
            merged_cop_rows = merge_csv_rows(
                cop_instance_path,
                cop_rows,
                key_fields=(
                    "hardware_family",
                    "family",
                    "size",
                    "instance_index",
                    "method",
                    "sqa_chain_strength_fraction",
                ),
                sort_key=_cop_row_sort_key,
            )
        cop_aggregate_rows = _embedding_aggregate_rows(
            merged_cop_rows
        )
        _write_rows_csv(cop_instance_path, merged_cop_rows)
        _write_rows_csv(
            cop_aggregate_path, cop_aggregate_rows
        )
        _log(f"wrote {cop_instance_path}")
        _log(f"wrote {cop_aggregate_path}")

    finally:
        if executor is not None:
            executor.shutdown(
                wait=True, cancel_futures=False
            )
        if progress is not None:
            progress.close()
        _ACTIVE_PROGRESS = None


if __name__ == "__main__":
    main()
