"""Compare topology-aware projections of tuned unbalanced penalties.

This experiment reuses the instance-generation, reference-loading, and
annealer-evaluation pipelines from
``experiments.compare_methods_baseline`` and
``experiments.compare_methods_embedding`` while changing only the projected
inequality target:

1. load the tuned unbalanced-penalty shape and deployment multipliers;
2. project the tuned raw UP row ``lambda1_shape * v + lambda2_shape * v^2``
   instead of projecting a Heaviside-style ideal penalty;
3. compare only topology-aware projected methods against the direct
   unbalanced baseline;
4. write both instance-level outputs and robust aggregate summaries.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

if __package__ in (None, ""):
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from experiments import (
    compare_methods_baseline as baseline_mod,
)
from experiments import (
    compare_methods_embedding as embedding_mod,
)
from experiments.experiment_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MEASURE_LAM,
    DEFAULT_NUM_INSTANCES,
    DEFAULT_PEGASUS_SIZE,
    DEFAULT_PROGRESS_UI,
    DEFAULT_PROJECTION_REG,
    DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
    DEFAULT_SEED,
    DEFAULT_SQA_NUM_READS,
    DEFAULT_SQA_NUM_SWEEPS,
    DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
    DEFAULT_TUNING_DIR,
    DEFAULT_TUNING_MIN,
    DEFAULT_TUNING_NM_FATOL,
    DEFAULT_TUNING_NM_MAXITER,
    DEFAULT_TUNING_NM_START_POINTS,
    DEFAULT_TUNING_NM_XATOL,
    DEFAULT_TUNING_SIZES,
    PROGRESS_UI_CHOICES,
)
from experiments.utils.baseline_progress import (
    BaselineComparisonProgressTotals,
    activate_progress_ui,
)
from experiments.utils.baseline_progress import (
    advance_progress_counter as advance_tuning_progress_counter,
)
from experiments.utils.baseline_progress import (
    deactivate_progress_ui,
)
from experiments.utils.baseline_progress import (
    log as tuning_log,
)
from experiments.utils.baseline_progress import (
    set_progress_status as set_tuning_progress_status,
)
from experiments.utils.cop_aggregation import (
    write_aggregate_outputs,
)
from experiments.utils.cplex_reference import (
    DEFAULT_REFERENCE_ATOL,
    cplex_reference_for_synthetic_problem,
    load_cplex_reference_index,
    resolve_cplex_reference_path,
)
from experiments.utils.driver_common import (
    FAMILY_CODES,
    binary_to_spin_states,
    build_child_rng,
    default_family_sizes,
    is_complete_pair_edge_set,
    num_inequality_quadratic_terms,
    problem_batch_getter,
    problem_provenance_fields,
    projection_regime_fields,
    write_rows_csv,
)
from experiments.utils.family_cli import (
    add_family_selection_arguments,
    selected_families_from_args,
    selected_family_sizes_from_args,
)
from experiments.utils.merge_outputs import (
    ensure_run_metadata,
    merge_csv_rows,
)
from experiments.utils.projected_method_selection import (
    template_row_fields,
)
from experiments.utils.projected_pipeline import (
    ProjectedPenaltyComponents,
    build_projected_components,
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
from experiments.utils.tuning_core import (
    tune_precomputed_projected_multipliers,
)
from experiments.utils.tuning_models import TuningInstance
from experiments.utils.tuning_summary import (
    load_selected_projected_configs,
    load_tuned_unbalanced_parameters,
    write_projected_tuning_summary_csv,
)
from experiments.utils.tuning_support import (
    objective_energies as tuning_objective_energies,
)
from experiments.utils.tuning_support import (
    optimum_states as tuning_optimum_states,
)
from experiments.utils.tuning_support import (
    projected_component_energies as tuning_projected_component_energies,
)
from experiments.utils.unbalanced_pipeline import (
    UP_NORMALIZATION_REGIME,
    build_unbalanced_components,
    build_unbalanced_qubo_from_components,
)
from fourier_projection.blp import BLP
from fourier_projection.penalties import IdealPenalty
from fourier_projection.projection import (
    project_penalty_values_importance,
)
from fourier_projection.topology import HardwareTopology

DEFAULT_OUTPUT_ROOT = (
    EXPERIMENTS_DIR
    / "results"
    / "compare_unb_pen_up_projection"
)
DEFAULT_PROJECTED_UP_TUNING_DIR = (
    DEFAULT_TUNING_DIR / "compare_unb_pen_projected"
)
DEFAULT_BOOTSTRAP_RESAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_INSTANCE_MANIFEST = (
    EXPERIMENTS_DIR
    / "manifests"
    / "shared_eval_seed_manifest.csv"
)
MODE_CHOICES = ("tune", "logical", "embedding", "both")

LOGICAL_PROJECTED_METHODS = (
    "projected_pegasus_unb_pen",
    "projected_chimera_unb_pen",
    "projected_zephyr_unb_pen",
)
LOGICAL_METHODS = ("unbalanced", *LOGICAL_PROJECTED_METHODS)
EMBEDDING_PROJECTED_METHOD = "projected_topology_unb_pen"
EMBEDDING_METHODS = (
    "unbalanced",
    EMBEDDING_PROJECTED_METHOD,
)
UP_PROJECTION_TARGET = "raw_tuned_unbalanced_row"

LOGICAL_METHOD_TO_SUMMARY = {
    "projected_pegasus_unb_pen": "projected_pegasus",
    "projected_chimera_unb_pen": "projected_chimera",
    "projected_zephyr_unb_pen": "projected_zephyr",
}
SUMMARY_METHOD_TO_LOGICAL = {
    summary_method: compare_method
    for compare_method, summary_method in LOGICAL_METHOD_TO_SUMMARY.items()
}
HARDWARE_TO_SUMMARY_METHOD = {
    "chimera": "projected_chimera",
    "pegasus": "projected_pegasus",
    "zephyr": "projected_zephyr",
}
HARDWARE_ORDER = {"chimera": 0, "pegasus": 1, "zephyr": 2}
ALL_FAMILY_ORDER = tuple(default_family_sizes().keys())
LOGICAL_METHOD_ORDER = {
    method: index
    for index, method in enumerate(LOGICAL_METHODS)
}
EMBEDDING_METHOD_ORDER = {
    method: index
    for index, method in enumerate(EMBEDDING_METHODS)
}


def _log(message: str) -> None:
    print(
        f"[compare_up_projection] {message}",
        file=sys.stderr,
        flush=True,
    )


def _default_tuning_sizes() -> dict[str, int]:
    """Return the anchor size used to tune each family."""
    return {
        family: int(DEFAULT_TUNING_SIZES[family])
        for family in ALL_FAMILY_ORDER
        if family in DEFAULT_TUNING_SIZES
    }


def _projected_up_tuning_summary_path(
    tuning_dir: Path,
) -> Path:
    """Return the experiment-9 projected-UP summary path."""
    return (
        Path(tuning_dir)
        / "projected_penalty_tuning_summary.csv"
    )


def _projected_up_selection_path(tuning_dir: Path) -> Path:
    """Return the experiment-9 projected-UP selection summary path."""
    return (
        Path(tuning_dir)
        / "projected_combo_selection_summary.csv"
    )


def _unbalanced_summary_path(tuning_dir: Path) -> Path:
    """Return the shared UP tuning summary path."""
    return (
        Path(tuning_dir)
        / "unbalanced_penalty_tuning_summary.csv"
    )


def _base_projected_summary_path(tuning_dir: Path) -> Path:
    """Return the base projected summary used for combo selection."""
    return (
        Path(tuning_dir)
        / "projected_penalty_tuning_summary.csv"
    )


def _load_compare_unb_pen_tuning_inputs(
    *,
    tuning_dir: Path,
    projected_up_tuning_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], Any],
    Path,
    Path,
]:
    """Load shared UP parameters plus experiment-9 projected tuning rows."""
    unbalanced_summary_path = _unbalanced_summary_path(
        tuning_dir
    )
    projected_up_summary_path = (
        _projected_up_tuning_summary_path(
            projected_up_tuning_dir
        )
    )
    if not unbalanced_summary_path.exists():
        raise FileNotFoundError(
            "missing UP tuning summary: "
            f"{unbalanced_summary_path}"
        )
    if not projected_up_summary_path.exists():
        raise FileNotFoundError(
            "missing projected-UP tuning summary: "
            f"{projected_up_summary_path}; run "
            "`python -m experiments.compare_up_projection --mode tune` first"
        )
    return (
        load_tuned_unbalanced_parameters(
            unbalanced_summary_path
        ),
        load_selected_projected_configs(
            projected_up_summary_path
        ),
        unbalanced_summary_path,
        projected_up_summary_path,
    )


def _projected_multiplier_initial_point(
    tuned_up: Any,
    *,
    has_equality: bool,
    param_min: float,
) -> np.ndarray:
    """Build the projected-multiplier warm start from tuned UP weights."""
    lower = max(float(param_min), 1e-8)
    if has_equality:
        equality_multiplier = (
            lower
            if tuned_up.up_equality_multiplier is None
            else max(
                float(tuned_up.up_equality_multiplier),
                lower,
            )
        )
        return np.asarray(
            [
                equality_multiplier,
                max(
                    float(
                        tuned_up.up_inequality_multiplier
                    ),
                    lower,
                ),
            ],
            dtype=float,
        )
    return np.asarray(
        [
            max(
                float(tuned_up.up_inequality_multiplier),
                lower,
            )
        ],
        dtype=float,
    )


def _prepare_unb_pen_projected_tuning_instances(
    *,
    family: str,
    anchor_size: int,
    problems: list[BLP],
    projection_summary_method: str,
    selected_config: Any,
    tuned_up: Any,
    base_seed: int,
    pegasus_size: int,
    sample_cap_log2: int,
    chunk_size: int,
    projection_reg: float,
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ],
) -> list[TuningInstance]:
    """Cache exact projected-UP tuning energies for one family/method."""
    instances: list[TuningInstance] = []
    projection_method = (
        selected_config.projection_method
        or projection_summary_method
    )
    for instance_index, problem in enumerate(problems):
        set_tuning_progress_status(
            stage="prepare_projected_tuning",
            activity="precomputing projected-UP energies",
            family=family,
            size=anchor_size,
            instance_index=instance_index,
            total_instances=len(problems),
            method=projection_summary_method,
            measure=selected_config.measure_name,
            template="quadratic",
            detail="precomputing exact tuning tables",
        )
        _, optimum_state_indices = tuning_optimum_states(
            problem,
            chunk_size=chunk_size,
        )
        cache_key = (
            embedding_mod._projected_components_cache_key(
                projection_method=projection_method,
                family=family,
                size=anchor_size,
                instance_index=instance_index,
                measure_name=selected_config.measure_name,
                measure_lam=DEFAULT_MEASURE_LAM,
                penalty_template="quadratic",
                penalty_template_kwargs={
                    "lambda1": float(
                        tuned_up.up_lambda1_shape
                    ),
                    "lambda2": float(
                        tuned_up.up_lambda2_shape
                    ),
                },
                standardize=bool(
                    selected_config.projected_standardize
                ),
            )
        )
        if cache_key not in components_cache:
            pair_edges = baseline_mod._projection_pair_edges(
                problem,
                projection_method=projection_summary_method,
                pegasus_size=pegasus_size,
            )
            components_cache[cache_key] = (
                _build_unb_pen_projected_components(
                    problem,
                    pair_edges=pair_edges,
                    family=family,
                    size=anchor_size,
                    instance_index=instance_index,
                    base_seed=base_seed,
                    measure_name=selected_config.measure_name,
                    measure_lam=DEFAULT_MEASURE_LAM,
                    sample_cap_log2=sample_cap_log2,
                    reg=projection_reg,
                    standardize=bool(
                        selected_config.projected_standardize
                    ),
                    lambda1_shape=float(
                        tuned_up.up_lambda1_shape
                    ),
                    lambda2_shape=float(
                        tuned_up.up_lambda2_shape
                    ),
                )
            )
        components = components_cache[cache_key]
        equality_energies, inequality_energies = (
            tuning_projected_component_energies(
                problem,
                components,
                chunk_size=chunk_size,
            )
        )
        instances.append(
            TuningInstance(
                num_states=problem.num_states,
                optimum_state_indices=optimum_state_indices,
                objective_energies=tuning_objective_energies(
                    problem,
                    chunk_size=chunk_size,
                ),
                equality_energies=equality_energies,
                inequality_energies=inequality_energies,
            )
        )
    return instances


def build_unbalanced_inequality_target_matrix(
    problem: BLP,
    bitstrings: np.ndarray,
    *,
    lambda1_shape: float,
    lambda2_shape: float,
) -> np.ndarray:
    """Return one sampled UP-row target column per inequality."""
    bits = np.asarray(bitstrings, dtype=float)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if int(problem.num_inequalities) == 0:
        return np.zeros((bits.shape[0], 0), dtype=float)

    spins = binary_to_spin_states(bits)
    columns: list[np.ndarray] = []
    for coeffs, rhs in zip(
        problem.A, problem.b, strict=True
    ):
        violation = IdealPenalty.violation(
            spins,
            np.asarray(coeffs, dtype=float),
            float(rhs),
        )
        columns.append(
            float(lambda1_shape) * violation
            + float(lambda2_shape) * (violation**2)
        )
    return np.column_stack(columns)


def _projected_up_qubo(
    problem: BLP,
    components: ProjectedPenaltyComponents,
    *,
    equality_multiplier: float | None,
    inequality_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose one projected-UP QUBO with tuned global multipliers."""
    terms = [objective_terms(problem)]
    if (
        problem.num_equalities
        and equality_multiplier is not None
    ):
        if abs(float(equality_multiplier)) > 1e-12:
            terms.append(
                scale_terms(
                    components.equality_terms,
                    float(equality_multiplier),
                )
            )
    if (
        problem.num_inequalities
        and abs(float(inequality_multiplier)) > 1e-12
    ):
        terms.append(
            scale_terms(
                components.inequality_terms,
                float(inequality_multiplier),
            )
        )
    total = add_qubo_terms(*terms)
    return total.quadratic, total.linear, total.const


def _unbalanced_qubo(
    problem: BLP,
    tuned_up: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the deployed UP comparison QUBO."""
    components = build_unbalanced_components(
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
    return build_unbalanced_qubo_from_components(
        problem,
        components,
        equality_multiplier=tuned_up.up_equality_multiplier,
        inequality_multiplier=float(
            tuned_up.up_inequality_multiplier
        ),
    )


def _projection_regime_fields_for_method(
    method: str,
    *,
    hardware_family: str | None = None,
) -> dict[str, object]:
    """Return normalized projection-regime fields for experiment-9 methods."""
    if method == "unbalanced":
        return projection_regime_fields("unbalanced")
    if method in LOGICAL_METHOD_TO_SUMMARY:
        return projection_regime_fields(
            LOGICAL_METHOD_TO_SUMMARY[method]
        )
    if method == EMBEDDING_PROJECTED_METHOD:
        if hardware_family is None:
            raise ValueError(
                "hardware_family is required for projected_topology_unb_pen rows"
            )
        return projection_regime_fields(
            "projected_topology",
            hardware_family=hardware_family,
        )
    raise ValueError(
        f"unknown experiment-9 comparison method: {method}"
    )


def _unbalanced_target_builder(
    problem: BLP,
    sample_bits: np.ndarray,
    *,
    lambda1_shape: float,
    lambda2_shape: float,
) -> np.ndarray:
    return build_unbalanced_inequality_target_matrix(
        problem,
        sample_bits,
        lambda1_shape=lambda1_shape,
        lambda2_shape=lambda2_shape,
    )


def _build_unb_pen_projected_components(
    problem: BLP,
    *,
    pair_edges: list[tuple[int, int]],
    family: str,
    size: int,
    instance_index: int,
    base_seed: int,
    measure_name: str,
    measure_lam: float,
    sample_cap_log2: int,
    reg: float,
    standardize: bool,
    lambda1_shape: float,
    lambda2_shape: float,
) -> ProjectedPenaltyComponents:
    """Fit projected components using the tuned raw UP row as target."""
    sample_size = projection_sample_size(
        problem, sample_cap_log2
    )
    sample_rng = build_child_rng(
        base_seed,
        2_000,
        FAMILY_CODES[family],
        int(size),
        int(instance_index),
    )
    return build_projected_components(
        problem,
        pair_edges=pair_edges,
        sample_size=sample_size,
        sample_rng=sample_rng,
        measure_name=measure_name,
        measure_lam=measure_lam,
        penalty_template="quadratic",
        penalty_template_kwargs={
            "lambda1": float(lambda1_shape),
            "lambda2": float(lambda2_shape),
        },
        reg=reg,
        standardize=standardize,
        build_projection_sampling_catalog=build_projection_sampling_catalog,
        sample_projection_states_with_inequality_support=(
            sample_projection_states_with_inequality_support
        ),
        build_unit_equality_constraint_qubos=build_unit_equality_constraint_qubos,
        combine_constraint_terms=combine_constraint_terms,
        project_penalty_values_importance=project_penalty_values_importance,
        hardware_topology_cls=HardwareTopology,
        ideal_penalty_cls=IdealPenalty,
        binary_to_spin_states=binary_to_spin_states,
        is_complete_pair_edge_set=is_complete_pair_edge_set,
        inequality_target_matrix_builder=lambda cur_problem, sample_bits: (
            _unbalanced_target_builder(
                cur_problem,
                sample_bits,
                lambda1_shape=lambda1_shape,
                lambda2_shape=lambda2_shape,
            )
        ),
    )


def _logical_output_dir(base_output_dir: Path) -> Path:
    return Path(base_output_dir) / "logical"


def _embedding_output_dir(
    base_output_dir: Path,
    *,
    chain_strength_fraction: float,
) -> Path:
    suffix = embedding_mod._chain_strength_fraction_dirname(
        float(chain_strength_fraction)
    )
    return Path(base_output_dir) / "embedding" / suffix


def _family_sort_key(family: object) -> int:
    normalized = str(family)
    try:
        return ALL_FAMILY_ORDER.index(normalized)
    except ValueError:
        return len(ALL_FAMILY_ORDER)


def _logical_cop_sort_key(
    row: dict[str, object],
) -> tuple[int, int, int, int]:
    return (
        _family_sort_key(row["family"]),
        int(row["size"]),
        int(row["instance_index"]),
        LOGICAL_METHOD_ORDER[str(row["method"])],
    )


def _embedding_cop_sort_key(
    row: dict[str, object],
) -> tuple[int, int, int, int, int, float]:
    return (
        _family_sort_key(row["family"]),
        int(row["size"]),
        int(row["instance_index"]),
        HARDWARE_ORDER[str(row["hardware_family"])],
        EMBEDDING_METHOD_ORDER[str(row["method"])],
        float(row["sqa_chain_strength_fraction"]),
    )


def _merge_rows(
    path: Path,
    rows: list[dict[str, object]],
    *,
    force: bool,
    key_fields: tuple[str, ...],
    sort_key,
) -> list[dict[str, object]]:
    if force:
        merged_rows = list(rows)
        merged_rows.sort(key=sort_key)
        return merged_rows
    return merge_csv_rows(
        path,
        rows,
        key_fields=key_fields,
        sort_key=sort_key,
    )


def _logical_projection_config_row(
    *,
    family: str,
    size: int,
    method: str,
    selected_config: Any,
    tuned_up: Any,
    tuning_source: Path,
) -> dict[str, object]:
    return {
        "family": family,
        "size": int(size),
        "method": method,
        "projection_summary_method": LOGICAL_METHOD_TO_SUMMARY[
            method
        ],
        "projection_method": (
            selected_config.projection_method
            or LOGICAL_METHOD_TO_SUMMARY[method]
        ),
        "projection_measure": selected_config.measure_name,
        "projection_selection_mode": selected_config.selection_mode,
        "projection_selection_source": selected_config.selection_source,
        "projection_candidate_rank": int(
            selected_config.candidate_rank
        ),
        "projected_standardize": bool(
            selected_config.projected_standardize
        ),
        "projection_target": UP_PROJECTION_TARGET,
        "projected_equality_multiplier": (
            selected_config.tuning.equality_multiplier
        ),
        "projected_inequality_multiplier": (
            selected_config.tuning.inequality_multiplier
        ),
        "projected_initializer_equality_multiplier": (
            tuned_up.up_equality_multiplier
        ),
        "projected_initializer_inequality_multiplier": (
            tuned_up.up_inequality_multiplier
        ),
        "projection_target_lambda1_shape": tuned_up.up_lambda1_shape,
        "projection_target_lambda2_shape": tuned_up.up_lambda2_shape,
        "penalty_config_source": str(
            tuning_source.resolve()
        ),
    }


def _embedding_projection_config_row(
    *,
    hardware_family: str,
    hardware_size: int,
    family: str,
    size: int,
    selected_config: Any,
    tuned_up: Any,
    projected_summary_path: Path,
    chain_strength_fraction: float,
) -> dict[str, object]:
    summary_method = HARDWARE_TO_SUMMARY_METHOD[
        hardware_family
    ]
    return {
        "hardware_family": hardware_family,
        "hardware_size": int(hardware_size),
        "family": family,
        "size": int(size),
        "method": EMBEDDING_PROJECTED_METHOD,
        "projection_summary_method": summary_method,
        "projection_method": selected_config.projection_method
        or summary_method,
        "projection_measure": selected_config.measure_name,
        "projection_selection_mode": selected_config.selection_mode,
        "projection_selection_source": selected_config.selection_source,
        "projection_candidate_rank": int(
            selected_config.candidate_rank
        ),
        "projected_standardize": bool(
            selected_config.projected_standardize
        ),
        "projection_target": UP_PROJECTION_TARGET,
        "projected_equality_multiplier": (
            selected_config.tuning.equality_multiplier
        ),
        "projected_inequality_multiplier": (
            selected_config.tuning.inequality_multiplier
        ),
        "projected_initializer_equality_multiplier": (
            tuned_up.up_equality_multiplier
        ),
        "projected_initializer_inequality_multiplier": (
            tuned_up.up_inequality_multiplier
        ),
        "projection_target_lambda1_shape": tuned_up.up_lambda1_shape,
        "projection_target_lambda2_shape": tuned_up.up_lambda2_shape,
        "sqa_chain_strength_fraction": float(
            chain_strength_fraction
        ),
        "penalty_config_source": str(
            projected_summary_path.resolve()
        ),
    }


def _unbalanced_config_row(
    *,
    family: str,
    size: int,
    tuned_up: Any,
    source_path: Path,
) -> dict[str, object]:
    return {
        "family": family,
        "size": int(size),
        "normalization_regime": (
            tuned_up.normalization_regime
            or UP_NORMALIZATION_REGIME
        ),
        "per_constraint_standardization": (
            True
            if tuned_up.per_constraint_standardization
            is None
            else bool(
                tuned_up.per_constraint_standardization
            )
        ),
        "up_equality_multiplier": tuned_up.up_equality_multiplier,
        "up_inequality_multiplier": tuned_up.up_inequality_multiplier,
        "up_lambda1_shape": tuned_up.up_lambda1_shape,
        "up_lambda2_shape": tuned_up.up_lambda2_shape,
        "up_lambda_gauge": tuned_up.up_lambda_gauge,
        "up_global_multiplier": tuned_up.global_multiplier,
        "up_lambda0": tuned_up.lambda0,
        "up_lambda1": tuned_up.lambda1,
        "up_lambda2": tuned_up.lambda2,
        "base_parameter_source": tuned_up.base_parameter_source,
        "penalty_config_source": str(source_path.resolve()),
    }


def _write_logical_outputs(
    *,
    output_dir: Path,
    active_families: tuple[str, ...],
    cop_rows: list[dict[str, object]],
    projection_config_rows: list[dict[str, object]],
    unbalanced_config_rows: list[dict[str, object]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cop_metadata_path = (
        output_dir / "cop_metadata_summary.csv"
    )
    cop_instance_path = (
        output_dir / "cop_instance_summary.csv"
    )
    projection_path = (
        output_dir / "projection_config_summary.csv"
    )
    unbalanced_path = (
        output_dir / "unbalanced_penalty_config_summary.csv"
    )

    metadata_rows = baseline_mod._cop_metadata_rows(
        active_families
    )
    for row in metadata_rows:
        row["experiment"] = "compare_up_projection"
        row["projection_target"] = UP_PROJECTION_TARGET
    write_rows_csv(cop_metadata_path, metadata_rows)

    merged_projection_rows = _merge_rows(
        projection_path,
        projection_config_rows,
        force=force,
        key_fields=("family", "size", "method"),
        sort_key=lambda row: (
            _family_sort_key(row["family"]),
            int(row["size"]),
            LOGICAL_METHOD_ORDER[str(row["method"])],
        ),
    )
    write_rows_csv(projection_path, merged_projection_rows)

    merged_unbalanced_rows = _merge_rows(
        unbalanced_path,
        unbalanced_config_rows,
        force=force,
        key_fields=("family", "size"),
        sort_key=lambda row: (
            _family_sort_key(row["family"]),
            int(row["size"]),
        ),
    )
    write_rows_csv(unbalanced_path, merged_unbalanced_rows)

    merged_cop_rows = _merge_rows(
        cop_instance_path,
        cop_rows,
        force=force,
        key_fields=(
            "family",
            "size",
            "instance_index",
            "method",
        ),
        sort_key=_logical_cop_sort_key,
    )
    write_rows_csv(cop_instance_path, merged_cop_rows)
    robust_path, legacy_path = write_aggregate_outputs(
        merged_cop_rows,
        output_dir=output_dir,
        mode="baseline",
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
        write_legacy_mean_std=True,
    )
    _log(f"wrote {cop_metadata_path}")
    _log(f"wrote {projection_path}")
    _log(f"wrote {unbalanced_path}")
    _log(f"wrote {cop_instance_path}")
    _log(f"wrote {robust_path}")
    if legacy_path is not None:
        _log(f"wrote {legacy_path}")


def _write_embedding_outputs(
    *,
    output_dir: Path,
    cop_rows: list[dict[str, object]],
    projection_config_rows: list[dict[str, object]],
    unbalanced_config_rows: list[dict[str, object]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cop_instance_path = (
        output_dir / "cop_instance_summary.csv"
    )
    projection_path = (
        output_dir / "projection_config_summary.csv"
    )
    unbalanced_path = (
        output_dir / "unbalanced_penalty_config_summary.csv"
    )

    merged_projection_rows = _merge_rows(
        projection_path,
        projection_config_rows,
        force=force,
        key_fields=(
            "hardware_family",
            "family",
            "size",
            "method",
        ),
        sort_key=lambda row: (
            _family_sort_key(row["family"]),
            int(row["size"]),
            HARDWARE_ORDER[str(row["hardware_family"])],
            EMBEDDING_METHOD_ORDER[str(row["method"])],
        ),
    )
    write_rows_csv(projection_path, merged_projection_rows)

    merged_unbalanced_rows = _merge_rows(
        unbalanced_path,
        unbalanced_config_rows,
        force=force,
        key_fields=("family", "size"),
        sort_key=lambda row: (
            _family_sort_key(row["family"]),
            int(row["size"]),
        ),
    )
    write_rows_csv(unbalanced_path, merged_unbalanced_rows)

    merged_cop_rows = _merge_rows(
        cop_instance_path,
        cop_rows,
        force=force,
        key_fields=(
            "hardware_family",
            "family",
            "size",
            "instance_index",
            "method",
            "sqa_chain_strength_fraction",
        ),
        sort_key=_embedding_cop_sort_key,
    )
    write_rows_csv(cop_instance_path, merged_cop_rows)
    robust_path, legacy_path = write_aggregate_outputs(
        merged_cop_rows,
        output_dir=output_dir,
        mode="embedding",
        bootstrap_resamples=int(bootstrap_resamples),
        bootstrap_seed=int(bootstrap_seed),
        write_legacy_mean_std=True,
    )
    _log(f"wrote {projection_path}")
    _log(f"wrote {unbalanced_path}")
    _log(f"wrote {cop_instance_path}")
    _log(f"wrote {robust_path}")
    if legacy_path is not None:
        _log(f"wrote {legacy_path}")


def run_tuning_mode(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    """Tune projected-UP global multipliers for experiment 9."""
    active_families = selected_families_from_args(args)
    tuning_sizes = _default_tuning_sizes()
    tuning_family_sizes = {
        family: [tuning_sizes[family]]
        for family in active_families
    }
    tuning_dir = Path(args.tuning_dir)
    projected_up_tuning_dir = Path(
        args.projected_up_tuning_dir
    )
    base_projected_summary_path = (
        _base_projected_summary_path(tuning_dir)
    )
    base_unbalanced_summary_path = _unbalanced_summary_path(
        tuning_dir
    )
    if not base_projected_summary_path.exists():
        raise FileNotFoundError(
            "missing base projected tuning summary: "
            f"{base_projected_summary_path}"
        )
    if not base_unbalanced_summary_path.exists():
        raise FileNotFoundError(
            "missing base UP tuning summary: "
            f"{base_unbalanced_summary_path}"
        )

    base_selected_configs = load_selected_projected_configs(
        base_projected_summary_path
    )
    tuned_unbalanced_params = (
        load_tuned_unbalanced_parameters(
            base_unbalanced_summary_path
        )
    )
    summary_methods = tuple(
        HARDWARE_TO_SUMMARY_METHOD.values()
    )
    missing_up_families = [
        family
        for family in active_families
        if family not in tuned_unbalanced_params
    ]
    if missing_up_families:
        raise RuntimeError(
            "missing base UP tuning rows for "
            f"{', '.join(missing_up_families)} in "
            f"{base_unbalanced_summary_path}"
        )
    missing_projected_configs = [
        f"{family}/{method}"
        for family in active_families
        for method in summary_methods
        if (family, method) not in base_selected_configs
    ]
    if missing_projected_configs:
        raise RuntimeError(
            "missing base projected selections for "
            f"{', '.join(missing_projected_configs)} in "
            f"{base_projected_summary_path}"
        )

    total_tuning_jobs = len(active_families) * len(
        summary_methods
    )
    tui = activate_progress_ui(
        mode=str(args.progress_ui),
        totals=BaselineComparisonProgressTotals(
            tuning_jobs=total_tuning_jobs,
        ),
    )
    # Prefer the manifest-sidecar CPLEX CSV when a manifest is used or
    # when the project-default shared manifest exists and no explicit
    # instance manifest was provided on the CLI.
    instance_manifest_arg = getattr(args, "instance_manifest", None)
    if instance_manifest_arg is None:
        default_manifest = (
            EXPERIMENTS_DIR / "manifests" / "shared_eval_seed_manifest.csv"
        )
        if default_manifest.exists():
            instance_manifest_arg = default_manifest
            tuning_log(
                f"auto-using instance manifest {instance_manifest_arg} (found project default)"
            )
    # Keep args.instance_manifest in sync with the resolved choice so
    # downstream code uses the same manifest as the CPLEX resolution would.
    args.instance_manifest = instance_manifest_arg
    get_problem_batch = problem_batch_getter(
        base_seed=int(args.seed),
        num_instances=int(args.num_instances),
        instance_manifest=args.instance_manifest,
        family_sizes=tuning_family_sizes,
    )
    projected_tuning_rows: list[dict[str, object]] = []
    projected_selection_rows: list[dict[str, object]] = []
    components_cache: dict[
        tuple[object, ...],
        ProjectedPenaltyComponents,
    ] = {}
    summary_method_order = {
        method: index
        for index, method in enumerate(summary_methods)
    }

    try:
        ensure_run_metadata(
            projected_up_tuning_dir,
            {
                "experiment": "compare_up_projection",
                "mode": "tune",
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
                "base_tuning_dir": str(
                    tuning_dir.resolve()
                ),
                "base_projected_summary_path": str(
                    base_projected_summary_path.resolve()
                ),
                "base_unbalanced_summary_path": str(
                    base_unbalanced_summary_path.resolve()
                ),
                "projection_target": UP_PROJECTION_TARGET,
                "pegasus_size": int(args.pegasus_size),
            },
            force=bool(args.force),
        )
        tuning_log(
            "tuning projected-UP multipliers using combo selections from "
            f"{base_projected_summary_path}"
        )
        tuning_log(
            "using UP initialization from "
            f"{base_unbalanced_summary_path}"
        )
        for family in active_families:
            anchor_size = tuning_sizes[family]
            problems = get_problem_batch(
                family, anchor_size
            )
            tuned_up = tuned_unbalanced_params[family]
            tuning_log(
                f"tuning projected-UP multipliers for {family} "
                f"on anchor size {anchor_size} using "
                f"{len(problems)} instance(s)"
            )
            has_equality = any(
                problem.num_equalities > 0
                for problem in problems
            )
            initial_point = (
                _projected_multiplier_initial_point(
                    tuned_up,
                    has_equality=has_equality,
                    param_min=DEFAULT_TUNING_MIN,
                )
            )
            for summary_method in summary_methods:
                selected_config = base_selected_configs[
                    (family, summary_method)
                ]
                set_tuning_progress_status(
                    stage="tuning_projected",
                    activity="tuning projected-UP penalty",
                    family=family,
                    size=anchor_size,
                    instance_index=None,
                    total_instances=len(problems),
                    method=summary_method,
                    measure=selected_config.measure_name,
                    template="quadratic",
                    detail="warming from tuned UP multipliers",
                )
                instances = _prepare_unb_pen_projected_tuning_instances(
                    family=family,
                    anchor_size=anchor_size,
                    problems=problems,
                    projection_summary_method=summary_method,
                    selected_config=selected_config,
                    tuned_up=tuned_up,
                    base_seed=int(args.seed),
                    pegasus_size=int(args.pegasus_size),
                    sample_cap_log2=DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    projection_reg=DEFAULT_PROJECTION_REG,
                    components_cache=components_cache,
                )
                tuned, records, _, _ = (
                    tune_precomputed_projected_multipliers(
                        summary_method,
                        family,
                        instances,
                        anchor_size=anchor_size,
                        has_equality=has_equality,
                        param_min=DEFAULT_TUNING_MIN,
                        start_points_per_dim=DEFAULT_TUNING_NM_START_POINTS,
                        nelder_mead_maxiter=DEFAULT_TUNING_NM_MAXITER,
                        nelder_mead_xatol=DEFAULT_TUNING_NM_XATOL,
                        nelder_mead_fatol=DEFAULT_TUNING_NM_FATOL,
                        clip_minimum=False,
                        initial_points=[initial_point],
                    )
                )
                advance_tuning_progress_counter(
                    "tuning_jobs", 1
                )

                enriched_records: list[
                    dict[str, object]
                ] = []
                for record in records:
                    row = dict(record)
                    row.update(
                        {
                            "method": summary_method,
                            "family": family,
                            "anchor_size": anchor_size,
                            "projection_method": (
                                selected_config.projection_method
                                or summary_method
                            ),
                            "projection_measure": (
                                selected_config.measure_name
                            ),
                            "projection_selection_mode": (
                                selected_config.selection_mode
                            ),
                            "projection_selection_source": (
                                selected_config.selection_source
                            ),
                            "projection_candidate_rank": int(
                                selected_config.candidate_rank
                            ),
                            "projected_standardize": bool(
                                selected_config.projected_standardize
                            ),
                            "projection_target": UP_PROJECTION_TARGET,
                            "projection_target_lambda1_shape": (
                                tuned_up.up_lambda1_shape
                            ),
                            "projection_target_lambda2_shape": (
                                tuned_up.up_lambda2_shape
                            ),
                            "projected_initializer_equality_multiplier": (
                                tuned_up.up_equality_multiplier
                            ),
                            "projected_initializer_inequality_multiplier": (
                                tuned_up.up_inequality_multiplier
                            ),
                            **template_row_fields(
                                "quadratic",
                                {
                                    "lambda1": float(
                                        tuned_up.up_lambda1_shape
                                    ),
                                    "lambda2": float(
                                        tuned_up.up_lambda2_shape
                                    ),
                                },
                            ),
                        }
                    )
                    enriched_records.append(row)
                records_path = (
                    projected_up_tuning_dir
                    / "projected_tuning_records"
                    / f"{family}_{summary_method}_anchor_{anchor_size}.csv"
                )
                write_rows_csv(
                    records_path, enriched_records
                )
                tuning_log(f"wrote {records_path}")

                summary_row = tuned.as_row()
                summary_row.update(
                    {
                        "projection_method": (
                            selected_config.projection_method
                            or summary_method
                        ),
                        "projection_measure": selected_config.measure_name,
                        "projection_selection_mode": (
                            selected_config.selection_mode
                        ),
                        "projection_selection_source": (
                            selected_config.selection_source
                        ),
                        "projection_candidate_rank": int(
                            selected_config.candidate_rank
                        ),
                        "projected_standardize": bool(
                            selected_config.projected_standardize
                        ),
                        "selected_projection_combo": True,
                        "projection_target": UP_PROJECTION_TARGET,
                        "projection_target_lambda1_shape": (
                            tuned_up.up_lambda1_shape
                        ),
                        "projection_target_lambda2_shape": (
                            tuned_up.up_lambda2_shape
                        ),
                        "projected_initializer_equality_multiplier": (
                            tuned_up.up_equality_multiplier
                        ),
                        "projected_initializer_inequality_multiplier": (
                            tuned_up.up_inequality_multiplier
                        ),
                        **template_row_fields(
                            "quadratic",
                            {
                                "lambda1": float(
                                    tuned_up.up_lambda1_shape
                                ),
                                "lambda2": float(
                                    tuned_up.up_lambda2_shape
                                ),
                            },
                        ),
                    }
                )
                projected_tuning_rows.append(
                    dict(summary_row)
                )
                projected_selection_rows.append(
                    dict(summary_row)
                )
                tuning_log(
                    f"selected projected-UP multipliers for "
                    f"{family}/{summary_method}: equality="
                    f"{tuned.equality_multiplier:.6g}, inequality="
                    f"{tuned.inequality_multiplier:.6g}, objective="
                    f"{tuned.objective_value:.6g}"
                )

        set_tuning_progress_status(
            stage="writing",
            activity="writing tuning outputs",
            family=None,
            size=None,
            instance_index=None,
            total_instances=None,
            method=None,
            measure=None,
            template=None,
            detail="writing projected-UP tuning summaries",
        )
        projected_tuning_path = (
            _projected_up_tuning_summary_path(
                projected_up_tuning_dir
            )
        )
        projected_sort_key = lambda row: (
            _family_sort_key(row["family"]),
            summary_method_order[str(row["method"])],
        )
        if args.force:
            merged_projected_rows = list(
                projected_tuning_rows
            )
            merged_projected_rows.sort(
                key=projected_sort_key
            )
        else:
            merged_projected_rows = merge_csv_rows(
                projected_tuning_path,
                projected_tuning_rows,
                key_fields=("family", "method"),
                sort_key=projected_sort_key,
            )
        write_projected_tuning_summary_csv(
            projected_tuning_path,
            merged_projected_rows,
        )
        tuning_log(f"wrote {projected_tuning_path}")

        projected_selection_path = (
            _projected_up_selection_path(
                projected_up_tuning_dir
            )
        )
        if args.force:
            merged_selection_rows = list(
                projected_selection_rows
            )
            merged_selection_rows.sort(
                key=projected_sort_key
            )
        else:
            merged_selection_rows = merge_csv_rows(
                projected_selection_path,
                projected_selection_rows,
                key_fields=("family", "method"),
                sort_key=projected_sort_key,
            )
        write_rows_csv(
            projected_selection_path, merged_selection_rows
        )
        tuning_log(f"wrote {projected_selection_path}")
        return {
            "projected_tuning_rows": projected_tuning_rows,
            "projected_selection_rows": projected_selection_rows,
        }
    finally:
        deactivate_progress_ui(tui)


def run_logical_mode(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    """Run the logical-SQA variant of experiment 9."""
    active_families = selected_families_from_args(args)
    family_sizes = selected_family_sizes_from_args(args)
    output_dir = _logical_output_dir(Path(args.output_dir))
    tuning_dir = Path(args.tuning_dir)
    projected_up_tuning_dir = Path(
        args.projected_up_tuning_dir
    )
    (
        tuned_unbalanced_params,
        projected_up_configs,
        unbalanced_summary_path,
        projected_up_summary_path,
    ) = _load_compare_unb_pen_tuning_inputs(
        tuning_dir=tuning_dir,
        projected_up_tuning_dir=projected_up_tuning_dir,
    )
    # Prefer the manifest-sidecar CPLEX CSV when a manifest is used or
    # when the project-default shared manifest exists and no explicit
    # instance manifest was provided on the CLI.
    instance_manifest_arg = getattr(args, "instance_manifest", None)
    if instance_manifest_arg is None:
        default_manifest = (
            EXPERIMENTS_DIR / "manifests" / "shared_eval_seed_manifest.csv"
        )
        if default_manifest.exists():
            instance_manifest_arg = default_manifest
            _log(
                f"auto-using instance manifest {instance_manifest_arg} (found project default)"
            )
    cplex_reference_path = resolve_cplex_reference_path(
        getattr(args, "cplex_reference_csv", None),
        instance_manifest=instance_manifest_arg,
    )
    # Keep args.instance_manifest in sync with the resolved choice so the
    # rest of the run (problem generation, metadata) uses the same manifest.
    args.instance_manifest = instance_manifest_arg
    cplex_reference_index = load_cplex_reference_index(
        cplex_reference_path
    )
    fixed_schedule = baseline_mod._fixed_sqa_schedule()
    get_problem_batch = problem_batch_getter(
        base_seed=int(args.seed),
        num_instances=int(args.num_instances),
        instance_manifest=args.instance_manifest,
        family_sizes=family_sizes,
    )

    ensure_run_metadata(
        output_dir,
        {
            "experiment": "compare_up_projection",
            "mode": "logical",
            "base_seed": int(args.seed),
            "num_instances": int(args.num_instances),
            "instance_manifest": (
                None
                if args.instance_manifest is None
                else str(
                    Path(args.instance_manifest).resolve()
                )
            ),
            "cplex_reference_csv": str(
                cplex_reference_path.resolve()
            ),
            "tuning_dir": str(tuning_dir.resolve()),
            "projected_up_tuning_dir": str(
                projected_up_tuning_dir.resolve()
            ),
            "unbalanced_summary_path": str(
                unbalanced_summary_path.resolve()
            ),
            "projected_up_summary_path": str(
                projected_up_summary_path.resolve()
            ),
            "projection_target": UP_PROJECTION_TARGET,
            "sqa_schedule_id": fixed_schedule.schedule_id,
            "sqa_schedule_kind": fixed_schedule.schedule_kind,
            "sqa_schedule_total_time": fixed_schedule.total_schedule_time,
        },
        force=bool(args.force),
    )

    missing_up_families = [
        family
        for family in active_families
        if family not in tuned_unbalanced_params
    ]
    if missing_up_families:
        raise RuntimeError(
            "missing unbalanced tuning rows for "
            f"{', '.join(missing_up_families)} in "
            f"{unbalanced_summary_path}"
        )

    missing_projected_configs = [
        f"{family}/{summary_method}"
        for family in active_families
        for summary_method in SUMMARY_METHOD_TO_LOGICAL
        if (family, summary_method)
        not in projected_up_configs
    ]
    if missing_projected_configs:
        raise RuntimeError(
            "missing projected-UP tuning rows for "
            f"{', '.join(missing_projected_configs)} in "
            f"{projected_up_summary_path}"
        )

    cop_rows: list[dict[str, object]] = []
    projection_config_rows: list[dict[str, object]] = []
    unbalanced_config_rows: list[dict[str, object]] = []
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ] = {}
    for family in active_families:
        tuned_up = tuned_unbalanced_params[family]
        selected_config_map = {
            method: projected_up_configs[
                (family, LOGICAL_METHOD_TO_SUMMARY[method])
            ]
            for method in LOGICAL_PROJECTED_METHODS
        }
        for size in family_sizes[family]:
            unbalanced_config_rows.append(
                _unbalanced_config_row(
                    family=family,
                    size=size,
                    tuned_up=tuned_up,
                    source_path=unbalanced_summary_path,
                )
            )
            for method in LOGICAL_PROJECTED_METHODS:
                projection_config_rows.append(
                    _logical_projection_config_row(
                        family=family,
                        size=size,
                        method=method,
                        selected_config=selected_config_map[
                            method
                        ],
                        tuned_up=tuned_up,
                        tuning_source=projected_up_summary_path,
                    )
                )

            problems = get_problem_batch(family, size)
            _log(
                f"logical: evaluating {family} size {size} "
                f"with {len(problems)} instance(s)"
            )
            for instance_index, problem in enumerate(
                problems
            ):
                reference = (
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
                )
                method_qubos: dict[
                    str,
                    tuple[np.ndarray, np.ndarray, float],
                ] = {
                    "unbalanced": _unbalanced_qubo(
                        problem, tuned_up
                    )
                }
                projected_components_map: dict[
                    str, ProjectedPenaltyComponents
                ] = {}
                for method in LOGICAL_PROJECTED_METHODS:
                    selected_config = selected_config_map[
                        method
                    ]
                    summary_method = (
                        LOGICAL_METHOD_TO_SUMMARY[method]
                    )
                    cache_key = embedding_mod._projected_components_cache_key(
                        projection_method=(
                            selected_config.projection_method
                            or summary_method
                        ),
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        measure_name=selected_config.measure_name,
                        measure_lam=DEFAULT_MEASURE_LAM,
                        penalty_template="quadratic",
                        penalty_template_kwargs={
                            "lambda1": float(
                                tuned_up.up_lambda1_shape
                            ),
                            "lambda2": float(
                                tuned_up.up_lambda2_shape
                            ),
                        },
                        standardize=bool(
                            selected_config.projected_standardize
                        ),
                    )
                    if cache_key not in components_cache:
                        pair_edges = baseline_mod._projection_pair_edges(
                            problem,
                            projection_method=summary_method,
                            pegasus_size=int(
                                args.pegasus_size
                            ),
                        )
                        components_cache[cache_key] = (
                            _build_unb_pen_projected_components(
                                problem,
                                pair_edges=pair_edges,
                                family=family,
                                size=size,
                                instance_index=instance_index,
                                base_seed=int(args.seed),
                                measure_name=selected_config.measure_name,
                                measure_lam=DEFAULT_MEASURE_LAM,
                                sample_cap_log2=DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                                reg=DEFAULT_PROJECTION_REG,
                                standardize=bool(
                                    selected_config.projected_standardize
                                ),
                                lambda1_shape=float(
                                    tuned_up.up_lambda1_shape
                                ),
                                lambda2_shape=float(
                                    tuned_up.up_lambda2_shape
                                ),
                            )
                        )
                    components = components_cache[cache_key]
                    projected_components_map[method] = (
                        components
                    )
                    method_qubos[method] = (
                        _projected_up_qubo(
                            problem,
                            components,
                            equality_multiplier=tuned_up.up_equality_multiplier,
                            inequality_multiplier=float(
                                tuned_up.up_inequality_multiplier
                            ),
                        )
                    )

                for method in LOGICAL_METHODS:
                    quadratic, linear, const = method_qubos[
                        method
                    ]
                    normalization_scale = baseline_mod._qubo_normalization_scale(
                        quadratic,
                        linear,
                        num_variables=problem.num_variables,
                    )
                    annealer_input = baseline_mod._scaled_annealer_input(
                        problem,
                        quadratic,
                        linear,
                        const,
                        normalization_scale=normalization_scale,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                    )
                    sqa_metrics = baseline_mod._logical_annealer_baseline_result(
                        problem,
                        annealer_input,
                        reference=reference,
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        method=method,
                        base_seed=int(args.seed),
                        num_reads=DEFAULT_SQA_NUM_READS,
                        schedule=fixed_schedule,
                        atol=DEFAULT_REFERENCE_ATOL,
                    )

                    selected_config = (
                        None
                        if method == "unbalanced"
                        else selected_config_map[method]
                    )
                    projected_components = (
                        None
                        if method == "unbalanced"
                        else projected_components_map[
                            method
                        ]
                    )
                    cop_rows.append(
                        {
                            "family": family,
                            "size": size,
                            "instance_index": instance_index,
                            "method": method,
                            **_projection_regime_fields_for_method(
                                method
                            ),
                            **problem_provenance_fields(
                                problem
                            ),
                            "comparison_family": "compare_up_projection",
                            "comparison_mode": "logical",
                            "num_variables": problem.num_variables,
                            "num_states": problem.num_states,
                            "reference_optimum_objective": (
                                float(
                                    reference.optimum_objective
                                )
                            ),
                            "reference_optimum_source": reference.optimum_source,
                            "reference_objective_sense": (
                                reference.objective_sense
                            ),
                            "reference_match_tolerance": DEFAULT_REFERENCE_ATOL,
                            "num_inequality_quadratic_terms": (
                                num_inequality_quadratic_terms(
                                    problem
                                )
                            ),
                            "sqa_logical_cop": sqa_metrics.logical_cop,
                            "sqa_num_reads": sqa_metrics.num_reads,
                            "sqa_fea": sqa_metrics.feasible_rate,
                            "sqa_gap": sqa_metrics.objective_gap,
                            "sqa_schedule_id": sqa_metrics.schedule_id,
                            "sqa_schedule_kind": sqa_metrics.schedule_kind,
                            "sqa_beta_scale": sqa_metrics.beta_scale,
                            "sqa_schedule_total_time": (
                                sqa_metrics.total_schedule_time
                            ),
                            "sqa_anneal_schedule": (
                                baseline_mod._shared_anneal_schedule_json(
                                    sqa_metrics.anneal_schedule
                                )
                            ),
                            "qubo_normalization_scale": normalization_scale,
                            "normalization_regime": (
                                tuned_up.normalization_regime
                                or UP_NORMALIZATION_REGIME
                            ),
                            "per_constraint_standardization": (
                                True
                                if method == "unbalanced"
                                and tuned_up.per_constraint_standardization
                                is None
                                else (
                                    bool(
                                        tuned_up.per_constraint_standardization
                                    )
                                    if method
                                    == "unbalanced"
                                    else bool(
                                        selected_config.projected_standardize
                                    )
                                )
                            ),
                            "projected_sample_size": (
                                None
                                if projected_components
                                is None
                                else projected_components.sample_size
                            ),
                            "projected_num_quadratic_couplers": (
                                None
                                if projected_components
                                is None
                                else projected_components.num_quadratic_couplers
                            ),
                            "projection_method": (
                                None
                                if selected_config is None
                                else (
                                    selected_config.projection_method
                                    or LOGICAL_METHOD_TO_SUMMARY[
                                        method
                                    ]
                                )
                            ),
                            "projection_summary_method": (
                                None
                                if method == "unbalanced"
                                else LOGICAL_METHOD_TO_SUMMARY[
                                    method
                                ]
                            ),
                            "projection_measure": (
                                None
                                if selected_config is None
                                else selected_config.measure_name
                            ),
                            "projection_selection_mode": (
                                None
                                if selected_config is None
                                else selected_config.selection_mode
                            ),
                            "projection_selection_source": (
                                None
                                if selected_config is None
                                else selected_config.selection_source
                            ),
                            "projection_candidate_rank": (
                                None
                                if selected_config is None
                                else int(
                                    selected_config.candidate_rank
                                )
                            ),
                            "projected_standardize": (
                                None
                                if selected_config is None
                                else bool(
                                    selected_config.projected_standardize
                                )
                            ),
                            "projection_target": (
                                None
                                if method == "unbalanced"
                                else UP_PROJECTION_TARGET
                            ),
                            "projection_target_lambda1_shape": (
                                None
                                if method == "unbalanced"
                                else tuned_up.up_lambda1_shape
                            ),
                            "projection_target_lambda2_shape": (
                                None
                                if method == "unbalanced"
                                else tuned_up.up_lambda2_shape
                            ),
                            "projected_equality_multiplier": (
                                None
                                if selected_config is None
                                else selected_config.tuning.equality_multiplier
                            ),
                            "projected_inequality_multiplier": (
                                None
                                if selected_config is None
                                else selected_config.tuning.inequality_multiplier
                            ),
                            "projected_initializer_equality_multiplier": (
                                None
                                if method == "unbalanced"
                                else tuned_up.up_equality_multiplier
                            ),
                            "projected_initializer_inequality_multiplier": (
                                None
                                if method == "unbalanced"
                                else tuned_up.up_inequality_multiplier
                            ),
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
                            "up_global_multiplier": (
                                tuned_up.global_multiplier
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

    _write_logical_outputs(
        output_dir=output_dir,
        active_families=active_families,
        cop_rows=cop_rows,
        projection_config_rows=projection_config_rows,
        unbalanced_config_rows=unbalanced_config_rows,
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        force=bool(args.force),
    )
    return {
        "cop_rows": cop_rows,
        "projection_config_rows": projection_config_rows,
        "unbalanced_config_rows": unbalanced_config_rows,
    }


def _evaluate_embedding_instance(
    *,
    family: str,
    size: int,
    instance_index: int,
    problem: BLP,
    reference: Any,
    tuned_up: Any,
    hardware_projected_config_map: dict[str, Any],
    hardware_families: tuple[str, ...],
    hardware_sizes: dict[str, int],
    base_seed: int,
    measure_lam: float,
    sample_cap_log2: int,
    projection_reg: float,
    sqa_num_reads: int,
    sqa_num_points: int,
    sqa_num_sweeps_per_beta: int,
    sqa_schedule: Any,
    sqa_chain_strength_fraction: float,
) -> list[dict[str, object]]:
    """Evaluate one instance across all hardware families."""
    annealer_backend = (
        embedding_mod.SimulatedDWaveAnnealingBackend()
    )
    hardware_graphs = {
        hardware_family: embedding_mod.build_dwave_graph(
            hardware_family,
            hardware_sizes[hardware_family],
        )
        for hardware_family in hardware_families
    }
    rows: list[dict[str, object]] = []
    components_cache: dict[
        tuple[object, ...], ProjectedPenaltyComponents
    ] = {}
    unbalanced_qubo = _unbalanced_qubo(problem, tuned_up)
    for hardware_family in hardware_families:
        hardware_size = int(hardware_sizes[hardware_family])
        hardware_graph = hardware_graphs[hardware_family]
        summary_method = HARDWARE_TO_SUMMARY_METHOD[
            hardware_family
        ]
        selected_config = hardware_projected_config_map[
            summary_method
        ]
        pair_edges, fixed_embedding = (
            embedding_mod._projection_topology_details(
                problem,
                projection_method=(
                    selected_config.projection_method
                    or summary_method
                ),
                projection_hardware_graph=hardware_graph,
            )
        )
        cache_key = embedding_mod._projected_components_cache_key(
            projection_method=selected_config.projection_method
            or summary_method,
            family=family,
            size=size,
            instance_index=instance_index,
            measure_name=selected_config.measure_name,
            measure_lam=measure_lam,
            penalty_template="quadratic",
            penalty_template_kwargs={
                "lambda1": float(tuned_up.up_lambda1_shape),
                "lambda2": float(tuned_up.up_lambda2_shape),
            },
            standardize=bool(
                selected_config.projected_standardize
            ),
            deployment_topology=hardware_family,
            deployment_topology_size=hardware_size,
        )
        if cache_key not in components_cache:
            components_cache[cache_key] = (
                _build_unb_pen_projected_components(
                    problem,
                    pair_edges=pair_edges,
                    family=family,
                    size=size,
                    instance_index=instance_index,
                    base_seed=base_seed,
                    measure_name=selected_config.measure_name,
                    measure_lam=measure_lam,
                    sample_cap_log2=sample_cap_log2,
                    reg=projection_reg,
                    standardize=bool(
                        selected_config.projected_standardize
                    ),
                    lambda1_shape=float(
                        tuned_up.up_lambda1_shape
                    ),
                    lambda2_shape=float(
                        tuned_up.up_lambda2_shape
                    ),
                )
            )
        projected_components = components_cache[cache_key]
        projected_qubo = _projected_up_qubo(
            problem,
            projected_components,
            equality_multiplier=tuned_up.up_equality_multiplier,
            inequality_multiplier=float(
                tuned_up.up_inequality_multiplier
            ),
        )
        fixed_embeddings = {
            "unbalanced": None,
            EMBEDDING_PROJECTED_METHOD: fixed_embedding,
        }
        method_qubos = {
            "unbalanced": unbalanced_qubo,
            EMBEDDING_PROJECTED_METHOD: projected_qubo,
        }
        for method in EMBEDDING_METHODS:
            quadratic, linear, const = method_qubos[method]
            normalization_scale = (
                embedding_mod._qaoa_normalization_scale(
                    quadratic,
                    linear,
                )
            )
            annealer_input = (
                embedding_mod._scaled_annealer_input(
                    problem,
                    quadratic,
                    linear,
                    const,
                    normalization_scale=normalization_scale,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                )
            )
            sqa_metrics = embedding_mod._embedded_sqa_penalized_metrics(
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
                fixed_embedding=fixed_embeddings[method],
            )
            projected_row = (
                method == EMBEDDING_PROJECTED_METHOD
            )
            rows.append(
                {
                    "hardware_family": hardware_family,
                    "hardware_size": hardware_size,
                    "family": family,
                    "size": size,
                    "instance_index": instance_index,
                    "method": method,
                    **_projection_regime_fields_for_method(
                        method,
                        hardware_family=hardware_family,
                    ),
                    **problem_provenance_fields(problem),
                    "comparison_family": "compare_up_projection",
                    "comparison_mode": "embedding",
                    "num_variables": problem.num_variables,
                    "num_states": problem.num_states,
                    "reference_optimum_objective": (
                        float(reference.optimum_objective)
                    ),
                    "reference_optimum_source": reference.optimum_source,
                    "reference_objective_sense": reference.objective_sense,
                    "reference_match_tolerance": DEFAULT_REFERENCE_ATOL,
                    "num_inequality_quadratic_terms": (
                        num_inequality_quadratic_terms(
                            problem
                        )
                    ),
                    "sqa_logical_cop": sqa_metrics.coefficient_of_performance,
                    "sqa_fea": sqa_metrics.feasible_read_fraction,
                    "sqa_gap": sqa_metrics.objective_gap,
                    "optimum_probability": sqa_metrics.optimum_probability,
                    "cop": sqa_metrics.coefficient_of_performance,
                    "true_optimum_objective": float(
                        reference.optimum_objective
                    ),
                    "best_feasible_objective": (
                        sqa_metrics.best_feasible_objective
                    ),
                    "best_feasible_gap": sqa_metrics.best_feasible_gap,
                    "objective_gap": sqa_metrics.objective_gap,
                    "num_feasible_reads": sqa_metrics.num_feasible_reads,
                    "feasible_read_fraction": (
                        sqa_metrics.feasible_read_fraction
                    ),
                    "sqa_num_reads": sqa_num_reads,
                    "sqa_num_sweeps": sqa_num_points,
                    "sqa_num_sweeps_per_beta": sqa_num_sweeps_per_beta,
                    "sqa_chain_strength_fraction": (
                        sqa_chain_strength_fraction
                    ),
                    "sqa_schedule_id": sqa_schedule.schedule_id,
                    "sqa_schedule_kind": sqa_schedule.schedule_kind,
                    "sqa_beta_scale": sqa_schedule.beta_scale,
                    "sqa_total_schedule_time": (
                        sqa_schedule.total_schedule_time
                    ),
                    "sqa_anneal_schedule": (
                        embedding_mod._shared_anneal_schedule_json(
                            sqa_schedule.anneal_schedule
                        )
                    ),
                    "mean_sqa_energy_gap": sqa_metrics.mean_sqa_energy_gap,
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
                    "qubo_normalization_scale": normalization_scale,
                    "normalization_regime": (
                        tuned_up.normalization_regime
                        or UP_NORMALIZATION_REGIME
                    ),
                    "per_constraint_standardization": (
                        True
                        if method == "unbalanced"
                        and tuned_up.per_constraint_standardization
                        is None
                        else (
                            bool(
                                tuned_up.per_constraint_standardization
                            )
                            if method == "unbalanced"
                            else bool(
                                selected_config.projected_standardize
                            )
                        )
                    ),
                    "projected_sample_size": (
                        projected_components.sample_size
                        if projected_row
                        else None
                    ),
                    "projected_num_quadratic_couplers": (
                        projected_components.num_quadratic_couplers
                        if projected_row
                        else None
                    ),
                    "projection_method": (
                        (
                            selected_config.projection_method
                            or summary_method
                        )
                        if projected_row
                        else None
                    ),
                    "projection_summary_method": (
                        summary_method
                        if projected_row
                        else None
                    ),
                    "projection_measure": (
                        selected_config.measure_name
                        if projected_row
                        else None
                    ),
                    "projection_selection_mode": (
                        selected_config.selection_mode
                        if projected_row
                        else None
                    ),
                    "projection_selection_source": (
                        selected_config.selection_source
                        if projected_row
                        else None
                    ),
                    "projection_candidate_rank": (
                        int(selected_config.candidate_rank)
                        if projected_row
                        else None
                    ),
                    "projected_standardize": (
                        bool(
                            selected_config.projected_standardize
                        )
                        if projected_row
                        else None
                    ),
                    "projection_target": (
                        UP_PROJECTION_TARGET
                        if projected_row
                        else None
                    ),
                    "projection_target_lambda1_shape": (
                        tuned_up.up_lambda1_shape
                        if projected_row
                        else None
                    ),
                    "projection_target_lambda2_shape": (
                        tuned_up.up_lambda2_shape
                        if projected_row
                        else None
                    ),
                    "projected_equality_multiplier": (
                        selected_config.tuning.equality_multiplier
                        if projected_row
                        else None
                    ),
                    "projected_inequality_multiplier": (
                        selected_config.tuning.inequality_multiplier
                        if projected_row
                        else None
                    ),
                    "projected_initializer_equality_multiplier": (
                        tuned_up.up_equality_multiplier
                        if projected_row
                        else None
                    ),
                    "projected_initializer_inequality_multiplier": (
                        tuned_up.up_inequality_multiplier
                        if projected_row
                        else None
                    ),
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


def _embedding_progress_totals(
    problem_batches: dict[tuple[str, int], list[BLP]],
    *,
    hardware_families: tuple[str, ...],
) -> embedding_mod.ProgressTotals:
    """Return embedding-mode totals for the shared compare-embedding UI."""
    instances = 0
    topologies = 0
    methods = 0
    for batch in problem_batches.values():
        instances += len(batch)
        for problem in batch:
            del problem
            topologies += len(hardware_families)
            methods += len(hardware_families) * len(
                EMBEDDING_METHODS
            )
    return embedding_mod.ProgressTotals(
        instances=instances,
        topologies=topologies,
        measures=methods,
    )


def run_embedding_mode(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    """Run the embedding-based variant of experiment 9."""
    active_families = selected_families_from_args(args)
    family_sizes = selected_family_sizes_from_args(args)
    sqa_chain_strength_fraction = float(
        args.sqa_chain_strength_fraction
    )
    if not (0.0 < sqa_chain_strength_fraction <= 1.0):
        raise ValueError(
            "--sqa-chain-strength-fraction must be in (0, 1]"
        )
    output_dir = _embedding_output_dir(
        Path(args.output_dir),
        chain_strength_fraction=sqa_chain_strength_fraction,
    )
    # Prefer the manifest-sidecar CPLEX CSV when a manifest is used or
    # when the project-default shared manifest exists and no explicit
    # instance manifest was provided on the CLI.
    instance_manifest_arg = getattr(args, "instance_manifest", None)
    if instance_manifest_arg is None:
        default_manifest = (
            EXPERIMENTS_DIR / "manifests" / "shared_eval_seed_manifest.csv"
        )
        if default_manifest.exists():
            instance_manifest_arg = default_manifest
            _log(
                f"auto-using instance manifest {instance_manifest_arg} (found project default)"
            )
    cplex_reference_path = resolve_cplex_reference_path(
        getattr(args, "cplex_reference_csv", None),
        instance_manifest=instance_manifest_arg,
    )
    # Keep args.instance_manifest in sync with the resolved choice so the
    # rest of the run uses the same manifest and CPLEX sidecar.
    args.instance_manifest = instance_manifest_arg
    cplex_reference_index = load_cplex_reference_index(
        cplex_reference_path
    )
    tuning_dir = Path(args.tuning_dir)
    projected_up_tuning_dir = Path(
        args.projected_up_tuning_dir
    )
    (
        tuned_unbalanced_params,
        projected_up_configs,
        unbalanced_summary_path,
        projected_summary_path,
    ) = _load_compare_unb_pen_tuning_inputs(
        tuning_dir=tuning_dir,
        projected_up_tuning_dir=projected_up_tuning_dir,
    )
    fixed_sqa_schedule = embedding_mod._fixed_sqa_schedule()
    hardware_families = tuple(
        dict.fromkeys(args.hardware_families)
    )
    hardware_sizes = {
        "chimera": int(args.chimera_size),
        "pegasus": int(args.pegasus_size),
        "zephyr": int(args.zephyr_size),
    }
    workers = max(1, int(args.workers))
    get_problem_batch = problem_batch_getter(
        base_seed=int(args.seed),
        num_instances=int(args.num_instances),
        instance_manifest=args.instance_manifest,
        family_sizes=family_sizes,
    )
    missing_up_families = [
        family
        for family in active_families
        if family not in tuned_unbalanced_params
    ]
    if missing_up_families:
        raise RuntimeError(
            "missing unbalanced tuning rows for "
            f"{', '.join(missing_up_families)} in "
            f"{unbalanced_summary_path}"
        )
    missing_projected_configs = [
        f"{family}/{summary_method}"
        for family in active_families
        for summary_method in HARDWARE_TO_SUMMARY_METHOD.values()
        if (family, summary_method)
        not in projected_up_configs
    ]
    if missing_projected_configs:
        raise RuntimeError(
            "missing projected-UP tuning rows for "
            f"{', '.join(missing_projected_configs)} in "
            f"{projected_summary_path}"
        )
    ensure_run_metadata(
        output_dir,
        {
            "experiment": "compare_up_projection",
            "mode": "embedding",
            "base_seed": int(args.seed),
            "num_instances": int(args.num_instances),
            "instance_manifest": (
                None
                if args.instance_manifest is None
                else str(
                    Path(args.instance_manifest).resolve()
                )
            ),
            "cplex_reference_csv": str(
                cplex_reference_path.resolve()
            ),
            "tuning_dir": str(tuning_dir.resolve()),
            "projected_up_tuning_dir": str(
                projected_up_tuning_dir.resolve()
            ),
            "projected_summary_path": str(
                projected_summary_path.resolve()
            ),
            "unbalanced_summary_path": str(
                unbalanced_summary_path.resolve()
            ),
            "hardware_families": list(hardware_families),
            "hardware_sizes": dict(hardware_sizes),
            "sqa_num_reads": int(args.sqa_num_reads),
            "sqa_num_points": DEFAULT_SQA_NUM_SWEEPS,
            "sqa_num_sweeps_per_beta": DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
            "sqa_chain_strength_fraction": sqa_chain_strength_fraction,
            "projection_target": UP_PROJECTION_TARGET,
            "sqa_schedule_id": fixed_sqa_schedule.schedule_id,
        },
        force=bool(args.force),
    )

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

    progress_totals = _embedding_progress_totals(
        problem_batches,
        hardware_families=hardware_families,
    )
    progress = (
        embedding_mod._build_compare_embedding_progress(
            mode=str(args.progress_ui),
            totals=progress_totals,
            worker_count=workers,
        )
    )
    previous_active_progress = (
        embedding_mod._ACTIVE_PROGRESS
    )
    embedding_mod._ACTIVE_PROGRESS = progress
    if progress is not None:
        progress.start()

    cop_rows: list[dict[str, object]] = []
    projection_config_rows: list[dict[str, object]] = []
    unbalanced_config_rows: list[dict[str, object]] = []
    for family in active_families:
        for size in family_sizes[family]:
            tuned_up = tuned_unbalanced_params[family]
            unbalanced_config_rows.append(
                _unbalanced_config_row(
                    family=family,
                    size=size,
                    tuned_up=tuned_up,
                    source_path=unbalanced_summary_path,
                )
            )
            for hardware_family in hardware_families:
                summary_method = HARDWARE_TO_SUMMARY_METHOD[
                    hardware_family
                ]
                projection_config_rows.append(
                    _embedding_projection_config_row(
                        hardware_family=hardware_family,
                        hardware_size=hardware_sizes[
                            hardware_family
                        ],
                        family=family,
                        size=size,
                        selected_config=projected_up_configs[
                            (family, summary_method)
                        ],
                        tuned_up=tuned_up,
                        projected_summary_path=projected_summary_path,
                        chain_strength_fraction=sqa_chain_strength_fraction,
                    )
                )

    executor: ProcessPoolExecutor | None = None
    methods_per_instance = len(hardware_families) * len(
        EMBEDDING_METHODS
    )
    topologies_per_instance = len(hardware_families)
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=workers
            )
        _log(
            f"embedding: using {workers} worker process(es)"
        )
        for family in active_families:
            for size in family_sizes[family]:
                tuned_up = tuned_unbalanced_params[family]
                projected_config_map = {
                    summary_method: projected_up_configs[
                        (family, summary_method)
                    ]
                    for summary_method in HARDWARE_TO_SUMMARY_METHOD.values()
                }
                problems = problem_batches[(family, size)]
                references = reference_batches[
                    (family, size)
                ]
                _log(
                    f"embedding: evaluating {family} size {size} "
                    f"with {len(problems)} instance(s)"
                )
                if executor is None:
                    for (
                        instance_index,
                        problem,
                    ) in enumerate(problems):
                        embedding_mod._set_progress_status(
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
                        cop_rows.extend(
                            _evaluate_embedding_instance(
                                family=family,
                                size=size,
                                instance_index=instance_index,
                                problem=problem,
                                reference=references[
                                    instance_index
                                ],
                                tuned_up=tuned_up,
                                hardware_projected_config_map=(
                                    projected_config_map
                                ),
                                hardware_families=hardware_families,
                                hardware_sizes=hardware_sizes,
                                base_seed=int(args.seed),
                                measure_lam=DEFAULT_MEASURE_LAM,
                                sample_cap_log2=DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                                projection_reg=DEFAULT_PROJECTION_REG,
                                sqa_num_reads=int(
                                    args.sqa_num_reads
                                ),
                                sqa_num_points=DEFAULT_SQA_NUM_SWEEPS,
                                sqa_num_sweeps_per_beta=(
                                    DEFAULT_SQA_NUM_SWEEPS_PER_BETA
                                ),
                                sqa_schedule=fixed_sqa_schedule,
                                sqa_chain_strength_fraction=(
                                    sqa_chain_strength_fraction
                                ),
                            )
                        )
                        embedding_mod._advance_progress_counter(
                            "methods",
                            methods_per_instance,
                        )
                        embedding_mod._advance_progress_counter(
                            "topologies",
                            topologies_per_instance,
                        )
                        embedding_mod._advance_progress_counter(
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
                        hardware_projected_config_map=projected_config_map,
                        hardware_families=hardware_families,
                        hardware_sizes=hardware_sizes,
                        base_seed=int(args.seed),
                        measure_lam=DEFAULT_MEASURE_LAM,
                        sample_cap_log2=DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
                        projection_reg=DEFAULT_PROJECTION_REG,
                        sqa_num_reads=int(
                            args.sqa_num_reads
                        ),
                        sqa_num_points=DEFAULT_SQA_NUM_SWEEPS,
                        sqa_num_sweeps_per_beta=DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
                        sqa_schedule=fixed_sqa_schedule,
                        sqa_chain_strength_fraction=sqa_chain_strength_fraction,
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
                    embedding_mod._set_progress_status(
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
                    embedding_mod._advance_progress_counter(
                        "methods",
                        methods_per_instance,
                    )
                    embedding_mod._advance_progress_counter(
                        "topologies",
                        topologies_per_instance,
                    )
                    embedding_mod._advance_progress_counter(
                        "instances", 1
                    )
                for instance_index in range(len(problems)):
                    cop_rows.extend(
                        completed_rows[instance_index]
                    )

        embedding_mod._set_progress_status(
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
        _write_embedding_outputs(
            output_dir=output_dir,
            cop_rows=cop_rows,
            projection_config_rows=projection_config_rows,
            unbalanced_config_rows=unbalanced_config_rows,
            bootstrap_resamples=int(
                args.bootstrap_resamples
            ),
            bootstrap_seed=int(args.bootstrap_seed),
            force=bool(args.force),
        )
        return {
            "cop_rows": cop_rows,
            "projection_config_rows": projection_config_rows,
            "unbalanced_config_rows": unbalanced_config_rows,
        }
    finally:
        if executor is not None:
            executor.shutdown(
                wait=True, cancel_futures=False
            )
        if progress is not None:
            progress.close()
        embedding_mod._ACTIVE_PROGRESS = (
            previous_active_progress
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the combined CLI for experiment 9."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=list(MODE_CHOICES),
        default="both",
        help="Which pipeline to run: tune, logical, embedding, or both eval modes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for logical and embedding result tables",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed used for instance generation and projections",
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
        default=DEFAULT_INSTANCE_MANIFEST,
        help=(
            "Seed manifest CSV used to construct matched instance sets. "
            "Defaults to the shared evaluation manifest so the matching "
            "manifest-sidecar CPLEX reference CSV is used automatically."
        ),
    )
    parser.add_argument(
        "--cplex-reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional CPLEX reference CSV. Defaults to the manifest-sidecar "
            "reference when --instance-manifest is provided."
        ),
    )
    add_family_selection_arguments(
        parser, include_sizes=True
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=DEFAULT_TUNING_DIR,
        help=(
            "Base tuning directory containing the shared UP summary and "
            "the original projected combo selections"
        ),
    )
    parser.add_argument(
        "--projected-up-tuning-dir",
        type=Path,
        default=DEFAULT_PROJECTED_UP_TUNING_DIR,
        help=(
            "Directory where experiment-9 projected-UP multiplier tuning "
            "summaries are written and later read"
        ),
    )
    parser.add_argument(
        "--pegasus-size",
        type=int,
        default=DEFAULT_PEGASUS_SIZE,
        help="Logical-topology size parameter used by the baseline-style mapper",
    )
    parser.add_argument(
        "--hardware-families",
        nargs="+",
        choices=list(HARDWARE_TO_SUMMARY_METHOD),
        default=list(HARDWARE_TO_SUMMARY_METHOD),
        help="One or more hardware graph families to evaluate in embedding mode",
    )
    parser.add_argument(
        "--chimera-size",
        type=int,
        default=embedding_mod.DEFAULT_CHIMERA_SIZE,
        help="Size parameter for Chimera hardware graphs in embedding mode",
    )
    parser.add_argument(
        "--zephyr-size",
        type=int,
        default=embedding_mod.DEFAULT_ZEPHYR_SIZE,
        help="Size parameter for Zephyr hardware graphs in embedding mode",
    )
    parser.add_argument(
        "--sqa-num-reads",
        type=int,
        default=DEFAULT_SQA_NUM_READS,
        help="Number of decoded reads drawn from SQA in embedding mode",
    )
    parser.add_argument(
        "--sqa-chain-strength-fraction",
        type=float,
        default=embedding_mod.DEFAULT_SQA_CHAIN_STRENGTH_FRACTION,
        help="Fraction of Ocean's default chain strength to use in embedding mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=embedding_mod.DEFAULT_WORKERS,
        help="Number of worker processes for embedding-mode evaluation",
    )
    parser.add_argument(
        "--progress-ui",
        choices=PROGRESS_UI_CHOICES,
        default=DEFAULT_PROGRESS_UI,
        help="Progress renderer to use in tune and embedding modes",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Number of bootstrap resamples used for robust CI bounds",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Base RNG seed used for robust aggregate bootstrap CIs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run metadata instead of rejecting mismatch",
    )
    return parser


def main() -> None:
    """Dispatch experiment 9 across the requested evaluation modes."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.mode == "tune":
        run_tuning_mode(args)
        return
    if args.mode in {"logical", "both"}:
        run_logical_mode(args)
    if args.mode in {"embedding", "both"}:
        run_embedding_mode(args)


if __name__ == "__main__":
    main()
