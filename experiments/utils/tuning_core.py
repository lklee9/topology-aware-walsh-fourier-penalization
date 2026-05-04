"""Shared tuning helpers used by the experiment entry points."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.optimize import minimize

from experiments.experiment_config import (
    DEFAULT_TUNING_OBJECTIVE,
)
from experiments.utils import baseline_progress
from experiments.utils.projected_qubo import (
    add_qubo_terms,
    build_unit_equality_constraint_qubos,
    combine_constraint_terms,
    projected_multiplier_vector,
    scale_terms,
)
from experiments.utils.qubo_standardization import (
    standardized_penalty_scale,
)
from experiments.utils.tuning_models import (
    GroundStateTuningMetrics,
    TunedProjectedMultipliers,
    TunedUnbalancedParameters,
    TuningInstance,
    TuningObjectiveResult,
    UnbalancedInequalityRowBasis,
    UnbalancedTuningInstance,
)
from experiments.utils.tuning_support import (
    build_projected_components,
    objective_energies,
    optimum_states,
    projected_component_energies,
    projected_components_cache_key,
    qubo_energies,
)
from experiments.utils.unbalanced_pipeline import (
    UP_BASE_PARAMETER_SOURCE,
    UP_LAMBDA_GAUGE,
    UP_NORMALIZATION_REGIME,
    unbalanced_inequality_linear_terms,
    unbalanced_inequality_quadratic_terms,
    unbalanced_multiplier_vector,
)


def prepare_tuning_instances(
    projection_method: str,
    family: str,
    problems: list[Any],
    *,
    size: int,
    components_cache: dict[tuple[object, ...], Any],
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
    projection_hardware_graph=None,
    rigetti_hardware_graph=None,
    progress_desc: str | None = None,
) -> list[TuningInstance]:
    """Cache the exact energy tables needed for projected tuning."""
    prepared: list[TuningInstance] = []
    iterable = enumerate(problems)
    with baseline_progress.progress(
        iterable,
        total=len(problems),
        desc=progress_desc
        or f"Prepare {family}/{projection_method} n={size}",
        leave=False,
    ) as progress_bar:
        for instance_index, problem in progress_bar:
            baseline_progress.set_progress_status(
                stage="prepare_projected_tuning",
                activity="precomputing projected energies",
                family=family,
                size=size,
                instance_index=instance_index,
                total_instances=len(problems),
                detail="precomputing projected tuning energies",
            )
            optimum_objective, optimum_state_indices = (
                optimum_states(
                    problem,
                    chunk_size=chunk_size,
                )
            )
            del optimum_objective
            key = projected_components_cache_key(
                projection_method=projection_method,
                family=family,
                size=size,
                instance_index=instance_index,
                measure_name=measure_name,
                measure_lam=measure_lam,
                penalty_template=penalty_template,
                penalty_template_kwargs=penalty_template_kwargs,
                standardize=standardize,
            )
            if key not in components_cache:
                components_cache[key] = (
                    build_projected_components(
                        problem,
                        projection_method=projection_method,
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        base_seed=base_seed,
                        measure_name=measure_name,
                        measure_lam=measure_lam,
                        penalty_template=penalty_template,
                        penalty_template_kwargs=penalty_template_kwargs,
                        pegasus_size=pegasus_size,
                        sample_cap_log2=sample_cap_log2,
                        chunk_size=chunk_size,
                        reg=reg,
                        standardize=standardize,
                        projection_hardware_graph=projection_hardware_graph,
                        rigetti_hardware_graph=rigetti_hardware_graph,
                        status_callback=baseline_progress.set_progress_status,
                    )
                )
            components = components_cache[key]
            equality_energies, inequality_energies = (
                projected_component_energies(
                    problem,
                    components,
                    chunk_size=chunk_size,
                )
            )
            prepared.append(
                TuningInstance(
                    num_states=problem.num_states,
                    optimum_state_indices=optimum_state_indices,
                    objective_energies=objective_energies(
                        problem,
                        chunk_size=chunk_size,
                    ),
                    equality_energies=equality_energies,
                    inequality_energies=inequality_energies,
                )
            )
    return prepared


def prepare_unbalanced_tuning_instances(
    problems: list[Any],
    *,
    family: str | None = None,
    anchor_size: int | None = None,
    chunk_size: int,
    progress_desc: str | None = None,
) -> list[UnbalancedTuningInstance]:
    """Cache the exact energy tables needed for UP tuning."""
    prepared: list[UnbalancedTuningInstance] = []
    with baseline_progress.progress(
        problems,
        total=len(problems),
        desc=progress_desc
        or (
            "Prepare UP"
            if family is None or anchor_size is None
            else f"Prepare UP {family} n={anchor_size}"
        ),
        leave=False,
    ) as progress_bar:
        for instance_index, problem in enumerate(
            progress_bar
        ):
            baseline_progress.set_progress_status(
                stage="prepare_unbalanced_tuning",
                activity="precomputing unbalanced energies",
                family=family,
                size=anchor_size,
                instance_index=instance_index,
                total_instances=len(problems),
                detail="precomputing unbalanced tuning energies",
            )
            optimum_objective, optimum_state_indices = (
                optimum_states(
                    problem,
                    chunk_size=chunk_size,
                )
            )
            del optimum_objective
            objective_values = objective_energies(
                problem,
                chunk_size=chunk_size,
            )
            equality_terms = combine_constraint_terms(
                problem,
                build_unit_equality_constraint_qubos(
                    problem
                ),
                standardize=True,
            )
            equality_energies = qubo_energies(
                problem,
                quadratic=equality_terms.quadratic,
                linear=equality_terms.linear,
                const=equality_terms.const,
                chunk_size=chunk_size,
            )
            inequality_row_bases: list[
                UnbalancedInequalityRowBasis
            ] = []
            for coeffs, rhs in zip(
                problem.A,
                problem.b,
                strict=True,
            ):
                linear_terms = (
                    unbalanced_inequality_linear_terms(
                        coeffs, rhs
                    )
                )
                quadratic_terms = (
                    unbalanced_inequality_quadratic_terms(
                        coeffs,
                        rhs,
                    )
                )
                linear_energies = qubo_energies(
                    problem,
                    quadratic=linear_terms.quadratic,
                    linear=linear_terms.linear,
                    const=linear_terms.const,
                    chunk_size=chunk_size,
                )
                quadratic_energies = qubo_energies(
                    problem,
                    quadratic=quadratic_terms.quadratic,
                    linear=quadratic_terms.linear,
                    const=quadratic_terms.const,
                    chunk_size=chunk_size,
                )
                inequality_row_bases.append(
                    UnbalancedInequalityRowBasis(
                        linear_terms=linear_terms,
                        quadratic_terms=quadratic_terms,
                        linear_energies=linear_energies,
                        quadratic_energies=quadratic_energies,
                    )
                )
            prepared.append(
                UnbalancedTuningInstance(
                    num_states=problem.num_states,
                    optimum_state_indices=optimum_state_indices,
                    objective_energies=objective_values,
                    objective_linear=np.asarray(
                        problem.c, dtype=float
                    ).copy(),
                    equality_energies=equality_energies,
                    inequality_row_bases=tuple(
                        inequality_row_bases
                    ),
                )
            )
    return prepared


def ground_state_tuning_metrics(
    energies: np.ndarray,
    optimum_state_indices: np.ndarray,
    *,
    atol: float = 1e-9,
) -> GroundStateTuningMetrics:
    """Return the gap metrics used by the simplified tuning sweep."""
    optimum_energy = float(
        np.min(energies[optimum_state_indices])
    )
    ground_energy = float(np.min(energies))
    non_optimal_mask = np.ones(
        energies.shape[0],
        dtype=bool,
    )
    non_optimal_mask[optimum_state_indices] = False
    non_optimal_energies = energies[non_optimal_mask]
    if non_optimal_energies.size == 0:
        tied_fraction = 0.0
    else:
        tied_fraction = float(
            np.count_nonzero(
                np.isclose(
                    non_optimal_energies,
                    optimum_energy,
                    atol=atol,
                    rtol=0.0,
                )
            )
        ) / float(energies.shape[0])
    return GroundStateTuningMetrics(
        gap=optimum_energy - ground_energy,
        best_optimum_percentile=float(
            np.count_nonzero(
                non_optimal_energies
                < (optimum_energy - atol)
            )
        )
        / float(energies.shape[0]),
        tied_fraction=tied_fraction,
    )


def tuning_objective_result(
    metrics: list[GroundStateTuningMetrics],
    *,
    param_sum: float,
) -> TuningObjectiveResult:
    """Aggregate per-instance tuning metrics into one result."""
    gap_values = np.array(
        [item.gap for item in metrics],
        dtype=float,
    )
    best_optimum_percentile_values = np.array(
        [item.best_optimum_percentile for item in metrics],
        dtype=float,
    )
    tied_fraction_values = np.array(
        [item.tied_fraction for item in metrics],
        dtype=float,
    )
    mean_gap = float(np.mean(gap_values))
    mean_best_optimum_percentile = float(
        np.mean(best_optimum_percentile_values)
    )
    mean_tied_fraction = float(
        np.mean(tied_fraction_values)
    )
    objective_value = (
        mean_gap
        + 1e-6 * mean_best_optimum_percentile
        + 1e-9 * mean_tied_fraction
        + 1e-8 * param_sum
    )
    sort_key = (
        mean_gap,
        mean_best_optimum_percentile,
        mean_tied_fraction,
        param_sum,
    )
    return TuningObjectiveResult(
        sort_key=sort_key,
        objective_value=float(objective_value),
        record_fields={
            "tuning_objective": DEFAULT_TUNING_OBJECTIVE,
            "mean_gap": mean_gap,
            "mean_best_optimum_percentile": (
                mean_best_optimum_percentile
            ),
            "mean_tied_fraction": mean_tied_fraction,
            "param_sum": float(param_sum),
        },
    )


def evaluate_projected_tuning_objective(
    equality_multiplier: float,
    inequality_multiplier: float,
    instances: list[TuningInstance],
) -> TuningObjectiveResult:
    """Return one projected-penalty tuning score."""
    equality_multiplier = max(
        0.0, float(equality_multiplier)
    )
    inequality_multiplier = max(
        0.0,
        float(inequality_multiplier),
    )
    metrics: list[GroundStateTuningMetrics] = []
    for item in instances:
        energies = item.objective_energies.copy()
        if equality_multiplier != 0.0:
            energies += (
                equality_multiplier * item.equality_energies
            )
        if inequality_multiplier != 0.0:
            energies += (
                inequality_multiplier
                * item.inequality_energies
            )
        metrics.append(
            ground_state_tuning_metrics(
                energies,
                item.optimum_state_indices,
            )
        )
    return tuning_objective_result(
        metrics,
        param_sum=equality_multiplier
        + inequality_multiplier,
    )


def projected_tuning_objective(
    raw_params: np.ndarray,
    instances: list[TuningInstance],
    *,
    has_equality: bool,
) -> TuningObjectiveResult:
    """Return one projected-penalty score over the anchor batch."""
    equality_multiplier, inequality_multiplier = (
        projected_multiplier_vector(
            raw_params,
            has_equality=has_equality,
        )
    )
    return evaluate_projected_tuning_objective(
        equality_multiplier,
        inequality_multiplier,
        instances,
    )


def evaluate_unbalanced_parameter_objective(
    params: tuple[float, float, float, float],
    instances: list[UnbalancedTuningInstance],
) -> TuningObjectiveResult:
    """Return one UP tuning score over the anchor-size batch."""
    (
        equality_multiplier,
        inequality_multiplier,
        lambda1_shape,
        lambda2_shape,
    ) = params
    metrics: list[GroundStateTuningMetrics] = []
    for item in instances:
        energies = item.objective_energies.copy()
        if abs(float(equality_multiplier)) > 1e-12:
            energies += (
                float(equality_multiplier)
                * item.equality_energies
            )
        if (
            abs(float(inequality_multiplier)) > 1e-12
            and item.inequality_row_bases
        ):
            inequality_energies = np.zeros(
                item.num_states, dtype=float
            )
            for row_basis in item.inequality_row_bases:
                row_terms = add_qubo_terms(
                    scale_terms(
                        row_basis.linear_terms,
                        float(lambda1_shape),
                    ),
                    scale_terms(
                        row_basis.quadratic_terms,
                        float(lambda2_shape),
                    ),
                )
                row_scale, _, _ = (
                    standardized_penalty_scale(
                        objective_linear=item.objective_linear,
                        penalty_quadratic=row_terms.quadratic,
                        penalty_linear=row_terms.linear,
                    )
                )
                if row_scale == 0.0:
                    continue
                row_energies = (
                    float(lambda1_shape)
                    * row_basis.linear_energies
                    + float(lambda2_shape)
                    * row_basis.quadratic_energies
                )
                inequality_energies += (
                    row_scale * row_energies
                )
            energies += (
                float(inequality_multiplier)
                * inequality_energies
            )
        metrics.append(
            ground_state_tuning_metrics(
                energies,
                item.optimum_state_indices,
            )
        )
    return tuning_objective_result(
        metrics,
        param_sum=(
            float(equality_multiplier)
            + float(inequality_multiplier)
            + float(lambda1_shape)
            + float(lambda2_shape)
        ),
    )


def unbalanced_tuning_objective(
    raw_params: np.ndarray,
    instances: list[UnbalancedTuningInstance],
    *,
    has_equality: bool,
) -> TuningObjectiveResult:
    """Return one split-block UP tuning score over the anchor batch."""
    params = unbalanced_multiplier_vector(
        raw_params,
        has_equality=has_equality,
    )
    result = evaluate_unbalanced_parameter_objective(
        params,
        instances,
    )
    (
        equality_multiplier,
        inequality_multiplier,
        lambda1_shape,
        lambda2_shape,
    ) = params
    record_fields = dict(result.record_fields)
    record_fields.update(
        {
            "up_equality_multiplier": (
                float(equality_multiplier)
                if has_equality
                else None
            ),
            "up_inequality_multiplier": float(
                inequality_multiplier
            ),
            "up_lambda1_shape": float(lambda1_shape),
            "up_lambda2_shape": float(lambda2_shape),
            "up_lambda_gauge": UP_LAMBDA_GAUGE,
        }
    )
    return TuningObjectiveResult(
        sort_key=result.sort_key,
        objective_value=result.objective_value,
        record_fields=record_fields,
    )


def _positive_start_grid(
    *,
    lower: float,
    num_points: int,
) -> np.ndarray:
    """Return a positive geometric ladder of Nelder-Mead start values."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    lo = max(float(lower), 1e-8)
    if num_points == 1:
        return np.array([lo], dtype=float)
    return lo * np.power(
        10.0,
        np.arange(num_points, dtype=float),
    )


def _nelder_mead_initial_simplex(
    log_point: np.ndarray,
    *,
    log_lower: np.ndarray,
    clip_minimum: bool,
) -> np.ndarray:
    """Build an initial simplex above one log-parameter point."""
    point = np.asarray(log_point, dtype=float).reshape(-1)
    lower = np.asarray(log_lower, dtype=float).reshape(-1)
    simplex = [point.copy()]
    for index in range(point.shape[0]):
        candidate = point.copy()
        span = max(
            0.25,
            0.1
            * max(
                abs(point[index]),
                abs(lower[index]),
                1.0,
            ),
        )
        candidate[index] += span
        if clip_minimum:
            candidate[index] = max(
                lower[index],
                candidate[index],
            )
        simplex.append(candidate)
    return np.asarray(simplex, dtype=float)


def _nelder_mead_search(
    objective,
    *,
    dim: int,
    param_min: float,
    start_points_per_dim: int,
    maxiter: int,
    xatol: float,
    fatol: float,
    clip_minimum: bool,
    initial_points: list[np.ndarray] | None = None,
    progress_desc: str | None = None,
) -> tuple[
    np.ndarray,
    float,
    int,
    list[dict[str, object]],
    bool,
    int,
    str,
]:
    """Minimize one low-dimensional positive box via multi-start NM."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if start_points_per_dim <= 0:
        raise ValueError(
            "start_points_per_dim must be positive"
        )
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    lower = np.full(
        dim,
        max(float(param_min), 1e-8),
        dtype=float,
    )
    best_point = lower.copy()
    best_value = float("inf")
    best_sort_key: tuple[float, ...] | None = None
    evaluations = 0
    records: list[dict[str, object]] = []
    eval_index = 0
    log_lower = np.log(lower)
    start_axis = _positive_start_grid(
        lower=param_min,
        num_points=start_points_per_dim,
    )
    grid_start_points = [
        np.asarray(coords, dtype=float)
        for coords in itertools.product(
            start_axis,
            repeat=dim,
        )
    ]
    start_points: list[np.ndarray] = []
    seen_start_points: set[tuple[float, ...]] = set()
    for point in (
        list(initial_points or []) + grid_start_points
    ):
        candidate = np.asarray(point, dtype=float).reshape(
            -1
        )
        if candidate.shape != (dim,):
            raise ValueError(
                "initial start point has wrong dimension: "
                f"expected {dim}, got {candidate.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(
                "initial start point must be finite"
            )
        candidate = np.maximum(candidate, lower)
        start_key = tuple(np.round(candidate, 12).tolist())
        if start_key in seen_start_points:
            continue
        seen_start_points.add(start_key)
        start_points.append(candidate)
    cache: dict[
        tuple[float, ...], TuningObjectiveResult
    ] = {}
    any_success = False
    successful_restarts = 0
    best_status = 1
    best_restart_index: int | None = None
    best_optimizer_message = "did not evaluate any point"
    with baseline_progress.progress(
        total=None,
        desc=progress_desc or "Nelder-Mead",
        leave=False,
    ) as progress_bar:
        baseline_progress.begin_optimizer_progress(
            len(start_points)
        )
        try:
            for restart_index, start_point in enumerate(
                start_points
            ):
                restart_improved_global = False
                baseline_progress.set_optimizer_restart(
                    restart_index + 1
                )
                baseline_progress.set_progress_status(
                    detail=(
                        "optimizer restart "
                        f"{restart_index + 1}/{len(start_points)}"
                    )
                )

                def wrapped_objective(
                    log_params: np.ndarray,
                ) -> float:
                    nonlocal best_point
                    nonlocal best_sort_key
                    nonlocal best_value
                    nonlocal evaluations
                    nonlocal eval_index
                    nonlocal restart_improved_global
                    log_point = np.asarray(
                        log_params,
                        dtype=float,
                    ).reshape(-1)
                    if clip_minimum:
                        log_point = np.maximum(
                            log_point,
                            log_lower,
                        )
                    point = np.exp(
                        np.clip(log_point, -700.0, 700.0)
                    )
                    cache_key = tuple(
                        np.round(point, 12).tolist()
                    )
                    if cache_key not in cache:
                        cache[cache_key] = objective(point)
                    result = cache[cache_key]
                    value = float(result.objective_value)
                    evaluations += 1
                    row: dict[str, object] = {
                        "level": int(restart_index),
                        "restart_index": int(restart_index),
                        "eval_index": int(eval_index),
                        "objective": float(value),
                        "search_method": "nelder_mead",
                    }
                    for j, val in enumerate(point.tolist()):
                        row[f"param_{j}"] = float(val)
                    row.update(result.record_fields)
                    records.append(row)
                    eval_index += 1
                    progress_bar.update(1)
                    if (
                        best_sort_key is None
                        or result.sort_key < best_sort_key
                    ):
                        best_sort_key = result.sort_key
                        best_value = value
                        best_point = point.copy()
                        restart_improved_global = True
                    baseline_progress.record_optimizer_eval(
                        best_value=best_value
                    )
                    return value

                result = minimize(
                    wrapped_objective,
                    np.log(start_point),
                    method="Nelder-Mead",
                    options={
                        "initial_simplex": _nelder_mead_initial_simplex(
                            np.log(start_point),
                            log_lower=log_lower,
                            clip_minimum=clip_minimum,
                        ),
                        "maxiter": int(maxiter),
                        "xatol": float(xatol),
                        "fatol": float(fatol),
                    },
                )
                if result.success:
                    any_success = True
                    successful_restarts += 1
                if restart_improved_global:
                    best_restart_index = int(restart_index)
                    best_status = int(result.status)
                    best_optimizer_message = str(
                        result.message
                    )
        finally:
            baseline_progress.finish_optimizer_progress()
    if best_sort_key is None:
        raise RuntimeError(
            "Nelder-Mead tuning did not evaluate any point"
        )
    best_message = (
        "nelder_mead "
        f"restarts={len(start_points)} "
        f"successful_restarts={successful_restarts} "
        f"start_points_per_dim={start_points_per_dim} "
        f"initial_points={len(initial_points or [])} "
        f"param_min={param_min} "
        "param_max=unbounded "
        f"min_clip={'on' if clip_minimum else 'off'} "
        f"maxiter={maxiter} "
        f"xatol={xatol} fatol={fatol} "
        f"evaluations={evaluations} "
        f"best_restart={best_restart_index} "
        f"optimizer_message={best_optimizer_message}"
    )
    return (
        best_point,
        best_value,
        evaluations,
        records,
        any_success or np.isfinite(best_value),
        best_status,
        best_message,
    )


def tune_precomputed_projected_multipliers(
    method: str,
    family: str,
    instances: list[TuningInstance],
    *,
    anchor_size: int,
    has_equality: bool,
    param_min: float,
    start_points_per_dim: int,
    nelder_mead_maxiter: int,
    nelder_mead_xatol: float,
    nelder_mead_fatol: float,
    clip_minimum: bool,
    initial_points: list[np.ndarray] | None = None,
) -> tuple[
    TunedProjectedMultipliers,
    list[dict[str, object]],
    np.ndarray,
    int,
]:
    """Tune projected equality/inequality multipliers on cached energies."""
    objective = (
        lambda raw_params: projected_tuning_objective(
            raw_params,
            instances,
            has_equality=has_equality,
        )
    )
    (
        best_point,
        best_value,
        evaluations,
        records,
        success,
        status,
        message,
    ) = _nelder_mead_search(
        objective,
        dim=2 if has_equality else 1,
        param_min=param_min,
        start_points_per_dim=start_points_per_dim,
        maxiter=nelder_mead_maxiter,
        xatol=nelder_mead_xatol,
        fatol=nelder_mead_fatol,
        clip_minimum=clip_minimum,
        initial_points=initial_points,
        progress_desc=f"Tune {method} {family} n={anchor_size}",
    )
    equality_multiplier, inequality_multiplier = (
        projected_multiplier_vector(
            best_point,
            has_equality=has_equality,
        )
    )
    tuned = TunedProjectedMultipliers(
        method=method,
        family=family,
        anchor_size=anchor_size,
        equality_multiplier=equality_multiplier,
        inequality_multiplier=inequality_multiplier,
        tuning_objective=DEFAULT_TUNING_OBJECTIVE,
        objective_value=float(best_value),
        success=success,
        status=status,
        message=(
            f"{message}; "
            f"tuning_objective={DEFAULT_TUNING_OBJECTIVE}; "
            f"best_point={best_point.tolist()}"
        ),
    )
    return tuned, records, best_point, evaluations


def tune_projected_multipliers(
    method: str,
    projection_method: str,
    family: str,
    problems: list[Any],
    *,
    anchor_size: int,
    components_cache: dict[tuple[object, ...], Any],
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
    param_min: float,
    start_points_per_dim: int,
    nelder_mead_maxiter: int,
    nelder_mead_xatol: float,
    nelder_mead_fatol: float,
    clip_minimum: bool,
    initial_points: list[np.ndarray] | None = None,
) -> tuple[
    TunedProjectedMultipliers,
    list[dict[str, object]],
    np.ndarray,
    int,
]:
    """Tune projected equality/inequality multipliers via Nelder-Mead."""
    instances = prepare_tuning_instances(
        projection_method,
        family,
        problems,
        size=anchor_size,
        components_cache=components_cache,
        base_seed=base_seed,
        measure_name=measure_name,
        measure_lam=measure_lam,
        penalty_template=penalty_template,
        penalty_template_kwargs=penalty_template_kwargs,
        pegasus_size=pegasus_size,
        sample_cap_log2=sample_cap_log2,
        chunk_size=chunk_size,
        reg=reg,
        standardize=standardize,
        progress_desc=f"Prepare {method} {family} n={anchor_size}",
    )
    has_equality = any(
        problem.num_equalities > 0 for problem in problems
    )
    return tune_precomputed_projected_multipliers(
        method,
        family,
        instances,
        anchor_size=anchor_size,
        has_equality=has_equality,
        param_min=param_min,
        start_points_per_dim=start_points_per_dim,
        nelder_mead_maxiter=nelder_mead_maxiter,
        nelder_mead_xatol=nelder_mead_xatol,
        nelder_mead_fatol=nelder_mead_fatol,
        clip_minimum=clip_minimum,
        initial_points=initial_points,
    )


def tune_unbalanced_parameters(
    family: str,
    problems: list[Any],
    *,
    anchor_size: int,
    chunk_size: int,
    param_min: float,
    start_points_per_dim: int,
    nelder_mead_maxiter: int,
    nelder_mead_xatol: float,
    nelder_mead_fatol: float,
    clip_minimum: bool,
) -> tuple[
    TunedUnbalancedParameters,
    list[dict[str, object]],
    np.ndarray,
    int,
]:
    """Tune UP block multipliers and inequality shape via Nelder-Mead."""
    instances = prepare_unbalanced_tuning_instances(
        problems,
        family=family,
        anchor_size=anchor_size,
        chunk_size=chunk_size,
        progress_desc=f"Prepare UP {family} n={anchor_size}",
    )
    has_equality = any(
        problem.num_equalities > 0 for problem in problems
    )
    objective = (
        lambda raw_params: unbalanced_tuning_objective(
            raw_params,
            instances,
            has_equality=has_equality,
        )
    )
    (
        best_point,
        best_value,
        evaluations,
        records,
        success,
        status,
        message,
    ) = _nelder_mead_search(
        objective,
        dim=4 if has_equality else 3,
        param_min=param_min,
        start_points_per_dim=start_points_per_dim,
        maxiter=nelder_mead_maxiter,
        xatol=nelder_mead_xatol,
        fatol=nelder_mead_fatol,
        clip_minimum=clip_minimum,
        progress_desc=f"Tune UP {family} n={anchor_size}",
    )
    (
        equality_multiplier,
        inequality_multiplier,
        lambda1_shape,
        lambda2_shape,
    ) = unbalanced_multiplier_vector(
        best_point,
        has_equality=has_equality,
    )
    tuned = TunedUnbalancedParameters(
        family=family,
        anchor_size=anchor_size,
        up_equality_multiplier=(
            float(equality_multiplier)
            if has_equality
            else None
        ),
        up_inequality_multiplier=float(
            inequality_multiplier
        ),
        up_lambda1_shape=float(lambda1_shape),
        up_lambda2_shape=float(lambda2_shape),
        up_lambda_gauge=UP_LAMBDA_GAUGE,
        normalization_regime=UP_NORMALIZATION_REGIME,
        per_constraint_standardization=True,
        global_multiplier=None,
        lambda0=(
            float(equality_multiplier)
            if has_equality
            else None
        ),
        lambda1=float(inequality_multiplier)
        * float(lambda1_shape),
        lambda2=float(inequality_multiplier)
        * float(lambda2_shape),
        base_parameter_source=UP_BASE_PARAMETER_SOURCE,
        tuning_objective=DEFAULT_TUNING_OBJECTIVE,
        objective_value=float(best_value),
        success=success,
        status=status,
        message=(
            f"{message}; "
            f"base_parameter_source={UP_BASE_PARAMETER_SOURCE}; "
            f"up_lambda_gauge={UP_LAMBDA_GAUGE}; "
            f"tuning_objective={DEFAULT_TUNING_OBJECTIVE}; "
            f"best_point={best_point.tolist()}"
        ),
    )
    return tuned, records, best_point, evaluations


__all__ = [
    "evaluate_projected_tuning_objective",
    "evaluate_unbalanced_parameter_objective",
    "ground_state_tuning_metrics",
    "prepare_tuning_instances",
    "prepare_unbalanced_tuning_instances",
    "projected_tuning_objective",
    "tune_precomputed_projected_multipliers",
    "tune_projected_multipliers",
    "tune_unbalanced_parameters",
    "tuning_objective_result",
    "unbalanced_tuning_objective",
]
