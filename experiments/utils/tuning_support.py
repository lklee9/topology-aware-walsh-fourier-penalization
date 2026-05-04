"""Shared low-level helpers for tuning and projected-penalty evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import networkx as nx
import numpy as np

from experiments.experiment_config import FAMILY_CODES
from experiments.utils.driver_common import (
    binary_to_spin_states,
    build_child_rng,
    full_pair_edges,
    is_complete_pair_edge_set,
)
from experiments.utils.problems import iter_state_chunks
from experiments.utils.projected_pipeline import (
    ProjectedPenaltyComponents,
)
from experiments.utils.projected_pipeline import (
    build_projected_components as _shared_build_projected_components,
)
from experiments.utils.projected_qubo import (
    QuboTerms,
    add_qubo_terms,
    build_unit_equality_constraint_qubos,
    combine_constraint_terms,
    objective_terms,
    projection_sample_size,
    scale_terms,
)
from experiments.utils.projection_measure import (
    build_projection_sampling_catalog,
    canonical_projection_measure_name,
    sample_projection_states_with_inequality_support,
)
from experiments.utils.qa_simulator import build_dwave_graph
from experiments.utils.tuning_models import (
    TunedProjectedMultipliers,
)
from experiments.utils.unb_pen import (
    UnbalancedPenaltyParameters,
    qubo_energy_values,
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

_PAIR_SUPPORT_ATOL = 1e-12


def _template_signature(
    template_kwargs: dict[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    """Return a hashable signature for template keyword arguments."""
    if not template_kwargs:
        return ()
    return tuple(
        sorted(
            (str(key), float(value))
            for key, value in template_kwargs.items()
        )
    )


def projected_components_cache_key(
    *,
    projection_method: str,
    family: str,
    size: int,
    instance_index: int,
    measure_name: str,
    measure_lam: float,
    penalty_template: str,
    penalty_template_kwargs: dict[str, float] | None = None,
    standardize: bool | None = None,
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
        _template_signature(penalty_template_kwargs),
        None if standardize is None else bool(standardize),
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


def base_unbalanced_parameters(
    has_equality: bool,
) -> tuple[UnbalancedPenaltyParameters, str]:
    """Return the fixed UP template scaled during anchor tuning."""
    return (
        UnbalancedPenaltyParameters(
            lambda0=1.0 if has_equality else None,
            lambda1=1.0,
            lambda2=1.0,
        ),
        "unit_template",
    )


def scale_unbalanced_parameters(
    params: UnbalancedPenaltyParameters,
    multiplier: float,
) -> UnbalancedPenaltyParameters:
    """Scale a fixed UP template by one nonnegative global multiplier."""
    scale = max(0.0, float(multiplier))
    return UnbalancedPenaltyParameters(
        lambda0=(
            None
            if params.lambda0 is None
            else scale * float(params.lambda0)
        ),
        lambda1=scale * float(params.lambda1),
        lambda2=scale * float(params.lambda2),
    )


@lru_cache(maxsize=None)
def _cached_dwave_projection_graph(
    family: str,
    size: int,
) -> nx.Graph:
    """Return one cached D-Wave hardware graph used for projection."""
    return build_dwave_graph(family, size)


def _canonical_pair_edges(
    pair_edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return sorted unique logical pairs in ``i < j`` form."""
    canonical: set[tuple[int, int]] = set()
    for u, v in pair_edges:
        logical_u = int(u)
        logical_v = int(v)
        if logical_u == logical_v:
            continue
        canonical.add(tuple(sorted((logical_u, logical_v))))
    return sorted(canonical)


def _constraint_support_pair_edges(
    coeffs: np.ndarray,
) -> list[tuple[int, int]]:
    """Return the quadratic support induced by one constraint row."""
    active_indices = [
        int(index)
        for index in np.flatnonzero(
            np.abs(
                np.asarray(coeffs, dtype=float).reshape(-1)
            )
            > _PAIR_SUPPORT_ATOL
        )
    ]
    pair_edges = [
        (active_indices[left], active_indices[right])
        for left in range(len(active_indices))
        for right in range(left + 1, len(active_indices))
    ]
    return _canonical_pair_edges(pair_edges)


def _up_support_pair_edges(
    problem: BLP,
) -> list[tuple[int, int]]:
    """Return the union of row-local UP-support pairs across all constraints."""
    pair_edges: list[tuple[int, int]] = []
    if problem.num_equalities:
        for coeffs in problem.D:
            pair_edges.extend(
                _constraint_support_pair_edges(coeffs)
            )
    if problem.num_inequalities:
        for coeffs in problem.A:
            pair_edges.extend(
                _constraint_support_pair_edges(coeffs)
            )
    return _canonical_pair_edges(pair_edges)


def _fit_constraint_projection_terms(
    problem: BLP,
    sample_bits: np.ndarray,
    target_values: np.ndarray,
    importance_weights: np.ndarray,
    *,
    pair_edges: list[tuple[int, int]],
    reg: float,
) -> QuboTerms:
    """Fit one projected constraint QUBO on the requested pair support."""
    topology = HardwareTopology(
        problem.num_variables, pair_edges
    )
    fit = project_penalty_values_importance(
        sample_bits,
        topology,
        np.asarray(target_values, dtype=float),
        importance_weights=np.asarray(
            importance_weights, dtype=float
        ),
        reg=reg,
    )
    return QuboTerms(
        quadratic=np.triu(fit.quadratic, k=1),
        linear=np.asarray(fit.linear, dtype=float),
        const=float(fit.const),
    )


def _build_projected_up_support_components(
    problem: BLP,
    *,
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
    reg: float,
    standardize: bool,
    status_callback: Callable[..., None] | None = None,
) -> ProjectedPenaltyComponents:
    """Build the projected penalty using row-local UP support per constraint."""
    del pegasus_size

    sample_size = projection_sample_size(
        problem, sample_cap_log2
    )
    if status_callback is not None:
        status_callback(
            activity="computing projection",
            detail=f"sampling {sample_size} projection states",
        )
    sample_rng = build_child_rng(
        base_seed,
        2_000,
        FAMILY_CODES[family],
        size,
        instance_index,
    )
    if status_callback is not None:
        status_callback(
            activity="computing projection",
            detail="deriving row-local UP support pairs",
        )

    catalog = build_projection_sampling_catalog(
        problem,
        measure_name=measure_name,
        legacy_default_lam=measure_lam,
    )
    sample_bits, inequality_weight_vectors = (
        sample_projection_states_with_inequality_support(
            catalog,
            sample_size=sample_size,
            rng=sample_rng,
        )
    )

    equality_terms = combine_constraint_terms(
        problem,
        build_unit_equality_constraint_qubos(problem),
        standardize=standardize,
    )

    inequality_constraint_terms: list[QuboTerms] = []
    if problem.num_inequalities:
        spins = binary_to_spin_states(sample_bits)
        template_kwargs = (
            {}
            if penalty_template_kwargs is None
            else dict(penalty_template_kwargs)
        )
        for constraint_index in range(
            problem.num_inequalities
        ):
            inequality_constraint_terms.append(
                _fit_constraint_projection_terms(
                    problem,
                    sample_bits,
                    IdealPenalty.for_constraint(
                        spins,
                        problem,
                        constraint_index,
                        template=penalty_template,
                        **template_kwargs,
                    ),
                    inequality_weight_vectors[
                        constraint_index
                    ],
                    pair_edges=_constraint_support_pair_edges(
                        np.asarray(
                            problem.A[constraint_index],
                            dtype=float,
                        )
                    ),
                    reg=reg,
                )
            )

    inequality_terms = combine_constraint_terms(
        problem,
        inequality_constraint_terms,
        standardize=standardize,
    )
    return ProjectedPenaltyComponents(
        equality_terms=equality_terms,
        inequality_terms=inequality_terms,
        sample_size=int(sample_bits.shape[0]),
        num_quadratic_couplers=len(
            _up_support_pair_edges(problem)
        ),
    )


def projection_pair_edges(
    problem: BLP,
    *,
    projection_method: str,
    pegasus_size: int,
    projection_hardware_graph: nx.Graph | None = None,
    rigetti_hardware_graph: nx.Graph | None = None,
) -> list[tuple[int, int]]:
    """Return the admissible quadratic pairs for one projection method."""
    if projection_method == "projected_full":
        return full_pair_edges(problem.num_variables)
    if projection_method == "projected_up_support":
        return _up_support_pair_edges(problem)

    if projection_method in (
        "projected_pegasus",
        "projected_chimera",
        "projected_zephyr",
        "projected_rigetti",
    ):
        if projection_method == "projected_pegasus":
            hardware_graph = (
                projection_hardware_graph
                if projection_hardware_graph is not None
                else _cached_dwave_projection_graph(
                    "pegasus",
                    pegasus_size,
                )
            )
        elif projection_method == "projected_chimera":
            hardware_graph = _cached_dwave_projection_graph(
                "chimera",
                pegasus_size,
            )
        elif projection_method == "projected_zephyr":
            hardware_graph = _cached_dwave_projection_graph(
                "zephyr",
                pegasus_size,
            )
        else:
            if rigetti_hardware_graph is None:
                raise ValueError(
                    "rigetti_hardware_graph is required for "
                    "projected_rigetti"
                )
            hardware_graph = rigetti_hardware_graph

        placement, topology = (
            mapped_logical_topology_from_graph(
                problem.constraint_matrix,
                hardware_graph,
                logical_vertices=range(
                    problem.num_variables
                ),
            )
        )
        edges = [tuple(edge) for edge in topology.E]
        if all(
            0 <= u < problem.num_variables
            and 0 <= v < problem.num_variables
            for u, v in edges
        ):
            return sorted(
                tuple(sorted(edge)) for edge in edges
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
            return sorted(canonical_edges)
        raise ValueError(
            f"{projection_method} logical topology could not be "
            "canonicalized"
        )
    raise ValueError(
        f"unknown projected method: {projection_method}"
    )


def qubo_energies(
    problem: BLP,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Enumerate one QUBO energy landscape over the full hypercube."""
    energies = np.empty(problem.num_states, dtype=float)
    for start, bitstrings in iter_state_chunks(
        problem.num_variables,
        chunk_size=chunk_size,
        dtype=float,
    ):
        stop = start + bitstrings.shape[0]
        energies[start:stop] = qubo_energy_values(
            bitstrings,
            quadratic=quadratic,
            linear=linear,
            const=const,
        )
    return energies


def projected_component_energies(
    problem: BLP,
    components: ProjectedPenaltyComponents,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact energy tables for the combined projected penalties."""
    equality = np.zeros(problem.num_states, dtype=float)
    inequality = np.zeros(problem.num_states, dtype=float)
    if problem.num_equalities:
        equality = qubo_energies(
            problem,
            quadratic=components.equality_terms.quadratic,
            linear=components.equality_terms.linear,
            const=components.equality_terms.const,
            chunk_size=chunk_size,
        )
    if problem.num_inequalities:
        inequality = qubo_energies(
            problem,
            quadratic=components.inequality_terms.quadratic,
            linear=components.inequality_terms.linear,
            const=components.inequality_terms.const,
            chunk_size=chunk_size,
        )
    return equality, inequality


def build_projected_components(
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
    penalty_template_kwargs: dict[str, float] | None = None,
    pegasus_size: int,
    sample_cap_log2: int,
    chunk_size: int,
    reg: float,
    standardize: bool,
    projection_hardware_graph: nx.Graph | None = None,
    rigetti_hardware_graph: nx.Graph | None = None,
    status_callback: Callable[..., None] | None = None,
) -> ProjectedPenaltyComponents:
    """Construct projected-method pieces with shared defaults."""
    del chunk_size
    if projection_method == "projected_up_support":
        return _build_projected_up_support_components(
            problem,
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
            reg=reg,
            standardize=standardize,
            status_callback=status_callback,
        )

    sample_size = projection_sample_size(
        problem, sample_cap_log2
    )
    if status_callback is not None:
        status_callback(
            activity="computing projection",
            detail=f"sampling {sample_size} projection states",
        )
    sample_rng = build_child_rng(
        base_seed,
        2_000,
        FAMILY_CODES[family],
        size,
        instance_index,
    )
    if status_callback is not None:
        status_callback(
            activity="computing projection",
            detail=f"deriving {projection_method} logical topology",
        )
    pair_edges = projection_pair_edges(
        problem,
        projection_method=projection_method,
        pegasus_size=pegasus_size,
        projection_hardware_graph=projection_hardware_graph,
        rigetti_hardware_graph=rigetti_hardware_graph,
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
            build_unit_equality_constraint_qubos
        ),
        combine_constraint_terms=combine_constraint_terms,
        project_penalty_values_importance=(
            project_penalty_values_importance
        ),
        hardware_topology_cls=HardwareTopology,
        ideal_penalty_cls=IdealPenalty,
        binary_to_spin_states=binary_to_spin_states,
        is_complete_pair_edge_set=is_complete_pair_edge_set,
    )


def projected_full_qubo(
    problem: BLP,
    components: ProjectedPenaltyComponents,
    multipliers: TunedProjectedMultipliers,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine the projected pieces with the tuned family multipliers."""
    terms = [objective_terms(problem)]
    if (
        problem.num_equalities
        and multipliers.equality_multiplier != 0.0
    ):
        terms.append(
            scale_terms(
                components.equality_terms,
                multipliers.equality_multiplier,
            )
        )
    if (
        problem.num_inequalities
        and multipliers.inequality_multiplier != 0.0
    ):
        terms.append(
            scale_terms(
                components.inequality_terms,
                multipliers.inequality_multiplier,
            )
        )
    total = add_qubo_terms(*terms)
    return total.quadratic, total.linear, total.const


def objective_energies(
    problem: BLP,
    *,
    chunk_size: int,
    atol: float = 1e-9,
) -> np.ndarray:
    """Enumerate the raw constrained objective over the whole hypercube."""
    del atol
    values = np.empty(problem.num_states, dtype=float)
    for start, bitstrings in iter_state_chunks(
        problem.num_variables,
        chunk_size=chunk_size,
        dtype=float,
    ):
        stop = start + bitstrings.shape[0]
        values[start:stop] = problem.objective_values(
            bitstrings
        )
    return values


def optimum_states(
    problem: BLP,
    *,
    chunk_size: int,
    atol: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Return the constrained optimum objective value and optimal states."""
    best_value = np.inf
    optimum_chunks: list[np.ndarray] = []

    for start, bitstrings in iter_state_chunks(
        problem.num_variables,
        chunk_size=chunk_size,
        dtype=float,
    ):
        feasible = problem.feasible_mask(
            bitstrings,
            atol=atol,
        )
        if not np.any(feasible):
            continue

        objective_values = problem.objective_values(
            bitstrings
        )
        feasible_indices = np.flatnonzero(feasible)
        feasible_values = objective_values[feasible_indices]
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
            chunk_best,
            best_value,
            atol=atol,
            rtol=0.0,
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

    if not optimum_chunks:
        raise RuntimeError(
            f"problem {problem.name} has no feasible states"
        )
    return best_value, np.concatenate(optimum_chunks)


__all__ = [
    "base_unbalanced_parameters",
    "build_projected_components",
    "objective_energies",
    "optimum_states",
    "projected_component_energies",
    "projected_components_cache_key",
    "projected_full_qubo",
    "projection_pair_edges",
    "qubo_energies",
    "scale_unbalanced_parameters",
]
