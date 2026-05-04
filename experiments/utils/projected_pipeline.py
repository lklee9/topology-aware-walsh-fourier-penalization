"""Shared projected-penalty fitting helpers for experiment drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .projected_qubo import QuboTerms


@dataclass(frozen=True)
class ProjectedPenaltyComponents:
    """Combined projected equality and inequality QUBO pieces."""

    equality_terms: QuboTerms
    inequality_terms: QuboTerms
    sample_size: int
    num_quadratic_couplers: int


def inequality_template_matrix(
    problem: Any,
    bitstrings: np.ndarray,
    *,
    template_name: str,
    ideal_penalty_cls: Any,
    binary_to_spin_states: Callable[
        [np.ndarray], np.ndarray
    ],
    template_kwargs: dict[str, float] | None = None,
) -> np.ndarray:
    """Evaluate one ideal penalty column for each inequality constraint."""
    bits = np.asarray(bitstrings, dtype=float)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if int(problem.num_inequalities) == 0:
        return np.zeros((bits.shape[0], 0), dtype=float)

    spins = binary_to_spin_states(bits)
    kwargs = (
        {}
        if template_kwargs is None
        else dict(template_kwargs)
    )
    columns = [
        ideal_penalty_cls.for_constraint(
            spins,
            problem,
            constraint_index,
            template=template_name,
            **kwargs,
        )
        for constraint_index in range(
            problem.num_inequalities
        )
    ]
    return np.column_stack(columns)


def fit_sampled_projection_constraint_qubos(
    problem: Any,
    sample_bits: np.ndarray,
    *,
    pair_edges: list[tuple[int, int]],
    importance_weight_vectors: tuple[np.ndarray, ...],
    template_name: str,
    template_kwargs: dict[str, float] | None,
    reg: float,
    project_penalty_values_importance: Callable[..., Any],
    hardware_topology_cls: Any,
    ideal_penalty_cls: Any,
    binary_to_spin_states: Callable[
        [np.ndarray], np.ndarray
    ],
) -> list[QuboTerms]:
    """Fit one projected inequality QUBO per inequality constraint."""
    bits = np.asarray(sample_bits, dtype=float)
    if (
        bits.ndim != 2
        or bits.shape[1] != problem.num_variables
        or bits.shape[0] == 0
    ):
        raise ValueError(
            "sample_bits must have shape (N, n) with N > 0"
        )
    if int(problem.num_inequalities) == 0:
        return []
    if (
        len(importance_weight_vectors)
        != problem.num_inequalities
    ):
        raise ValueError(
            "importance_weight_vectors must contain one entry per inequality"
        )

    targets = inequality_template_matrix(
        problem,
        bits,
        template_name=template_name,
        ideal_penalty_cls=ideal_penalty_cls,
        binary_to_spin_states=binary_to_spin_states,
        template_kwargs=template_kwargs,
    )
    return fit_sampled_projection_constraint_qubos_from_targets(
        problem,
        bits,
        pair_edges=pair_edges,
        importance_weight_vectors=importance_weight_vectors,
        targets=targets,
        reg=reg,
        project_penalty_values_importance=project_penalty_values_importance,
        hardware_topology_cls=hardware_topology_cls,
    )


def fit_sampled_projection_constraint_qubos_from_targets(
    problem: Any,
    sample_bits: np.ndarray,
    *,
    pair_edges: list[tuple[int, int]],
    importance_weight_vectors: tuple[np.ndarray, ...],
    targets: np.ndarray,
    reg: float,
    project_penalty_values_importance: Callable[..., Any],
    hardware_topology_cls: Any,
) -> list[QuboTerms]:
    """Fit one projected inequality QUBO per inequality from explicit targets."""
    bits = np.asarray(sample_bits, dtype=float)
    if (
        bits.ndim != 2
        or bits.shape[1] != problem.num_variables
        or bits.shape[0] == 0
    ):
        raise ValueError(
            "sample_bits must have shape (N, n) with N > 0"
        )
    if int(problem.num_inequalities) == 0:
        return []
    if (
        len(importance_weight_vectors)
        != problem.num_inequalities
    ):
        raise ValueError(
            "importance_weight_vectors must contain one entry per inequality"
        )

    target_matrix = np.asarray(targets, dtype=float)
    if target_matrix.shape != (
        bits.shape[0],
        int(problem.num_inequalities),
    ):
        raise ValueError(
            "targets must have shape "
            f"({bits.shape[0]}, {int(problem.num_inequalities)})"
        )

    topology = hardware_topology_cls(
        problem.num_variables, pair_edges
    )

    terms: list[QuboTerms] = []
    for constraint_index in range(problem.num_inequalities):
        fit = project_penalty_values_importance(
            bits,
            topology,
            target_matrix[:, constraint_index],
            importance_weights=importance_weight_vectors[
                constraint_index
            ],
            reg=reg,
        )
        terms.append(
            QuboTerms(
                quadratic=np.triu(fit.quadratic, k=1),
                linear=np.asarray(fit.linear, dtype=float),
                const=float(fit.const),
            )
        )
    return terms


def fit_sampled_projection_equality_constraint_qubos(
    problem: Any,
    sample_bits: np.ndarray,
    *,
    pair_edges: list[tuple[int, int]],
    importance_weight_vectors: tuple[np.ndarray, ...],
    reg: float,
    project_penalty_values_importance: Callable[..., Any],
    hardware_topology_cls: Any,
    ideal_penalty_cls: Any,
    binary_to_spin_states: Callable[
        [np.ndarray], np.ndarray
    ],
) -> list[QuboTerms]:
    """Fit one projected equality QUBO per equality constraint."""
    bits = np.asarray(sample_bits, dtype=float)
    if (
        bits.ndim != 2
        or bits.shape[1] != problem.num_variables
        or bits.shape[0] == 0
    ):
        raise ValueError(
            "sample_bits must have shape (N, n) with N > 0"
        )
    if int(problem.num_equalities) == 0:
        return []
    if (
        len(importance_weight_vectors)
        != problem.num_equalities
    ):
        raise ValueError(
            "importance_weight_vectors must contain one entry per equality"
        )

    topology = hardware_topology_cls(
        problem.num_variables, pair_edges
    )
    spins = binary_to_spin_states(bits)

    terms: list[QuboTerms] = []
    for constraint_index in range(problem.num_equalities):
        target_values = (
            ideal_penalty_cls.for_equality_constraint(
                spins,
                problem,
                constraint_index,
                weight=1.0,
            )
        )
        fit = project_penalty_values_importance(
            bits,
            topology,
            target_values,
            importance_weights=importance_weight_vectors[
                constraint_index
            ],
            reg=reg,
        )
        terms.append(
            QuboTerms(
                quadratic=np.triu(fit.quadratic, k=1),
                linear=np.asarray(fit.linear, dtype=float),
                const=float(fit.const),
            )
        )
    return terms


def build_projected_components(
    problem: Any,
    *,
    pair_edges: list[tuple[int, int]],
    sample_size: int,
    sample_rng: np.random.Generator,
    measure_name: str,
    measure_lam: float,
    penalty_template: str,
    penalty_template_kwargs: dict[str, float] | None,
    reg: float,
    standardize: bool,
    build_projection_sampling_catalog: Callable[..., Any],
    sample_projection_states_with_inequality_support: Callable[
        ..., tuple[np.ndarray, tuple[np.ndarray, ...]]
    ],
    build_unit_equality_constraint_qubos: Callable[
        [Any], list[QuboTerms]
    ],
    combine_constraint_terms: Callable[..., QuboTerms],
    project_penalty_values_importance: Callable[..., Any],
    hardware_topology_cls: Any,
    ideal_penalty_cls: Any,
    binary_to_spin_states: Callable[
        [np.ndarray], np.ndarray
    ],
    is_complete_pair_edge_set: Callable[
        [int, list[tuple[int, int]]], bool
    ],
    inequality_target_matrix_builder: (
        Callable[[Any, np.ndarray], np.ndarray] | None
    ) = None,
) -> ProjectedPenaltyComponents:
    """Build the combined projected equality and inequality penalty pieces."""
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

    if is_complete_pair_edge_set(
        problem.num_variables, pair_edges
    ):
        equality_constraint_terms = (
            build_unit_equality_constraint_qubos(problem)
        )
    else:
        equality_constraint_terms = fit_sampled_projection_equality_constraint_qubos(
            problem,
            sample_bits,
            pair_edges=pair_edges,
            importance_weight_vectors=(
                catalog.equality_importance_weights(
                    sample_bits
                )
            ),
            reg=reg,
            project_penalty_values_importance=(
                project_penalty_values_importance
            ),
            hardware_topology_cls=hardware_topology_cls,
            ideal_penalty_cls=ideal_penalty_cls,
            binary_to_spin_states=binary_to_spin_states,
        )

    if inequality_target_matrix_builder is None:
        inequality_constraint_terms = fit_sampled_projection_constraint_qubos(
            problem,
            sample_bits,
            pair_edges=pair_edges,
            importance_weight_vectors=inequality_weight_vectors,
            template_name=penalty_template,
            template_kwargs=penalty_template_kwargs,
            reg=reg,
            project_penalty_values_importance=project_penalty_values_importance,
            hardware_topology_cls=hardware_topology_cls,
            ideal_penalty_cls=ideal_penalty_cls,
            binary_to_spin_states=binary_to_spin_states,
        )
    else:
        inequality_constraint_terms = fit_sampled_projection_constraint_qubos_from_targets(
            problem,
            sample_bits,
            pair_edges=pair_edges,
            importance_weight_vectors=inequality_weight_vectors,
            targets=inequality_target_matrix_builder(
                problem, sample_bits
            ),
            reg=reg,
            project_penalty_values_importance=project_penalty_values_importance,
            hardware_topology_cls=hardware_topology_cls,
        )

    equality_terms = combine_constraint_terms(
        problem,
        equality_constraint_terms,
        standardize=standardize,
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
        num_quadratic_couplers=len(pair_edges),
    )


__all__ = [
    "ProjectedPenaltyComponents",
    "build_projected_components",
    "fit_sampled_projection_constraint_qubos_from_targets",
    "fit_sampled_projection_constraint_qubos",
    "fit_sampled_projection_equality_constraint_qubos",
    "inequality_template_matrix",
]
