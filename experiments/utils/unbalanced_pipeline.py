"""Shared experiment-layer UP helpers with row standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.utils.projected_qubo import (
    QuboTerms,
    add_qubo_terms,
    build_unit_equality_constraint_qubos,
    combine_constraint_terms,
    objective_terms,
    scale_terms,
    squared_linear_form_qubo,
    zero_terms,
)
from fourier_projection.blp import BLP

UP_LAMBDA_GAUGE = "sum_to_one"
UP_NORMALIZATION_REGIME = (
    "per_constraint_standardized_rows_v1"
)
UP_BASE_PARAMETER_SOURCE = (
    "per_constraint_standardized_split_blocks"
)


@dataclass(frozen=True)
class UnbalancedPenaltyComponents:
    """Standardized UP equality/inequality blocks for one problem."""

    equality_terms: QuboTerms
    inequality_terms: QuboTerms
    lambda1_shape: float
    lambda2_shape: float
    lambda_gauge: str = UP_LAMBDA_GAUGE


def normalized_unbalanced_lambda_shape(
    raw_lambda1: float,
    raw_lambda2: float,
) -> tuple[float, float]:
    """Project nonnegative raw UP weights onto the sum-to-one gauge."""
    lambda1 = max(0.0, float(raw_lambda1))
    lambda2 = max(0.0, float(raw_lambda2))
    total = lambda1 + lambda2
    if total <= 1e-12:
        return 0.5, 0.5
    return lambda1 / total, lambda2 / total


def unbalanced_multiplier_vector(
    raw_params: np.ndarray,
    *,
    has_equality: bool,
) -> tuple[float, float, float, float]:
    """Map raw NM coordinates onto UP block multipliers and shape."""
    params = np.asarray(raw_params, dtype=float).reshape(-1)
    clipped = np.clip(params, 0.0, np.inf)
    if has_equality:
        if clipped.shape[0] != 4:
            raise ValueError(
                "expected four parameters when equality tuning is enabled"
            )
        equality_multiplier = float(clipped[0])
        inequality_multiplier = float(clipped[1])
        lambda1_shape, lambda2_shape = (
            normalized_unbalanced_lambda_shape(
                float(clipped[2]),
                float(clipped[3]),
            )
        )
        return (
            equality_multiplier,
            inequality_multiplier,
            lambda1_shape,
            lambda2_shape,
        )

    if clipped.shape[0] != 3:
        raise ValueError(
            "expected three parameters when only inequality tuning is enabled"
        )
    inequality_multiplier = float(clipped[0])
    lambda1_shape, lambda2_shape = (
        normalized_unbalanced_lambda_shape(
            float(clipped[1]),
            float(clipped[2]),
        )
    )
    return (
        0.0,
        inequality_multiplier,
        lambda1_shape,
        lambda2_shape,
    )


def unbalanced_inequality_linear_terms(
    coeffs: np.ndarray,
    rhs: float,
) -> QuboTerms:
    """Return the UP inequality row's linear slack term."""
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    num_variables = coeffs.shape[0]
    return QuboTerms(
        quadratic=np.zeros(
            (num_variables, num_variables),
            dtype=float,
        ),
        linear=-coeffs,
        const=float(rhs),
    )


def unbalanced_inequality_quadratic_terms(
    coeffs: np.ndarray,
    rhs: float,
) -> QuboTerms:
    """Return the UP inequality row's quadratic slack term."""
    return squared_linear_form_qubo(
        np.asarray(coeffs, dtype=float),
        float(rhs),
    )


def unbalanced_inequality_row_terms(
    coeffs: np.ndarray,
    rhs: float,
    *,
    lambda1_shape: float,
    lambda2_shape: float,
) -> QuboTerms:
    """Return one full row-level UP inequality QUBO."""
    pieces = []
    if abs(float(lambda1_shape)) > 1e-12:
        pieces.append(
            scale_terms(
                unbalanced_inequality_linear_terms(
                    coeffs, rhs
                ),
                float(lambda1_shape),
            )
        )
    if abs(float(lambda2_shape)) > 1e-12:
        pieces.append(
            scale_terms(
                unbalanced_inequality_quadratic_terms(
                    coeffs, rhs
                ),
                float(lambda2_shape),
            )
        )
    if not pieces:
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        return zero_terms(coeffs.shape[0])
    return add_qubo_terms(*pieces)


def build_unbalanced_inequality_constraint_qubos(
    problem: BLP,
    *,
    lambda1_shape: float,
    lambda2_shape: float,
) -> list[QuboTerms]:
    """Return one full UP row QUBO per inequality constraint."""
    constraint_terms: list[QuboTerms] = []
    if problem.num_inequalities == 0:
        return constraint_terms
    for coeffs, rhs in zip(
        problem.A,
        problem.b,
        strict=True,
    ):
        constraint_terms.append(
            unbalanced_inequality_row_terms(
                np.asarray(coeffs, dtype=float),
                float(rhs),
                lambda1_shape=float(lambda1_shape),
                lambda2_shape=float(lambda2_shape),
            )
        )
    return constraint_terms


def build_unbalanced_components(
    problem: BLP,
    *,
    lambda1_shape: float,
    lambda2_shape: float,
    standardize: bool,
) -> UnbalancedPenaltyComponents:
    """Build the standardized UP equality and inequality blocks."""
    equality_terms = combine_constraint_terms(
        problem,
        build_unit_equality_constraint_qubos(problem),
        standardize=standardize,
    )
    inequality_terms = combine_constraint_terms(
        problem,
        build_unbalanced_inequality_constraint_qubos(
            problem,
            lambda1_shape=float(lambda1_shape),
            lambda2_shape=float(lambda2_shape),
        ),
        standardize=standardize,
    )
    return UnbalancedPenaltyComponents(
        equality_terms=equality_terms,
        inequality_terms=inequality_terms,
        lambda1_shape=float(lambda1_shape),
        lambda2_shape=float(lambda2_shape),
    )


def build_unbalanced_qubo_from_components(
    problem: BLP,
    components: UnbalancedPenaltyComponents,
    *,
    equality_multiplier: float,
    inequality_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose the deployed UP comparison QUBO."""
    terms = [objective_terms(problem)]
    if (
        problem.num_equalities
        and abs(float(equality_multiplier)) > 1e-12
    ):
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


__all__ = [
    "UP_BASE_PARAMETER_SOURCE",
    "UP_LAMBDA_GAUGE",
    "UP_NORMALIZATION_REGIME",
    "UnbalancedPenaltyComponents",
    "build_unbalanced_components",
    "build_unbalanced_inequality_constraint_qubos",
    "build_unbalanced_qubo_from_components",
    "normalized_unbalanced_lambda_shape",
    "unbalanced_inequality_linear_terms",
    "unbalanced_inequality_quadratic_terms",
    "unbalanced_inequality_row_terms",
    "unbalanced_multiplier_vector",
]
