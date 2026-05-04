"""Shared projected-penalty QUBO helpers used across experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.utils.qubo_standardization import (
    standardized_penalty_scale,
)
from experiments.utils.unb_pen import (
    UnbalancedPenaltyParameters,
)
from fourier_projection.blp import BLP


@dataclass(frozen=True)
class QuboTerms:
    """One QUBO in ``const + linear^T x + sum_{i<j} Q_ij x_i x_j`` form."""

    quadratic: np.ndarray
    linear: np.ndarray
    const: float


def zero_terms(num_variables: int) -> QuboTerms:
    """Return the all-zero QUBO of the requested size."""
    return QuboTerms(
        quadratic=np.zeros(
            (num_variables, num_variables), dtype=float
        ),
        linear=np.zeros(num_variables, dtype=float),
        const=0.0,
    )


def scale_terms(
    terms: QuboTerms, scale: float
) -> QuboTerms:
    """Scale every coefficient of one QUBO."""
    return QuboTerms(
        quadratic=float(scale) * terms.quadratic,
        linear=float(scale) * terms.linear,
        const=float(scale) * terms.const,
    )


def objective_terms(problem: BLP) -> QuboTerms:
    """Return the bare linear objective as QUBO terms."""
    return QuboTerms(
        quadratic=np.zeros(
            (problem.num_variables, problem.num_variables),
            dtype=float,
        ),
        linear=problem.c.copy(),
        const=float(problem.objective_constant),
    )


def add_qubo_terms(*terms: QuboTerms) -> QuboTerms:
    """Add a sequence of aligned QUBOs."""
    if not terms:
        raise ValueError(
            "at least one QUBO term bundle is required"
        )
    quadratic = np.zeros_like(terms[0].quadratic)
    linear = np.zeros_like(terms[0].linear)
    const = 0.0
    for item in terms:
        quadratic += item.quadratic
        linear += item.linear
        const += float(item.const)
    return QuboTerms(
        quadratic=quadratic, linear=linear, const=const
    )


def projection_sample_size(
    problem: BLP, sample_cap_log2: int
) -> int:
    """Return the sampled-projection budget ``2^min(n-1, sample_cap_log2)``."""
    if sample_cap_log2 < 0:
        raise ValueError(
            "sample_cap_log2 must be nonnegative"
        )
    exponent = max(
        0, min(problem.num_variables - 1, sample_cap_log2)
    )
    return 1 << exponent


def projected_multiplier_vector(
    raw_params: np.ndarray,
    *,
    has_equality: bool,
) -> tuple[float, float]:
    """Map linear tuning coordinates to nonnegative projected multipliers."""
    params = np.asarray(raw_params, dtype=float).reshape(-1)
    clipped = np.clip(params, 0.0, np.inf)
    if has_equality:
        if clipped.shape[0] != 2:
            raise ValueError(
                "expected two parameters when equality tuning is enabled"
            )
        return float(clipped[0]), float(clipped[1])
    if clipped.shape[0] != 1:
        raise ValueError(
            "expected one parameter when only inequality tuning is enabled"
        )
    return 0.0, float(clipped[0])


def unbalanced_parameter_vector(
    raw_params: np.ndarray,
    *,
    has_equality: bool,
) -> UnbalancedPenaltyParameters:
    """Map linear tuning coordinates to nonnegative unbalanced lambdas."""
    params = np.asarray(raw_params, dtype=float).reshape(-1)
    clipped = np.clip(params, 0.0, np.inf)
    if has_equality:
        if clipped.shape[0] != 3:
            raise ValueError(
                "expected three parameters when equality tuning is enabled"
            )
        return UnbalancedPenaltyParameters(
            lambda0=float(clipped[0]),
            lambda1=float(clipped[1]),
            lambda2=float(clipped[2]),
        )
    if clipped.shape[0] != 2:
        raise ValueError(
            "expected two parameters when only inequality tuning is enabled"
        )
    return UnbalancedPenaltyParameters(
        lambda0=None,
        lambda1=float(clipped[0]),
        lambda2=float(clipped[1]),
    )


def squared_linear_form_qubo(
    coeffs: np.ndarray, rhs: float
) -> QuboTerms:
    """Return ``(coeffs^T x - rhs)^2`` as one QUBO bundle."""
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    return QuboTerms(
        quadratic=2.0
        * np.triu(np.outer(coeffs, coeffs), k=1),
        linear=coeffs**2 - 2.0 * float(rhs) * coeffs,
        const=float(rhs) ** 2,
    )


def combine_constraint_terms(
    problem: BLP,
    constraint_terms: list[QuboTerms],
    *,
    standardize: bool,
) -> QuboTerms:
    """Optionally standardize each constraint QUBO before summing it."""
    if not constraint_terms:
        return zero_terms(problem.num_variables)

    combined_terms: list[QuboTerms] = []
    for terms in constraint_terms:
        scale = 1.0
        if standardize:
            scale, _, _ = standardized_penalty_scale(
                objective_linear=problem.c,
                penalty_quadratic=terms.quadratic,
                penalty_linear=terms.linear,
            )
        if scale == 0.0:
            continue
        if scale == 1.0:
            combined_terms.append(terms)
        else:
            combined_terms.append(scale_terms(terms, scale))

    if not combined_terms:
        return zero_terms(problem.num_variables)
    return add_qubo_terms(*combined_terms)


def build_unit_equality_constraint_qubos(
    problem: BLP,
) -> list[QuboTerms]:
    """Return one exact equality-penalty QUBO per equality constraint."""
    constraint_terms: list[QuboTerms] = []
    for coeffs, rhs in zip(
        problem.D,
        problem.e,
        strict=True,
    ):
        constraint_terms.append(
            squared_linear_form_qubo(
                np.asarray(coeffs, dtype=float), float(rhs)
            )
        )
    return constraint_terms


def build_unit_equality_qubo(problem: BLP) -> QuboTerms:
    """Return the exact equality penalty ``sum_r (a_r^T x - b_r)^2``."""
    constraint_terms = build_unit_equality_constraint_qubos(
        problem
    )
    if not constraint_terms:
        return zero_terms(problem.num_variables)
    return add_qubo_terms(*constraint_terms)


__all__ = [
    "QuboTerms",
    "add_qubo_terms",
    "build_unit_equality_constraint_qubos",
    "build_unit_equality_qubo",
    "combine_constraint_terms",
    "objective_terms",
    "projected_multiplier_vector",
    "projection_sample_size",
    "scale_terms",
    "squared_linear_form_qubo",
    "unbalanced_parameter_vector",
    "zero_terms",
]
