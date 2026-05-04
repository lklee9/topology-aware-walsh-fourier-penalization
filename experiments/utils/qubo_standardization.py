"""
qubo_standardization.py
=======================
Helpers for variance-based standardization of QUBO objectives.

The main use case is to combine a linear BLP objective and a projected penalty
QUBO without expensive per-instance tuning. We standardize each QUBO by its
standard deviation under iid Bernoulli(0.5) variables, following the same
high-level idea as Lee et al. (2025).
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency.
    from qubolite import qubo as _qubolite_qubo
except (
    ImportError
):  # pragma: no cover - exercised when qubolite missing.
    _qubolite_qubo = None


def upper_triangular_qubo_matrix(
    quadratic: np.ndarray,
    linear: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert ``quadratic``/``linear`` coefficients into an upper-triangular QUBO.

    The returned matrix ``U`` is interpreted via ``x^T U x`` with binary
    variables, so the diagonal stores the linear terms and the strict upper
    triangle stores the pairwise couplings exactly once.
    """
    quadratic = np.asarray(quadratic, dtype=float)
    if (
        quadratic.ndim != 2
        or quadratic.shape[0] != quadratic.shape[1]
    ):
        raise ValueError(
            "quadratic must be a square matrix"
        )

    n = quadratic.shape[0]
    if linear is None:
        linear_vec = np.zeros(n, dtype=float)
    else:
        linear_vec = np.asarray(
            linear, dtype=float
        ).reshape(-1)
        if linear_vec.shape[0] != n:
            raise ValueError(f"linear must have length {n}")

    U = np.triu(quadratic, k=1).astype(float, copy=True)
    U[np.diag_indices(n)] = np.diag(quadratic) + linear_vec
    return U


def qubo_variance_matrix(
    qubo_matrix: np.ndarray,
    use_qubolite: bool = True,
) -> float:
    """
    Return the exact variance of ``x^T Q x`` for iid ``Bernoulli(0.5)`` inputs.

    The input ``qubo_matrix`` is interpreted in upper-triangular QUBO form:
    diagonal entries are linear terms and strict upper-triangular entries are
    pairwise couplings. Constant offsets are ignored because they do not affect
    the variance.

    When ``qubolite`` is installed we use its variance routine first. Otherwise
    we fall back to a closed-form expression derived from the change of
    variables ``x_i = 1/2 + y_i`` with independent centered ``y_i``.
    """
    Q = np.asarray(qubo_matrix, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("qubo_matrix must be square")

    upper = np.triu(Q)
    if use_qubolite and _qubolite_qubo is not None:
        try:
            return float(_qubolite_qubo(upper).variance())
        except Exception:
            pass

    linear = np.diag(upper).copy()
    quadratic = np.triu(upper, k=1)

    # Expanding x_i x_j = 1/4 + 1/2 y_i + 1/2 y_j + y_i y_j with
    # y_i = x_i - 1/2 yields an orthogonal decomposition under iid
    # Bernoulli(0.5), so the variance is a sum of squares.
    centered_linear = linear + 0.5 * (
        quadratic.sum(axis=0) + quadratic.sum(axis=1)
    )
    variance = 0.25 * float(
        np.dot(centered_linear, centered_linear)
    )
    variance += 0.0625 * float(np.sum(quadratic**2))
    return max(variance, 0.0)


def qubo_standard_deviation(
    qubo_matrix: np.ndarray,
    use_qubolite: bool = True,
) -> float:
    """Return the exact standard deviation of ``x^T Q x`` under iid Bernoulli(0.5)."""
    return float(
        np.sqrt(
            qubo_variance_matrix(
                qubo_matrix, use_qubolite=use_qubolite
            )
        )
    )


def standardized_penalty_scale(
    objective_linear: np.ndarray,
    penalty_quadratic: np.ndarray,
    penalty_linear: np.ndarray,
    *,
    eps: float = 1e-12,
    use_qubolite: bool = True,
) -> tuple[float, float, float]:
    """
    Return the deployment-style penalty scale from QUBO standard deviations.

    If we combine the standardized objective and penalty

        ``f(x) / sigma_f + g(x) / sigma_g``,

    then, up to multiplication by the positive constant ``sigma_f``, this is
    equivalent to minimizing

        ``f(x) + lambda * g(x)``

    with ``lambda = sigma_f / sigma_g``.

    The returned tuple is ``(lambda, sigma_f, sigma_g)``. If the projected
    penalty is effectively constant under the Bernoulli(0.5) reference law,
    then ``sigma_g`` is treated as zero and the returned scale is ``lambda=0``.
    """
    objective_linear = np.asarray(
        objective_linear, dtype=float
    ).reshape(-1)
    objective_matrix = np.diag(objective_linear)
    penalty_matrix = upper_triangular_qubo_matrix(
        penalty_quadratic, penalty_linear
    )

    objective_std = qubo_standard_deviation(
        objective_matrix, use_qubolite=use_qubolite
    )
    penalty_std = qubo_standard_deviation(
        penalty_matrix, use_qubolite=use_qubolite
    )

    if penalty_std <= eps:
        return 0.0, max(objective_std, eps), 0.0
    return (
        max(objective_std, eps) / penalty_std,
        objective_std,
        penalty_std,
    )


__all__ = [
    "qubo_standard_deviation",
    "qubo_variance_matrix",
    "standardized_penalty_scale",
    "upper_triangular_qubo_matrix",
]
