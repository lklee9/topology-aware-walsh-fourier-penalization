"""Paper-faithful unbalanced-penalty evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnbalancedPenaltyParameters:
    """Table-1 penalty parameters from Montanez-Barrera et al. (2024)."""

    lambda0: float | None
    lambda1: float
    lambda2: float


def qubo_energy_values(
    bitstrings: np.ndarray,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float = 0.0,
) -> np.ndarray:
    """Evaluate one upper-triangular binary QUBO."""
    X = np.asarray(bitstrings, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    linear = np.asarray(linear, dtype=float).reshape(-1)
    quadratic = np.asarray(quadratic, dtype=float)
    if X.shape[1] != linear.shape[0]:
        raise ValueError(
            "bitstrings and linear term have incompatible sizes"
        )
    if quadratic.shape != (
        linear.shape[0],
        linear.shape[0],
    ):
        raise ValueError(
            "quadratic must be square with the same dimension as linear"
        )

    energies = np.full(
        X.shape[0], float(const), dtype=float
    )
    energies += X @ linear
    if np.any(quadratic):
        energies += np.einsum(
            "bi,ij,bj->b", X, np.triu(quadratic, k=1), X
        )
    return energies


__all__ = [
    "UnbalancedPenaltyParameters",
    "qubo_energy_values",
]
