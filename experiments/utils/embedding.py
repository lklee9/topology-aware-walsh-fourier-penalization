"""Small utilities for converting QUBO arrays to a dimod BQM.

This centralises the helper so callers don't duplicate logic and avoids
accidental divergences between modules.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import dimod  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dimod = None  # type: ignore


def _qubo_arrays_to_bqm(
    quadratic: np.ndarray, linear: np.ndarray, const: float
) -> "dimod.BQM":
    """Convert QUBO arrays to a dimod BQM.

    The function accepts dense quadratic and linear arrays and returns a
    dimod.BinaryQuadraticModel (BQM). If dimod is unavailable the function
    raises ImportError to let callers handle the fallback.
    """
    if (
        dimod is None
    ):  # pragma: no cover - exercised only when dimod present
        raise ImportError(
            "dimod is required to convert QUBO arrays to BQM"
        )

    n = linear.shape[0]
    # Create a dict for quadratic terms where keys are (i, j) with i < j
    quad: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = float(quadratic[i, j])
            if v != 0.0:
                quad[(i, j)] = v

    linear_dict = {i: float(linear[i]) for i in range(n)}

    # dimod's BQM uses vartype=BINARY for binary variables
    bqm = dimod.BinaryQuadraticModel(
        linear_dict,
        quad,
        float(const),
        vartype=dimod.BINARY,
    )
    return bqm
