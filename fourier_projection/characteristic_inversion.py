from __future__ import annotations

import math
from itertools import product
from typing import Sequence

import numpy as np
from scipy.integrate import quad


def _sinc_unscaled(x: float) -> float:
    """Return sin(x)/x with the removable singularity handled at x = 0."""
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


def slack_interval_prob_p_half(
    a: Sequence[float],
    b: float,
    ell: float,
    r: float,
    *,
    initial_panels: int = 16,
    max_panels: int = 4096,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    clip_to_unit_interval: bool = True,
) -> float:
    r"""
    Compute P(ell < s <= r) for

        s = a^T x - b

    where x_i are i.i.d. Bernoulli(1/2), using the centered one-sided
    characteristic-function inversion formula

        P(ell < s <= r)
        = (2h/pi) \int_0^\infty
            [prod_i cos(w a_i / 2)] cos(w m) sinc(w h) dw,

    with
        mu = (1/2) * sum_i a_i - b,
        m  = (ell + r)/2 - mu,
        h  = (r - ell)/2,
        sinc(u) = sin(u)/u.

    Numerical method
    ----------------
    The integral is split into panels at the zeros of sinc(w h), i.e.

        w_k = k * pi / h,   k = 0, 1, 2, ...

    Each panel is integrated with scipy.integrate.quad, and the number of
    panels is doubled until the partial sums stabilize.

    Parameters
    ----------
    a : sequence of floats
        Real coefficients.
    b : float
        Scalar offset.
    ell, r : float
        Interval endpoints for the half-open bin (ell, r].
    initial_panels : int
        Initial number of sinc-zero panels to integrate.
    max_panels : int
        Hard cap on the number of panels.
    epsabs, epsrel : float
        Absolute and relative stopping tolerances for both the per-panel quad
        calls and the outer tail-stabilization test.
    clip_to_unit_interval : bool
        If True, clip the final answer to [0, 1] to suppress tiny numerical
        spillover.

    Returns
    -------
    float
        Approximation to P(ell < s <= r).

    Notes
    -----
    - This implementation is specialized to p_i = 1/2.
    - For very narrow bins or highly oscillatory coefficient sets, you may need
      more panels and/or tighter tolerances.
    """
    if not (ell < r):
        return 0.0

    a = np.asarray(a, dtype=float)
    b = float(b)
    ell = float(ell)
    r = float(r)

    h = 0.5 * (r - ell)
    if h <= 0.0:
        return 0.0

    mu = 0.5 * float(np.sum(a)) - b
    m = 0.5 * (ell + r) - mu

    prefactor = 2.0 * h / math.pi
    panel_width = math.pi / h

    def integrand(w: float) -> float:
        # Product characteristic function for the centered variable:
        # psi(w) = prod_i cos(w a_i / 2)
        psi = float(np.prod(np.cos(0.5 * a * w)))
        return (
            prefactor
            * psi
            * math.cos(m * w)
            * _sinc_unscaled(h * w)
        )

    def panel_integral(k: int) -> float:
        left = k * panel_width
        right = (k + 1) * panel_width
        value, _ = quad(
            integrand,
            left,
            right,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=200,
        )
        return value

    def partial_sum(num_panels: int) -> float:
        total = 0.0
        for k in range(num_panels):
            total += panel_integral(k)
        return total

    num_panels = max(4, int(initial_panels))
    previous = partial_sum(num_panels)

    while True:
        next_num_panels = min(2 * num_panels, max_panels)
        current = partial_sum(next_num_panels)

        if abs(current - previous) <= max(
            epsabs, epsrel * abs(current)
        ):
            result = current
            break

        if next_num_panels >= max_panels:
            result = current
            break

        num_panels = next_num_panels
        previous = current

    if clip_to_unit_interval:
        result = min(1.0, max(0.0, result))

    return result


def slack_cdf_p_half(
    a: Sequence[float],
    b: float,
    t: float,
    **kwargs,
) -> float:
    """
    Compute P(s <= t) for s = a^T x - b with x_i ~ Bernoulli(1/2).

    Since the interval formula uses a half-open bin (ell, r], we approximate
    P(s <= t) by integrating over a wide interval (L, t], where L is chosen
    below the minimum possible support point.
    """
    a = np.asarray(a, dtype=float)
    min_support = float(np.sum(np.minimum(a, 0.0)) - b)
    left = min_support - 1.0
    return slack_interval_prob_p_half(
        a, b, left, t, **kwargs
    )


def brute_force_slack_interval_prob_p_half(
    a: Sequence[float],
    b: float,
    ell: float,
    r: float,
) -> float:
    """
    Exact brute-force reference for small n.

    Computes P(ell < s <= r) by enumerating all 2^n binary vectors.
    Only suitable for small n.
    """
    a = list(map(float, a))
    b = float(b)
    ell = float(ell)
    r = float(r)

    n = len(a)
    total = 0.0
    atom_prob = 2.0 ** (-n)

    for x in product((0, 1), repeat=n):
        s = sum(ai * xi for ai, xi in zip(a, x)) - b
        if ell < s <= r:
            total += atom_prob

    return total
