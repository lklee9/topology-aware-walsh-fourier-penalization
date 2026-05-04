from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class BinMassEstimate:
    """
    Estimated masses for user-specified bins of s(x) = a^T x - b.

    Attributes
    ----------
    bin_edges : np.ndarray
        Strictly increasing bin edges. Bin j is [bin_edges[j], bin_edges[j+1]),
        except the last bin is closed on the right for convenience when assigning
        quantized support points that land exactly on the final edge.
    bin_masses : np.ndarray
        Estimated probability mass in each requested bin.
    delta : float
        Internal quantization resolution used for the DP.
    k : np.ndarray
        Integer quantized coefficients, k_i = round(a_i / delta) by default.
    total_mass : float
        Sum of returned bin masses. This can be < 1 if the requested bins do not
        cover the full support of the quantized approximation.
    lost_mass : float
        Quantized mass that fell outside the requested bins.
    b : float
        Shift parameter in s(x) = a^T x - b.
    """

    bin_edges: np.ndarray
    bin_masses: np.ndarray
    delta: float
    k: np.ndarray
    total_mass: float
    lost_mass: float
    b: float

    def prob_geq(self, threshold: float) -> float:
        """
        Approximate P(s >= threshold) using the already-aggregated user bins.

        This is exact only when the threshold aligns with bin boundaries; otherwise
        it is a coarse bin-level approximation.
        """
        left = self.bin_edges[:-1]
        return float(
            self.bin_masses[left >= threshold].sum()
        )

    def prob_leq(self, threshold: float) -> float:
        """
        Approximate P(s <= threshold) using the already-aggregated user bins.

        This is exact only when the threshold aligns with bin boundaries; otherwise
        it is a coarse bin-level approximation.
        """
        right = self.bin_edges[1:]
        return float(
            self.bin_masses[right <= threshold].sum()
        )


def estimate_masses_in_bins_quantized_dp(
    a: Iterable[float],
    p: Iterable[float],
    *,
    bin_edges: Iterable[float],
    b: float = 0.0,
    delta: Optional[float] = None,
    rounding: str = "nearest",
    drop_tolerance: Optional[float] = None,
) -> BinMassEstimate:
    """
    Estimate the probability mass of user-specified bins for

        s(x) = a^T x - b,

    where x_i are independent Bernoulli(p_i), using quantization + sparse DP.

    Parameters
    ----------
    a : iterable of float
        Real coefficients a_i.
    p : iterable of float
        Bernoulli probabilities p_i in [0, 1].
    bin_edges : iterable of float
        Strictly increasing bin edges. The returned masses correspond to bins
        [bin_edges[j], bin_edges[j+1]) for j = 0, ..., m-2, with the final edge
        treated as inclusive when assigning quantized support points.
    b : float, default=0.0
        Shift in s(x) = a^T x - b.
    delta : float or None, default=None
        Internal quantization resolution. If None, it is chosen as one quarter of
        the smallest requested bin width. A smaller delta usually improves accuracy
        but increases runtime and memory.
    rounding : {"nearest", "floor", "ceil"}, default="nearest"
        Rule used to map a_i / delta to integer k_i.
    drop_tolerance : float or None, default=None
        Optional pruning threshold for the sparse DP. States with mass below this
        threshold are discarded after each update. This speeds things up but adds
        extra approximation on top of quantization.

    Returns
    -------
    BinMassEstimate
        Estimated masses of the requested bins.

    Method
    ------
    We quantize each coefficient as

        k_i = round(a_i / delta),
        a_i ≈ delta * k_i.

    Then we compute exactly the pmf of the integer-valued lattice sum

        Z = sum_i k_i x_i

    using the recursion

        f_i(m) = (1 - p_i) f_{i-1}(m) + p_i f_{i-1}(m - k_i).

    Finally, each lattice state m is mapped to the quantized real value

        s_tilde = delta * m - b,

    and its mass is aggregated into the requested user bin containing s_tilde.

    Interpretation
    --------------
    The returned masses are best understood as approximate bin masses for the true
    distribution of s. They are not exact atomic masses of the original real-valued
    subset-sum distribution.
    """
    a = np.asarray(list(a), dtype=float)
    p = np.asarray(list(p), dtype=float)
    bin_edges = np.asarray(list(bin_edges), dtype=float)

    if a.ndim != 1 or p.ndim != 1 or len(a) != len(p):
        raise ValueError(
            "a and p must be 1D arrays of the same length."
        )
    if len(a) == 0:
        raise ValueError("a and p must be non-empty.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("All p_i must lie in [0, 1].")
    if not np.isfinite(b):
        raise ValueError("b must be finite.")
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
        raise ValueError(
            "bin_edges must be a 1D array-like with at least two values."
        )
    if np.any(~np.isfinite(bin_edges)):
        raise ValueError("bin_edges must be finite.")
    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError(
            "bin_edges must be strictly increasing."
        )
    if rounding not in {"nearest", "floor", "ceil"}:
        raise ValueError(
            "rounding must be one of {'nearest', 'floor', 'ceil'}."
        )
    if drop_tolerance is not None and drop_tolerance < 0:
        raise ValueError(
            "drop_tolerance must be nonnegative or None."
        )

    if delta is None:
        min_bin_width = float(np.min(np.diff(bin_edges)))
        delta = min_bin_width / 4.0
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(
            "delta must be a positive finite number."
        )

    scaled = a / delta
    if rounding == "nearest":
        k = np.rint(scaled).astype(int)
    elif rounding == "floor":
        k = np.floor(scaled).astype(int)
    else:
        k = np.ceil(scaled).astype(int)

    # Sparse DP for the quantized lattice pmf of Z = sum_i k_i x_i.
    pmf: dict[int, float] = {0: 1.0}
    for ki, pi in zip(k, p):
        new_pmf = defaultdict(float)
        if pi == 0.0:
            for m, mass in pmf.items():
                new_pmf[m] += mass
        elif pi == 1.0:
            for m, mass in pmf.items():
                new_pmf[m + ki] += mass
        else:
            q = 1.0 - pi
            for m, mass in pmf.items():
                new_pmf[m] += q * mass
                new_pmf[m + ki] += pi * mass

        if drop_tolerance is None:
            pmf = dict(new_pmf)
        else:
            pmf = {
                m: mass
                for m, mass in new_pmf.items()
                if mass >= drop_tolerance
            }

    bin_masses = np.zeros(len(bin_edges) - 1, dtype=float)
    lost_mass = 0.0

    for m, mass in pmf.items():
        s_tilde = delta * m - b

        # Assign to bin j such that bin_edges[j] <= s_tilde < bin_edges[j+1].
        j = (
            np.searchsorted(
                bin_edges, s_tilde, side="right"
            )
            - 1
        )

        if np.isclose(s_tilde, bin_edges[-1]):
            j = len(bin_masses) - 1

        if 0 <= j < len(bin_masses):
            # Guard against values below the left edge that still produce j == 0.
            if s_tilde < bin_edges[0] and not np.isclose(
                s_tilde, bin_edges[0]
            ):
                lost_mass += mass
            else:
                bin_masses[j] += mass
        else:
            lost_mass += mass

    return BinMassEstimate(
        bin_edges=bin_edges,
        bin_masses=bin_masses,
        delta=float(delta),
        k=k,
        total_mass=float(bin_masses.sum()),
        lost_mass=float(lost_mass),
        b=float(b),
    )


if __name__ == "__main__":
    a = [0.9, -1.3, 2.2, 0.55]
    p = [0.2, 0.5, 0.7, 0.4]
    b = 0.4
    bins = [-3.0, -1.0, 0.0, 1.0, 3.0]

    est = estimate_masses_in_bins_quantized_dp(
        a, p, bin_edges=bins, b=b, delta=0.1
    )

    print("delta:", est.delta)
    print("quantized k:", est.k)
    print("total captured mass:", est.total_mass)
    print(
        "lost mass outside requested bins:", est.lost_mass
    )
    print()

    for left, right, mass in zip(
        est.bin_edges[:-1],
        est.bin_edges[1:],
        est.bin_masses,
    ):
        print(
            f"[{left:6.2f}, {right:6.2f})  mass={mass:.6f}"
        )
