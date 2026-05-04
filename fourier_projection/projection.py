"""
projection.py
=============
Hardware-aware penalty function approximation for Binary Linear Programs (BLPs)
on quantum annealers via Walsh-Hadamard Fourier projection.

The core routines in this module work with either

1. the full hypercube ``{0, 1}^n`` for exact moment computation, or
2. an arbitrary sample of bitstrings for empirical/importance-weighted
   estimates of the same moments.

In both cases we fit the hardware-restricted projection by solving the
normal equations from Eq. 33 of ``brainstorm/main.tex``:

    G theta = c,

where ``G`` is the Gram matrix of the admissible Walsh features under the
chosen measure and ``c`` is their correlation vector with the target penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    import torch
except (
    ImportError
):  # pragma: no cover - optional runtime dependency.
    torch = None

try:
    from .blp import BLP
    from .measures import GaussianSlackTarget
    from .penalties import IdealPenalty
    from .sampling import BinnedTargetSaddlepointIS
    from .topology import HardwareTopology
except (
    ImportError
):  # pragma: no cover - allows running the file as a script.
    from blp import BLP
    from measures import GaussianSlackTarget
    from penalties import IdealPenalty
    from sampling import BinnedTargetSaddlepointIS
    from topology import HardwareTopology


DEFAULT_PROJECTION_BACKEND = "torch"


def _as_binary_matrix(
    sample_bits: np.ndarray, n: int
) -> np.ndarray:
    """Validate a binary sample matrix and return it as ``float``."""
    X = np.asarray(sample_bits)
    if X.ndim != 2:
        raise ValueError(
            "sample_bits must be a 2D array of shape (N, n)"
        )
    if X.shape[1] != n:
        raise ValueError(
            f"sample_bits must have {n} columns"
        )
    if X.shape[0] == 0:
        raise ValueError(
            "sample_bits must contain at least one bitstring"
        )
    if not np.all((X == 0) | (X == 1)):
        raise ValueError(
            "sample_bits must contain only 0/1 values"
        )
    return X.astype(float, copy=False)


def _as_spin_matrix(sample_spins: np.ndarray) -> np.ndarray:
    """Validate a spin sample matrix with entries in ``{-1, 1}``."""
    Z = np.asarray(sample_spins)
    if Z.ndim != 2:
        raise ValueError(
            "sample_spins must be a 2D array of shape (N, n)"
        )
    if Z.shape[0] == 0:
        raise ValueError(
            "sample_spins must contain at least one spin vector"
        )
    if not np.all((Z == -1) | (Z == 1)):
        raise ValueError(
            "sample_spins must contain only -1/+1 values"
        )
    return Z.astype(float, copy=False)


def _as_vector(
    values: np.ndarray, length: int, name: str
) -> np.ndarray:
    """Validate a vector aligned with the sample rows."""
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.shape[0] != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError(
            f"{name} must contain only finite values"
        )
    return vec


def _as_penalty_matrix(
    values: np.ndarray,
    num_rows: int,
    name: str,
) -> np.ndarray:
    """Validate a matrix of aligned penalty vectors."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != num_rows:
            raise ValueError(
                f"{name} must have {num_rows} rows"
            )
        arr = arr.reshape(num_rows, 1)
    if arr.ndim != 2 or arr.shape[0] != num_rows:
        raise ValueError(
            f"{name} must have shape ({num_rows}, k)"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} must contain only finite values"
        )
    return arr


def _resolve_projection_backend(backend: str) -> str:
    """Validate one projection backend name."""
    backend_name = str(backend).lower()
    if backend_name not in {"numpy", "torch"}:
        raise ValueError(
            "backend must be either 'numpy' or 'torch'"
        )
    if backend_name == "torch" and torch is None:
        raise ImportError(
            "backend='torch' requires PyTorch to be installed"
        )
    return backend_name


def _resolve_torch_dtype(torch_dtype: object | None):
    """Return one torch dtype, defaulting to float64 for stability."""
    if torch is None:
        raise ImportError(
            "PyTorch is required to resolve a torch dtype"
        )
    if torch_dtype is None:
        return torch.float64
    if isinstance(torch_dtype, str):
        dtype_name = torch_dtype.lower()
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float64":
            return torch.float64
        raise ValueError(
            "torch_dtype must be float32, float64, "
            "torch.float32, or torch.float64"
        )
    if torch_dtype in {torch.float32, torch.float64}:
        return torch_dtype
    try:
        dtype_name = np.dtype(torch_dtype).name
    except TypeError as exc:
        raise ValueError(
            "torch_dtype must be float32, float64, "
            "torch.float32, or torch.float64"
        ) from exc
    return _resolve_torch_dtype(dtype_name)


def _resolve_torch_device(torch_device: str | None):
    """Return one torch device, preferring CUDA when available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required to resolve a torch device"
        )
    if torch_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(torch_device)
    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "torch_device='cuda' was requested but CUDA is not available"
        )
    return device


def _tensor_to_numpy(value: object) -> np.ndarray:
    """Convert one backend value into a NumPy array."""
    if torch is not None and isinstance(
        value, torch.Tensor
    ):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalise_weights(
    mu: np.ndarray | None, num_rows: int
) -> np.ndarray:
    """
    Return a probability vector aligned with the supplied rows.

    If ``mu`` is omitted we interpret the rows as an unweighted sample and use
    the empirical distribution, i.e. each row receives mass ``1 / num_rows``.
    """
    if mu is None:
        return np.full(num_rows, 1.0 / num_rows)

    weights = _as_vector(mu, num_rows, "mu")
    if np.any(weights < 0):
        raise ValueError("mu must be nonnegative")

    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("mu must have positive total mass")
    return weights / total


def _importance_weights_from_law(
    importance_law: object,
    sample_bits: np.ndarray,
) -> np.ndarray:
    """Return raw importance ratios from one supported score-law object."""
    if hasattr(importance_law, "importance_weights"):
        weights = importance_law.importance_weights(
            X=sample_bits
        )
    elif hasattr(importance_law, "law") and hasattr(
        importance_law.law, "importance_weights"
    ):
        weights = importance_law.law.importance_weights(
            X=sample_bits
        )
    else:
        raise TypeError(
            "importance_law must expose importance_weights(X=...) directly "
            "or through a .law attribute"
        )
    return np.asarray(weights, dtype=float)


def _importance_weights(
    *,
    sample_bits: np.ndarray,
    num_rows: int,
    importance_weights: np.ndarray | None = None,
    importance_law: object | None = None,
) -> np.ndarray:
    """Return self-normalized importance weights aligned with sample rows."""
    if (importance_weights is None) == (
        importance_law is None
    ):
        raise ValueError(
            "provide exactly one of importance_weights or importance_law"
        )
    if importance_law is not None:
        weights = _importance_weights_from_law(
            importance_law, sample_bits
        )
    else:
        weights = importance_weights
    ratios = _as_vector(
        weights, num_rows, "importance_weights"
    )
    if np.any(ratios < 0.0):
        raise ValueError(
            "importance_weights must be nonnegative"
        )
    total = float(np.sum(ratios))
    if total <= 0.0:
        raise ValueError(
            "importance_weights must have positive total mass"
        )
    return ratios / total


def _weighted_normal_equation(
    Phi: np.ndarray,
    weights: np.ndarray,
    P_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted Gram matrix and correlation matrix."""
    penalties = _as_penalty_matrix(
        P_vals,
        Phi.shape[0],
        "P_vals",
    )
    weighted_phi = Phi * weights[:, np.newaxis]
    G = Phi.T @ weighted_phi
    c_matrix = weighted_phi.T @ penalties
    return G, c_matrix


def _solve_regularized_gram_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve one regularized Gram system, preferring Cholesky."""
    try:
        factor, lower = cho_factor(
            matrix,
            lower=True,
            check_finite=False,
        )
        return cho_solve(
            (factor, lower),
            rhs,
            check_finite=False,
        )
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix, rhs)


def _evaluate_theta_on_enum(
    enum: "HypercubeSampleEnumerator",
    calE,
    theta: dict[frozenset[int], float],
) -> np.ndarray:
    """Evaluate fitted Walsh coefficients on an arbitrary enumerator."""
    phi = enum.design_matrix(calE)
    theta_vec = np.array(
        [float(theta.get(S, 0.0)) for S in calE],
        dtype=float,
    )
    return phi @ theta_vec


def _as_optional_measure_sequence(
    values: Sequence[np.ndarray | None] | None,
    length: int,
    name: str,
) -> tuple[np.ndarray | None, ...]:
    """Validate a sequence of optional measure vectors."""
    if values is None:
        return tuple(None for _ in range(length))
    seq = tuple(values)
    if len(seq) != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return seq


def _as_optional_vector_sequence(
    values: Sequence[np.ndarray | None] | None,
    length: int,
    name: str,
) -> tuple[np.ndarray | None, ...]:
    """Validate a sequence of optional vectors without reshaping yet."""
    if values is None:
        return tuple(None for _ in range(length))
    seq = tuple(values)
    if len(seq) != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return seq


def _as_optional_object_sequence(
    values: Sequence[object | None] | None,
    length: int,
    name: str,
) -> tuple[object | None, ...]:
    """Validate a sequence of optional objects without inspecting members."""
    if values is None:
        return tuple(None for _ in range(length))
    seq = tuple(values)
    if len(seq) != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return seq


def _broadcast_optional_vector(
    values: np.ndarray | None,
    length: int,
) -> tuple[np.ndarray | None, ...]:
    """Broadcast one optional vector across several constraint fits."""
    return tuple(values for _ in range(length))


def _broadcast_optional_object(
    values: object | None,
    length: int,
) -> tuple[object | None, ...]:
    """Broadcast one optional object across several constraint fits."""
    return tuple(values for _ in range(length))


def _as_enumerator_sequence(
    values: Sequence["HypercubeSampleEnumerator"] | None,
    default: "HypercubeSampleEnumerator" | None,
    length: int,
    name: str,
) -> tuple["HypercubeSampleEnumerator", ...]:
    """Validate a sequence of enumerators or broadcast one default enumerator."""
    if values is None:
        if default is None:
            raise ValueError(f"{name} must be provided")
        return tuple(default for _ in range(length))
    seq = tuple(values)
    if len(seq) != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return seq


def _as_weight_sequence(
    values: float | Sequence[float],
    length: int,
    name: str,
) -> tuple[float, ...]:
    """Validate a scalar or one weight per component."""
    if np.isscalar(values):
        return tuple(float(values) for _ in range(length))
    seq = tuple(
        float(value)
        for value in np.asarray(
            values, dtype=float
        ).reshape(-1)
    )
    if len(seq) != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return seq


def _sum_projected_fits(
    fits: Sequence[ProjectedPenaltyFit],
) -> ProjectedPenaltyFit | None:
    """Aggregate several projected fits into one compatibility summary."""
    seq = tuple(fits)
    if not seq:
        return None
    theta_total: dict[frozenset[int], float] = {}
    for fit in seq:
        for subset, value in fit.theta.items():
            theta_total[subset] = theta_total.get(
                subset, 0.0
            ) + float(value)
    return ProjectedPenaltyFit(
        values=np.sum([fit.values for fit in seq], axis=0),
        quadratic=np.sum(
            [fit.quadratic for fit in seq], axis=0
        ),
        linear=np.sum([fit.linear for fit in seq], axis=0),
        const=float(sum(fit.const for fit in seq)),
        theta=theta_total,
        target_values=np.sum(
            [fit.target_values for fit in seq], axis=0
        ),
        measure=None,
        fit_enum_size=seq[0].fit_enum_size,
    )


def _equality_gaussian_measure(
    enum: "HypercubeSampleEnumerator",
    blp: BLP,
    equality_idx: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Return the Gaussian measure centered on equality satisfaction."""
    if equality_idx < 0 or equality_idx >= blp.p:
        raise IndexError(
            f"equality index out of range: {equality_idx}"
        )
    coeffs = np.asarray(blp.D[equality_idx], dtype=float)
    rhs = float(blp.e[equality_idx])
    residuals = enum.X @ coeffs - rhs
    if sigma is None:
        sigma2 = max(
            0.25 * float(np.dot(coeffs, coeffs)), 1e-12
        )
        sigma = float(np.sqrt(sigma2))
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    log_weights = -0.5 * (residuals / sigma) ** 2
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return _normalise_weights(weights, enum.N)


def _equality_gaussian_log_unnormalized(
    enum: "HypercubeSampleEnumerator",
    blp: BLP,
    equality_idx: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Return unnormalized log-weights centered on equality satisfaction."""
    if equality_idx < 0 or equality_idx >= blp.p:
        raise IndexError(
            f"equality index out of range: {equality_idx}"
        )
    coeffs = np.asarray(blp.D[equality_idx], dtype=float)
    rhs = float(blp.e[equality_idx])
    residuals = enum.X @ coeffs - rhs
    if sigma is None:
        sigma2 = max(
            0.25 * float(np.dot(coeffs, coeffs)), 1e-12
        )
        sigma = float(np.sqrt(sigma2))
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return -0.5 * (residuals / sigma) ** 2


def _default_equality_importance_law(
    blp: BLP,
    equality_idx: int,
    sigma: float | None = None,
) -> BinnedTargetSaddlepointIS:
    """Return the default equality score law centered on zero residual."""
    proposal_probs = np.full(blp.n, 0.5, dtype=float)
    target = (
        GaussianSlackTarget(center=0.0)
        if sigma is None
        else GaussianSlackTarget(
            center=0.0, scale=float(sigma)
        )
    )
    return BinnedTargetSaddlepointIS.from_problem(
        a=np.asarray(blp.D[equality_idx], dtype=float),
        b=float(blp.e[equality_idx]),
        p=proposal_probs,
        target=target,
        delta=None,
        origin=0.0,
    )


@dataclass(frozen=True)
class ProjectedPenaltyFit:
    """One fitted projected penalty component."""

    values: np.ndarray
    quadratic: np.ndarray
    linear: np.ndarray
    const: float
    theta: dict[frozenset[int], float]
    target_values: np.ndarray
    measure: np.ndarray | None
    fit_enum_size: int


@dataclass(frozen=True)
class ProjectedBLPPenalty:
    """Projected inequality and equality penalties plus their total sum."""

    inequality: ProjectedPenaltyFit | None
    inequalities: tuple[ProjectedPenaltyFit, ...]
    equalities: tuple[ProjectedPenaltyFit, ...]
    total_values: np.ndarray
    quadratic: np.ndarray
    linear: np.ndarray
    const: float


class HypercubeSampleEnumerator:
    """
    Store either the full hypercube or an arbitrary sample of bitstrings.

    Attributes
    ----------
    X : ndarray (N, n)
        Binary states in ``{0, 1}^n``.
    Z : ndarray (N, n)
        Spin states in ``{-1, 1}^n`` via ``z_i = 1 - 2 x_i``.
    N : int
        Number of supplied states.
    """

    def __init__(self, n: int, sample_bits: np.ndarray):
        self.n = n
        self.X = _as_binary_matrix(sample_bits, n)
        self.N = self.X.shape[0]
        self.Z = 1.0 - 2.0 * self.X

    @classmethod
    def full(cls, n: int) -> "HypercubeSampleEnumerator":
        """Enumerate all ``2^n`` bitstrings in ``{0, 1}^n``."""
        indices = np.arange(1 << n)
        bits = (
            (indices[:, None] >> np.arange(n - 1, -1, -1))
            & 1
        ).astype(float)
        return cls(n, bits)

    @classmethod
    def from_spins(
        cls, sample_spins: np.ndarray
    ) -> "HypercubeSampleEnumerator":
        """Build an enumerator from spin samples in ``{-1, 1}^n``."""
        Z = _as_spin_matrix(sample_spins)
        bits = 0.5 * (1.0 - Z)
        return cls(Z.shape[1], bits)

    def chi(self, S) -> np.ndarray:
        """
        Character ``chi_S(z) = prod_{i in S} z_i`` evaluated on all rows.

        Parameters
        ----------
        S : frozenset or iterable of ints

        Returns
        -------
        ndarray of shape (N,)
        """
        S = list(S)
        if len(S) == 0:
            return np.ones(self.N)
        return np.prod(self.Z[:, S], axis=1)

    def design_matrix(self, calE) -> np.ndarray:
        """
        Build the feature matrix whose columns are the admissible characters.

        Parameters
        ----------
        calE : iterable of subsets
            Feature index set, typically ``{emptyset} ∪ {{i}} ∪ E``.

        Returns
        -------
        ndarray (N, |calE|)
        """
        subsets = tuple(calE)
        Phi = np.empty((self.N, len(subsets)), dtype=float)
        for idx, subset in enumerate(subsets):
            indices = tuple(sorted(subset))
            if len(indices) == 0:
                Phi[:, idx] = 1.0
            elif len(indices) == 1:
                Phi[:, idx] = self.Z[:, indices[0]]
            elif len(indices) == 2:
                i, j = indices
                Phi[:, idx] = self.Z[:, i] * self.Z[:, j]
            else:
                Phi[:, idx] = np.prod(
                    self.Z[:, indices],
                    axis=1,
                )
        return Phi


# Backwards-compatible alias for older code that uses the full-hypercube name.
HypercubeEnumerator = HypercubeSampleEnumerator


class FourierAnalysis:
    """
    Walsh-Hadamard Fourier analysis on a supplied set of states.

    When the rows of the enumerator cover the full hypercube and ``mu`` stores
    the target measure over all states, the returned moments are exact. When
    the rows are samples, the same formulas return empirical or weighted
    estimates depending on the choice of ``mu``.
    """

    def __init__(self, enum: HypercubeSampleEnumerator):
        self.enum = enum

    def weights(
        self, mu: np.ndarray | None = None
    ) -> np.ndarray:
        """Return normalized row weights for exact or sample-based moments."""
        return _normalise_weights(mu, self.enum.N)

    def importance_weights(
        self,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ) -> np.ndarray:
        """Return self-normalized importance weights on the stored rows."""
        return _importance_weights(
            sample_bits=self.enum.X,
            num_rows=self.enum.N,
            importance_weights=importance_weights,
            importance_law=importance_law,
        )

    def coefficient(self, P_vals, S, mu=None) -> float:
        """Compute ``E[P(z) chi_S(z)]`` under the supplied row weights."""
        weights = self.weights(mu)
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        return float(
            np.dot(weights, P_vals * self.enum.chi(S))
        )

    def gram_matrix(self, calE, mu=None):
        """
        Compute the Gram matrix of the admissible Walsh features.

        Returns
        -------
        G : ndarray (K, K)
            Exact or empirical Gram matrix.
        chi_cache : dict
            Cached feature values keyed by the corresponding subset.
        """
        Phi = self.enum.design_matrix(calE)
        weights = self.weights(mu)
        G = np.einsum(
            "ni,n,nj->ij", Phi, weights, Phi, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return G, chi_cache

    def gram_matrix_importance(
        self,
        calE,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Compute the sampled Gram matrix under self-normalized IS weights."""
        Phi = self.enum.design_matrix(calE)
        weights = self.importance_weights(
            importance_weights=importance_weights,
            importance_law=importance_law,
        )
        G = np.einsum(
            "ni,n,nj->ij", Phi, weights, Phi, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return G, chi_cache

    def correlation_vector(self, P_vals, calE, mu=None):
        """
        Compute the vector ``c_a = E[P(z) g_a(z)]`` from Eq. 33.

        Returns
        -------
        c : ndarray (K,)
            Exact or empirical correlation vector.
        chi_cache : dict
            Cached feature values keyed by the corresponding subset.
        """
        Phi = self.enum.design_matrix(calE)
        weights = self.weights(mu)
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        c = np.einsum(
            "ni,n,n->i", Phi, weights, P_vals, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return c, chi_cache

    def correlation_vector_importance(
        self,
        P_vals,
        calE,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Compute the sampled correlation vector under IS weights."""
        Phi = self.enum.design_matrix(calE)
        weights = self.importance_weights(
            importance_weights=importance_weights,
            importance_law=importance_law,
        )
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        c = np.einsum(
            "ni,n,n->i", Phi, weights, P_vals, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return c, chi_cache

    def normal_equation(self, P_vals, calE, mu=None):
        """
        Compute the exact/estimated moments in the normal equation ``G theta = c``.

        Returns
        -------
        G : ndarray (K, K)
            Gram matrix of admissible features.
        c : ndarray (K,)
            Feature-penalty correlation vector.
        chi_cache : dict
            Cached feature values keyed by the corresponding subset.
        """
        Phi = self.enum.design_matrix(calE)
        weights = self.weights(mu)
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        G = np.einsum(
            "ni,n,nj->ij", Phi, weights, Phi, optimize=True
        )
        c = np.einsum(
            "ni,n,n->i", Phi, weights, P_vals, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return G, c, chi_cache

    def normal_equation_importance(
        self,
        P_vals,
        calE,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Compute the sampled normal equation under self-normalized IS."""
        Phi = self.enum.design_matrix(calE)
        weights = self.importance_weights(
            importance_weights=importance_weights,
            importance_law=importance_law,
        )
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        G = np.einsum(
            "ni,n,nj->ij", Phi, weights, Phi, optimize=True
        )
        c = np.einsum(
            "ni,n,n->i", Phi, weights, P_vals, optimize=True
        )
        chi_cache = {
            S: Phi[:, idx] for idx, S in enumerate(calE)
        }
        return G, c, chi_cache


class HardwarePenaltyProjection:
    """
    Project an ideal penalty onto the hardware-constrained Walsh subspace

        V_E = span{chi_S : S in calE}

    using either exact moments over the full hypercube or empirical moments
    estimated from sampled bitstrings.

    Parameters
    ----------
    enum : HypercubeSampleEnumerator
        Full enumeration or sample used to estimate the moments.
    topology : HardwareTopology
        Supplies the admissible constant, linear, and quadratic features.
    reg : float, optional
        Tikhonov regularisation used only in the linear solve.
    """

    def __init__(
        self,
        enum: HypercubeSampleEnumerator,
        topology: HardwareTopology,
        reg: float = 1e-8,
        *,
        backend: str = DEFAULT_PROJECTION_BACKEND,
        torch_device: str | None = None,
        torch_dtype: object | None = None,
    ):
        self.enum = enum
        self.topo = topology
        self.fourier = FourierAnalysis(enum)
        self.calE = topology.calE
        self.reg = reg
        self.backend = _resolve_projection_backend(backend)
        self._linear_indices = np.arange(
            self.enum.n, dtype=int
        )
        if self.topo.E:
            self._quadratic_pairs = np.asarray(
                self.topo.E,
                dtype=int,
            ).reshape(-1, 2)
        else:
            self._quadratic_pairs = np.empty(
                (0, 2), dtype=int
            )
        self._const_index = 0
        self._linear_slice = slice(
            1,
            1 + self.enum.n,
        )
        self._quadratic_slice = slice(
            1 + self.enum.n,
            1 + self.enum.n + len(self.topo.E),
        )
        self._Phi = self._build_feature_matrix()
        self._torch_device = (
            _resolve_torch_device(torch_device)
            if self.backend == "torch"
            else None
        )
        self._torch_dtype = (
            _resolve_torch_dtype(torch_dtype)
            if self.backend == "torch"
            else None
        )
        self._Phi_torch = (
            torch.as_tensor(
                self._Phi,
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            if self.backend == "torch"
            else None
        )
        self._cached_weights: np.ndarray | None = None
        self._cached_weighted_phi: object | None = None
        self._cached_gram_matrix: object | None = None
        self._cached_solver: tuple[str, object] | None = (
            None
        )

    def _design_matrix(self) -> np.ndarray:
        """Return the cached admissible Walsh feature matrix."""
        return self._Phi

    def _build_feature_matrix(self) -> np.ndarray:
        """Build the constant/linear/quadratic feature matrix."""
        cols = [
            np.ones((self.enum.N, 1), dtype=float),
            self.enum.Z,
        ]
        if self._quadratic_pairs.size:
            pair_values = (
                self.enum.Z[:, self._quadratic_pairs[:, 0]]
                * self.enum.Z[
                    :, self._quadratic_pairs[:, 1]
                ]
            )
            cols.append(pair_values)
        return np.concatenate(cols, axis=1)

    def _theta_vector_from_mapping(
        self,
        theta: dict[frozenset[int], float],
    ) -> np.ndarray:
        """Return one theta vector in the cached basis order."""
        return np.array(
            [
                float(theta.get(subset, 0.0))
                for subset in self.calE
            ],
            dtype=float,
        )

    def _theta_matrix(
        self,
        theta_values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate one or more coefficient vectors."""
        theta = np.asarray(theta_values, dtype=float)
        num_features = self._Phi.shape[1]
        if theta.ndim == 1:
            if theta.shape[0] != num_features:
                raise ValueError(
                    f"{name} must have length {num_features}"
                )
            return theta.reshape(num_features, 1)
        if (
            theta.ndim != 2
            or theta.shape[0] != num_features
        ):
            raise ValueError(
                f"{name} must have shape ({num_features}, k)"
            )
        return theta

    def _weighted_system(
        self,
        weights: np.ndarray,
    ) -> tuple[object, object, tuple[str, object]]:
        """Return the weighted design matrix, Gram matrix, and solver."""
        if (
            self._cached_weights is not None
            and np.array_equal(
                weights, self._cached_weights
            )
            and self._cached_weighted_phi is not None
            and self._cached_gram_matrix is not None
            and self._cached_solver is not None
        ):
            return (
                self._cached_weighted_phi,
                self._cached_gram_matrix,
                self._cached_solver,
            )

        if self.backend == "torch":
            if torch is None or self._Phi_torch is None:
                raise RuntimeError(
                    "torch backend requested without torch support"
                )
            weights_tensor = torch.as_tensor(
                weights,
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            weighted_phi = (
                self._Phi_torch * weights_tensor[:, None]
            )
            gram_matrix = self._Phi_torch.T @ weighted_phi
            solve_matrix = gram_matrix.clone()
            diag_idx = torch.arange(
                solve_matrix.shape[0],
                device=self._torch_device,
            )
            solve_matrix[diag_idx, diag_idx] += self.reg
            try:
                solver = (
                    "cho",
                    torch.linalg.cholesky(solve_matrix),
                )
            except RuntimeError:
                solver = ("solve", solve_matrix)
        else:
            weighted_phi = (
                self._Phi * weights[:, np.newaxis]
            )
            gram_matrix = self._Phi.T @ weighted_phi
            solve_matrix = np.array(gram_matrix, copy=True)
            solve_matrix.flat[
                :: solve_matrix.shape[0] + 1
            ] += self.reg
            try:
                solver = (
                    "cho",
                    cho_factor(
                        solve_matrix,
                        lower=True,
                        check_finite=False,
                    ),
                )
            except np.linalg.LinAlgError:
                solver = ("solve", solve_matrix)

        self._cached_weights = np.array(weights, copy=True)
        self._cached_weighted_phi = weighted_phi
        self._cached_gram_matrix = gram_matrix
        self._cached_solver = solver
        return weighted_phi, gram_matrix, solver

    def _solve_cached_system(
        self,
        solver: tuple[str, object],
        rhs: np.ndarray,
    ) -> object:
        """Solve one cached regularized Gram system."""
        solver_kind, solver_data = solver
        if self.backend == "torch":
            if torch is None:
                raise RuntimeError(
                    "torch backend requested without torch support"
                )
            rhs_tensor = torch.as_tensor(
                rhs,
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            if solver_kind == "cho":
                return torch.cholesky_solve(
                    rhs_tensor,
                    solver_data,
                    upper=False,
                )
            try:
                return torch.linalg.solve(
                    solver_data, rhs_tensor
                )
            except RuntimeError:
                # Torch can still report singularity for numerically rank-deficient
                # regularized systems, especially at lower precision. Fall back to
                # the minimum-norm least-squares solution instead of aborting.
                return torch.linalg.lstsq(
                    solver_data, rhs_tensor
                ).solution
        if solver_kind == "cho":
            factor, lower = solver_data
            return cho_solve(
                (factor, lower),
                rhs,
                check_finite=False,
            )
        return np.linalg.solve(solver_data, rhs)

    def _solve_projection(
        self,
        penalties: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Solve one weighted normal equation for one or more penalties."""
        penalties = _as_penalty_matrix(
            penalties,
            self.enum.N,
            "penalties",
        )
        weighted_phi, gram_matrix, solver = (
            self._weighted_system(weights)
        )
        if self.backend == "torch":
            penalties_backend = torch.as_tensor(
                penalties,
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            c_matrix = weighted_phi.T @ penalties_backend
        else:
            c_matrix = weighted_phi.T @ penalties
        theta_matrix = self._solve_cached_system(
            solver,
            c_matrix,
        )
        if self.backend == "torch":
            H_matrix = self._Phi_torch @ theta_matrix
        else:
            H_matrix = self._Phi @ theta_matrix
        return (
            np.array(
                _tensor_to_numpy(theta_matrix), copy=True
            ),
            np.array(
                _tensor_to_numpy(gram_matrix), copy=True
            ),
            np.array(_tensor_to_numpy(c_matrix), copy=True),
            np.array(_tensor_to_numpy(H_matrix), copy=True),
        )

    def gram_matrix(self, mu=None) -> np.ndarray:
        """Return the exact or empirical Gram matrix of the admissible features."""
        weights = self.fourier.weights(mu)
        _, gram_matrix, _ = self._weighted_system(weights)
        return np.array(
            _tensor_to_numpy(gram_matrix), copy=True
        )

    def normal_equation(self, P_vals: np.ndarray, mu=None):
        """
        Return the moments in the normal equation from Eq. 33.

        Parameters
        ----------
        P_vals : ndarray (N,)
            Penalty values evaluated on the supplied rows of ``enum``.
        mu : ndarray (N,), optional
            Exact probabilities over the rows, empirical sample weights, or
            importance weights. If omitted, each row receives weight ``1 / N``.

        Returns
        -------
        G : ndarray (K, K)
            Gram matrix of the admissible features.
        c : ndarray (K,)
            Correlation vector between the penalty and the admissible features.
        """
        weights = self.fourier.weights(mu)
        penalties = _as_vector(
            P_vals, self.enum.N, "P_vals"
        )
        weighted_phi, gram_matrix, _ = (
            self._weighted_system(weights)
        )
        if self.backend == "torch":
            penalties_backend = torch.as_tensor(
                penalties.reshape(-1, 1),
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            c_matrix = weighted_phi.T @ penalties_backend
        else:
            c_matrix = weighted_phi.T @ penalties.reshape(
                -1, 1
            )
        c = _tensor_to_numpy(c_matrix)[:, 0]
        return (
            np.array(
                _tensor_to_numpy(gram_matrix), copy=True
            ),
            c,
        )

    def normal_equation_importance(
        self,
        P_vals: np.ndarray,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Return sampled normal-equation moments under self-normalized IS."""
        weights = self.fourier.importance_weights(
            importance_weights=importance_weights,
            importance_law=importance_law,
        )
        penalties = _as_vector(
            P_vals, self.enum.N, "P_vals"
        )
        weighted_phi, gram_matrix, _ = (
            self._weighted_system(weights)
        )
        if self.backend == "torch":
            penalties_backend = torch.as_tensor(
                penalties.reshape(-1, 1),
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
            c_matrix = weighted_phi.T @ penalties_backend
        else:
            c_matrix = weighted_phi.T @ penalties.reshape(
                -1, 1
            )
        return (
            np.array(
                _tensor_to_numpy(gram_matrix), copy=True
            ),
            _tensor_to_numpy(c_matrix)[:, 0],
        )

    def project_many(
        self,
        P_vals_matrix: np.ndarray,
        mu=None,
    ):
        """
        Project multiple penalties that share one topology and measure.

        Parameters
        ----------
        P_vals_matrix : ndarray (N, k)
            One penalty vector per column.
        mu : ndarray (N,), optional
            Exact probabilities or sample weights aligned with the rows.

        Returns
        -------
        theta_matrix : ndarray (|calE|, k)
            Projection coefficients in the cached basis order.
        G : ndarray (|calE|, |calE|)
            Exact or estimated Gram matrix from Eq. 33.
        c_matrix : ndarray (|calE|, k)
            One correlation vector per penalty column.
        H_matrix : ndarray (N, k)
            Projected values on the supplied rows.
        """
        penalties = _as_penalty_matrix(
            P_vals_matrix,
            self.enum.N,
            "P_vals_matrix",
        )
        weights = self.fourier.weights(mu)
        return self._solve_projection(penalties, weights)

    def project_many_importance(
        self,
        P_vals_matrix: np.ndarray,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Project multiple penalties from sampled rows via IS weights."""
        penalties = _as_penalty_matrix(
            P_vals_matrix,
            self.enum.N,
            "P_vals_matrix",
        )
        weights = self.fourier.importance_weights(
            importance_weights=importance_weights,
            importance_law=importance_law,
        )
        return self._solve_projection(penalties, weights)

    def project(self, P_vals: np.ndarray, mu=None):
        """
        Compute the hardware-restricted projection of ``P``.

        The returned ``G`` and ``c`` are the exact or estimated moments from the
        normal equation ``G theta = c``. The solve itself uses
        ``G + reg * I`` for numerical stability when ``reg > 0``.

        Parameters
        ----------
        P_vals : ndarray (N,)
            Penalty values evaluated on the supplied rows of ``enum``.
        mu : ndarray (N,), optional
            Exact probabilities or sample weights aligned with ``P_vals``.

        Returns
        -------
        theta : dict frozenset -> float
            Projection coefficients in Walsh coordinates.
        G : ndarray (K, K)
            Exact or estimated Gram matrix from Eq. 33.
        c : ndarray (K,)
            Exact or estimated correlation vector from Eq. 33.
        H_vals : ndarray (N,)
            Projected values on the supplied rows.
        """
        penalties = _as_vector(
            P_vals, self.enum.N, "P_vals"
        )
        theta_matrix, G, c_matrix, H_matrix = (
            self.project_many(
                penalties,
                mu,
            )
        )
        theta_vec = theta_matrix[:, 0]
        theta = {
            S: float(theta_vec[idx])
            for idx, S in enumerate(self.calE)
        }
        return theta, G, c_matrix[:, 0], H_matrix[:, 0]

    def project_importance(
        self,
        P_vals: np.ndarray,
        *,
        importance_weights: np.ndarray | None = None,
        importance_law: object | None = None,
    ):
        """Project one penalty vector from proposal-sampled rows via IS."""
        penalties = _as_vector(
            P_vals, self.enum.N, "P_vals"
        )
        theta_matrix, G, c_matrix, H_matrix = (
            self.project_many_importance(
                penalties,
                importance_weights=importance_weights,
                importance_law=importance_law,
            )
        )
        theta_vec = theta_matrix[:, 0]
        theta = {
            S: float(theta_vec[idx])
            for idx, S in enumerate(self.calE)
        }
        return theta, G, c_matrix[:, 0], H_matrix[:, 0]

    def mse(
        self,
        P_vals: np.ndarray,
        H_vals: np.ndarray,
        mu=None,
    ) -> float:
        """Compute the exact or empirical weighted mean-squared error."""
        weights = self.fourier.weights(mu)
        P_vals = _as_vector(P_vals, self.enum.N, "P_vals")
        H_vals = _as_vector(H_vals, self.enum.N, "H_vals")
        return float(
            np.dot(weights, (P_vals - H_vals) ** 2)
        )

    def irreducible_error(
        self, P_vals: np.ndarray, mu=None
    ) -> float:
        """
        Return the residual error of the fitted projection under the same weights.

        When ``reg = 0`` this is the exact projection error. For ``reg > 0`` it
        is the residual error of the regularized fit returned by :meth:`project`.
        """
        _, _, _, H_vals = self.project(P_vals, mu)
        return self.mse(P_vals, H_vals, mu)

    def to_qubo(self, theta: dict):
        """
        Convert spin-domain coefficients ``theta`` to a ``{0, 1}^n`` QUBO form.

        Uses the substitution ``z_i = 1 - 2 x_i``:

            ``chi_{i}(z)  = z_i = 1 - 2 x_i``
            ``chi_{ij}(z) = z_i z_j = (1 - 2 x_i)(1 - 2 x_j)``

        Parameters
        ----------
        theta : dict frozenset -> float
            Spin-domain projection coefficients.

        Returns
        -------
        Q : ndarray (n, n)
            Symmetric quadratic couplings.
        h : ndarray (n,)
            Linear coefficients.
        const : float
            Constant offset.
        """
        if isinstance(theta, dict):
            theta_matrix = self._theta_vector_from_mapping(
                theta
            )
        else:
            theta_matrix = np.asarray(theta, dtype=float)
        quadratic, linear, const = self.to_qubo_many(
            theta_matrix,
        )
        return quadratic[0], linear[0], float(const[0])

    def to_qubo_many(
        self,
        theta_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert one or more spin-domain coefficient vectors to QUBOs."""
        theta = self._theta_matrix(
            theta_values, "theta_values"
        )
        num_terms = theta.shape[1]
        n = self.enum.n
        quadratic = np.zeros((num_terms, n, n), dtype=float)
        linear = np.zeros((num_terms, n), dtype=float)
        const = theta[self._const_index, :].copy()

        linear_coeffs = theta[self._linear_slice, :].T
        if linear_coeffs.size:
            const += np.sum(linear_coeffs, axis=1)
            linear -= 2.0 * linear_coeffs

        pair_coeffs = theta[self._quadratic_slice, :].T
        if pair_coeffs.size:
            const += np.sum(pair_coeffs, axis=1)
            for pair_idx, (i, j) in enumerate(
                self._quadratic_pairs
            ):
                coeffs = pair_coeffs[:, pair_idx]
                linear[:, i] -= 2.0 * coeffs
                linear[:, j] -= 2.0 * coeffs
                quadratic[:, i, j] = 4.0 * coeffs
                quadratic[:, j, i] = 4.0 * coeffs

        return quadratic, linear, const

    def build_full_qubo(
        self,
        blp: "BLP",
        constraint_idx: int,
        mu: np.ndarray | None,
        penalty_fn,
        lam: float = 1.0,
    ):
        """
        Build the penalized QUBO for one constraint using the fitted projection.

        Parameters
        ----------
        blp : BLP
            Binary linear program.
        constraint_idx : int
            Constraint index to penalize.
        mu : ndarray (N,), optional
            Exact probabilities or sample weights aligned with ``enum``.
        penalty_fn : callable
            Function returning penalty values on spin states from the original
            binary-domain constraint data. If equality constraints are used,
            the callable should accept an optional ``kind`` keyword.
        lam : float, optional
            Penalty strength.

        Returns
        -------
        Q_full : ndarray (n, n)
            Upper-triangular QUBO matrix.
        const : float
            Constant offset from the projected penalty.
        """
        kind, a, b = blp.constraint_data(constraint_idx)
        try:
            P_vals = penalty_fn(
                self.enum.Z, a, b, kind=kind
            )
        except TypeError:
            if kind != "ineq":
                raise TypeError(
                    "equality-constraint projection requires a penalty_fn that accepts kind='eq'"
                ) from None
            P_vals = penalty_fn(self.enum.Z, a, b)
        theta, _, _, _ = self.project(P_vals, mu)
        Q, h, const = self.to_qubo(theta)

        n = self.enum.n
        Q_full = np.zeros((n, n))
        np.fill_diagonal(Q_full, blp.c + lam * h)
        for i in range(n):
            for j in range(i + 1, n):
                Q_full[i, j] = lam * Q[i, j]

        return Q_full, const


def project_penalty_values(
    fit_enum: HypercubeSampleEnumerator,
    topology: HardwareTopology,
    penalty_values: np.ndarray,
    *,
    mu: np.ndarray | None = None,
    eval_enum: HypercubeSampleEnumerator | None = None,
    reg: float = 1e-8,
    backend: str = DEFAULT_PROJECTION_BACKEND,
    torch_device: str | None = None,
    torch_dtype: object | None = None,
) -> ProjectedPenaltyFit:
    """
    Project one penalty vector onto the supplied topology.

    The normal equation is fit on ``fit_enum`` with optional row weights
    ``mu``. The returned projected values are evaluated on ``eval_enum`` when
    provided, otherwise on ``fit_enum`` itself.
    """
    eval_enum = fit_enum if eval_enum is None else eval_enum
    if fit_enum.n != eval_enum.n:
        raise ValueError(
            "fit_enum and eval_enum must have the same dimension"
        )

    projection = HardwarePenaltyProjection(
        fit_enum,
        topology,
        reg=reg,
        backend=backend,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )
    theta, _, _, fit_values = projection.project(
        penalty_values, mu
    )
    quadratic, linear, const = projection.to_qubo(theta)
    values = (
        fit_values
        if eval_enum is fit_enum
        else _evaluate_theta_on_enum(
            eval_enum, projection.calE, theta
        )
    )
    return ProjectedPenaltyFit(
        values=np.asarray(values, dtype=float),
        quadratic=np.asarray(quadratic, dtype=float),
        linear=np.asarray(linear, dtype=float),
        const=float(const),
        theta=dict(theta),
        target_values=np.asarray(
            penalty_values, dtype=float
        ).reshape(-1),
        measure=(
            None
            if mu is None
            else np.asarray(mu, dtype=float).reshape(-1)
        ),
        fit_enum_size=fit_enum.N,
    )


def project_penalty_values_importance(
    sample_bits: np.ndarray,
    topology: HardwareTopology,
    penalty_values: np.ndarray,
    *,
    importance_weights: np.ndarray | None = None,
    importance_law: object | None = None,
    eval_enum: HypercubeSampleEnumerator | None = None,
    reg: float = 1e-8,
    backend: str = DEFAULT_PROJECTION_BACKEND,
    torch_device: str | None = None,
    torch_dtype: object | None = None,
) -> ProjectedPenaltyFit:
    """Project one penalty vector from sampled states via IS weights."""
    penalty_values = np.asarray(
        penalty_values, dtype=float
    ).reshape(-1)
    if penalty_values.ndim != 1:
        raise ValueError(
            "penalty_values must be a 1D array"
        )
    fit_enum = HypercubeSampleEnumerator(
        topology.n,
        np.asarray(sample_bits, dtype=float),
    )
    if penalty_values.shape[0] != fit_enum.N:
        raise ValueError(
            "penalty_values must align with sample_bits"
        )
    eval_enum = fit_enum if eval_enum is None else eval_enum
    if fit_enum.n != eval_enum.n:
        raise ValueError(
            "fit_enum and eval_enum must have the same dimension"
        )

    projection = HardwarePenaltyProjection(
        fit_enum,
        topology,
        reg=reg,
        backend=backend,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )
    theta, _, _, fit_values = projection.project_importance(
        penalty_values,
        importance_weights=importance_weights,
        importance_law=importance_law,
    )
    quadratic, linear, const = projection.to_qubo(theta)
    values = (
        fit_values
        if eval_enum is fit_enum
        else _evaluate_theta_on_enum(
            eval_enum, projection.calE, theta
        )
    )
    weights = projection.fourier.importance_weights(
        importance_weights=importance_weights,
        importance_law=importance_law,
    )
    return ProjectedPenaltyFit(
        values=np.asarray(values, dtype=float),
        quadratic=np.asarray(quadratic, dtype=float),
        linear=np.asarray(linear, dtype=float),
        const=float(const),
        theta=dict(theta),
        target_values=penalty_values,
        measure=weights,
        fit_enum_size=fit_enum.N,
    )


def project_blp_penalty(
    blp: BLP,
    topology: HardwareTopology,
    *,
    fit_enum: HypercubeSampleEnumerator | None = None,
    inequality_fit_enum: (
        HypercubeSampleEnumerator | None
    ) = None,
    inequality_template: (
        str | Callable[..., np.ndarray]
    ) = "hinge",
    inequality_template_kwargs: dict | None = None,
    inequality_measure: np.ndarray | None = None,
    inequality_measures: (
        Sequence[np.ndarray | None] | None
    ) = None,
    inequality_weights: float | Sequence[float] = 1.0,
    equality_fit_enums: (
        Sequence[HypercubeSampleEnumerator] | None
    ) = None,
    equality_measures: (
        Sequence[np.ndarray | None] | None
    ) = None,
    equality_weights: float | Sequence[float] = 1.0,
    equality_sigmas: float | Sequence[float] | None = None,
    eval_enum: HypercubeSampleEnumerator | None = None,
    reg: float = 1e-8,
    backend: str = DEFAULT_PROJECTION_BACKEND,
    torch_device: str | None = None,
    torch_dtype: object | None = None,
) -> ProjectedBLPPenalty:
    """
    Project a mixed-constraint BLP penalty onto one hardware topology.

    ``fit_enum`` is the common projection sample/enumerator. When supplied it
    is used as the default fit enumerator for both inequalities and
    equalities, while the more specific ``inequality_fit_enum`` and
    ``equality_fit_enums`` override it when needed.

    Inequalities are projected separately, one per constraint. Each equality
    is also projected separately using the
    fixed ideal square penalty ``(d_j^T x - e_j)^2`` together with a Gaussian
    measure centered at satisfaction unless explicit equality measures are
    provided.
    """
    if blp.m == 0 and blp.p == 0:
        if eval_enum is None:
            raise ValueError(
                "eval_enum must be provided when the BLP has no constraints"
            )
        zeros = np.zeros(eval_enum.N, dtype=float)
        return ProjectedBLPPenalty(
            inequality=None,
            inequalities=(),
            equalities=(),
            total_values=zeros,
            quadratic=np.zeros((blp.n, blp.n), dtype=float),
            linear=np.zeros(blp.n, dtype=float),
            const=0.0,
        )

    inequality_template_kwargs = (
        {}
        if inequality_template_kwargs is None
        else dict(inequality_template_kwargs)
    )
    if fit_enum is not None and fit_enum.n != blp.n:
        raise ValueError(
            "fit_enum dimension must match blp.n"
        )
    if inequality_fit_enum is None:
        inequality_fit_enum = fit_enum

    default_enum = (
        inequality_fit_enum
        if inequality_fit_enum is not None
        else fit_enum
    )
    if default_enum is None:
        default_enum = eval_enum
    if default_enum is None:
        raise ValueError(
            "at least one of fit_enum, inequality_fit_enum, or eval_enum must be provided"
        )
    if default_enum.n != blp.n:
        raise ValueError(
            "enumerator dimension must match blp.n"
        )
    eval_enum = (
        default_enum if eval_enum is None else eval_enum
    )
    if eval_enum.n != blp.n:
        raise ValueError(
            "eval_enum dimension must match blp.n"
        )

    total_values = np.zeros(eval_enum.N, dtype=float)
    total_quadratic = np.zeros((blp.n, blp.n), dtype=float)
    total_linear = np.zeros(blp.n, dtype=float)
    total_const = 0.0

    inequality_fit_weight_seq = _as_weight_sequence(
        inequality_weights,
        blp.m,
        "inequality_weights",
    )
    inequality_measure_seq = _as_optional_measure_sequence(
        inequality_measures,
        blp.m,
        "inequality_measures",
    )
    if (
        inequality_measures is None
        and inequality_measure is not None
    ):
        inequality_measure_seq = _broadcast_optional_vector(
            inequality_measure,
            blp.m,
        )

    inequality_fits: list[ProjectedPenaltyFit] = []
    if blp.m:
        if inequality_fit_enum is None:
            raise ValueError(
                "inequality_fit_enum must be provided when the BLP has inequalities"
            )
        if inequality_fit_enum.n != blp.n:
            raise ValueError(
                "inequality_fit_enum dimension must match blp.n"
            )
        for inequality_idx in range(blp.m):
            ineq_values = IdealPenalty.for_constraint(
                inequality_fit_enum.Z,
                blp,
                inequality_idx,
                template=inequality_template,
                weight=inequality_fit_weight_seq[
                    inequality_idx
                ],
                **inequality_template_kwargs,
            )
            fit = project_penalty_values(
                inequality_fit_enum,
                topology,
                ineq_values,
                mu=inequality_measure_seq[inequality_idx],
                eval_enum=eval_enum,
                reg=reg,
                backend=backend,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
            )
            inequality_fits.append(fit)
            total_values += fit.values
            total_quadratic += fit.quadratic
            total_linear += fit.linear
            total_const += fit.const

    equality_enum_seq = _as_enumerator_sequence(
        equality_fit_enums,
        default_enum,
        blp.p,
        "equality_fit_enums",
    )
    equality_measure_seq = _as_optional_measure_sequence(
        equality_measures,
        blp.p,
        "equality_measures",
    )
    equality_weight_seq = _as_weight_sequence(
        equality_weights,
        blp.p,
        "equality_weights",
    )
    if equality_sigmas is None:
        equality_sigma_seq = tuple(
            None for _ in range(blp.p)
        )
    else:
        equality_sigma_seq = _as_weight_sequence(
            equality_sigmas, blp.p, "equality_sigmas"
        )

    equality_fits: list[ProjectedPenaltyFit] = []
    for equality_idx in range(blp.p):
        fit_enum = equality_enum_seq[equality_idx]
        if fit_enum.n != blp.n:
            raise ValueError(
                "equality fit enumerator dimension must match blp.n"
            )
        eq_values = IdealPenalty.for_equality_constraint(
            fit_enum.Z,
            blp,
            equality_idx,
            weight=equality_weight_seq[equality_idx],
        )
        eq_measure = equality_measure_seq[equality_idx]
        if eq_measure is None:
            eq_measure = _equality_gaussian_measure(
                fit_enum,
                blp,
                equality_idx,
                sigma=equality_sigma_seq[equality_idx],
            )
        fit = project_penalty_values(
            fit_enum,
            topology,
            eq_values,
            mu=eq_measure,
            eval_enum=eval_enum,
            reg=reg,
            backend=backend,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )
        equality_fits.append(fit)
        total_values += fit.values
        total_quadratic += fit.quadratic
        total_linear += fit.linear
        total_const += fit.const

    return ProjectedBLPPenalty(
        inequality=_sum_projected_fits(inequality_fits),
        inequalities=tuple(inequality_fits),
        equalities=tuple(equality_fits),
        total_values=total_values,
        quadratic=total_quadratic,
        linear=total_linear,
        const=float(total_const),
    )


def project_blp_penalty_importance(
    blp: BLP,
    topology: HardwareTopology,
    *,
    fit_enum: HypercubeSampleEnumerator | None = None,
    inequality_fit_enum: (
        HypercubeSampleEnumerator | None
    ) = None,
    inequality_template: (
        str | Callable[..., np.ndarray]
    ) = "hinge",
    inequality_template_kwargs: dict | None = None,
    inequality_importance_weights: (
        Sequence[np.ndarray | None] | None
    ) = None,
    inequality_importance_laws: (
        Sequence[object | None] | None
    ) = None,
    inequality_weights: float | Sequence[float] = 1.0,
    equality_fit_enums: (
        Sequence[HypercubeSampleEnumerator] | None
    ) = None,
    equality_importance_weights: (
        Sequence[np.ndarray | None] | None
    ) = None,
    equality_importance_laws: (
        Sequence[object | None] | None
    ) = None,
    equality_weights: float | Sequence[float] = 1.0,
    equality_sigmas: float | Sequence[float] | None = None,
    eval_enum: HypercubeSampleEnumerator | None = None,
    reg: float = 1e-8,
    backend: str = DEFAULT_PROJECTION_BACKEND,
    torch_device: str | None = None,
    torch_dtype: object | None = None,
) -> ProjectedBLPPenalty:
    """
    Project a mixed-constraint BLP penalty via proposal-based importance sampling.

    ``fit_enum`` is the common sampled enumerator. As in :func:`project_blp_penalty`,
    inequalities and equalities are projected one by one and then summed.
    Importance inputs are supplied either as direct row-weight vectors or as
    score-law objects exposing ``importance_weights(X=...)``.
    """
    if blp.m == 0 and blp.p == 0:
        if eval_enum is None:
            raise ValueError(
                "eval_enum must be provided when the BLP has no constraints"
            )
        zeros = np.zeros(eval_enum.N, dtype=float)
        return ProjectedBLPPenalty(
            inequality=None,
            inequalities=(),
            equalities=(),
            total_values=zeros,
            quadratic=np.zeros((blp.n, blp.n), dtype=float),
            linear=np.zeros(blp.n, dtype=float),
            const=0.0,
        )

    inequality_template_kwargs = (
        {}
        if inequality_template_kwargs is None
        else dict(inequality_template_kwargs)
    )
    if fit_enum is not None and fit_enum.n != blp.n:
        raise ValueError(
            "fit_enum dimension must match blp.n"
        )
    if inequality_fit_enum is None:
        inequality_fit_enum = fit_enum

    default_enum = (
        inequality_fit_enum
        if inequality_fit_enum is not None
        else fit_enum
    )
    if default_enum is None:
        default_enum = eval_enum
    if default_enum is None:
        raise ValueError(
            "at least one of fit_enum, inequality_fit_enum, or eval_enum must be provided"
        )
    if default_enum.n != blp.n:
        raise ValueError(
            "enumerator dimension must match blp.n"
        )
    eval_enum = (
        default_enum if eval_enum is None else eval_enum
    )
    if eval_enum.n != blp.n:
        raise ValueError(
            "eval_enum dimension must match blp.n"
        )

    total_values = np.zeros(eval_enum.N, dtype=float)
    total_quadratic = np.zeros((blp.n, blp.n), dtype=float)
    total_linear = np.zeros(blp.n, dtype=float)
    total_const = 0.0

    inequality_fit_weight_seq = _as_weight_sequence(
        inequality_weights,
        blp.m,
        "inequality_weights",
    )
    inequality_importance_weight_seq = (
        _as_optional_vector_sequence(
            inequality_importance_weights,
            blp.m,
            "inequality_importance_weights",
        )
    )
    inequality_importance_law_seq = (
        _as_optional_object_sequence(
            inequality_importance_laws,
            blp.m,
            "inequality_importance_laws",
        )
    )

    inequality_fits: list[ProjectedPenaltyFit] = []
    if blp.m:
        if inequality_fit_enum is None:
            raise ValueError(
                "inequality_fit_enum must be provided when the BLP has inequalities"
            )
        if inequality_fit_enum.n != blp.n:
            raise ValueError(
                "inequality_fit_enum dimension must match blp.n"
            )
        for inequality_idx in range(blp.m):
            ineq_values = IdealPenalty.for_constraint(
                inequality_fit_enum.Z,
                blp,
                inequality_idx,
                template=inequality_template,
                weight=inequality_fit_weight_seq[
                    inequality_idx
                ],
                **inequality_template_kwargs,
            )
            ineq_importance_weights = (
                inequality_importance_weight_seq[
                    inequality_idx
                ]
            )
            ineq_importance_law = (
                inequality_importance_law_seq[
                    inequality_idx
                ]
            )
            if (
                ineq_importance_weights is None
                and ineq_importance_law is None
            ):
                raise ValueError(
                    "inequality_importance_weights or inequality_importance_laws "
                    "must be provided for every inequality"
                )
            fit = project_penalty_values_importance(
                inequality_fit_enum.X,
                topology,
                ineq_values,
                importance_weights=ineq_importance_weights,
                importance_law=ineq_importance_law,
                eval_enum=eval_enum,
                reg=reg,
                backend=backend,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
            )
            inequality_fits.append(fit)
            total_values += fit.values
            total_quadratic += fit.quadratic
            total_linear += fit.linear
            total_const += fit.const

    equality_enum_seq = _as_enumerator_sequence(
        equality_fit_enums,
        default_enum,
        blp.p,
        "equality_fit_enums",
    )
    equality_importance_weight_seq = (
        _as_optional_vector_sequence(
            equality_importance_weights,
            blp.p,
            "equality_importance_weights",
        )
    )
    equality_importance_law_seq = (
        _as_optional_object_sequence(
            equality_importance_laws,
            blp.p,
            "equality_importance_laws",
        )
    )
    equality_weight_seq = _as_weight_sequence(
        equality_weights,
        blp.p,
        "equality_weights",
    )
    if equality_sigmas is None:
        equality_sigma_seq = tuple(
            None for _ in range(blp.p)
        )
    else:
        equality_sigma_seq = _as_weight_sequence(
            equality_sigmas, blp.p, "equality_sigmas"
        )

    equality_fits: list[ProjectedPenaltyFit] = []
    for equality_idx in range(blp.p):
        fit_enum_eq = equality_enum_seq[equality_idx]
        if fit_enum_eq.n != blp.n:
            raise ValueError(
                "equality fit enumerator dimension must match blp.n"
            )
        eq_values = IdealPenalty.for_equality_constraint(
            fit_enum_eq.Z,
            blp,
            equality_idx,
            weight=equality_weight_seq[equality_idx],
        )
        eq_importance_weights = (
            equality_importance_weight_seq[equality_idx]
        )
        eq_importance_law = equality_importance_law_seq[
            equality_idx
        ]
        if (
            eq_importance_weights is None
            and eq_importance_law is None
        ):
            eq_importance_law = (
                _default_equality_importance_law(
                    blp,
                    equality_idx,
                    sigma=equality_sigma_seq[equality_idx],
                )
            )
        fit = project_penalty_values_importance(
            fit_enum_eq.X,
            topology,
            eq_values,
            importance_weights=eq_importance_weights,
            importance_law=eq_importance_law,
            eval_enum=eval_enum,
            reg=reg,
            backend=backend,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )
        equality_fits.append(fit)
        total_values += fit.values
        total_quadratic += fit.quadratic
        total_linear += fit.linear
        total_const += fit.const

    return ProjectedBLPPenalty(
        inequality=_sum_projected_fits(inequality_fits),
        inequalities=tuple(inequality_fits),
        equalities=tuple(equality_fits),
        total_values=total_values,
        quadratic=total_quadratic,
        linear=total_linear,
        const=float(total_const),
    )
