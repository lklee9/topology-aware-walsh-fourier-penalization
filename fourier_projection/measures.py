"""Slack-target measures used by hardware-aware Fourier projection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from scipy.stats import norm

try:
    from .blp import BLP
except (
    ImportError
):  # pragma: no cover - allows running the file as a script.
    from blp import BLP

if TYPE_CHECKING:  # pragma: no cover
    from .projection import HypercubeSampleEnumerator


ArrayLike = Union[np.ndarray, list, tuple]
_MIN_INTERVAL_WIDTH = 1e-6


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Return a probability vector proportional to the supplied weights."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "weights must have positive total mass"
        )
    return weights / total


def _positive_scale(
    values: np.ndarray, default: float = 1.0
) -> float:
    """Return a robust positive scale from the supplied values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    positive = values[np.abs(values) > 1e-9]
    if positive.size == 0:
        return float(default)
    return float(np.median(np.abs(positive)))


class SlackTargetDistribution(ABC):
    """
    Abstract base class for a target distribution on one slack value

        s(x) = a^T x - b.

    The target distribution lives on the scalar slack axis rather than on the
    full binary state space.
    """

    name: str = "abstract"

    @staticmethod
    def default_sigma(
        a: ArrayLike, p: Optional[ArrayLike] = None
    ) -> float:
        """Return the default proposal-scale proxy for one slack value."""
        a = np.asarray(a, dtype=float)
        if a.ndim != 1:
            raise ValueError("a must be one-dimensional.")

        if p is None:
            return float(0.5 * np.linalg.norm(a))

        p = np.asarray(p, dtype=float)
        if p.shape != a.shape:
            raise ValueError(
                "p must have the same shape as a."
            )
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError(
                "Each proposal probability must satisfy 0 <= p_i <= 1."
            )
        return float(
            np.sqrt(np.sum((a**2) * p * (1.0 - p)))
        )

    @abstractmethod
    def bin_masses(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        centers: np.ndarray,
        *,
        a: ArrayLike,
        b: float,
        p: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return normalized target masses for one slack-axis binning."""
        raise NotImplementedError

    def suggest_delta(
        self, a: ArrayLike, p: Optional[ArrayLike] = None
    ) -> float:
        """Suggest one default slack-bin width."""
        sigma_q = self.default_sigma(a=a, p=p)
        if sigma_q <= 0.0:
            raise ValueError(
                "The implied proposal scale is non-positive."
            )
        return float(sigma_q / 8.0)

    def summary(
        self,
        a: Optional[ArrayLike] = None,
        p: Optional[ArrayLike] = None,
    ) -> dict[str, Any]:
        """Return a small diagnostic summary."""
        out: dict[str, Any] = {"target_type": self.name}
        if a is not None:
            out["default_sigma"] = self.default_sigma(
                a=a, p=p
            )
            out["suggested_delta"] = self.suggest_delta(
                a=a, p=p
            )
        return out


@dataclass
class GaussianSlackTarget(SlackTargetDistribution):
    """Gaussian-shaped target on the slack axis."""

    center: float
    scale: Optional[float] = None
    name: str = field(init=False, default="gaussian")

    def resolved_scale(
        self, a: ArrayLike, p: Optional[ArrayLike] = None
    ) -> float:
        """Return the active Gaussian scale."""
        if self.scale is not None:
            if self.scale <= 0.0:
                raise ValueError(
                    "Gaussian target scale must be positive."
                )
            return float(self.scale)
        sigma = self.default_sigma(a=a, p=p)
        if sigma <= 0.0:
            raise ValueError(
                "Resolved Gaussian target scale must be positive."
            )
        return float(sigma)

    def bin_masses(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        centers: np.ndarray,
        *,
        a: ArrayLike,
        b: float,
        p: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return normalized Gaussian masses on the supplied slack bins."""
        del centers, b
        scale = self.resolved_scale(a=a, p=p)
        center = float(self.center)
        masses = norm.cdf(
            (upper - center) / scale
        ) - norm.cdf((lower - center) / scale)
        masses = np.maximum(masses, 0.0)
        total = masses.sum()
        if total <= 0.0:
            raise ValueError(
                "Gaussian target has zero mass on the supplied bins."
            )
        return masses / total

    def suggest_delta(
        self, a: ArrayLike, p: Optional[ArrayLike] = None
    ) -> float:
        """Suggest a slack-bin width for the Gaussian target."""
        sigma_q = self.default_sigma(a=a, p=p)
        scale = self.resolved_scale(a=a, p=p)
        return float(min(sigma_q / 8.0, scale / 8.0))

    def summary(
        self,
        a: Optional[ArrayLike] = None,
        p: Optional[ArrayLike] = None,
    ) -> dict[str, Any]:
        """Return a diagnostic summary of the Gaussian target."""
        out = super().summary(a=a, p=p)
        out["center"] = float(self.center)
        out["user_scale"] = (
            None
            if self.scale is None
            else float(self.scale)
        )
        if a is not None:
            out["resolved_scale"] = self.resolved_scale(
                a=a, p=p
            )
        return out


@dataclass
class UniformSlackTarget(SlackTargetDistribution):
    """Uniform target on one slack interval."""

    low: float
    high: float
    name: str = field(init=False, default="uniform")

    def __post_init__(self) -> None:
        self.low = float(self.low)
        self.high = float(self.high)
        if not self.low < self.high:
            raise ValueError(
                "Uniform target requires low < high."
            )

    @property
    def width(self) -> float:
        """Return the interval width."""
        return float(self.high - self.low)

    def bin_masses(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        centers: np.ndarray,
        *,
        a: ArrayLike,
        b: float,
        p: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return normalized uniform masses on the supplied slack bins."""
        del centers, a, b, p
        overlap = np.maximum(
            0.0,
            np.minimum(upper, self.high)
            - np.maximum(lower, self.low),
        )
        total = overlap.sum()
        if total <= 0.0:
            raise ValueError(
                "Uniform target interval has zero overlap with the supplied bins."
            )
        return overlap / total

    def suggest_delta(
        self, a: ArrayLike, p: Optional[ArrayLike] = None
    ) -> float:
        """Suggest a slack-bin width for the uniform target."""
        sigma_q = self.default_sigma(a=a, p=p)
        return float(min(sigma_q / 8.0, self.width / 30.0))

    def summary(
        self,
        a: Optional[ArrayLike] = None,
        p: Optional[ArrayLike] = None,
    ) -> dict[str, Any]:
        """Return a diagnostic summary of the uniform target."""
        out = super().summary(a=a, p=p)
        out["low"] = self.low
        out["high"] = self.high
        out["width"] = self.width
        return out


@dataclass
class CustomSlackTarget(SlackTargetDistribution):
    """Custom target defined directly on the slack bins."""

    mass_function: Any
    name: str = "custom"

    def bin_masses(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        centers: np.ndarray,
        *,
        a: ArrayLike,
        b: float,
        p: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Return normalized masses from the user-supplied bin function."""
        del a, b, p
        masses = np.asarray(
            self.mass_function(
                lower.copy(), upper.copy(), centers.copy()
            ),
            dtype=float,
        )
        if masses.shape != centers.shape:
            raise ValueError(
                "mass_function must return one value per bin."
            )
        if np.any(masses < 0.0):
            raise ValueError(
                "mass_function must return nonnegative masses."
            )
        total = masses.sum()
        if total <= 0.0:
            raise ValueError(
                "Custom target has zero mass on the supplied bins."
            )
        return masses / total


class ProjectionMeasures:
    """User-facing factory for the slack targets used in the experiments."""

    def __init__(
        self,
        enum: "HypercubeSampleEnumerator | None",
        blp: BLP,
        proposal_probs: ArrayLike | None = None,
    ):
        self.enum = enum
        self.blp = blp
        if proposal_probs is None:
            self.proposal_probs = np.full(
                blp.n, 0.5, dtype=float
            )
        else:
            probs = np.asarray(
                proposal_probs, dtype=float
            ).reshape(-1)
            if probs.shape[0] != blp.n:
                raise ValueError(
                    "proposal_probs must have one entry per variable"
                )
            self.proposal_probs = probs

    def uniform_hypercube(self) -> np.ndarray:
        """Return the uniform measure over the stored hypercube/sample rows."""
        if self.enum is None:
            raise ValueError(
                "enum is required for the hypercube-uniform measure"
            )
        return np.full(
            self.enum.N,
            1.0 / float(self.enum.N),
            dtype=float,
        )

    def _check_inequality_idx(
        self, inequality_idx: int
    ) -> None:
        if (
            inequality_idx < 0
            or inequality_idx >= self.blp.m
        ):
            raise IndexError(
                f"inequality index out of range: {inequality_idx}"
            )

    def _check_equality_idx(
        self, equality_idx: int
    ) -> None:
        if equality_idx < 0 or equality_idx >= self.blp.p:
            raise IndexError(
                f"equality index out of range: {equality_idx}"
            )

    def _inequality_max_slack(
        self, inequality_idx: int
    ) -> float:
        self._check_inequality_idx(inequality_idx)
        coeffs = np.asarray(
            self.blp.A[inequality_idx], dtype=float
        )
        rhs = float(self.blp.b[inequality_idx])
        return max(
            float(np.sum(np.maximum(coeffs, 0.0)) - rhs),
            0.0,
        )

    def _inequality_min_slack(
        self, inequality_idx: int
    ) -> float:
        self._check_inequality_idx(inequality_idx)
        coeffs = np.asarray(
            self.blp.A[inequality_idx], dtype=float
        )
        rhs = float(self.blp.b[inequality_idx])
        return min(
            float(np.sum(np.minimum(coeffs, 0.0)) - rhs),
            0.0,
        )

    def _inequality_sigma(
        self, inequality_idx: int
    ) -> float:
        self._check_inequality_idx(inequality_idx)
        coeffs = np.asarray(
            self.blp.A[inequality_idx], dtype=float
        )
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        variance = 0.25 * float(np.sum(coeffs**2))
        sigma2 = min(variance, (max_slack / 4.0) ** 2)
        return float(np.sqrt(max(sigma2, 1e-12)))

    def _equality_sigma(self, equality_idx: int) -> float:
        self._check_equality_idx(equality_idx)
        coeffs = np.asarray(
            self.blp.D[equality_idx], dtype=float
        )
        return float(
            np.sqrt(
                max(
                    0.25 * float(np.dot(coeffs, coeffs)),
                    1e-12,
                )
            )
        )

    def uniform_slack(
        self, inequality_idx: int
    ) -> UniformSlackTarget:
        """Return the uniform-over-slack target for one inequality."""
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        min_slack = self._inequality_min_slack(
            inequality_idx
        )
        sigma = self._inequality_sigma(inequality_idx)
        width = max(
            float(max_slack),
            float(sigma),
            _MIN_INTERVAL_WIDTH,
        )
        return UniformSlackTarget(low=min_slack, high=width)

    def q0(
        self, inequality_idx: int
    ) -> GaussianSlackTarget:
        """Return the canonical boundary-centered slack Gaussian."""
        return GaussianSlackTarget(
            center=0.0,
            scale=self._inequality_sigma(inequality_idx),
        )

    def q1(
        self, inequality_idx: int
    ) -> GaussianSlackTarget:
        """Return the first-quartile slack Gaussian."""
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        return GaussianSlackTarget(
            center=0.25 * max_slack,
            scale=self._inequality_sigma(inequality_idx),
        )

    def q2(
        self, inequality_idx: int
    ) -> GaussianSlackTarget:
        """Return the mid-slack Gaussian."""
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        return GaussianSlackTarget(
            center=0.5 * max_slack,
            scale=self._inequality_sigma(inequality_idx),
        )

    def q3(
        self, inequality_idx: int
    ) -> GaussianSlackTarget:
        """Return the third-quartile slack Gaussian."""
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        return GaussianSlackTarget(
            center=0.75 * max_slack,
            scale=self._inequality_sigma(inequality_idx),
        )

    def q4(
        self, inequality_idx: int
    ) -> GaussianSlackTarget:
        """Return the end-slack Gaussian."""
        max_slack = self._inequality_max_slack(
            inequality_idx
        )
        return GaussianSlackTarget(
            center=1.0 * max_slack,
            scale=self._inequality_sigma(inequality_idx),
        )

    def equality_zero_centered(
        self, equality_idx: int
    ) -> GaussianSlackTarget:
        """Return the equality slack Gaussian centered at zero residual."""
        return GaussianSlackTarget(
            center=0.0,
            scale=self._equality_sigma(equality_idx),
        )


# Compatibility aliases while the rest of the repo finishes the naming shift.
ScoreTargetDistribution = SlackTargetDistribution
GaussianScoreTarget = GaussianSlackTarget
UniformScoreTarget = UniformSlackTarget
CustomScoreTarget = CustomSlackTarget


__all__ = [
    "ArrayLike",
    "CustomScoreTarget",
    "CustomSlackTarget",
    "GaussianScoreTarget",
    "GaussianSlackTarget",
    "ProjectionMeasures",
    "ScoreTargetDistribution",
    "SlackTargetDistribution",
    "UniformScoreTarget",
    "UniformSlackTarget",
]
