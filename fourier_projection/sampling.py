from __future__ import annotations

"""
Binned importance sampling for score-constrained binary distributions.

Overview
========
This module implements a reusable framework for constructing and using an
approximate target distribution on the score

    S(X) = a^T X - b,

where X is a binary random vector sampled from an easy proposal distribution Q,
chosen here to be an independent Bernoulli product law

    X_i ~ Bernoulli(p_i),  i = 1, ..., n.

The main use case is the following.

1. We want the induced distribution of the scalar score S(X) to have a desired
   shape, such as a Gaussian-shaped law centered near a target value c, or a
   uniform-shaped law over some interval [L, U].
2. Directly constructing a distribution on all binary vectors x in {0,1}^n that
   realizes that score law may be difficult.
3. Instead, we:
   (a) discretize score space into bins,
   (b) approximate the proposal mass in each bin using a saddlepoint method,
   (c) define desired target masses on the same bins,
   (d) reweight proposal samples by the bin-wise ratio

           weight = target_bin_mass / proposal_bin_mass.

The code is organized into four layers.

Structure of the module
=======================
1. ScoreTargetDistribution
   Abstract interface for a target law on score space. A concrete target knows
   how to assign masses to score bins and how to suggest a reasonable default
   bin width delta.

2. GaussianScoreTarget, UniformScoreTarget, CustomScoreTarget
   Concrete target-law implementations.

3. BinnedScoreTargetLaw
   Deterministic construction of the approximate binned target law. This class
   owns:
   - the score bins,
   - the proposal bin masses q_hat,
   - the target bin masses pi,
   - the per-bin importance ratios pi / q_hat,
   - the saddlepoint approximation machinery.

   It does not perform Monte Carlo sampling.

4. BinnedTargetSaddlepointIS
   Thin Monte Carlo wrapper around BinnedScoreTargetLaw. This class draws
   proposal samples and uses the law object to compute weights and estimate
   expectations.

Mathematical summary
====================
Let the score bins be

    B_j = [origin + (j - 1/2) delta, origin + (j + 1/2) delta).

Let q_j = Q(S(X) in B_j) be the proposal bin masses and let pi_j be the desired
bin masses under the target score law. Then the binned target distribution on
binary vectors is defined by

    P(x) = Q(x) * pi_{j(x)} / q_{j(x)},

where j(x) is the bin index such that S(x) in B_{j(x)}.

This construction guarantees that, at the bin level,

    P(S(X) in B_j) = pi_j.

In practice q_j is usually not available in closed form for arbitrary real
coefficients a. We therefore approximate q_j by a saddlepoint approximation:

    q_j ≈ F_hat(u_j) - F_hat(l_j),

where l_j and u_j are the lower and upper edges of bin B_j, and F_hat is the
Lugannani-Rice approximation to the proposal CDF of S.

How to use this module
======================
Typical workflow:

1. Choose a target law, for example

       target = GaussianScoreTarget(center=0.5, scale=0.8)

   or

       target = UniformScoreTarget(low=-1.0, high=1.0)

2. Construct an estimator directly from the problem specification:

       estimator = BinnedTargetSaddlepointIS.from_problem(
           a=a,
           b=b,
           p=p,
           target=target,
           delta=None,
           origin=0.0,
       )

   If delta is omitted, the target object suggests a default value.

3. Estimate an expectation under the approximate target law:

       result = estimator.estimate_expectation(g, n_samples=20000, rng=123)

4. Inspect the approximate score law on bins via the underlying deterministic
   law object:

       law = estimator.law
       law.centers     # bin centers
       law.q_hat       # approximate proposal score pmf on bins
       law.pi          # target score pmf on bins
       law.weight_by_bin

You can also reuse an already constructed law with external samples through

    estimator.estimate_from_samples(...)

which avoids rebuilding the same binned target law multiple times.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

try:
    from .measures import (
        CustomScoreTarget,
        CustomSlackTarget,
        GaussianScoreTarget,
        GaussianSlackTarget,
        ScoreTargetDistribution,
        SlackTargetDistribution,
        UniformScoreTarget,
        UniformSlackTarget,
    )
except (
    ImportError
):  # pragma: no cover - allows running the file as a script.
    from measures import (
        CustomScoreTarget,
        CustomSlackTarget,
        GaussianScoreTarget,
        GaussianSlackTarget,
        ScoreTargetDistribution,
        SlackTargetDistribution,
        UniformScoreTarget,
        UniformSlackTarget,
    )


ArrayLike = Union[np.ndarray, list, tuple]


@dataclass
class BinnedScoreTargetLaw:
    """
    Reusable approximate binned target law for

        S(X) = a^T X - b,

    under an independent Bernoulli proposal.

    This class performs all deterministic construction steps needed for the
    approximate binned target law:

    1. Build a bin system covering the support of S.
    2. Approximate the proposal mass in each bin by a saddlepoint method.
    3. Compute the desired target mass in each bin.
    4. Form approximate importance ratios pi_j / q_hat_j.

    It does not sample from the proposal or perform Monte Carlo estimation.
    Those tasks are handled by BinnedTargetSaddlepointIS.

    Parameters
    ----------
    a : array-like of shape (n,)
        Real coefficient vector defining the score.
    b : float
        Scalar offset in the score definition.
    p : array-like of shape (n,)
        Bernoulli proposal probabilities.
    target : SlackTargetDistribution
        Target distribution object.
    delta : float, optional
        Bin width. If omitted, target.suggest_delta(a, p) is used.
    origin : float, default=0.0
        Origin of the score binning system.
    q_floor : float, default=1e-15
        Positive floor applied to approximate proposal bin masses before
        renormalization.
    mean_tol : float, default=1e-8
        Tolerance used in the Lugannani-Rice approximation near the proposal
        mean.
    """

    a: ArrayLike
    b: float
    p: ArrayLike
    target: SlackTargetDistribution
    delta: Optional[float] = None
    origin: float = 0.0
    q_floor: float = 1e-15
    mean_tol: float = 1e-8

    j: np.ndarray = field(init=False)
    centers: np.ndarray = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)
    q_hat: np.ndarray = field(init=False)
    pi: np.ndarray = field(init=False)
    weight_by_bin: np.ndarray = field(init=False)
    s_min: float = field(init=False)
    s_max: float = field(init=False)
    logit_p: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Validate inputs and build the full approximate binned target law.

        The support bounds of the score are

            s_min = -b + sum_{i : a_i < 0} a_i,
            s_max = -b + sum_{i : a_i > 0} a_i,

        because each binary variable x_i can only take the values 0 or 1.
        These bounds are then used to create a finite bin system covering the
        entire support of S.
        """
        self.a = np.asarray(self.a, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        self.b = float(self.b)
        self.origin = float(self.origin)
        self.q_floor = float(self.q_floor)
        self.mean_tol = float(self.mean_tol)

        if self.a.ndim != 1 or self.p.ndim != 1:
            raise ValueError(
                "a and p must be one-dimensional arrays."
            )
        if self.a.shape != self.p.shape:
            raise ValueError(
                "a and p must have the same shape."
            )
        if np.any(self.p <= 0.0) or np.any(self.p >= 1.0):
            raise ValueError(
                "Each p_i must satisfy 0 < p_i < 1."
            )
        if self.q_floor <= 0.0:
            raise ValueError("q_floor must be positive.")
        if not isinstance(
            self.target, SlackTargetDistribution
        ):
            raise TypeError(
                "target must be an instance of SlackTargetDistribution."
            )

        if self.delta is None:
            self.delta = float(
                self.target.suggest_delta(
                    a=self.a, p=self.p
                )
            )
        else:
            self.delta = float(self.delta)

        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")

        # Cached logit(p_i) values are used to evaluate derivatives of the CGF
        # in a numerically stable logistic form.
        self.logit_p = np.log(self.p) - np.log1p(-self.p)

        # Support of S(x) over x in {0,1}^n. Positive coefficients are included
        # at their maximum by setting x_i = 1, while negative coefficients are
        # included at their minimum by setting x_i = 1.
        self.s_min = float(
            np.sum(self.a[self.a < 0.0]) - self.b
        )
        self.s_max = float(
            np.sum(self.a[self.a > 0.0]) - self.b
        )

        if self.s_min > self.s_max:
            raise ValueError("Inconsistent support bounds.")

        self._build_bins()
        self.q_hat = self._approximate_bin_masses()
        self.pi = self.target.bin_masses(
            self.lower,
            self.upper,
            self.centers,
            a=self.a,
            b=self.b,
            p=self.p,
        )
        self.weight_by_bin = self.pi / self.q_hat

    # ------------------------------------------------------------------
    # Bin construction
    # ------------------------------------------------------------------

    def _build_bins(self) -> None:
        """
        Construct a finite set of score bins covering [s_min, s_max].

        The bins are indexed by integers j and have the form

            B_j = [origin + (j - 1/2) delta,
                   origin + (j + 1/2) delta).

        The implementation chooses the smallest and largest j values needed to
        ensure that the full support interval [s_min, s_max] is covered.
        """
        j_min = int(
            np.floor(
                (self.s_min - self.origin) / self.delta
                - 0.5
            )
        )
        j_max = int(
            np.ceil(
                (self.s_max - self.origin) / self.delta
                + 0.5
            )
        )
        self.j = np.arange(j_min, j_max + 1, dtype=int)

        self.centers = self.origin + self.j * self.delta
        self.lower = (
            self.origin + (self.j - 0.5) * self.delta
        )
        self.upper = (
            self.origin + (self.j + 0.5) * self.delta
        )

        if self.j.size == 0:
            raise ValueError("No bins were constructed.")

        self._j0 = int(self.j[0])

    # ------------------------------------------------------------------
    # Score and proposal-CGF utilities
    # ------------------------------------------------------------------

    def slack_from_X(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the slack

            S = a^T X - b

        for a batch of binary vectors X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n)
            Binary vectors.

        Returns
        -------
        ndarray of shape (n_samples,)
            Slack values corresponding to the rows of X.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.a.size:
            raise ValueError(
                "X must have shape (n_samples, n_features)."
            )
        return X @ self.a - self.b

    def score_from_X(self, X: np.ndarray) -> np.ndarray:
        """Compatibility alias for :meth:`slack_from_X`."""
        return self.slack_from_X(X)

    def K(self, t: float) -> float:
        """
        Cumulant generating function of the proposal score.

        Mathematical background
        -----------------------
        Under the Bernoulli product proposal,

            S(X) = sum_i a_i X_i - b,

        and therefore

            E_Q[e^{t S(X)}]
            = e^{-bt} prod_i (1 - p_i + p_i e^{a_i t}).

        Hence the cumulant generating function is

            K(t) = log E_Q[e^{t S(X)}]
                 = -bt + sum_i log(1 - p_i + p_i e^{a_i t}).

        This function is the starting point for the saddlepoint approximation.
        """
        t = float(t)
        term = np.logaddexp(
            np.log1p(-self.p), np.log(self.p) + self.a * t
        )
        return float(-self.b * t + np.sum(term))

    def K1(self, t: float) -> float:
        """
        First derivative of the cumulant generating function.

        Mathematical background
        -----------------------
        Differentiating K gives

            K'(t)
            = -b + sum_i a_i p_i e^{a_i t} / (1 - p_i + p_i e^{a_i t}).

        This quantity is strictly increasing in t and is used to solve the
        saddlepoint equation

            K'(t_hat) = s.

        The implementation evaluates the ratio in logistic form for numerical
        stability.
        """
        t = float(t)
        r = expit(self.logit_p + self.a * t)
        return float(-self.b + np.sum(self.a * r))

    def K2(self, t: float) -> float:
        """
        Second derivative of the cumulant generating function.

        Mathematical background
        -----------------------
        The second derivative is

            K''(t)
            = sum_i a_i^2 p_i (1 - p_i) e^{a_i t} / (1 - p_i + p_i e^{a_i t})^2.

        It is nonnegative and strictly positive whenever the score is not
        degenerate. In the saddlepoint approximation, K''(t_hat) determines the
        local curvature of K at the saddlepoint.
        """
        t = float(t)
        r = expit(self.logit_p + self.a * t)
        return float(np.sum((self.a**2) * r * (1.0 - r)))

    def K3(self, t: float) -> float:
        """
        Third derivative of the cumulant generating function.

        This quantity is not essential for the main Lugannani-Rice formula used
        below, but it is often useful for higher-order diagnostics and related
        local approximations.
        """
        t = float(t)
        r = expit(self.logit_p + self.a * t)
        return float(
            np.sum(
                (self.a**3)
                * r
                * (1.0 - r)
                * (1.0 - 2.0 * r)
            )
        )

    # ------------------------------------------------------------------
    # Saddlepoint machinery
    # ------------------------------------------------------------------

    def _solve_saddlepoint(
        self, s: float, max_expand: int = 100
    ) -> float:
        """
        Solve the saddlepoint equation

            K'(t_hat) = s

        using bracketing and Brent's method.

        Mathematical background
        -----------------------
        Since K''(t) >= 0, the function K'(t) is monotone increasing. Therefore,
        for any score value s in the interior of the support of S, there is a
        unique saddlepoint t_hat solving K'(t_hat) = s.

        The algorithm expands a symmetric bracket until the root is enclosed and
        then calls scipy.optimize.brentq.
        """
        s = float(s)

        def f(t: float) -> float:
            return self.K1(t) - s

        lo, hi = -1.0, 1.0
        flo, fhi = f(lo), f(hi)

        expand_count = 0
        while (
            flo > 0.0 or fhi < 0.0
        ) and expand_count < max_expand:
            lo *= 2.0
            hi *= 2.0
            flo, fhi = f(lo), f(hi)
            expand_count += 1

        if flo > 0.0 or fhi < 0.0:
            raise RuntimeError(
                "Failed to bracket the saddlepoint root. "
                "This usually indicates that s is numerically too close "
                "to the support boundary."
            )

        return float(brentq(f, lo, hi))

    def cdf_saddlepoint(self, s: float) -> float:
        """
        Approximate the proposal CDF F_S(s) = P_Q(S <= s) by the
        Lugannani-Rice saddlepoint formula.

        Mathematical background
        -----------------------
        Let t_hat solve

            K'(t_hat) = s.

        Define

            w(s) = sign(t_hat) * sqrt(2 (t_hat s - K(t_hat))),
            u(s) = t_hat * sqrt(K''(t_hat)).

        Then the Lugannani-Rice approximation is

            F_hat(s) = Phi(w) + phi(w) (1/w - 1/u),

        where Phi and phi are the standard normal CDF and PDF.

        This approximation is typically accurate even in moderately non-Gaussian
        settings and is especially useful for weighted Bernoulli sums such as the
        present score S(X).

        Numerical notes
        ---------------
        Near the mean of the distribution, the quantities w and u can become very
        small and cause cancellation. In that regime this method falls back to a
        simple Gaussian approximation based on the proposal mean and variance.
        """
        s = float(s)

        if s <= self.s_min:
            return 0.0
        if s >= self.s_max:
            return 1.0

        mu = self.K1(0.0)
        var = self.K2(0.0)

        if var <= 0.0:
            return 1.0 if s >= mu else 0.0

        if abs(s - mu) <= self.mean_tol * max(
            1.0, abs(mu), np.sqrt(var)
        ):
            z = (s - mu) / np.sqrt(var)
            return float(norm.cdf(z))

        t_hat = self._solve_saddlepoint(s)
        K_hat = self.K(t_hat)
        K2_hat = self.K2(t_hat)

        if K2_hat <= 0.0:
            z = (s - mu) / np.sqrt(var)
            return float(norm.cdf(z))

        arg = 2.0 * (t_hat * s - K_hat)
        arg = max(arg, 0.0)

        w = np.sign(t_hat) * np.sqrt(arg)
        u = t_hat * np.sqrt(K2_hat)

        if abs(w) < 1e-12 or abs(u) < 1e-12:
            z = (s - mu) / np.sqrt(var)
            return float(norm.cdf(z))

        val = norm.cdf(w) + norm.pdf(w) * (
            1.0 / w - 1.0 / u
        )
        return float(np.clip(val, 0.0, 1.0))

    def _approximate_bin_masses(self) -> np.ndarray:
        """
        Approximate the proposal bin masses q_j.

        Mathematical background
        -----------------------
        If B_j = [lower_j, upper_j), then

            q_j = P_Q(S in B_j)
                = F_S(upper_j) - F_S(lower_j).

        We approximate this by replacing the exact proposal CDF F_S with the
        Lugannani-Rice approximation F_hat:

            q_hat_j = F_hat(upper_j) - F_hat(lower_j).

        After computing these approximate masses, a small positive floor is
        applied before renormalization to prevent numerical underflow from
        producing exactly zero proposal masses, which would make the importance
        ratios unstable.
        """
        q = np.empty_like(self.centers, dtype=float)
        for k, (lo, hi) in enumerate(
            zip(self.lower, self.upper)
        ):
            q[k] = self.cdf_saddlepoint(
                hi
            ) - self.cdf_saddlepoint(lo)
            # q[k] = slack_interval_prob_p_half(self.a, self.b, lo, hi)

        q = np.maximum(q, self.q_floor)
        q /= q.sum()
        return q

    # ------------------------------------------------------------------
    # Bin lookup and approximate weights
    # ------------------------------------------------------------------

    def bin_index(
        self, s: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Map score value(s) to internal bin indices.

        Given a score s, the corresponding bin label j is computed by rounding
        to the nearest bin center according to the chosen origin and bin width.
        The returned array indexes into the stored arrays self.centers,
        self.lower, self.upper, self.q_hat, and self.pi.
        """
        s_arr = np.asarray(s, dtype=float)
        j_vals = np.floor(
            (s_arr - self.origin) / self.delta + 0.5
        ).astype(int)
        idx = j_vals - self._j0
        idx = np.clip(idx, 0, self.j.size - 1)
        return idx

    def weights_from_slacks(
        self, s: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Return approximate importance weights for slack values.

        Mathematical background
        -----------------------
        At the bin level, the importance ratio is

            weight_j = pi_j / q_hat_j.

        Therefore any sample whose slack falls in bin B_j receives that bin's
        ratio as its approximate importance weight.
        """
        idx = self.bin_index(s)
        return self.weight_by_bin[idx]

    def weights_from_scores(
        self, s: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compatibility alias for :meth:`weights_from_slacks`."""
        return self.weights_from_slacks(s)

    def importance_weights_from_X(
        self, X: np.ndarray
    ) -> np.ndarray:
        """
        Return approximate importance weights for a batch of binary samples X.

        This is a convenience wrapper that first computes the slacks

            S = a^T X - b,

        and then applies weights_from_slacks.
        """
        S = self.slack_from_X(X)
        return self.weights_from_slacks(S)

    def importance_weights(
        self,
        X: Optional[np.ndarray] = None,
        S: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Return approximate importance weights from either binary samples X or
        precomputed score values S.

        Exactly one of X or S must be supplied.
        """
        if (X is None) == (S is None):
            raise ValueError(
                "Provide exactly one of X or S."
            )

        if X is not None:
            return self.importance_weights_from_X(X)

        return self.weights_from_slacks(S)

    def summary(self) -> Dict[str, Any]:
        """
        Return basic diagnostic information about the constructed law.
        """
        return {
            "support_min": self.s_min,
            "support_max": self.s_max,
            "num_bins": int(self.j.size),
            "delta": self.delta,
            "origin": self.origin,
            "target": self.target.summary(
                a=self.a, p=self.p
            ),
        }


@dataclass
class BinnedTargetSaddlepointIS:
    """
    Monte Carlo wrapper for a BinnedScoreTargetLaw.

    This class adds proposal sampling and expectation estimation on top of the
    deterministic binned target law. It is intentionally thin: all score-space
    logic is delegated to the law object.

    Parameters
    ----------
    law : BinnedScoreTargetLaw
        Reusable approximate binned target law.
    """

    law: BinnedScoreTargetLaw

    @classmethod
    def from_problem(
        cls,
        a: ArrayLike,
        b: float,
        p: ArrayLike,
        target: SlackTargetDistribution,
        delta: Optional[float] = None,
        origin: float = 0.0,
        q_floor: float = 1e-15,
        mean_tol: float = 1e-8,
    ) -> "BinnedTargetSaddlepointIS":
        """
        Construct the estimator directly from the problem specification.

        This is a convenience constructor that builds the deterministic law
        object internally and wraps it in a Monte Carlo estimator.

        It allows the user to work in a one-line style without manually creating
        BinnedScoreTargetLaw first.
        """
        law = BinnedScoreTargetLaw(
            a=a,
            b=b,
            p=p,
            target=target,
            delta=delta,
            origin=origin,
            q_floor=q_floor,
            mean_tol=mean_tol,
        )
        return cls(law=law)

    @property
    def a(self) -> np.ndarray:
        """
        Expose the score coefficient vector from the underlying law.
        """
        return self.law.a

    @property
    def b(self) -> float:
        """
        Expose the score offset from the underlying law.
        """
        return self.law.b

    @property
    def p(self) -> np.ndarray:
        """
        Expose the Bernoulli proposal probabilities from the underlying law.
        """
        return self.law.p

    @property
    def target(self) -> SlackTargetDistribution:
        """
        Expose the target distribution object from the underlying law.
        """
        return self.law.target

    @property
    def delta(self) -> float:
        """
        Expose the active bin width from the underlying law.
        """
        return float(self.law.delta)

    @property
    def origin(self) -> float:
        """
        Expose the score-bin origin from the underlying law.
        """
        return float(self.law.origin)

    def sample_proposal(
        self,
        n_samples: int,
        rng: Optional[
            Union[int, np.random.Generator]
        ] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample binary vectors from the Bernoulli proposal Q.

        Returns both the raw binary samples X and their associated score values

            S = a^T X - b.

        This is the main entry point when the user wants to estimate expectations
        using newly generated proposal samples.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if isinstance(rng, np.random.Generator):
            generator = rng
        else:
            generator = np.random.default_rng(rng)

        X = generator.binomial(
            1, self.p, size=(n_samples, self.a.size)
        ).astype(np.int8)
        S = self.law.slack_from_X(X)
        return X, S

    def estimate_from_samples(
        self,
        g: Union[
            Callable[[np.ndarray], np.ndarray], np.ndarray
        ],
        *,
        X: Optional[np.ndarray] = None,
        S: Optional[Union[float, np.ndarray]] = None,
        self_normalized: bool = True,
        return_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate an expectation under the approximate target law from an existing
        batch of proposal samples.

        Mathematical background
        -----------------------
        If w_m are the approximate importance weights and g_m are the values of
        the integrand on the supplied samples, then the self-normalized
        importance-sampling estimator is

            estimate = sum_m w_m g_m / sum_m w_m.

        This is the default because the proposal bin masses q_hat are only
        approximate. If self_normalized=False, the method instead returns the raw
        importance-weighted average

            mean_m(w_m g_m).

        Inputs
        ------
        Exactly one of X or S must be provided.

        - If g is callable, X must be supplied because g is evaluated on the
          binary vectors.
        - If g is already a vector of values and S is known, it is sufficient to
          supply S.
        """
        if (X is None) == (S is None):
            raise ValueError(
                "Provide exactly one of X or S."
            )
        if callable(g) and X is None:
            raise ValueError(
                "When g is callable, X must be provided."
            )

        w = self.law.importance_weights(X=X, S=S)

        if callable(g):
            g_vals = np.asarray(g(X), dtype=float)
        else:
            g_vals = np.asarray(g, dtype=float)

        if g_vals.ndim != 1:
            raise ValueError(
                "g must return or provide a one-dimensional array."
            )
        if g_vals.shape[0] != w.shape[0]:
            raise ValueError(
                "The number of g-values must match the number of weights."
            )

        if self_normalized:
            estimate = np.sum(w * g_vals) / np.sum(w)
        else:
            estimate = np.mean(w * g_vals)

        # Standard effective sample size diagnostic used in importance sampling.
        ess = (np.sum(w) ** 2) / np.sum(w**2)

        result: Dict[str, Any] = {
            "estimate": float(estimate),
            "weights": w,
            "g_values": g_vals,
            "ess": float(ess),
        }

        if return_details:
            result.update(
                {
                    "q_hat": self.law.q_hat.copy(),
                    "pi": self.law.pi.copy(),
                    "bin_centers": self.law.centers.copy(),
                    "bin_lower": self.law.lower.copy(),
                    "bin_upper": self.law.upper.copy(),
                }
            )
            if X is not None:
                result["X"] = X
            if S is not None:
                result["S"] = np.asarray(S, dtype=float)
        return result


# Preferred slack-oriented aliases.
BinnedSlackTargetLaw = BinnedScoreTargetLaw
BinnedSlackTargetIS = BinnedTargetSaddlepointIS
