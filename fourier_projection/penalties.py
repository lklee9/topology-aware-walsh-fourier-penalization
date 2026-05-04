"""
penalties.py
============
Ideal penalty templates for Fourier-projected BLP penalties.

The functions in this module are vectorized over rows of spin states
``Z in {-1, 1}^{N x n}`` and are designed to plug directly into the
projection code in ``projection.py``. The paper-level templates from
Eq. (penalties) in ``brainstorm/main.tex`` are included, together with a
few extra smooth variants that are useful during early fitting.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

import numpy as np

try:
    from .blp import BLP
except (
    ImportError
):  # pragma: no cover - allows running the file as a script.
    from blp import BLP


def _as_state_matrix(Z: np.ndarray) -> np.ndarray:
    """Validate a 2D spin-state array."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(
            "Z must be a 2D array of spin states"
        )
    return Z


def _as_vector(
    values, length: int, name: str
) -> np.ndarray:
    """Validate a 1D vector of the requested length."""
    vec = np.asarray(values, dtype=float).reshape(-1)
    if vec.shape[0] != length:
        raise ValueError(
            f"{name} must have length {length}"
        )
    return vec


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _safe_exp(values: np.ndarray) -> np.ndarray:
    """Exponentiate with clipping to avoid floating-point overflow."""
    return np.exp(np.clip(values, -700.0, 700.0))


class IdealPenalty:
    """
    Vectorized ideal penalty templates for inequality constraints.

    Throughout this module the binary-domain violation score is

        ``v(x) = b - a^T x``,

    where ``v(x) > 0`` means that the underlying binary state
    violates the constraint. Piecewise templates such as step and ReLU
    vanish on feasible states. Smooth templates such as
    sigmoid, negative exponential, and softplus stay strictly positive but
    still emphasize the infeasible side.
    """

    @staticmethod
    def violation(Z, a, b) -> np.ndarray:
        """
        Return the binary-domain violation score ``v(x(z)) = b - a^T x(z)``.

        The incoming states are still given in spin form; we convert them to the
        binary convention used everywhere else in the BLP code:

            ``x = (1 - z) / 2``.
        """
        Z = _as_state_matrix(Z)
        a = _as_vector(a, Z.shape[1], "a")
        X = 0.5 * (1.0 - Z)
        return float(b) - X @ a

    @staticmethod
    def equality_residual(Z, a, b) -> np.ndarray:
        """Return the signed equality residual ``a^T x(z) - b``."""
        Z = _as_state_matrix(Z)
        a = _as_vector(a, Z.shape[1], "a")
        X = 0.5 * (1.0 - Z)
        return X @ a - float(b)

    @staticmethod
    def equality_squared(
        Z,
        a,
        b,
        weight: float = 1.0,
        kind: str | None = None,
    ) -> np.ndarray:
        """Return the weighted quadratic equality penalty ``weight * (a^T x - b)^2``."""
        if weight < 0.0:
            raise ValueError("weight must be nonnegative")
        residual = IdealPenalty.equality_residual(Z, a, b)
        return float(weight) * residual**2

    @staticmethod
    def heaviside(
        Z, a, b, kind: str = "ineq"
    ) -> np.ndarray:
        """Flat indicator: ``1[v(x(z)) > 0]``."""
        if kind != "ineq":
            raise ValueError(
                "heaviside is defined only for inequality constraints"
            )
        return (
            IdealPenalty.violation(Z, a, b) > 0.0
        ).astype(float)

    @staticmethod
    def step(Z, a, b, kind: str = "ineq") -> np.ndarray:
        """Alias for the paper's step penalty."""
        return IdealPenalty.heaviside(Z, a, b, kind=kind)

    @staticmethod
    def heaviside_inv(
        Z, a, b, kind: str = "ineq"
    ) -> np.ndarray:
        """Indicator of the feasible side: ``1[v(x(z)) < 0]``."""
        if kind != "ineq":
            raise ValueError(
                "heaviside_inv is defined only for inequality constraints"
            )
        return (
            IdealPenalty.violation(Z, a, b) < 0.0
        ).astype(float)

    @staticmethod
    def sigmoid(
        Z, a, b, beta: float = 1.0, kind: str = "ineq"
    ) -> np.ndarray:
        """
        Smooth sigmoid template

            ``psi_sig(t; beta) = 1 / (1 + exp(-beta t))``.
        """
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if kind != "ineq":
            raise ValueError(
                "sigmoid is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        return _sigmoid(beta * violation)

    @staticmethod
    def relu(Z, a, b, kind: str = "ineq") -> np.ndarray:
        """ReLU template: ``max(0, v(x(z)))``."""
        if kind != "ineq":
            raise ValueError(
                "relu is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        return np.maximum(0.0, violation)

    @staticmethod
    def negative_exponential(
        Z, a, b, kind: str = "ineq"
    ) -> np.ndarray:
        """
        Negative-exponential template

            ``psi_exp(t) = exp(-t)`` with ``t = a^T x - b``.

        Since ``v(x) = b - a^T x = -t``, this is exactly ``exp(v(x))``.
        """
        if kind != "ineq":
            raise ValueError(
                "negative_exponential is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        return _safe_exp(violation)

    @staticmethod
    def quadratic(
        Z,
        a,
        b,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        kind: str = "ineq",
        **kwargs,
    ) -> np.ndarray:
        """Return ``lambda1 * v + lambda2 * v^2`` on the violation score."""
        if "coef_linear" in kwargs:
            lambda1 = kwargs.pop("coef_linear")
        if "coef_quadratic" in kwargs:
            lambda2 = kwargs.pop("coef_quadratic")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"unexpected quadratic penalty keyword(s): {unknown}"
            )
        if kind != "ineq":
            raise ValueError(
                "quadratic is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        return (
            float(lambda1) * violation
            + float(lambda2) * violation**2
        )

    @staticmethod
    def hinge(
        Z, a, b, delta: float = 0.0, kind: str = "ineq"
    ) -> np.ndarray:
        """
        Shifted hinge template

            ``psi_hinge(t; delta) = max(0, t + delta)``.
        """
        if kind != "ineq":
            raise ValueError(
                "hinge is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        return np.maximum(0.0, violation + delta)

    @staticmethod
    def squared_hinge(
        Z, a, b, delta: float = 0.0, kind: str = "ineq"
    ) -> np.ndarray:
        """Quadratic hinge penalty: ``max(0, v(x(z)) + delta)^2``."""
        hinge_vals = IdealPenalty.hinge(
            Z, a, b, delta=delta, kind=kind
        )
        return hinge_vals**2

    @staticmethod
    def softplus(
        Z, a, b, beta: float = 1.0, kind: str = "ineq"
    ) -> np.ndarray:
        """
        Smooth ReLU approximation

            ``softplus_beta(t) = log(1 + exp(beta t)) / beta``.
        """
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if kind != "ineq":
            raise ValueError(
                "softplus is defined only for inequality constraints"
            )
        violation = IdealPenalty.violation(Z, a, b)
        scaled = beta * violation
        return (
            np.maximum(scaled, 0.0)
            + np.log1p(np.exp(-np.abs(scaled)))
        ) / beta

    @staticmethod
    def template(name: str) -> Callable[..., np.ndarray]:
        """Return a penalty template by name."""
        lookup = {
            "heaviside": IdealPenalty.heaviside,
            "step": IdealPenalty.step,
            "heaviside_inv": IdealPenalty.heaviside_inv,
            "sigmoid": IdealPenalty.sigmoid,
            "relu": IdealPenalty.relu,
            "negative_exponential": IdealPenalty.negative_exponential,
            "negexp": IdealPenalty.negative_exponential,
            "quadratic": IdealPenalty.quadratic,
            "hinge": IdealPenalty.hinge,
            "squared_hinge": IdealPenalty.squared_hinge,
            "softplus": IdealPenalty.softplus,
        }
        try:
            return lookup[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown penalty template: {name}"
            ) from exc

    @staticmethod
    def constraint(
        Z: np.ndarray,
        a,
        b,
        template: str | Callable[..., np.ndarray] = "hinge",
        weight: float = 1.0,
        kind: str = "ineq",
        **template_kwargs,
    ) -> np.ndarray:
        """
        Evaluate one weighted constraint penalty on the supplied spin states.
        """
        if weight < 0.0:
            raise ValueError("weight must be nonnegative")
        penalty_fn = (
            IdealPenalty.template(template)
            if isinstance(template, str)
            else template
        )
        if isinstance(template, str):
            return float(weight) * penalty_fn(
                Z, a, b, kind=kind, **template_kwargs
            )

        signature = inspect.signature(penalty_fn)
        accepts_kind = "kind" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kind:
            return float(weight) * penalty_fn(
                Z, a, b, kind=kind, **template_kwargs
            )
        return float(weight) * penalty_fn(
            Z, a, b, **template_kwargs
        )

    @staticmethod
    def for_constraint(
        Z: np.ndarray,
        blp: "BLP",
        constraint_idx: int,
        template: str | Callable[..., np.ndarray] = "hinge",
        weight: float = 1.0,
        **template_kwargs,
    ) -> np.ndarray:
        """
        Evaluate one BLP inequality penalty using the original binary data.
        """
        if constraint_idx < 0 or constraint_idx >= blp.m:
            raise IndexError(
                f"inequality index out of range: {constraint_idx}"
            )
        a = blp.A[constraint_idx]
        b = blp.b[constraint_idx]
        return IdealPenalty.constraint(
            Z,
            a,
            b,
            template=template,
            weight=weight,
            kind="ineq",
            **template_kwargs,
        )

    @staticmethod
    def for_equality_constraint(
        Z: np.ndarray,
        blp: "BLP",
        equality_idx: int,
        weight: float = 1.0,
    ) -> np.ndarray:
        """Evaluate one quadratic equality penalty using the original binary data."""
        if equality_idx < 0 or equality_idx >= blp.p:
            raise IndexError(
                f"equality index out of range: {equality_idx}"
            )
        return IdealPenalty.equality_squared(
            Z,
            blp.D[equality_idx],
            blp.e[equality_idx],
            weight=weight,
        )

    @staticmethod
    def total(
        Z: np.ndarray,
        blp: "BLP",
        template: (
            str
            | Callable[..., np.ndarray]
            | Sequence[str | Callable[..., np.ndarray]]
        ) = "hinge",
        weights: float | Sequence[float] = 1.0,
        template_kwargs: (
            dict | Sequence[dict] | None
        ) = None,
    ) -> np.ndarray:
        """
        Build the total ideal penalty

            ``P(z) = sum_r lambda_r psi_r(v_r(x(z)))``

        across every BLP inequality.

        Parameters
        ----------
        Z : ndarray (N, n)
            Spin states on which the total penalty is evaluated.
        blp : BLP
            Binary linear program.
        template : str, callable, or sequence
            A single template shared by all constraints, or one template per
            constraint.
        weights : float or sequence
            Penalty weights ``lambda_r``.
        template_kwargs : dict or sequence of dict, optional
            Shared template arguments or one dict per constraint.
        """
        Z = _as_state_matrix(Z)
        m = blp.m

        if isinstance(
            template, Sequence
        ) and not isinstance(template, (str, bytes)):
            templates = list(template)
        else:
            templates = [template] * m
        if len(templates) != m:
            raise ValueError(
                "template must provide one entry per constraint"
            )

        if np.isscalar(weights):
            weight_vec = [float(weights)] * m
        else:
            weight_vec = list(
                np.asarray(weights, dtype=float).reshape(-1)
            )
        if len(weight_vec) != m:
            raise ValueError(
                "weights must provide one entry per constraint"
            )

        if template_kwargs is None:
            kwargs_list = [{} for _ in range(m)]
        elif isinstance(
            template_kwargs, Sequence
        ) and not isinstance(template_kwargs, dict):
            kwargs_list = [
                dict(kwargs) for kwargs in template_kwargs
            ]
        else:
            kwargs_list = [
                dict(template_kwargs) for _ in range(m)
            ]
        if len(kwargs_list) != m:
            raise ValueError(
                "template_kwargs must provide one dict per constraint"
            )

        total_penalty = np.zeros(Z.shape[0], dtype=float)
        for idx in range(m):
            total_penalty += IdealPenalty.for_constraint(
                Z,
                blp,
                idx,
                template=templates[idx],
                weight=weight_vec[idx],
                **kwargs_list[idx],
            )
        return total_penalty

    @staticmethod
    def total_equalities(
        Z: np.ndarray,
        blp: "BLP",
        weights: float | Sequence[float] = 1.0,
    ) -> np.ndarray:
        """Return the total quadratic penalty across all BLP equalities."""
        Z = _as_state_matrix(Z)
        p = blp.p
        if np.isscalar(weights):
            weight_vec = [float(weights)] * p
        else:
            weight_vec = list(
                np.asarray(weights, dtype=float).reshape(-1)
            )
        if len(weight_vec) != p:
            raise ValueError(
                "weights must provide one entry per equality"
            )

        total_penalty = np.zeros(Z.shape[0], dtype=float)
        for idx in range(p):
            total_penalty += (
                IdealPenalty.for_equality_constraint(
                    Z,
                    blp,
                    idx,
                    weight=weight_vec[idx],
                )
            )
        return total_penalty
