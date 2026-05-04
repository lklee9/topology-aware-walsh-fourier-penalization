"""Core Binary Linear Program data structure."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_constraint_matrix(
    values, num_variables: int, name: str
) -> np.ndarray:
    """Return one constraint matrix with the requested number of columns."""
    if values is None:
        return np.zeros((0, num_variables), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((0, num_variables), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != num_variables:
            raise ValueError(
                f"{name} must have {num_variables} columns"
            )
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[1] != num_variables:
        raise ValueError(
            f"{name} must have {num_variables} columns"
        )
    return arr


def _as_rhs_vector(
    values, expected_length: int, name: str
) -> np.ndarray:
    """Return one RHS vector with the expected length."""
    if values is None:
        return np.zeros(expected_length, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}"
        )
    return arr


class BLP:
    """
    Binary Linear Program:
        min  c^T x
        s.t. A x - b >= 0
             D x - e  = 0
             x in {0,1}^n

    Parameters
    ----------
    c : array_like (n,)
        Objective coefficients.
    A : array_like (m, n), optional
        Inequality-constraint matrix.
    b : array_like (m,), optional
        Inequality-constraint RHS vector.
    D : array_like (p, n), optional
        Equality-constraint matrix.
    e : array_like (p,), optional
        Equality-constraint RHS vector.
    """

    def __init__(
        self,
        c,
        A=None,
        b=None,
        D=None,
        e=None,
        *,
        objective_constant: float = 0.0,
        name: str | None = None,
        variable_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.c = np.asarray(c, dtype=float).reshape(-1)
        self.n = int(self.c.shape[0])

        self.A = _as_constraint_matrix(A, self.n, "A")
        self.b = _as_rhs_vector(b, self.A.shape[0], "b")
        self.D = _as_constraint_matrix(D, self.n, "D")
        self.e = _as_rhs_vector(e, self.D.shape[0], "e")
        self.objective_constant = float(objective_constant)
        self.name = (
            f"BLP_{self.n}" if name is None else str(name)
        )
        if variable_names is None:
            self.variable_names = [
                f"x_{index}" for index in range(self.n)
            ]
        else:
            if len(variable_names) != self.n:
                raise ValueError(
                    "variable_names must have one entry per variable"
                )
            self.variable_names = list(variable_names)
        self.metadata = dict(metadata or {})

        self.m = int(self.A.shape[0])
        self.p = int(self.D.shape[0])

    def _as_state_matrix(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Return ``x`` as a 2D array and whether the input was one state."""
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            if X.shape[0] != self.n:
                raise ValueError(
                    f"x must have length {self.n}"
                )
            return X.reshape(1, -1), True
        if X.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if X.shape[1] != self.n:
            raise ValueError(
                f"x must have {self.n} columns"
            )
        return X, False

    def _restore_row_shape(
        self, values: np.ndarray, squeeze: bool
    ) -> np.ndarray:
        """Undo the row promotion used for 1D state inputs."""
        if squeeze:
            return values.reshape(-1)
        return values

    @property
    def num_inequalities(self) -> int:
        """Return the number of inequality constraints."""
        return self.m

    @property
    def num_equalities(self) -> int:
        """Return the number of equality constraints."""
        return self.p

    @property
    def num_constraints(self) -> int:
        """Return the total number of linear constraints."""
        return self.m + self.p

    @property
    def num_variables(self) -> int:
        """Return the number of binary decision variables."""
        return self.n

    @property
    def objective_linear(self) -> np.ndarray:
        """Backward-compatible alias for the objective coefficients."""
        return self.c

    @property
    def equality_matrix(self) -> np.ndarray:
        """Backward-compatible alias for the equality-constraint matrix."""
        return self.D

    @property
    def equality_rhs(self) -> np.ndarray:
        """Backward-compatible alias for the equality-constraint RHS."""
        return self.e

    @property
    def inequality_matrix(self) -> np.ndarray:
        """Paper-style alias for the inequality matrix in ``lhs <= rhs`` form."""
        return -self.A

    @property
    def inequality_rhs(self) -> np.ndarray:
        """Paper-style alias for the inequality RHS in ``lhs <= rhs`` form."""
        return -self.b

    @property
    def num_states(self) -> int:
        """Return the size of the binary hypercube."""
        return 1 << self.n

    @property
    def constraint_matrix(self) -> np.ndarray:
        """Stack equalities then inequalities for topology heuristics."""
        if self.p and self.m:
            return np.vstack([self.D, self.A])
        if self.p:
            return self.D.copy()
        if self.m:
            return self.A.copy()
        return np.zeros((0, self.n), dtype=float)

    def inequality_violation(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return ``A x - b``; negative entries indicate violated inequalities."""
        X, squeeze = self._as_state_matrix(x)
        if self.m == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        values = X @ self.A.T - self.b
        return self._restore_row_shape(values, squeeze)

    def equality_residual(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return ``D x - e``; feasible states satisfy this exactly."""
        X, squeeze = self._as_state_matrix(x)
        if self.p == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        values = X @ self.D.T - self.e
        return self._restore_row_shape(values, squeeze)

    def equality_residuals(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return equality residuals for one state or a batch of states."""
        return self.equality_residual(x)

    def inequality_penalty_argument(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return the one-sided inequality violation scores ``b - A x``."""
        X, squeeze = self._as_state_matrix(x)
        if self.m == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        values = self.b - X @ self.A.T
        return self._restore_row_shape(values, squeeze)

    def inequality_slacks(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return paper-style slacks where positive values are feasible."""
        return self.inequality_violation(x)

    def equality_distance(
        self, x: np.ndarray
    ) -> np.ndarray:
        """Return the absolute equality residuals ``|D x - e|``."""
        X, squeeze = self._as_state_matrix(x)
        if self.p == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        values = np.abs(X @ self.D.T - self.e)
        return self._restore_row_shape(values, squeeze)

    def feasible_mask(
        self, x: np.ndarray, atol: float = 1e-9
    ) -> np.ndarray | bool:
        """Return a boolean feasibility mask aligned with the supplied states."""
        X, squeeze = self._as_state_matrix(x)
        mask = np.ones(X.shape[0], dtype=bool)
        if self.m:
            mask &= np.all(
                X @ self.A.T - self.b >= -atol, axis=1
            )
        if self.p:
            mask &= np.all(
                np.abs(X @ self.D.T - self.e) <= atol,
                axis=1,
            )
        if squeeze:
            return bool(mask[0])
        return mask

    def violation(self, x: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for the inequality residual ``A x - b``."""
        return self.inequality_violation(x)

    def is_feasible(
        self, x: np.ndarray, atol: float = 1e-9
    ) -> bool:
        """Return ``True`` when all inequality and equality constraints hold."""
        return bool(self.feasible_mask(x, atol=atol))

    def objective(
        self, x: np.ndarray
    ) -> float | np.ndarray:
        """Evaluate the affine objective ``objective_constant + c^T x``."""
        X, squeeze = self._as_state_matrix(x)
        values = self.objective_constant + X @ self.c
        if squeeze:
            return float(values[0])
        return values

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        """Return objective values for a batch of states."""
        values = self.objective(x)
        if isinstance(values, float):
            return np.asarray([values], dtype=float)
        return np.asarray(values, dtype=float)

    def constraint_kind(self, j: int) -> str:
        """Return ``'ineq'`` or ``'eq'`` for constraint index ``j``."""
        if j < 0 or j >= self.num_constraints:
            raise IndexError(
                f"constraint index out of range: {j}"
            )
        return "ineq" if j < self.m else "eq"

    def constraint_data(
        self, j: int
    ) -> tuple[str, np.ndarray, float]:
        """Return ``(kind, coeffs, rhs)`` for one inequality/equality constraint."""
        kind = self.constraint_kind(j)
        if kind == "ineq":
            return kind, self.A[j].copy(), float(self.b[j])
        idx = j - self.m
        return kind, self.D[idx].copy(), float(self.e[idx])

    def spin_constraint(
        self, j: int
    ) -> tuple[np.ndarray, float]:
        """
        Return the spin-space form of constraint ``j``.

        With ``x_i = (1 - z_i) / 2``, both inequality and equality constraints
        of the form ``a^T x`` vs. ``b`` become a spin-domain affine residual

            ``a^T z - (1^T a - 2 b)``.

        For inequalities, positive residual means infeasible. For equalities,
        feasible states are exactly those with zero residual.
        """
        _, coeffs, rhs = self.constraint_data(j)
        return coeffs, float(np.sum(coeffs) - 2.0 * rhs)


__all__ = ["BLP"]
