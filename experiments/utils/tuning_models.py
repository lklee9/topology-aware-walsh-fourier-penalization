"""Shared tuning dataclasses used across experiment drivers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.utils.projected_qubo import QuboTerms


@dataclass(frozen=True)
class GroundStateTuningMetrics:
    """Gap-only tuning metrics for one penalized energy spectrum."""

    gap: float
    best_optimum_percentile: float
    tied_fraction: float


@dataclass(frozen=True)
class TuningObjectiveResult:
    """Comparable tuning score plus summary metrics for one point."""

    sort_key: tuple[float, ...]
    objective_value: float
    record_fields: dict[str, object]


@dataclass(frozen=True)
class TunedProjectedMultipliers:
    """Family-level projected-penalty multipliers selected on one anchor size."""

    method: str
    family: str
    anchor_size: int
    equality_multiplier: float
    inequality_multiplier: float
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str

    def as_row(self) -> dict[str, object]:
        """Serialize the tuned projected multipliers."""
        return {
            "method": self.method,
            "family": self.family,
            "anchor_size": self.anchor_size,
            "equality_multiplier": self.equality_multiplier,
            "inequality_multiplier": self.inequality_multiplier,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class SelectedProjectedConfig:
    """One projected-method combo selected for downstream evaluation."""

    method: str
    family: str
    measure_name: str
    penalty_template: str
    selection_mode: str
    selection_source: str
    candidate_rank: int
    tuning: TunedProjectedMultipliers
    projection_method: str | None = None
    penalty_template_kwargs: dict[str, float] = field(
        default_factory=dict
    )
    projected_standardize: bool | None = None


@dataclass(frozen=True)
class TunedUnbalancedParameters:
    """Family-level unbalanced-penalty parameters selected on one anchor size."""

    family: str
    anchor_size: int
    up_equality_multiplier: float | None
    up_inequality_multiplier: float
    up_lambda1_shape: float
    up_lambda2_shape: float
    up_lambda_gauge: str
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str
    normalization_regime: str | None = None
    per_constraint_standardization: bool | None = None
    global_multiplier: float | None = None
    lambda0: float | None = None
    lambda1: float | None = None
    lambda2: float | None = None
    base_parameter_source: str | None = None

    def as_row(self) -> dict[str, object]:
        """Serialize the tuned unbalanced-penalty parameters."""
        return {
            "family": self.family,
            "anchor_size": self.anchor_size,
            "up_equality_multiplier": self.up_equality_multiplier,
            "up_inequality_multiplier": self.up_inequality_multiplier,
            "up_lambda1_shape": self.up_lambda1_shape,
            "up_lambda2_shape": self.up_lambda2_shape,
            "up_lambda_gauge": self.up_lambda_gauge,
            "normalization_regime": self.normalization_regime,
            "per_constraint_standardization": (
                self.per_constraint_standardization
            ),
            "global_multiplier": self.global_multiplier,
            "lambda0": self.lambda0,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "base_parameter_source": self.base_parameter_source,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class TuningRunOutputs:
    """Family-level tuning artifacts reused by downstream experiments."""

    projected_tuning_rows: list[dict[str, object]]
    projected_selection_rows: list[dict[str, object]]
    up_tuning_rows: list[dict[str, object]]
    tuned_unbalanced_params: dict[
        str, TunedUnbalancedParameters
    ]
    selected_projected_configs: dict[
        tuple[str, str], SelectedProjectedConfig
    ]


@dataclass(frozen=True)
class TuningInstance:
    """Cached exact energy tables for one anchor instance."""

    num_states: int
    optimum_state_indices: np.ndarray
    objective_energies: np.ndarray
    equality_energies: np.ndarray
    inequality_energies: np.ndarray


@dataclass(frozen=True)
class UnbalancedTuningInstance:
    """Cached exact energy tables for UP tuning on one anchor instance."""

    num_states: int
    optimum_state_indices: np.ndarray
    objective_energies: np.ndarray
    objective_linear: np.ndarray
    equality_energies: np.ndarray
    inequality_row_bases: tuple[
        "UnbalancedInequalityRowBasis", ...
    ]


@dataclass(frozen=True)
class UnbalancedInequalityRowBasis:
    """Cached row-level UP building blocks for one inequality."""

    linear_terms: QuboTerms
    quadratic_terms: QuboTerms
    linear_energies: np.ndarray
    quadratic_energies: np.ndarray


__all__ = [
    "GroundStateTuningMetrics",
    "SelectedProjectedConfig",
    "TunedProjectedMultipliers",
    "TunedUnbalancedParameters",
    "UnbalancedInequalityRowBasis",
    "TuningInstance",
    "TuningObjectiveResult",
    "TuningRunOutputs",
    "UnbalancedTuningInstance",
]
