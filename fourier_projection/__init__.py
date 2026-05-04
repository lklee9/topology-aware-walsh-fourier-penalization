"""Public package API for Fourier-projected BLP penalties."""

from __future__ import annotations

from .blp import BLP
from .measures import (
    CustomSlackTarget,
    GaussianSlackTarget,
    ProjectionMeasures,
    SlackTargetDistribution,
    UniformSlackTarget,
)
from .penalties import IdealPenalty
from .projection import (
    DEFAULT_PROJECTION_BACKEND,
    FourierAnalysis,
    HardwarePenaltyProjection,
    HypercubeEnumerator,
    HypercubeSampleEnumerator,
    ProjectedBLPPenalty,
    ProjectedPenaltyFit,
    project_blp_penalty,
    project_blp_penalty_importance,
    project_penalty_values,
    project_penalty_values_importance,
)
from .sampling import BinnedTargetSaddlepointIS
from .topology import HardwareTopology

__version__ = "0.1.0"

__all__ = [
    "BLP",
    "BinnedTargetSaddlepointIS",
    "CustomSlackTarget",
    "DEFAULT_PROJECTION_BACKEND",
    "FourierAnalysis",
    "GaussianSlackTarget",
    "HardwarePenaltyProjection",
    "HardwareTopology",
    "HypercubeEnumerator",
    "HypercubeSampleEnumerator",
    "IdealPenalty",
    "ProjectedBLPPenalty",
    "ProjectedPenaltyFit",
    "ProjectionMeasures",
    "SlackTargetDistribution",
    "UniformSlackTarget",
    "project_blp_penalty",
    "project_blp_penalty_importance",
    "project_penalty_values",
    "project_penalty_values_importance",
]
