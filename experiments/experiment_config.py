"""Experiment-wide configuration constants used by the drivers.

This central module collects the top-level DEFAULT_* and other experiment
constants so they can be imported from a single place by the tuning and
comparison entry points.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_DIR / "results" / "unbalanced_penalization"
)
DEFAULT_TUNING_DIR = EXPERIMENTS_DIR / "tunings"
DEFAULT_METHODS = (
    "unbalanced",
    "projected_full",
    "projected_up_support",
    "projected_pegasus",
    "projected_chimera",
    "projected_zephyr",
)
DEFAULT_SEED = 1
DEFAULT_NUM_INSTANCES = 20
DEFAULT_PROJECTION_MEASURE = "q2"
DEFAULT_PROJECTION_PENALTY_TEMPLATE = "heaviside"
DEFAULT_PROJECTION_SELECTION_MODE = "fixed"
DEFAULT_MDKP_SIZES = [5, 10, 15, 20]
DEFAULT_MIS_SIZES = [8, 12, 16, 20]
DEFAULT_MEASURE_LAM = 0.01
DEFAULT_TUNING_SIZES = {
    "mdkp": 15,
    "mis": 16,
}
DEFAULT_CHUNK_SIZE = 1 << 15
DEFAULT_PEGASUS_SIZE = 16
DEFAULT_PROJECTION_SAMPLE_CAP_LOG2 = 15
DEFAULT_PROJECTION_REG = 1e-8
DEFAULT_PROJECTED_STANDARDIZE = True
DEFAULT_QAOA_GRID_SIZE = 50
DEFAULT_QAOA_NUM_READS = 1000
DEFAULT_LOGICAL_ANNEALER_SAMPLERS = ("sqa",)
DEFAULT_PROGRESS_UI = "plain"
DEFAULT_SQA_NUM_READS = 1_000
DEFAULT_SQA_NUM_SWEEPS = 1_000
DEFAULT_SQA_NUM_SWEEPS_PER_BETA = 1
DEFAULT_SQA_BETA_SCALE = 1.0
DEFAULT_QPU_STANDARD_ANNEAL_TIME = 50.0
QAOA_SELECTION_RULE_CHOICES = (
    "max_logical_optimum_probability",
    "min_expected_energy",
    "min_feasible_objective_expectation",
)
DEFAULT_QAOA_SELECTION_RULES = (
    "max_logical_optimum_probability",
)
DEFAULT_TUNING_MIN = 0.01
DEFAULT_TUNING_NM_START_POINTS = 3
DEFAULT_TUNING_NM_MAXITER = 200
DEFAULT_TUNING_NM_XATOL = 1e-3
DEFAULT_TUNING_NM_FATOL = 1e-8
DEFAULT_TUNING_OBJECTIVE = "gap"
FAMILY_ORDER = (
    "mdkp",
    "mis",
)
FAMILY_LABELS = {
    "mdkp": "MDKP",
    "mis": "MIS",
}
METHOD_LABELS = {
    "unbalanced": "Unbalanced Penalization",
    "projected_full": "Projected Penalty (Full Pairwise)",
    "projected_up_support": "Projected Penalty (UP Support)",
    "projected_pegasus": "Projected Penalty (Pegasus Topology)",
    "projected_chimera": "Projected Penalty (Chimera Topology)",
    "projected_zephyr": "Projected Penalty (Zephyr Topology)",
}
FAMILY_CODES = {
    "mdkp": 3,
    "mis": 4,
}
METHOD_CODES = {
    "unbalanced": 1,
    "projected_full": 2,
    "projected_up_support": 3,
    "projected_pegasus": 4,
    "projected_chimera": 5,
    "projected_zephyr": 6,
}
SQA_SCHEDULE_KIND_RANK = {
    "standard": 0,
    "pause": 1,
    "quench": 2,
}
PROJECTION_SELECTION_MODES = ("fixed",)
PROGRESS_UI_CHOICES = ("plain", "tui", "rich")
