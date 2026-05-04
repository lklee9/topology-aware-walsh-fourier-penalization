"""Shared helpers for the active baseline-comparison entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECTED_METHODS = (
    "projected_full",
    "projected_up_support",
    "projected_pegasus",
    "projected_chimera",
    "projected_zephyr",
)
PROJECTION_PENALTY_TEMPLATE_CHOICES = ("heaviside",)


def projected_methods() -> tuple[str, ...]:
    """Return the projected-method variants compared downstream."""
    return PROJECTED_METHODS


def openqaoa_qubo_terms_and_scale(
    quadratic: np.ndarray,
    linear: np.ndarray,
    *,
    num_variables: int,
) -> tuple[list[list[int]], list[float], float]:
    """Build OpenQAOA-style QUBO terms and the matching scale."""
    terms: list[list[int]] = []
    weights: list[float] = []
    referenced_qubits: set[int] = set()

    for index, value in enumerate(
        np.asarray(linear, dtype=float).reshape(-1)
    ):
        if abs(value) > 1e-12:
            terms.append([index])
            weights.append(float(value))
            referenced_qubits.add(index)

    quadratic = np.asarray(quadratic, dtype=float)
    for row in range(num_variables):
        for col in range(row + 1, num_variables):
            value = float(quadratic[row, col])
            if abs(value) > 1e-12:
                terms.append([row, col])
                weights.append(value)
                referenced_qubits.add(row)
                referenced_qubits.add(col)

    for qubit in range(num_variables):
        if qubit not in referenced_qubits:
            terms.append([qubit])
            weights.append(0.0)

    if not terms:
        terms = [[0]]
        weights = [0.0]

    abs_weights = np.unique(
        np.abs(np.asarray(weights, dtype=float))
    )
    abs_weights = abs_weights[np.isfinite(abs_weights)]
    if abs_weights.size == 0:
        return terms, weights, 1.0

    scale = float(abs_weights[np.argsort(abs_weights)[-1]])
    if scale <= 1e-12:
        scale = 1.0
    return terms, weights, scale


def qubo_normalization_scale(
    quadratic: np.ndarray,
    linear: np.ndarray,
    *,
    num_variables: int,
) -> float:
    """Return the shared whole-QUBO annealer normalization scale."""
    _, _, scale = openqaoa_qubo_terms_and_scale(
        quadratic,
        linear,
        num_variables=num_variables,
    )
    return float(scale)


def scale_qubo_coefficients(
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    normalization_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the QUBO coefficients divided by one safe scale."""
    scale = float(normalization_scale)
    if not np.isfinite(scale) or abs(scale) <= 1e-12:
        scale = 1.0
    return (
        np.asarray(quadratic, dtype=float) / scale,
        np.asarray(linear, dtype=float) / scale,
        float(const) / scale,
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_qaoa_selection_rule: bool,
    output_dir: Path,
    seed: int,
    num_instances: int,
    progress_ui_choices: tuple[str, ...],
    progress_ui_default: str,
    projection_measure_default: str,
    projection_penalty_template_default: str,
    projection_selection_modes: tuple[str, ...],
    projection_selection_mode_default: str,
    qaoa_selection_rule_choices: (
        tuple[str, ...] | None
    ) = None,
) -> argparse.ArgumentParser:
    """Populate the shared CLI flags used by the entry points."""
    if tuple(projection_selection_modes) != ("fixed",):
        raise ValueError(
            "only fixed projection selection is supported"
        )
    if projection_selection_mode_default != "fixed":
        raise ValueError(
            "projection selection must default to 'fixed'"
        )
    if (
        projection_penalty_template_default
        not in PROJECTION_PENALTY_TEMPLATE_CHOICES
    ):
        raise ValueError(
            "unsupported projected penalty template default: "
            f"{projection_penalty_template_default}"
        )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory for figures and CSV summaries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help=(
            "Base RNG seed used for instance generation and "
            "sampled projection"
        ),
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=num_instances,
        help="Number of random instances per family/size",
    )
    parser.add_argument(
        "--progress-ui",
        choices=progress_ui_choices,
        default=progress_ui_default,
        help="Console progress rendering mode",
    )
    parser.add_argument(
        "--projection-measure",
        choices=[projection_measure_default],
        default=projection_measure_default,
        help=(
            "Projection measure used to sample states for fitting the "
            "topology-restricted projected penalties. Only the fixed "
            f"'{projection_measure_default}' choice is supported."
        ),
    )
    parser.add_argument(
        "--projection-penalty-template",
        choices=[projection_penalty_template_default],
        default=projection_penalty_template_default,
        help=(
            "Ideal inequality penalty template fitted by the projected "
            "method. Only the fixed "
            f"'{projection_penalty_template_default}' choice is supported."
        ),
    )
    parser.add_argument(
        "--projection-selection-mode",
        choices=["fixed"],
        default="fixed",
        help=(
            "How the projected measure/template combo is chosen before "
            "scalar multiplier tuning. Only 'fixed' is supported."
        ),
    )
    if include_qaoa_selection_rule:
        if qaoa_selection_rule_choices is None:
            raise ValueError(
                "qaoa_selection_rule_choices must be provided when "
                "include_qaoa_selection_rule is True"
            )
        parser.add_argument(
            "--qaoa-selection-rule",
            action="append",
            choices=list(qaoa_selection_rule_choices),
            default=None,
            help=(
                "Grid-point selector used for the p=1 QAOA baseline. "
                "Repeat this flag to emit side-by-side summaries for "
                "multiple selectors."
            ),
        )
    return parser
