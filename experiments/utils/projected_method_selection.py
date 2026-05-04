"""Helpers for the fixed projected penalty choice used downstream."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.experiment_config import (
    DEFAULT_MEASURE_LAM,
)
from experiments.utils.projection_measure import (
    canonical_projection_measure_name,
)
from experiments.utils.tuning_models import (
    SelectedProjectedConfig,
    TunedProjectedMultipliers,
)


@dataclass(frozen=True)
class ProjectionComboChoice:
    """One projected penalty-template / measure candidate."""

    family: str
    topology: str
    penalty_template: str
    measure_name: str
    source: str
    candidate_rank: int


def projected_method_topology(method: str) -> str:
    """Map one projected method to its logical topology family."""
    if method in {"projected_full", "projected_up_support"}:
        return "fully_connected"
    if method in {"projected_pegasus"}:
        return "pegasus"
    if method in {"projected_chimera"}:
        return "chimera"
    if method in {"projected_zephyr"}:
        return "zephyr"
    raise ValueError(f"unknown projected method: {method}")


def load_projection_combo_candidates(
    *,
    selection_mode: str,
    family: str,
    method: str,
    fixed_measure_name: str,
    fixed_penalty_template: str,
) -> list[ProjectionComboChoice]:
    """Return the fixed projected combo used by active experiments."""
    if selection_mode != "fixed":
        raise ValueError(
            "only 'fixed' projection selection is supported"
        )
    topology = projected_method_topology(method)
    return [
        ProjectionComboChoice(
            family=family,
            topology=topology,
            penalty_template=fixed_penalty_template,
            measure_name=canonical_projection_measure_name(
                fixed_measure_name,
                legacy_default_lam=DEFAULT_MEASURE_LAM,
            ),
            source="fixed_cli",
            candidate_rank=1,
        )
    ]


def template_row_fields(
    template_name: str,
    template_kwargs: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return flat CSV fields that describe one selected template."""
    row: dict[str, object] = {
        "projection_penalty_template": template_name
    }
    for key, value in (template_kwargs or {}).items():
        row[f"projection_penalty_template_{key}"] = float(
            value
        )
    return row


def _family_projection_measure_key(
    measure_name: str,
) -> tuple[str]:
    """Return a family-level key for one projection measure."""
    return (
        canonical_projection_measure_name(
            measure_name,
            legacy_default_lam=DEFAULT_MEASURE_LAM,
        ),
    )


def family_projection_combo_key(
    candidate: ProjectionComboChoice,
) -> tuple[object, ...]:
    """Return a family-level key for one penalty/measure candidate."""
    return (
        int(candidate.candidate_rank),
        candidate.penalty_template,
        *_family_projection_measure_key(
            candidate.measure_name
        ),
    )


def shared_projected_selection_sort_key(
    candidate: ProjectionComboChoice,
    *,
    method_configs: dict[str, SelectedProjectedConfig],
) -> tuple[object, ...]:
    """Aggregate method-level tuning objectives into one family score."""
    objective_values = [
        float(config.tuning.objective_value)
        for config in method_configs.values()
    ]
    return (
        float(sum(objective_values)),
        float(max(objective_values)),
        int(candidate.candidate_rank),
        candidate.penalty_template,
        canonical_projection_measure_name(
            candidate.measure_name,
            legacy_default_lam=DEFAULT_MEASURE_LAM,
        ),
    )


def projected_candidate_spec(
    method: str,
    *,
    family: str,
    candidate: ProjectionComboChoice,
    selection_mode: str,
    tuned: TunedProjectedMultipliers,
    default_template: str | None = None,
    projection_method: str | None = None,
    penalty_template_kwargs: dict[str, float] | None = None,
    projected_standardize: bool | None = None,
) -> SelectedProjectedConfig:
    """Materialize one selected projected config."""
    resolved_projection_method = projection_method
    if resolved_projection_method is None:
        if method in {
            "projected_full",
            "projected_up_support",
            "projected_pegasus",
            "projected_chimera",
            "projected_zephyr",
        }:
            resolved_projection_method = method
        else:
            raise ValueError(
                f"unknown projected method: {method}"
            )

    resolved_template = candidate.penalty_template
    if not resolved_template:
        if default_template is None:
            raise ValueError(
                "default_template is required when the "
                "candidate does not define a template"
            )
        resolved_template = default_template

    return SelectedProjectedConfig(
        method=method,
        family=family,
        projection_method=resolved_projection_method,
        measure_name=canonical_projection_measure_name(
            candidate.measure_name,
            legacy_default_lam=DEFAULT_MEASURE_LAM,
        ),
        penalty_template=resolved_template,
        penalty_template_kwargs=dict(
            penalty_template_kwargs or {}
        ),
        selection_mode=selection_mode,
        selection_source=candidate.source,
        candidate_rank=int(candidate.candidate_rank),
        projected_standardize=projected_standardize,
        tuning=tuned,
    )
