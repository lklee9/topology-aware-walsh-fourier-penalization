"""CSV I/O helpers for shared tuning artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from experiments.experiment_config import (
    DEFAULT_PROJECTED_STANDARDIZE,
)
from experiments.utils.driver_common import write_rows_csv
from experiments.utils.tuning_models import (
    SelectedProjectedConfig,
    TunedProjectedMultipliers,
    TunedUnbalancedParameters,
    TuningRunOutputs,
)
from experiments.utils.unbalanced_pipeline import (
    UP_LAMBDA_GAUGE,
    normalized_unbalanced_lambda_shape,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
GLOBAL_PROJECTED_PENALTY_SUMMARY_PATH = (
    EXPERIMENTS_DIR
    / "results"
    / "unbalanced_penalization"
    / "projected_penalty_tuning_summary.csv"
)
GLOBAL_UNBALANCED_PENALTY_SUMMARY_PATH = (
    EXPERIMENTS_DIR
    / "results"
    / "unbalanced_penalization"
    / "unbalanced_penalty_tuning_summary.csv"
)

_FAMILY_ALIASES = {
    "bpp": "mis",
    "kp": "mdkp",
}


def _normalise_family_name(family: str) -> str:
    """Map summary-file family labels to canonical repo names."""
    key = str(family).strip().lower()
    return _FAMILY_ALIASES.get(key, key)


def _parse_bool(raw_value: object) -> bool:
    """Parse one CSV boolean field."""
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    return value in {"1", "true", "yes", "t", "y"}


def _parse_optional_float(
    raw_value: object,
) -> float | None:
    """Parse one optional floating-point CSV field."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return float(value)


def _parse_optional_int(raw_value: object) -> int | None:
    """Parse one optional integer CSV field."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return int(value)


def _parse_optional_str(raw_value: object) -> str | None:
    """Parse one optional string CSV field."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return value


@dataclass(frozen=True)
class LoadedProjectedPenaltySummary:
    """One method/family projected-weight summary row."""

    method: str
    family: str
    anchor_size: int
    equality_multiplier: float
    inequality_multiplier: float
    projection_method: str | None
    projection_measure: str | None
    projection_penalty_template: str | None
    projection_selection_mode: str | None
    projection_selection_source: str | None
    projection_candidate_rank: int | None
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str


@dataclass(frozen=True)
class LoadedUnbalancedPenaltySummary:
    """One family-level UP summary row."""

    family: str
    anchor_size: int
    up_equality_multiplier: float | None
    up_inequality_multiplier: float
    up_lambda1_shape: float
    up_lambda2_shape: float
    up_lambda_gauge: str | None
    normalization_regime: str | None
    per_constraint_standardization: bool | None
    global_multiplier: float | None
    lambda0: float | None
    lambda1: float | None
    lambda2: float | None
    base_parameter_source: str | None
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str


def _parse_optional_bool(raw_value: object) -> bool | None:
    """Parse one optional CSV boolean field."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return _parse_bool(value)


def _projected_tuning_summary_fieldnames(
    rows: list[dict[str, object]],
) -> list[str]:
    """Return the preferred field order for projected tuning summaries."""
    preferred = [
        "method",
        "family",
        "anchor_size",
        "equality_multiplier",
        "inequality_multiplier",
        "projection_measure",
        "projection_penalty_template",
    ]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in rows):
            seen.add(key)
            fieldnames.append(key)
    template_param_keys = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith(
                "projection_penalty_template_"
            )
            and key != "projection_penalty_template"
        }
    )
    for key in template_param_keys:
        seen.add(key)
        fieldnames.append(key)
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def write_projected_tuning_summary_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the projected tuning summary with a stable column order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _projected_tuning_summary_fieldnames(rows)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def load_projected_penalty_summaries(
    path: Path = GLOBAL_PROJECTED_PENALTY_SUMMARY_PATH,
) -> dict[tuple[str, str], LoadedProjectedPenaltySummary]:
    """Return projected tuning summaries keyed by ``(method, family)``."""
    summaries: dict[
        tuple[str, str], LoadedProjectedPenaltySummary
    ] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row["method"]).strip()
            family = _normalise_family_name(row["family"])
            key = (method, family)
            if key in summaries:
                raise ValueError(
                    "duplicate projected penalty summary row "
                    f"for {key} in {path}"
                )
            summaries[key] = LoadedProjectedPenaltySummary(
                method=method,
                family=family,
                anchor_size=int(row["anchor_size"]),
                equality_multiplier=float(
                    row["equality_multiplier"]
                ),
                inequality_multiplier=float(
                    row["inequality_multiplier"]
                ),
                projection_method=_parse_optional_str(
                    row.get("projection_method")
                ),
                projection_measure=_parse_optional_str(
                    row.get("projection_measure")
                ),
                projection_penalty_template=_parse_optional_str(
                    row.get("projection_penalty_template")
                ),
                projection_selection_mode=_parse_optional_str(
                    row.get("projection_selection_mode")
                ),
                projection_selection_source=_parse_optional_str(
                    row.get("projection_selection_source")
                ),
                projection_candidate_rank=_parse_optional_int(
                    row.get("projection_candidate_rank")
                ),
                tuning_objective=str(
                    row["tuning_objective"]
                ).strip(),
                objective_value=float(
                    row["objective_value"]
                ),
                success=_parse_bool(row["success"]),
                status=int(row["status"]),
                message=str(row["message"]).strip(),
            )
    return summaries


def load_unbalanced_penalty_summaries(
    path: Path = GLOBAL_UNBALANCED_PENALTY_SUMMARY_PATH,
) -> dict[str, LoadedUnbalancedPenaltySummary]:
    """Return UP tuning summaries keyed by canonical family name."""
    summaries: dict[str, LoadedUnbalancedPenaltySummary] = (
        {}
    )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            family = _normalise_family_name(row["family"])
            if family in summaries:
                raise ValueError(
                    "duplicate unbalanced penalty summary row "
                    f"for {family} in {path}"
                )
            global_multiplier = _parse_optional_float(
                row.get("global_multiplier")
            )
            lambda0 = _parse_optional_float(
                row.get("lambda0")
            )
            lambda1 = _parse_optional_float(
                row.get("lambda1")
            )
            lambda2 = _parse_optional_float(
                row.get("lambda2")
            )
            up_equality_multiplier = _parse_optional_float(
                row.get("up_equality_multiplier")
            )
            up_inequality_multiplier = (
                _parse_optional_float(
                    row.get("up_inequality_multiplier")
                )
            )
            up_lambda1_shape = _parse_optional_float(
                row.get("up_lambda1_shape")
            )
            up_lambda2_shape = _parse_optional_float(
                row.get("up_lambda2_shape")
            )
            up_lambda_gauge = _parse_optional_str(
                row.get("up_lambda_gauge")
            )

            if (
                up_lambda1_shape is None
                or up_lambda2_shape is None
            ):
                legacy_lambda1 = (
                    0.0
                    if lambda1 is None
                    else float(lambda1)
                )
                legacy_lambda2 = (
                    0.0
                    if lambda2 is None
                    else float(lambda2)
                )
                up_lambda1_shape, up_lambda2_shape = (
                    normalized_unbalanced_lambda_shape(
                        legacy_lambda1,
                        legacy_lambda2,
                    )
                )
            if up_lambda_gauge is None:
                up_lambda_gauge = UP_LAMBDA_GAUGE
            if up_equality_multiplier is None:
                up_equality_multiplier = lambda0
            if up_inequality_multiplier is None:
                if global_multiplier is not None:
                    up_inequality_multiplier = float(
                        global_multiplier
                    )
                else:
                    up_inequality_multiplier = float(
                        (up_lambda1_shape or 0.0)
                        + (up_lambda2_shape or 0.0)
                    )
            summaries[family] = (
                LoadedUnbalancedPenaltySummary(
                    family=family,
                    anchor_size=int(row["anchor_size"]),
                    up_equality_multiplier=up_equality_multiplier,
                    up_inequality_multiplier=float(
                        up_inequality_multiplier
                    ),
                    up_lambda1_shape=float(
                        up_lambda1_shape
                    ),
                    up_lambda2_shape=float(
                        up_lambda2_shape
                    ),
                    up_lambda_gauge=up_lambda_gauge,
                    normalization_regime=_parse_optional_str(
                        row.get("normalization_regime")
                    ),
                    per_constraint_standardization=_parse_optional_bool(
                        row.get(
                            "per_constraint_standardization"
                        )
                    ),
                    global_multiplier=global_multiplier,
                    lambda0=lambda0,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    base_parameter_source=_parse_optional_str(
                        row.get("base_parameter_source")
                    ),
                    tuning_objective=str(
                        row["tuning_objective"]
                    ).strip(),
                    objective_value=float(
                        row["objective_value"]
                    ),
                    success=_parse_bool(row["success"]),
                    status=int(row["status"]),
                    message=str(row["message"]).strip(),
                )
            )
    return summaries


def _selected_projected_template_kwargs(
    row: dict[str, str],
) -> dict[str, float]:
    """Extract template kwargs stored as flat CSV columns."""
    prefix = "projection_penalty_template_"
    return {
        key[len(prefix) :]: float(value)
        for key, value in row.items()
        if key.startswith(prefix)
        and key != "projection_penalty_template"
        and value not in ("", None)
    }


def load_tuned_unbalanced_parameters(
    path: Path,
) -> dict[str, TunedUnbalancedParameters]:
    """Load family-level UP tuning outputs from CSV."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    loaded: dict[str, TunedUnbalancedParameters] = {}
    for row in rows:
        family = _normalise_family_name(row["family"])
        global_multiplier = _parse_optional_float(
            row.get("global_multiplier")
        )
        lambda0 = _parse_optional_float(row.get("lambda0"))
        lambda1 = _parse_optional_float(row.get("lambda1"))
        lambda2 = _parse_optional_float(row.get("lambda2"))
        up_equality_multiplier = _parse_optional_float(
            row.get("up_equality_multiplier")
        )
        up_inequality_multiplier = _parse_optional_float(
            row.get("up_inequality_multiplier")
        )
        up_lambda1_shape = _parse_optional_float(
            row.get("up_lambda1_shape")
        )
        up_lambda2_shape = _parse_optional_float(
            row.get("up_lambda2_shape")
        )
        if (
            up_lambda1_shape is None
            or up_lambda2_shape is None
        ):
            up_lambda1_shape, up_lambda2_shape = (
                normalized_unbalanced_lambda_shape(
                    (
                        0.0
                        if lambda1 is None
                        else float(lambda1)
                    ),
                    (
                        0.0
                        if lambda2 is None
                        else float(lambda2)
                    ),
                )
            )
        if up_equality_multiplier is None:
            up_equality_multiplier = lambda0
        if up_inequality_multiplier is None:
            if global_multiplier is not None:
                up_inequality_multiplier = float(
                    global_multiplier
                )
            else:
                up_inequality_multiplier = float(
                    up_lambda1_shape + up_lambda2_shape
                )
        loaded[family] = TunedUnbalancedParameters(
            family=family,
            anchor_size=int(row["anchor_size"]),
            up_equality_multiplier=up_equality_multiplier,
            up_inequality_multiplier=float(
                up_inequality_multiplier
            ),
            up_lambda1_shape=float(up_lambda1_shape),
            up_lambda2_shape=float(up_lambda2_shape),
            up_lambda_gauge=str(
                row.get("up_lambda_gauge", UP_LAMBDA_GAUGE)
            ),
            normalization_regime=_parse_optional_str(
                row.get("normalization_regime")
            ),
            per_constraint_standardization=_parse_optional_bool(
                row.get("per_constraint_standardization")
            ),
            global_multiplier=global_multiplier,
            lambda0=lambda0,
            lambda1=lambda1,
            lambda2=lambda2,
            base_parameter_source=_parse_optional_str(
                row.get("base_parameter_source")
            ),
            tuning_objective=str(
                row.get("tuning_objective", "gap")
            ),
            objective_value=float(row["objective_value"]),
            success=_parse_bool(row.get("success")),
            status=int(row["status"]),
            message=str(row.get("message", "")),
        )
    return loaded


def load_selected_projected_configs(
    path: Path,
) -> dict[tuple[str, str], SelectedProjectedConfig]:
    """Load the selected projected configs from the tuning summary CSV."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    loaded: dict[
        tuple[str, str], SelectedProjectedConfig
    ] = {}
    for row in rows:
        family = _normalise_family_name(row["family"])
        method = str(row["method"])
        tuned = TunedProjectedMultipliers(
            method=method,
            family=family,
            anchor_size=int(row["anchor_size"]),
            equality_multiplier=float(
                row["equality_multiplier"]
            ),
            inequality_multiplier=float(
                row["inequality_multiplier"]
            ),
            tuning_objective=str(
                row.get("tuning_objective", "gap")
            ),
            objective_value=float(row["objective_value"]),
            success=_parse_bool(row.get("success")),
            status=int(row["status"]),
            message=str(row.get("message", "")),
        )
        loaded[(family, method)] = SelectedProjectedConfig(
            method=method,
            family=family,
            projection_method=_parse_optional_str(
                row.get("projection_method")
            ),
            measure_name=str(row["projection_measure"]),
            penalty_template=str(
                row["projection_penalty_template"]
            ),
            penalty_template_kwargs=_selected_projected_template_kwargs(
                row
            ),
            selection_mode=str(
                row["projection_selection_mode"]
            ),
            selection_source=str(
                row["projection_selection_source"]
            ),
            candidate_rank=int(
                row["projection_candidate_rank"]
            ),
            projected_standardize=(
                DEFAULT_PROJECTED_STANDARDIZE
                if _parse_optional_bool(
                    row.get("projected_standardize")
                )
                is None
                else bool(
                    _parse_optional_bool(
                        row.get("projected_standardize")
                    )
                )
            ),
            tuning=tuned,
        )
    return loaded


def load_tuning_outputs(
    output_dir: Path,
) -> TuningRunOutputs:
    """Load the persisted tuning outputs needed downstream."""
    projected_tuning_path = (
        output_dir / "projected_penalty_tuning_summary.csv"
    )
    projected_selection_path = (
        output_dir / "projected_combo_selection_summary.csv"
    )
    up_tuning_path = (
        output_dir / "unbalanced_penalty_tuning_summary.csv"
    )

    if not projected_tuning_path.exists():
        raise FileNotFoundError(
            "missing projected tuning summary: "
            f"{projected_tuning_path}"
        )
    if not up_tuning_path.exists():
        raise FileNotFoundError(
            f"missing UP tuning summary: {up_tuning_path}"
        )

    projected_tuning_rows: list[dict[str, object]] = []
    projected_selection_rows: list[dict[str, object]] = []
    up_tuning_rows: list[dict[str, object]] = []

    with projected_tuning_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        projected_tuning_rows = list(csv.DictReader(handle))
    if projected_selection_path.exists():
        with projected_selection_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            projected_selection_rows = list(
                csv.DictReader(handle)
            )
    with up_tuning_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        up_tuning_rows = list(csv.DictReader(handle))

    return TuningRunOutputs(
        projected_tuning_rows=projected_tuning_rows,
        projected_selection_rows=projected_selection_rows,
        up_tuning_rows=up_tuning_rows,
        tuned_unbalanced_params=load_tuned_unbalanced_parameters(
            up_tuning_path
        ),
        selected_projected_configs=load_selected_projected_configs(
            projected_tuning_path
        ),
    )


__all__ = [
    "GLOBAL_PROJECTED_PENALTY_SUMMARY_PATH",
    "GLOBAL_UNBALANCED_PENALTY_SUMMARY_PATH",
    "LoadedProjectedPenaltySummary",
    "LoadedUnbalancedPenaltySummary",
    "_normalise_family_name",
    "_parse_bool",
    "_parse_optional_float",
    "_parse_optional_int",
    "_parse_optional_str",
    "load_projected_penalty_summaries",
    "load_selected_projected_configs",
    "load_tuned_unbalanced_parameters",
    "load_tuning_outputs",
    "load_unbalanced_penalty_summaries",
    "write_projected_tuning_summary_csv",
    "write_rows_csv",
]
