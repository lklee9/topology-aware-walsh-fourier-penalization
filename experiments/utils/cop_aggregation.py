"""Standalone CoP table aggregation helpers.

This module rebuilds aggregate comparison tables from the instance-level
``cop_instance_summary.csv`` outputs written by the experiment runners.
It supports both the baseline-style tables used by
``unbalanced_penalization`` and the embedding-aware tables used by
``compare_embedding``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.utils.driver_common import write_rows_csv

ModeName = str


@dataclass(frozen=True)
class MetricSpec:
    """One canonical output metric and its input aliases."""

    key: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ModeConfig:
    """Aggregation schema for one input family."""

    name: ModeName
    group_keys: tuple[str, ...]
    sort_keys: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]


BASELINE_METRICS = (
    MetricSpec("sqa_cop", ("sqa_logical_cop",)),
    MetricSpec("sqa_fea", ("sqa_fea",)),
    MetricSpec("sqa_gap", ("sqa_gap",)),
)

EMBEDDING_METRICS = (
    MetricSpec("sqa_cop", ("sqa_logical_cop", "cop")),
    MetricSpec(
        "sqa_fea", ("sqa_fea", "feasible_read_fraction")
    ),
    MetricSpec("sqa_gap", ("sqa_gap",)),
    MetricSpec(
        "optimum_probability", ("optimum_probability",)
    ),
    MetricSpec("sqa_energy_gap", ("mean_sqa_energy_gap",)),
    MetricSpec("sqa_optimum_rate", ("sqa_optimum_rate",)),
    MetricSpec(
        "mean_chain_break_fraction",
        ("mean_chain_break_fraction",),
    ),
    MetricSpec("broken_read_rate", ("broken_read_rate",)),
)

BASELINE_MODE = ModeConfig(
    name="baseline",
    group_keys=("family", "size", "method"),
    sort_keys=("family", "size", "method"),
    metrics=BASELINE_METRICS,
)

EMBEDDING_MODE = ModeConfig(
    name="embedding",
    group_keys=(
        "hardware_family",
        "hardware_size",
        "family",
        "size",
        "method",
    ),
    sort_keys=(
        "hardware_family",
        "family",
        "size",
        "method",
    ),
    metrics=EMBEDDING_METRICS,
)

MODE_CONFIGS = {
    BASELINE_MODE.name: BASELINE_MODE,
    EMBEDDING_MODE.name: EMBEDDING_MODE,
}

ROBUST_OUTPUT_FILENAME = "cop_aggregate_robust_summary.csv"
LEGACY_OUTPUT_FILENAME = "cop_aggregate_summary.csv"
ZERO_ATOL = 1e-12


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into dictionaries."""
    with path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def infer_mode(
    rows: Iterable[dict[str, object]],
) -> ModeName:
    """Infer the aggregation mode from the available columns."""
    for row in rows:
        if (
            _clean_optional_text(row.get("hardware_family"))
            is not None
        ):
            return EMBEDDING_MODE.name
    return BASELINE_MODE.name


def aggregate_robust_rows(
    rows: list[dict[str, object]],
    *,
    mode: ModeName,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> list[dict[str, object]]:
    """Return robust aggregate rows for one instance-summary table."""
    config = _config_for_mode(mode)
    grouped = _group_rows(rows, config)
    out: list[dict[str, object]] = []
    for group_index, key in enumerate(
        _sorted_group_keys(grouped, config)
    ):
        items = grouped[key]
        row = _identity_row(items[0], len(items), config)
        for metric_index, metric in enumerate(
            config.metrics
        ):
            values = _metric_values(items, metric)
            row[f"{metric.key}_median"] = _finite_stat(
                values,
                lambda finite: float(np.median(finite)),
            )
            row[f"{metric.key}_q25"] = _finite_stat(
                values,
                lambda finite: float(
                    np.percentile(finite, 25)
                ),
            )
            row[f"{metric.key}_q75"] = _finite_stat(
                values,
                lambda finite: float(
                    np.percentile(finite, 75)
                ),
            )
            row[f"{metric.key}_mean"] = _finite_stat(
                values,
                lambda finite: float(np.mean(finite)),
            )
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                resamples=bootstrap_resamples,
                seed_components=(
                    bootstrap_seed,
                    group_index,
                    metric_index,
                ),
            )
            row[f"{metric.key}_ci95_low"] = ci_low
            row[f"{metric.key}_ci95_high"] = ci_high

        sqa_cop = _metric_values(items, config.metrics[0])
        sqa_gap = _metric_values(
            items, _metric_spec(config, "sqa_gap")
        )
        row["sqa_cop_nonzero_rate"] = _event_rate(
            sqa_cop,
            predicate=lambda finite: finite > ZERO_ATOL,
        )
        row["sqa_gap_zero_rate"] = _event_rate(
            sqa_gap,
            predicate=lambda finite: np.isclose(
                finite,
                0.0,
                atol=ZERO_ATOL,
            ),
        )
        out.append(row)
    return out


def aggregate_legacy_rows(
    rows: list[dict[str, object]],
    *,
    mode: ModeName,
) -> list[dict[str, object]]:
    """Return the legacy mean/std aggregate rows."""
    config = _config_for_mode(mode)
    grouped = _group_rows(rows, config)
    out: list[dict[str, object]] = []
    for key in _sorted_group_keys(grouped, config):
        items = grouped[key]
        metric_values = {
            metric.key: _metric_values(items, metric)
            for metric in config.metrics
        }
        for stat_name in ("mean", "std"):
            row = _legacy_identity_row(
                items[0], len(items), config, stat_name
            )
            for metric in config.metrics:
                values = metric_values[metric.key]
                if stat_name == "mean":
                    row[metric.key] = _finite_stat(
                        values,
                        lambda finite: float(
                            np.mean(finite)
                        ),
                    )
                else:
                    row[metric.key] = _finite_stat(
                        values,
                        lambda finite: float(
                            np.std(finite, ddof=0)
                        ),
                    )
            out.append(row)
    return out


def write_aggregate_outputs(
    rows: list[dict[str, object]],
    *,
    output_dir: Path,
    mode: ModeName,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    write_legacy_mean_std: bool,
) -> tuple[Path, Path | None]:
    """Write robust and optional legacy aggregate outputs."""
    robust_rows = aggregate_robust_rows(
        rows,
        mode=mode,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    robust_path = output_dir / ROBUST_OUTPUT_FILENAME
    write_rows_csv(robust_path, robust_rows)

    legacy_path: Path | None = None
    if write_legacy_mean_std:
        legacy_path = output_dir / LEGACY_OUTPUT_FILENAME
        write_rows_csv(
            legacy_path,
            aggregate_legacy_rows(rows, mode=mode),
        )
    return robust_path, legacy_path


def sanitize_output_dir_name(path: Path) -> str:
    """Return one stable directory-safe label for an input path."""
    parent = path.resolve().parent
    parts = parent.parts[-3:]
    text = "__".join(parts)
    cleaned = [
        (
            char
            if char.isalnum() or char in {"-", "_"}
            else "_"
        )
        for char in text
    ]
    return "".join(cleaned).strip("_") or "aggregate"


def _config_for_mode(mode: ModeName) -> ModeConfig:
    try:
        return MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unsupported aggregation mode: {mode}"
        ) from exc


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _safe_float(value: object) -> float:
    text = _clean_optional_text(value)
    if text is None:
        return math.nan
    return float(text)


def _safe_int(value: object) -> int:
    text = _clean_optional_text(value)
    if text is None:
        raise ValueError("missing integer field")
    return int(float(text))


def _group_rows(
    rows: list[dict[str, object]],
    config: ModeConfig,
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[
        tuple[object, ...], list[dict[str, object]]
    ] = {}
    for row in rows:
        key = tuple(
            _group_value(row, field)
            for field in config.group_keys
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _group_value(
    row: dict[str, object], field: str
) -> object:
    value = row.get(field)
    if field in {"size", "hardware_size"}:
        return _safe_int(value)
    text = _clean_optional_text(value)
    if text is None:
        raise ValueError(f"missing grouping field: {field}")
    return text


def _sorted_group_keys(
    grouped: dict[
        tuple[object, ...], list[dict[str, object]]
    ],
    config: ModeConfig,
) -> list[tuple[object, ...]]:
    def _sort_tuple(
        key: tuple[object, ...],
    ) -> tuple[object, ...]:
        mapping = {
            field: value
            for field, value in zip(config.group_keys, key)
        }
        values = [
            mapping[field] for field in config.sort_keys
        ]
        if config.name == EMBEDDING_MODE.name:
            values.append(mapping["hardware_size"])
        return tuple(values)

    return sorted(grouped, key=_sort_tuple)


def _identity_row(
    item: dict[str, object],
    inst: int,
    config: ModeConfig,
) -> dict[str, object]:
    row: dict[str, object] = {}
    if config.name == EMBEDDING_MODE.name:
        row["hardware_family"] = str(
            item["hardware_family"]
        )
        row["hardware_size"] = _safe_int(
            item["hardware_size"]
        )
    row["family"] = str(item["family"])
    row["size"] = _safe_int(item["size"])
    row["method"] = str(item["method"])
    row["n"] = _safe_int(item["num_variables"])
    row["inst"] = int(inst)
    return row


def _legacy_identity_row(
    item: dict[str, object],
    inst: int,
    config: ModeConfig,
    stat_name: str,
) -> dict[str, object]:
    row = _identity_row(item, inst, config)
    if config.name == BASELINE_MODE.name:
        ordered = {
            "family": row["family"],
            "size": row["size"],
            "method": row["method"],
            "n": row["n"],
            "stat": stat_name,
            "inst": row["inst"],
        }
        return ordered

    ordered = {
        "hardware_family": row["hardware_family"],
        "hardware_size": row["hardware_size"],
        "family": row["family"],
        "size": row["size"],
        "method": row["method"],
        "n": row["n"],
        "stat": stat_name,
        "inst": row["inst"],
    }
    return ordered


def _metric_spec(
    config: ModeConfig, key: str
) -> MetricSpec:
    for metric in config.metrics:
        if metric.key == key:
            return metric
    raise KeyError(key)


def _metric_values(
    items: list[dict[str, object]],
    metric: MetricSpec,
) -> np.ndarray:
    values = np.array(
        [
            _first_float(item, metric.aliases)
            for item in items
        ],
        dtype=float,
    )
    finite = np.isfinite(values)
    return values[finite]


def _first_float(
    item: dict[str, object],
    aliases: tuple[str, ...],
) -> float:
    for field in aliases:
        value = _safe_float(item.get(field))
        if np.isfinite(value):
            return value
    return math.nan


def _finite_stat(
    values: np.ndarray,
    fn,
) -> float:
    if values.size == 0:
        return math.nan
    return float(fn(values))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    seed_components: tuple[int, int, int],
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if resamples <= 0:
        raise ValueError(
            "bootstrap resamples must be positive"
        )
    if values.size == 1:
        value = float(values[0])
        return value, value

    seed = np.random.SeedSequence(
        [int(part) for part in seed_components]
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values.size,
        size=(int(resamples), values.size),
    )
    means = np.mean(values[sample_indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _event_rate(
    values: np.ndarray,
    *,
    predicate,
) -> float:
    """Return one mildly boundary-clipped event rate.

    The aggregate tables are primarily used to highlight strongly zero- or
    nonzero-inflated groups. Rare events at or below 10% are therefore
    collapsed to 0.0, and symmetric near-certain events at or above 90%
    are collapsed to 1.0.
    """
    if values.size == 0:
        return math.nan
    rate = float(np.mean(predicate(values)))
    if rate <= 0.1 + 1e-12:
        return 0.0
    if rate >= 0.9 - 1e-12:
        return 1.0
    return rate


__all__ = [
    "LEGACY_OUTPUT_FILENAME",
    "ROBUST_OUTPUT_FILENAME",
    "aggregate_legacy_rows",
    "aggregate_robust_rows",
    "infer_mode",
    "read_csv_rows",
    "sanitize_output_dir_name",
    "write_aggregate_outputs",
]
