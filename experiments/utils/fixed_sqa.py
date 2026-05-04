"""Shared fixed-SQA schedule helpers used by experiment drivers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from experiments.experiment_config import (
    DEFAULT_QPU_STANDARD_ANNEAL_TIME,
    DEFAULT_SQA_BETA_SCALE,
)
from experiments.utils.qa_simulator import (
    build_standard_anneal_schedule,
)


@dataclass(frozen=True)
class FixedSqaSchedule:
    """One fixed QPU-style schedule paired with the shared beta scale."""

    schedule_id: str
    schedule_kind: str
    anneal_schedule: tuple[tuple[float, float], ...]
    beta_scale: float
    total_schedule_time: float


def _schedule_token(value: float) -> str:
    """Return one compact identifier token for a float."""
    return f"{float(value):g}".replace(".", "p")


@lru_cache(maxsize=1)
def fixed_sqa_schedule() -> FixedSqaSchedule:
    """Return the single fixed SQA schedule used by all experiments."""
    total_time = float(DEFAULT_QPU_STANDARD_ANNEAL_TIME)
    return FixedSqaSchedule(
        schedule_id=f"standard_t{_schedule_token(total_time)}",
        schedule_kind="standard",
        anneal_schedule=build_standard_anneal_schedule(
            total_time
        ),
        beta_scale=float(DEFAULT_SQA_BETA_SCALE),
        total_schedule_time=total_time,
    )


def sqa_schedule_catalog() -> tuple[FixedSqaSchedule, ...]:
    """Return the catalog of supported fixed SQA schedules."""
    return (fixed_sqa_schedule(),)


def anneal_schedule_json(
    anneal_schedule: tuple[tuple[float, float], ...],
) -> str:
    """Return one stable JSON encoding of an anneal schedule."""
    return json.dumps(
        [
            [float(time), float(anneal_fraction)]
            for time, anneal_fraction in anneal_schedule
        ],
        separators=(",", ":"),
    )


def anneal_schedule_from_json(
    raw_value: str,
) -> tuple[tuple[float, float], ...]:
    """Parse one anneal schedule serialized by ``anneal_schedule_json``."""
    payload = json.loads(raw_value)
    return tuple(
        (float(time_value), float(anneal_fraction))
        for time_value, anneal_fraction in payload
    )


__all__ = [
    "FixedSqaSchedule",
    "anneal_schedule_from_json",
    "anneal_schedule_json",
    "fixed_sqa_schedule",
    "sqa_schedule_catalog",
]
