"""Shared helpers for synthetic benchmark problem batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.utils.driver_common import (
    build_problem_instances,
    problem_batch_getter,
)
from fourier_projection.blp import BLP

DEFAULT_SYNTHETIC_FAMILY_SIZES = {
    "mdkp": [50, 60],
    "mis": [50, 60],
}
_DEFAULT_SYNTHETIC_FAMILIES = tuple(
    DEFAULT_SYNTHETIC_FAMILY_SIZES
)


@dataclass(frozen=True)
class SyntheticProblemInstance:
    """One generated problem plus its stable synthetic key."""

    family: str
    size: int
    instance_index: int
    problem_seed: int
    instance_name: str
    blp: BLP


def synthetic_instance_name(
    family: str,
    size: int,
    problem_seed: int,
) -> str:
    """Return the stable descriptive name used for one instance."""
    normalized_family = (
        str(family).strip().lower().replace("-", "_")
    )
    return (
        f"synthetic_{normalized_family}_size_{int(size)}"
        f"_seed_{int(problem_seed)}"
    )


def synthetic_reference_key(
    *,
    family: str,
    size: int,
    problem_seed: int,
) -> tuple[str, int, int]:
    """Return the canonical join key for synthetic references."""
    return (
        str(family).strip().lower(),
        int(size),
        int(problem_seed),
    )


def load_synthetic_problem_batches(
    *,
    base_seed: int,
    num_instances: int,
    instance_manifest: str | Path | None,
    family_sizes: dict[str, list[int]],
) -> dict[tuple[str, int], list[SyntheticProblemInstance]]:
    """Return generated synthetic problems keyed by family and size."""
    get_problem_batch = problem_batch_getter(
        base_seed=base_seed,
        num_instances=num_instances,
        instance_manifest=instance_manifest,
        family_sizes=family_sizes,
    )
    batches: dict[
        tuple[str, int], list[SyntheticProblemInstance]
    ] = {}
    for family in _DEFAULT_SYNTHETIC_FAMILIES:
        for size in family_sizes.get(family, []):
            problems = get_problem_batch(family, int(size))
            batches[(family, int(size))] = [
                SyntheticProblemInstance(
                    family=family,
                    size=int(size),
                    instance_index=int(instance_index),
                    problem_seed=int(
                        problem.metadata["problem_seed"]
                    ),
                    instance_name=synthetic_instance_name(
                        family,
                        int(size),
                        int(
                            problem.metadata["problem_seed"]
                        ),
                    ),
                    blp=problem,
                )
                for instance_index, problem in enumerate(
                    problems
                )
            ]
    return batches


def reconstruct_synthetic_problem(
    *,
    family: str,
    size: int,
    problem_seed: int,
) -> SyntheticProblemInstance:
    """Rebuild one generated problem from its canonical key."""
    problems = build_problem_instances(
        str(family).strip().lower(),
        int(size),
        base_seed=0,
        num_instances=1,
        instance_seeds=[int(problem_seed)],
    )
    if len(problems) != 1:
        raise RuntimeError(
            "expected exactly one reconstructed synthetic problem, found "
            f"{len(problems)}"
        )
    problem = problems[0]
    return SyntheticProblemInstance(
        family=str(family).strip().lower(),
        size=int(size),
        instance_index=0,
        problem_seed=int(problem_seed),
        instance_name=synthetic_instance_name(
            family, size, problem_seed
        ),
        blp=problem,
    )


__all__ = [
    "DEFAULT_SYNTHETIC_FAMILY_SIZES",
    "SyntheticProblemInstance",
    "load_synthetic_problem_batches",
    "reconstruct_synthetic_problem",
    "synthetic_instance_name",
    "synthetic_reference_key",
]
