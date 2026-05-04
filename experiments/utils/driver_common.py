"""Shared helpers used across experiment drivers.

The retained experiment entrypoints still keep their own orchestration,
reporting, and backend-specific logic. This module only centralizes the
small deterministic helpers that were previously copy-pasted between
drivers.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.experiment_config import (
    DEFAULT_MDKP_SIZES,
    DEFAULT_MIS_SIZES,
    FAMILY_CODES,
    FAMILY_ORDER,
)
from experiments.utils.problems import (
    sample_mdkp_problem,
    sample_mis_problem,
)
from fourier_projection.blp import BLP

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_MANIFEST_FIELDNAMES = (
    "family",
    "size",
    "instance_index",
    "problem_seed",
)
PROJECTED_TOPOLOGY_REGIME = "projected_topology"
_PAIR_SUPPORT_ATOL = 1e-12


def repo_relative_path(path: str | Path) -> str:
    """Return one repo-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def default_family_sizes() -> dict[str, list[int]]:
    """Return the fixed family/size grid used by the main evaluations."""
    return {
        "mdkp": list(DEFAULT_MDKP_SIZES),
        "mis": list(DEFAULT_MIS_SIZES),
    }


def build_child_rng(
    base_seed: int,
    *components: int,
) -> np.random.Generator:
    """Return one deterministic RNG derived from integer seed components."""
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            *[int(value) for value in components],
        ]
    )
    return np.random.default_rng(sequence)


def build_child_seed(
    base_seed: int,
    *components: int,
) -> int:
    """Return one deterministic integer seed derived from components."""
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            *[int(value) for value in components],
        ]
    )
    return int(
        sequence.generate_state(1, dtype=np.uint64)[0]
    )


def binary_to_spin_states(
    bitstrings: np.ndarray,
) -> np.ndarray:
    """Map binary states ``x`` to Ising spins ``z = 1 - 2x``."""
    bits = np.asarray(bitstrings, dtype=float)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    return 1.0 - 2.0 * bits


def decoded_sampleset_bits(
    sampleset: Any,
    *,
    num_variables: int,
) -> np.ndarray:
    """Return decoded logical bits in canonical variable order ``0..n-1``."""
    variable_to_index = {
        var: idx
        for idx, var in enumerate(sampleset.variables)
    }
    try:
        ordered_indices = [
            variable_to_index[i]
            for i in range(int(num_variables))
        ]
    except KeyError as exc:
        raise RuntimeError(
            "decoded sampleset variables do not match canonical logical indices"
        ) from exc
    return np.asarray(
        sampleset.record.sample, dtype=np.uint8
    )[:, ordered_indices]


def full_pair_edges(
    num_variables: int,
) -> list[tuple[int, int]]:
    """Return the complete logical pair set on ``num_variables`` vertices."""
    return [
        (i, j)
        for i in range(int(num_variables))
        for j in range(i + 1, int(num_variables))
    ]


def num_inequality_quadratic_terms(problem: Any) -> int:
    """Return the distinct variable-pair count induced by inequalities."""
    num_inequalities = int(
        getattr(problem, "num_inequalities", 0)
    )
    if num_inequalities <= 0:
        return 0

    inequality_matrix = np.asarray(
        problem.A, dtype=float
    ).reshape(
        num_inequalities,
        int(problem.num_variables),
    )
    support_pairs: set[tuple[int, int]] = set()
    for coeffs in inequality_matrix:
        active_indices = np.flatnonzero(
            np.abs(coeffs) > _PAIR_SUPPORT_ATOL
        )
        for left in range(len(active_indices)):
            for right in range(
                left + 1, len(active_indices)
            ):
                support_pairs.add(
                    (
                        int(active_indices[left]),
                        int(active_indices[right]),
                    )
                )
    return len(support_pairs)


def is_complete_pair_edge_set(
    num_variables: int,
    pair_edges: list[tuple[int, int]],
) -> bool:
    """Return whether ``pair_edges`` contains every distinct variable pair."""
    canonical_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v in pair_edges
    }
    expected_edges = (
        int(num_variables) * (int(num_variables) - 1)
    ) // 2
    return len(canonical_edges) == expected_edges


def normalised_gap(
    values: np.ndarray | float,
    optimum: float,
) -> np.ndarray:
    """Normalize energies or objectives against a reference optimum."""
    return (
        np.asarray(values, dtype=float) - float(optimum)
    ) / max(
        1.0,
        abs(float(optimum)),
    )


def objective_gap_ratio(
    found_best_feasible_objective: float | None,
    optimum_objective: float,
    *,
    atol: float = 1e-12,
) -> float:
    """Return the normalized minimization gap for the best feasible solution."""
    del atol  # Kept for the existing public call signature.
    if found_best_feasible_objective is None:
        return math.nan
    return float(
        normalised_gap(
            found_best_feasible_objective, optimum_objective
        )
    )


def write_rows_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows to CSV while preserving first-seen field order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def build_instance_seed_rows(
    *,
    base_seed: int,
    num_instances: int,
    family_sizes: dict[str, list[int]] | None = None,
) -> list[dict[str, object]]:
    """Build one deterministic manifest row set for synthetic instances."""
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    selected_family_sizes = (
        default_family_sizes()
        if family_sizes is None
        else family_sizes
    )
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        sizes = selected_family_sizes.get(family, [])
        for size in sizes:
            for instance_index in range(num_instances):
                rows.append(
                    {
                        "family": family,
                        "size": int(size),
                        "instance_index": int(
                            instance_index
                        ),
                        "problem_seed": build_child_seed(
                            base_seed,
                            1_000,
                            FAMILY_CODES[family],
                            size,
                            instance_index,
                        ),
                    }
                )
    return rows


def load_instance_seed_manifest(
    path: str | Path,
    *,
    family_sizes: dict[str, list[int]] | None = None,
) -> dict[tuple[str, int], list[int]]:
    """Load one seed manifest and return ordered seeds by family/size."""
    manifest_path = Path(path).resolve()
    with manifest_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"instance manifest is empty: {manifest_path}"
            )
        missing_columns = [
            name
            for name in INSTANCE_MANIFEST_FIELDNAMES
            if name not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "instance manifest is missing required columns: "
                f"{', '.join(missing_columns)}"
            )
        rows = list(reader)

    expected_blocks: set[tuple[str, int]] | None = None
    if family_sizes is not None:
        expected_blocks = {
            (family, int(size))
            for family in FAMILY_ORDER
            for size in family_sizes.get(family, [])
        }

    grouped: dict[tuple[str, int], dict[int, int]] = {}
    for row in rows:
        family = str(row["family"]).strip().lower()
        if family not in FAMILY_CODES:
            raise ValueError(
                f"unknown manifest family: {family}"
            )
        size = int(row["size"])
        block = (family, size)
        instance_index = int(row["instance_index"])
        problem_seed = int(row["problem_seed"])
        if instance_index < 0:
            raise ValueError(
                "instance_index must be non-negative"
            )
        block_rows = grouped.setdefault(block, {})
        if instance_index in block_rows:
            raise ValueError(
                "manifest contains duplicate family/size/index rows for "
                f"{family}/{size}/{instance_index}"
            )
        block_rows[instance_index] = problem_seed

    if expected_blocks is not None:
        missing_blocks = sorted(
            expected_blocks - set(grouped)
        )
        if missing_blocks:
            raise ValueError(
                "manifest is missing required family/size blocks: "
                + ", ".join(
                    f"{family}/{size}"
                    for family, size in missing_blocks
                )
            )

    ordered: dict[tuple[str, int], list[int]] = {}
    for block, block_rows in grouped.items():
        ordered_indices = sorted(block_rows)
        expected_indices = list(range(len(ordered_indices)))
        if ordered_indices != expected_indices:
            family, size = block
            raise ValueError(
                "manifest instance_index values must be contiguous from 0 for "
                f"{family}/{size}"
            )
        ordered[block] = [
            block_rows[index] for index in ordered_indices
        ]
    return ordered


def _annotate_problem_instance(
    problem: BLP,
    *,
    problem_seed: int,
    instance_source: str,
    instance_manifest: Path | None = None,
) -> BLP:
    """Attach one stable provenance payload to the sampled problem."""
    metadata = dict(problem.metadata)
    metadata["problem_seed"] = int(problem_seed)
    metadata.setdefault("seed", int(problem_seed))
    metadata["instance_source"] = str(instance_source)
    metadata["instance_manifest"] = (
        repo_relative_path(instance_manifest)
        if instance_manifest is not None
        else None
    )
    problem.metadata = metadata
    return problem


def problem_provenance_fields(
    problem: BLP,
) -> dict[str, object]:
    """Return one consistent row payload describing instance provenance."""
    metadata = dict(problem.metadata)
    return {
        "instance_source": str(
            metadata.get("instance_source", "generated")
        ),
        "problem_seed": (
            None
            if metadata.get("problem_seed") is None
            else int(metadata["problem_seed"])
        ),
        "instance_manifest": metadata.get(
            "instance_manifest"
        ),
    }


def projection_regime_fields(
    method: str,
    *,
    hardware_family: str | None = None,
) -> dict[str, object]:
    """Normalize one method name into paper-level projection categories."""
    normalized_method = str(method).strip().lower()
    if normalized_method == "unbalanced":
        return {
            "projection_regime": "unbalanced",
            "projection_topology_family": None,
        }
    if normalized_method == "projected_full":
        return {
            "projection_regime": "projected_full",
            "projection_topology_family": None,
        }
    if normalized_method == "projected_up_support":
        return {
            "projection_regime": "projected_up_support",
            "projection_topology_family": None,
        }
    if normalized_method == "projected_topology":
        if hardware_family is None:
            raise ValueError(
                "hardware_family is required for projected_topology rows"
            )
        return {
            "projection_regime": PROJECTED_TOPOLOGY_REGIME,
            "projection_topology_family": str(
                hardware_family
            )
            .strip()
            .lower(),
        }
    if normalized_method.startswith("projected_"):
        topology_family = normalized_method.removeprefix(
            "projected_"
        )
        if topology_family in {
            "chimera",
            "pegasus",
            "zephyr",
        }:
            return {
                "projection_regime": PROJECTED_TOPOLOGY_REGIME,
                "projection_topology_family": topology_family,
            }
    raise ValueError(
        f"unknown projection method for normalization: {method}"
    )


def build_problem_instances(
    family: str,
    size: int,
    *,
    base_seed: int,
    num_instances: int,
    instance_seeds: list[int] | None = None,
    instance_manifest: str | Path | None = None,
) -> list[BLP]:
    """Sample the fixed instance batch for one problem family and size."""
    manifest_path = None
    if instance_manifest is not None:
        manifest_path = Path(instance_manifest).resolve()

    problems: list[BLP] = []
    if instance_seeds is None:
        family_code = FAMILY_CODES[family]
        seeds = [
            build_child_seed(
                base_seed,
                1_000,
                family_code,
                size,
                instance_index,
            )
            for instance_index in range(num_instances)
        ]
        instance_source = "generated"
    else:
        seeds = [int(seed) for seed in instance_seeds]
        instance_source = "seed_manifest"

    if family == "mdkp":
        for problem_seed in seeds:
            problems.append(
                _annotate_problem_instance(
                    sample_mdkp_problem(size, problem_seed),
                    problem_seed=problem_seed,
                    instance_source=instance_source,
                    instance_manifest=manifest_path,
                )
            )
        return problems

    if family == "mis":
        for problem_seed in seeds:
            problems.append(
                _annotate_problem_instance(
                    sample_mis_problem(size, problem_seed),
                    problem_seed=problem_seed,
                    instance_source=instance_source,
                    instance_manifest=manifest_path,
                )
            )
        return problems

    raise ValueError(f"unknown family: {family}")


def problem_batch_getter(
    *,
    base_seed: int,
    num_instances: int,
    instance_manifest: str | Path | None = None,
    family_sizes: dict[str, list[int]] | None = None,
):
    """Return one memoized problem-batch loader."""
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    problem_batches: dict[tuple[str, int], list[BLP]] = {}
    manifest_path = None
    manifest_seed_map: (
        dict[tuple[str, int], list[int]] | None
    ) = None
    if instance_manifest is not None:
        manifest_path = Path(instance_manifest).resolve()
        manifest_seed_map = load_instance_seed_manifest(
            manifest_path,
            family_sizes=family_sizes,
        )

    def get_problem_batch(
        family: str,
        size: int,
    ) -> list[BLP]:
        key = (family, size)
        if key not in problem_batches:
            if manifest_seed_map is None:
                problem_batches[key] = (
                    build_problem_instances(
                        family,
                        size,
                        base_seed=base_seed,
                        num_instances=num_instances,
                    )
                )
            else:
                try:
                    instance_seeds = manifest_seed_map[key]
                except KeyError as exc:
                    raise ValueError(
                        "instance manifest does not define a block for "
                        f"{family}/{size}"
                    ) from exc
                instance_seeds = instance_seeds[
                    : min(
                        len(instance_seeds), num_instances
                    )
                ]
                problem_batches[key] = (
                    build_problem_instances(
                        family,
                        size,
                        base_seed=base_seed,
                        num_instances=len(instance_seeds),
                        instance_seeds=instance_seeds,
                        instance_manifest=manifest_path,
                    )
                )
        return problem_batches[key]

    return get_problem_batch


__all__ = [
    "binary_to_spin_states",
    "build_instance_seed_rows",
    "build_problem_instances",
    "build_child_rng",
    "default_family_sizes",
    "full_pair_edges",
    "is_complete_pair_edge_set",
    "decoded_sampleset_bits",
    "INSTANCE_MANIFEST_FIELDNAMES",
    "load_instance_seed_manifest",
    "normalised_gap",
    "num_inequality_quadratic_terms",
    "objective_gap_ratio",
    "problem_batch_getter",
    "problem_provenance_fields",
    "projection_regime_fields",
    "repo_relative_path",
    "write_rows_csv",
]
