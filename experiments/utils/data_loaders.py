"""Unified dataset loaders for the active benchmark families.

This module centralizes the MIS and MDKP parsers used by the
experiment-side benchmark runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

try:
    from fourier_projection.blp import BLP
except (
    ImportError
):  # pragma: no cover - supports direct script usage.
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fourier_projection.blp import BLP


def _normalise_selected_names(
    names: Iterable[str] | None,
) -> set[str] | None:
    """Return the requested file-name filters as a normalized set."""
    if names is None:
        return None
    selected = {
        name.strip() for name in names if name.strip()
    }
    return selected or None


def _filter_selected_paths(
    paths: list[Path],
    selected: set[str] | None,
) -> list[Path]:
    """Keep only requested file names or stems when filtered."""
    if selected is None:
        return paths
    return [
        path
        for path in paths
        if path.name in selected or path.stem in selected
    ]


def _apply_limit(
    paths: list[Path], limit: int | None
) -> list[Path]:
    """Apply a positive instance-count limit when provided."""
    if limit is None:
        return paths
    if limit <= 0:
        raise ValueError("limit must be positive")
    return paths[:limit]


@dataclass(frozen=True)
class MISInstance:
    """One graph instance for maximum independent set."""

    name: str
    path: Path
    graph: nx.Graph

    @property
    def n(self) -> int:
        return int(self.graph.number_of_nodes())

    @property
    def m(self) -> int:
        return int(self.graph.number_of_edges())


def load_mis_instance(path: str | Path) -> MISInstance:
    """Load one DIMACS edge-list graph instance."""
    path = Path(path).resolve()
    graph = nx.Graph()
    declared_nodes: int | None = None
    declared_edges: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "p":
                if len(parts) < 4:
                    raise ValueError(
                        f"invalid DIMACS problem line in {path}: {line}"
                    )
                declared_nodes = int(parts[2])
                declared_edges = int(parts[3])
                graph.add_nodes_from(
                    range(1, declared_nodes + 1)
                )
                continue

            if tag == "e":
                if len(parts) < 3:
                    raise ValueError(
                        f"invalid DIMACS edge line in {path}: {line}"
                    )
                graph.add_edge(int(parts[1]), int(parts[2]))

    if declared_nodes is None:
        raise ValueError(
            f"MIS file missing problem line: {path}"
        )

    if (
        declared_edges is not None
        and graph.number_of_edges() != declared_edges
    ):
        raise ValueError(
            "MIS file edge count does not match problem line: "
            f"{path}"
        )

    return MISInstance(
        name=path.stem,
        path=path,
        graph=graph,
    )


def load_mis_directory(
    directory: str | Path,
    *,
    limit: int | None = None,
    names: Iterable[str] | None = None,
    recursive: bool = True,
) -> list[MISInstance]:
    """Load all or a selected subset of MIS graph files."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(
            f"MIS directory does not exist: {directory}"
        )

    paths: list[Path] = []
    for suffix in ("*.txt", "*.mis"):
        path_iter = (
            directory.rglob(suffix)
            if recursive
            else directory.glob(suffix)
        )
        paths.extend(
            path for path in path_iter if path.is_file()
        )

    paths = sorted(set(paths))
    paths = _filter_selected_paths(
        paths,
        _normalise_selected_names(names),
    )
    paths = _apply_limit(paths, limit)
    return [load_mis_instance(path) for path in paths]


@dataclass(frozen=True)
class MDKPInstance:
    """One benchmark MDKP instance together with its BLP conversion."""

    name: str
    path: Path
    profits: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    optimal_profit: float | None
    blp: BLP

    @property
    def n(self) -> int:
        return int(self.profits.shape[0])

    @property
    def m(self) -> int:
        return int(self.capacities.shape[0])

    @property
    def known_feasible_objective(self) -> float | None:
        if self.optimal_profit is None:
            return None
        return float(-self.optimal_profit)

    def to_unbalanced_penalty_terms(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the quadratic-penalty QUBO terms for row squares."""
        quadratic = np.zeros((self.n, self.n), dtype=float)
        linear = np.zeros(self.n, dtype=float)
        const = 0.0

        for weights_row, capacity in zip(
            self.weights, self.capacities
        ):
            quadratic += np.outer(weights_row, weights_row)
            linear -= 2.0 * float(capacity) * weights_row
            const += float(capacity) ** 2

        return quadratic, linear, const


def _parse_mdkp_tokens(path: Path) -> list[int]:
    """Parse the standard OR-Library style MDKP file into integers."""
    text = path.read_text(encoding="utf-8")
    tokens = text.replace("\\n", " ").split()
    if not tokens:
        raise ValueError(f"empty MDKP file: {path}")

    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            f"failed to parse MDKP file as integers: {path}"
        ) from exc


def load_mdkp_instance(path: str | Path) -> MDKPInstance:
    """Load one MDKP instance and convert it to the repo BLP convention."""
    path = Path(path).resolve()
    nums = _parse_mdkp_tokens(path)
    if len(nums) < 3:
        raise ValueError(f"MDKP file is too short: {path}")

    idx = 0
    n = int(nums[idx])
    idx += 1
    m = int(nums[idx])
    idx += 1
    optimal_profit = float(nums[idx])
    idx += 1

    expected_length = 3 + n + m * n + m
    if len(nums) != expected_length:
        raise ValueError(
            f"MDKP file has {len(nums)} integers but expected "
            f"{expected_length}: {path}"
        )

    profits = np.asarray(nums[idx : idx + n], dtype=float)
    idx += n

    weights = np.asarray(
        nums[idx : idx + m * n],
        dtype=float,
    ).reshape(m, n)
    idx += m * n

    capacities = np.asarray(
        nums[idx : idx + m], dtype=float
    )
    blp = BLP(
        c=-profits,
        A=-weights,
        b=-capacities,
    )
    return MDKPInstance(
        name=path.stem,
        path=path,
        profits=profits,
        weights=weights,
        capacities=capacities,
        optimal_profit=optimal_profit,
        blp=blp,
    )


def load_mdkp_directory(
    directory: str | Path,
    *,
    limit: int | None = None,
    names: Iterable[str] | None = None,
) -> list[MDKPInstance]:
    """Load all or a selected subset of MDKP .dat files."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(
            f"MDKP directory does not exist: {directory}"
        )

    paths = sorted(directory.glob("*.dat"))
    paths = _filter_selected_paths(
        paths,
        _normalise_selected_names(names),
    )
    paths = _apply_limit(paths, limit)
    return [load_mdkp_instance(path) for path in paths]


__all__ = [
    "MDKPInstance",
    "MISInstance",
    "load_mdkp_directory",
    "load_mdkp_instance",
    "load_mis_directory",
    "load_mis_instance",
]
