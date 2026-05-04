"""Hardware topology descriptors for projected QUBO fits."""

from __future__ import annotations

from typing import Iterable


class HardwareTopology:
    """One logical topology induced by the available pair couplings.

    Parameters
    ----------
    n:
        Number of logical variables.
    E:
        Iterable of undirected pair couplings. Each edge is normalized to
        ``(min(i, j), max(i, j))`` and duplicates are removed while keeping
        the first occurrence.
    """

    def __init__(
        self,
        n: int,
        E: Iterable[tuple[int, int]],
    ):
        num_variables = int(n)
        if num_variables < 0:
            raise ValueError("n must be nonnegative")

        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for raw_edge in E:
            if len(raw_edge) != 2:
                raise ValueError(
                    "each edge must contain exactly two indices"
                )
            u = int(raw_edge[0])
            v = int(raw_edge[1])
            if u == v:
                raise ValueError(
                    "self-loops are not allowed in HardwareTopology"
                )
            if (
                u < 0
                or v < 0
                or u >= num_variables
                or v >= num_variables
            ):
                raise ValueError(
                    "edge indices must lie in the range [0, n)"
                )
            edge = (u, v) if u < v else (v, u)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)

        self.n = num_variables
        self.E = edges
        self.calE = self._build_feature_index_set()

    def _build_feature_index_set(
        self,
    ) -> list[frozenset[int]]:
        """Return the admissible Walsh feature index set."""
        calE = [frozenset()]
        calE.extend(
            frozenset({index}) for index in range(self.n)
        )
        calE.extend(frozenset(edge) for edge in self.E)
        return calE

    @property
    def num_edges(self) -> int:
        """Return the number of available quadratic couplings."""
        return len(self.E)

    @classmethod
    def full(
        cls,
        n: int,
    ) -> "HardwareTopology":
        """Return the complete pairwise logical topology on ``n`` variables."""
        num_variables = int(n)
        return cls(
            num_variables,
            (
                (i, j)
                for i in range(num_variables)
                for j in range(i + 1, num_variables)
            ),
        )


__all__ = ["HardwareTopology"]
