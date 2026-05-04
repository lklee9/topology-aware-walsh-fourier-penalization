"""
greedy_mapping.py
=================
Tractable heuristics for mapping logical BLP variables onto a hardware graph.

The central routine in this module improves on the earlier breadth-first
"take the first connected patch" rule by using a weighted logical interaction
graph derived from the constraint matrix. It then performs a deterministic
multi-start frontier placement:

1. build logical edge weights from constraint co-occurrence,
2. restrict to the largest connected hardware component,
3. try several seed pairs of important logical variables and central hardware
   qubits,
4. grow the placement greedily to preserve as much logical edge weight as
   possible, and
5. apply a small pairwise-swap local improvement pass.

This remains inexpensive on D-Wave-scale sparse graphs, but it usually
produces a more useful logical topology for the projection code than the older
pure BFS ordering.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Hashable, TypeVar

try:
    from .topology import HardwareTopology
except (
    ImportError
):  # pragma: no cover - allows running the file as a script.
    from topology import HardwareTopology


LogicalVertex = TypeVar("LogicalVertex", bound=Hashable)
HardwareVertex = TypeVar("HardwareVertex", bound=Hashable)


def heuristic_topology_aware_mapping(
    constraint_matrix: Sequence[Sequence[float]],
    hardware_vertices: Iterable[HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
    logical_vertices: Iterable[LogicalVertex] | None = None,
    num_restarts: int = 16,
    seed_candidates: int = 6,
    swap_rounds: int = 2,
    node_bonus: float = 0.05,
) -> dict[LogicalVertex, HardwareVertex]:
    """
    Map logical variables to hardware qubits using a weighted frontier heuristic.

    Parameters
    ----------
    constraint_matrix
        Constraint matrix ``A`` of the BLP. Nonzero co-occurrence in a row is
        treated as evidence that two logical variables would benefit from being
        adjacent in the mapped topology.
    hardware_vertices, hardware_edges
        Vertices and undirected edges of the hardware graph. D-Wave graphs from
        ``dwave_networkx`` can be passed via ``graph.nodes`` and ``graph.edges``.
    logical_vertices
        Optional labels for the logical variables. If omitted, the logical
        variables are labeled ``0, 1, ..., n - 1``.
    num_restarts
        Number of seed-pair restarts tried by the heuristic.
    seed_candidates
        Number of top logical and hardware seed candidates considered.
    swap_rounds
        Number of pairwise local-improvement passes after the initial greedy
        placement.
    node_bonus
        Small regularization term that prefers mapping more active logical
        variables onto more central hardware qubits.

    Returns
    -------
    dict
        Injective logical-to-hardware placement.
    """
    if num_restarts <= 0:
        raise ValueError("num_restarts must be positive")
    if seed_candidates <= 0:
        raise ValueError("seed_candidates must be positive")
    if swap_rounds < 0:
        raise ValueError("swap_rounds must be nonnegative")
    if node_bonus < 0.0:
        raise ValueError("node_bonus must be nonnegative")

    num_columns = _num_columns(constraint_matrix)
    if logical_vertices is None:
        logical_list = list(range(num_columns))
    else:
        logical_list = _unique_preserving_order(
            logical_vertices
        )
        if len(logical_list) != num_columns:
            raise ValueError(
                "the number of logical vertices must equal the number of columns in the constraint matrix"
            )

    hardware_list = _unique_preserving_order(
        hardware_vertices
    )
    if not logical_list:
        return {}
    if len(hardware_list) < len(logical_list):
        raise ValueError(
            "hardware graph has fewer vertices than the logical graph"
        )

    logical_order = {
        vertex: idx
        for idx, vertex in enumerate(logical_list)
    }
    hardware_order = {
        vertex: idx
        for idx, vertex in enumerate(hardware_list)
    }
    hardware_adj = _build_hardware_graph(
        hardware_list, hardware_edges
    )

    component = _largest_connected_component(
        hardware_list, hardware_adj, hardware_order
    )
    if len(component) < len(logical_list):
        raise ValueError(
            "the largest connected component of the hardware graph is too small for an injective assignment"
        )

    (
        logical_adj,
        logical_edge_weights,
        logical_vertex_score,
        node_activity,
    ) = _logical_graph_from_constraints(
        constraint_matrix, logical_list, logical_order
    )
    hardware_vertex_score = _hardware_vertex_scores(
        component, hardware_adj
    )

    logical_seed_candidates = _top_logical_seeds(
        logical_list,
        logical_vertex_score,
        node_activity,
        logical_order,
        limit=min(seed_candidates, len(logical_list)),
    )
    hardware_seed_candidates = _top_hardware_seeds(
        component,
        hardware_adj,
        hardware_vertex_score,
        hardware_order,
        limit=min(seed_candidates, len(component)),
    )
    seed_pairs = _seed_pairs(
        logical_seed_candidates,
        hardware_seed_candidates,
        num_restarts=num_restarts,
    )

    best_mapping: (
        dict[LogicalVertex, HardwareVertex] | None
    ) = None
    best_score: tuple[float, tuple[int, ...]] | None = None

    for logical_seed, hardware_seed in seed_pairs:
        placement = _greedy_frontier_mapping(
            logical_list=logical_list,
            component=component,
            hardware_adj=hardware_adj,
            logical_adj=logical_adj,
            logical_vertex_score=logical_vertex_score,
            node_activity=node_activity,
            hardware_vertex_score=hardware_vertex_score,
            logical_order=logical_order,
            hardware_order=hardware_order,
            logical_seed=logical_seed,
            hardware_seed=hardware_seed,
        )
        placement = _improve_placement_by_swaps(
            placement,
            logical_list=logical_list,
            logical_adj=logical_adj,
            node_activity=node_activity,
            hardware_adj=hardware_adj,
            hardware_vertex_score=hardware_vertex_score,
            swap_rounds=swap_rounds,
            node_bonus=node_bonus,
        )
        objective = _placement_objective(
            placement,
            logical_edge_weights=logical_edge_weights,
            node_activity=node_activity,
            hardware_adj=hardware_adj,
            hardware_vertex_score=hardware_vertex_score,
            node_bonus=node_bonus,
        )
        tie_break = tuple(
            hardware_order[placement[logical]]
            for logical in logical_list
        )
        score = (
            objective,
            tuple(-value for value in tie_break),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_mapping = placement

    if best_mapping is None:
        raise RuntimeError(
            "failed to construct a logical-to-hardware placement"
        )
    return best_mapping


def simple_topology_aware_mapping(
    constraint_matrix: Sequence[Sequence[float]],
    hardware_vertices: Iterable[HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
    logical_vertices: Iterable[LogicalVertex] | None = None,
) -> dict[LogicalVertex, HardwareVertex]:
    """
    Backwards-compatible entry point for the improved mapping heuristic.

    The name is kept for compatibility with the older experiments, but the
    implementation now uses the weighted multi-start frontier placement above.
    """
    return heuristic_topology_aware_mapping(
        constraint_matrix=constraint_matrix,
        hardware_vertices=hardware_vertices,
        hardware_edges=hardware_edges,
        logical_vertices=logical_vertices,
    )


def logical_couplings_from_placement(
    placement: Mapping[LogicalVertex, HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
    logical_vertices: Iterable[LogicalVertex] | None = None,
) -> list[tuple[LogicalVertex, LogicalVertex]]:
    """
    Return logical pairs whose mapped hardware qubits are directly coupled.

    Parameters
    ----------
    placement
        Injective logical-to-hardware map.
    hardware_edges
        Undirected couplers of the hardware graph.
    logical_vertices
        Optional logical ordering used to canonicalize returned pairs.
    """
    if logical_vertices is None:
        logical_list = list(placement.keys())
    else:
        logical_list = _unique_preserving_order(
            logical_vertices
        )

    logical_order = {
        vertex: idx
        for idx, vertex in enumerate(logical_list)
    }
    # Build a reverse lookup so edge filtering stays O(|E_hardware|).
    reverse_placement = {
        hardware: logical
        for logical, hardware in placement.items()
    }
    couplings: list[tuple[LogicalVertex, LogicalVertex]] = (
        []
    )
    seen: set[tuple[LogicalVertex, LogicalVertex]] = set()

    for hardware_u, hardware_v in hardware_edges:
        if (
            hardware_u not in reverse_placement
            or hardware_v not in reverse_placement
        ):
            continue
        logical_u = reverse_placement[hardware_u]
        logical_v = reverse_placement[hardware_v]
        if logical_u == logical_v:
            continue

        pair = _canonical_edge(
            logical_u, logical_v, logical_order
        )
        if pair not in seen:
            seen.add(pair)
            couplings.append(pair)
    return couplings


def logical_topology_from_placement(
    placement: Mapping[LogicalVertex, HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
    logical_vertices: Iterable[LogicalVertex] | None = None,
) -> HardwareTopology:
    """
    Convert a logical-to-hardware placement into the projection topology.

    The returned topology is indexed in the order given by ``logical_vertices``
    (or the insertion order of ``placement`` if omitted).
    """
    if logical_vertices is None:
        logical_list = list(placement.keys())
    else:
        logical_list = _unique_preserving_order(
            logical_vertices
        )

    if set(logical_list) != set(placement.keys()):
        raise ValueError(
            "logical_vertices must match the keys of placement"
        )

    logical_order = {
        vertex: idx
        for idx, vertex in enumerate(logical_list)
    }
    logical_edges = logical_couplings_from_placement(
        placement,
        hardware_edges,
        logical_vertices=logical_list,
    )
    indexed_edges = [
        (logical_order[u], logical_order[v])
        for u, v in logical_edges
    ]
    return HardwareTopology(
        len(logical_list), indexed_edges
    )


def mapped_logical_topology(
    constraint_matrix: Sequence[Sequence[float]],
    hardware_vertices: Iterable[HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
    logical_vertices: Iterable[LogicalVertex] | None = None,
    **mapping_kwargs,
) -> tuple[
    dict[LogicalVertex, HardwareVertex], HardwareTopology
]:
    """
    Build both the logical-to-hardware placement and the induced topology.
    """
    placement = heuristic_topology_aware_mapping(
        constraint_matrix=constraint_matrix,
        hardware_vertices=hardware_vertices,
        hardware_edges=hardware_edges,
        logical_vertices=logical_vertices,
        **mapping_kwargs,
    )
    topology = logical_topology_from_placement(
        placement,
        hardware_edges,
        logical_vertices=logical_vertices,
    )
    return placement, topology


def mapped_logical_topology_from_graph(
    constraint_matrix: Sequence[Sequence[float]],
    hardware_graph,
    logical_vertices: Iterable[LogicalVertex] | None = None,
    **mapping_kwargs,
) -> tuple[
    dict[LogicalVertex, HardwareVertex], HardwareTopology
]:
    """Graph-object wrapper for :func:`mapped_logical_topology`."""
    hardware_vertices, hardware_edges = (
        _graph_vertices_edges(hardware_graph)
    )
    return mapped_logical_topology(
        constraint_matrix=constraint_matrix,
        hardware_vertices=hardware_vertices,
        hardware_edges=hardware_edges,
        logical_vertices=logical_vertices,
        **mapping_kwargs,
    )


def _num_columns(
    constraint_matrix: Sequence[Sequence[float]],
) -> int:
    if len(constraint_matrix) == 0:
        return 0
    num_columns = len(constraint_matrix[0])
    for row in constraint_matrix:
        if len(row) != num_columns:
            raise ValueError(
                "constraint matrix must be rectangular"
            )
    return num_columns


def _unique_preserving_order(
    vertices: Iterable[Hashable],
) -> list[Hashable]:
    seen: set[Hashable] = set()
    ordered: list[Hashable] = []
    for vertex in vertices:
        if vertex not in seen:
            seen.add(vertex)
            ordered.append(vertex)
    return ordered


def _graph_vertices_edges(
    graph,
) -> tuple[
    list[HardwareVertex],
    list[tuple[HardwareVertex, HardwareVertex]],
]:
    """Extract ``nodes`` and ``edges`` from a graph-like object."""
    try:
        hardware_vertices = list(graph.nodes)
        hardware_edges = list(graph.edges)
    except AttributeError as exc:
        raise TypeError(
            "hardware_graph must expose .nodes and .edges"
        ) from exc
    return hardware_vertices, hardware_edges


def _build_hardware_graph(
    hardware_vertices: list[HardwareVertex],
    hardware_edges: Iterable[
        tuple[HardwareVertex, HardwareVertex]
    ],
) -> dict[HardwareVertex, set[HardwareVertex]]:
    hardware_set = set(hardware_vertices)
    adjacency: dict[HardwareVertex, set[HardwareVertex]] = {
        vertex: set() for vertex in hardware_vertices
    }

    for u, v in hardware_edges:
        if u == v:
            continue
        if u not in hardware_set or v not in hardware_set:
            raise ValueError(
                "hardware edge references an unknown hardware vertex"
            )
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _connected_components(
    hardware_vertices: list[HardwareVertex],
    adjacency: dict[HardwareVertex, set[HardwareVertex]],
    hardware_order: dict[HardwareVertex, int],
) -> list[set[HardwareVertex]]:
    remaining = set(hardware_vertices)
    components: list[set[HardwareVertex]] = []

    for root in hardware_vertices:
        if root not in remaining:
            continue
        queue = deque([root])
        component = {root}
        remaining.remove(root)

        while queue:
            current = queue.popleft()
            neighbors = sorted(
                adjacency[current],
                key=lambda vertex: hardware_order[vertex],
            )
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _largest_connected_component(
    hardware_vertices: list[HardwareVertex],
    adjacency: dict[HardwareVertex, set[HardwareVertex]],
    hardware_order: dict[HardwareVertex, int],
) -> set[HardwareVertex]:
    components = _connected_components(
        hardware_vertices, adjacency, hardware_order
    )
    if not components:
        return set()
    components.sort(
        key=lambda component: (
            -len(component),
            min(
                hardware_order[vertex]
                for vertex in component
            ),
        )
    )
    return components[0]


def _logical_graph_from_constraints(
    constraint_matrix: Sequence[Sequence[float]],
    logical_list: list[LogicalVertex],
    logical_order: dict[LogicalVertex, int],
) -> tuple[
    dict[LogicalVertex, dict[LogicalVertex, float]],
    dict[tuple[LogicalVertex, LogicalVertex], float],
    dict[LogicalVertex, float],
    dict[LogicalVertex, float],
]:
    """Build a weighted logical graph from constraint co-occurrence."""
    node_activity = {vertex: 0.0 for vertex in logical_list}
    logical_adj = {vertex: {} for vertex in logical_list}
    logical_edge_weights: dict[
        tuple[LogicalVertex, LogicalVertex], float
    ] = {}

    for row in constraint_matrix:
        active = [
            (logical_list[idx], abs(float(value)))
            for idx, value in enumerate(row)
            if float(value) != 0.0
        ]
        for vertex, magnitude in active:
            node_activity[vertex] += magnitude
        for left_idx, (u, weight_u) in enumerate(active):
            for v, weight_v in active[left_idx + 1 :]:
                pair = _canonical_edge(u, v, logical_order)
                logical_edge_weights[pair] = (
                    logical_edge_weights.get(pair, 0.0)
                    + weight_u * weight_v
                )

    for (u, v), weight in logical_edge_weights.items():
        logical_adj[u][v] = weight
        logical_adj[v][u] = weight

    logical_vertex_score = {
        vertex: sum(logical_adj[vertex].values())
        + 0.25 * node_activity[vertex]
        for vertex in logical_list
    }
    return (
        logical_adj,
        logical_edge_weights,
        logical_vertex_score,
        node_activity,
    )


def _hardware_vertex_scores(
    component: set[HardwareVertex],
    hardware_adj: dict[HardwareVertex, set[HardwareVertex]],
) -> dict[HardwareVertex, float]:
    """Return a cheap centrality proxy on the chosen hardware component."""
    scores: dict[HardwareVertex, float] = {}
    for vertex in component:
        neighbors = hardware_adj[vertex] & component
        second_hop: set[HardwareVertex] = set()
        for neighbor in neighbors:
            second_hop.update(
                hardware_adj[neighbor] & component
            )
        second_hop.discard(vertex)
        scores[vertex] = float(
            len(neighbors) + 0.25 * len(second_hop)
        )
    return scores


def _top_logical_seeds(
    logical_list: list[LogicalVertex],
    logical_vertex_score: dict[LogicalVertex, float],
    node_activity: dict[LogicalVertex, float],
    logical_order: dict[LogicalVertex, int],
    limit: int,
) -> list[LogicalVertex]:
    return sorted(
        logical_list,
        key=lambda vertex: (
            -logical_vertex_score[vertex],
            -node_activity[vertex],
            logical_order[vertex],
        ),
    )[:limit]


def _top_hardware_seeds(
    component: set[HardwareVertex],
    hardware_adj: dict[HardwareVertex, set[HardwareVertex]],
    hardware_vertex_score: dict[HardwareVertex, float],
    hardware_order: dict[HardwareVertex, int],
    limit: int,
) -> list[HardwareVertex]:
    return sorted(
        component,
        key=lambda vertex: (
            -hardware_vertex_score[vertex],
            -len(hardware_adj[vertex] & component),
            hardware_order[vertex],
        ),
    )[:limit]


def _seed_pairs(
    logical_seeds: Sequence[LogicalVertex],
    hardware_seeds: Sequence[HardwareVertex],
    num_restarts: int,
) -> list[tuple[LogicalVertex, HardwareVertex]]:
    pairs: list[tuple[LogicalVertex, HardwareVertex]] = []
    for logical_seed in logical_seeds:
        for hardware_seed in hardware_seeds:
            pairs.append((logical_seed, hardware_seed))
            if len(pairs) >= num_restarts:
                return pairs
    return pairs


def _greedy_frontier_mapping(
    logical_list: list[LogicalVertex],
    component: set[HardwareVertex],
    hardware_adj: dict[HardwareVertex, set[HardwareVertex]],
    logical_adj: dict[
        LogicalVertex, dict[LogicalVertex, float]
    ],
    logical_vertex_score: dict[LogicalVertex, float],
    node_activity: dict[LogicalVertex, float],
    hardware_vertex_score: dict[HardwareVertex, float],
    logical_order: dict[LogicalVertex, int],
    hardware_order: dict[HardwareVertex, int],
    logical_seed: LogicalVertex,
    hardware_seed: HardwareVertex,
) -> dict[LogicalVertex, HardwareVertex]:
    placement = {logical_seed: hardware_seed}
    unmapped_logical = set(logical_list)
    unmapped_logical.remove(logical_seed)
    available_hardware = set(component)
    available_hardware.remove(hardware_seed)

    while unmapped_logical:
        logical_vertex = max(
            unmapped_logical,
            key=lambda vertex: (
                sum(
                    logical_adj[vertex].get(mapped, 0.0)
                    for mapped in placement
                ),
                logical_vertex_score[vertex],
                node_activity[vertex],
                -logical_order[vertex],
            ),
        )

        candidate_hardware = _hardware_frontier(
            placement.values(),
            available_hardware,
            hardware_adj,
        )
        if not candidate_hardware:
            candidate_hardware = set(available_hardware)

        hardware_vertex = max(
            candidate_hardware,
            key=lambda vertex: _hardware_candidate_key(
                logical_vertex=logical_vertex,
                hardware_vertex=vertex,
                placement=placement,
                available_hardware=available_hardware,
                logical_adj=logical_adj,
                hardware_adj=hardware_adj,
                hardware_vertex_score=hardware_vertex_score,
                hardware_order=hardware_order,
            ),
        )

        placement[logical_vertex] = hardware_vertex
        unmapped_logical.remove(logical_vertex)
        available_hardware.remove(hardware_vertex)

    return placement


def _hardware_frontier(
    mapped_hardware: Iterable[HardwareVertex],
    available_hardware: set[HardwareVertex],
    hardware_adj: dict[HardwareVertex, set[HardwareVertex]],
) -> set[HardwareVertex]:
    frontier: set[HardwareVertex] = set()
    for hardware_vertex in mapped_hardware:
        frontier.update(
            hardware_adj[hardware_vertex]
            & available_hardware
        )
    return frontier


def _hardware_candidate_key(
    logical_vertex: LogicalVertex,
    hardware_vertex: HardwareVertex,
    placement: Mapping[LogicalVertex, HardwareVertex],
    available_hardware: set[HardwareVertex],
    logical_adj: Mapping[
        LogicalVertex, Mapping[LogicalVertex, float]
    ],
    hardware_adj: Mapping[
        HardwareVertex, set[HardwareVertex]
    ],
    hardware_vertex_score: Mapping[HardwareVertex, float],
    hardware_order: Mapping[HardwareVertex, int],
) -> tuple[float, float, int, float, int]:
    direct_score = 0.0
    two_hop_score = 0.0

    for (
        mapped_logical,
        mapped_hardware,
    ) in placement.items():
        weight = logical_adj[logical_vertex].get(
            mapped_logical, 0.0
        )
        if weight == 0.0:
            continue
        if mapped_hardware in hardware_adj[hardware_vertex]:
            direct_score += weight
        elif (
            hardware_adj[hardware_vertex]
            & hardware_adj[mapped_hardware]
        ):
            two_hop_score += 0.25 * weight

    free_degree = len(
        hardware_adj[hardware_vertex] & available_hardware
    )
    return (
        direct_score,
        two_hop_score,
        free_degree,
        hardware_vertex_score[hardware_vertex],
        -hardware_order[hardware_vertex],
    )


def _placement_objective(
    placement: Mapping[LogicalVertex, HardwareVertex],
    logical_edge_weights: Mapping[
        tuple[LogicalVertex, LogicalVertex], float
    ],
    node_activity: Mapping[LogicalVertex, float],
    hardware_adj: Mapping[
        HardwareVertex, set[HardwareVertex]
    ],
    hardware_vertex_score: Mapping[HardwareVertex, float],
    node_bonus: float,
) -> float:
    """Score a placement by preserved logical edge weight plus a small node bonus."""
    score = 0.0
    for (
        logical_u,
        logical_v,
    ), weight in logical_edge_weights.items():
        if (
            placement[logical_v]
            in hardware_adj[placement[logical_u]]
        ):
            score += weight

    if node_bonus > 0.0 and placement:
        max_activity = max(
            node_activity.values(), default=1.0
        )
        max_hardware_score = max(
            hardware_vertex_score.values(), default=1.0
        )
        for (
            logical_vertex,
            hardware_vertex,
        ) in placement.items():
            score += (
                node_bonus
                * (
                    node_activity[logical_vertex]
                    / max_activity
                )
                * (
                    hardware_vertex_score[hardware_vertex]
                    / max_hardware_score
                )
            )
    return score


def _improve_placement_by_swaps(
    placement: dict[LogicalVertex, HardwareVertex],
    logical_list: Sequence[LogicalVertex],
    logical_adj: Mapping[
        LogicalVertex, Mapping[LogicalVertex, float]
    ],
    node_activity: Mapping[LogicalVertex, float],
    hardware_adj: Mapping[
        HardwareVertex, set[HardwareVertex]
    ],
    hardware_vertex_score: Mapping[HardwareVertex, float],
    swap_rounds: int,
    node_bonus: float,
) -> dict[LogicalVertex, HardwareVertex]:
    """Run a small greedy swap pass to improve the preserved logical edge weight."""
    if swap_rounds == 0 or len(placement) <= 1:
        return dict(placement)

    improved = dict(placement)
    max_activity = max(node_activity.values(), default=1.0)
    max_hardware_score = max(
        hardware_vertex_score.values(), default=1.0
    )

    def node_term(
        logical_vertex: LogicalVertex,
        hardware_vertex: HardwareVertex,
    ) -> float:
        if node_bonus == 0.0:
            return 0.0
        return (
            node_bonus
            * (node_activity[logical_vertex] / max_activity)
            * (
                hardware_vertex_score[hardware_vertex]
                / max_hardware_score
            )
        )

    for _ in range(swap_rounds):
        best_pair: (
            tuple[LogicalVertex, LogicalVertex] | None
        ) = None
        best_delta = 0.0

        for left_idx, logical_u in enumerate(logical_list):
            for logical_v in logical_list[left_idx + 1 :]:
                hardware_u = improved[logical_u]
                hardware_v = improved[logical_v]
                if hardware_u == hardware_v:
                    continue

                delta = (
                    node_term(logical_u, hardware_v)
                    + node_term(logical_v, hardware_u)
                    - node_term(logical_u, hardware_u)
                    - node_term(logical_v, hardware_v)
                )

                for logical_nbr, weight in logical_adj[
                    logical_u
                ].items():
                    if logical_nbr == logical_v:
                        continue
                    hardware_nbr = improved[logical_nbr]
                    old_connected = (
                        1.0
                        if hardware_nbr
                        in hardware_adj[hardware_u]
                        else 0.0
                    )
                    new_connected = (
                        1.0
                        if hardware_nbr
                        in hardware_adj[hardware_v]
                        else 0.0
                    )
                    delta += weight * (
                        new_connected - old_connected
                    )

                for logical_nbr, weight in logical_adj[
                    logical_v
                ].items():
                    if logical_nbr == logical_u:
                        continue
                    hardware_nbr = improved[logical_nbr]
                    old_connected = (
                        1.0
                        if hardware_nbr
                        in hardware_adj[hardware_v]
                        else 0.0
                    )
                    new_connected = (
                        1.0
                        if hardware_nbr
                        in hardware_adj[hardware_u]
                        else 0.0
                    )
                    delta += weight * (
                        new_connected - old_connected
                    )

                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_pair = (logical_u, logical_v)

        if best_pair is None:
            break

        logical_u, logical_v = best_pair
        improved[logical_u], improved[logical_v] = (
            improved[logical_v],
            improved[logical_u],
        )

    return improved


def _canonical_edge(
    u: Hashable,
    v: Hashable,
    order: Mapping[Hashable, int],
) -> tuple[Hashable, Hashable]:
    if u not in order or v not in order:
        raise ValueError(
            "edge references a vertex that is missing from the chosen ordering"
        )
    return (u, v) if order[u] <= order[v] else (v, u)


__all__ = [
    "heuristic_topology_aware_mapping",
    "logical_couplings_from_placement",
    "logical_topology_from_placement",
    "mapped_logical_topology",
    "mapped_logical_topology_from_graph",
    "simple_topology_aware_mapping",
]
