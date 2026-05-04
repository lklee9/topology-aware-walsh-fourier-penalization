"""Combinatorial problem families and format conversions used in experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from docplex.mp.model import Model

try:
    from fourier_projection.blp import BLP
except (
    ImportError
):  # pragma: no cover - supports direct script usage.
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fourier_projection.blp import BLP


def _as_2d_matrix(
    matrix: np.ndarray | list[list[float]], num_columns: int
) -> np.ndarray:
    """Return one constraint matrix with a fixed number of columns."""
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return np.zeros((0, num_columns), dtype=float)
    if arr.ndim != 2:
        raise ValueError("constraint matrix must be 2D")
    if arr.shape[1] != num_columns:
        raise ValueError(
            f"constraint matrix must have {num_columns} columns"
        )
    return arr


def _as_rhs_vector(
    values: np.ndarray | list[float], expected_length: int
) -> np.ndarray:
    """Return one right-hand-side vector."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if expected_length == 0:
        return np.zeros(0, dtype=float)
    if arr.shape[0] != expected_length:
        raise ValueError(
            f"rhs must have length {expected_length}"
        )
    return arr


def state_indices_to_bits(
    indices: np.ndarray,
    num_variables: int,
    *,
    dtype: np.dtype | type = float,
) -> np.ndarray:
    """Convert integer state indices into a binary matrix with MSB-first columns."""
    if num_variables <= 0:
        raise ValueError("num_variables must be positive")
    values = np.asarray(indices, dtype=np.uint64).reshape(
        -1
    )
    shifts = np.arange(
        num_variables - 1, -1, -1, dtype=np.uint64
    )
    bits = (
        (values[:, None] >> shifts[None, :]) & 1
    ).astype(dtype, copy=False)
    return bits


def iter_state_chunks(
    num_variables: int,
    *,
    chunk_size: int = 1 << 15,
    dtype: np.dtype | type = float,
):
    """
    Yield ``(start_index, bits)`` chunks across the full binary hypercube.

    The returned bit matrices are ordered lexicographically by their integer
    state index with the most-significant bit in column 0.
    """
    if num_variables <= 0:
        raise ValueError("num_variables must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    num_states = 1 << num_variables
    for start in range(0, num_states, chunk_size):
        stop = min(start + chunk_size, num_states)
        indices = np.arange(start, stop, dtype=np.uint64)
        yield start, state_indices_to_bits(
            indices, num_variables, dtype=dtype
        )


def _eliminate_fixed_variables(
    objective_linear: np.ndarray,
    objective_constant: float,
    equality_matrix: np.ndarray,
    equality_rhs: np.ndarray,
    inequality_matrix: np.ndarray,
    inequality_rhs: np.ndarray,
    variable_names: list[str],
    fixed_variables: dict[str, float] | None = None,
) -> tuple[
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, float],
]:
    """Substitute fixed binary variables into the linear objective and constraints."""
    if fixed_variables is None:
        return (
            np.asarray(
                objective_linear, dtype=float
            ).reshape(-1),
            float(objective_constant),
            np.asarray(equality_matrix, dtype=float),
            np.asarray(equality_rhs, dtype=float).reshape(
                -1
            ),
            np.asarray(inequality_matrix, dtype=float),
            np.asarray(inequality_rhs, dtype=float).reshape(
                -1
            ),
            list(variable_names),
            {},
        )

    obj = np.asarray(objective_linear, dtype=float).reshape(
        -1
    )
    if obj.shape[0] != len(variable_names):
        raise ValueError(
            "objective_linear and variable_names must have the same length"
        )

    name_to_index = {
        name: idx for idx, name in enumerate(variable_names)
    }
    fixed_index_to_value: dict[int, float] = {}
    for name, value in fixed_variables.items():
        if name not in name_to_index:
            raise ValueError(
                f"unknown fixed variable: {name}"
            )
        fixed_index_to_value[name_to_index[name]] = float(
            value
        )

    fixed_indices = sorted(fixed_index_to_value)
    free_indices = [
        idx
        for idx in range(len(variable_names))
        if idx not in fixed_index_to_value
    ]

    if fixed_indices:
        fixed_values = np.array(
            [
                fixed_index_to_value[idx]
                for idx in fixed_indices
            ],
            dtype=float,
        )
    else:
        fixed_values = np.zeros(0, dtype=float)

    const = float(objective_constant)
    if fixed_indices:
        const += float(obj[fixed_indices] @ fixed_values)
    reduced_objective = obj[free_indices]
    reduced_names = [
        variable_names[idx] for idx in free_indices
    ]

    def _reduce_constraints(
        matrix: np.ndarray, rhs: np.ndarray, sense: str
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(matrix, dtype=float)
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if matrix.shape[0] == 0:
            return np.zeros(
                (0, len(free_indices)), dtype=float
            ), np.zeros(0, dtype=float)

        reduced_rhs = rhs.copy()
        if fixed_indices:
            reduced_rhs -= (
                matrix[:, fixed_indices] @ fixed_values
            )
        reduced_matrix = matrix[:, free_indices]

        keep_rows: list[int] = []
        for row_idx in range(reduced_matrix.shape[0]):
            row = reduced_matrix[row_idx]
            if np.any(np.abs(row) > 1e-12):
                keep_rows.append(row_idx)
                continue

            rhs_value = float(reduced_rhs[row_idx])
            if sense == "eq":
                if abs(rhs_value) > 1e-9:
                    raise ValueError(
                        "fixed-variable substitution produced an inconsistent equality"
                    )
            else:
                if rhs_value < -1e-9:
                    raise ValueError(
                        "fixed-variable substitution produced an inconsistent inequality"
                    )

        if not keep_rows:
            return np.zeros(
                (0, len(free_indices)), dtype=float
            ), np.zeros(0, dtype=float)
        return (
            reduced_matrix[keep_rows],
            reduced_rhs[keep_rows],
        )

    reduced_eq_matrix, reduced_eq_rhs = _reduce_constraints(
        equality_matrix, equality_rhs, "eq"
    )
    reduced_ineq_matrix, reduced_ineq_rhs = (
        _reduce_constraints(
            inequality_matrix, inequality_rhs, "le"
        )
    )

    fixed_name_to_value = {
        variable_names[idx]: fixed_index_to_value[idx]
        for idx in fixed_indices
    }
    return (
        reduced_objective,
        const,
        reduced_eq_matrix,
        reduced_eq_rhs,
        reduced_ineq_matrix,
        reduced_ineq_rhs,
        reduced_names,
        fixed_name_to_value,
    )


def _docplex_linear_coefficients(
    expression: Any,
    *,
    variable_to_index: dict[Any, int],
    num_variables: int,
) -> tuple[np.ndarray, float]:
    """Return the linear coefficients and constant term of one docplex expression."""
    coefficients = np.zeros(num_variables, dtype=float)
    for variable, value in expression.iter_terms():
        index = variable_to_index.get(variable)
        if index is None:
            raise ValueError(
                f"expression references unknown variable {variable!r}"
            )
        coefficients[index] += float(value)
    return coefficients, float(expression.constant)


def docplex_model_to_blp(
    model: Any,
    *,
    name: str | None = None,
    fixed_variables: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BLP:
    """
    Convert one authors' docplex model into the repo's canonical problem form.

    The returned ``BLP`` is always a minimization problem in repo convention:

    - inequalities: ``A x - b >= 0``
    - equalities: ``D x - e = 0``
    """
    variables = sorted(
        model.iter_binary_vars(),
        key=lambda variable: variable.index,
    )
    num_variables = len(variables)
    variable_names = [
        variable.name for variable in variables
    ]
    variable_to_index = {
        variable: index
        for index, variable in enumerate(variables)
    }

    objective_linear, objective_constant = (
        _docplex_linear_coefficients(
            model.objective_expr,
            variable_to_index=variable_to_index,
            num_variables=num_variables,
        )
    )
    if model.is_maximized():
        objective_linear = -objective_linear
        objective_constant = -objective_constant

    equality_rows: list[np.ndarray] = []
    equality_rhs: list[float] = []
    inequality_rows: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    for constraint in model.iter_constraints():
        left_linear, left_constant = (
            _docplex_linear_coefficients(
                constraint.left_expr,
                variable_to_index=variable_to_index,
                num_variables=num_variables,
            )
        )
        right_linear, right_constant = (
            _docplex_linear_coefficients(
                constraint.right_expr,
                variable_to_index=variable_to_index,
                num_variables=num_variables,
            )
        )
        coefficients = left_linear - right_linear
        rhs = float(right_constant - left_constant)

        if constraint.sense_string == "EQ":
            equality_rows.append(coefficients)
            equality_rhs.append(rhs)
        elif constraint.sense_string == "LE":
            inequality_rows.append(coefficients)
            inequality_rhs.append(rhs)
        elif constraint.sense_string == "GE":
            inequality_rows.append(-coefficients)
            inequality_rhs.append(-rhs)
        else:
            raise ValueError(
                f"unsupported docplex constraint sense: {constraint.sense_string}"
            )

    equality_matrix = np.asarray(
        equality_rows, dtype=float
    ).reshape(-1, num_variables)
    equality_rhs_array = np.asarray(
        equality_rhs, dtype=float
    ).reshape(-1)
    inequality_matrix = np.asarray(
        inequality_rows, dtype=float
    ).reshape(-1, num_variables)
    inequality_rhs_array = np.asarray(
        inequality_rhs, dtype=float
    ).reshape(-1)

    (
        reduced_objective,
        reduced_constant,
        reduced_eq_matrix,
        reduced_eq_rhs,
        reduced_ineq_matrix,
        reduced_ineq_rhs,
        reduced_names,
        fixed_name_to_value,
    ) = _eliminate_fixed_variables(
        objective_linear=objective_linear,
        objective_constant=objective_constant,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs_array,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs_array,
        variable_names=variable_names,
        fixed_variables=fixed_variables,
    )

    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault(
        "docplex_model_name", model.name
    )
    if fixed_variables is not None:
        merged_metadata.setdefault(
            "original_variable_names", variable_names
        )
        merged_metadata["fixed_variables"] = (
            fixed_name_to_value
        )

    return BLP(
        c=reduced_objective,
        A=-reduced_ineq_matrix,
        b=-reduced_ineq_rhs,
        D=reduced_eq_matrix,
        e=reduced_eq_rhs,
        objective_constant=reduced_constant,
        name=name or model.name,
        variable_names=reduced_names,
        metadata=merged_metadata,
    )


def blp_to_docplex_model(
    problem: BLP,
    *,
    name: str | None = None,
) -> Model:
    """Convert one repo-convention BLP into a DOcplex linear model."""
    model = Model(name=name or problem.name)
    variables = [
        model.binary_var(name=variable_name)
        for variable_name in problem.variable_names
    ]

    objective = float(problem.objective_constant)
    objective += model.sum(
        float(coefficient) * variables[index]
        for index, coefficient in enumerate(
            problem.objective_linear
        )
        if abs(float(coefficient)) > 1e-12
    )
    model.minimize(objective)

    for row_index, (coefficients, rhs) in enumerate(
        zip(
            problem.D,
            problem.e,
            strict=True,
        )
    ):
        expression = model.sum(
            float(value) * variables[index]
            for index, value in enumerate(coefficients)
            if abs(float(value)) > 1e-12
        )
        model.add_constraint(
            expression == float(rhs),
            ctname=f"eq_{row_index}",
        )

    for row_index, (coefficients, rhs) in enumerate(
        zip(
            problem.A,
            problem.b,
            strict=True,
        )
    ):
        expression = model.sum(
            float(value) * variables[index]
            for index, value in enumerate(coefficients)
            if abs(float(value)) > 1e-12
        )
        model.add_constraint(
            expression >= float(rhs),
            ctname=f"ineq_{row_index}",
        )

    return model


def _rng_from_seed(seed: int) -> np.random.Generator:
    """Return one RNG created from a single integer seed."""
    return np.random.default_rng(int(seed))


def _sample_random_density_graph(
    num_nodes: int,
    *,
    target_density: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Sample one graph with an exact target number of random edges."""
    graph = nx.Graph()
    graph.add_nodes_from(range(1, num_nodes + 1))

    if num_nodes < 2 or target_density <= 0.0:
        return graph

    triu_rows, triu_cols = np.triu_indices(num_nodes, k=1)
    total_pairs = len(triu_rows)
    target_edges = int(
        round(float(target_density) * float(total_pairs))
    )

    if target_edges <= 0:
        return graph

    if target_edges >= total_pairs:
        selected_indices = np.arange(total_pairs, dtype=int)
    else:
        selected_indices = np.sort(
            rng.choice(
                total_pairs,
                size=target_edges,
                replace=False,
            )
        )

    selected_pairs = zip(
        triu_rows[selected_indices],
        triu_cols[selected_indices],
    )
    for i, j in selected_pairs:
        graph.add_edge(int(i + 1), int(j + 1))

    return graph


def _graph_edge_density(graph: nx.Graph) -> float:
    """Return the undirected edge density of one graph."""
    num_nodes = int(graph.number_of_nodes())
    if num_nodes < 2:
        return 0.0
    num_edges = int(graph.number_of_edges())
    return (
        2.0
        * float(num_edges)
        / float(num_nodes * (num_nodes - 1))
    )


def _mis_graph_to_blp(
    graph: nx.Graph,
    *,
    family: str = "mis",
    name: str,
    metadata: dict[str, Any],
) -> BLP:
    """Convert one MIS graph instance into the repo-convention BLP."""
    nodes = list(sorted(graph.nodes()))
    node_to_index = {
        node: index for index, node in enumerate(nodes)
    }
    edges = list(
        sorted(
            tuple(sorted(edge)) for edge in graph.edges()
        )
    )
    num_nodes = len(nodes)

    inequality_rows: list[np.ndarray] = []
    for u, v in edges:
        row = np.zeros(num_nodes, dtype=float)
        row[node_to_index[u]] = 1.0
        row[node_to_index[v]] = 1.0
        inequality_rows.append(row)

    inequality_matrix = np.asarray(
        inequality_rows, dtype=float
    ).reshape(-1, num_nodes)
    inequality_rhs = np.ones(len(edges), dtype=float)

    merged_metadata = dict(metadata)
    merged_metadata.update(
        {
            "problem_family": str(family),
            "num_nodes": int(num_nodes),
            "num_edges": int(len(edges)),
            "nodes": tuple(nodes),
            "edges": tuple(edges),
            "edge_density": _graph_edge_density(graph),
        }
    )

    return BLP(
        c=-np.ones(num_nodes, dtype=float),
        A=-inequality_matrix,
        b=-inequality_rhs,
        D=np.zeros((0, num_nodes), dtype=float),
        e=np.zeros(0, dtype=float),
        name=name,
        variable_names=[f"x_{node}" for node in nodes],
        metadata=merged_metadata,
    )


def sample_mis_problem(
    num_nodes: int,
    seed: int,
) -> BLP:
    """
    Sample one reproducible MIS instance from a target edge density.

    The density is sampled uniformly from 0.095 to 0.215 and undirected edges
    are added uniformly at random until that density is reached.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    rng = _rng_from_seed(seed)
    edge_density = float(rng.uniform(0.095, 0.215))
    graph = _sample_random_density_graph(
        num_nodes,
        target_density=edge_density,
        rng=rng,
    )

    return _mis_graph_to_blp(
        graph,
        family="mis",
        name=f"MIS_{num_nodes}",
        metadata={
            "graph_model": "random_density_graph",
            "seed": int(seed),
            "target_edge_density": float(edge_density),
        },
    )


def sample_mdkp_problem(
    num_items: int,
    seed: int,
) -> BLP:
    """
    Sample one reproducible MDKP instance from simple benchmark ranges.

    The generator uses ``num_items - 2`` constraints, profits from
    ``[22, 42000]``, weights from ``[0, 310]``, and capacities as random
    fractions of each row sum.
    """
    if num_items <= 0:
        raise ValueError("num_items must be positive")

    rng = _rng_from_seed(seed)
    if num_items < 3:
        raise ValueError(
            "num_items must be at least 3 for MDKP generation"
        )

    num_constraints = num_items - 2
    values = rng.integers(22, 42001, size=num_items).astype(
        float
    )
    weight_matrix = rng.integers(
        0,
        311,
        size=(num_constraints, num_items),
    ).astype(float)
    ratios = rng.uniform(0.4, 0.8, size=num_constraints)
    capacities = np.floor(
        ratios * np.sum(weight_matrix, axis=1)
    ).astype(float)
    capacities = np.maximum(capacities, 1.0)

    return BLP(
        c=-np.asarray(values, dtype=float),
        A=-np.asarray(weight_matrix, dtype=float),
        b=-capacities,
        D=np.zeros((0, num_items), dtype=float),
        e=np.zeros(0, dtype=float),
        name=f"MDKP_{num_items}",
        variable_names=[
            f"x_{index}" for index in range(num_items)
        ],
        metadata={
            "problem_family": "mdkp",
            "num_items": int(num_items),
            "num_constraints": int(num_constraints),
            "seed": int(seed),
            "values": np.asarray(values, dtype=float),
            "weights": np.asarray(
                weight_matrix, dtype=float
            ),
            "weight_matrix": np.asarray(
                weight_matrix, dtype=float
            ),
            "capacities": capacities,
            "capacity_ratios": ratios,
        },
    )


__all__ = [
    "BLP",
    "blp_to_docplex_model",
    "docplex_model_to_blp",
    "iter_state_chunks",
    "sample_mis_problem",
    "sample_mdkp_problem",
    "state_indices_to_bits",
]
