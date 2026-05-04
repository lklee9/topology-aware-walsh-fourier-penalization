"""
qa_simulator.py
===============
Reusable Ocean annealing simulation utilities for embedded QUBOs on D-Wave-style
hardware graphs.

The functions in this module are intentionally experiment-agnostic. They
construct hardware graphs, find minor embeddings, build the embedded physical
model, draw reads with either D-Wave's simulated quantum annealer or a cheap
random baseline, and decode those reads back to the logical variables while
tracking chain-break fractions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

try:
    import dimod
    import dwave_networkx as dnx
    import minorminer
    from dwave.embedding import (
        chain_breaks,
        embed_bqm,
        unembed_sampleset,
    )
    from dwave.embedding.chain_strength import (
        uniform_torque_compensation,
    )
    from dwave.samplers import PathIntegralAnnealingSampler
except (
    ImportError
) as exc:  # pragma: no cover - exercised only when deps missing.
    raise ImportError(
        "experiments.utils.qa_simulator requires dimod, dwave_networkx, "
        "minorminer, dwave.embedding, and dwave.samplers to be installed."
    ) from exc

try:
    from dwave.system.coupling_groups import (
        coupling_groups as _coupling_groups,
    )
except (
    ImportError
):  # pragma: no cover - optional helper for Zephyr scaling.
    _coupling_groups = None


warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)

ZERO_COUPLER_ATOL = 1e-12
DEFAULT_QPU_H_RANGE = (-4.0, 4.0)
DEFAULT_QPU_J_RANGE = (-1.0, 1.0)
DEFAULT_QPU_EXTENDED_J_RANGE = (-2.0, 1.0)
DEFAULT_PEGASUS_PER_QUBIT_COUPLING_RANGE = (-18.0, 15.0)
DEFAULT_ZEPHYR_PER_GROUP_COUPLING_RANGE = (-13.0, 10.0)


def _sanitize_logical_bqm(
    logical_bqm: dimod.BinaryQuadraticModel,
) -> dimod.BinaryQuadraticModel:
    """
    Drop numerically zero quadratic couplers while preserving all variables.

    Ocean's default chain-strength heuristic and minorminer both interpret the
    stored quadratic interactions as the problem graph. Explicit zero-valued
    couplers should not affect either the logical topology or the chosen chain
    strength, so we remove them before embedding/sampling.
    """
    if all(
        abs(float(bias)) > ZERO_COUPLER_ATOL
        for bias in logical_bqm.quadratic.values()
    ):
        return logical_bqm

    quadratic = {
        interaction: float(bias)
        for interaction, bias in logical_bqm.quadratic.items()
        if abs(float(bias)) > ZERO_COUPLER_ATOL
    }
    return dimod.BinaryQuadraticModel(
        dict(logical_bqm.linear),
        quadratic,
        logical_bqm.offset,
        logical_bqm.vartype,
    )


def default_qpu_solver_properties(
    hardware_family: str,
    *,
    hardware_size: int,
) -> dict[str, Any]:
    """Return documented QPU coefficient ranges for offline dry runs."""
    family = str(hardware_family).strip().lower()
    properties: dict[str, Any] = {
        "topology": {
            "type": family,
            "shape": [int(hardware_size)],
        },
        "h_range": list(DEFAULT_QPU_H_RANGE),
        "j_range": list(DEFAULT_QPU_J_RANGE),
        "extended_j_range": list(
            DEFAULT_QPU_EXTENDED_J_RANGE
        ),
    }
    if family == "pegasus":
        properties["per_qubit_coupling_range"] = list(
            DEFAULT_PEGASUS_PER_QUBIT_COUPLING_RANGE
        )
    elif family == "zephyr":
        properties["per_group_coupling_range"] = list(
            DEFAULT_ZEPHYR_PER_GROUP_COUPLING_RANGE
        )
    return properties


def _range_pair(
    value: Any,
) -> tuple[float, float] | None:
    """Return one finite numeric range pair."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        return None
    try:
        lower = float(value[0])
        upper = float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return lower, upper


def _positive_range_ratio(
    values: Sequence[float],
    value_range: tuple[float, float] | None,
) -> float:
    """Return the non-negative scale ratio induced by one solver range."""
    if value_range is None:
        return 0.0
    finite_values = np.asarray(list(values), dtype=float)
    if finite_values.size == 0:
        return 0.0
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("auto-scale values must be finite")

    lower, upper = value_range
    limit = 0.0
    if upper > 0.0:
        limit = max(
            limit, float(np.max(finite_values) / upper)
        )
    if lower < 0.0:
        limit = max(
            limit, float(np.min(finite_values) / lower)
        )
    return max(limit, 0.0)


def _quadratic_bias(
    bqm: dimod.BinaryQuadraticModel,
    u: Any,
    v: Any,
) -> float:
    """Return one quadratic bias, defaulting to zero when absent."""
    if u not in bqm.variables or v not in bqm.variables:
        return 0.0
    return float(bqm.get_quadratic(u, v, default=0.0))


def _per_qubit_coupling_limit(
    spin_bqm: dimod.BinaryQuadraticModel,
    coupling_range: tuple[float, float] | None,
) -> float:
    """Return the per-qubit coupling contribution to the scale factor."""
    if coupling_range is None:
        return 0.0
    totals: dict[Any, float] = {
        variable: 0.0 for variable in spin_bqm.variables
    }
    for (u, v), bias in spin_bqm.quadratic.items():
        coupling = float(bias)
        totals[u] = totals.get(u, 0.0) + coupling
        totals[v] = totals.get(v, 0.0) + coupling
    return _positive_range_ratio(
        tuple(totals.values()), coupling_range
    )


def _per_group_coupling_limit(
    spin_bqm: dimod.BinaryQuadraticModel,
    hardware_graph,
    coupling_range: tuple[float, float] | None,
) -> float:
    """Return the per-group coupling contribution to the scale factor."""
    if coupling_range is None:
        return 0.0
    if _coupling_groups is None:
        raise RuntimeError(
            "Zephyr auto-scale emulation requires dwave-system's "
            "coupling_groups helper"
        )
    group_totals: list[float] = []
    for group in _coupling_groups(hardware_graph):
        total = 0.0
        for u, v in group:
            total += _quadratic_bias(spin_bqm, u, v)
        group_totals.append(total)
    return _positive_range_ratio(
        group_totals, coupling_range
    )


def dwave_auto_scale_factor(
    bqm: dimod.BinaryQuadraticModel,
    *,
    hardware_graph,
    solver_properties: dict[str, Any] | None,
) -> float:
    """Return the D-Wave-style auto-scale divisor for one physical BQM."""
    properties = (
        {}
        if solver_properties is None
        else dict(solver_properties)
    )
    spin_bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    h_limit = _positive_range_ratio(
        [float(bias) for bias in spin_bqm.linear.values()],
        _range_pair(properties.get("h_range"))
        or DEFAULT_QPU_H_RANGE,
    )
    j_limit = _positive_range_ratio(
        [
            float(bias)
            for bias in spin_bqm.quadratic.values()
        ],
        _range_pair(properties.get("extended_j_range"))
        or _range_pair(properties.get("j_range"))
        or DEFAULT_QPU_EXTENDED_J_RANGE,
    )
    per_group_range = _range_pair(
        properties.get("per_group_coupling_range")
    )
    if per_group_range is not None:
        coupling_limit = _per_group_coupling_limit(
            spin_bqm,
            hardware_graph,
            per_group_range,
        )
    else:
        coupling_limit = _per_qubit_coupling_limit(
            spin_bqm,
            _range_pair(
                properties.get("per_qubit_coupling_range")
            ),
        )

    scale = max(h_limit, j_limit, coupling_limit)
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return float(scale)


def auto_scale_dwave_physical_bqm(
    bqm: dimod.BinaryQuadraticModel,
    *,
    hardware_graph,
    solver_properties: dict[str, Any] | None,
) -> tuple[dimod.BinaryQuadraticModel, float]:
    """Return one uniformly scaled physical BQM plus its scale factor."""
    factor = dwave_auto_scale_factor(
        bqm,
        hardware_graph=hardware_graph,
        solver_properties=solver_properties,
    )
    if np.isclose(factor, 1.0, atol=1e-12, rtol=0.0):
        return bqm, 1.0
    scaled_bqm = bqm.copy()
    scaled_bqm.scale(1.0 / factor)
    return scaled_bqm, float(factor)


@dataclass(frozen=True)
class ClassicalAnnealerSimulation:
    """Embedded annealer readout simulation for one sampler/chain-strength setting."""

    sampler_name: str
    decoder_name: str
    embedding: dict[Any, list[Any]]
    effective_chain_strength: float
    physical_bqm: dimod.BinaryQuadraticModel
    physical_sampleset: dimod.SampleSet
    decoded_sampleset: dimod.SampleSet
    chain_break_fraction: np.ndarray
    summary: Any | None = None
    auto_scale_factor: float = 1.0


@dataclass(frozen=True)
class QpuAnnealScheduleSpec:
    """One QPU-compatible anneal schedule shape."""

    schedule_id: str
    schedule_kind: str
    anneal_schedule: tuple[tuple[float, float], ...]
    total_time: float


def _validated_anneal_schedule(
    anneal_schedule: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Return one validated anneal schedule."""
    schedule_array = np.asarray(
        anneal_schedule, dtype=float
    )
    if (
        schedule_array.ndim != 2
        or schedule_array.shape[1] != 2
    ):
        raise ValueError(
            "anneal_schedule must have shape (n, 2)"
        )
    if schedule_array.shape[0] < 2:
        raise ValueError(
            "anneal_schedule must contain at least two points"
        )
    if not np.all(np.isfinite(schedule_array)):
        raise ValueError(
            "anneal_schedule must contain only finite values"
        )
    times = schedule_array[:, 0]
    anneal_fractions = schedule_array[:, 1]
    if np.any(np.diff(times) < 0.0):
        raise ValueError(
            "anneal_schedule times must be nondecreasing"
        )
    if times[0] != 0.0:
        raise ValueError(
            "anneal_schedule must start at time 0.0"
        )
    if times[-1] <= 0.0:
        raise ValueError(
            "anneal_schedule must end at a positive time"
        )
    if np.any(anneal_fractions < 0.0) or np.any(
        anneal_fractions > 1.0
    ):
        raise ValueError(
            "anneal_schedule anneal fractions must lie in [0, 1]"
        )
    if anneal_fractions[0] != 0.0:
        raise ValueError(
            "anneal_schedule must start at anneal fraction 0.0"
        )
    if anneal_fractions[-1] != 1.0:
        raise ValueError(
            "anneal_schedule must end at anneal fraction 1.0"
        )
    return tuple(
        (float(time), float(anneal_fraction))
        for time, anneal_fraction in schedule_array.tolist()
    )


def build_standard_anneal_schedule(
    total_time: float,
) -> tuple[tuple[float, float], ...]:
    """Return one standard forward anneal schedule."""
    duration = float(total_time)
    if duration <= 0.0:
        raise ValueError("total_time must be positive")
    return _validated_anneal_schedule(
        ((0.0, 0.0), (duration, 1.0))
    )


def build_pause_anneal_schedule(
    total_time: float,
    pause_fraction: float,
    pause_duration: float,
) -> tuple[tuple[float, float], ...]:
    """Return one forward anneal with a mid-anneal pause."""
    duration = float(total_time)
    pause_s = float(pause_fraction)
    pause_time = float(pause_duration)
    if duration <= 0.0:
        raise ValueError("total_time must be positive")
    if not 0.0 < pause_s < 1.0:
        raise ValueError(
            "pause_fraction must lie strictly between 0 and 1"
        )
    if pause_time < 0.0:
        raise ValueError(
            "pause_duration must be non-negative"
        )
    pause_start = duration * pause_s
    return _validated_anneal_schedule(
        (
            (0.0, 0.0),
            (pause_start, pause_s),
            (pause_start + pause_time, pause_s),
            (duration + pause_time, 1.0),
        )
    )


def build_quench_anneal_schedule(
    total_time: float,
    quench_fraction: float,
    quench_duration: float,
) -> tuple[tuple[float, float], ...]:
    """Return one forward anneal with a steep final ramp."""
    duration = float(total_time)
    quench_s = float(quench_fraction)
    quench_time = float(quench_duration)
    if duration <= 0.0:
        raise ValueError("total_time must be positive")
    if not 0.0 < quench_s < 1.0:
        raise ValueError(
            "quench_fraction must lie strictly between 0 and 1"
        )
    if quench_time <= 0.0:
        raise ValueError("quench_duration must be positive")
    quench_start = duration * quench_s
    return _validated_anneal_schedule(
        (
            (0.0, 0.0),
            (quench_start, quench_s),
            (quench_start + quench_time, 1.0),
        )
    )


def qpu_anneal_schedule_to_sqa_fields(
    anneal_schedule: Sequence[Sequence[float]],
    beta_scale: float,
    *,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one QPU-style anneal schedule to SQA custom fields."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    scale = float(beta_scale)
    if scale <= 0.0:
        raise ValueError("beta_scale must be positive")
    validated_schedule = _validated_anneal_schedule(
        anneal_schedule
    )
    schedule_array = np.asarray(
        validated_schedule, dtype=float
    )
    times = schedule_array[:, 0]
    anneal_fractions = schedule_array[:, 1]
    time_grid = np.linspace(
        float(times[0]),
        float(times[-1]),
        num_points,
        dtype=float,
    )
    anneal_fraction_grid = np.interp(
        time_grid, times, anneal_fractions
    )
    longitudinal_field = scale * anneal_fraction_grid
    transverse_field = scale * (1.0 - anneal_fraction_grid)
    return (
        np.asarray(longitudinal_field, dtype=float),
        np.asarray(transverse_field, dtype=float),
    )


def _chain_break_method(
    decoder_name: str,
    logical_bqm: dimod.BinaryQuadraticModel,
    embedding: dict[Any, list[Any]],
):
    """Resolve one supported Ocean chain-break decoder."""
    if decoder_name == "majority_vote":
        return chain_breaks.majority_vote
    raise ValueError(
        f"unknown chain-break decoder: {decoder_name}"
    )


@lru_cache(maxsize=None)
def build_dwave_graph(
    family: str,
    size: int,
):
    """Construct a D-Wave hardware graph from ``dwave_networkx``."""
    if size <= 0:
        raise ValueError("size must be positive")

    family = family.lower()
    if family == "chimera":
        return dnx.chimera_graph(size)
    if family == "pegasus":
        return dnx.pegasus_graph(size)
    if family == "zephyr":
        return dnx.zephyr_graph(size)
    raise ValueError(f"unknown hardware family: {family}")


def find_minor_embedding(
    logical_bqm: dimod.BinaryQuadraticModel,
    hardware_graph,
    random_seed: int | None = None,
) -> dict[Any, list[Any]]:
    """
    Find a minor embedding for the logical quadratic graph.

    Variables that do not participate in any quadratic term are assigned to
    unused hardware vertices after the minor embedding is found.
    """
    sanitized_bqm = _sanitize_logical_bqm(logical_bqm)
    source_edges = list(sanitized_bqm.quadratic)
    if source_edges:
        embedding = minorminer.find_embedding(
            source_edges,
            list(hardware_graph.edges),
            random_seed=random_seed,
        )
        if not embedding:
            raise ValueError(
                "minorminer failed to find an embedding on the chosen hardware graph"
            )
    else:
        embedding = {}

    embedding = {
        var: list(chain) for var, chain in embedding.items()
    }
    missing = [
        var
        for var in sanitized_bqm.variables
        if var not in embedding
    ]
    if not missing:
        return embedding

    used_vertices = {
        vertex
        for chain in embedding.values()
        for vertex in chain
    }
    available_vertices = [
        vertex
        for vertex in hardware_graph.nodes
        if vertex not in used_vertices
    ]
    if len(available_vertices) < len(missing):
        raise ValueError(
            "hardware graph does not have enough unused vertices for isolated logical variables"
        )

    for var, hardware_vertex in zip(
        missing, available_vertices
    ):
        embedding[var] = [hardware_vertex]
    return embedding


def sqa_dwave_annealer_samples(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    rng: np.random.Generator,
    *,
    num_sweeps: int | None = 1_000,
    num_sweeps_per_beta: int = 1,
    beta_range: tuple[float, float] | None = None,
    beta_schedule_type: str = "geometric",
    hp_field: Sequence[float] | np.ndarray | None = None,
    hd_field: Sequence[float] | np.ndarray | None = None,
) -> dimod.SampleSet:
    """Sample a physical BQM with Ocean's path-integral simulated annealer."""
    if num_reads <= 0:
        raise ValueError("num_reads must be positive")
    if num_sweeps_per_beta <= 0:
        raise ValueError(
            "num_sweeps_per_beta must be positive"
        )
    use_custom_schedule = hp_field is not None
    if use_custom_schedule:
        hp_array = np.asarray(hp_field, dtype=float)
        if hp_array.ndim != 1 or hp_array.size == 0:
            raise ValueError(
                "hp_field must be a non-empty 1D array"
            )
        if not np.all(np.isfinite(hp_array)):
            raise ValueError(
                "hp_field must contain only finite values"
            )
        if np.any(hp_array < 0.0):
            raise ValueError(
                "hp_field must be non-negative"
            )
        if hd_field is None:
            raise ValueError(
                "hd_field must be provided with hp_field"
            )
        hd_array = np.asarray(hd_field, dtype=float)
        if hd_array.shape != hp_array.shape:
            raise ValueError(
                "hd_field must match hp_field shape"
            )
        if not np.all(np.isfinite(hd_array)):
            raise ValueError(
                "hd_field must contain only finite values"
            )
        if np.any(hd_array < 0.0):
            raise ValueError(
                "hd_field must be non-negative"
            )
        if beta_range is not None:
            raise ValueError(
                "beta_range cannot be combined with hp_field"
            )
        sample_num_sweeps = (
            int(num_sweeps)
            if num_sweeps is not None
            else int(hp_array.size)
            * int(num_sweeps_per_beta)
        )
        if sample_num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")
        sample_beta_schedule_type = "custom"
    else:
        if hd_field is not None:
            raise ValueError(
                "hd_field cannot be provided without hp_field"
            )
        if num_sweeps is None or num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")
        sample_num_sweeps = int(num_sweeps)
        sample_beta_schedule_type = beta_schedule_type

    sampler = PathIntegralAnnealingSampler()
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    with np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        sample_kwargs: dict[str, object] = {
            "num_reads": num_reads,
            "num_sweeps": sample_num_sweeps,
            "num_sweeps_per_beta": num_sweeps_per_beta,
            "seed": seed,
        }
        if use_custom_schedule:
            sample_kwargs["beta_schedule_type"] = (
                sample_beta_schedule_type
            )
            sample_kwargs["Hp_field"] = hp_array
            sample_kwargs["Hd_field"] = hd_array
        else:
            sample_kwargs["beta_range"] = beta_range
            sample_kwargs["beta_schedule_type"] = (
                sample_beta_schedule_type
            )
        return sampler.sample(
            bqm,
            **sample_kwargs,
        )


def random_dwave_annealer_samples(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    rng: np.random.Generator,
) -> dimod.SampleSet:
    """Sample a physical BQM with independent random assignments."""
    if num_reads <= 0:
        raise ValueError("num_reads must be positive")

    num_variables = len(bqm.variables)
    if bqm.vartype is dimod.SPIN:
        random_states = rng.choice(
            np.asarray([-1, 1], dtype=np.int8),
            size=(num_reads, num_variables),
            replace=True,
        )
    elif bqm.vartype is dimod.BINARY:
        random_states = rng.integers(
            0,
            2,
            size=(num_reads, num_variables),
            dtype=np.int8,
        )
    else:
        raise ValueError(
            f"unsupported vartype: {bqm.vartype}"
        )

    return dimod.SampleSet.from_samples_bqm(
        (random_states, list(bqm.variables)),
        bqm,
    )


def dwave_annealer_samples(
    sampler_name: str,
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    rng: np.random.Generator,
    *,
    num_sweeps: int | None = 1_000,
    num_sweeps_per_beta: int = 1,
    beta_range: tuple[float, float] | None = None,
    beta_schedule_type: str = "geometric",
    hp_field: Sequence[float] | np.ndarray | None = None,
    hd_field: Sequence[float] | np.ndarray | None = None,
) -> dimod.SampleSet:
    """Dispatch to the supported Ocean annealing backend."""
    if sampler_name == "sqa":
        return sqa_dwave_annealer_samples(
            bqm,
            num_reads=num_reads,
            rng=rng,
            num_sweeps=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
            beta_range=beta_range,
            beta_schedule_type=beta_schedule_type,
            hp_field=hp_field,
            hd_field=hd_field,
        )
    if sampler_name == "random":
        return random_dwave_annealer_samples(
            bqm,
            num_reads=num_reads,
            rng=rng,
        )
    raise ValueError(
        "unknown annealing sampler: "
        f"{sampler_name}; supported values are 'sqa' and 'random'"
    )


def chain_break_fractions(
    physical_sampleset: dimod.SampleSet,
    embedding: dict[Any, list[Any]],
) -> np.ndarray:
    """Return the per-read fraction of logical variables whose chains are broken."""
    if not embedding:
        return np.zeros(
            len(physical_sampleset), dtype=float
        )

    variable_to_index = {
        var: idx
        for idx, var in enumerate(
            physical_sampleset.variables
        )
    }
    samples = np.asarray(physical_sampleset.record.sample)
    broken_counts = np.zeros(samples.shape[0], dtype=float)

    for chain in embedding.values():
        if len(chain) <= 1:
            continue
        chain_indices = [
            variable_to_index[node] for node in chain
        ]
        chain_values = samples[:, chain_indices]
        broken = np.any(
            chain_values != chain_values[:, [0]], axis=1
        )
        broken_counts += broken.astype(float)

    return broken_counts / max(len(embedding), 1)


def embedded_chain_to_problem_ratio(
    physical_bqm: dimod.BinaryQuadraticModel,
    embedding: dict[Any, list[Any]],
) -> float | None:
    """
    Return the ratio between chain-coupler scale and problem-coefficient scale.

    The numerator is the maximum absolute chain-coupler magnitude. The
    denominator is the maximum absolute non-chain physical coefficient across
    linear biases and non-chain quadratic couplers.
    """
    if not embedding:
        return None

    node_to_variable: dict[Any, Any] = {}
    for variable, chain in embedding.items():
        for node in chain:
            node_to_variable[node] = variable

    chain_abs: list[float] = []
    problem_abs: list[float] = [
        abs(float(bias))
        for bias in physical_bqm.linear.values()
    ]

    for (u, v), bias in physical_bqm.quadratic.items():
        abs_bias = abs(float(bias))
        if node_to_variable.get(u) == node_to_variable.get(
            v
        ):
            chain_abs.append(abs_bias)
        else:
            problem_abs.append(abs_bias)

    if not chain_abs:
        return None

    problem_scale = max(problem_abs, default=0.0)
    if problem_scale <= 1e-12:
        return None
    return max(chain_abs) / problem_scale


def decode_embedded_sampleset(
    physical_sampleset: dimod.SampleSet,
    embedding: dict[Any, list[Any]],
    logical_bqm: dimod.BinaryQuadraticModel,
    decoder_name: str,
) -> dimod.SampleSet:
    """Decode one embedded sampleset with a supported chain-break rule."""
    decoded_sampleset = unembed_sampleset(
        physical_sampleset,
        embedding,
        logical_bqm,
        chain_break_method=_chain_break_method(
            decoder_name, logical_bqm, embedding
        ),
        chain_break_fraction=True,
    )
    if len(decoded_sampleset) != len(physical_sampleset):
        raise RuntimeError(
            "decoded sampleset changed the number of reads unexpectedly"
        )
    return decoded_sampleset


def simulate_dwave_annealer_classically(
    logical_bqm: dimod.BinaryQuadraticModel,
    hardware_graph,
    chain_strength_multiplier: float | None,
    num_reads: int,
    rng: np.random.Generator,
    embedding: dict[Any, list[Any]] | None = None,
    sampler_name: str = "sqa",
    decoder_name: str = "majority_vote",
    num_sweeps: int | None = 1_000,
    num_sweeps_per_beta: int = 1,
    beta_range: tuple[float, float] | None = None,
    beta_schedule_type: str = "geometric",
    hp_field: Sequence[float] | np.ndarray | None = None,
    hd_field: Sequence[float] | np.ndarray | None = None,
    effective_chain_strength: float | None = None,
    solver_properties: dict[str, Any] | None = None,
    auto_scale: bool = False,
) -> ClassicalAnnealerSimulation:
    """
    Simulate an embedded D-Wave run with a supported classical sampler.

    The logical BQM is first minor-embedded into the chosen hardware graph.
    We then sample the embedded physical BQM with the requested classical
    backend. The resulting physical reads are decoded to obtain logical
    samples, and chain-break as well as solution-quality statistics are
    returned to the caller. Ocean's default
    ``uniform_torque_compensation`` rule is first computed for the logical
    BQM and embedding; when
    ``chain_strength_multiplier`` is provided, the effective chain strength is
    that default multiplied by the requested factor. The decoding rule is
    controlled by ``decoder_name``. Custom SQA schedules can be supplied
    through ``hp_field`` and ``hd_field`` when ``sampler_name="sqa"``.
    """
    sanitized_logical_bqm = _sanitize_logical_bqm(
        logical_bqm
    )
    if embedding is None:
        embedding = find_minor_embedding(
            sanitized_logical_bqm, hardware_graph
        )

    default_chain_strength = float(
        uniform_torque_compensation(
            sanitized_logical_bqm, embedding
        )
    )
    if effective_chain_strength is not None:
        effective_chain_strength = float(
            effective_chain_strength
        )
    elif chain_strength_multiplier is None:
        effective_chain_strength = default_chain_strength
    else:
        effective_chain_strength = (
            float(chain_strength_multiplier)
            * default_chain_strength
        )
    if effective_chain_strength <= 0.0:
        raise ValueError(
            "effective chain strength must be positive"
        )

    physical_bqm = embed_bqm(
        sanitized_logical_bqm,
        embedding=embedding,
        target_adjacency=hardware_graph.adj,
        chain_strength=effective_chain_strength,
    )
    auto_scale_factor = 1.0
    if auto_scale:
        physical_bqm, auto_scale_factor = (
            auto_scale_dwave_physical_bqm(
                physical_bqm,
                hardware_graph=hardware_graph,
                solver_properties=solver_properties,
            )
        )
        effective_chain_strength = float(
            effective_chain_strength
        ) / float(auto_scale_factor)
    physical_sampleset = dwave_annealer_samples(
        sampler_name,
        physical_bqm,
        num_reads=num_reads,
        rng=rng,
        num_sweeps=num_sweeps,
        num_sweeps_per_beta=num_sweeps_per_beta,
        beta_range=beta_range,
        beta_schedule_type=beta_schedule_type,
        hp_field=hp_field,
        hd_field=hd_field,
    )
    decoded_sampleset = decode_embedded_sampleset(
        physical_sampleset,
        embedding,
        logical_bqm,
        decoder_name=decoder_name,
    )

    return ClassicalAnnealerSimulation(
        sampler_name=sampler_name,
        decoder_name=decoder_name,
        embedding=embedding,
        effective_chain_strength=effective_chain_strength,
        physical_bqm=physical_bqm,
        physical_sampleset=physical_sampleset,
        decoded_sampleset=decoded_sampleset,
        chain_break_fraction=chain_break_fractions(
            physical_sampleset, embedding
        ),
        auto_scale_factor=float(auto_scale_factor),
    )


__all__ = [
    "ClassicalAnnealerSimulation",
    "QpuAnnealScheduleSpec",
    "auto_scale_dwave_physical_bqm",
    "build_pause_anneal_schedule",
    "build_quench_anneal_schedule",
    "build_standard_anneal_schedule",
    "build_dwave_graph",
    "chain_break_fractions",
    "default_qpu_solver_properties",
    "decode_embedded_sampleset",
    "dwave_annealer_samples",
    "dwave_auto_scale_factor",
    "embedded_chain_to_problem_ratio",
    "find_minor_embedding",
    "qpu_anneal_schedule_to_sqa_fields",
    "random_dwave_annealer_samples",
    "simulate_dwave_annealer_classically",
    "sqa_dwave_annealer_samples",
]
