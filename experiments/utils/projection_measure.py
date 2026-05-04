"""Shared projection-measure catalog backed by slack-law sampling."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from fourier_projection.blp import BLP
from fourier_projection.measures import (
    CustomSlackTarget,
    ProjectionMeasures,
    SlackTargetDistribution,
    UniformSlackTarget,
)
from fourier_projection.sampling import (
    BinnedTargetSaddlepointIS,
)

MIXTURE_LAMS = (0.01,)
BASE_PROJECTION_MEASURE_NAMES = (
    "uniform",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
)
_LEGACY_MEASURE_ALIASES = {
    "bnd_mix": "q0",
    "q1_mix": "q1",
    "mid_mix": "q2",
    "q3_mix": "q3",
    "end_mix": "q4",
}
_PROPOSAL_PROBABILITY = 0.5
_MIN_INTERVAL_WIDTH = 1e-6
_MAX_IMPORTANCE_SUPPORT_ROUNDS = 16


@dataclass(frozen=True)
class ProjectionMeasureSpec:
    """Canonical description of one supported public measure name."""

    name: str
    components: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ProjectionSamplingCatalog:
    """Per-problem slack-law estimators used by the experiments."""

    canonical_name: str
    proposal_probs: np.ndarray
    inequality_estimators: tuple[
        BinnedTargetSaddlepointIS, ...
    ]
    equality_estimators: tuple[
        BinnedTargetSaddlepointIS, ...
    ]
    proposal_sampler: BinnedTargetSaddlepointIS

    def sample_proposal(
        self,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw Bernoulli-proposal states from the shared estimator."""
        X, _ = self.proposal_sampler.sample_proposal(
            int(sample_size),
            rng=rng,
        )
        return np.asarray(X, dtype=float)

    def inequality_importance_weights(
        self,
        sample_bits: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return one raw importance-ratio vector per inequality."""
        X = np.asarray(sample_bits, dtype=float)
        return tuple(
            np.asarray(
                estimator.law.importance_weights(X=X),
                dtype=float,
            )
            for estimator in self.inequality_estimators
        )

    def equality_importance_weights(
        self,
        sample_bits: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return one raw importance-ratio vector per equality."""
        X = np.asarray(sample_bits, dtype=float)
        return tuple(
            np.asarray(
                estimator.law.importance_weights(X=X),
                dtype=float,
            )
            for estimator in self.equality_estimators
        )


def _missing_positive_mass_constraints(
    weight_vectors: tuple[np.ndarray, ...],
) -> tuple[int, ...]:
    """Return inequality indices whose weights have zero total mass."""
    missing: list[int] = []
    for index, weights in enumerate(weight_vectors):
        vector = np.asarray(weights, dtype=float).reshape(
            -1
        )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "importance weights must be finite for every inequality"
            )
        if np.any(vector < 0.0):
            raise ValueError(
                "importance weights must be nonnegative for every inequality"
            )
        if float(vector.sum()) <= 0.0:
            missing.append(index)
    return tuple(missing)


def sample_projection_states_with_inequality_support(
    catalog: ProjectionSamplingCatalog,
    *,
    sample_size: int,
    rng: np.random.Generator,
    max_rounds: int = _MAX_IMPORTANCE_SUPPORT_ROUNDS,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Draw proposal states until every inequality has positive IS mass."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")

    if not catalog.inequality_estimators:
        return (
            np.asarray(
                catalog.sample_proposal(
                    sample_size=sample_size, rng=rng
                ),
                dtype=float,
            ),
            tuple(),
        )

    sample_batches: list[np.ndarray] = []
    weight_batches: list[list[np.ndarray]] = [
        [] for _ in catalog.inequality_estimators
    ]
    missing = tuple(
        range(len(catalog.inequality_estimators))
    )

    for _ in range(max_rounds):
        batch = np.asarray(
            catalog.sample_proposal(
                sample_size=sample_size, rng=rng
            ),
            dtype=float,
        )
        if batch.ndim != 2 or batch.shape[0] == 0:
            raise ValueError(
                "proposal sampler must return a non-empty 2D array"
            )
        batch_weights = (
            catalog.inequality_importance_weights(batch)
        )
        if len(batch_weights) != len(weight_batches):
            raise ValueError(
                "inequality_importance_weights returned the wrong number "
                "of constraint vectors"
            )

        sample_batches.append(batch)
        for index, weights in enumerate(batch_weights):
            vector = np.asarray(
                weights, dtype=float
            ).reshape(-1)
            if vector.shape[0] != batch.shape[0]:
                raise ValueError(
                    "importance weights must align with sampled rows"
                )
            weight_batches[index].append(vector)

        combined_weights = tuple(
            np.concatenate(parts).astype(float, copy=False)
            for parts in weight_batches
        )
        missing = _missing_positive_mass_constraints(
            combined_weights
        )
        if not missing:
            return (
                np.vstack(sample_batches),
                combined_weights,
            )

    raise ValueError(
        "failed to draw projection samples with positive importance mass "
        f"for inequalities {missing} under measure "
        f"'{catalog.canonical_name}' after {max_rounds} batch(es) "
        f"of size {sample_size}"
    )


def _canonical_name(
    base_name: str, lam: float | None = None
) -> str:
    """Return the stable external name used in CSV outputs and CLIs."""
    if lam is None:
        return base_name
    return f"{base_name}_mix_{float(lam):g}"


def build_projection_measure_specs(
    mixture_lams: tuple[float, ...] = MIXTURE_LAMS,
) -> tuple[ProjectionMeasureSpec, ...]:
    """Return the canonical measure catalog used throughout the repo."""
    specs: list[ProjectionMeasureSpec] = [
        ProjectionMeasureSpec(
            "uniform", (("uniform", 1.0),)
        )
    ]
    for base_name in ("q0", "q1", "q2", "q3", "q4"):
        specs.append(
            ProjectionMeasureSpec(
                base_name, ((base_name, 1.0),)
            )
        )
        for lam in mixture_lams:
            specs.append(
                ProjectionMeasureSpec(
                    _canonical_name(base_name, lam),
                    (
                        (base_name, 1.0 - float(lam)),
                        ("uniform", float(lam)),
                    ),
                )
            )
    return tuple(specs)


PROJECTION_MEASURE_SPECS = build_projection_measure_specs()
PROJECTION_MEASURE_SPECS_BY_NAME = {
    spec.name: spec for spec in PROJECTION_MEASURE_SPECS
}


def parse_projection_measure_name(
    name: str,
    *,
    legacy_default_lam: float = 0.01,
) -> ProjectionMeasureSpec:
    """Parse one canonical or legacy measure name into its mixture spec."""
    name = str(name).strip()
    if not name:
        raise ValueError(
            "projection measure name must not be empty"
        )
    if name in PROJECTION_MEASURE_SPECS_BY_NAME:
        return PROJECTION_MEASURE_SPECS_BY_NAME[name]

    if name in _LEGACY_MEASURE_ALIASES:
        base_name = _LEGACY_MEASURE_ALIASES[name]
        if base_name == "uniform":
            return PROJECTION_MEASURE_SPECS_BY_NAME[
                "uniform"
            ]
        canonical = _canonical_name(
            base_name, legacy_default_lam
        )
        return ProjectionMeasureSpec(
            canonical,
            (
                (
                    base_name,
                    1.0 - float(legacy_default_lam),
                ),
                ("uniform", float(legacy_default_lam)),
            ),
        )

    match = re.fullmatch(
        r"(q[0-4])_mix_([0-9.eE+-]+)", name
    )
    if match:
        base_name = match.group(1)
        lam = float(match.group(2))
        if not (0.0 <= lam <= 1.0):
            raise ValueError(
                "measure mixture weight must lie in [0, 1]"
            )
        return ProjectionMeasureSpec(
            _canonical_name(base_name, lam),
            (
                (base_name, 1.0 - lam),
                ("uniform", lam),
            ),
        )

    if name in BASE_PROJECTION_MEASURE_NAMES:
        return ProjectionMeasureSpec(name, ((name, 1.0),))

    raise ValueError(f"unknown projection measure: {name}")


def canonical_projection_measure_name(
    name: str,
    *,
    legacy_default_lam: float = 0.01,
) -> str:
    """Return the stable canonical name for one measure identifier."""
    return parse_projection_measure_name(
        name,
        legacy_default_lam=legacy_default_lam,
    ).name


def _default_proposal_probs(problem: BLP) -> np.ndarray:
    """Return the shared Bernoulli proposal for slack-law sampling."""
    return np.full(
        problem.num_variables,
        _PROPOSAL_PROBABILITY,
        dtype=float,
    )


def _component_target(
    problem: BLP,
    *,
    inequality_idx: int,
    base_name: str,
) -> SlackTargetDistribution:
    """Return one slack target for a single inequality."""
    measures = ProjectionMeasures(None, problem)
    if base_name == "uniform":
        return measures.uniform_slack(inequality_idx)
    if base_name == "q0":
        return measures.q0(inequality_idx)
    if base_name == "q1":
        return measures.q1(inequality_idx)
    if base_name == "q2":
        return measures.q2(inequality_idx)
    if base_name == "q3":
        return measures.q3(inequality_idx)
    if base_name == "q4":
        return measures.q4(inequality_idx)
    raise ValueError(
        f"unknown base projection measure: {base_name}"
    )


def _mixed_target(
    components: tuple[
        tuple[SlackTargetDistribution, float], ...
    ],
    *,
    a: np.ndarray,
    b: float,
    p: np.ndarray,
    name: str,
) -> SlackTargetDistribution:
    """Return one bin-mass mixture of already-parameterized slack targets."""

    def _mass_function(
        lower: np.ndarray,
        upper: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        masses = np.zeros_like(centers, dtype=float)
        for target, weight in components:
            masses += float(weight) * target.bin_masses(
                lower,
                upper,
                centers,
                a=a,
                b=b,
                p=p,
            )
        return masses

    return CustomSlackTarget(
        mass_function=_mass_function, name=name
    )


def _inequality_target(
    problem: BLP,
    *,
    inequality_idx: int,
    spec: ProjectionMeasureSpec,
    proposal_probs: np.ndarray,
) -> SlackTargetDistribution:
    """Return one mixed slack target for a single inequality."""
    coeffs = np.asarray(
        problem.A[inequality_idx], dtype=float
    )
    rhs = float(problem.b[inequality_idx])
    components = tuple(
        (
            _component_target(
                problem,
                inequality_idx=inequality_idx,
                base_name=base_name,
            ),
            float(component_weight),
        )
        for base_name, component_weight in spec.components
    )
    if len(components) == 1:
        return components[0][0]
    return _mixed_target(
        components,
        a=coeffs,
        b=rhs,
        p=proposal_probs,
        name=spec.name,
    )


def _build_estimator(
    *,
    a: np.ndarray,
    b: float,
    proposal_probs: np.ndarray,
    target: SlackTargetDistribution,
) -> BinnedTargetSaddlepointIS:
    """Build one slack-law estimator with the shared Bernoulli proposal."""
    return BinnedTargetSaddlepointIS.from_problem(
        a=np.asarray(a, dtype=float),
        b=float(b),
        p=np.asarray(proposal_probs, dtype=float),
        target=target,
        delta=None,
        origin=0.0,
    )


def build_equality_sampling_estimators(
    problem: BLP,
    *,
    proposal_probs: np.ndarray | None = None,
) -> tuple[BinnedTargetSaddlepointIS, ...]:
    """Return one equality-centered Gaussian slack estimator per equality."""
    proposal = (
        _default_proposal_probs(problem)
        if proposal_probs is None
        else np.asarray(proposal_probs, dtype=float)
    )
    measures = ProjectionMeasures(
        None, problem, proposal_probs=proposal
    )
    estimators: list[BinnedTargetSaddlepointIS] = []
    for equality_idx in range(problem.num_equalities):
        estimators.append(
            _build_estimator(
                a=np.asarray(
                    problem.D[equality_idx], dtype=float
                ),
                b=float(problem.e[equality_idx]),
                proposal_probs=proposal,
                target=measures.equality_zero_centered(
                    equality_idx
                ),
            )
        )
    return tuple(estimators)


def build_projection_sampling_catalog(
    problem: BLP,
    *,
    measure_name: str,
    legacy_default_lam: float = 0.01,
    proposal_probs: np.ndarray | None = None,
    include_equality_estimators: bool = True,
) -> ProjectionSamplingCatalog:
    """Return the slack-law estimators used for one experiment run."""
    proposal = (
        _default_proposal_probs(problem)
        if proposal_probs is None
        else np.asarray(proposal_probs, dtype=float)
    )
    spec = parse_projection_measure_name(
        measure_name,
        legacy_default_lam=legacy_default_lam,
    )

    inequality_estimators = tuple(
        _build_estimator(
            a=np.asarray(
                problem.A[inequality_idx], dtype=float
            ),
            b=float(problem.b[inequality_idx]),
            proposal_probs=proposal,
            target=_inequality_target(
                problem,
                inequality_idx=inequality_idx,
                spec=spec,
                proposal_probs=proposal,
            ),
        )
        for inequality_idx in range(
            problem.num_inequalities
        )
    )
    equality_estimators = (
        build_equality_sampling_estimators(
            problem,
            proposal_probs=proposal,
        )
        if include_equality_estimators
        else tuple()
    )

    if inequality_estimators:
        proposal_sampler = inequality_estimators[0]
    elif equality_estimators:
        proposal_sampler = equality_estimators[0]
    else:
        proposal_sampler = _build_estimator(
            a=np.ones(problem.num_variables, dtype=float),
            b=0.0,
            proposal_probs=proposal,
            target=UniformSlackTarget(
                low=0.0,
                high=max(
                    float(problem.num_variables),
                    _MIN_INTERVAL_WIDTH,
                ),
            ),
        )

    return ProjectionSamplingCatalog(
        canonical_name=spec.name,
        proposal_probs=proposal,
        inequality_estimators=inequality_estimators,
        equality_estimators=equality_estimators,
        proposal_sampler=proposal_sampler,
    )
