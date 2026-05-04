"""Sampled importance-projection example."""

from __future__ import annotations

import numpy as np

from fourier_projection import (
    BLP,
    GaussianSlackTarget,
    HardwareTopology,
    HypercubeSampleEnumerator,
    IdealPenalty,
    project_penalty_values_importance,
)
from fourier_projection.sampling import (
    BinnedTargetSaddlepointIS,
)


def main() -> None:
    """Fit one inequality penalty from proposal samples and IS weights."""
    blp = BLP(
        c=np.zeros(3),
        A=[[1.0, 1.0, 0.0]],
        b=[0.0],
        name="toy_importance",
    )
    topology = HardwareTopology.full(blp.num_variables)
    proposal_probs = np.full(
        blp.num_variables, 0.5, dtype=float
    )
    law = BinnedTargetSaddlepointIS.from_problem(
        a=np.asarray(blp.A[0], dtype=float),
        b=float(blp.b[0]),
        p=proposal_probs,
        target=GaussianSlackTarget(center=0.0, scale=0.75),
    )

    rng = np.random.default_rng(7)
    sample_bits, _ = law.sample_proposal(4096, rng=rng)
    sample_enum = HypercubeSampleEnumerator(
        blp.num_variables, sample_bits
    )
    penalty_values = IdealPenalty.for_constraint(
        sample_enum.Z,
        blp,
        0,
        template="hinge",
    )

    fit = project_penalty_values_importance(
        sample_bits,
        topology,
        penalty_values,
        importance_law=law,
    )

    print("fit enum size:", fit.fit_enum_size)
    print("linear:", fit.linear)
    print("constant:", fit.const)


if __name__ == "__main__":
    main()
