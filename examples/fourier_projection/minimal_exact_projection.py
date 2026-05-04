"""Minimal exact Fourier-projection example."""

from __future__ import annotations

import numpy as np

from fourier_projection import (
    BLP,
    HardwareTopology,
    HypercubeSampleEnumerator,
    IdealPenalty,
    project_penalty_values,
)


def main() -> None:
    """Project one inequality penalty on the full hypercube."""
    blp = BLP(
        c=np.zeros(3),
        A=[[1.0, 1.0, 0.0]],
        b=[1.0],
        name="toy_inequality",
    )
    enum = HypercubeSampleEnumerator.full(blp.num_variables)
    topology = HardwareTopology.full(blp.num_variables)

    target_penalty = IdealPenalty.for_constraint(
        enum.Z,
        blp,
        0,
        template="hinge",
    )
    fit = project_penalty_values(
        enum,
        topology,
        target_penalty,
    )

    print("linear:", fit.linear)
    print("quadratic:\n", fit.quadratic)
    print("constant:", fit.const)


if __name__ == "__main__":
    main()
