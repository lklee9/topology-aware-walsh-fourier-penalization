"""Mixed-constraint BLP projection example."""

from __future__ import annotations

import numpy as np

from fourier_projection import (
    BLP,
    HardwareTopology,
    HypercubeSampleEnumerator,
    project_blp_penalty,
)


def main() -> None:
    """Project one BLP with both equality and inequality constraints."""
    blp = BLP(
        c=np.array([1.0, 0.0, 0.5]),
        A=[[1.0, 1.0, 0.0]],
        b=[1.0],
        D=[[0.0, 1.0, 1.0]],
        e=[1.0],
        name="toy_mixed",
    )
    enum = HypercubeSampleEnumerator.full(blp.num_variables)
    topology = HardwareTopology.full(blp.num_variables)

    artifact = project_blp_penalty(
        blp,
        topology,
        fit_enum=enum,
    )

    print("total linear:", artifact.linear)
    print("total quadratic:\n", artifact.quadratic)
    print(
        "number of equality fits:", len(artifact.equalities)
    )
    print(
        "number of inequality fits:",
        len(artifact.inequalities),
    )


if __name__ == "__main__":
    main()
