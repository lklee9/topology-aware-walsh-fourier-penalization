# fourier_projection

`fourier_projection` is the reusable library in this repository for
fitting hardware-restricted Walsh/Fourier penalty surrogates for binary
linear programs (BLPs).

## Installation

Minimal install:

```bash
pip install -e .[fourier]
```

Optional PyTorch backend:

```bash
pip install -e .[fourier,torch]
```

The package is designed so that users who only need the projection code
can avoid the full experiment and QPU stack.

## Public API

The stable package surface is exported from
`fourier_projection/__init__.py`.

Core objects:

- `BLP`
- `HardwareTopology`
- `ProjectionMeasures`
- `IdealPenalty`
- `HypercubeSampleEnumerator`
- `HardwarePenaltyProjection`
- `BinnedTargetSaddlepointIS`

Convenience wrappers:

- `project_penalty_values`
- `project_penalty_values_importance`
- `project_blp_penalty`
- `project_blp_penalty_importance`

Result containers:

- `ProjectedPenaltyFit`
- `ProjectedBLPPenalty`

## Quickstart

Exact projection on the full hypercube:

```python
import numpy as np

from fourier_projection import (
    BLP,
    HardwareTopology,
    HypercubeSampleEnumerator,
    IdealPenalty,
    project_penalty_values,
)

blp = BLP(
    c=np.zeros(3),
    A=[[1.0, 1.0, 0.0]],
    b=[1.0],
)
enum = HypercubeSampleEnumerator.full(blp.num_variables)
penalty = IdealPenalty.for_constraint(enum.Z, blp, 0, template="hinge")
fit = project_penalty_values(
    enum,
    HardwareTopology.full(blp.num_variables),
    penalty,
)

print(fit.linear)
print(fit.quadratic)
```

## Workflows

### 1. Exact / row-weighted fitting

Use `project_penalty_values(...)` when you either:

- enumerate the full hypercube exactly, or
- already have a fixed set of sampled rows plus explicit row weights.

The key inputs are:

- a `HypercubeSampleEnumerator` or sample-backed enumerator,
- a `HardwareTopology`,
- one vector of penalty values aligned with the enumerated rows, and
- optional row weights `mu`.

### 2. Proposal-based importance fitting

Use `project_penalty_values_importance(...)` when rows come from a
proposal distribution and you want self-normalized importance weights.

Typical pattern:

1. define a proposal / target law,
2. draw samples,
3. evaluate the target penalty on those samples, and
4. fit with `importance_law=` or `importance_weights=`.

### 3. Mixed-constraint BLP projection

Use `project_blp_penalty(...)` or
`project_blp_penalty_importance(...)` to project the full BLP penalty by
handling inequalities and equalities constraint-by-constraint and then
summing the results.

## Target measures

`ProjectionMeasures` exposes the current target factories used by the
repo:

- `uniform_hypercube()`
- `uniform_slack(j)`
- `q0(j)` through `q4(j)`
- `equality_zero_centered(j)`

These replace older README references such as `uniform()` and
`uniform_log_unnormalized(...)`.

## Examples

Runnable examples live in `examples/fourier_projection/`:

- `minimal_exact_projection.py`
- `sampled_importance_projection.py`
- `mixed_constraint_blp.py`

Run them from the repository root, for example:

```bash
python examples/fourier_projection/minimal_exact_projection.py
```

## Troubleshooting

- If you see `ImportError: backend='torch' requires PyTorch`, install the
  optional `torch` extra or pass `backend="numpy"`.
- If you only need the library, prefer `pip install -e .[fourier]`
  instead of the full repository requirements.
- The full experiment environment still lives in `requirements.txt`
  because it includes many optional research dependencies.
