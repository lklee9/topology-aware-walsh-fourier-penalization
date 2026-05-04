# blp-qubo

Research code for hardware-aware QUBO reformulations of binary linear
programs, with a focus on Walsh/Fourier projection, unbalanced
penalization, and topology-aware embedding studies.

## Repository layout

- `fourier_projection/` — reusable projection library
- `experiments/` — active experiment runners, analyses, and utilities
- `plots/` — active paper-facing figure and table generation
- `brainstorm/` — paper source and published artifacts
- `docs/paper_artifacts.md` — current paper artifact inventory
- `experiments/archive/`, `plots/archive/` — retired exploratory code

Only descriptive module names are supported. Legacy numbered wrappers and
other compatibility shims were removed in the hard-cutover cleanup pass.

## Installation

### Minimal library-only install

If you only want the Fourier projection package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[fourier]
```

Optional PyTorch backend:

```bash
pip install -e .[fourier,torch]
```

### Full paper/experiment environment

For the full experiment stack, plotting, and benchmark tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick checks

Package import:

```bash
python -c "import fourier_projection; print(fourier_projection.__version__)"
```

Paper artifact input smoke check:

```bash
python -m plots.paper_artifacts --check
```

Tests:

```bash
pytest -q
```

Formatting/linting:

```bash
isort . && black . && flake8 .
```

## Reproducing paper artifacts

The published figures and tables used by `brainstorm/main.tex` are
tracked in `docs/paper_artifacts.md`.

Typical paper-facing commands are:

```bash
python -m plots.plot_measures
python -m plots.plot_slack_dist
python -m plots.compare_full_vs_unbalanced_mean
python -m plots.compare_before_after_mean
python -m plots.compare_up_projection_before_after
python -m plots.export_compare_up_projection_cop_table
python -m plots.export_compare_qpu_gap_sample_stats_table
```

The QPU sample-statistics exporter auto-discovers the newest analysed
D-Wave session under `experiments/results/dwave_bench/`; use
`--session-dir` to pin one specific run.

## Running experiments

See `experiments/README.md` for the experiment taxonomy, dataset
locations, primary entrypoints, and notes about optional dependencies
such as CPLEX and D-Wave credentials.

Run order (experiment-side workflows)

Run the experiment-side workflows in sequence; later steps produce artifacts consumed by the plotting/export scripts. See `experiments/README.md` for details and optional arguments.

1. Prepare instance manifest (if needed)

```bash
python -m experiments.manifests.create_instance_seed_manifest
```

2. Exact slack-measure definitions

```bash
python -m experiments.measure_definitions
```

3. Tune penalty multipliers

```bash
python -m experiments.tune_multipliers
```

4. Logical baseline comparisons

```bash
python -m experiments.compare_methods_baseline
```

5. Embedding comparisons (uses tuning summaries)

```bash
python -m experiments.compare_methods_embedding
```

6. Tuned UP-projection comparison

```bash
python -m experiments.compare_up_projection
```

7. (Optional) Run D-Wave benchmarks (requires Ocean credentials/hardware)

```bash
python -m experiments.dwave_bench
```

8. Analyze D-Wave benchmark outputs

```bash
python -m experiments.results.analyze_dwave_bench
```

9. Aggregate CoP tables (helper for rebuilding aggregate tables)

```bash
python -m experiments.results.aggregate_cop_tables
```


Regenerate paper figures & tables (paper/main.tex order)

After completing the experiment steps (or if you already have `experiments/results/`), run the following to rebuild the figures and LaTeX tables used by `brainstorm/main.tex` in the order they appear in the paper:

```bash
# Slack histogram (paper figure)
python -m plots.plot_measures

# Gram--Schmidt illustration (LaTeX source)
# from the project root:
cd brainstorm/figures && latexmk -pdf -quiet gram_schmidt_projection.tex
# (or: pdflatex -interaction=nonstopmode gram_schmidt_projection.tex)

# Slack distribution figure (paper copy)
python -m plots.plot_slack_dist

# Heatmaps: full vs unbalanced (mean CoP before/after embed)
python -m plots.compare_full_vs_unbalanced_mean

# Heatmaps: topology vs unbalanced (mean CoP before/after embed)
python -m plots.compare_before_after_mean

# UP-projection heatmaps (multiple panels; paper uses the mean-CoP panel)
python -m plots.compare_up_projection_before_after

# Tables: UP-projection mean CoP / feasibility table
python -m plots.export_compare_up_projection_cop_table

# Tables: QPU sample-level objective-gap statistics (auto-discovers latest analysed D-Wave session)
python -m plots.export_compare_qpu_gap_sample_stats_table [--session-dir <session_dir>]
```

Notes:

- The QPU exporter auto-discovers the newest analysed session under `experiments/results/dwave_bench/`; pass `--session-dir` to pin a specific run.
- Many plotting entrypoints accept `--output-dir` and other arguments; run with `--help` to see options.

## Using `fourier_projection`

The installable library exports a stable public API from
`fourier_projection/__init__.py`, including:

- `BLP`
- `HardwareTopology`
- `ProjectionMeasures`
- `IdealPenalty`
- `HypercubeSampleEnumerator`
- `HardwarePenaltyProjection`
- `project_penalty_values`
- `project_penalty_values_importance`
- `project_blp_penalty`
- `project_blp_penalty_importance`

See `fourier_projection/README.md` and
`examples/fourier_projection/` for minimal usage examples.
