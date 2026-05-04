# Experiments

This directory contains the active experiment runners,
analyses, manifests, results helpers, and shared utilities.

Projected-penalty workflows now use one fixed projected
configuration downstream:

- projection measure: `q2`
- projection penalty template: `heaviside`
- projection selection mode: `fixed`

## Supported entrypoints

Use only the descriptive module names below.

| Purpose | Module |
| --- | --- |
| Exact slack-measure definitions | `experiments.measure_definitions` |
| Tune multipliers | `experiments.tune_multipliers` |
| Logical baseline comparison | `experiments.compare_methods_baseline` |
| Embedding comparison | `experiments.compare_methods_embedding` |
| Tuned UP-projection comparison | `experiments.compare_up_projection` |
| D-Wave benchmark runner | `experiments.dwave_bench` |
| D-Wave benchmark analysis | `experiments.results.analyze_dwave_bench` |
| Shared instance-manifest builder | `experiments.manifests.create_instance_seed_manifest` |
| CoP aggregation helper | `experiments.results.aggregate_cop_tables` |

Legacy numbered entrypoints and retired comparison scripts were
removed.

## Run order

Run the experiment-side workflows in the order listed below.
Later steps depend on artifacts produced by earlier steps.
Do not skip ahead unless you already have the required outputs.

1. `experiments.manifests.create_instance_seed_manifest`
   - helper for matched synthetic instance sets when needed
1. `experiments.measure_definitions`
   - exact slack-distribution reference study
2. `experiments.tune_multipliers`
   - writes the family-level tuning summaries consumed downstream
3. `experiments.compare_methods_baseline`
   - uses the tuning summaries from step 2
4. `experiments.compare_methods_embedding`
   - uses the tuning summaries from step 2
5. `experiments.compare_up_projection`
   - compares the tuned UP-projection variants
6. `experiments.dwave_bench`
   - runs the benchmark QPU sampling jobs
7. `experiments.results.analyze_dwave_bench`
   - analyzes raw outputs from step 6
9. `experiments.results.aggregate_cop_tables`
   - helper for rebuilding aggregate CoP tables from instance CSVs

## Dataset locations

- MIS benchmark instances: `data/mis_instances/`
- MDKP benchmark instances: `data/mdkp_instances/`
- Dataset loaders: `experiments/utils/data_loaders.py`
- Synthetic problem generators: `experiments/utils/problems.py`
- Synthetic benchmark reconstruction helpers:
  `experiments/utils/synthetic_bench.py`
- Shared benchmark conversion helpers:
  `experiments/utils/benchmark_data.py`

## Output conventions

- experiment results: `experiments/results/`
- tuning summaries: `experiments/tunings/`
- manifests: `experiments/manifests/`
- paper-facing figures/tables: generated from `plots/`

`experiments/` should only produce data artifacts consumed by
plotting or analysis code.

## Typical commands

```bash
python -m experiments.measure_definitions --help
python -m experiments.tune_multipliers --help
python -m experiments.compare_methods_baseline --help
python -m experiments.compare_methods_embedding --help
python -m experiments.compare_up_projection --help
python -m experiments.dwave_bench --help
python -m experiments.results.analyze_dwave_bench --help
python -m experiments.manifests.create_instance_seed_manifest --help
python -m experiments.results.aggregate_cop_tables --help
```

## Dependency notes

- `docplex` / `cplex` are required for some reference-solver
  paths.
- D-Wave workflows require Ocean credentials and hardware
  access.
- The full repository environment is still easiest to
  reproduce with `pip install -r requirements.txt`.
- If you only need the reusable projection code, install
  `pip install -e .[fourier]` instead.

## Archived code

Exploratory or retired experiment-side scripts live under
`experiments/archive/`.
