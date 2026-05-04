# Plots

This directory is the active home for paper-facing figures and table
exports.

## Active paper entrypoints

| Artifact(s) | Entry point | Main inputs |
| --- | --- | --- |
| `brainstorm/figures/slack-uniform.pdf` | `python -m plots.plot_measures` | `data/mdkp_instances/pet3.dat` |
| `brainstorm/figures/slack-dist.pdf` | `python -m plots.plot_slack_dist` | seeded synthetic problem generation |
| `brainstorm/figures/full_vs_unbalanced_before_after_embedding_cop_heatmap_mean.pdf` | `python -m plots.compare_full_vs_unbalanced_mean` | baseline + embedding `cop_instance_summary.csv` |
| `brainstorm/figures/topology_vs_unbalanced_before_after_cop_heatmap_mean.pdf` | `python -m plots.compare_before_after_mean` | baseline + embedding `cop_instance_summary.csv` |
| `brainstorm/figures/topology_vs_unbalanced_before_after_cop_mean_heatmap_up_projection.pdf` and related UP-projection heatmaps | `python -m plots.compare_up_projection_before_after` | `compare_unb_pen_up_projection` logical + embedding summaries |
| `brainstorm/tabs/cop_mean_table.tex` | `python -m plots.export_compare_up_projection_cop_table` | UP-projection aggregate summaries |
| `brainstorm/tabs/qpu_gap_sample_stats_table.tex` | `python -m plots.export_compare_qpu_gap_sample_stats_table` | latest analysed D-Wave session, or `--session-dir` |

See `docs/paper_artifacts.md` for the full artifact inventory and exact
paths.

## Shared support modules

- `plots.compare_before_after`
- `plots.compare_full_vs_unbalanced`
- `plots.heatmap_layout`
- `plots.paper_artifacts`

## Archived scripts

Non-paper plotting scripts were moved out of the active root and now live
under `plots/archive/`.

## Style conventions

Heatmap sizing and shared styling guidance live in `plots/STYLE.md`.
