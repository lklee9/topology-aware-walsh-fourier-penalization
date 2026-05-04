"""Paper artifact manifest and input smoke checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from plots.export_compare_qpu_gap_sample_stats_table import (
    DEFAULT_RESULTS_ROOT as DEFAULT_DWAVE_RESULTS_ROOT,
)
from plots.export_compare_qpu_gap_sample_stats_table import (
    _discover_qpu_catalogs,
    _latest_session_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compare_input_paths() -> tuple[Path, ...]:
    root = REPO_ROOT / "experiments" / "results"
    return (
        root
        / "unbalanced_penalization"
        / "cop_instance_summary.csv",
        root
        / "compare_embedding"
        / "sqa_chain_strength_fraction_1"
        / "cop_instance_summary.csv",
    )


def _compare_unb_pen_input_paths() -> tuple[Path, ...]:
    root = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "compare_unb_pen_up_projection"
    )
    return (
        root / "logical" / "cop_instance_summary.csv",
        root
        / "embedding"
        / "sqa_chain_strength_fraction_1"
        / "cop_instance_summary.csv",
        root / "logical" / "cop_aggregate_summary.csv",
        root
        / "embedding"
        / "sqa_chain_strength_fraction_1"
        / "cop_aggregate_summary.csv",
    )


def _latest_dwave_catalog_paths() -> tuple[Path, ...]:
    session_dir = _latest_session_dir(
        DEFAULT_DWAVE_RESULTS_ROOT
    )
    catalogs = _discover_qpu_catalogs(session_dir)
    return tuple(catalogs.values())


@dataclass(frozen=True)
class PaperArtifact:
    """One figure or table included in ``brainstorm/main.tex``."""

    output_path: Path
    generator: str
    command: str
    inputs: tuple[str, ...]
    notes: str = ""
    paper_critical: bool = True
    path_checks: tuple[Path, ...] = field(
        default_factory=tuple
    )
    dynamic_check: Callable[[], tuple[Path, ...]] | None = (
        None
    )

    def resolved_checks(self) -> tuple[Path, ...]:
        """Return concrete filesystem paths checked by ``--check``."""
        if self.dynamic_check is not None:
            return tuple(
                Path(path) for path in self.dynamic_check()
            )
        return tuple(
            Path(path) for path in self.path_checks
        )


ARTIFACTS = (
    PaperArtifact(
        output_path=REPO_ROOT
        / "brainstorm"
        / "figures"
        / "slack-uniform.pdf",
        generator="plots.plot_measures",
        command="python -m plots.plot_measures",
        inputs=("data/mdkp_instances/pet3.dat",),
        notes="Hard-coded MDKP instance used for the paper figure.",
        path_checks=(
            REPO_ROOT
            / "data"
            / "mdkp_instances"
            / "pet3.dat",
        ),
    ),
    PaperArtifact(
        output_path=REPO_ROOT
        / "brainstorm"
        / "figures"
        / "slack-dist.pdf",
        generator="plots.plot_slack_dist",
        command="python -m plots.plot_slack_dist",
        inputs=(
            "No external result files; generated from a seeded synthetic problem.",
        ),
        notes="Defaults to the seeded MDKP slack-distribution figure used in the paper.",
    ),
    PaperArtifact(
        output_path=(
            REPO_ROOT
            / "brainstorm"
            / "figures"
            / "full_vs_unbalanced_before_after_embedding_cop_heatmap_mean.pdf"
        ),
        generator="plots.compare_full_vs_unbalanced_mean",
        command="python -m plots.compare_full_vs_unbalanced_mean",
        inputs=tuple(
            str(path.relative_to(REPO_ROOT))
            for path in _compare_input_paths()
        ),
        path_checks=_compare_input_paths(),
    ),
    PaperArtifact(
        output_path=(
            REPO_ROOT
            / "brainstorm"
            / "figures"
            / "topology_vs_unbalanced_before_after_cop_heatmap_mean.pdf"
        ),
        generator="plots.compare_before_after_mean",
        command="python -m plots.compare_before_after_mean",
        inputs=tuple(
            str(path.relative_to(REPO_ROOT))
            for path in _compare_input_paths()
        ),
        path_checks=_compare_input_paths(),
    ),
    PaperArtifact(
        output_path=(
            REPO_ROOT
            / "brainstorm"
            / "figures"
            / "topology_vs_unbalanced_before_after_cop_mean_heatmap_up_projection.pdf"
        ),
        generator="plots.compare_up_projection_before_after",
        command="python -m plots.compare_up_projection_before_after",
        inputs=(
            "experiments/results/compare_unb_pen_up_projection/logical/cop_instance_summary.csv",
            "experiments/results/compare_unb_pen_up_projection/embedding/sqa_chain_strength_fraction_1/cop_instance_summary.csv",
        ),
        notes="The command emits multiple related heatmaps; the paper uses the mean CoP panel.",
        path_checks=(
            REPO_ROOT
            / "experiments"
            / "results"
            / "compare_unb_pen_up_projection"
            / "logical"
            / "cop_instance_summary.csv",
            REPO_ROOT
            / "experiments"
            / "results"
            / "compare_unb_pen_up_projection"
            / "embedding"
            / "sqa_chain_strength_fraction_1"
            / "cop_instance_summary.csv",
        ),
    ),
    PaperArtifact(
        output_path=REPO_ROOT
        / "brainstorm"
        / "tabs"
        / "cop_mean_table.tex",
        generator="plots.export_compare_up_projection_cop_table",
        command="python -m plots.export_compare_up_projection_cop_table",
        inputs=(
            "experiments/results/compare_unb_pen_up_projection/logical/cop_aggregate_summary.csv",
            "experiments/results/compare_unb_pen_up_projection/embedding/sqa_chain_strength_fraction_1/cop_aggregate_summary.csv",
        ),
        path_checks=(
            REPO_ROOT
            / "experiments"
            / "results"
            / "compare_unb_pen_up_projection"
            / "logical"
            / "cop_aggregate_summary.csv",
            REPO_ROOT
            / "experiments"
            / "results"
            / "compare_unb_pen_up_projection"
            / "embedding"
            / "sqa_chain_strength_fraction_1"
            / "cop_aggregate_summary.csv",
        ),
    ),
    PaperArtifact(
        output_path=(
            REPO_ROOT
            / "brainstorm"
            / "tabs"
            / "qpu_gap_sample_stats_table.tex"
        ),
        generator="plots.export_compare_qpu_gap_sample_stats_table",
        command="python -m plots.export_compare_qpu_gap_sample_stats_table",
        inputs=(
            "Latest analysed D-Wave session under experiments/results/dwave_bench/",
            "Advantage */run_metrics_catalog.csv",
            "Advantage2 */run_metrics_catalog.csv",
        ),
        notes="The exporter auto-discovers the newest analysed session unless --session-dir is supplied.",
        dynamic_check=_latest_dwave_catalog_paths,
    ),
)


def _check_artifact_inputs() -> list[str]:
    """Return one error string per missing required input."""
    errors: list[str] = []
    for artifact in ARTIFACTS:
        try:
            checks = artifact.resolved_checks()
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive CLI path.
            errors.append(f"{artifact.output_path}: {exc}")
            continue
        for path in checks:
            if not Path(path).exists():
                errors.append(
                    f"{artifact.output_path}: missing {Path(path)}"
                )
    return errors


def _print_manifest() -> None:
    """Print the current artifact manifest in a readable text format."""
    for artifact in ARTIFACTS:
        print(
            f"- {artifact.output_path.relative_to(REPO_ROOT)}"
        )
        print(f"  generator: {artifact.generator}")
        print(f"  command:   {artifact.command}")
        if artifact.inputs:
            print("  inputs:")
            for item in artifact.inputs:
                print(f"    - {item}")
        if artifact.notes:
            print(f"  notes:    {artifact.notes}")


def main(argv: list[str] | None = None) -> int:
    """Run the manifest listing or input smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that all required artifact inputs exist.",
    )
    args = parser.parse_args(argv)

    _print_manifest()
    if not args.check:
        return 0

    errors = _check_artifact_inputs()
    if errors:
        print("\nInput check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nAll required artifact inputs are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
