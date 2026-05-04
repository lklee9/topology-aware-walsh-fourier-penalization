#!/usr/bin/env python3
"""Regenerate paper figures and tables in the README.md order.

This script runs the plotting and table-export entrypoints listed in
`README.md` in the order they appear in the paper build. It intentionally
invokes the package with the current Python interpreter so that virtual
environment activation is not required when calling via `python3`.

Usage:
  python3 scripts/regenerate_paper_figures.py [--run-latex] [--session-dir PATH]
      [--dry-run] [--continue-on-error]

Options:
  --run-latex           Run `latexmk -pdf -quiet gram_schmidt_projection.tex`
                        in brainstorm/figures. Disabled by default because many
                        environments do not have latexmk installed.
  --session-dir PATH    Pass this path to the QPU exporter
                        (plots.export_compare_qpu_gap_sample_stats_table).
  --dry-run             Print the commands without executing them.
  --continue-on-error   Continue executing later steps even if one fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-latex",
        action="store_true",
        help=(
            "Run latexmk for gram_schmidt_projection.tex in "
            "brainstorm/figures. Disabled by default."
        ),
    )
    p.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=(
            "Analysed D-Wave session directory to pass to the QPU exporter "
            "(plots.export_compare_qpu_gap_sample_stats_table)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue executing later steps even if one step fails. "
            "Default behaviour is to abort on first error."
        ),
    )
    return p


def _run_command(
    cmd: Iterable[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    cmd_list = [str(c) for c in cmd]
    print("\n==> Running: {}".format(" ".join(cmd_list)))
    if cwd is not None:
        print("    cwd:", str(cwd))
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd_list, cwd=(cwd or None), check=True)
        return int(res.returncode)
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed (exit {exc.returncode}): {' '.join(cmd_list)}"
        )
        if continue_on_error:
            return int(exc.returncode)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    python = sys.executable

    steps = [
        # Slack histogram (paper figure)
        ( [python, "-m", "plots.plot_measures"], REPO_ROOT ),

        # Gram--Schmidt illustration (LaTeX source)
        # user must opt-in with --run-latex
        (
            ["latexmk", "-pdf", "-quiet", "gram_schmidt_projection.tex"],
            REPO_ROOT / "brainstorm" / "figures",
        ),

        # Slack distribution figure
        ( [python, "-m", "plots.plot_slack_dist"], REPO_ROOT ),

        # Heatmaps: full vs unbalanced (mean CoP before/after embed)
        ( [python, "-m", "plots.compare_full_vs_unbalanced_mean"], REPO_ROOT ),

        # Heatmaps: topology vs unbalanced (mean CoP before/after embed)
        ( [python, "-m", "plots.compare_before_after_mean"], REPO_ROOT ),

        # UP-projection heatmaps (paper uses the mean-CoP panel)
        ( [python, "-m", "plots.compare_up_projection_before_after"], REPO_ROOT ),

        # Tables: UP-projection mean CoP / feasibility table
        ( [python, "-m", "plots.export_compare_up_projection_cop_table"], REPO_ROOT ),

        # Tables: QPU sample-level objective-gap statistics
        ( [python, "-m", "plots.export_compare_qpu_gap_sample_stats_table"], REPO_ROOT ),
    ]

    # Execute in order
    for cmd, cwd in steps:
        # latex step requires explicit opt-in
        if cmd and cmd[0] == "latexmk":
            if not args.run_latex:
                print("\n==> Skipping LaTeX step (pass --run-latex to enable)")
                continue
        # append session-dir to the QPU exporter when provided
        if (
            cmd[0] == python
            and len(cmd) >= 3
            and cmd[2] == "plots.export_compare_qpu_gap_sample_stats_table"
        ):
            if args.session_dir is not None:
                cmd = list(cmd) + ["--session-dir", str(args.session_dir)]

        try:
            _run_command(
                cmd,
                cwd=cwd,
                dry_run=bool(args.dry_run),
                continue_on_error=bool(args.continue_on_error),
            )
        except Exception as exc:  # pragma: no cover - orchestration script
            print("\nAborting due to failure:", exc)
            return 1

    print("\nAll requested steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
