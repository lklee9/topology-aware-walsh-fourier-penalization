"""Rebuild aggregate CoP summary tables from instance-level CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    import sys

    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from experiments.utils.cop_aggregation import (
    infer_mode,
    read_csv_rows,
    sanitize_output_dir_name,
    write_aggregate_outputs,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to one cop_instance_summary.csv file. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "baseline", "embedding"),
        default="auto",
        help="Aggregation schema to apply.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to each input's parent directory.",
    )
    parser.add_argument(
        "--write-legacy-mean-std",
        action="store_true",
        help=(
            "Also write the legacy mean/std "
            "cop_aggregate_summary.csv."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help=(
            "Number of bootstrap resamples used for robust CI bounds."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Base RNG seed for bootstrap CI estimation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone aggregation CLI."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    multiple_inputs = len(args.input) > 1
    for input_path in args.input:
        resolved_input = input_path.resolve()
        if not resolved_input.exists():
            raise FileNotFoundError(
                f"input CSV not found: {resolved_input}"
            )
        rows = read_csv_rows(resolved_input)
        mode = (
            infer_mode(rows)
            if args.mode == "auto"
            else str(args.mode)
        )
        output_dir = _resolve_output_dir(
            resolved_input,
            requested_output_dir=args.output_dir,
            multiple_inputs=multiple_inputs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        robust_path, legacy_path = write_aggregate_outputs(
            rows,
            output_dir=output_dir,
            mode=mode,
            bootstrap_resamples=int(
                args.bootstrap_resamples
            ),
            bootstrap_seed=int(args.bootstrap_seed),
            write_legacy_mean_std=bool(
                args.write_legacy_mean_std
            ),
        )
        print(f"wrote {robust_path}")
        if legacy_path is not None:
            print(f"wrote {legacy_path}")
    return 0


def _resolve_output_dir(
    input_path: Path,
    *,
    requested_output_dir: Path | None,
    multiple_inputs: bool,
) -> Path:
    if requested_output_dir is None:
        return input_path.parent
    if not multiple_inputs:
        return requested_output_dir.resolve()
    return (
        requested_output_dir.resolve()
        / sanitize_output_dir_name(input_path)
    )


if __name__ == "__main__":
    raise SystemExit(main())
