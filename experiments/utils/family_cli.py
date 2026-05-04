"""Shared family-selection CLI helpers for experiment drivers."""

from __future__ import annotations

import argparse

from experiments.experiment_config import (
    DEFAULT_MDKP_SIZES,
    DEFAULT_MIS_SIZES,
    FAMILY_ORDER,
)

DEFAULT_FAMILY_SIZES = {
    "mdkp": list(DEFAULT_MDKP_SIZES),
    "mis": list(DEFAULT_MIS_SIZES),
}

FAMILY_SIZE_ARGUMENTS = {
    "mdkp": "mdkp_sizes",
    "mis": "mis_sizes",
}


def _parse_unique_choice_list(
    text: str,
    *,
    valid_choices: tuple[str, ...],
    field_name: str,
) -> list[str]:
    """Parse one comma-separated unique choice list."""
    values: list[str] = []
    seen: set[str] = set()
    for item in str(text).split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value not in valid_choices:
            raise argparse.ArgumentTypeError(
                f"invalid {field_name} value: {value}"
            )
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(
            f"{field_name} must not be empty"
        )
    return values


def parse_families(text: str) -> list[str]:
    """Parse one comma-separated problem-family list."""
    return _parse_unique_choice_list(
        text,
        valid_choices=FAMILY_ORDER,
        field_name="families",
    )


def parse_size_list(text: str) -> list[int]:
    """Parse one comma-separated positive integer list."""
    values: list[int] = []
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer size: {token}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "sizes must be positive"
            )
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(
            "sizes must not be empty"
        )
    return values


def add_family_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_sizes: bool,
) -> argparse.ArgumentParser:
    """Attach shared family-selection flags to one parser."""
    parser.add_argument(
        "--families",
        type=parse_families,
        default=list(FAMILY_ORDER),
        help="Comma-separated family list, e.g. 'mdkp,mis'",
    )
    if not include_sizes:
        return parser

    parser.add_argument(
        "--mdkp-sizes",
        type=parse_size_list,
        default=list(DEFAULT_FAMILY_SIZES["mdkp"]),
        help="Comma-separated MDKP sizes",
    )
    parser.add_argument(
        "--mis-sizes",
        type=parse_size_list,
        default=list(DEFAULT_FAMILY_SIZES["mis"]),
        help="Comma-separated MIS sizes",
    )
    return parser


def selected_families_from_args(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Return the ordered family tuple requested on the CLI."""
    families = tuple(
        str(family) for family in args.families
    )
    return tuple(dict.fromkeys(families))


def selected_family_sizes_from_args(
    args: argparse.Namespace,
) -> dict[str, list[int]]:
    """Return size selections keyed by all known families."""
    selected = set(selected_families_from_args(args))
    family_sizes: dict[str, list[int]] = {}
    for family in FAMILY_ORDER:
        if family not in selected:
            family_sizes[family] = []
            continue
        argument_name = FAMILY_SIZE_ARGUMENTS[family]
        family_sizes[family] = list(
            getattr(args, argument_name)
        )
    return family_sizes
