"""Helpers for incremental CSV output merges and metadata checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable


def _normalized_key(
    row: dict[str, object],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Return one normalized merge key for CSV-backed rows."""
    return tuple(
        "" if row[field] is None else str(row[field])
        for field in fields
    )


def metadata_sidecar_path(path: Path) -> Path:
    """Return the metadata sidecar path for one file or directory target."""
    resolved = Path(path)
    if resolved.suffix:
        return resolved.with_name(
            f"{resolved.stem}_metadata.json"
        )
    return resolved / "run_metadata.json"


def ensure_run_metadata(
    target: Path,
    metadata: dict[str, object],
    *,
    force: bool = False,
) -> Path:
    """Persist one run-metadata sidecar and enforce compatibility."""
    sidecar_path = metadata_sidecar_path(target)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.exists() and not force:
        with sidecar_path.open(
            "r", encoding="utf-8"
        ) as handle:
            existing = json.load(handle)
        if existing != metadata:
            raise RuntimeError(
                "existing outputs were generated with a different run setup: "
                f"{sidecar_path}. Rerun with --force to overwrite this output set."
            )
        return sidecar_path

    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata, handle, indent=2, sort_keys=True
        )
        handle.write("\n")
    return sidecar_path


def load_rows_csv(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into row dictionaries when it exists."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def merge_rows_by_keys(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """Merge rows by replacing any existing row with the same key."""
    merged: dict[tuple[str, ...], dict[str, object]] = {}
    order: list[tuple[str, ...]] = []
    for row in existing_rows:
        key = _normalized_key(row, key_fields)
        if key in merged:
            continue
        merged[key] = dict(row)
        order.append(key)
    for row in new_rows:
        key = _normalized_key(row, key_fields)
        if key not in merged:
            order.append(key)
        merged[key] = dict(row)
    return [merged[key] for key in order]


def replace_rows_by_blocks(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    block_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """Replace all existing rows whose block key appears in ``new_rows``."""
    replaced_blocks = {
        _normalized_key(row, block_fields)
        for row in new_rows
    }
    preserved_rows = [
        dict(row)
        for row in existing_rows
        if _normalized_key(row, block_fields)
        not in replaced_blocks
    ]
    preserved_rows.extend(dict(row) for row in new_rows)
    return preserved_rows


def merge_csv_rows(
    path: Path,
    new_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...] | None = None,
    block_fields: tuple[str, ...] | None = None,
    sort_key: (
        Callable[[dict[str, object]], object] | None
    ) = None,
) -> list[dict[str, object]]:
    """Load, merge, and return CSV rows for one path."""
    existing_rows = [
        dict(row) for row in load_rows_csv(path)
    ]
    if key_fields is not None and block_fields is not None:
        raise ValueError(
            "specify key_fields or block_fields, not both"
        )
    if key_fields is not None:
        merged_rows = merge_rows_by_keys(
            existing_rows,
            new_rows,
            key_fields=key_fields,
        )
    elif block_fields is not None:
        merged_rows = replace_rows_by_blocks(
            existing_rows,
            new_rows,
            block_fields=block_fields,
        )
    else:
        merged_rows = [dict(row) for row in new_rows]
    if sort_key is not None:
        merged_rows.sort(key=sort_key)
    return merged_rows
