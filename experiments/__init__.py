"""Canonical experiment entrypoints.

Only the descriptive module names in this package are supported.
Legacy numbered wrappers and retired comparison scripts were removed.
"""

from __future__ import annotations

__all__ = [
    "compare_methods_baseline",
    "compare_methods_embedding",
    "compare_up_projection",
    "dwave_bench",
    "measure_definitions",
    "tune_multipliers",
]
