"""Shared heatmap layout helpers.

These helpers size figures from the number of heatmap cells rather than
from per-script ad hoc width scales. The goal is to keep heatmap cell
dimensions visually consistent across plotting scripts.

Future heatmap scripts in ``plots/`` should import one of these helpers
instead of hard-coding figure widths and heights.

Example
-------
```python
from plots.heatmap_layout import heatmap_figsize_for_shape

matrix = np.full((len(blocks), len(columns)), np.nan)
fig, ax = plt.subplots(
    figsize=heatmap_figsize_for_shape(matrix.shape),
    constrained_layout=True,
)
```
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np

HEATMAP_CELL_WIDTH = 0.45
HEATMAP_CELL_HEIGHT = 0.175
HEATMAP_LEFT_PAD = 1.0
HEATMAP_RIGHT_PAD = 0.12
HEATMAP_TOP_PAD = 0.58
HEATMAP_BOTTOM_PAD = 0.52
HEATMAP_PANEL_GAP = 0.16
HEATMAP_COLORBAR_WIDTH = 0.22
HEATMAP_COLORBAR_PAD = 0.18
STANDARD_COLORBAR_FRACTION = 0.05
STANDARD_COLORBAR_PAD = 0.03
HEATMAP_BAD_COLOR = "#d9d9d9"
HEATMAP_FACE_COLOR = "#f5f5f5"
HEATMAP_GRID_COLOR = "#ffffff"
HEATMAP_ANNOTATION_COLOR = "#111111"
HEATMAP_TICK_FONTSIZE = 8
HEATMAP_ANNOTATION_FONTSIZE = 7
HEATMAP_PANEL_TITLE_FONTSIZE = 10
HEATMAP_TITLE_PAD = 5


def heatmap_figsize_for_cells(
    num_rows: int,
    num_cols_per_panel: int,
    *,
    num_panels: int = 1,
    include_colorbar: bool = True,
    cell_width_scale: float = 1.0,
) -> tuple[float, float]:
    """Return a figure size with approximately fixed cell dimensions."""
    total_cols = max(1, int(num_cols_per_panel)) * max(
        1, int(num_panels)
    )
    scaled_cell_width = HEATMAP_CELL_WIDTH * max(
        float(cell_width_scale), 0.1
    )
    width = (
        HEATMAP_LEFT_PAD
        + total_cols * scaled_cell_width
        + max(0, int(num_panels) - 1) * HEATMAP_PANEL_GAP
        + HEATMAP_RIGHT_PAD
    )
    if include_colorbar:
        width += (
            HEATMAP_COLORBAR_PAD + HEATMAP_COLORBAR_WIDTH
        )
    height = (
        HEATMAP_TOP_PAD
        + max(1, int(num_rows)) * HEATMAP_CELL_HEIGHT
        + HEATMAP_BOTTOM_PAD
    )
    return width, height


def heatmap_figsize_for_shape(
    shape: tuple[int, int],
    *,
    num_panels: int = 1,
    include_colorbar: bool = True,
    cell_width_scale: float = 1.0,
) -> tuple[float, float]:
    """Return a figure size directly from one heatmap matrix shape."""
    num_rows, num_cols = shape
    return heatmap_figsize_for_cells(
        num_rows,
        num_cols,
        num_panels=num_panels,
        include_colorbar=include_colorbar,
        cell_width_scale=cell_width_scale,
    )


def cop_heatmap_cmap() -> matplotlib.colors.Colormap:
    """Return the shared diverging colormap for CoP-style heatmaps.

    Positive values map to green, while negative values map to purple.
    """
    cmap = matplotlib.colormaps["PRGn"].copy()
    cmap.set_bad(HEATMAP_BAD_COLOR)
    return cmap


def add_heatmap_grid(
    axis,
    shape: tuple[int, int],
    *,
    color: str = HEATMAP_GRID_COLOR,
    linewidth: float = 1.2,
) -> None:
    """Draw one white grid over heatmap cells."""
    rows, cols = shape
    axis.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(
        which="minor",
        color=color,
        linestyle="-",
        linewidth=linewidth,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


def style_heatmap_xaxis(
    axis,
    labels: list[str],
    *,
    fontsize: float = HEATMAP_TICK_FONTSIZE,
    rotation: float = 0,
    ha: str = "center",
) -> None:
    """Apply the shared x-axis tick styling for heatmaps."""
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(
        labels,
        rotation=rotation,
        ha=ha,
        fontsize=fontsize,
    )


def style_heatmap_yaxis(
    axis,
    labels: list[str],
    *,
    show_labels: bool = True,
    fontsize: float = HEATMAP_TICK_FONTSIZE,
) -> None:
    """Apply the shared y-axis tick styling for heatmaps."""
    axis.set_yticks(np.arange(len(labels)))
    if show_labels:
        axis.set_yticklabels(labels, fontsize=fontsize)
        axis.tick_params(axis="y", labelsize=fontsize)
        return
    axis.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )


def cap_infinite_heatmap_values(
    matrix: np.ndarray,
    *,
    positive_cap: float,
    negative_cap: float | None = None,
) -> np.ndarray:
    """Return one heatmap matrix copy with infinities capped for colouring."""
    capped = np.asarray(matrix, dtype=float).copy()
    negative_limit = (
        -positive_cap
        if negative_cap is None
        else negative_cap
    )
    capped[np.isposinf(capped)] = positive_cap
    capped[np.isneginf(capped)] = negative_limit
    return capped


def heatmap_text_color(
    cmap,
    norm,
    value: float,
    *,
    dark_text: str = HEATMAP_ANNOTATION_COLOR,
    light_text: str = "white",
    luminance_threshold: float = 0.45,
) -> str:
    """Return one annotation color with contrast against a heatmap cell."""
    red, green, blue, alpha = cmap(norm(value))
    luminance = (
        0.2126 * red + 0.7152 * green + 0.0722 * blue
    )
    if alpha > 0 and luminance < luminance_threshold:
        return light_text
    return dark_text


def heatmap_text_color_white_default(
    norm,
    value: float,
    *,
    dark_text: str = HEATMAP_ANNOTATION_COLOR,
    light_text: str = "white",
    center_band: float = 0.30,
) -> str:
    """Return white text by default and dark text only near zero.

    The ``center_band`` is expressed in normalized color space, where
    diverging norms map zero close to ``0.5``.
    """
    position = _norm_scalar_position(norm, value)
    if (
        math.isfinite(position)
        and abs(position - 0.5) <= center_band
    ):
        return dark_text
    return light_text


def _norm_scalar_position(norm, value: float) -> float:
    """Return one normalized scalar position for a tick candidate."""
    normalized = norm(np.asarray([value], dtype=float))
    return float(np.asarray(normalized).reshape(-1)[0])


def _compact_colorbar_tick_label(value: float) -> str:
    """Return one compact numeric colorbar tick label."""
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    return f"{value:.3g}"


def _contains_close_value(
    values: list[float], candidate: float
) -> bool:
    """Return whether one numeric list already contains a close value."""
    return any(
        math.isclose(
            value, candidate, rel_tol=1e-9, abs_tol=1e-12
        )
        for value in values
    )


def _sym_log_tick_candidates(norm) -> list[float]:
    """Return readable major tick candidates for one SymLogNorm colorbar."""
    vmin = float(
        norm.vmin if norm.vmin is not None else -1.0
    )
    vmax = float(
        norm.vmax if norm.vmax is not None else 1.0
    )
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return []

    candidates: list[float] = []
    if vmin < 0.0:
        candidates.append(vmin)
    if vmin < 0.0 < vmax:
        candidates.append(0.0)
    if vmax > 0.0:
        candidates.append(vmax)

    max_magnitude = max(abs(vmin), abs(vmax))
    if max_magnitude <= 0.0:
        return sorted(candidates)

    linthresh = float(getattr(norm, "linthresh", 1.0))
    if not math.isfinite(linthresh) or linthresh <= 0.0:
        linthresh = 1.0
    base = float(getattr(norm, "base", 10.0))
    if not math.isfinite(base) or base <= 1.0:
        base = 10.0

    min_exponent = int(math.ceil(math.log(linthresh, base)))
    max_exponent = int(
        math.floor(math.log(max_magnitude, base))
    )
    for exponent in range(min_exponent, max_exponent + 1):
        magnitude = base**exponent
        if (
            magnitude <= 0.0
            or magnitude > max_magnitude * 1.0000001
        ):
            continue
        if vmin < 0.0:
            candidates.append(-magnitude)
        if vmax > 0.0:
            candidates.append(magnitude)

    unique_candidates: list[float] = []
    for candidate in sorted(candidates):
        if not _contains_close_value(
            unique_candidates, candidate
        ):
            unique_candidates.append(candidate)
    return unique_candidates


def _configure_sym_log_colorbar_ticks(
    colorbar,
    norm,
    *,
    max_ticks: int = 7,
    min_spacing: float = 0.11,
) -> None:
    """Apply sparse, non-overlapping ticks to one SymLogNorm colorbar."""
    candidates = _sym_log_tick_candidates(norm)
    if not candidates:
        return

    vmin = float(
        norm.vmin if norm.vmin is not None else -1.0
    )
    vmax = float(
        norm.vmax if norm.vmax is not None else 1.0
    )
    essential: list[float] = []
    if vmin < 0.0:
        essential.append(vmin)
    if vmin < 0.0 < vmax:
        essential.append(0.0)
    if vmax > 0.0:
        essential.append(vmax)

    selected: list[float] = []
    selected_positions: list[float] = []

    def try_add_tick(value: float) -> None:
        if len(selected) >= max_ticks:
            return
        if _contains_close_value(selected, value):
            return
        position = _norm_scalar_position(norm, value)
        if not math.isfinite(position):
            return
        if any(
            abs(position - other) < min_spacing
            for other in selected_positions
        ):
            return
        selected.append(value)
        selected_positions.append(position)

    for value in essential:
        try_add_tick(value)

    remaining = [
        value
        for value in candidates
        if not _contains_close_value(selected, value)
    ]
    remaining.sort(key=lambda value: (-abs(value), value))
    for value in remaining:
        try_add_tick(value)

    if not selected:
        return

    ticks = sorted(selected)
    colorbar.locator = matplotlib.ticker.FixedLocator(ticks)
    colorbar.formatter = matplotlib.ticker.FuncFormatter(
        lambda value, _position: _compact_colorbar_tick_label(
            float(value)
        )
    )
    colorbar.update_ticks()
    colorbar.ax.minorticks_off()


def add_standard_heatmap_colorbar(
    figure,
    image,
    *,
    ax,
    label: str,
    tick_fontsize: float = 8,
    fraction: float = STANDARD_COLORBAR_FRACTION,
    pad: float = STANDARD_COLORBAR_PAD,
    orientation: str = "vertical",
    extend: str | None = None,
):
    """Add one heatmap colorbar with the shared sizing/style.

    This matches the sizing used by ``plots/compare_before_after.py``.
    """
    kwargs = {
        "ax": ax,
        "fraction": fraction,
        "pad": pad,
        "orientation": orientation,
    }
    if extend is not None:
        kwargs["extend"] = extend
    colorbar = figure.colorbar(image, **kwargs)
    if isinstance(image.norm, matplotlib.colors.SymLogNorm):
        _configure_sym_log_colorbar_ticks(
            colorbar, image.norm
        )
    colorbar.ax.tick_params(labelsize=tick_fontsize)
    colorbar.set_label(label)
    return colorbar
