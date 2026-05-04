"""Shared progress helpers for the baseline-comparison workflow."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

_NESTED_PROGRESS_ENABLED = True
_ACTIVE_TUI = None


class _NullProgressBar:
    """Fallback progress bar used when ``tqdm`` is unavailable."""

    def __init__(self, iterable=None, **kwargs):
        del kwargs
        self._iterable = iterable

    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        del exc_type, exc, exc_tb
        return False

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, n: int = 1) -> None:
        del n

    def close(self) -> None:
        return None


def progress(iterable=None, **kwargs):
    """Return a ``tqdm`` progress bar when available."""
    if not _NESTED_PROGRESS_ENABLED or _tqdm is None:
        return _NullProgressBar(iterable=iterable, **kwargs)
    return _tqdm(iterable, **kwargs)


def log(message: str) -> None:
    """Emit one stable progress log line."""
    if _ACTIVE_TUI is not None:
        _ACTIVE_TUI.log(message)
        return
    print(
        f"[compare_methods_baseline] {message}",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class BaselineComparisonProgressTotals:
    """Aggregate totals shown in the baseline-comparison TUI."""

    tuning_jobs: int = 0
    component_builds: int = 0
    baseline_methods: int = 0
    instances: int = 0


@dataclass
class BaselineComparisonProgressState:
    """Current nested experiment context shown in the TUI."""

    stage: str = "initializing"
    activity: str | None = None
    family: str | None = None
    size: int | None = None
    instance_index: int | None = None
    total_instances: int | None = None
    method: str | None = None
    measure: str | None = None
    template: str | None = None
    candidate_index: int | None = None
    total_candidates: int | None = None
    detail: str | None = None
    optimizer_restart: int | None = None
    optimizer_total_restarts: int | None = None
    optimizer_evals: int = 0
    optimizer_best: float | None = None
    qaoa_done: int | None = None
    qaoa_total: int | None = None

    def merge(self, **kwargs: object) -> None:
        """Apply explicit field updates in place."""
        for field_name, value in kwargs.items():
            if hasattr(self, field_name):
                setattr(self, field_name, value)


class BaselineComparisonTui:
    """Rich live TUI tailored to the baseline-comparison workflow."""

    def __init__(
        self,
        *,
        totals: BaselineComparisonProgressTotals,
        stream=None,
    ) -> None:
        try:
            from rich.console import Console, Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.text import Text
        except ImportError as exc:
            raise RuntimeError(
                "baseline comparison TUI requested, but "
                "'rich' is not installed."
            ) from exc

        self._Console = Console
        self._Group = Group
        self._Live = Live
        self._Panel = Panel
        self._Progress = Progress
        self._BarColumn = BarColumn
        self._SpinnerColumn = SpinnerColumn
        self._TaskProgressColumn = TaskProgressColumn
        self._TextColumn = TextColumn
        self._TimeElapsedColumn = TimeElapsedColumn
        self._Text = Text
        self._stream = stream or sys.stderr
        self._console = self._Console(
            file=self._stream,
            force_terminal=True,
        )
        self._totals = totals
        self._status = BaselineComparisonProgressState()
        self._logs: deque[str] = deque(maxlen=12)
        self._progress = self._Progress(
            self._SpinnerColumn(),
            self._TextColumn(
                "[progress.description]{task.description}"
            ),
            self._BarColumn(bar_width=None),
            self._TaskProgressColumn(),
            self._TimeElapsedColumn(),
            expand=True,
            console=self._console,
        )
        self._live = self._Live(
            self._renderable(),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._tuning_task_id = self._progress.add_task(
            "Tuning jobs",
            total=max(1, self._totals.tuning_jobs),
        )
        self._component_task_id = self._progress.add_task(
            "Projection builds",
            total=max(1, self._totals.component_builds),
        )
        self._baseline_task_id = self._progress.add_task(
            "Baseline methods",
            total=max(1, self._totals.baseline_methods),
        )
        self._instance_task_id = self._progress.add_task(
            "Instances",
            total=max(1, self._totals.instances),
        )
        self._optimizer_task_id: int | None = None
        self._qaoa_task_id: int | None = None

    def start(self) -> None:
        """Start the live UI."""
        self._live.start()
        self._refresh()

    def close(self) -> None:
        """Stop the live UI."""
        self._live.stop()

    def update_status(self, **kwargs: object) -> None:
        """Update the visible experiment status."""
        self._status.merge(**kwargs)
        self._refresh()

    def advance_tuning_jobs(self, amount: int = 1) -> None:
        """Advance the tuning-job counter."""
        self._progress.advance(self._tuning_task_id, amount)
        self._refresh()

    def advance_component_builds(
        self, amount: int = 1
    ) -> None:
        """Advance the projected-component counter."""
        self._progress.advance(
            self._component_task_id, amount
        )
        self._refresh()

    def advance_baseline_methods(
        self, amount: int = 1
    ) -> None:
        """Advance the baseline-method counter."""
        self._progress.advance(
            self._baseline_task_id, amount
        )
        self._refresh()

    def advance_instances(self, amount: int = 1) -> None:
        """Advance the instance counter."""
        self._progress.advance(
            self._instance_task_id, amount
        )
        self._refresh()

    def begin_optimizer(
        self,
        *,
        total_restarts: int,
    ) -> None:
        """Show the active Nelder-Mead optimizer task."""
        self._status.optimizer_restart = 0
        self._status.optimizer_total_restarts = int(
            total_restarts
        )
        self._status.optimizer_evals = 0
        self._status.optimizer_best = None
        if self._optimizer_task_id is None:
            self._optimizer_task_id = (
                self._progress.add_task(
                    "Optimizer restarts",
                    total=max(1, int(total_restarts)),
                )
            )
        else:
            self._progress.reset(
                self._optimizer_task_id,
                total=max(1, int(total_restarts)),
                completed=0,
            )
        self._refresh()

    def set_optimizer_restart(
        self, restart_index: int
    ) -> None:
        """Update the active optimizer restart index."""
        self._status.optimizer_restart = int(restart_index)
        if self._optimizer_task_id is not None:
            self._progress.update(
                self._optimizer_task_id,
                completed=int(restart_index),
            )
        self._refresh()

    def record_optimizer_eval(
        self,
        *,
        best_value: float | None = None,
    ) -> None:
        """Update the active optimizer evaluation counters."""
        self._status.optimizer_evals += 1
        if best_value is not None:
            self._status.optimizer_best = float(best_value)
        self._refresh()

    def finish_optimizer(self) -> None:
        """Hide the optimizer task."""
        if self._optimizer_task_id is not None:
            self._progress.remove_task(
                self._optimizer_task_id
            )
            self._optimizer_task_id = None
        self._status.optimizer_restart = None
        self._status.optimizer_total_restarts = None
        self._status.optimizer_evals = 0
        self._status.optimizer_best = None
        self._refresh()

    def begin_qaoa_grid(self, *, total_points: int) -> None:
        """Show the active QAOA-grid task."""
        self._status.qaoa_done = 0
        self._status.qaoa_total = int(total_points)
        if self._qaoa_task_id is None:
            self._qaoa_task_id = self._progress.add_task(
                "QAOA grid",
                total=max(1, int(total_points)),
            )
        else:
            self._progress.reset(
                self._qaoa_task_id,
                total=max(1, int(total_points)),
                completed=0,
            )
        self._refresh()

    def advance_qaoa_grid(self, amount: int = 1) -> None:
        """Advance the active QAOA-grid task."""
        if self._qaoa_task_id is None:
            return
        self._progress.advance(self._qaoa_task_id, amount)
        self._status.qaoa_done = int(
            (self._status.qaoa_done or 0) + amount
        )
        self._refresh()

    def finish_qaoa_grid(self) -> None:
        """Hide the QAOA-grid task."""
        if self._qaoa_task_id is not None:
            self._progress.remove_task(self._qaoa_task_id)
            self._qaoa_task_id = None
        self._status.qaoa_done = None
        self._status.qaoa_total = None
        self._refresh()

    def log(self, message: str) -> None:
        """Append one event message to the live log panel."""
        self._logs.append(str(message))
        self._refresh()

    def _refresh(self) -> None:
        self._live.update(self._renderable(), refresh=True)

    def _renderable(self):
        lines = [
            f"Stage: {self._status.stage}",
        ]
        if self._status.activity is not None:
            lines.append(
                f"Activity: {self._status.activity}"
            )
        if self._status.family is not None:
            if self._status.size is not None:
                lines.append(
                    f"Problem: {self._status.family} "
                    f"n={self._status.size}"
                )
            else:
                lines.append(
                    f"Family: {self._status.family}"
                )
        elif self._status.size is not None:
            lines.append(f"Size: {self._status.size}")
        if (
            self._status.instance_index is not None
            and self._status.total_instances is not None
        ):
            lines.append(
                "Instance: "
                f"{self._status.instance_index + 1}/"
                f"{self._status.total_instances}"
            )
        if self._status.method is not None:
            lines.append(f"Method: {self._status.method}")
        if self._status.measure is not None:
            lines.append(f"Measure: {self._status.measure}")
        if self._status.template is not None:
            lines.append(
                f"Template: {self._status.template}"
            )
        if (
            self._status.candidate_index is not None
            and self._status.total_candidates is not None
        ):
            lines.append(
                "Candidate: "
                f"{self._status.candidate_index}/"
                f"{self._status.total_candidates}"
            )
        if (
            self._status.optimizer_total_restarts
            is not None
        ):
            line = (
                "Optimizer: "
                f"{self._status.optimizer_restart or 0}/"
                f"{self._status.optimizer_total_restarts}, "
                f"evals={self._status.optimizer_evals}"
            )
            if self._status.optimizer_best is not None:
                line += (
                    ", best="
                    f"{self._status.optimizer_best:.6g}"
                )
            lines.append(line)
        if self._status.qaoa_total is not None:
            lines.append(
                "QAOA grid: "
                f"{self._status.qaoa_done or 0}/"
                f"{self._status.qaoa_total}"
            )
        if self._status.detail is not None:
            lines.append(f"Detail: {self._status.detail}")

        if self._logs:
            log_text = self._Text("\n".join(self._logs))
        else:
            log_text = self._Text("No recent events")

        return self._Group(
            self._Panel.fit(
                self._Text("\n".join(lines)),
                title="compare_methods_baseline status",
            ),
            self._progress,
            self._Panel.fit(
                log_text,
                title="Recent events",
            ),
        )


def _build_baseline_comparison_tui(
    *,
    mode: str,
    totals: BaselineComparisonProgressTotals,
    stream=None,
) -> BaselineComparisonTui | None:
    """Build the requested baseline-comparison progress TUI."""
    if mode == "plain":
        return None
    if mode not in ("tui", "rich"):
        raise ValueError(
            "unsupported baseline comparison progress mode: "
            f"{mode}"
        )
    target_stream = stream or sys.stderr
    if not getattr(
        target_stream, "isatty", lambda: False
    )():
        print(
            "[compare_methods_baseline] TUI requested in a "
            "non-TTY context; falling back to plain progress.",
            file=target_stream,
            flush=True,
        )
        return None
    return BaselineComparisonTui(
        totals=totals,
        stream=target_stream,
    )


def activate_progress_ui(
    *,
    mode: str,
    totals: BaselineComparisonProgressTotals,
) -> BaselineComparisonTui | None:
    """Activate the shared baseline-comparison progress machinery."""
    tui = _build_baseline_comparison_tui(
        mode=mode,
        totals=totals,
        stream=sys.stderr,
    )
    global _ACTIVE_TUI
    global _NESTED_PROGRESS_ENABLED
    _ACTIVE_TUI = tui
    _NESTED_PROGRESS_ENABLED = tui is None
    if tui is not None:
        tui.start()
    return tui


def deactivate_progress_ui(
    tui: BaselineComparisonTui | None,
) -> None:
    """Reset the shared baseline-comparison progress machinery."""
    global _ACTIVE_TUI
    global _NESTED_PROGRESS_ENABLED
    if tui is not None:
        tui.close()
    _ACTIVE_TUI = None
    _NESTED_PROGRESS_ENABLED = True


def set_progress_status(**kwargs: object) -> None:
    """Update the active baseline-comparison TUI, if any."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.update_status(**kwargs)


def advance_progress_counter(
    counter: str,
    amount: int = 1,
) -> None:
    """Advance one baseline-comparison aggregate counter."""
    if _ACTIVE_TUI is None:
        return
    if counter == "tuning_jobs":
        _ACTIVE_TUI.advance_tuning_jobs(amount)
        return
    if counter == "component_builds":
        _ACTIVE_TUI.advance_component_builds(amount)
        return
    if counter == "baseline_methods":
        _ACTIVE_TUI.advance_baseline_methods(amount)
        return
    if counter == "instances":
        _ACTIVE_TUI.advance_instances(amount)
        return
    raise ValueError(
        "unknown baseline comparison progress counter: "
        f"{counter}"
    )


def begin_optimizer_progress(total_restarts: int) -> None:
    """Show one active Nelder-Mead progress task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.begin_optimizer(
        total_restarts=total_restarts
    )


def set_optimizer_restart(restart_index: int) -> None:
    """Update the currently active optimizer restart."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.set_optimizer_restart(restart_index)


def record_optimizer_eval(
    best_value: float | None = None,
) -> None:
    """Record one optimizer objective evaluation in the TUI."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.record_optimizer_eval(best_value=best_value)


def finish_optimizer_progress() -> None:
    """Hide the active Nelder-Mead progress task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.finish_optimizer()


def begin_qaoa_progress(total_points: int) -> None:
    """Show one active QAOA grid-search task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.begin_qaoa_grid(total_points=total_points)


def advance_qaoa_progress(amount: int = 1) -> None:
    """Advance the active QAOA grid-search task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.advance_qaoa_grid(amount)


def finish_qaoa_progress() -> None:
    """Hide the active QAOA grid-search task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.finish_qaoa_grid()


__all__ = [
    "BaselineComparisonProgressTotals",
    "activate_progress_ui",
    "advance_progress_counter",
    "advance_qaoa_progress",
    "begin_optimizer_progress",
    "begin_qaoa_progress",
    "deactivate_progress_ui",
    "finish_optimizer_progress",
    "finish_qaoa_progress",
    "log",
    "progress",
    "record_optimizer_eval",
    "set_optimizer_restart",
    "set_progress_status",
]
