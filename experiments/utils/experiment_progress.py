"""Reusable console progress reporters for long-running experiment scripts."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class ProgressTotals:
    """Aggregate counts shown by an experiment progress UI."""

    instances: int = 0
    topologies: int = 0
    measures: int = 0


@dataclass
class ProgressStatus:
    """Latest human-facing status metadata for an experiment run."""

    stage: str = "initializing"
    family: str | None = None
    size: int | None = None
    instance_index: int | None = None
    total_instances: int | None = None
    topology: str | None = None
    measure: str | None = None
    detail: str | None = None
    workers: int | None = None

    def merge(self, **kwargs: object) -> "ProgressStatus":
        """Apply explicit field updates in place."""
        for field_name, value in kwargs.items():
            if hasattr(self, field_name):
                setattr(self, field_name, value)
        return self

    def instance_label(self) -> str:
        """Return the current family/size/instance label."""
        if self.family is None or self.size is None:
            return "n/a"
        if (
            self.instance_index is None
            or self.total_instances is None
        ):
            return f"{self.family} n={self.size}"
        return (
            f"{self.family} n={self.size} "
            f"i={self.instance_index + 1}/{self.total_instances}"
        )


class ExperimentProgressReporter:
    """Shared interface for plain and rich progress reporters."""

    def __init__(
        self,
        *,
        totals: ProgressTotals,
        worker_count: int,
        stream: TextIO | None = None,
    ) -> None:
        self._totals = totals
        self._worker_count = int(worker_count)
        self._stream = stream or sys.stdout
        self._status = ProgressStatus(
            workers=self._worker_count
        )
        self._instances_done = 0
        self._topologies_done = 0
        self._measures_done = 0
        self._started = False
        self._closed = False

    def start(self) -> None:
        """Begin rendering progress."""
        self._started = True

    def update_status(self, **kwargs: object) -> None:
        """Update the current run status."""
        self._status.merge(**kwargs)
        self._render_status()

    def advance_instances(self, amount: int = 1) -> None:
        """Advance the instance counter."""
        self._instances_done += int(amount)
        self._render_counters()

    def advance_topologies(self, amount: int = 1) -> None:
        """Advance the topology counter."""
        self._topologies_done += int(amount)
        self._render_counters()

    def advance_measures(self, amount: int = 1) -> None:
        """Advance the measure counter."""
        self._measures_done += int(amount)
        self._render_counters()

    def close(self) -> None:
        """Stop rendering progress."""
        self._closed = True

    def _render_status(self) -> None:
        """Handle a status update."""

    def _render_counters(self) -> None:
        """Handle a counter update."""

    def _postfix(self) -> str:
        """Return a compact plain-text status summary."""
        parts = [
            f"topo {self._topologies_done}/{self._totals.topologies}",
            f"meas {self._measures_done}/{self._totals.measures}",
            self._status.instance_label(),
            f"stage={self._status.stage}",
        ]
        if self._status.topology is not None:
            parts.append(
                f"topology={self._status.topology}"
            )
        if self._status.measure is not None:
            parts.append(f"measure={self._status.measure}")
        if self._status.detail is not None:
            parts.append(self._status.detail)
        return " | ".join(parts)


class PlainProgressReporter(ExperimentProgressReporter):
    """Compact console reporter with one stable outer progress bar."""

    def __init__(
        self,
        *,
        totals: ProgressTotals,
        worker_count: int,
        stream: TextIO | None = None,
    ) -> None:
        super().__init__(
            totals=totals,
            worker_count=worker_count,
            stream=stream,
        )
        self._bar = None
        self._last_refresh = 0.0
        self._refresh_interval_seconds = 0.25

    def start(self) -> None:
        """Start the outer instance bar."""
        super().start()
        try:
            from tqdm.auto import tqdm
        except ImportError:
            self._bar = None
            return
        self._bar = tqdm(
            total=self._totals.instances,
            desc="Instances",
            unit="instance",
            file=self._stream,
            leave=True,
            disable=self._totals.instances <= 0,
        )
        self._render_counters()

    def close(self) -> None:
        """Close the outer progress bar."""
        if self._bar is not None:
            self._bar.close()
        super().close()

    def _render_status(self) -> None:
        self._refresh_bar(
            force=self._status.stage != "measure"
        )

    def _render_counters(self) -> None:
        if self._bar is None:
            return
        self._refresh_bar(
            force=self._instances_done
            >= self._totals.instances
        )

    def _refresh_bar(self, *, force: bool) -> None:
        """Apply the latest counters/status to the visible tqdm bar."""
        if self._bar is None:
            return
        now = time.monotonic()
        if (
            not force
            and (now - self._last_refresh)
            < self._refresh_interval_seconds
        ):
            return
        self._bar.n = min(
            self._instances_done, self._totals.instances
        )
        self._bar.set_postfix_str(
            self._postfix(), refresh=False
        )
        self._bar.refresh()
        self._last_refresh = now


class RichProgressReporter(ExperimentProgressReporter):
    """Live multi-bar terminal UI backed by ``rich``."""

    def __init__(
        self,
        *,
        totals: ProgressTotals,
        worker_count: int,
        stream: TextIO | None = None,
    ) -> None:
        super().__init__(
            totals=totals,
            worker_count=worker_count,
            stream=stream,
        )
        try:
            from rich.console import Console, Group
            from rich.live import Live
            from rich.panel import Panel
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.text import Text
        except ImportError as exc:
            raise RuntimeError(
                "Rich progress UI requested, but 'rich' is not installed. "
                "Install it with 'pip install rich' or via requirements.txt."
            ) from exc

        self._Console = Console
        self._Group = Group
        self._Live = Live
        self._Panel = Panel
        self._Progress = Progress
        self._BarColumn = BarColumn
        self._TaskProgressColumn = TaskProgressColumn
        self._TextColumn = TextColumn
        self._TimeElapsedColumn = TimeElapsedColumn
        self._Text = Text
        self._console = self._Console(
            file=self._stream, force_terminal=True
        )
        self._progress = None
        self._live = None
        self._instance_task_id = None
        self._topology_task_id = None
        self._measure_task_id = None

    def start(self) -> None:
        """Start the live TUI."""
        super().start()
        self._progress = self._Progress(
            self._TextColumn("[bold]{task.description}"),
            self._BarColumn(bar_width=None),
            self._TaskProgressColumn(),
            self._TimeElapsedColumn(),
            expand=True,
            console=self._console,
        )
        self._instance_task_id = self._progress.add_task(
            "Instances",
            total=max(1, self._totals.instances),
        )
        self._topology_task_id = self._progress.add_task(
            "Topologies",
            total=max(1, self._totals.topologies),
        )
        self._measure_task_id = self._progress.add_task(
            "Measures",
            total=max(1, self._totals.measures),
        )
        self._live = self._Live(
            self._renderable(),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()
        self._refresh()

    def close(self) -> None:
        """Stop the live TUI."""
        if self._live is not None:
            self._live.stop()
        super().close()

    def _render_status(self) -> None:
        self._refresh()

    def _render_counters(self) -> None:
        if self._progress is None:
            return
        self._progress.update(
            self._instance_task_id,
            completed=min(
                self._instances_done, self._totals.instances
            ),
        )
        self._progress.update(
            self._topology_task_id,
            completed=min(
                self._topologies_done,
                self._totals.topologies,
            ),
        )
        self._progress.update(
            self._measure_task_id,
            completed=min(
                self._measures_done, self._totals.measures
            ),
        )
        self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        self._live.update(self._renderable(), refresh=True)

    def _renderable(self):
        lines = [
            f"Stage: {self._status.stage}",
            f"Current: {self._status.instance_label()}",
            f"Topology: {self._status.topology or 'n/a'}",
            f"Measure: {self._status.measure or 'n/a'}",
            f"Workers: {self._worker_count}",
        ]
        if self._status.detail is not None:
            lines.append(f"Detail: {self._status.detail}")
        status_text = self._Text("\n".join(lines))
        return self._Group(
            self._Panel.fit(
                status_text, title="Experiment Status"
            ),
            self._progress,
        )


def build_progress_reporter(
    *,
    mode: str,
    totals: ProgressTotals,
    worker_count: int,
    stream: TextIO | None = None,
) -> ExperimentProgressReporter:
    """Return the requested reporter, falling back when needed."""
    target_stream = stream or sys.stdout
    if mode == "rich":
        if not getattr(
            target_stream, "isatty", lambda: False
        )():
            print(
                "[experiment_progress] rich progress requested in a non-TTY "
                "context; falling back to plain progress.",
                file=target_stream,
                flush=True,
            )
            return PlainProgressReporter(
                totals=totals,
                worker_count=worker_count,
                stream=target_stream,
            )
        return RichProgressReporter(
            totals=totals,
            worker_count=worker_count,
            stream=target_stream,
        )
    if mode != "plain":
        raise ValueError(
            f"unsupported progress reporter mode: {mode}"
        )
    return PlainProgressReporter(
        totals=totals,
        worker_count=worker_count,
        stream=target_stream,
    )
