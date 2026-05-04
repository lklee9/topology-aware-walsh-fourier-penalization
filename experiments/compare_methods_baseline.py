"""Shared engine for tuning penalties and comparing logical baselines.

This script follows the protocol in ``brainstorm/outline.md`` with the
current experiment-specific adjustments:

1. compare unbalanced penalization, the projected penalty on the full
   logical topology, and the projected penalty with couplings restricted
   to a Pegasus-induced logical topology;

   additionally compare the full-topology projected penalty when its
   inequality template is the quadratic penalty fitted from the tuned
   unbalanced parameters;

2. fit topology-restricted projected penalties from sampled states using
   the chosen projection measure and inequality penalty template;
   equality QUBOs remain exact on the full logical topology but are
   projected on sparse logical topologies using the same sampled states
   and importance-fitting pipeline as the inequality constraints;

3. tune one global unbalanced-penalty multiplier and separate projected
   equality/inequality multipliers on one anchor size per problem
   family with a deterministic multi-start Nelder-Mead search, then
   reuse those weights for all other sizes in that family;

4. report a logical SQA baseline on the same penalized QUBOs.

"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

if __package__ in (None, ""):
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

# Top-level D-Wave annealer imports. Let import-time failures propagate so
# the module fails to import if these dependencies aren't present.
from experiments.experiment_config import *
from experiments.utils import (
    projected_method_selection as _projected_selection_utils,
)
from experiments.utils import (
    tuning_support as _shared_tuning_support,
)
from experiments.utils.baseline_common import (
    add_common_arguments as _shared_add_common_arguments,
)
from experiments.utils.baseline_common import (
    projected_methods as _shared_projected_methods,
)
from experiments.utils.baseline_common import (
    qubo_normalization_scale as _shared_qubo_normalization_scale,
)
from experiments.utils.baseline_common import (
    scale_qubo_coefficients as _shared_scale_qubo_coefficients,
)
from experiments.utils.cplex_reference import (
    DEFAULT_REFERENCE_ATOL,
    CplexReference,
    cplex_reference_for_synthetic_problem,
    load_cplex_reference_index,
    reference_objective_gap_ratio,
    reference_objective_hit_mask,
    resolve_cplex_reference_path,
)
from experiments.utils.driver_common import (
    build_child_rng as _rng_from_seed,
)
from experiments.utils.driver_common import (
    decoded_sampleset_bits as _decoded_sampleset_bits,
)
from experiments.utils.driver_common import (
    default_family_sizes as _shared_default_family_sizes,
)
from experiments.utils.driver_common import (
    full_pair_edges as _full_pair_edges,
)
from experiments.utils.driver_common import (
    num_inequality_quadratic_terms as _num_inequality_quadratic_terms,
)
from experiments.utils.driver_common import (
    problem_batch_getter as _shared_problem_batch_getter,
)
from experiments.utils.driver_common import (
    problem_provenance_fields as _problem_provenance_fields,
)
from experiments.utils.driver_common import (
    projection_regime_fields as _projection_regime_fields,
)
from experiments.utils.driver_common import (
    write_rows_csv as _write_rows_csv,
)
from experiments.utils.embedding import _qubo_arrays_to_bqm
from experiments.utils.family_cli import (
    add_family_selection_arguments,
    selected_families_from_args,
    selected_family_sizes_from_args,
)
from experiments.utils.fixed_sqa import (
    anneal_schedule_json as _shared_anneal_schedule_json,
)
from experiments.utils.fixed_sqa import (
    fixed_sqa_schedule as _fixed_sqa_schedule,
)
from experiments.utils.fixed_sqa import (
    sqa_schedule_catalog as _shared_sqa_schedule_catalog,
)
from experiments.utils.merge_outputs import (
    ensure_run_metadata,
    merge_csv_rows,
)
from experiments.utils.qa_simulator import (
    build_dwave_graph,
    qpu_anneal_schedule_to_sqa_fields,
    sqa_dwave_annealer_samples,
)
from experiments.utils.tuning_summary import (
    load_tuning_outputs as _shared_load_tuning_outputs,
)
from experiments.utils.unbalanced_pipeline import (
    UP_NORMALIZATION_REGIME as _UP_NORMALIZATION_REGIME,
)
from experiments.utils.unbalanced_pipeline import (
    build_unbalanced_components as _build_unbalanced_components,
)
from experiments.utils.unbalanced_pipeline import (
    build_unbalanced_qubo_from_components as _build_unbalanced_qubo_from_components,
)
from fourier_projection.blp import BLP
from fourier_projection.greedy_mapping import (
    mapped_logical_topology_from_graph,
)

_SQA_WARNING_EMITTED = False
_NESTED_PROGRESS_ENABLED = True
_ACTIVE_TUI = None
_ANNEALER_SEED_DOMAIN = 4_000


def _stable_seed_component(label: object) -> int:
    """Return one stable integer seed component for a text label."""
    digest = hashlib.blake2b(
        str(label).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(
        digest, byteorder="little", signed=False
    )


def _problem_seed(problem: BLP) -> int:
    """Return the stable manifest/problem seed attached to one instance."""
    metadata = dict(getattr(problem, "metadata", {}) or {})
    if metadata.get("problem_seed") is None:
        raise ValueError(
            "problem metadata is missing 'problem_seed'; "
            "cannot align annealer randomness across experiments"
        )
    return int(metadata["problem_seed"])


def _annealer_schedule_seed_label(
    *,
    schedule_id: str,
    schedule_kind: str,
    beta_scale: float,
    num_points: int,
    num_sweeps_per_beta: int,
) -> str:
    """Return the canonical schedule label used for annealer seeding."""
    return (
        f"{schedule_kind}:{schedule_id}:beta={float(beta_scale):.12g}:"
        f"points={int(num_points)}:spb={int(num_sweeps_per_beta)}"
    )


def _logical_annealer_rng(
    problem: BLP,
    *,
    family: str,
    size: int,
    method: str,
    base_seed: int,
    schedule_id: str,
    schedule_kind: str,
    beta_scale: float,
    num_points: int,
    num_sweeps_per_beta: int,
) -> np.random.Generator:
    """Return the shared logical-SQA RNG used across experiments."""
    schedule_label = _annealer_schedule_seed_label(
        schedule_id=schedule_id,
        schedule_kind=schedule_kind,
        beta_scale=beta_scale,
        num_points=num_points,
        num_sweeps_per_beta=num_sweeps_per_beta,
    )
    return _rng_from_seed(
        base_seed,
        _ANNEALER_SEED_DOMAIN,
        _problem_seed(problem),
        FAMILY_CODES[family],
        int(size),
        _stable_seed_component(str(method)),
        _stable_seed_component(schedule_label),
    )


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


def _progress(iterable=None, **kwargs):
    """Return a ``tqdm`` progress bar when available."""
    if not _NESTED_PROGRESS_ENABLED or _tqdm is None:
        return _NullProgressBar(iterable=iterable, **kwargs)
    return _tqdm(iterable, **kwargs)


def _log(message: str) -> None:
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
                "baseline comparison TUI requested, but 'rich' is not installed."
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
        self._logs: deque[str] = deque(maxlen=8)
        self._progress = None
        self._live = None
        self._last_refresh = 0.0
        self._refresh_interval_seconds = 0.1
        self._tuning_task_id = None
        self._components_task_id = None
        self._baselines_task_id = None
        self._instances_task_id = None
        self._optimizer_task_id = None
        self._tuning_jobs_done = 0
        self._component_builds_done = 0
        self._baseline_methods_done = 0
        self._instances_done = 0

    def start(self) -> None:
        """Start the live TUI."""
        self._progress = self._Progress(
            self._SpinnerColumn(style="cyan"),
            self._TextColumn("{task.description}"),
            self._BarColumn(bar_width=None),
            self._TaskProgressColumn(),
            self._TimeElapsedColumn(),
            expand=True,
            console=self._console,
        )
        self._tuning_task_id = self._progress.add_task(
            "Anchor tuning",
            total=max(1, self._totals.tuning_jobs),
            visible=self._totals.tuning_jobs > 0,
        )
        self._components_task_id = self._progress.add_task(
            "Projected components",
            total=max(1, self._totals.component_builds),
            visible=self._totals.component_builds > 0,
        )
        self._baselines_task_id = self._progress.add_task(
            "Baseline methods",
            total=max(1, self._totals.baseline_methods),
            visible=self._totals.baseline_methods > 0,
        )
        self._instances_task_id = self._progress.add_task(
            "Completed instances",
            total=max(1, self._totals.instances),
            visible=self._totals.instances > 0,
        )
        self._optimizer_task_id = self._progress.add_task(
            "Optimizer restarts",
            total=1,
            visible=False,
        )
        self._live = self._Live(
            self._renderable(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()
        self._refresh(force=True)

    def close(self) -> None:
        """Stop the live TUI."""
        if self._live is not None:
            self._live.stop()

    def update_status(self, **kwargs: object) -> None:
        """Update the current experiment context."""
        self._status.merge(**kwargs)
        self._refresh(force=False)

    def log(self, message: str) -> None:
        """Append one recent-event line to the TUI."""
        self._logs.append(message)
        self._refresh(force=True)

    def advance_tuning_jobs(self, amount: int = 1) -> None:
        """Advance the completed tuning-job counter."""
        self._tuning_jobs_done += int(amount)
        self._update_task(
            self._tuning_task_id,
            completed=min(
                self._tuning_jobs_done,
                self._totals.tuning_jobs,
            ),
        )
        self._refresh(force=False)

    def advance_component_builds(
        self, amount: int = 1
    ) -> None:
        """Advance the projected-component counter."""
        self._component_builds_done += int(amount)
        self._update_task(
            self._components_task_id,
            completed=min(
                self._component_builds_done,
                self._totals.component_builds,
            ),
        )
        self._refresh(force=False)

    def advance_baseline_methods(
        self, amount: int = 1
    ) -> None:
        """Advance the baseline-method counter."""
        self._baseline_methods_done += int(amount)
        self._update_task(
            self._baselines_task_id,
            completed=min(
                self._baseline_methods_done,
                self._totals.baseline_methods,
            ),
        )
        self._refresh(force=False)

    def advance_instances(self, amount: int = 1) -> None:
        """Advance the completed-instance counter."""
        self._instances_done += int(amount)
        self._update_task(
            self._instances_task_id,
            completed=min(
                self._instances_done,
                self._totals.instances,
            ),
        )
        self._refresh(force=False)

    def begin_optimizer(
        self, *, total_restarts: int
    ) -> None:
        """Show one current Nelder-Mead optimizer task."""
        self._status.merge(
            optimizer_restart=0,
            optimizer_total_restarts=int(total_restarts),
            optimizer_evals=0,
            optimizer_best=None,
        )
        self._update_task(
            self._optimizer_task_id,
            description="Optimizer restarts",
            total=max(1, int(total_restarts)),
            completed=0,
            visible=True,
        )
        self._refresh(force=True)

    def set_optimizer_restart(
        self, restart_index: int
    ) -> None:
        """Update the active optimizer restart index."""
        total = self._status.optimizer_total_restarts
        completed = (
            0 if restart_index <= 0 else restart_index - 1
        )
        self._status.optimizer_restart = int(restart_index)
        self._update_task(
            self._optimizer_task_id,
            completed=min(completed, total or completed),
        )
        self._refresh(force=False)

    def record_optimizer_eval(
        self,
        *,
        best_value: float | None = None,
    ) -> None:
        """Advance the visible optimizer evaluation count."""
        self._status.optimizer_evals += 1
        if best_value is not None:
            current_best = self._status.optimizer_best
            if (
                current_best is None
                or best_value < current_best
            ):
                self._status.optimizer_best = float(
                    best_value
                )
        self._refresh(force=False)

    def finish_optimizer(self) -> None:
        """Hide the optimizer task after one search finishes."""
        total = self._status.optimizer_total_restarts
        if total is not None:
            self._update_task(
                self._optimizer_task_id,
                completed=int(total),
            )
        self._update_task(
            self._optimizer_task_id, visible=False
        )
        self._status.merge(
            optimizer_restart=None,
            optimizer_total_restarts=None,
        )
        self._refresh(force=True)

    def _update_task(
        self, task_id, **kwargs: object
    ) -> None:
        """Update one rich progress task if the TUI is active."""
        if self._progress is None or task_id is None:
            return
        self._progress.update(task_id, **kwargs)

    def _refresh(self, *, force: bool) -> None:
        """Refresh the live TUI with throttling."""
        if self._live is None:
            return
        now = time.monotonic()
        if (
            not force
            and (now - self._last_refresh)
            < self._refresh_interval_seconds
        ):
            return
        self._live.update(self._renderable(), refresh=True)
        self._last_refresh = now

    def _renderable(self):
        lines = [
            f"Stage: {self._status.stage}",
            f"Activity: {self._status.activity or 'n/a'}",
            (
                f"Family/size: {self._status.family or 'n/a'}"
                f" / {self._status.size if self._status.size is not None else 'n/a'}"
            ),
            (
                "Instance: "
                + (
                    "n/a"
                    if self._status.instance_index is None
                    or self._status.total_instances is None
                    else (
                        f"{self._status.instance_index + 1}/"
                        f"{self._status.total_instances}"
                    )
                )
            ),
            f"Method: {self._status.method or 'n/a'}",
            f"Measure: {self._status.measure or 'n/a'}",
        ]
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
            restart = self._status.optimizer_restart or 0
            line = (
                "Optimizer: "
                f"restart {restart}/"
                f"{self._status.optimizer_total_restarts}"
                f", evals {self._status.optimizer_evals}"
            )
            if self._status.optimizer_best is not None:
                line += (
                    ", best="
                    f"{self._status.optimizer_best:.6g}"
                )
            lines.append(line)
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
                log_text, title="Recent events"
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
            f"unsupported baseline comparison progress mode: {mode}"
        )
    target_stream = stream or sys.stderr
    if not getattr(
        target_stream, "isatty", lambda: False
    )():
        print(
            "[compare_methods_baseline] TUI requested in a non-TTY context; "
            "falling back to plain progress.",
            file=target_stream,
            flush=True,
        )
        return None
    return BaselineComparisonTui(
        totals=totals, stream=target_stream
    )


def _set_progress_status(**kwargs: object) -> None:
    """Update the active baseline-comparison TUI, if any."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.update_status(**kwargs)


def _advance_progress_counter(
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
        f"unknown baseline comparison progress counter: {counter}"
    )


def _begin_optimizer_progress(total_restarts: int) -> None:
    """Show one active Nelder-Mead progress task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.begin_optimizer(
        total_restarts=total_restarts
    )


def _set_optimizer_restart(restart_index: int) -> None:
    """Update the currently active optimizer restart."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.set_optimizer_restart(restart_index)


def _record_optimizer_eval(
    best_value: float | None = None,
) -> None:
    """Record one optimizer objective evaluation in the TUI."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.record_optimizer_eval(best_value=best_value)


def _finish_optimizer_progress() -> None:
    """Hide the active Nelder-Mead progress task."""
    if _ACTIVE_TUI is None:
        return
    _ACTIVE_TUI.finish_optimizer()


@dataclass(frozen=True)
class GroundStateTuningMetrics:
    """Gap-only tuning metrics for one penalized energy spectrum."""

    gap: float
    best_optimum_percentile: float
    tied_fraction: float


@dataclass(frozen=True)
class TuningObjectiveResult:
    """Comparable tuning score plus summary metrics for one parameter
    point."""

    sort_key: tuple[float, ...]
    objective_value: float
    record_fields: dict[str, object]


@dataclass(frozen=True)
class LogicalAnnealerMetrics:
    """Summary metrics for one logical annealer run on a penalized QUBO."""

    mean_penalized_energy_gap: float
    penalized_ground_state_rate: float
    logical_optimum_probability: float
    logical_cop: float
    feasible_rate: float
    best_feasible_objective: float | None
    objective_gap: float
    schedule_id: str
    schedule_kind: str
    beta_scale: float
    total_schedule_time: float
    anneal_schedule: tuple[tuple[float, float], ...]
    num_reads: int


@dataclass(frozen=True)
class SqaScheduleCandidate:
    """One candidate QPU-style schedule paired with one beta scale."""

    schedule_id: str
    schedule_kind: str
    anneal_schedule: tuple[tuple[float, float], ...]
    beta_scale: float
    total_schedule_time: float


@dataclass(frozen=True)
class TunedSqaSchedule:
    """Family-level logical SQA schedule selected on the anchor size."""

    family: str
    anchor_size: int
    schedule_id: str
    schedule_kind: str
    anneal_schedule: tuple[tuple[float, float], ...]
    beta_scale: float
    total_schedule_time: float
    mean_logical_optimum_probability: float
    mean_feasible_rate: float
    mean_objective_gap: float
    mean_penalized_energy_gap: float
    objective_value: tuple[float, ...]

    def as_row(self) -> dict[str, object]:
        return {
            "family": self.family,
            "anchor_size": self.anchor_size,
            "schedule_id": self.schedule_id,
            "schedule_kind": self.schedule_kind,
            "beta_scale": self.beta_scale,
            "total_schedule_time": self.total_schedule_time,
            "mean_logical_optimum_probability": (
                self.mean_logical_optimum_probability
            ),
            "mean_feasible_rate": self.mean_feasible_rate,
            "mean_objective_gap": self.mean_objective_gap,
            "mean_penalized_energy_gap": (
                self.mean_penalized_energy_gap
            ),
            "anneal_schedule": _shared_anneal_schedule_json(
                self.anneal_schedule
            ),
        }


@dataclass(frozen=True)
class ScaledAnnealerInput:
    """Scaled logical-QUBO data used by the SQA baseline."""

    quadratic: np.ndarray
    linear: np.ndarray
    const: float


@dataclass(frozen=True)
class PreparedSqaBaselineInstance:
    """Anchor instance data reused across family-level schedule tuning."""

    problem: BLP
    reference: CplexReference
    annealer_inputs: dict[str, ScaledAnnealerInput]


@dataclass(frozen=True)
class TunedProjectedMultipliers:
    """Family-level projected-penalty multipliers selected on the anchor
    size."""

    method: str
    family: str
    anchor_size: int
    equality_multiplier: float
    inequality_multiplier: float
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "family": self.family,
            "anchor_size": self.anchor_size,
            "equality_multiplier": self.equality_multiplier,
            "inequality_multiplier": self.inequality_multiplier,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class SelectedProjectedConfig:
    """One projected-method combo selected for downstream evaluation."""

    method: str
    family: str
    projection_method: str
    measure_name: str
    penalty_template: str
    penalty_template_kwargs: dict[str, float]
    selection_mode: str
    selection_source: str
    candidate_rank: int
    tuning: TunedProjectedMultipliers


@dataclass(frozen=True)
class TuningRunOutputs:
    """Family-level tuning artifacts reused by the baseline comparison."""

    projected_tuning_rows: list[dict[str, object]]
    projected_selection_rows: list[dict[str, object]]
    up_tuning_rows: list[dict[str, object]]
    tuned_unbalanced_params: dict[
        str, "TunedUnbalancedParameters"
    ]
    selected_projected_configs: dict[
        tuple[str, str], SelectedProjectedConfig
    ]


@dataclass(frozen=True)
class TunedUnbalancedParameters:
    """Family-level unbalanced-penalty parameters selected on the anchor
    size."""

    family: str
    anchor_size: int
    global_multiplier: float
    lambda0: float | None
    lambda1: float
    lambda2: float
    base_parameter_source: str
    tuning_objective: str
    objective_value: float
    success: bool
    status: int
    message: str

    def as_row(self) -> dict[str, object]:
        return {
            "family": self.family,
            "anchor_size": self.anchor_size,
            "global_multiplier": self.global_multiplier,
            "lambda0": self.lambda0,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "base_parameter_source": self.base_parameter_source,
            "tuning_objective": self.tuning_objective,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class TuningInstance:
    """Cached exact energy tables for one anchor instance."""

    num_states: int
    optimum_state_indices: np.ndarray
    objective_energies: np.ndarray
    equality_energies: np.ndarray
    inequality_energies: np.ndarray


@dataclass(frozen=True)
class UnbalancedTuningInstance:
    """Cached exact energy tables for UP tuning on one anchor
    instance."""

    num_states: int
    optimum_state_indices: np.ndarray
    objective_energies: np.ndarray
    equality_energies: np.ndarray
    inequality_linear_energies: np.ndarray
    inequality_quadratic_energies: np.ndarray


def _warn_sqa_unavailable(message: str) -> None:
    """Log one SQA warning while letting the experiment continue."""
    global _SQA_WARNING_EMITTED
    if _SQA_WARNING_EMITTED:
        return
    _log(f"warning: {message}")
    _SQA_WARNING_EMITTED = True


@lru_cache(maxsize=None)
def _pegasus_hardware_graph(size: int):
    """Build and cache the Pegasus hardware graph used for topology restriction."""
    return build_dwave_graph("pegasus", int(size))


@lru_cache(maxsize=None)
def _chimera_hardware_graph(size: int):
    """Build and cache the Chimera hardware graph used for topology restriction."""
    return build_dwave_graph("chimera", int(size))


@lru_cache(maxsize=None)
def _zephyr_hardware_graph(size: int):
    """Build and cache the Zephyr hardware graph used for topology restriction."""
    return build_dwave_graph("zephyr", int(size))


def _unbalanced_qubo(
    problem: BLP,
    tuned_params: "TunedUnbalancedParameters",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the shared standardized UP comparison QUBO."""
    components = _build_unbalanced_components(
        problem,
        lambda1_shape=float(tuned_params.up_lambda1_shape),
        lambda2_shape=float(tuned_params.up_lambda2_shape),
        standardize=(
            True
            if tuned_params.per_constraint_standardization
            is None
            else bool(
                tuned_params.per_constraint_standardization
            )
        ),
    )
    return _build_unbalanced_qubo_from_components(
        problem,
        components,
        equality_multiplier=(
            0.0
            if tuned_params.up_equality_multiplier is None
            else float(tuned_params.up_equality_multiplier)
        ),
        inequality_multiplier=float(
            tuned_params.up_inequality_multiplier
        ),
    )


def _qubo_normalization_scale(
    quadratic: np.ndarray,
    linear: np.ndarray,
    *,
    num_variables: int,
) -> float:
    """Return the QUBO normalization scale used by the baseline notebook."""
    return _shared_qubo_normalization_scale(
        quadratic,
        linear,
        num_variables=num_variables,
    )


def _projection_pair_edges(
    problem: BLP,
    *,
    projection_method: str,
    pegasus_size: int,
) -> list[tuple[int, int]]:
    """Return the admissible quadratic pairs for one projected-method variant."""
    if projection_method == "projected_full":
        return _full_pair_edges(problem.num_variables)

    if projection_method in (
        "projected_pegasus",
        "projected_chimera",
        "projected_zephyr",
    ):
        if projection_method == "projected_pegasus":
            hardware_graph = _pegasus_hardware_graph(
                pegasus_size
            )
            family_name = "Pegasus"
        elif projection_method == "projected_chimera":
            hardware_graph = _chimera_hardware_graph(
                pegasus_size
            )
            family_name = "Chimera"
        else:
            hardware_graph = _zephyr_hardware_graph(
                pegasus_size
            )
            family_name = "Zephyr"

        placement, topology = (
            mapped_logical_topology_from_graph(
                problem.constraint_matrix,
                hardware_graph,
                logical_vertices=range(
                    problem.num_variables
                ),
            )
        )
        edges = [tuple(edge) for edge in topology.E]

        if all(
            0 <= u < problem.num_variables
            and 0 <= v < problem.num_variables
            for u, v in edges
        ):
            return edges

        # Some experiment-local mapper copies return hardware-labeled edges.
        # Recover the induced logical topology from the placement when that happens.
        hardware_to_logical = {
            hardware_vertex: logical_vertex
            for logical_vertex, hardware_vertex in placement.items()
        }
        canonical_edges: set[tuple[int, int]] = set()
        for u, v in edges:
            if (
                u not in hardware_to_logical
                or v not in hardware_to_logical
            ):
                continue
            logical_u = int(hardware_to_logical[u])
            logical_v = int(hardware_to_logical[v])
            if logical_u == logical_v:
                continue
            canonical_edges.add(
                tuple(sorted((logical_u, logical_v)))
            )

        if canonical_edges:
            return sorted(canonical_edges)

        raise ValueError(
            f"{family_name} logical topology contains out-of-range edge labels and could "
            "not be canonicalized to logical variable indices"
        )
    raise ValueError(
        f"unknown projected method: {projection_method}"
    )


def _scaled_annealer_input(
    problem: BLP,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    normalization_scale: float,
    chunk_size: int,
) -> ScaledAnnealerInput:
    """Return the normalized logical QUBO used by the SQA baseline."""
    del problem
    del chunk_size
    scaled_quadratic, scaled_linear, scaled_const = (
        _shared_scale_qubo_coefficients(
            quadratic,
            linear,
            const,
            normalization_scale=normalization_scale,
        )
    )
    return ScaledAnnealerInput(
        quadratic=scaled_quadratic,
        linear=scaled_linear,
        const=scaled_const,
    )


def _logical_annealer_penalized_metrics(
    problem: BLP,
    quadratic: np.ndarray,
    linear: np.ndarray,
    const: float,
    *,
    reference: CplexReference,
    family: str,
    size: int,
    instance_index: int,
    method: str,
    base_seed: int,
    schedule_index: int,
    num_reads: int,
    schedule: SqaScheduleCandidate,
    atol: float,
) -> LogicalAnnealerMetrics:
    """Run logical SQA on a penalized QUBO and summarize metrics."""
    _set_progress_status(
        stage="sqa_sampling",
        activity="running SQA",
        family=family,
        size=size,
        instance_index=instance_index,
        method=method,
        detail=(
            f"{schedule.schedule_id} "
            f"(kind={schedule.schedule_kind}, "
            f"beta_scale={schedule.beta_scale:.3g}), "
            f"{num_reads} reads"
        ),
    )
    try:
        # Sample directly on the logical QUBO (no embedding/chains).
        bqm = _qubo_arrays_to_bqm(quadratic, linear, const)
        hp_field, hd_field = (
            qpu_anneal_schedule_to_sqa_fields(
                schedule.anneal_schedule,
                schedule.beta_scale,
                num_points=DEFAULT_SQA_NUM_SWEEPS,
            )
        )
        del schedule_index
        rng = _logical_annealer_rng(
            problem,
            family=family,
            size=size,
            method=method,
            base_seed=base_seed,
            schedule_id=schedule.schedule_id,
            schedule_kind=schedule.schedule_kind,
            beta_scale=float(schedule.beta_scale),
            num_points=DEFAULT_SQA_NUM_SWEEPS,
            num_sweeps_per_beta=DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
        )
        sampleset = sqa_dwave_annealer_samples(
            bqm,
            num_reads=num_reads,
            rng=rng,
            num_sweeps=int(hp_field.size),
            num_sweeps_per_beta=DEFAULT_SQA_NUM_SWEEPS_PER_BETA,
            hp_field=hp_field,
            hd_field=hd_field,
        )
    except (RuntimeError, ValueError) as exc:
        _warn_sqa_unavailable(str(exc))
        return LogicalAnnealerMetrics(
            mean_penalized_energy_gap=math.nan,
            penalized_ground_state_rate=math.nan,
            logical_optimum_probability=math.nan,
            logical_cop=math.nan,
            feasible_rate=math.nan,
            best_feasible_objective=None,
            objective_gap=math.nan,
            schedule_id=schedule.schedule_id,
            schedule_kind=schedule.schedule_kind,
            beta_scale=float(schedule.beta_scale),
            total_schedule_time=float(
                schedule.total_schedule_time
            ),
            anneal_schedule=tuple(schedule.anneal_schedule),
            num_reads=int(num_reads),
        )

    sample_energies = np.asarray(
        sampleset.record.energy, dtype=float
    )
    occurrences = np.asarray(
        sampleset.record.num_occurrences, dtype=float
    )
    total_reads = float(np.sum(occurrences))
    if total_reads <= 0:
        return LogicalAnnealerMetrics(
            mean_penalized_energy_gap=math.nan,
            penalized_ground_state_rate=math.nan,
            logical_optimum_probability=math.nan,
            logical_cop=math.nan,
            feasible_rate=math.nan,
            best_feasible_objective=None,
            objective_gap=math.nan,
            schedule_id=schedule.schedule_id,
            schedule_kind=schedule.schedule_kind,
            beta_scale=float(schedule.beta_scale),
            total_schedule_time=float(
                schedule.total_schedule_time
            ),
            anneal_schedule=tuple(schedule.anneal_schedule),
            num_reads=int(num_reads),
        )

    bitstrings = _decoded_sampleset_bits(
        sampleset,
        num_variables=problem.num_variables,
    )
    bits_float = bitstrings.astype(float, copy=False)
    feasible_mask = problem.feasible_mask(bits_float)
    objective_values = problem.objective_values(bits_float)
    hit_mask = reference_objective_hit_mask(
        objective_values,
        feasible_mask,
        optimum_objective=float(
            reference.optimum_objective
        ),
        objective_sense=reference.objective_sense,
        atol=atol,
    )
    feasible_rate = float(
        np.sum(occurrences[feasible_mask]) / total_reads
    )
    logical_optimum_probability = float(
        np.sum(occurrences[hit_mask]) / total_reads
    )
    if np.any(feasible_mask):
        mean_gap = float(
            np.average(
                reference_objective_gap_ratio(
                    objective_values[feasible_mask],
                    float(reference.optimum_objective),
                    objective_sense=reference.objective_sense,
                ),
                weights=occurrences[feasible_mask],
            )
        )
    else:
        mean_gap = math.nan
    if np.any(feasible_mask):
        feasible_objectives = objective_values[
            feasible_mask
        ]
        if (
            str(reference.objective_sense).strip().lower()
            == "max"
        ):
            best_feasible_objective = float(
                np.max(feasible_objectives)
            )
        else:
            best_feasible_objective = float(
                np.min(feasible_objectives)
            )
    else:
        best_feasible_objective = None
    return LogicalAnnealerMetrics(
        mean_penalized_energy_gap=mean_gap,
        penalized_ground_state_rate=logical_optimum_probability,
        logical_optimum_probability=logical_optimum_probability,
        logical_cop=logical_optimum_probability
        * float(1 << problem.num_variables),
        feasible_rate=feasible_rate,
        best_feasible_objective=best_feasible_objective,
        objective_gap=reference_objective_gap_ratio(
            best_feasible_objective,
            float(reference.optimum_objective),
            objective_sense=reference.objective_sense,
        ),
        schedule_id=schedule.schedule_id,
        schedule_kind=schedule.schedule_kind,
        beta_scale=float(schedule.beta_scale),
        total_schedule_time=float(
            schedule.total_schedule_time
        ),
        anneal_schedule=tuple(schedule.anneal_schedule),
        num_reads=int(num_reads),
    )


def _logical_annealer_baseline_result(
    problem: BLP,
    annealer_input: ScaledAnnealerInput,
    *,
    reference: CplexReference,
    family: str,
    size: int,
    instance_index: int,
    method: str,
    base_seed: int,
    num_reads: int,
    schedule: SqaScheduleCandidate,
    schedule_index: int = 0,
    atol: float,
) -> LogicalAnnealerMetrics:
    """Run the logical SQA baseline with one selected schedule."""
    return _logical_annealer_penalized_metrics(
        problem,
        annealer_input.quadratic,
        annealer_input.linear,
        annealer_input.const,
        reference=reference,
        family=family,
        size=size,
        instance_index=instance_index,
        method=method,
        base_seed=base_seed,
        schedule_index=schedule_index,
        num_reads=num_reads,
        schedule=schedule,
        atol=atol,
    )


# Exact-spectrum CSV export removed
def _aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-instance SQA rows into a compact comparison table."""
    grouped: dict[
        tuple[str, int, str], list[dict[str, object]]
    ] = {}
    for row in rows:
        key = (
            str(row["family"]),
            int(row["size"]),
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key in sorted(grouped):
        family, size, method = key
        items = grouped[key]
        num_variables = int(items[0]["num_variables"])
        logical_cop = np.array(
            [
                float(item.get("sqa_logical_cop", math.nan))
                for item in items
            ],
            dtype=float,
        )
        feasible_rate = np.array(
            [
                float(item.get("sqa_fea", math.nan))
                for item in items
            ],
            dtype=float,
        )
        objective_gap = np.array(
            [
                float(item.get("sqa_gap", math.nan))
                for item in items
            ],
            dtype=float,
        )

        mean_row: dict[str, object] = {
            "family": family,
            "size": size,
            "method": method,
            "n": num_variables,
            "stat": "mean",
            "inst": len(items),
        }
        finite_cop = np.isfinite(logical_cop)
        finite_fea = np.isfinite(feasible_rate)
        finite_gap = np.isfinite(objective_gap)
        mean_row["sqa_cop"] = (
            float(np.mean(logical_cop[finite_cop]))
            if np.any(finite_cop)
            else math.nan
        )
        mean_row["sqa_fea"] = (
            float(np.mean(feasible_rate[finite_fea]))
            if np.any(finite_fea)
            else math.nan
        )
        mean_row["sqa_gap"] = (
            float(np.mean(objective_gap[finite_gap]))
            if np.any(finite_gap)
            else math.nan
        )

        std_row: dict[str, object] = {
            "family": family,
            "size": size,
            "method": method,
            "n": num_variables,
            "stat": "std",
            "inst": len(items),
        }
        std_row["sqa_cop"] = (
            float(np.std(logical_cop[finite_cop], ddof=0))
            if np.any(finite_cop)
            else math.nan
        )
        std_row["sqa_fea"] = (
            float(np.std(feasible_rate[finite_fea], ddof=0))
            if np.any(finite_fea)
            else math.nan
        )
        std_row["sqa_gap"] = (
            float(np.std(objective_gap[finite_gap], ddof=0))
            if np.any(finite_gap)
            else math.nan
        )

        out.append(mean_row)
        out.append(std_row)
    return out


def _cop_metadata_rows(
    families: tuple[str, ...] = FAMILY_ORDER,
) -> list[dict[str, object]]:
    """Return a compact metadata table for the result CSVs."""
    fixed_schedule = _fixed_sqa_schedule()
    catalog_ids = ";".join(
        candidate.schedule_id
        for candidate in _shared_sqa_schedule_catalog()
    )
    selected_ids = ";".join(
        f"{family}:{fixed_schedule.schedule_id}"
        for family in families
    )
    return [
        {
            "samplers": ";".join(
                DEFAULT_LOGICAL_ANNEALER_SAMPLERS
            ),
            "sqa_reads": str(DEFAULT_SQA_NUM_READS),
            "sqa_mode": "fixed_qpu_default",
            "sqa_beta_scales": ";".join(
                f"{candidate.beta_scale:g}"
                for candidate in _shared_sqa_schedule_catalog()
            ),
            "sqa_num_sweeps": DEFAULT_SQA_NUM_SWEEPS,
            "sqa_schedule_catalog": catalog_ids,
            "sqa_selected_schedules": selected_ids,
            "sqa_anneal_time": float(
                fixed_schedule.total_schedule_time
            ),
        }
    ]


def _progress_totals(
    *,
    family_sizes: dict[str, list[int]],
    num_instances: int,
    projected_candidates: dict[
        tuple[str, str],
        tuple[ProjectionComboChoice, ...],
    ],
    num_projected_methods: int,
) -> BaselineComparisonProgressTotals:
    """Return aggregate counters for the baseline-comparison TUI."""
    total_instances = int(
        sum(
            len(family_sizes.get(family, []))
            for family in FAMILY_ORDER
        )
        * num_instances
    )
    total_tuning_jobs = int(
        sum(
            1
            for family in FAMILY_ORDER
            if family_sizes.get(family)
        )
        + sum(
            len(candidates)
            for candidates in projected_candidates.values()
        )
    )
    total_component_builds = int(
        total_instances * num_projected_methods
    )
    total_baseline_methods = int(
        total_instances * len(DEFAULT_METHODS)
    )
    return BaselineComparisonProgressTotals(
        tuning_jobs=total_tuning_jobs,
        component_builds=total_component_builds,
        baseline_methods=total_baseline_methods,
        instances=total_instances,
    )


def _default_family_sizes() -> dict[str, list[int]]:
    """Return the fixed family/size protocol used by this experiment."""
    return _shared_default_family_sizes()


def _projected_methods() -> tuple[str, ...]:
    """Return the projected-method variants compared downstream."""
    return _shared_projected_methods()


def _add_common_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Populate the shared CLI flags used by all entry points."""
    return _shared_add_common_arguments(
        parser,
        include_qaoa_selection_rule=False,
        output_dir=DEFAULT_OUTPUT_DIR,
        seed=DEFAULT_SEED,
        num_instances=DEFAULT_NUM_INSTANCES,
        progress_ui_choices=PROGRESS_UI_CHOICES,
        progress_ui_default=DEFAULT_PROGRESS_UI,
        projection_measure_default=DEFAULT_PROJECTION_MEASURE,
        projection_penalty_template_default=DEFAULT_PROJECTION_PENALTY_TEMPLATE,
        projection_selection_modes=PROJECTION_SELECTION_MODES,
        projection_selection_mode_default=DEFAULT_PROJECTION_SELECTION_MODE,
    )


def _activate_progress_ui(
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


def _deactivate_progress_ui(
    tui: BaselineComparisonTui | None,
) -> None:
    """Reset the shared baseline-comparison progress machinery."""
    global _ACTIVE_TUI
    global _NESTED_PROGRESS_ENABLED
    if tui is not None:
        tui.close()
    _ACTIVE_TUI = None
    _NESTED_PROGRESS_ENABLED = True


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the intentionally small CLI surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(parser)
    add_family_selection_arguments(
        parser, include_sizes=True
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=None,
        help=(
            "Optional seed manifest CSV used to construct one matched "
            "instance set for evaluation."
        ),
    )
    parser.add_argument(
        "--cplex-reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional CPLEX reference CSV. Defaults to the manifest-sidecar "
            "reference when --instance-manifest is provided, otherwise "
            "`data/classical_baselines/cplex_optima.csv`."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSVs and run metadata instead of merging.",
    )
    return parser


def build_comparison_argument_parser() -> (
    argparse.ArgumentParser
):
    """Create the CLI for the baseline-comparison entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the logical-SQA method comparison using previously tuned "
            "penalty multipliers."
        )
    )
    _add_common_arguments(parser)
    add_family_selection_arguments(
        parser, include_sizes=True
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=None,
        help=(
            "Optional seed manifest CSV used to construct one matched "
            "instance set for evaluation."
        ),
    )
    parser.add_argument(
        "--cplex-reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional CPLEX reference CSV. Defaults to the manifest-sidecar "
            "reference when --instance-manifest is provided, otherwise "
            "`data/classical_baselines/cplex_optima.csv`."
        ),
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=DEFAULT_TUNING_DIR,
        help=(
            "Directory containing the tuning CSV outputs. Defaults to "
            "`experiments/tunings`."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSVs and run metadata instead of merging.",
    )
    return parser


def run_comparison_experiment(
    args: argparse.Namespace,
    *,
    tuning_outputs: TuningRunOutputs | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Run only the logical-SQA baseline comparison stage."""
    active_families = selected_families_from_args(args)
    family_sizes = selected_family_sizes_from_args(args)
    output_dir = args.output_dir
    tuning_dir = (
        getattr(args, "tuning_dir", None)
        or DEFAULT_TUNING_DIR
    )
    # Prefer the manifest-sidecar CPLEX CSV when a manifest is used or
    # when the project-default shared manifest exists and no explicit
    # instance manifest was provided on the CLI.
    instance_manifest_arg = getattr(args, "instance_manifest", None)
    if instance_manifest_arg is None:
        default_manifest = EXPERIMENTS_DIR / "manifests" / "shared_eval_seed_manifest.csv"
        if default_manifest.exists():
            instance_manifest_arg = default_manifest
            _log(
                f"auto-using instance manifest {instance_manifest_arg} (found project default)"
            )
    cplex_reference_path = resolve_cplex_reference_path(
        getattr(args, "cplex_reference_csv", None),
        instance_manifest=instance_manifest_arg,
    )
    # Keep args.instance_manifest in sync with the resolved manifest so the rest
    # of this run uses the same manifest (and the corresponding CPLEX sidecar).
    args.instance_manifest = instance_manifest_arg
    if tuning_outputs is None:
        tuning_outputs = _shared_load_tuning_outputs(
            tuning_dir
        )
    projected_methods = _projected_methods()
    cplex_reference_index = load_cplex_reference_index(
        cplex_reference_path
    )

    missing_up_families = [
        family
        for family in active_families
        if family
        not in tuning_outputs.tuned_unbalanced_params
    ]
    if missing_up_families:
        raise RuntimeError(
            "missing unbalanced tuning rows for "
            f"{', '.join(missing_up_families)} in "
            f"{tuning_dir / 'unbalanced_penalty_tuning_summary.csv'}; "
            "regenerate tuning artifacts for those families before running "
            "this command"
        )

    missing_projected_configs = [
        f"{family}/{method}"
        for family in active_families
        for method in projected_methods
        if (family, method)
        not in tuning_outputs.selected_projected_configs
    ]
    if missing_projected_configs:
        raise RuntimeError(
            "missing projected tuning rows for "
            f"{', '.join(missing_projected_configs)} in "
            f"{tuning_dir / 'projected_penalty_tuning_summary.csv'}; "
            "regenerate tuning artifacts for those families before running "
            "this command"
        )

    chunk_size = DEFAULT_CHUNK_SIZE
    measure_lam = DEFAULT_MEASURE_LAM
    sample_cap_log2 = DEFAULT_PROJECTION_SAMPLE_CAP_LOG2
    projection_reg = DEFAULT_PROJECTION_REG
    projected_standardize = DEFAULT_PROJECTED_STANDARDIZE
    pegasus_size = DEFAULT_PEGASUS_SIZE
    components_cache: dict[
        tuple[object, ...],
        ProjectedPenaltyComponents,
    ] = {}
    get_problem_batch = _shared_problem_batch_getter(
        base_seed=args.seed,
        num_instances=args.num_instances,
        instance_manifest=args.instance_manifest,
        family_sizes=family_sizes,
    )
    fixed_schedule = _fixed_sqa_schedule()

    total_instances = sum(
        len(get_problem_batch(family, size))
        for family in FAMILY_ORDER
        for size in family_sizes[family]
    )
    tui = _activate_progress_ui(
        mode=args.progress_ui,
        totals=BaselineComparisonProgressTotals(
            component_builds=total_instances
            * len(projected_methods),
            baseline_methods=total_instances
            * len(DEFAULT_METHODS),
            instances=total_instances,
        ),
    )
    _set_progress_status(
        stage="initializing",
        activity="starting comparison",
        family=None,
        size=None,
        instance_index=None,
        total_instances=None,
        method=None,
        measure=None,
        template=None,
        candidate_index=None,
        total_candidates=None,
        detail="starting baseline comparison",
    )

    try:
        ensure_run_metadata(
            output_dir,
            {
                "base_seed": int(args.seed),
                "num_instances": int(args.num_instances),
                "instance_manifest": (
                    None
                    if args.instance_manifest is None
                    else str(
                        Path(
                            args.instance_manifest
                        ).resolve()
                    )
                ),
                "cplex_reference_csv": str(
                    cplex_reference_path.resolve()
                ),
                "tuning_dir": str(
                    Path(tuning_dir).resolve()
                ),
                "sqa_schedule_id": fixed_schedule.schedule_id,
                "sqa_schedule_kind": fixed_schedule.schedule_kind,
                "sqa_schedule_total_time": fixed_schedule.total_schedule_time,
            },
            force=bool(getattr(args, "force", False)),
        )
        if args.instance_manifest is None:
            _log(
                "using generated evaluation instances from "
                f"seed={args.seed} with num_instances={args.num_instances}"
            )
        else:
            _log(
                "using matched evaluation instances from manifest "
                f"{Path(args.instance_manifest).resolve()}"
            )
        _log(
            f"using CPLEX references from {cplex_reference_path}"
        )
        cop_rows: list[dict[str, object]] = []
        used_combo_rows: list[dict[str, object]] = []

        for family in active_families:
            tuned_up = (
                tuning_outputs.tuned_unbalanced_params[
                    family
                ]
            )
            projected_config_map = {
                method: tuning_outputs.selected_projected_configs[
                    (family, method)
                ]
                for method in projected_methods
            }
            projected_multiplier_map = {
                method: projected_config_map[method].tuning
                for method in projected_methods
            }

            for size in family_sizes[family]:
                for method in DEFAULT_METHODS:
                    if method == "unbalanced":
                        used_combo_rows.append(
                            {
                                "family": family,
                                "size": size,
                                "method": "unbalanced",
                                "normalization_regime": (
                                    tuned_up.normalization_regime
                                    or _UP_NORMALIZATION_REGIME
                                ),
                                "per_constraint_standardization": (
                                    True
                                    if tuned_up.per_constraint_standardization
                                    is None
                                    else bool(
                                        tuned_up.per_constraint_standardization
                                    )
                                ),
                                "up_equality_multiplier": (
                                    tuned_up.up_equality_multiplier
                                ),
                                "up_inequality_multiplier": (
                                    tuned_up.up_inequality_multiplier
                                ),
                                "up_lambda1_shape": tuned_up.up_lambda1_shape,
                                "up_lambda2_shape": tuned_up.up_lambda2_shape,
                                "up_lambda_gauge": tuned_up.up_lambda_gauge,
                                "up_global_multiplier": tuned_up.global_multiplier,
                                "up_lambda0": tuned_up.lambda0,
                                "up_lambda1": tuned_up.lambda1,
                                "up_lambda2": tuned_up.lambda2,
                                "sqa_schedule_id": fixed_schedule.schedule_id,
                                "sqa_schedule_kind": fixed_schedule.schedule_kind,
                                "sqa_beta_scale": fixed_schedule.beta_scale,
                                "sqa_schedule_total_time": (
                                    fixed_schedule.total_schedule_time
                                ),
                            }
                        )
                    else:
                        sel_config = (
                            projected_config_map.get(method)
                        )
                        proj_mult = (
                            projected_multiplier_map.get(
                                method
                            )
                        )
                        if (
                            sel_config is None
                            or proj_mult is None
                        ):
                            continue
                        row = {
                            "family": family,
                            "size": size,
                            "method": method,
                            "normalization_regime": _UP_NORMALIZATION_REGIME,
                            "per_constraint_standardization": (
                                projected_standardize
                            ),
                            "projection_method": sel_config.projection_method,
                            "projection_measure": sel_config.measure_name,
                            "projection_penalty_template": sel_config.penalty_template,
                            "projection_selection_mode": sel_config.selection_mode,
                            "projection_selection_source": sel_config.selection_source,
                            "projection_candidate_rank": int(
                                sel_config.candidate_rank
                            ),
                            "projected_equality_multiplier": (
                                proj_mult.equality_multiplier
                            ),
                            "projected_inequality_multiplier": (
                                proj_mult.inequality_multiplier
                            ),
                            "projected_standardize": projected_standardize,
                            "sqa_schedule_id": fixed_schedule.schedule_id,
                            "sqa_schedule_kind": fixed_schedule.schedule_kind,
                            "sqa_beta_scale": fixed_schedule.beta_scale,
                            "sqa_schedule_total_time": (
                                fixed_schedule.total_schedule_time
                            ),
                        }
                        row.update(
                            _projected_selection_utils.template_row_fields(
                                sel_config.penalty_template,
                                sel_config.penalty_template_kwargs,
                            )
                        )
                        used_combo_rows.append(row)

                problems = get_problem_batch(family, size)
                _set_progress_status(
                    stage="evaluating",
                    activity="starting evaluation batch",
                    family=family,
                    size=size,
                    instance_index=None,
                    total_instances=len(problems),
                    method=None,
                    measure=None,
                    template=None,
                    candidate_index=None,
                    total_candidates=None,
                    detail="starting instance evaluation batch",
                )
                _log(
                    f"evaluating {family} size {size} with "
                    f"{len(problems)} instance(s)"
                )

                for instance_index, problem in enumerate(
                    problems
                ):
                    problem_seed = int(
                        problem.metadata["problem_seed"]
                    )
                    _set_progress_status(
                        stage="evaluating",
                        activity="loading CPLEX reference",
                        family=family,
                        size=size,
                        instance_index=instance_index,
                        total_instances=len(problems),
                        method=None,
                        measure=None,
                        template=None,
                        detail="loading constrained optimum from reference CSV",
                    )
                    reference = cplex_reference_for_synthetic_problem(
                        index=cplex_reference_index,
                        family=family,
                        size=size,
                        problem_seed=problem_seed,
                        problem=problem,
                        atol=DEFAULT_REFERENCE_ATOL,
                    )
                    projected_components_map: dict[
                        str,
                        ProjectedPenaltyComponents,
                    ] = {}

                    for method in projected_methods:
                        selected_config = (
                            projected_config_map[method]
                        )
                        _set_progress_status(
                            stage="building_components",
                            activity="computing projection",
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            total_instances=len(problems),
                            method=method,
                            measure=selected_config.measure_name,
                            template=selected_config.penalty_template,
                            detail="building projected penalty components",
                        )
                        cache_key = _shared_tuning_support.projected_components_cache_key(
                            projection_method=selected_config.projection_method,
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            measure_name=selected_config.measure_name,
                            measure_lam=measure_lam,
                            penalty_template=selected_config.penalty_template,
                            penalty_template_kwargs=(
                                selected_config.penalty_template_kwargs
                            ),
                            standardize=projected_standardize,
                        )
                        if (
                            cache_key
                            not in components_cache
                        ):
                            components_cache[cache_key] = (
                                _shared_tuning_support.build_projected_components(
                                    problem,
                                    projection_method=selected_config.projection_method,
                                    family=family,
                                    size=size,
                                    instance_index=instance_index,
                                    base_seed=args.seed,
                                    measure_name=selected_config.measure_name,
                                    measure_lam=measure_lam,
                                    penalty_template=selected_config.penalty_template,
                                    penalty_template_kwargs=(
                                        selected_config.penalty_template_kwargs
                                    ),
                                    pegasus_size=pegasus_size,
                                    sample_cap_log2=sample_cap_log2,
                                    chunk_size=chunk_size,
                                    reg=projection_reg,
                                    standardize=projected_standardize,
                                    status_callback=_set_progress_status,
                                )
                            )
                        projected_components_map[method] = (
                            components_cache[cache_key]
                        )
                        _advance_progress_counter(
                            "component_builds", 1
                        )

                    method_qubos: dict[
                        str,
                        tuple[
                            np.ndarray, np.ndarray, float
                        ],
                    ] = {
                        "unbalanced": _unbalanced_qubo(
                            problem, tuned_up
                        ),
                        "projected_full": _shared_tuning_support.projected_full_qubo(
                            problem,
                            projected_components_map[
                                "projected_full"
                            ],
                            projected_multiplier_map[
                                "projected_full"
                            ],
                        ),
                        "projected_up_support": _shared_tuning_support.projected_full_qubo(
                            problem,
                            projected_components_map[
                                "projected_up_support"
                            ],
                            projected_multiplier_map[
                                "projected_up_support"
                            ],
                        ),
                        "projected_pegasus": _shared_tuning_support.projected_full_qubo(
                            problem,
                            projected_components_map[
                                "projected_pegasus"
                            ],
                            projected_multiplier_map[
                                "projected_pegasus"
                            ],
                        ),
                        "projected_chimera": _shared_tuning_support.projected_full_qubo(
                            problem,
                            projected_components_map[
                                "projected_chimera"
                            ],
                            projected_multiplier_map[
                                "projected_chimera"
                            ],
                        ),
                        "projected_zephyr": _shared_tuning_support.projected_full_qubo(
                            problem,
                            projected_components_map[
                                "projected_zephyr"
                            ],
                            projected_multiplier_map[
                                "projected_zephyr"
                            ],
                        ),
                    }

                    _log(
                        f"instance {instance_index + 1}/{len(problems)} in "
                        f"{family} n={size}: {problem.num_variables} qubits, "
                        f"reference optimum={reference.optimum_objective:0.6g}"
                    )
                    sqa_num_reads = DEFAULT_SQA_NUM_READS
                    num_inequality_quadratic_terms = (
                        _num_inequality_quadratic_terms(
                            problem
                        )
                    )

                    for method in DEFAULT_METHODS:
                        _set_progress_status(
                            stage="baselines",
                            activity="preparing baseline method",
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            total_instances=len(problems),
                            method=method,
                            measure=None,
                            template=None,
                            detail="running logical SQA baseline",
                        )
                        quadratic, linear, const = (
                            method_qubos[method]
                        )
                        normalization_scale = _qubo_normalization_scale(
                            quadratic,
                            linear,
                            num_variables=problem.num_variables,
                        )
                        _set_progress_status(
                            activity="running SQA",
                            detail="running logical SQA with selected family schedule",
                        )
                        annealer_input = _scaled_annealer_input(
                            problem,
                            quadratic,
                            linear,
                            const,
                            normalization_scale=normalization_scale,
                            chunk_size=chunk_size,
                        )
                        sqa_metrics = _logical_annealer_baseline_result(
                            problem,
                            annealer_input,
                            reference=reference,
                            family=family,
                            size=size,
                            instance_index=instance_index,
                            method=method,
                            base_seed=args.seed,
                            num_reads=sqa_num_reads,
                            schedule=fixed_schedule,
                            atol=DEFAULT_REFERENCE_ATOL,
                        )

                        projected_components = (
                            projected_components_map.get(
                                method
                            )
                        )
                        selected_config = (
                            projected_config_map.get(method)
                        )
                        projected_multipliers = (
                            projected_multiplier_map.get(
                                method
                            )
                        )
                        projection_method = None
                        penalty_template = None
                        penalty_template_kwargs: dict[
                            str, float
                        ] = {}
                        projection_measure = None
                        projection_selection_mode = None
                        projection_selection_source = None
                        projection_candidate_rank = None
                        if projected_components is not None:
                            if selected_config is None:
                                raise RuntimeError(
                                    f"missing selected config for {family}/{method}"
                                )
                            projection_method = (
                                selected_config.projection_method
                            )
                            penalty_template = (
                                selected_config.penalty_template
                            )
                            penalty_template_kwargs = (
                                selected_config.penalty_template_kwargs
                            )
                            projection_measure = (
                                selected_config.measure_name
                            )
                            projection_selection_mode = (
                                selected_config.selection_mode
                            )
                            projection_selection_source = (
                                selected_config.selection_source
                            )
                            projection_candidate_rank = int(
                                selected_config.candidate_rank
                            )

                        cop_rows.append(
                            {
                                "family": family,
                                "size": size,
                                "instance_index": instance_index,
                                "method": method,
                                **_problem_provenance_fields(
                                    problem
                                ),
                                **_projection_regime_fields(
                                    method
                                ),
                                "num_variables": problem.num_variables,
                                "num_states": problem.num_states,
                                "reference_optimum_objective": (
                                    float(
                                        reference.optimum_objective
                                    )
                                ),
                                "reference_optimum_source": reference.optimum_source,
                                "reference_objective_sense": (
                                    reference.objective_sense
                                ),
                                "reference_match_tolerance": DEFAULT_REFERENCE_ATOL,
                                "num_inequality_quadratic_terms": (
                                    num_inequality_quadratic_terms
                                ),
                                "sqa_logical_cop": sqa_metrics.logical_cop,
                                "sqa_num_reads": sqa_metrics.num_reads,
                                "sqa_fea": sqa_metrics.feasible_rate,
                                "sqa_gap": sqa_metrics.objective_gap,
                                "sqa_schedule_id": sqa_metrics.schedule_id,
                                "sqa_schedule_kind": sqa_metrics.schedule_kind,
                                "sqa_beta_scale": sqa_metrics.beta_scale,
                                "sqa_schedule_total_time": (
                                    sqa_metrics.total_schedule_time
                                ),
                                "sqa_anneal_schedule": _shared_anneal_schedule_json(
                                    sqa_metrics.anneal_schedule
                                ),
                                "projected_sample_size": (
                                    projected_components.sample_size
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projected_num_quadratic_couplers": (
                                    projected_components.num_quadratic_couplers
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projection_method": (
                                    projection_method
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "comparison_method": (
                                    method
                                    if method.startswith(
                                        "projected_"
                                    )
                                    else None
                                ),
                                "projection_measure": (
                                    projection_measure
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projection_measure_lam": (
                                    None
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projection_penalty_template": (
                                    penalty_template
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                **(
                                    _projected_selection_utils.template_row_fields(
                                        penalty_template,
                                        penalty_template_kwargs,
                                    )
                                    if projected_components
                                    is not None
                                    else {}
                                ),
                                "projection_selection_mode": (
                                    projection_selection_mode
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projection_selection_source": (
                                    projection_selection_source
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projection_candidate_rank": (
                                    projection_candidate_rank
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projected_standardize": (
                                    projected_standardize
                                    if projected_components
                                    is not None
                                    else None
                                ),
                                "projected_equality_multiplier": (
                                    projected_multipliers.equality_multiplier
                                    if projected_multipliers
                                    is not None
                                    else None
                                ),
                                "projected_inequality_multiplier": (
                                    projected_multipliers.inequality_multiplier
                                    if projected_multipliers
                                    is not None
                                    else None
                                ),
                                "normalization_regime": (
                                    (
                                        tuned_up.normalization_regime
                                        or _UP_NORMALIZATION_REGIME
                                    )
                                    if method
                                    == "unbalanced"
                                    else _UP_NORMALIZATION_REGIME
                                ),
                                "per_constraint_standardization": (
                                    (
                                        True
                                        if tuned_up.per_constraint_standardization
                                        is None
                                        else bool(
                                            tuned_up.per_constraint_standardization
                                        )
                                    )
                                    if method
                                    == "unbalanced"
                                    else projected_standardize
                                ),
                                "qubo_normalization_scale": (
                                    normalization_scale
                                ),
                                "up_equality_multiplier": (
                                    tuned_up.up_equality_multiplier
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_inequality_multiplier": (
                                    tuned_up.up_inequality_multiplier
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda1_shape": (
                                    tuned_up.up_lambda1_shape
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda2_shape": (
                                    tuned_up.up_lambda2_shape
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda_gauge": (
                                    tuned_up.up_lambda_gauge
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_global_multiplier": (
                                    tuned_up.global_multiplier
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda0": (
                                    tuned_up.lambda0
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda1": (
                                    tuned_up.lambda1
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                                "up_lambda2": (
                                    tuned_up.lambda2
                                    if method
                                    == "unbalanced"
                                    else None
                                ),
                            }
                        )
                        _advance_progress_counter(
                            "baseline_methods", 1
                        )

                    _advance_progress_counter(
                        "instances", 1
                    )

        _set_progress_status(
            stage="writing",
            activity="writing outputs",
            family=None,
            size=None,
            instance_index=None,
            total_instances=None,
            method=None,
            measure=None,
            template=None,
            candidate_index=None,
            total_candidates=None,
            detail="writing comparison CSV summaries",
        )

        cop_instance_path = (
            output_dir / "cop_instance_summary.csv"
        )
        cop_aggregate_path = (
            output_dir / "cop_aggregate_summary.csv"
        )
        cop_metadata_path = (
            output_dir / "cop_metadata_summary.csv"
        )
        cop_metadata_rows = _cop_metadata_rows(
            active_families
        )
        if getattr(args, "force", False):
            merged_cop_rows = list(cop_rows)
            merged_cop_rows.sort(
                key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    int(row["instance_index"]),
                    str(row["method"]),
                )
            )
        else:
            merged_cop_rows = merge_csv_rows(
                cop_instance_path,
                cop_rows,
                key_fields=(
                    "family",
                    "size",
                    "instance_index",
                    "method",
                ),
                sort_key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    int(row["instance_index"]),
                    str(row["method"]),
                ),
            )
        cop_aggregate_rows = _aggregate_rows(
            merged_cop_rows
        )
        _write_rows_csv(
            cop_metadata_path, cop_metadata_rows
        )
        _write_rows_csv(cop_instance_path, merged_cop_rows)
        _write_rows_csv(
            cop_aggregate_path, cop_aggregate_rows
        )
        _log(f"wrote {cop_metadata_path}")
        _log(f"wrote {cop_instance_path}")
        _log(f"wrote {cop_aggregate_path}")

        used_combo_path = (
            tuning_dir
            / "used_penalty_measure_multipliers_by_size.csv"
        )
        if getattr(args, "force", False):
            merged_used_combo_rows = list(used_combo_rows)
            merged_used_combo_rows.sort(
                key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    str(row["method"]),
                )
            )
        else:
            merged_used_combo_rows = merge_csv_rows(
                used_combo_path,
                used_combo_rows,
                key_fields=("family", "size", "method"),
                sort_key=lambda row: (
                    FAMILY_ORDER.index(str(row["family"])),
                    int(row["size"]),
                    str(row["method"]),
                ),
            )
        _write_rows_csv(
            used_combo_path, merged_used_combo_rows
        )
        _log(f"wrote {used_combo_path}")

        return {
            "cop_rows": merged_cop_rows,
            "used_combo_rows": merged_used_combo_rows,
        }
    finally:
        _deactivate_progress_ui(tui)


def main() -> None:
    """Run the baseline comparison using previously tuned outputs."""
    parser = build_comparison_argument_parser()
    args = parser.parse_args()
    run_comparison_experiment(args)


if __name__ == "__main__":
    main()
