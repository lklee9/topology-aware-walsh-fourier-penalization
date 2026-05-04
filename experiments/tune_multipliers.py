"""Tune the multipliers used by the unbalanced-penalty comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from experiments.experiment_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MEASURE_LAM,
    DEFAULT_NUM_INSTANCES,
    DEFAULT_PEGASUS_SIZE,
    DEFAULT_PROGRESS_UI,
    DEFAULT_PROJECTED_STANDARDIZE,
    DEFAULT_PROJECTION_MEASURE,
    DEFAULT_PROJECTION_PENALTY_TEMPLATE,
    DEFAULT_PROJECTION_REG,
    DEFAULT_PROJECTION_SAMPLE_CAP_LOG2,
    DEFAULT_PROJECTION_SELECTION_MODE,
    DEFAULT_SEED,
    DEFAULT_TUNING_DIR,
    DEFAULT_TUNING_MIN,
    DEFAULT_TUNING_NM_FATOL,
    DEFAULT_TUNING_NM_MAXITER,
    DEFAULT_TUNING_NM_START_POINTS,
    DEFAULT_TUNING_NM_XATOL,
    DEFAULT_TUNING_SIZES,
    FAMILY_ORDER,
    PROGRESS_UI_CHOICES,
    PROJECTION_SELECTION_MODES,
)
from experiments.utils.baseline_common import (
    add_common_arguments as _shared_add_common_arguments,
)
from experiments.utils.baseline_common import (
    projected_methods as _shared_projected_methods,
)
from experiments.utils.baseline_progress import (
    BaselineComparisonProgressTotals,
    activate_progress_ui,
    advance_progress_counter,
    deactivate_progress_ui,
    log,
    set_progress_status,
)
from experiments.utils.driver_common import (
    problem_batch_getter,
    write_rows_csv,
)
from experiments.utils.family_cli import (
    add_family_selection_arguments,
    selected_families_from_args,
)
from experiments.utils.merge_outputs import (
    ensure_run_metadata,
    merge_csv_rows,
)
from experiments.utils.projected_method_selection import (
    ProjectionComboChoice,
    family_projection_combo_key,
    load_projection_combo_candidates,
    projected_candidate_spec,
    shared_projected_selection_sort_key,
    template_row_fields,
)
from experiments.utils.tuning_core import (
    tune_projected_multipliers,
    tune_unbalanced_parameters,
)
from experiments.utils.tuning_models import TuningRunOutputs
from experiments.utils.tuning_summary import (
    write_projected_tuning_summary_csv,
)

TOPOLOGY_PROJECTED_METHODS = (
    "projected_pegasus",
    "projected_chimera",
    "projected_zephyr",
)


def _default_tuning_sizes() -> dict[str, int]:
    """Return the anchor size used to tune each family."""
    return {
        family: int(DEFAULT_TUNING_SIZES[family])
        for family in FAMILY_ORDER
    }


def _projected_methods() -> tuple[str, ...]:
    """Return the projected-method variants compared downstream."""
    return _shared_projected_methods()


def _topology_projected_methods() -> tuple[str, ...]:
    """Return projected methods that fit onto hardware topologies."""
    return TOPOLOGY_PROJECTED_METHODS


def _selected_projected_methods(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Return the projected methods requested for this tuning run."""
    if bool(getattr(args, "topology_only", False)):
        return _topology_projected_methods()
    return _projected_methods()


def _is_quadratic_projection_method(
    row: dict[str, object],
) -> bool:
    """Return whether one row belongs to a removed quadratic method."""
    return (
        str(row.get("method", ""))
        .strip()
        .endswith("_quadratic")
    )


def _drop_quadratic_projection_methods(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Drop obsolete quadratic projected-method rows from merged summaries."""
    return [
        row
        for row in rows
        if not _is_quadratic_projection_method(row)
    ]


def build_tuning_argument_parser() -> (
    argparse.ArgumentParser
):
    """Create the CLI for the tuning-only entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Tune the unbalanced and projected penalty "
            "multipliers used by the unbalanced-penalization "
            "comparison."
        )
    )
    _shared_add_common_arguments(
        parser,
        include_qaoa_selection_rule=False,
        output_dir=DEFAULT_TUNING_DIR,
        seed=DEFAULT_SEED,
        num_instances=DEFAULT_NUM_INSTANCES,
        progress_ui_choices=PROGRESS_UI_CHOICES,
        progress_ui_default=DEFAULT_PROGRESS_UI,
        projection_measure_default=DEFAULT_PROJECTION_MEASURE,
        projection_penalty_template_default=(
            DEFAULT_PROJECTION_PENALTY_TEMPLATE
        ),
        projection_selection_modes=PROJECTION_SELECTION_MODES,
        projection_selection_mode_default=(
            DEFAULT_PROJECTION_SELECTION_MODE
        ),
    )
    add_family_selection_arguments(
        parser, include_sizes=False
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run metadata instead of rejecting mismatch.",
    )
    parser.add_argument(
        "--topology-only",
        "--topology-projections-only",
        action="store_true",
        dest="topology_only",
        help=(
            "Tune only projected_pegasus, projected_chimera, and "
            "projected_zephyr, then merge those rows into existing tuning "
            "summaries."
        ),
    )
    return parser


def _projected_candidates(
    *,
    families: tuple[str, ...],
    methods: tuple[str, ...],
    selection_mode: str,
    fixed_measure_name: str,
    fixed_penalty_template: str,
) -> tuple[
    dict[
        tuple[str, str], tuple[ProjectionComboChoice, ...]
    ],
    dict[str, tuple[ProjectionComboChoice, ...]],
]:
    """Materialize the projected candidate lists used during tuning."""
    projected_candidates_map: dict[
        tuple[str, str],
        tuple[ProjectionComboChoice, ...],
    ] = {}
    family_projected_candidates: dict[
        str,
        tuple[ProjectionComboChoice, ...],
    ] = {}
    if not methods:
        raise ValueError(
            "at least one projected method must be selected"
        )
    for family in families:
        for method in methods:
            projected_candidates_map[(family, method)] = (
                tuple(
                    load_projection_combo_candidates(
                        selection_mode=selection_mode,
                        family=family,
                        method=method,
                        fixed_measure_name=fixed_measure_name,
                        fixed_penalty_template=fixed_penalty_template,
                    )
                )
            )
        family_projected_candidates[family] = (
            projected_candidates_map[(family, methods[0])]
        )
    return (
        projected_candidates_map,
        family_projected_candidates,
    )


def run_tuning_experiment(
    args: argparse.Namespace,
) -> TuningRunOutputs:
    """Run only the multiplier-tuning stage and persist its CSV outputs."""
    tuning_sizes = _default_tuning_sizes()
    active_families = selected_families_from_args(args)
    methods = _selected_projected_methods(args)
    tune_unbalanced = not bool(
        getattr(args, "topology_only", False)
    )
    output_dir = args.output_dir
    chunk_size = DEFAULT_CHUNK_SIZE
    measure_lam = DEFAULT_MEASURE_LAM
    sample_cap_log2 = DEFAULT_PROJECTION_SAMPLE_CAP_LOG2
    projection_reg = DEFAULT_PROJECTION_REG
    projected_standardize = DEFAULT_PROJECTED_STANDARDIZE
    tuning_min = DEFAULT_TUNING_MIN
    tuning_start_points_per_dim = (
        DEFAULT_TUNING_NM_START_POINTS
    )
    tuning_nelder_mead_maxiter = DEFAULT_TUNING_NM_MAXITER
    tuning_nelder_mead_xatol = DEFAULT_TUNING_NM_XATOL
    tuning_nelder_mead_fatol = DEFAULT_TUNING_NM_FATOL
    tuning_min_clip = False
    pegasus_size = DEFAULT_PEGASUS_SIZE
    components_cache: dict[tuple[object, ...], object] = {}
    get_problem_batch = problem_batch_getter(
        base_seed=args.seed,
        num_instances=args.num_instances,
    )
    (
        projected_candidates_map,
        family_projected_candidates,
    ) = _projected_candidates(
        families=active_families,
        methods=methods,
        selection_mode=args.projection_selection_mode,
        fixed_measure_name=args.projection_measure,
        fixed_penalty_template=args.projection_penalty_template,
    )
    total_tuning_jobs = int(
        (len(active_families) if tune_unbalanced else 0)
        + sum(
            len(candidates)
            for candidates in projected_candidates_map.values()
        )
    )
    tui = activate_progress_ui(
        mode=args.progress_ui,
        totals=BaselineComparisonProgressTotals(
            tuning_jobs=total_tuning_jobs,
        ),
    )

    projected_tuning_rows: list[dict[str, object]] = []
    projected_selection_rows: list[dict[str, object]] = []
    up_tuning_rows: list[dict[str, object]] = []
    selected_projected_configs: dict[
        tuple[str, str], object
    ] = {}
    tuned_unbalanced_params: dict[str, object] = {}

    set_progress_status(
        stage="initializing",
        activity="starting tuning",
        family=None,
        size=None,
        instance_index=None,
        total_instances=None,
        method=None,
        measure=None,
        template=None,
        candidate_index=None,
        total_candidates=None,
        detail="starting multiplier tuning",
    )

    try:
        ensure_run_metadata(
            output_dir,
            {
                "base_seed": int(args.seed),
                "num_instances": int(args.num_instances),
                "projection_selection_mode": str(
                    args.projection_selection_mode
                ),
                "projection_measure": str(
                    args.projection_measure
                ),
                "projection_penalty_template": str(
                    args.projection_penalty_template
                ),
            },
            force=bool(getattr(args, "force", False)),
        )
        for family in active_families:
            anchor_size = tuning_sizes[family]
            anchor_problems = get_problem_batch(
                family,
                anchor_size,
            )
            if tune_unbalanced:
                set_progress_status(
                    stage="tuning_unbalanced",
                    activity="tuning unbalanced penalty",
                    family=family,
                    size=anchor_size,
                    instance_index=None,
                    total_instances=len(anchor_problems),
                    method="unbalanced",
                    measure=None,
                    template=None,
                    candidate_index=None,
                    total_candidates=None,
                    detail="tuning global unbalanced multiplier",
                )
                log(
                    f"tuning unbalanced multiplier for {family} "
                    f"on anchor size {anchor_size} using "
                    f"{len(anchor_problems)} instance(s)"
                )
                tuned_up, up_records, _, _ = (
                    tune_unbalanced_parameters(
                        family,
                        anchor_problems,
                        anchor_size=anchor_size,
                        chunk_size=chunk_size,
                        param_min=tuning_min,
                        start_points_per_dim=(
                            tuning_start_points_per_dim
                        ),
                        nelder_mead_maxiter=(
                            tuning_nelder_mead_maxiter
                        ),
                        nelder_mead_xatol=(
                            tuning_nelder_mead_xatol
                        ),
                        nelder_mead_fatol=(
                            tuning_nelder_mead_fatol
                        ),
                        clip_minimum=tuning_min_clip,
                    )
                )
                up_records_path = (
                    output_dir
                    / "up_tuning_records"
                    / f"{family}_anchor_{anchor_size}.csv"
                )
                write_rows_csv(up_records_path, up_records)
                log(f"wrote {up_records_path}")
                tuned_unbalanced_params[family] = tuned_up
                up_tuning_rows.append(tuned_up.as_row())
                advance_progress_counter("tuning_jobs", 1)
                log(
                    f"selected unbalanced multiplier for {family}: "
                    f"equality_multiplier={tuned_up.up_equality_multiplier}, "
                    "inequality_multiplier="
                    f"{tuned_up.up_inequality_multiplier:.6g}, "
                    f"lambda1_shape={tuned_up.up_lambda1_shape:.6g}, "
                    f"lambda2_shape={tuned_up.up_lambda2_shape:.6g}, "
                    f"lambda_gauge={tuned_up.up_lambda_gauge}, "
                    f"objective={tuned_up.objective_value:.6g}, "
                    f"success={tuned_up.success}"
                )
            else:
                log(
                    f"topology-only mode: skipping unbalanced tuning for {family}"
                )

            family_candidates = family_projected_candidates[
                family
            ]
            method_candidate_configs: dict[
                str,
                dict[tuple[object, ...], object],
            ] = {method: {} for method in methods}
            summary_rows_by_method_key: dict[
                tuple[str, tuple[object, ...]],
                list[dict[str, object]],
            ] = {}

            for method in methods:
                candidates = projected_candidates_map[
                    (family, method)
                ]
                set_progress_status(
                    stage="tuning_projected",
                    activity="tuning projected penalty",
                    family=family,
                    size=anchor_size,
                    instance_index=None,
                    total_instances=len(anchor_problems),
                    method=method,
                    measure=None,
                    template=None,
                    candidate_index=None,
                    total_candidates=len(candidates),
                    detail=(
                        f"{len(candidates)} projected combo "
                        "candidate(s)"
                    ),
                )
                log(
                    f"tuning {method} multipliers for {family} "
                    f"on anchor size {anchor_size} using "
                    f"{len(anchor_problems)} instance(s) across "
                    f"{len(candidates)} combo candidate(s)"
                )

                candidate_summary_rows: list[
                    dict[str, object]
                ] = []
                candidate_records: list[
                    dict[str, object]
                ] = []

                for candidate_index, candidate in enumerate(
                    candidates,
                    start=1,
                ):
                    set_progress_status(
                        stage="tuning_projected",
                        activity="tuning projected penalty",
                        family=family,
                        size=anchor_size,
                        instance_index=None,
                        total_instances=len(
                            anchor_problems
                        ),
                        method=method,
                        measure=candidate.measure_name,
                        template=candidate.penalty_template,
                        candidate_index=candidate_index,
                        total_candidates=len(candidates),
                        detail=(
                            f"candidate {candidate_index}/"
                            f"{len(candidates)} "
                            f"template={candidate.penalty_template}"
                        ),
                    )
                    tuned, proj_records, _, _ = (
                        tune_projected_multipliers(
                            method,
                            method,
                            family,
                            anchor_problems,
                            anchor_size=anchor_size,
                            components_cache=components_cache,
                            base_seed=args.seed,
                            measure_name=candidate.measure_name,
                            measure_lam=measure_lam,
                            penalty_template=(
                                candidate.penalty_template
                            ),
                            penalty_template_kwargs=None,
                            pegasus_size=pegasus_size,
                            sample_cap_log2=sample_cap_log2,
                            chunk_size=chunk_size,
                            reg=projection_reg,
                            standardize=projected_standardize,
                            param_min=tuning_min,
                            start_points_per_dim=(
                                tuning_start_points_per_dim
                            ),
                            nelder_mead_maxiter=(
                                tuning_nelder_mead_maxiter
                            ),
                            nelder_mead_xatol=(
                                tuning_nelder_mead_xatol
                            ),
                            nelder_mead_fatol=(
                                tuning_nelder_mead_fatol
                            ),
                            clip_minimum=tuning_min_clip,
                        )
                    )
                    advance_progress_counter(
                        "tuning_jobs", 1
                    )
                    selected_config = projected_candidate_spec(
                        method,
                        family=family,
                        candidate=candidate,
                        selection_mode=(
                            args.projection_selection_mode
                        ),
                        tuned=tuned,
                        default_template=(
                            args.projection_penalty_template
                        ),
                    )
                    summary_row = tuned.as_row()
                    summary_row.update(
                        {
                            "projection_method": (
                                selected_config.projection_method
                            ),
                            "projection_selection_mode": (
                                selected_config.selection_mode
                            ),
                            "projection_selection_source": (
                                selected_config.selection_source
                            ),
                            "projection_candidate_rank": int(
                                selected_config.candidate_rank
                            ),
                            "projection_measure": (
                                selected_config.measure_name
                            ),
                            "projected_standardize": projected_standardize,
                            **template_row_fields(
                                selected_config.penalty_template,
                                selected_config.penalty_template_kwargs,
                            ),
                            "selected_projection_combo": False,
                        }
                    )
                    candidate_summary_rows.append(
                        summary_row
                    )
                    config_key = (
                        family_projection_combo_key(
                            candidate
                        )
                    )
                    method_candidate_configs[method][
                        config_key
                    ] = selected_config
                    summary_rows_by_method_key.setdefault(
                        (method, config_key),
                        [],
                    ).append(summary_row)

                    for record in proj_records:
                        row = dict(record)
                        row.update(
                            {
                                "method": method,
                                "family": family,
                                "anchor_size": anchor_size,
                                "projection_method": (
                                    selected_config.projection_method
                                ),
                                "projection_selection_mode": (
                                    selected_config.selection_mode
                                ),
                                "projection_selection_source": (
                                    selected_config.selection_source
                                ),
                                "projection_candidate_rank": int(
                                    selected_config.candidate_rank
                                ),
                                "projection_measure": (
                                    selected_config.measure_name
                                ),
                                **template_row_fields(
                                    selected_config.penalty_template,
                                    selected_config.penalty_template_kwargs,
                                ),
                            }
                        )
                        candidate_records.append(row)

                proj_records_path = (
                    output_dir
                    / "projected_tuning_records"
                    / f"{family}_{method}_anchor_{anchor_size}.csv"
                )
                write_rows_csv(
                    proj_records_path,
                    candidate_records,
                )
                log(f"wrote {proj_records_path}")
                projected_selection_rows.extend(
                    candidate_summary_rows
                )

            best_family_candidate = None
            best_family_method_configs = None
            best_shared_sort_key: (
                tuple[object, ...] | None
            ) = None
            for family_candidate in family_candidates:
                family_key = family_projection_combo_key(
                    family_candidate
                )
                method_configs_for_candidate: dict[
                    str, object
                ] = {}
                for method in methods:
                    try:
                        method_configs_for_candidate[
                            method
                        ] = method_candidate_configs[
                            method
                        ][
                            family_key
                        ]
                    except KeyError as exc:
                        raise RuntimeError(
                            "missing projected tuning result for "
                            f"{family}/{method} candidate "
                            f"{family_candidate.penalty_template}/"
                            f"{family_candidate.measure_name}"
                        ) from exc

                sort_key = (
                    shared_projected_selection_sort_key(
                        family_candidate,
                        method_configs=(
                            method_configs_for_candidate
                        ),
                    )
                )
                if (
                    best_shared_sort_key is None
                    or sort_key < best_shared_sort_key
                ):
                    best_shared_sort_key = sort_key
                    best_family_candidate = family_candidate
                    best_family_method_configs = (
                        method_configs_for_candidate
                    )

            if (
                best_family_candidate is None
                or best_family_method_configs is None
            ):
                raise RuntimeError(
                    "no shared projected combo was selected "
                    f"for {family}"
                )

            best_family_key = family_projection_combo_key(
                best_family_candidate
            )
            log(
                f"selected shared projected combo for {family}: "
                f"{best_family_candidate.penalty_template}/"
                f"{best_family_candidate.measure_name}"
            )
            for method in methods:
                matching_rows = (
                    summary_rows_by_method_key.get(
                        (method, best_family_key),
                        [],
                    )
                )
                if not matching_rows:
                    raise RuntimeError(
                        "missing selected projected summary row "
                        f"for {family}/{method}"
                    )
                for row in matching_rows:
                    row["selected_projection_combo"] = True
                projected_tuning_rows.extend(matching_rows)
                best_config = best_family_method_configs[
                    method
                ]
                selected_projected_configs[
                    (family, method)
                ] = best_config
                log(
                    f"selected {best_config.penalty_template}/"
                    f"{best_config.measure_name} for "
                    f"{family}/{method}: equality multiplier="
                    f"{best_config.tuning.equality_multiplier:.6g}, "
                    "inequality multiplier="
                    f"{best_config.tuning.inequality_multiplier:.6g}, "
                    f"objective={best_config.tuning.objective_value:.6g}, "
                    f"source={best_config.selection_source}"
                )

        set_progress_status(
            stage="writing",
            activity="writing tuning outputs",
            family=None,
            size=None,
            instance_index=None,
            total_instances=None,
            method=None,
            measure=None,
            template=None,
            candidate_index=None,
            total_candidates=None,
            detail="writing tuning CSV summaries",
        )

        projected_tuning_path = (
            output_dir
            / "projected_penalty_tuning_summary.csv"
        )
        merged_projected_rows = merge_csv_rows(
            projected_tuning_path,
            projected_tuning_rows,
            key_fields=("family", "method"),
        )
        merged_projected_rows = (
            _drop_quadratic_projection_methods(
                merged_projected_rows
            )
        )
        write_projected_tuning_summary_csv(
            projected_tuning_path,
            merged_projected_rows,
        )
        log(f"wrote {projected_tuning_path}")

        projected_selection_path = (
            output_dir
            / "projected_combo_selection_summary.csv"
        )
        merged_selection_rows = merge_csv_rows(
            projected_selection_path,
            projected_selection_rows,
            block_fields=("family", "method"),
        )
        merged_selection_rows = (
            _drop_quadratic_projection_methods(
                merged_selection_rows
            )
        )
        write_rows_csv(
            projected_selection_path, merged_selection_rows
        )
        log(f"wrote {projected_selection_path}")

        up_tuning_path = (
            output_dir
            / "unbalanced_penalty_tuning_summary.csv"
        )
        merged_up_rows = merge_csv_rows(
            up_tuning_path,
            up_tuning_rows,
            key_fields=("family",),
        )
        write_rows_csv(up_tuning_path, merged_up_rows)
        log(f"wrote {up_tuning_path}")

        return TuningRunOutputs(
            projected_tuning_rows=projected_tuning_rows,
            projected_selection_rows=projected_selection_rows,
            up_tuning_rows=up_tuning_rows,
            tuned_unbalanced_params=tuned_unbalanced_params,
            selected_projected_configs=(
                selected_projected_configs
            ),
        )
    finally:
        deactivate_progress_ui(tui)


def main() -> None:
    """Run only the tuning stage and persist its CSV outputs."""
    parser = build_tuning_argument_parser()
    args = parser.parse_args()
    run_tuning_experiment(args)


if __name__ == "__main__":
    main()
