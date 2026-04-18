from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datasets import DatasetExperimentConfig
from .manifold_validation import deterministic_dataframe_sample
from .proposed_setups import PROPOSED_SAMPLER_SETUPS
from .sensitivity_validation import SENSITIVITY_VALIDATION_COLUMNS, _evaluate_variant_config


PARAMETER_SENSITIVITY_DATASET = "adult"
PARAMETER_SENSITIVITY_COLUMNS = SENSITIVITY_VALIDATION_COLUMNS
QUALITY_METRICS = [
    "distribution_similarity_score",
    "utility_lift",
    "discrimination_accuracy",
    "nn_distance_ratio",
]

TARGET_METRIC_VALUES = {
    "discrimination_accuracy": 0.5,
    "nn_distance_ratio": 1.0,
}

LINE_SEARCH_PARAMETER_ORDER = ["n_iterations", "n_components", "nca_fit_sample_size", "lambda_"]
NCA_ONLY_LINE_SEARCH_PARAMETERS = {"n_components", "nca_fit_sample_size"}
LINE_SEARCH_METRIC_LABELS = {
    "distribution_similarity_score": "Distribution score ↑",
    "utility_lift": "Utility lift ↑",
    "discrimination_accuracy": "Discrimination accuracy → 0.5",
    "nn_distance_ratio": "NN ratio → 1.0",
}
LINE_SEARCH_DISPLAY_COLUMNS = [
    "parameter",
    "value",
    "average_quality_rank",
    "distribution_similarity_score",
    "utility_lift",
    "discrimination_accuracy",
    "nn_distance_ratio",
]


def run_adult_parameter_sensitivity(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    max_train_rows: int = 250,
    n_samples: int | None = None,
    n_components_grid: Iterable[int] = tuple(range(1, 11)),
    nca_fit_sample_size_grid: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    lambda_grid: Iterable[float] = tuple(np.linspace(0.1, 2.0, 10)),
    n_iterations_grid: Iterable[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """Run one-at-a-time parameter grids on capped Adult data.

    The function keeps the default setup fixed and changes only one parameter at
    a time. Results are written to ``adult_parameter_sensitivity.csv``.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if config.dataset_name != PARAMETER_SENSITIVITY_DATASET or dataframe.empty:
        report = pd.DataFrame(columns=PARAMETER_SENSITIVITY_COLUMNS)
    else:
        work = deterministic_dataframe_sample(
            dataframe,
            max_rows=max_train_rows,
            random_state=config.random_state,
        )
        report = parameter_sensitivity_report(
            dataframe=work,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            sampler_config=sampler_config or config.sampler_config,
            n_samples=n_samples or min(max(1, config.n_generated), len(work)),
            random_state=config.random_state,
            n_components_grid=n_components_grid,
            nca_fit_sample_size_grid=nca_fit_sample_size_grid,
            lambda_grid=lambda_grid,
            n_iterations_grid=n_iterations_grid,
        )

    if not report.empty:
        report.to_csv(results_path / f"{config.dataset_name}_parameter_sensitivity.csv", index=False)
    return report


def parameter_sensitivity_report(
    *,
    dataframe: pd.DataFrame,
    dataset_name: str,
    target_column: str | None,
    sampler_config: Mapping[str, Any] | None = None,
    n_samples: int = 250,
    random_state: int = 42,
    n_components_grid: Iterable[int] = tuple(range(1, 11)),
    nca_fit_sample_size_grid: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    lambda_grid: Iterable[float] = tuple(np.linspace(0.1, 2.0, 10)),
    n_iterations_grid: Iterable[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=PARAMETER_SENSITIVITY_COLUMNS)

    base_config = dict(sampler_config or {})
    base_config.setdefault("random_state", random_state)
    base_config.update(PROPOSED_SAMPLER_SETUPS[1].sampler_config)

    rows = []
    for parameter, values in [
        ("n_components", n_components_grid),
        ("nca_fit_sample_size", nca_fit_sample_size_grid),
        ("lambda_", lambda_grid),
        ("n_iterations", n_iterations_grid),
    ]:
        for value in values:
            clean_value = _clean_parameter_value(value)
            variant_config = dict(base_config)
            variant_config[parameter] = clean_value
            rows.append(
                _evaluate_variant_config(
                    dataframe,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    parameter=parameter,
                    value=clean_value,
                    setup_label=f"{parameter}={clean_value}",
                    sampler_config=variant_config,
                    n_samples=n_samples,
                    random_state=random_state,
                )
            )
    return pd.DataFrame(rows, columns=PARAMETER_SENSITIVITY_COLUMNS)


def plot_parameter_sensitivity(
    report: pd.DataFrame,
    *,
    figures_dir: str | Path | None = None,
    filename: str = "adult_parameter_sensitivity.pdf",
    metrics: Iterable[str] = QUALITY_METRICS,
):
    if report.empty:
        raise ValueError("report must contain at least one sensitivity row.")
    metrics = list(metrics)
    parameters = ["n_components", "nca_fit_sample_size", "lambda_", "n_iterations"]
    labels = {
        "distribution_similarity_score": "Distribution score",
        "utility_lift": "Utility lift",
        "discrimination_accuracy": "Discrimination accuracy",
        "nn_distance_ratio": "NN ratio",
    }
    fig, axes = plt.subplots(
        len(parameters),
        len(metrics),
        figsize=(4.0 * len(metrics), 2.8 * len(parameters)),
        squeeze=False,
        sharex=False,
    )
    for row_idx, parameter in enumerate(parameters):
        subset = report[report["parameter"] == parameter].copy()
        subset["value_numeric"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset.sort_values("value_numeric")
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            ax.plot(subset["value_numeric"], subset[metric], marker="o", linewidth=1.5)
            best = _best_sensitivity_row(subset, metric)
            if best is not None:
                best_x = best["value_numeric"]
                best_y = best[metric]
                ax.axvline(best_x, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.scatter(
                    [best_x],
                    [best_y],
                    marker="*",
                    s=120,
                    color="tab:red",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
            if row_idx == 0:
                ax.set_title(labels.get(metric, metric))
            if col_idx == 0:
                ax.set_ylabel(parameter)
            ax.set_xlabel("value")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    if figures_dir is not None:
        output = Path(figures_dir) / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    return fig


def run_iterative_parameter_line_search(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    base_config: Mapping[str, Any],
    parameter_grids: Mapping[str, Iterable[Any]],
    parameter_order: Iterable[str] = LINE_SEARCH_PARAMETER_ORDER,
    max_train_rows: int = 250,
    n_samples: int = 250,
    random_state: int = 42,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a one-parameter-at-a-time line search using average quality ranks."""
    capped = deterministic_dataframe_sample(dataframe, max_rows=max_train_rows, random_state=random_state)
    current_config = dict(base_config)
    parameter_order = list(parameter_order)
    history_rows = []
    sweep_reports = []

    for step, parameter in enumerate(parameter_order, start=1):
        if parameter in NCA_ONLY_LINE_SEARCH_PARAMETERS and int(current_config.get("n_iterations", 0)) == 0:
            history_row = {
                "step": step,
                "parameter": parameter,
                "selected_value": current_config.get(parameter),
                "average_quality_rank": pd.NA,
                "reason": "skipped because n_iterations=0",
                **{key: current_config.get(key) for key in parameter_order},
            }
            history_rows.append(history_row)
            _emit_line_search_progress(
                progress_callback,
                step=step,
                parameter=parameter,
                history_row=history_row,
                sweep=pd.DataFrame(),
                current_config=current_config,
            )
            continue
        rows = []
        for value in parameter_grids[parameter]:
            clean_value = _clean_parameter_value(value)
            candidate_config = dict(current_config)
            candidate_config[parameter] = clean_value
            rows.append(
                _evaluate_variant_config(
                    capped,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    parameter=parameter,
                    value=clean_value,
                    setup_label=f"line_search_{parameter}={clean_value}",
                    sampler_config=candidate_config,
                    n_samples=n_samples,
                    random_state=random_state,
                )
            )

        sweep = pd.DataFrame(rows, columns=PARAMETER_SENSITIVITY_COLUMNS)
        ranked = rank_parameter_sweep(sweep)
        best = ranked.iloc[0]
        best_value = _clean_parameter_value(best["value"])
        current_config[parameter] = best_value
        ranked["line_search_step"] = step
        ranked["selected_value"] = best_value
        ranked["current_config_after_step"] = repr(
            {key: current_config.get(key) for key in parameter_order}
        )
        sweep_reports.append(ranked)
        history_row = {
            "step": step,
            "parameter": parameter,
            "selected_value": best_value,
            "average_quality_rank": best["average_quality_rank"],
            "reason": "ok",
            **{key: current_config.get(key) for key in parameter_order},
        }
        history_rows.append(history_row)
        _emit_line_search_progress(
            progress_callback,
            step=step,
            parameter=parameter,
            history_row=history_row,
            sweep=ranked,
            current_config=current_config,
        )

    history = pd.DataFrame(history_rows)
    sweeps = _concat_nonempty_frames(sweep_reports)
    return history, sweeps


def run_repeated_iterative_parameter_line_search(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    base_config: Mapping[str, Any],
    parameter_grids: Mapping[str, Iterable[Any]],
    seeds: Iterable[int],
    parameter_order: Iterable[str] = LINE_SEARCH_PARAMETER_ORDER,
    max_train_rows: int = 250,
    n_samples: int = 250,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run iterative line search with each sweep repeated across seeds.

    For each parameter step, every candidate value is evaluated on every
    deterministic capped subset. The selected value is chosen from the
    aggregated mean metrics for that step, then the shared configuration is
    updated before moving to the next parameter.
    """
    seeds = list(seeds)
    current_config = dict(base_config)
    parameter_order = list(parameter_order)
    histories = []
    sweeps = []
    summaries = []
    for step, parameter in enumerate(parameter_order, start=1):
        if parameter in NCA_ONLY_LINE_SEARCH_PARAMETERS and int(current_config.get("n_iterations", 0)) == 0:
            history_row = {
                "step": step,
                "parameter": parameter,
                "selected_value": current_config.get(parameter),
                "average_quality_rank": pd.NA,
                "n_seeds": len(seeds),
                "reason": "skipped because n_iterations=0",
                **{key: current_config.get(key) for key in parameter_order},
            }
            histories.append(history_row)
            _emit_line_search_progress(
                progress_callback,
                step=step,
                parameter=parameter,
                history_row=history_row,
                sweep=pd.DataFrame(),
                current_config=current_config,
                summary=pd.DataFrame(),
            )
            continue

        step_seed_sweeps = []
        for seed in seeds:
            capped = deterministic_dataframe_sample(dataframe, max_rows=max_train_rows, random_state=seed)
            rows = []
            for value in parameter_grids[parameter]:
                clean_value = _clean_parameter_value(value)
                candidate_config = {**current_config, "random_state": seed, parameter: clean_value}
                rows.append(
                    _evaluate_variant_config(
                        capped,
                        dataset_name=dataset_name,
                        target_column=target_column,
                        parameter=parameter,
                        value=clean_value,
                        setup_label=f"line_search_{parameter}={clean_value}",
                        sampler_config=candidate_config,
                        n_samples=n_samples,
                        random_state=seed,
                    )
                )
            seed_sweep = pd.DataFrame(rows, columns=PARAMETER_SENSITIVITY_COLUMNS)
            ranked_seed_sweep = rank_parameter_sweep(seed_sweep)
            ranked_seed_sweep["line_search_step"] = step
            ranked_seed_sweep["seed"] = seed
            step_seed_sweeps.append(ranked_seed_sweep)

        step_sweep = _concat_nonempty_frames(step_seed_sweeps)
        step_summary = summarize_repeated_line_search(step_sweep)
        best = step_summary.iloc[0]
        best_value = _clean_parameter_value(best["value"])
        current_config[parameter] = best_value

        step_sweep["selected_value"] = best_value
        step_sweep["current_config_after_step"] = repr(
            {key: current_config.get(key) for key in parameter_order}
        )
        step_summary["selected_value"] = best_value
        step_summary["current_config_after_step"] = repr(
            {key: current_config.get(key) for key in parameter_order}
        )
        step_summary["selection_count"] = (step_summary["value"] == best_value).astype(int)
        sweeps.append(step_sweep)
        summaries.append(step_summary)

        history_row = {
            "step": step,
            "parameter": parameter,
            "selected_value": best_value,
            "average_quality_rank": best["mean_average_quality_rank"],
            "n_seeds": int(best["n_seeds"]),
            "reason": "ok",
            **{key: current_config.get(key) for key in parameter_order},
        }
        histories.append(history_row)
        _emit_line_search_progress(
            progress_callback,
            step=step,
            parameter=parameter,
            history_row=history_row,
            sweep=step_sweep,
            current_config=current_config,
            summary=step_summary,
        )

    history_df = pd.DataFrame(histories)
    sweep_df = _concat_nonempty_frames(sweeps)
    summary_df = _concat_nonempty_frames(summaries)
    return history_df, sweep_df, summary_df


def _emit_line_search_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    step: int,
    parameter: str,
    history_row: Mapping[str, Any],
    sweep: pd.DataFrame,
    current_config: Mapping[str, Any],
    summary: pd.DataFrame | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "step": step,
            "parameter": parameter,
            "history_row": dict(history_row),
            "sweep": sweep,
            "summary": pd.DataFrame() if summary is None else summary,
            "current_config": dict(current_config),
        }
    )


def summarize_repeated_line_search(
    sweeps: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate repeated line-search sweeps by step, parameter, and value."""
    if sweeps.empty:
        return pd.DataFrame()
    grouped = sweeps.groupby(["line_search_step", "parameter", "value"], dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_seed_average_quality_rank=("average_quality_rank", "mean"),
        std_seed_average_quality_rank=("average_quality_rank", "std"),
        mean_distribution_similarity_score=("distribution_similarity_score", "mean"),
        std_distribution_similarity_score=("distribution_similarity_score", "std"),
        mean_utility_lift=("utility_lift", "mean"),
        std_utility_lift=("utility_lift", "std"),
        mean_discrimination_accuracy=("discrimination_accuracy", "mean"),
        std_discrimination_accuracy=("discrimination_accuracy", "std"),
        mean_nn_distance_ratio=("nn_distance_ratio", "mean"),
        std_nn_distance_ratio=("nn_distance_ratio", "std"),
    ).reset_index()
    if history is not None and not history.empty:
        selected = history[history["reason"].eq("ok")]
        selection_counts = (
            selected.groupby(["step", "parameter", "selected_value"], dropna=False)
            .size()
            .rename("selection_count")
            .reset_index()
            .rename(
                columns={
                    "step": "line_search_step",
                    "selected_value": "value",
                }
            )
        )
        summary = summary.merge(
            selection_counts,
            on=["line_search_step", "parameter", "value"],
            how="left",
        )
    if "selection_count" not in summary:
        summary["selection_count"] = 0
    summary["selection_count"] = summary["selection_count"].fillna(0).astype(int)
    summary["value_numeric"] = pd.to_numeric(summary["value"], errors="coerce")
    summary = rank_repeated_line_search_summary(summary)
    return summary.sort_values(
        ["line_search_step", "mean_average_quality_rank", "value_numeric"]
    ).reset_index(drop=True)


def rank_repeated_line_search_summary(
    summary: pd.DataFrame,
    metrics: Iterable[str] = QUALITY_METRICS,
) -> pd.DataFrame:
    """Rank repeated line-search values from aggregated mean metrics."""
    ranked = summary.copy()
    rank_columns = []
    for metric in metrics:
        mean_column = f"mean_{metric}"
        if mean_column not in ranked:
            continue
        rank_column = f"{mean_column}_rank"
        ranked[rank_column] = np.nan
        for _, step_index in ranked.groupby("line_search_step").groups.items():
            values = pd.to_numeric(ranked.loc[step_index, mean_column], errors="coerce")
            valid_index = values[values.notna()].index
            if len(valid_index) == 0:
                continue
            if metric in TARGET_METRIC_VALUES:
                score = (values.loc[valid_index] - TARGET_METRIC_VALUES[metric]).abs()
            else:
                score = -values.loc[valid_index]
            ranked.loc[valid_index, rank_column] = score.rank(method="min", ascending=True)
        rank_columns.append(rank_column)
    ranked["mean_average_quality_rank"] = ranked[rank_columns].mean(axis=1)
    return ranked


def rank_parameter_sweep(
    sweep: pd.DataFrame,
    metrics: Iterable[str] = QUALITY_METRICS,
) -> pd.DataFrame:
    """Rank sweep rows by average metric rank using each metric's direction."""
    ranked = sweep.copy()
    rank_columns = []
    for metric in metrics:
        values = pd.to_numeric(ranked[metric], errors="coerce")
        valid = values.notna()
        rank_column = f"{metric}_rank"
        ranked[rank_column] = np.nan
        if valid.any():
            if metric in TARGET_METRIC_VALUES:
                score = (values[valid] - TARGET_METRIC_VALUES[metric]).abs()
            else:
                score = -values[valid]
            ranked.loc[valid, rank_column] = score.rank(method="min", ascending=True)
            rank_columns.append(rank_column)
    ranked["average_quality_rank"] = ranked[rank_columns].mean(axis=1)
    ranked["value_numeric"] = pd.to_numeric(ranked["value"], errors="coerce")
    return ranked.sort_values(["average_quality_rank", "value_numeric"]).reset_index(drop=True)


def plot_line_search_step(
    sweep: pd.DataFrame,
    parameter: str,
    selected_value: Any,
    *,
    metrics: Iterable[str] = QUALITY_METRICS,
):
    """Plot one line-search sweep and mark the selected average-rank value."""
    metrics = list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.0), squeeze=False)
    ordered = sweep.copy()
    ordered["value_numeric"] = pd.to_numeric(ordered["value"], errors="coerce")
    ordered = ordered.sort_values("value_numeric")
    selected_value = float(selected_value)
    flat_axes = axes.ravel()
    for ax, metric in zip(flat_axes, metrics):
        y = pd.to_numeric(ordered[metric], errors="coerce")
        x = ordered["value_numeric"]
        ax.plot(x, y, marker="o", linewidth=1.5)
        metric_best = _best_xy_for_metric(x, y, metric)
        if metric_best is not None:
            best_x, best_y = metric_best
            ax.axvline(best_x, color="0.35", linestyle=":", linewidth=1.2, alpha=0.8)
            ax.scatter(
                [best_x],
                [best_y],
                marker="D",
                s=70,
                color="0.35",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        ax.axvline(selected_value, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.85)
        selected = ordered[np.isclose(x, selected_value)]
        if not selected.empty:
            ax.scatter(
                selected["value_numeric"],
                selected[metric],
                marker="*",
                s=130,
                color="tab:red",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        ax.set_title(LINE_SEARCH_METRIC_LABELS.get(metric, metric))
        ax.set_xlabel(parameter)
        ax.grid(alpha=0.25)
    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(f"Line-search step: {parameter} = {selected_value:g}", y=1.02)
    fig.tight_layout()
    return fig


def display_line_search_progress(
    event: Mapping[str, Any],
    *,
    columns: Iterable[str] = LINE_SEARCH_DISPLAY_COLUMNS,
) -> None:
    """Display one completed line-search step in a notebook.

    The helper is intentionally kept in the experiment module so notebooks can
    remain thin and only configure the run.
    """
    seed = event.get("seed")
    step = event["step"]
    parameter = event["parameter"]
    history_row = dict(event["history_row"])
    sweep = event["sweep"]
    summary = event.get("summary", pd.DataFrame())
    selected_value = history_row.get("selected_value")
    display_func = _notebook_display()

    prefix = f"Seed {seed}, " if seed is not None else ""
    print(f"{prefix}step {step}: {parameter} -> {selected_value}")
    display_func(pd.DataFrame([{**history_row, **({"seed": seed} if seed is not None else {})}]))
    display_func(pd.DataFrame([event["current_config"]]))
    if sweep.empty:
        return

    if summary is not None and not summary.empty:
        summary_columns = [
            "line_search_step",
            "parameter",
            "value",
            "n_seeds",
            "selection_count",
            "mean_average_quality_rank",
            "std_seed_average_quality_rank",
            "mean_distribution_similarity_score",
            "mean_utility_lift",
            "mean_discrimination_accuracy",
            "mean_nn_distance_ratio",
        ]
        display_columns = [column for column in summary_columns if column in summary.columns]
        display_func(summary[display_columns])
        fig = plot_repeated_line_search_summary(sweep, summary)
    else:
        display_columns = [column for column in columns if column in sweep.columns]
        if display_columns:
            display_func(sweep[display_columns])
        fig = plot_line_search_step(sweep, parameter, selected_value)
    display_func(fig)
    plt.close(fig)


def plot_repeated_line_search_summary(
    sweeps: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metrics: Iterable[str] = QUALITY_METRICS,
):
    """Plot repeated line-search sweeps with per-seed traces and mean error bars."""
    if sweeps.empty or summary.empty:
        raise ValueError("sweeps and summary must contain line-search rows.")
    metrics = list(metrics)
    steps = list(summary[["line_search_step", "parameter"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        len(steps),
        len(metrics),
        figsize=(4.4 * len(metrics), 2.8 * len(steps)),
        squeeze=False,
    )
    for row_idx, (step, parameter) in enumerate(steps):
        step_sweeps = sweeps[sweeps["line_search_step"].eq(step)].copy()
        step_summary = summary[summary["line_search_step"].eq(step)].copy()
        step_sweeps["value_numeric"] = pd.to_numeric(step_sweeps["value"], errors="coerce")
        step_summary["value_numeric"] = pd.to_numeric(step_summary["value"], errors="coerce")
        best = step_summary.sort_values(["mean_average_quality_rank", "value_numeric"]).iloc[0]
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            mean_col = f"mean_{metric}"
            std_col = f"std_{metric}"
            for _, seed_rows in step_sweeps.groupby("seed"):
                seed_rows = seed_rows.sort_values("value_numeric")
                ax.plot(
                    seed_rows["value_numeric"],
                    seed_rows[metric],
                    color="tab:gray",
                    marker="o",
                    linewidth=0.8,
                    alpha=0.28,
                )
            ordered = step_summary.sort_values("value_numeric")
            yerr = ordered[std_col].fillna(0.0) if std_col in ordered else None
            ax.errorbar(
                ordered["value_numeric"],
                ordered[mean_col],
                yerr=yerr,
                color="tab:blue",
                marker="o",
                linewidth=1.8,
                capsize=3,
                label="mean ± sd" if row_idx == 0 and col_idx == 0 else None,
            )
            metric_best = _best_xy_for_metric(ordered["value_numeric"], ordered[mean_col], metric)
            if metric_best is not None:
                best_x, best_y = metric_best
                ax.axvline(
                    best_x,
                    color="0.35",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label="metric best" if row_idx == 0 and col_idx == 0 else None,
                )
                ax.scatter(
                    [best_x],
                    [best_y],
                    marker="D",
                    s=70,
                    color="0.35",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
            ax.axvline(best["value_numeric"], color="tab:red", linestyle="--", linewidth=1.0, alpha=0.85)
            selected = ordered[np.isclose(ordered["value_numeric"], best["value_numeric"])]
            if not selected.empty:
                ax.scatter(
                    selected["value_numeric"],
                    selected[mean_col],
                    marker="*",
                    s=130,
                    color="tab:red",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                    label="best mean rank" if row_idx == 0 and col_idx == 0 else None,
                )
            if row_idx == 0:
                ax.set_title(LINE_SEARCH_METRIC_LABELS.get(metric, metric))
            if col_idx == 0:
                ax.set_ylabel(parameter)
            ax.set_xlabel("value")
            ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _best_sensitivity_row(subset: pd.DataFrame, metric: str) -> pd.Series | None:
    values = pd.to_numeric(subset[metric], errors="coerce")
    candidates = subset.loc[values.notna()].copy()
    if candidates.empty:
        return None
    metric_values = pd.to_numeric(candidates[metric], errors="coerce")
    if metric in TARGET_METRIC_VALUES:
        score = (metric_values - TARGET_METRIC_VALUES[metric]).abs()
        best_idx = score.idxmin()
    else:
        best_idx = metric_values.idxmax()
    return candidates.loc[best_idx]


def _concat_nonempty_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _notebook_display() -> Callable[[Any], None]:
    try:
        from IPython.display import display
    except ImportError:
        return print
    return display


def _best_xy_for_metric(x: pd.Series, y: pd.Series, metric: str) -> tuple[float, float] | None:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    valid_x = x[valid]
    valid_y = y[valid]
    if metric in TARGET_METRIC_VALUES:
        best_idx = (valid_y - TARGET_METRIC_VALUES[metric]).abs().idxmin()
    else:
        best_idx = valid_y.idxmax()
    return float(valid_x.loc[best_idx]), float(valid_y.loc[best_idx])


def _clean_parameter_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value
