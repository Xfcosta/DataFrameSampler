from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _clean_parameter_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value
