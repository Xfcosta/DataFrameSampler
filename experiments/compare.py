from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataframe_sampler import DataFrameSampler

from .baselines import BaselineSpec, simple_baselines
from .instrumentation import measure_call
from .metrics import categorical_similarity, dependence_similarity, main_measure_report, numeric_similarity


def run_dataset_comparison(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    results_dir: str | Path,
    dataframe_sampler_config: Mapping[str, Any] | None = None,
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run DataFrameSampler and simple baselines, then write metric summaries."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    method_specs = dataframe_sampler_methods(
        sampler_config=dataframe_sampler_config,
        random_state=random_state,
    )
    method_specs.extend(simple_baselines(target_column=target_column, random_state=random_state))

    rows = []
    for spec in method_specs:
        fit = measure_call(lambda: spec.estimator.fit(dataframe))

        if isinstance(spec.estimator, DataFrameSampler):
            sample = measure_call(lambda: spec.estimator.generate(n_samples=n_samples))
        else:
            sample = measure_call(lambda: spec.estimator.sample(n_samples=n_samples))
        synthetic = sample.value

        synthetic_path = results_dir / f"{dataset_name}_{spec.name}_generated.csv"
        synthetic.to_csv(synthetic_path, index=False)

        summary = summarize_synthetic_sample(
            dataframe,
            synthetic,
            dataset_name=dataset_name,
            method_name=spec.name,
            fit_seconds=fit.seconds,
            sample_seconds=sample.seconds,
            fit_peak_memory_mb=fit.peak_memory_mb,
            sample_peak_memory_mb=sample.peak_memory_mb,
            target_column=target_column,
            random_state=random_state,
        )
        rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    summary_df.to_csv(results_dir / f"{dataset_name}_baseline_comparison.csv", index=False)
    return summary_df


def dataframe_sampler_methods(
    *,
    sampler_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> list[BaselineSpec]:
    """Return the DataFrameSampler method used in comparisons."""
    config = dict(sampler_config or {})
    config.setdefault("random_state", random_state)
    return [BaselineSpec("dataframe_sampler", DataFrameSampler(**config))]


def summarize_synthetic_sample(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    dataset_name: str,
    method_name: str,
    fit_seconds: float | None = None,
    sample_seconds: float | None = None,
    fit_peak_memory_mb: float | None = None,
    sample_peak_memory_mb: float | None = None,
    target_column: str | None = None,
    random_state: int = 42,
) -> dict[str, float | str | int | None]:
    numeric = numeric_similarity(real, synthetic)
    categorical = categorical_similarity(real, synthetic)
    dependence = dependence_similarity(real, synthetic)
    summary = {
        "dataset": dataset_name,
        "method": method_name,
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "numeric_mean_abs_error": _mean_or_nan(numeric, "mean_abs_error"),
        "numeric_std_abs_error": _mean_or_nan(numeric, "std_abs_error"),
        "numeric_ks_statistic": _mean_or_nan(numeric, "ks_statistic"),
        "numeric_wasserstein_distance": _mean_or_nan(numeric, "wasserstein_distance"),
        "numeric_histogram_overlap": _mean_or_nan(numeric, "histogram_overlap"),
        "categorical_total_variation": _mean_or_nan(categorical, "total_variation_distance"),
        "categorical_jensen_shannon": _mean_or_nan(categorical, "jensen_shannon_divergence"),
        "categorical_coverage": _mean_or_nan(categorical, "category_coverage"),
        "rare_category_preservation": _mean_or_nan(categorical, "rare_category_preservation"),
        "mean_abs_association_difference": dependence["mean_abs_association_difference"],
        "max_abs_association_difference": dependence["max_abs_association_difference"],
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "fit_peak_memory_mb": fit_peak_memory_mb,
        "sample_peak_memory_mb": sample_peak_memory_mb,
        "peak_memory_mb": _max_optional(fit_peak_memory_mb, sample_peak_memory_mb),
    }
    summary.update(
        main_measure_report(
            real,
            synthetic,
            target_column=target_column,
            random_state=random_state,
        )
    )
    return summary


def _mean_or_nan(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe.empty or column not in dataframe:
        return np.nan
    return float(dataframe[column].mean(skipna=True))


def _max_optional(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)
