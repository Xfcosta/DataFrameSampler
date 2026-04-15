from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataframe_sampler import ConcreteDataFrameSampler

from .baselines import BaselineSpec, simple_baselines
from .metrics import categorical_similarity, dependence_similarity, numeric_similarity


def run_dataset_comparison(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    results_dir: str | Path,
    dataframe_sampler_config: Mapping[str, Any] | None = None,
    llm_assisted_config: Mapping[str, Any] | None = None,
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run DataFrameSampler and simple baselines, then write metric summaries."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    method_specs = dataframe_sampler_configuration_competitors(
        manual_config=dataframe_sampler_config,
        llm_assisted_config=llm_assisted_config,
        random_state=random_state,
    )
    method_specs.extend(simple_baselines(target_column=target_column, random_state=random_state))

    rows = []
    for spec in method_specs:
        fit_start = time.perf_counter()
        spec.estimator.fit(dataframe)
        fit_seconds = time.perf_counter() - fit_start

        sample_start = time.perf_counter()
        synthetic = spec.estimator.sample(n_samples=n_samples)
        sample_seconds = time.perf_counter() - sample_start

        synthetic_path = results_dir / f"{dataset_name}_{spec.name}_generated.csv"
        synthetic.to_csv(synthetic_path, index=False)

        summary = summarize_synthetic_sample(
            dataframe,
            synthetic,
            dataset_name=dataset_name,
            method_name=spec.name,
            fit_seconds=fit_seconds,
            sample_seconds=sample_seconds,
        )
        rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    summary_df.to_csv(results_dir / f"{dataset_name}_baseline_comparison.csv", index=False)
    return summary_df


def dataframe_sampler_configuration_competitors(
    *,
    manual_config: Mapping[str, Any] | None = None,
    llm_assisted_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> list[BaselineSpec]:
    """Return DataFrameSampler default/manual/LLM-style configuration competitors."""
    default_config = {"random_state": random_state}
    manual = dict(manual_config or {})
    manual.setdefault("random_state", random_state)
    specs = [
        BaselineSpec("dataframe_sampler_default", ConcreteDataFrameSampler(**default_config)),
        BaselineSpec("dataframe_sampler_manual", ConcreteDataFrameSampler(**manual)),
    ]
    if llm_assisted_config is not None:
        llm_config = dict(llm_assisted_config)
        llm_config.setdefault("random_state", random_state)
        specs.append(
            BaselineSpec("dataframe_sampler_llm_assisted", ConcreteDataFrameSampler(**llm_config))
        )
    return specs


def summarize_synthetic_sample(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    dataset_name: str,
    method_name: str,
    fit_seconds: float | None = None,
    sample_seconds: float | None = None,
) -> dict[str, float | str | int | None]:
    numeric = numeric_similarity(real, synthetic)
    categorical = categorical_similarity(real, synthetic)
    dependence = dependence_similarity(real, synthetic)
    return {
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
    }


def _mean_or_nan(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe.empty or column not in dataframe:
        return np.nan
    return float(dataframe[column].mean(skipna=True))
