from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from dataframe_sampler import DataFrameSampler

from .baselines import simple_baselines
from .compare import summarize_synthetic_sample
from .datasets import DatasetExperimentConfig
from .instrumentation import measure_call
from .manifold_validation import deterministic_dataframe_sample


PRIMARY_UNCERTAINTY_DATASET = "adult"

PRIMARY_UNCERTAINTY_COLUMNS = [
    "dataset",
    "seed",
    "method",
    "n_real",
    "n_synthetic",
    "fit_seconds",
    "sample_seconds",
    "nn_distance_ratio",
    "nn_suspiciously_close_rate",
    "discrimination_accuracy",
    "utility_lift",
    "distribution_similarity_score",
    "reason",
]


def run_primary_uncertainty_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    seeds: Iterable[int] = (42, 43, 44),
    max_train_rows: int = 120,
    n_samples: int = 120,
) -> pd.DataFrame:
    """Run a small repeated-seed primary-metric diagnostic on Adult.

    This is descriptive uncertainty evidence, not a formal statistical test.
    """
    if config.dataset_name != PRIMARY_UNCERTAINTY_DATASET or dataframe.empty or len(dataframe) < 50:
        return pd.DataFrame(columns=PRIMARY_UNCERTAINTY_COLUMNS)

    work = deterministic_dataframe_sample(
        dataframe,
        max_rows=max_train_rows,
        random_state=config.random_state,
    )
    rows = []
    for seed in seeds:
        rows.extend(
            _seed_rows(
                work,
                dataset_name=config.dataset_name,
                target_column=config.target_column,
                sampler_config=sampler_config or config.sampler_config,
                n_samples=min(n_samples, len(work)),
                seed=seed,
            )
        )
    report = pd.DataFrame(rows, columns=PRIMARY_UNCERTAINTY_COLUMNS)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    report.to_csv(results_path / f"{config.dataset_name}_primary_uncertainty.csv", index=False)
    return report


def summarize_primary_uncertainty(rows: pd.DataFrame) -> pd.DataFrame:
    valid = rows[rows["reason"].fillna("") == ""].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["dataset", "method"], dropna=False)
        .agg(
            runs=("seed", "nunique"),
            distribution_similarity_mean=("distribution_similarity_score", "mean"),
            distribution_similarity_std=("distribution_similarity_score", "std"),
            discrimination_accuracy_mean=("discrimination_accuracy", "mean"),
            discrimination_accuracy_std=("discrimination_accuracy", "std"),
            utility_lift_mean=("utility_lift", "mean"),
            utility_lift_std=("utility_lift", "std"),
            nn_distance_ratio_mean=("nn_distance_ratio", "mean"),
            nn_distance_ratio_std=("nn_distance_ratio", "std"),
        )
        .reset_index()
    )


def _seed_rows(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    sampler_config: Mapping[str, Any],
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    method_specs = [("dataframe_sampler", DataFrameSampler(**_sampler_config(sampler_config, seed)))]
    method_specs.extend((spec.name, spec.estimator) for spec in simple_baselines(target_column=target_column, random_state=seed))
    rows = []
    for method_name, estimator in method_specs:
        try:
            fit = measure_call(lambda: estimator.fit(dataframe))
            if isinstance(estimator, DataFrameSampler):
                sample = measure_call(lambda: estimator.generate(n_samples=n_samples))
            else:
                sample = measure_call(lambda: estimator.sample(n_samples=n_samples))
            summary = summarize_synthetic_sample(
                dataframe,
                sample.value,
                dataset_name=dataset_name,
                method_name=method_name,
                fit_seconds=fit.seconds,
                sample_seconds=sample.seconds,
                fit_peak_memory_mb=fit.peak_memory_mb,
                sample_peak_memory_mb=sample.peak_memory_mb,
                target_column=target_column,
                random_state=seed,
            )
            rows.append({**{column: summary.get(column) for column in PRIMARY_UNCERTAINTY_COLUMNS}, "seed": seed, "reason": ""})
        except Exception as exc:  # pragma: no cover - diagnostic row for notebooks.
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": method_name,
                    "n_real": len(dataframe),
                    "n_synthetic": n_samples,
                    "fit_seconds": pd.NA,
                    "sample_seconds": pd.NA,
                    "nn_distance_ratio": pd.NA,
                    "nn_suspiciously_close_rate": pd.NA,
                    "discrimination_accuracy": pd.NA,
                    "utility_lift": pd.NA,
                    "distribution_similarity_score": pd.NA,
                    "reason": f"failed:{type(exc).__name__}",
                }
            )
    return rows


def _sampler_config(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    sampler_config = dict(config)
    sampler_config.setdefault("random_state", seed)
    sampler_config["random_state"] = seed
    return sampler_config
