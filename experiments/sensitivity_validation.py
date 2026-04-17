from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from dataframe_sampler import DataFrameSampler

from .compare import summarize_synthetic_sample
from .datasets import DatasetExperimentConfig
from .instrumentation import measure_call
from .manifold_validation import deterministic_dataframe_sample
from .proposed_setups import PROPOSED_SAMPLER_SETUPS, ProposedSamplerSetup


SENSITIVITY_VALIDATION_DATASET = "adult"

SENSITIVITY_VALIDATION_COLUMNS = [
    "dataset",
    "parameter",
    "value",
    "setup",
    "setup_label",
    "n_iterations",
    "max_constraint_retries",
    "calibrate_decoders",
    "n_train",
    "n_generated",
    "fit_seconds",
    "sample_seconds",
    "fit_peak_memory_mb",
    "sample_peak_memory_mb",
    "peak_memory_mb",
    "nn_distance_ratio",
    "nn_suspiciously_close_rate",
    "discrimination_accuracy",
    "discrimination_privacy_score",
    "utility_task",
    "utility_real_score",
    "utility_augmented_score",
    "utility_lift",
    "distribution_histogram_overlap",
    "distribution_categorical_jsd",
    "distribution_similarity_score",
    "decoder_calibration_enabled",
    "reason",
]


def run_sensitivity_validation_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    max_train_rows: int = 800,
    n_samples: int | None = None,
    setups: Iterable[ProposedSamplerSetup] = PROPOSED_SAMPLER_SETUPS,
) -> pd.DataFrame:
    """Run capped DataFrameSampler setup checks for the designated dataset."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if config.dataset_name != SENSITIVITY_VALIDATION_DATASET or dataframe.empty:
        report = _empty_sensitivity_frame(config.dataset_name)
    else:
        work = deterministic_dataframe_sample(
            dataframe,
            max_rows=max_train_rows,
            random_state=config.random_state,
        )
        report = sensitivity_validation_report(
            dataframe=work,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            sampler_config=sampler_config or config.sampler_config,
            n_samples=n_samples or min(max(1, config.n_generated), len(work)),
            random_state=config.random_state,
            setups=setups,
        )

    if not report.empty:
        report.to_csv(results_path / f"{config.dataset_name}_sensitivity_validation.csv", index=False)
    return report


def sensitivity_validation_report(
    *,
    dataframe: pd.DataFrame,
    dataset_name: str,
    target_column: str | None,
    sampler_config: Mapping[str, Any] | None = None,
    n_samples: int = 250,
    random_state: int = 42,
    setups: Iterable[ProposedSamplerSetup] = PROPOSED_SAMPLER_SETUPS,
) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_sensitivity_frame(dataset_name)

    base_config = dict(sampler_config or {})
    base_config.setdefault("random_state", random_state)
    rows = []
    for setup in setups:
        variant_config = dict(base_config)
        variant_config.update(setup.sampler_config)
        rows.append(
            _evaluate_variant(
                dataframe,
                dataset_name=dataset_name,
                target_column=target_column,
                setup=setup,
                sampler_config=variant_config,
                n_samples=n_samples,
                random_state=random_state,
            )
        )
    return pd.DataFrame(rows, columns=SENSITIVITY_VALIDATION_COLUMNS)


def summarize_sensitivity_validation(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "parameter",
                "value",
                "setup_label",
                "datasets_evaluated",
                "n_iterations",
                "max_constraint_retries",
                "calibrate_decoders",
                "mean_nn_distance_ratio",
                "mean_discrimination_accuracy",
                "mean_utility_lift",
                "mean_distribution_similarity_score",
                "mean_fit_seconds",
                "mean_sample_seconds",
            ]
        )
    rows = rows.copy()
    if "setup_label" not in rows:
        rows["setup_label"] = rows.get("value", pd.Series(dtype=object)).astype(str)
    if "n_iterations" not in rows:
        rows["n_iterations"] = pd.NA
    if "max_constraint_retries" not in rows:
        rows["max_constraint_retries"] = pd.NA
    if "calibrate_decoders" not in rows:
        rows["calibrate_decoders"] = rows.get("decoder_calibration_enabled", pd.NA)
    grouped = rows.groupby(
        [
            "parameter",
            "value",
            "setup_label",
            "n_iterations",
            "max_constraint_retries",
            "calibrate_decoders",
        ],
        dropna=False,
    )
    return grouped.agg(
        datasets_evaluated=("dataset", "nunique"),
        mean_nn_distance_ratio=("nn_distance_ratio", "mean"),
        mean_discrimination_accuracy=("discrimination_accuracy", "mean"),
        mean_utility_lift=("utility_lift", "mean"),
        mean_distribution_similarity_score=("distribution_similarity_score", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_sample_seconds=("sample_seconds", "mean"),
    ).reset_index()


def _evaluate_variant(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    target_column: str | None,
    setup: ProposedSamplerSetup,
    sampler_config: Mapping[str, Any],
    n_samples: int,
    random_state: int,
) -> dict[str, Any]:
    sampler = DataFrameSampler(**dict(sampler_config))
    try:
        fit = measure_call(lambda: sampler.fit(dataframe))
        sample = measure_call(lambda: sampler.generate(n_samples=n_samples))
        synthetic = sample.value
        summary = summarize_synthetic_sample(
            dataframe,
            synthetic,
            dataset_name=dataset_name,
            method_name=f"sensitivity_setup_{setup.key}",
            fit_seconds=fit.seconds,
            sample_seconds=sample.seconds,
            fit_peak_memory_mb=fit.peak_memory_mb,
            sample_peak_memory_mb=sample.peak_memory_mb,
            target_column=target_column,
            random_state=random_state,
        )
        return _row_from_summary(
            summary,
            setup=setup,
            decoder_calibration_enabled=bool(sampler_config.get("calibrate_decoders", False)),
            reason="ok",
        )
    except Exception as exc:  # pragma: no cover - defensive row for long notebooks.
        return {
            "dataset": dataset_name,
            "parameter": "setup",
            "value": setup.key,
            "setup": setup.key,
            "setup_label": setup.label,
            "n_iterations": setup.n_iterations,
            "max_constraint_retries": setup.max_constraint_retries,
            "calibrate_decoders": setup.calibrate_decoders,
            "n_train": len(dataframe),
            "n_generated": n_samples,
            "fit_seconds": pd.NA,
            "sample_seconds": pd.NA,
            "fit_peak_memory_mb": pd.NA,
            "sample_peak_memory_mb": pd.NA,
            "peak_memory_mb": pd.NA,
            "nn_distance_ratio": pd.NA,
            "nn_suspiciously_close_rate": pd.NA,
            "discrimination_accuracy": pd.NA,
            "discrimination_privacy_score": pd.NA,
            "utility_task": pd.NA,
            "utility_real_score": pd.NA,
            "utility_augmented_score": pd.NA,
            "utility_lift": pd.NA,
            "distribution_histogram_overlap": pd.NA,
            "distribution_categorical_jsd": pd.NA,
            "distribution_similarity_score": pd.NA,
            "decoder_calibration_enabled": bool(sampler_config.get("calibrate_decoders", False)),
            "reason": f"failed:{type(exc).__name__}",
        }


def _row_from_summary(
    summary: Mapping[str, Any],
    *,
    setup: ProposedSamplerSetup,
    decoder_calibration_enabled: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": summary["dataset"],
        "parameter": "setup",
        "value": setup.key,
        "setup": setup.key,
        "setup_label": setup.label,
        "n_iterations": setup.n_iterations,
        "max_constraint_retries": setup.max_constraint_retries,
        "calibrate_decoders": setup.calibrate_decoders,
        "n_train": summary["n_real"],
        "n_generated": summary["n_synthetic"],
        "fit_seconds": summary["fit_seconds"],
        "sample_seconds": summary["sample_seconds"],
        "fit_peak_memory_mb": summary["fit_peak_memory_mb"],
        "sample_peak_memory_mb": summary["sample_peak_memory_mb"],
        "peak_memory_mb": summary["peak_memory_mb"],
        "nn_distance_ratio": summary["nn_distance_ratio"],
        "nn_suspiciously_close_rate": summary["nn_suspiciously_close_rate"],
        "discrimination_accuracy": summary["discrimination_accuracy"],
        "discrimination_privacy_score": summary["discrimination_privacy_score"],
        "utility_task": summary["utility_task"],
        "utility_real_score": summary["utility_real_score"],
        "utility_augmented_score": summary["utility_augmented_score"],
        "utility_lift": summary["utility_lift"],
        "distribution_histogram_overlap": summary["distribution_histogram_overlap"],
        "distribution_categorical_jsd": summary["distribution_categorical_jsd"],
        "distribution_similarity_score": summary["distribution_similarity_score"],
        "decoder_calibration_enabled": decoder_calibration_enabled,
        "reason": reason,
    }


def _empty_sensitivity_frame(dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=SENSITIVITY_VALIDATION_COLUMNS)
    if dataset_name:
        frame["dataset"] = frame.get("dataset", pd.Series(dtype=object))
    return frame
