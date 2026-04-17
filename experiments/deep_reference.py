from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import SdvCtganBaseline
from .compare import summarize_synthetic_sample
from .datasets import DatasetExperimentConfig
from .instrumentation import measure_call
from .manifold_validation import deterministic_dataframe_sample


DEEP_REFERENCE_DATASET = "adult"

DEEP_REFERENCE_COLUMNS = [
    "dataset",
    "method",
    "method_label",
    "n_real",
    "n_synthetic",
    "distribution_similarity_score",
    "discrimination_accuracy",
    "utility_lift",
    "fit_seconds",
    "sample_seconds",
    "peak_memory_mb",
    "reason",
]


def run_deep_reference_comparison_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    n_samples: int | None = None,
    max_train_rows: int = 800,
    baseline_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Run optional high-capacity CTGAN reference comparison for Adult only."""
    if config.dataset_name != DEEP_REFERENCE_DATASET or dataframe.empty:
        return _empty_deep_reference_frame(config.dataset_name)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    work = deterministic_dataframe_sample(
        dataframe,
        max_rows=max_train_rows,
        random_state=config.random_state,
    )
    n_rows = n_samples or min(max(1, config.n_generated), len(work))
    baseline = baseline_factory() if baseline_factory is not None else SdvCtganBaseline(random_state=config.random_state)

    fit = measure_call(lambda: baseline.fit(work))
    sample = measure_call(lambda: baseline.sample(n_rows))
    synthetic = sample.value
    synthetic.to_csv(results_path / f"{config.dataset_name}_ctgan_generated.csv", index=False)

    summary = summarize_synthetic_sample(
        work,
        synthetic,
        dataset_name=config.dataset_name,
        method_name="ctgan",
        fit_seconds=fit.seconds,
        sample_seconds=sample.seconds,
        fit_peak_memory_mb=fit.peak_memory_mb,
        sample_peak_memory_mb=sample.peak_memory_mb,
        target_column=config.target_column,
        random_state=config.random_state,
    )
    row = {
        "dataset": summary["dataset"],
        "method": summary["method"],
        "method_label": "CTGAN",
        "n_real": summary["n_real"],
        "n_synthetic": summary["n_synthetic"],
        "distribution_similarity_score": summary["distribution_similarity_score"],
        "discrimination_accuracy": summary["discrimination_accuracy"],
        "utility_lift": summary["utility_lift"],
        "fit_seconds": summary["fit_seconds"],
        "sample_seconds": summary["sample_seconds"],
        "peak_memory_mb": summary["peak_memory_mb"],
        "reason": "ok",
    }
    report = pd.DataFrame([row], columns=DEEP_REFERENCE_COLUMNS)
    report.to_csv(results_path / f"{config.dataset_name}_deep_reference_comparison.csv", index=False)
    return report


def _empty_deep_reference_frame(dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=DEEP_REFERENCE_COLUMNS)
    if dataset_name:
        frame["dataset"] = frame.get("dataset", pd.Series(dtype=object))
    return frame
