from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataframe_sampler import ConcreteDataFrameSampler

from .compare import run_dataset_comparison
from .datasets import DatasetExperimentConfig
from .instrumentation import measure_call


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    data_path: Path
    results_dir: Path


@dataclass(frozen=True)
class SamplerRun:
    generated: pd.DataFrame
    similarity_report: pd.DataFrame
    runtime: pd.DataFrame
    fit_seconds: float
    sample_seconds: float
    fit_peak_memory_mb: float
    sample_peak_memory_mb: float
    peak_memory_mb: float


@dataclass(frozen=True)
class DatasetExperimentResult:
    dataframe: pd.DataFrame
    working_dataframe: pd.DataFrame
    profile: pd.DataFrame
    starter_run: SamplerRun
    comparison: pd.DataFrame
    paths: ExperimentPaths


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find the project root from a notebook, script, or repository cwd."""
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "experiments").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {current}")


def experiment_paths(
    config: DatasetExperimentConfig,
    *,
    root: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> ExperimentPaths:
    project_root = resolve_project_root(root)
    result_path = Path(results_dir) if results_dir is not None else project_root / "experiments" / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root=project_root,
        data_path=project_root / "experiments" / "data" / "processed" / config.data_filename,
        results_dir=result_path,
    )


def notebook_environment(paths: ExperimentPaths) -> dict[str, str]:
    """Return lightweight version information displayed by notebooks."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "dataset": str(paths.data_path),
    }


def load_dataset(config: DatasetExperimentConfig, *, root: str | Path | None = None) -> pd.DataFrame:
    paths = experiment_paths(config, root=root)
    return pd.read_csv(paths.data_path)


def dataset_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing": dataframe.isna().sum(),
            "unique": dataframe.nunique(dropna=True),
        }
    )


def working_dataframe(dataframe: pd.DataFrame, config: DatasetExperimentConfig) -> pd.DataFrame:
    if config.working_sample_size is None or config.working_sample_size >= len(dataframe):
        return dataframe.reset_index(drop=True)
    return dataframe.sample(n=config.working_sample_size, random_state=config.random_state).reset_index(drop=True)


def sampler_config_with_random_state(config: dict[str, Any], random_state: int) -> dict[str, Any]:
    sampler_config = dict(config)
    sampler_config.setdefault("random_state", random_state)
    return sampler_config


def quick_similarity_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    numeric = real.select_dtypes(include="number").columns
    categorical = [column for column in real.columns if column not in numeric]
    rows: list[dict[str, Any]] = []
    for column in numeric:
        rows.append(
            {
                "column": column,
                "kind": "numeric",
                "real_mean": real[column].mean(),
                "synthetic_mean": synthetic[column].mean(),
                "abs_mean_delta": abs(real[column].mean() - synthetic[column].mean()),
                "real_missing": real[column].isna().mean(),
                "synthetic_missing": synthetic[column].isna().mean(),
            }
        )
    for column in categorical:
        real_values = set(real[column].dropna().astype(str))
        synthetic_values = set(synthetic[column].dropna().astype(str))
        rows.append(
            {
                "column": column,
                "kind": "categorical",
                "real_unique": len(real_values),
                "synthetic_unique": len(synthetic_values),
                "category_coverage": len(real_values & synthetic_values) / max(len(real_values), 1),
                "real_missing": real[column].isna().mean(),
                "synthetic_missing": synthetic[column].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def run_starter_sampler(
    dataframe: pd.DataFrame,
    config: DatasetExperimentConfig,
    *,
    results_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> SamplerRun:
    output_dir = Path(results_dir) if results_dir is not None else None
    sampler = ConcreteDataFrameSampler(
        **sampler_config_with_random_state(config.manual_sampler_config, config.random_state)
    )

    fit = measure_call(lambda: sampler.fit(dataframe))
    sample = measure_call(lambda: sampler.sample(n_samples=config.n_generated))
    generated = sample.value
    peak_memory_mb = max(fit.peak_memory_mb, sample.peak_memory_mb)

    report = quick_similarity_report(dataframe, generated)
    runtime = pd.DataFrame(
        [
            {
                "dataset": config.dataset_name,
                "rows_used": len(dataframe),
                "generated_rows": len(generated),
                "fit_seconds": fit.seconds,
                "sample_seconds": sample.seconds,
                "fit_peak_memory_mb": fit.peak_memory_mb,
                "sample_peak_memory_mb": sample.peak_memory_mb,
                "peak_memory_mb": peak_memory_mb,
            }
        ]
    )
    if write_outputs:
        if output_dir is None:
            raise ValueError("results_dir is required when write_outputs=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        generated.to_csv(output_dir / f"{config.dataset_name}_generated_start.csv", index=False)
        report.to_csv(output_dir / f"{config.dataset_name}_similarity_start.csv", index=False)
        runtime.to_csv(output_dir / f"{config.dataset_name}_runtime_start.csv", index=False)

    return SamplerRun(
        generated=generated,
        similarity_report=report,
        runtime=runtime,
        fit_seconds=fit.seconds,
        sample_seconds=sample.seconds,
        fit_peak_memory_mb=fit.peak_memory_mb,
        sample_peak_memory_mb=sample.peak_memory_mb,
        peak_memory_mb=peak_memory_mb,
    )


def run_configured_dataset_experiment(
    config: DatasetExperimentConfig,
    *,
    root: str | Path | None = None,
    results_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> DatasetExperimentResult:
    np.random.seed(config.random_state)
    paths = experiment_paths(config, root=root, results_dir=results_dir)
    dataframe = pd.read_csv(paths.data_path)
    work = working_dataframe(dataframe, config)
    starter_run = run_starter_sampler(
        work,
        config,
        results_dir=paths.results_dir,
        write_outputs=write_outputs,
    )
    comparison = run_dataset_comparison(
        work,
        dataset_name=config.dataset_name,
        target_column=config.target_column,
        results_dir=paths.results_dir,
        dataframe_sampler_config=sampler_config_with_random_state(
            config.manual_sampler_config,
            config.random_state,
        ),
        llm_assisted_config=(
            sampler_config_with_random_state(config.llm_assisted_config, config.random_state)
            if config.llm_assisted_config is not None
            else None
        ),
        n_samples=config.n_generated,
        random_state=config.random_state,
    )
    return DatasetExperimentResult(
        dataframe=dataframe,
        working_dataframe=work,
        profile=dataset_profile(dataframe),
        starter_run=starter_run,
        comparison=comparison,
        paths=paths,
    )
