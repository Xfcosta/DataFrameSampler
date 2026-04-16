from dataclasses import replace

import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.workflow import (
    dataset_profile,
    quick_similarity_report,
    run_starter_sampler,
    working_dataframe,
)


def make_workflow_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "fare": [10.0, 12.5, 20.0, 25.0, 30.0],
            "group": ["a", "a", "b", "b", "c"],
        }
    )


def make_workflow_config(**overrides):
    config = DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column=None,
        working_sample_size=3,
        n_generated=6,
        random_state=1,
        manual_sampler_config={
            "n_bins": 3,
            "n_neighbours": 2,
            "knn_backend": "sklearn",
        },
    )
    return replace(config, **overrides)


def test_dataset_profile_reports_dtype_missing_and_unique_counts():
    df = make_workflow_dataframe()

    profile = dataset_profile(df)

    assert list(profile.columns) == ["dtype", "missing", "unique"]
    assert profile.loc["age", "unique"] == 5
    assert profile.loc["group", "missing"] == 0


def test_working_dataframe_applies_reproducible_sample_size():
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=3)

    sampled = working_dataframe(df, config)

    assert sampled.shape == (3, 3)
    assert sampled.equals(working_dataframe(df, config))


def test_working_dataframe_uses_full_frame_when_sample_size_is_none():
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=None)

    sampled = working_dataframe(df, config)

    assert sampled.equals(df.reset_index(drop=True))


def test_quick_similarity_report_handles_numeric_and_categorical_columns():
    real = make_workflow_dataframe()
    synthetic = real.sample(n=5, replace=True, random_state=2).reset_index(drop=True)

    report = quick_similarity_report(real, synthetic)

    assert set(report["kind"]) == {"numeric", "categorical"}
    assert {"age", "fare", "group"} == set(report["column"])
    assert report.loc[report["column"] == "group", "category_coverage"].iloc[0] > 0


def test_run_starter_sampler_writes_reusable_outputs(tmp_path):
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=None, n_generated=6)

    run = run_starter_sampler(df, config, results_dir=tmp_path)

    assert run.generated.shape == (6, 3)
    assert run.peak_memory_mb >= 0
    assert {
        "fit_peak_memory_mb",
        "sample_peak_memory_mb",
        "peak_memory_mb",
    }.issubset(run.runtime.columns)
    assert (tmp_path / "toy_generated_start.csv").exists()
    assert (tmp_path / "toy_similarity_start.csv").exists()
    assert (tmp_path / "toy_runtime_start.csv").exists()
