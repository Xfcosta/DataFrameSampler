import pandas as pd
import pytest

from experiments.baselines import (
    GaussianCopulaEmpiricalBaseline,
    IndependentColumnBaseline,
    RowBootstrapBaseline,
    SmoteNcBaseline,
    StratifiedColumnBaseline,
    simple_baselines,
)
from experiments.compare import run_dataset_comparison


def make_baseline_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "fare": [10.0, 12.5, 20.0, 25.0, 30.0, 40.0],
            "group": ["a", "a", "b", "b", "b", "c"],
            "target": [0, 0, 1, 1, 1, 0],
        }
    )


def assert_sample_shape(sample):
    assert sample.shape == (12, 4)
    assert list(sample.columns) == ["age", "fare", "group", "target"]


def test_row_bootstrap_baseline_samples_complete_rows():
    df = make_baseline_dataframe()
    sample = RowBootstrapBaseline(random_state=1).fit(df).sample(12)

    assert_sample_shape(sample)
    assert set(sample["group"]).issubset(set(df["group"]))


def test_independent_column_baseline_samples_each_column():
    df = make_baseline_dataframe()
    sample = IndependentColumnBaseline(random_state=1).fit(df).sample(12)

    assert_sample_shape(sample)
    assert set(sample["age"]).issubset(set(df["age"]))
    assert set(sample["group"]).issubset(set(df["group"]))


def test_stratified_column_baseline_preserves_target_values():
    df = make_baseline_dataframe()
    sample = StratifiedColumnBaseline("target", random_state=1).fit(df).sample(12)

    assert_sample_shape(sample)
    assert set(sample["target"]).issubset({0, 1})


def test_gaussian_copula_empirical_baseline_generates_mixed_columns():
    df = make_baseline_dataframe()
    sample = GaussianCopulaEmpiricalBaseline(random_state=1).fit(df).sample(12)

    assert_sample_shape(sample)
    assert sample["age"].notna().all()
    assert set(sample["group"]).issubset(set(df["group"]))


def test_simple_baselines_include_target_dependent_baseline_when_requested():
    names = [spec.name for spec in simple_baselines(target_column="target", random_state=1)]

    assert names == [
        "row_bootstrap",
        "independent_columns",
        "gaussian_copula_empirical",
        "stratified_columns",
    ]


def test_smotenc_baseline_has_actionable_optional_dependency_message():
    pytest.importorskip("imblearn")
    df = make_baseline_dataframe()
    sample = SmoteNcBaseline("target", random_state=1, k_neighbors=1).fit(df).sample(12)

    assert_sample_shape(sample)


def test_run_dataset_comparison_writes_simple_baseline_summary(tmp_path):
    df = make_baseline_dataframe()

    summary = run_dataset_comparison(
        df,
        dataset_name="toy",
        target_column="target",
        results_dir=tmp_path,
        dataframe_sampler_config={
            "n_neighbours": 2,
            "knn_backend": "sklearn",
            "random_state": 1,
        },
        n_samples=10,
        random_state=1,
    )

    assert set(summary["method"]) == {
        "dataframe_sampler_default",
        "dataframe_sampler_manual",
        "row_bootstrap",
        "independent_columns",
        "gaussian_copula_empirical",
        "stratified_columns",
    }
    assert (tmp_path / "toy_baseline_comparison.csv").exists()
    assert {
        "fit_peak_memory_mb",
        "sample_peak_memory_mb",
        "peak_memory_mb",
        "nn_distance_ratio",
        "discrimination_accuracy",
        "utility_lift",
        "distribution_histogram_overlap",
    }.issubset(summary.columns)
    assert summary["peak_memory_mb"].notna().all()
    assert (summary["peak_memory_mb"] >= 0).all()
    for method in summary["method"]:
        assert (tmp_path / f"toy_{method}_generated.csv").exists()

