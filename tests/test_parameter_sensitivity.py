import numpy as np
import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.parameter_sensitivity import (
    parameter_sensitivity_report,
    plot_parameter_sensitivity,
    run_adult_parameter_sensitivity,
)


def make_parameter_sensitivity_dataframe(n_rows=40):
    x = np.linspace(0.0, 1.0, n_rows)
    return pd.DataFrame(
        {
            "age": 20 + (x * 40).round().astype(int),
            "hours": 20 + (x * 30),
            "education": np.where(x < 0.5, "HS-grad", "Bachelors"),
            "income": np.where(x + 0.1 * np.sin(10 * x) > 0.55, ">50K", "<=50K"),
        }
    )


def make_adult_config():
    return DatasetExperimentConfig(
        dataset_name="adult",
        title="Adult",
        data_filename="adult.csv",
        target_column="income",
        n_generated=12,
        random_state=7,
        sampler_config={
            "n_neighbours": 2,
            "knn_backend": "sklearn",
            "nca_fit_sample_size": 0.25,
            "decoder_kwargs": {"n_estimators": 5, "n_jobs": 1},
            "nca_kwargs": {"max_iter": 5},
        },
    )


def test_parameter_sensitivity_report_runs_one_at_a_time_grids():
    report = parameter_sensitivity_report(
        dataframe=make_parameter_sensitivity_dataframe(),
        dataset_name="adult",
        target_column="income",
        sampler_config=make_adult_config().sampler_config,
        n_samples=8,
        random_state=7,
        n_components_grid=[1, 2],
        nca_fit_sample_size_grid=[0.25],
        lambda_grid=[0.5, 1.0],
        n_iterations_grid=[0, 1],
    )

    assert set(report["parameter"]) == {"n_components", "nca_fit_sample_size", "lambda_", "n_iterations"}
    assert len(report) == 7
    assert {"distribution_similarity_score", "utility_lift", "discrimination_accuracy"}.issubset(report.columns)


def test_run_adult_parameter_sensitivity_writes_csv_and_plot(tmp_path):
    config = make_adult_config()
    report = run_adult_parameter_sensitivity(
        config,
        make_parameter_sensitivity_dataframe(),
        results_dir=tmp_path,
        max_train_rows=30,
        n_samples=8,
        n_components_grid=[1],
        nca_fit_sample_size_grid=[0.25],
        lambda_grid=[1.0],
        n_iterations_grid=[1],
    )

    assert len(report) == 4
    assert (tmp_path / "adult_parameter_sensitivity.csv").exists()
    fig = plot_parameter_sensitivity(report, figures_dir=tmp_path)
    assert fig is not None
    assert (tmp_path / "adult_parameter_sensitivity.pdf").exists()
