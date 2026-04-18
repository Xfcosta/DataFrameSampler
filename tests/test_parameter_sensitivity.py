import numpy as np
import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.parameter_sensitivity import (
    _best_sensitivity_row,
    parameter_sensitivity_report,
    plot_line_search_step,
    plot_parameter_sensitivity,
    plot_repeated_line_search_summary,
    rank_parameter_sweep,
    run_adult_parameter_sensitivity,
    run_iterative_parameter_line_search,
    run_repeated_iterative_parameter_line_search,
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


def test_best_sensitivity_row_uses_metric_direction_and_targets():
    rows = pd.DataFrame(
        {
            "value_numeric": [0.1, 0.5, 1.0],
            "distribution_similarity_score": [0.8, 0.9, 0.7],
            "discrimination_accuracy": [0.9, 0.52, 0.4],
            "nn_distance_ratio": [0.2, 0.8, 1.15],
        }
    )

    assert _best_sensitivity_row(rows, "distribution_similarity_score")["value_numeric"] == 0.5
    assert _best_sensitivity_row(rows, "discrimination_accuracy")["value_numeric"] == 0.5
    assert _best_sensitivity_row(rows, "nn_distance_ratio")["value_numeric"] == 1.0


def test_rank_parameter_sweep_and_line_search_plot_mark_average_best():
    rows = pd.DataFrame(
        {
            "parameter": ["lambda_", "lambda_", "lambda_"],
            "value": [0.1, 0.5, 1.0],
            "distribution_similarity_score": [0.8, 0.9, 0.7],
            "utility_lift": [0.0, 0.2, 0.1],
            "discrimination_accuracy": [0.9, 0.52, 0.4],
            "nn_distance_ratio": [0.2, 0.8, 1.15],
        }
    )

    ranked = rank_parameter_sweep(rows)

    assert ranked.iloc[0]["value"] == 0.5
    assert "average_quality_rank" in ranked
    fig = plot_line_search_step(ranked, "lambda_", ranked.iloc[0]["value"])
    assert fig is not None


def test_iterative_line_search_skips_nca_parameters_when_iterations_zero():
    config = make_adult_config()
    events = []
    history, sweeps = run_iterative_parameter_line_search(
        make_parameter_sensitivity_dataframe(n_rows=24),
        dataset_name="adult",
        target_column="income",
        base_config=config.sampler_config,
        parameter_grids={
            "n_iterations": [0],
            "n_components": [1, 2],
            "nca_fit_sample_size": [0.25, 0.5],
            "lambda_": [0.25],
        },
        max_train_rows=20,
        n_samples=4,
        random_state=7,
        progress_callback=events.append,
    )

    skipped = history[history["reason"].str.startswith("skipped")]

    assert {"n_components", "nca_fit_sample_size"} == set(skipped["parameter"])
    assert set(sweeps["parameter"]) == {"n_iterations", "lambda_"}
    assert [event["parameter"] for event in events] == [
        "n_iterations",
        "n_components",
        "nca_fit_sample_size",
        "lambda_",
    ]
    assert events[1]["sweep"].empty


def test_repeated_line_search_aggregates_seed_traces_and_plots():
    config = make_adult_config()
    events = []
    history, sweeps, summary = run_repeated_iterative_parameter_line_search(
        make_parameter_sensitivity_dataframe(n_rows=28),
        dataset_name="adult",
        target_column="income",
        base_config=config.sampler_config,
        parameter_grids={
            "n_iterations": [0],
            "n_components": [1, 2],
            "nca_fit_sample_size": [0.25, 0.5],
            "lambda_": [0.25, 0.5],
        },
        seeds=[7, 8, 9],
        max_train_rows=20,
        n_samples=4,
        progress_callback=events.append,
    )

    assert "seed" not in history.columns
    assert set(history["parameter"]) == {"n_iterations", "n_components", "nca_fit_sample_size", "lambda_"}
    assert set(sweeps["seed"]) == {7, 8, 9}
    assert {"n_seeds", "selection_count", "mean_average_quality_rank"}.issubset(summary.columns)
    assert summary["n_seeds"].max() == 3
    assert [event["parameter"] for event in events] == [
        "n_iterations",
        "n_components",
        "nca_fit_sample_size",
        "lambda_",
    ]
    assert len(events) == 4
    assert events[0]["summary"]["n_seeds"].max() == 3
    assert events[1]["summary"].empty
    fig = plot_repeated_line_search_summary(sweeps, summary)
    assert fig is not None
