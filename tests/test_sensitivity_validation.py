import numpy as np
import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.make_tables import (
    DatasetTableMetadata,
    generate_all_tables,
    write_sensitivity_validation_table,
)
from experiments.plot_results import DistributionDashboardSpec, generate_all_figures, plot_sensitivity_validation
from experiments.sensitivity_validation import (
    run_sensitivity_validation_for_config,
    sensitivity_validation_report,
    summarize_sensitivity_validation,
)
from experiments.workflow import run_configured_dataset_experiment


def make_sensitivity_dataframe(n_rows=36):
    x = np.linspace(0.0, 1.0, n_rows)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.cos(x * np.pi),
            "binary": np.where(x > 0.5, 1, 0),
            "group": np.where(x < 0.33, "low", np.where(x < 0.66, "mid", "high")),
        }
    )


def make_sensitivity_config(**overrides):
    config = DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column="binary",
        sampler_config={
            "n_iterations": 1,
            "n_neighbours": 2,
            "knn_backend": "sklearn",
        },
        n_generated=8,
        random_state=5,
    )
    return DatasetExperimentConfig(**{**config.__dict__, **overrides})


def test_sensitivity_validation_reports_proposed_setup_rows():
    report = sensitivity_validation_report(
        dataframe=make_sensitivity_dataframe(),
        dataset_name="toy",
        target_column="binary",
        sampler_config={"n_iterations": 1, "n_neighbours": 2, "knn_backend": "sklearn"},
        n_samples=8,
        random_state=1,
    )

    assert set(report["parameter"]) == {"setup"}
    assert set(report["value"]) == {"fast", "default", "accurate"}
    assert set(report["n_iterations"]) == {0, 1, 2}
    assert set(report["max_constraint_retries"]) == {0, 5, 20}
    assert {"nn_distance_ratio", "discrimination_accuracy", "utility_lift", "reason"}.issubset(report.columns)
    assert (report["reason"] == "ok").all()
    assert report[["nn_distance_ratio", "discrimination_accuracy"]].notna().all().all()


def test_sensitivity_summary_aggregates_by_parameter_and_value():
    rows = pd.DataFrame(
        {
            "dataset": ["a", "b"],
            "parameter": ["setup", "setup"],
            "value": ["default", "default"],
            "setup_label": ["DataFrameSampler default", "DataFrameSampler default"],
            "n_iterations": [1, 1],
            "max_constraint_retries": [5, 5],
            "calibrate_decoders": [False, False],
            "nn_distance_ratio": [1.0, 1.4],
            "discrimination_accuracy": [0.5, 0.7],
            "utility_lift": [0.0, 0.2],
            "distribution_similarity_score": [0.8, 1.0],
            "fit_seconds": [1.0, 3.0],
            "sample_seconds": [0.5, 1.5],
        }
    )

    summary = summarize_sensitivity_validation(rows)

    assert summary.loc[0, "datasets_evaluated"] == 2
    assert np.isclose(summary.loc[0, "mean_nn_distance_ratio"], 1.2)
    assert np.isclose(summary.loc[0, "mean_utility_lift"], 0.1)
    assert summary.loc[0, "setup_label"] == "DataFrameSampler default"


def test_run_sensitivity_validation_for_config_writes_csv(tmp_path):
    config = make_sensitivity_config()

    report = run_sensitivity_validation_for_config(
        config,
        make_sensitivity_dataframe(),
        results_dir=tmp_path,
    )

    assert report.empty
    assert not (tmp_path / "toy_sensitivity_validation.csv").exists()


def test_sensitivity_table_and_figure_helpers_write_outputs(tmp_path):
    rows = pd.DataFrame(
        {
            "dataset": ["toy", "toy", "toy"],
            "parameter": ["setup", "setup", "setup"],
            "value": ["fast", "default", "accurate"],
            "setup_label": [
                "DataFrameSampler fast",
                "DataFrameSampler default",
                "DataFrameSampler accurate",
            ],
            "n_iterations": [0, 1, 2],
            "max_constraint_retries": [0, 5, 20],
            "calibrate_decoders": [False, False, True],
            "nn_distance_ratio": [1.0, 1.1, 1.2],
            "discrimination_accuracy": [0.5, 0.55, 0.6],
            "utility_lift": [0.0, 0.1, 0.05],
            "distribution_similarity_score": [0.9, 0.8, 0.85],
            "fit_seconds": [1.0, 1.1, 1.2],
            "sample_seconds": [0.2, 0.3, 0.4],
        }
    )

    assert write_sensitivity_validation_table(rows, tables_dir=tmp_path).exists()
    assert plot_sensitivity_validation(rows, figures_dir=tmp_path).exists()


def test_workflow_writes_sensitivity_output(tmp_path):
    processed_dir = tmp_path / "experiments" / "data" / "processed"
    processed_dir.mkdir(parents=True)
    make_sensitivity_dataframe().to_csv(processed_dir / "adult.csv", index=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n")
    (tmp_path / "experiments").mkdir(exist_ok=True)

    result = run_configured_dataset_experiment(
        make_sensitivity_config(dataset_name="adult", data_filename="adult.csv"),
        root=tmp_path,
        results_dir=tmp_path / "results",
    )

    assert not result.sensitivity_validation.empty
    assert (tmp_path / "results" / "adult_sensitivity_validation.csv").exists()


def test_generate_all_tables_and_figures_include_sensitivity_outputs(tmp_path):
    results_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    results_dir.mkdir()
    processed_dir.mkdir()
    make_sensitivity_dataframe(8).to_csv(processed_dir / "toy.csv", index=False)
    make_sensitivity_dataframe(8).to_csv(results_dir / "toy_dataframe_sampler_generated.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "method": ["dataframe_sampler"],
            "n_real": [8],
            "n_synthetic": [8],
            "numeric_ks_statistic": [0.0],
            "categorical_total_variation": [0.0],
            "mean_abs_association_difference": [0.0],
            "numeric_histogram_overlap": [1.0],
            "categorical_coverage": [1.0],
            "rare_category_preservation": [1.0],
            "nn_distance_ratio": [1.0],
            "nn_suspiciously_close_rate": [0.0],
            "discrimination_accuracy": [0.5],
            "discrimination_privacy_score": [1.0],
            "utility_task": ["classification"],
            "utility_real_score": [1.0],
            "utility_augmented_score": [1.0],
            "utility_lift": [0.0],
            "distribution_histogram_overlap": [1.0],
            "distribution_numeric_kl": [0.0],
            "distribution_categorical_jsd": [0.0],
            "distribution_similarity_score": [1.0],
            "fit_seconds": [0.1],
            "sample_seconds": [0.1],
            "fit_peak_memory_mb": [0.0],
            "sample_peak_memory_mb": [0.0],
            "peak_memory_mb": [0.0],
        }
    ).to_csv(results_dir / "toy_baseline_comparison.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "parameter": ["setup"],
            "value": ["default"],
            "setup_label": ["DataFrameSampler default"],
            "n_iterations": [1],
            "max_constraint_retries": [5],
            "calibrate_decoders": [False],
            "nn_distance_ratio": [1.0],
            "discrimination_accuracy": [0.5],
            "utility_lift": [0.0],
            "distribution_similarity_score": [1.0],
            "fit_seconds": [0.1],
            "sample_seconds": [0.1],
        }
    ).to_csv(results_dir / "toy_sensitivity_validation.csv", index=False)

    table_outputs = generate_all_tables(
        results_dir=results_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        dataset_metadata=[DatasetTableMetadata("toy", "Toy", "Toy", "None", "Test")],
    )
    figure_outputs = generate_all_figures(
        results_dir=results_dir,
        data_dir=processed_dir,
        figures_dir=figures_dir,
        dashboard_spec=DistributionDashboardSpec(
            dataset_name="toy",
            generated_method="dataframe_sampler",
            numeric_column="x",
            categorical_column="group",
            correlation_columns=["x", "y"],
        ),
    )

    assert tables_dir / "sensitivity_validation.tex" in table_outputs
    assert figures_dir / "sensitivity_validation.pdf" in figure_outputs
