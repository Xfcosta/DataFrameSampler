import numpy as np
import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.make_tables import (
    DatasetTableMetadata,
    generate_all_tables,
    write_decoder_calibration_table,
    write_mechanism_validation_table,
)
from experiments.mechanism_validation import (
    mechanism_validation_report,
    run_mechanism_validation_for_config,
    summarize_decoder_calibration,
    summarize_mechanism_validation,
)
from experiments.plot_results import (
    DistributionDashboardSpec,
    generate_all_figures,
    plot_decoder_calibration,
    plot_mechanism_validation,
)
from experiments.workflow import run_configured_dataset_experiment


def make_mechanism_dataframe(n_rows=48):
    x = np.linspace(0.0, 1.0, n_rows)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x * np.pi),
            "binary": np.where(x > 0.5, 1, 0),
            "group": np.where(x < 0.33, "low", np.where(x < 0.66, "mid", "high")),
            "constant": "same",
            "id_like": [f"id-{idx:03d}" for idx in range(n_rows)],
        }
    )


def make_mechanism_config(**overrides):
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
        n_generated=12,
        random_state=3,
    )
    return DatasetExperimentConfig(**{**config.__dict__, **overrides})


def test_mechanism_validation_reports_expected_columns_and_edge_cases():
    train = make_mechanism_dataframe(36).iloc[:24].reset_index(drop=True)
    test = make_mechanism_dataframe(36).iloc[24:].reset_index(drop=True)

    mechanism, calibration = mechanism_validation_report(
        train=train,
        test=test,
        dataset_name="toy",
        sampler_config={"n_iterations": 1, "n_neighbours": 2, "knn_backend": "sklearn"},
        random_state=1,
    )

    assert {
        "dataset",
        "column",
        "nca_accuracy",
        "pca_accuracy",
        "nca_accuracy_lift_over_majority",
        "nca_accuracy_lift_over_pca",
    }.issubset(mechanism.columns)
    assert {"binary", "group", "constant", "id_like"}.issubset(set(mechanism["column"]))
    assert mechanism.loc[mechanism["column"] == "constant", "reason"].iloc[0] == "one_train_class"
    assert {
        "negative_log_loss",
        "brier_score",
        "expected_calibration_error",
        "cardinality_bucket",
    }.issubset(calibration.columns)
    ok_calibration = calibration[calibration["reason"] == "ok"]
    assert (ok_calibration[["negative_log_loss", "brier_score", "expected_calibration_error"]] >= 0).all().all()


def test_summaries_aggregate_mechanism_and_calibration_rows():
    mechanism = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "column": ["a", "b"],
            "cardinality": [2, 4],
            "nca_accuracy": [0.8, 0.5],
            "majority_accuracy": [0.6, 0.4],
            "pca_accuracy": [0.7, 0.3],
            "raw_context_accuracy": [0.9, 0.6],
            "nca_accuracy_lift_over_majority": [0.2, 0.1],
            "nca_accuracy_lift_over_pca": [0.1, 0.2],
        }
    )
    calibration = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "cardinality_bucket": ["binary", "low"],
            "column": ["a", "b"],
            "accuracy": [0.8, 0.5],
            "mean_top_confidence": [0.7, 0.6],
            "confidence_gap": [-0.1, 0.1],
            "negative_log_loss": [0.3, 0.9],
            "brier_score": [0.2, 0.4],
            "expected_calibration_error": [0.1, 0.2],
        }
    )

    mechanism_summary = summarize_mechanism_validation(mechanism)
    calibration_summary = summarize_decoder_calibration(calibration)

    assert mechanism_summary.loc[0, "columns_evaluated"] == 2
    assert np.isclose(mechanism_summary.loc[0, "mean_lift_over_pca"], 0.15)
    assert set(calibration_summary["cardinality_bucket"]) == {"binary", "low"}


def test_run_mechanism_validation_for_config_writes_csvs(tmp_path):
    config = make_mechanism_config()

    mechanism, calibration = run_mechanism_validation_for_config(
        config,
        make_mechanism_dataframe(),
        results_dir=tmp_path,
    )

    assert not mechanism.empty
    assert not calibration.empty
    assert (tmp_path / "toy_mechanism_validation.csv").exists()
    assert (tmp_path / "toy_decoder_calibration.csv").exists()


def test_mechanism_table_and_figure_helpers_write_outputs(tmp_path):
    mechanism = pd.DataFrame(
        {
            "dataset": ["toy"],
            "column": ["group"],
            "cardinality": [3],
            "nca_accuracy": [0.8],
            "majority_accuracy": [0.4],
            "pca_accuracy": [0.6],
            "raw_context_accuracy": [0.9],
            "nca_accuracy_lift_over_majority": [0.4],
            "nca_accuracy_lift_over_pca": [0.2],
        }
    )
    calibration = pd.DataFrame(
        {
            "dataset": ["toy"],
            "cardinality_bucket": ["low"],
            "column": ["group"],
            "accuracy": [0.8],
            "mean_top_confidence": [0.75],
            "confidence_gap": [-0.05],
            "negative_log_loss": [0.4],
            "brier_score": [0.2],
            "expected_calibration_error": [0.1],
        }
    )

    assert write_mechanism_validation_table(mechanism, tables_dir=tmp_path).exists()
    assert write_decoder_calibration_table(calibration, tables_dir=tmp_path).exists()
    assert plot_mechanism_validation(mechanism, figures_dir=tmp_path).exists()
    assert plot_decoder_calibration(calibration, figures_dir=tmp_path).exists()


def test_workflow_writes_mechanism_and_calibration_outputs(tmp_path):
    processed_dir = tmp_path / "experiments" / "data" / "processed"
    processed_dir.mkdir(parents=True)
    make_mechanism_dataframe().to_csv(processed_dir / "toy.csv", index=False)
    (tmp_path / "pyproject.toml").write_text("[project]\\nname='toy'\\n")
    (tmp_path / "experiments").mkdir(exist_ok=True)
    config = make_mechanism_config(data_filename="toy.csv")
    result = run_configured_dataset_experiment(config, root=tmp_path, results_dir=tmp_path / "results")

    assert not result.mechanism_validation.empty
    assert not result.decoder_calibration.empty
    assert (tmp_path / "results" / "toy_mechanism_validation.csv").exists()
    assert (tmp_path / "results" / "toy_decoder_calibration.csv").exists()


def test_generate_all_tables_and_figures_include_mechanism_outputs(tmp_path):
    results_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    data_dir = processed_dir
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "target": [0, 1, 0], "group": ["a", "b", "a"]}).to_csv(
        processed_dir / "toy.csv",
        index=False,
    )
    pd.DataFrame({"x": [0.1, 1.1, 1.8], "target": [0, 1, 0], "group": ["a", "b", "a"]}).to_csv(
        results_dir / "toy_dataframe_sampler_generated.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "method": ["dataframe_sampler"],
            "n_real": [3],
            "n_synthetic": [3],
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
            "column": ["group"],
            "cardinality": [2],
            "nca_accuracy": [0.8],
            "majority_accuracy": [0.5],
            "pca_accuracy": [0.6],
            "raw_context_accuracy": [0.9],
            "nca_accuracy_lift_over_majority": [0.3],
            "nca_accuracy_lift_over_pca": [0.2],
        }
    ).to_csv(results_dir / "toy_mechanism_validation.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "cardinality_bucket": ["binary"],
            "column": ["group"],
            "accuracy": [0.8],
            "mean_top_confidence": [0.7],
            "confidence_gap": [-0.1],
            "negative_log_loss": [0.4],
            "brier_score": [0.2],
            "expected_calibration_error": [0.1],
        }
    ).to_csv(results_dir / "toy_decoder_calibration.csv", index=False)

    table_outputs = generate_all_tables(
        results_dir=results_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        dataset_metadata=[DatasetTableMetadata("toy", "Toy", "Toy", "None", "Test")],
    )
    figure_outputs = generate_all_figures(
        results_dir=results_dir,
        data_dir=data_dir,
        figures_dir=figures_dir,
        dashboard_spec=DistributionDashboardSpec(
            dataset_name="toy",
            generated_method="dataframe_sampler",
            numeric_column="x",
            categorical_column="group",
            correlation_columns=["x", "target"],
        ),
    )

    assert tables_dir / "mechanism_validation.tex" in table_outputs
    assert tables_dir / "decoder_calibration.tex" in table_outputs
    assert figures_dir / "mechanism_validation.pdf" in figure_outputs
    assert figures_dir / "decoder_calibration.pdf" in figure_outputs
    assert figures_dir / "what_context.pdf" not in figure_outputs
    assert figures_dir / "how_pipeline.pdf" not in figure_outputs
