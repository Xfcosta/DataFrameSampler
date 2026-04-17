import numpy as np
import pandas as pd

from experiments.metrics import (
    categorical_similarity,
    classification_scores,
    dependence_similarity,
    downstream_utility_scores,
    inspectability_metrics,
    main_measure_report,
    nearest_neighbor_distance_test,
    numeric_similarity,
    practicality_metrics,
    regression_scores,
    theils_u,
    utility_lift_test,
)


def make_real_and_synthetic():
    real = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [10.0, 20.0, 30.0, 40.0],
            "segment": ["a", "a", "b", "rare"],
            "region": ["north", "north", "south", "west"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": [22, 31, 39, 48],
            "income": [11.0, 21.0, 31.0, 39.0],
            "segment": ["a", "b", "b", "new"],
            "region": ["north", "south", "south", "west"],
        }
    )
    return real, synthetic


def test_numeric_similarity_reports_todo_metrics():
    real, synthetic = make_real_and_synthetic()

    metrics = numeric_similarity(real, synthetic, columns=["age"])

    assert list(metrics["column"]) == ["age"]
    assert metrics.loc[0, "mean_abs_error"] == 0.0
    assert metrics.loc[0, "std_abs_error"] >= 0.0
    assert 0.0 <= metrics.loc[0, "ks_statistic"] <= 1.0
    assert metrics.loc[0, "wasserstein_distance"] >= 0.0
    assert 0.0 <= metrics.loc[0, "histogram_overlap"] <= 1.0


def test_categorical_similarity_reports_todo_metrics():
    real, synthetic = make_real_and_synthetic()

    metrics = categorical_similarity(real, synthetic, columns=["segment"], rare_threshold=0.26)

    assert list(metrics["column"]) == ["segment"]
    assert 0.0 <= metrics.loc[0, "total_variation_distance"] <= 1.0
    assert 0.0 <= metrics.loc[0, "jensen_shannon_divergence"] <= 1.0
    assert 0.0 <= metrics.loc[0, "category_coverage"] <= 1.0
    assert 0.0 <= metrics.loc[0, "rare_category_preservation"] <= 1.0


def test_dependence_similarity_reports_mixed_association_difference():
    real, synthetic = make_real_and_synthetic()

    metrics = dependence_similarity(real, synthetic, columns=["age", "income", "segment"])

    assert metrics["real_association"].shape == (3, 3)
    assert metrics["synthetic_association"].shape == (3, 3)
    assert metrics["mean_abs_association_difference"] >= 0.0
    assert metrics["max_abs_association_difference"] >= 0.0
    assert 0.0 <= theils_u(real["segment"], real["region"]) <= 1.0


def test_downstream_classification_and_regression_scores():
    classification = classification_scores(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0.1, 0.8, 0.4, 0.2],
    )
    regression = regression_scores([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])
    utility = downstream_utility_scores(
        train_on_synthetic_test_on_real=classification,
        train_on_real_test_on_real={"accuracy": 1.0},
        train_on_bootstrap_test_on_real={"accuracy": 0.75},
    )

    assert set(["accuracy", "f1", "roc_auc", "brier_score"]).issubset(classification)
    assert set(["mae", "rmse", "r2"]).issubset(regression)
    assert utility["evaluation"].tolist() == [
        "train_on_synthetic_test_on_real",
        "train_on_real_test_on_real",
        "train_on_bootstrap_test_on_real",
    ]


def test_inspectability_and_practicality_metrics():
    traces = [
        {
            "anchor": 1,
            "neighbour": 2,
            "second_neighbour": 3,
            "anchor_bins": [0],
            "neighbour_bins": [1],
            "second_neighbour_bins": [2],
            "generated_bins": [1],
            "decoded_bins": {"age": 1},
        },
        {"anchor": 1},
    ]

    inspectability = inspectability_metrics(traces)
    practicality = practicality_metrics(
        configuration={"n_components": 2, "n_iterations": 2},
        python_code="sampler.fit(df)\ngenerated = sampler.generate(10)\n",
        cli_command="dataframe-sampler -i input.csv -o output.csv -n 10",
        fit_seconds=0.5,
        sample_seconds=0.2,
        peak_memory_mb=128.0,
    ).to_dict()

    assert inspectability["trace_completeness_rate"] == 0.5
    assert inspectability["average_trace_size"] == np.mean([8, 1])
    assert practicality["configuration_choices"] == 2
    assert practicality["lines_of_python"] == 2
    assert practicality["cli_command_length"] == 7



def test_primary_measure_report_contains_four_experiment_measures():
    real, synthetic = make_real_and_synthetic()
    real["target"] = [0, 0, 1, 1]
    synthetic["target"] = [0, 1, 1, 1]

    report = main_measure_report(real, synthetic, target_column="target", random_state=1)

    assert "nn_distance_ratio" in report
    assert "discrimination_accuracy" in report
    assert "utility_lift" in report
    assert "distribution_histogram_overlap" in report
    assert 0.0 <= report["distribution_histogram_overlap"] <= 1.0


def test_utility_lift_snaps_continuous_synthetic_labels_for_classification():
    real = pd.DataFrame(
        {
            "feature": np.linspace(0.0, 1.0, 40),
            "segment": ["a", "b"] * 20,
            "target": [0, 1] * 20,
        }
    )
    synthetic = pd.DataFrame(
        {
            "feature": np.linspace(0.05, 0.95, 24),
            "segment": ["a", "b"] * 12,
            "target": np.linspace(-0.1, 1.1, 24),
        }
    )

    report = utility_lift_test(real, synthetic, target_column="target", random_state=1)

    assert report["utility_task"] == "classification"
    assert np.isfinite(report["utility_real_score"])
    assert np.isfinite(report["utility_augmented_score"])
    assert np.isfinite(report["utility_lift"])


def test_nearest_neighbor_distance_test_compares_to_natural_distances():
    real = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "group": ["a", "a", "b", "b"]})
    synthetic = pd.DataFrame({"x": [0.1, 2.9], "group": ["a", "b"]})

    report = nearest_neighbor_distance_test(real, synthetic)

    assert report["nn_synthetic_to_real_mean"] >= 0.0
    assert report["nn_real_to_real_mean"] > 0.0
    assert report["nn_distance_ratio"] >= 0.0


def test_nearest_neighbor_distance_handles_nullable_pandas_missing_values():
    real = pd.DataFrame(
        {
            "age": pd.Series([20, 30, pd.NA, 50], dtype="Int64"),
            "flag": pd.Series([True, False, pd.NA, True], dtype="boolean"),
            "group": pd.Series(["a", pd.NA, "b", "b"], dtype="string"),
        }
    )
    synthetic = pd.DataFrame(
        {
            "age": pd.Series([21, pd.NA, 49], dtype="Int64"),
            "flag": pd.Series([False, True, pd.NA], dtype="boolean"),
            "group": pd.Series(["a", "b", pd.NA], dtype="string"),
        }
    )

    report = nearest_neighbor_distance_test(real, synthetic)

    assert np.isfinite(report["nn_synthetic_to_real_mean"])
    assert np.isfinite(report["nn_real_to_real_mean"])
    assert report["nn_distance_ratio"] >= 0.0
