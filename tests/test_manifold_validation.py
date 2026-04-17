import numpy as np
import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.make_tables import DatasetTableMetadata, generate_all_tables, write_manifold_validation_table
from experiments.manifold_validation import (
    FrozenIsomapGeometry,
    convex_hull_membership,
    latent_bootstrap_baseline,
    latent_interpolation_baseline,
    manifold_validation_report,
    run_manifold_validation_for_config,
    summarize_manifold_validation,
)
from experiments.plot_results import DistributionDashboardSpec, generate_all_figures, plot_manifold_validation_stress


def test_convex_hull_membership_marks_inside_and_outside_triangle():
    train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = np.array([[0.25, 0.25], [1.0, 1.0]])

    inside = convex_hull_membership(train, query)

    assert inside.tolist() == [True, False]


def test_convex_hull_membership_handles_rank_deficient_training_data():
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    query = np.array([[1.0, 1.0], [3.0, 3.0]])

    inside = convex_hull_membership(train, query)

    assert inside.tolist() == [True, False]


def test_latent_interpolation_baseline_stays_inside_simple_convex_hull():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    generated = latent_interpolation_baseline(train, n_samples=20, n_neighbors=2, random_state=1)

    inside = convex_hull_membership(train, generated)

    assert inside.all()


def test_latent_bootstrap_baseline_reuses_training_latent_rows():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    generated = latent_bootstrap_baseline(train, n_samples=20, random_state=1)

    assert set(generated.ravel()).issubset(set(train.ravel()))


def test_frozen_isomap_insertion_stress_is_finite_and_non_negative():
    theta = np.linspace(0.0, np.pi, 12)
    train = np.column_stack([np.cos(theta), np.sin(theta)])
    points = train[[2, 5, 8]] + 0.01

    geometry = FrozenIsomapGeometry.fit(train, n_neighbors=3)
    stress = geometry.insertion_stress(points)

    assert stress["stress"].notna().all()
    assert (stress["stress"] >= 0).all()


def test_manifold_validation_summary_uses_real_q95_acceptance_threshold():
    pointwise = pd.DataFrame(
        {
            "dataset": ["toy"] * 7,
            "method": [
                "held_out_real",
                "held_out_real",
                "held_out_real",
                "dataframe_sampler_manual",
                "dataframe_sampler_manual",
                "dataframe_sampler_manual",
                "latent_interpolation",
            ],
            "sample_type": ["real_test", "real_test", "real_test", "generated", "generated", "generated", "generated"],
            "stress": [0.1, 0.2, 0.3, 0.15, 0.29, 0.5, 0.12],
            "out_hull": [False, False, False, True, True, False, False],
        }
    )

    summary = summarize_manifold_validation(pointwise)
    dfs = summary[summary["method"] == "dataframe_sampler_manual"].iloc[0]

    assert dfs["out_hull_rate"] == 2 / 3
    assert dfs["out_hull_acceptance_at_real_q95"] == 1.0


def test_manifold_validation_report_contains_expected_pointwise_columns():
    train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    real_test = np.array([[1.5], [3.5]])
    generated = {"dataframe_sampler_manual": np.array([[6.0], [2.5]])}

    report = manifold_validation_report(
        train_latent=train,
        real_test_latent=real_test,
        generated_latents=generated,
        dataset_name="toy",
        max_eval_points=10,
        isomap_neighbors=2,
        random_state=1,
    )

    assert {"dataset", "method", "sample_type", "stress", "in_hull", "out_hull"}.issubset(report.columns)
    assert set(report["sample_type"]) == {"real_test", "generated"}


def test_run_manifold_validation_for_config_writes_csv(tmp_path):
    dataframe = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, 18),
            "y": np.linspace(1.0, 2.0, 18),
            "group": ["a", "b", "c"] * 6,
            "target": [0, 1] * 9,
        }
    )
    config = DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column="target",
        manual_sampler_config={"n_iterations": 1, "n_neighbours": 2, "knn_backend": "sklearn"},
        n_generated=8,
        random_state=1,
    )

    report = run_manifold_validation_for_config(
        config,
        dataframe,
        results_dir=tmp_path,
        max_eval_points=6,
        isomap_neighbors=2,
    )

    assert not report.empty
    assert (tmp_path / "toy_manifold_validation.csv").exists()
    assert {"held_out_real", "dataframe_sampler_manual", "latent_interpolation", "latent_bootstrap"}.issubset(
        set(report["method"])
    )


def test_manifold_table_and_figure_helpers_write_outputs(tmp_path):
    pointwise = pd.DataFrame(
        {
            "dataset": ["toy"] * 6,
            "method": [
                "held_out_real",
                "held_out_real",
                "dataframe_sampler_manual",
                "dataframe_sampler_manual",
                "latent_interpolation",
                "latent_interpolation",
            ],
            "sample_type": ["real_test", "real_test", "generated", "generated", "generated", "generated"],
            "stress": [0.1, 0.2, 0.11, 0.3, 0.1, 0.12],
            "out_hull": [False, False, True, False, False, False],
        }
    )

    table_path = write_manifold_validation_table(pointwise, tables_dir=tmp_path)
    figure_path = plot_manifold_validation_stress(pointwise, figures_dir=tmp_path)

    assert table_path.exists()
    assert table_path.with_suffix(".csv").exists()
    assert figure_path.exists()


def test_generate_all_tables_includes_manifold_table_when_csv_exists(tmp_path):
    results_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    results_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame({"x": [0, 1], "target": [0, 1]}).to_csv(processed_dir / "toy.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "method": ["dataframe_sampler_manual"],
            "method_label": ["DFS manual"],
            "n_real": [2],
            "n_synthetic": [2],
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
            "dataset": ["toy", "toy", "toy"],
            "method": ["held_out_real", "dataframe_sampler_manual", "latent_interpolation"],
            "sample_type": ["real_test", "generated", "generated"],
            "stress": [0.1, 0.12, 0.08],
            "out_hull": [False, True, False],
        }
    ).to_csv(results_dir / "toy_manifold_validation.csv", index=False)
    outputs = generate_all_tables(
        results_dir=results_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        dataset_metadata=[
            DatasetTableMetadata(
                key="toy",
                name="Toy",
                domain="Toy",
                sensitive="None",
                rationale="Test",
            )
        ],
    )

    assert tables_dir / "manifold_validation.tex" in outputs


def test_generate_all_figures_includes_manifold_figure_when_csv_exists(tmp_path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    data_dir.mkdir()
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "target": [0, 1, 0], "group": ["a", "b", "a"]}).to_csv(
        data_dir / "toy.csv",
        index=False,
    )
    pd.DataFrame({"x": [0.1, 1.1, 1.8], "target": [0, 1, 0], "group": ["a", "b", "a"]}).to_csv(
        results_dir / "toy_dataframe_sampler_manual_generated.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "dataset": ["toy"],
            "method": ["dataframe_sampler_manual"],
            "nn_distance_ratio": [1.0],
            "discrimination_accuracy": [0.5],
            "utility_lift": [0.0],
            "distribution_similarity_score": [1.0],
            "sample_seconds": [0.1],
        }
    ).to_csv(results_dir / "toy_baseline_comparison.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["toy", "toy", "toy"],
            "method": ["held_out_real", "dataframe_sampler_manual", "latent_interpolation"],
            "sample_type": ["real_test", "generated", "generated"],
            "stress": [0.1, 0.12, 0.08],
            "out_hull": [False, True, False],
        }
    ).to_csv(results_dir / "toy_manifold_validation.csv", index=False)

    outputs = generate_all_figures(
        results_dir=results_dir,
        data_dir=data_dir,
        figures_dir=figures_dir,
        dashboard_spec=DistributionDashboardSpec(
            dataset_name="toy",
            generated_method="dataframe_sampler_manual",
            numeric_column="x",
            categorical_column="group",
            correlation_columns=["x", "target"],
        ),
    )

    assert figures_dir / "manifold_validation_stress.pdf" in outputs
