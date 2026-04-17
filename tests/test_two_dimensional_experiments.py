import matplotlib.pyplot as plt
import numpy as np

from experiments.two_dimensional import (
    DEFAULT_TWO_DIMENSIONAL_SAMPLER_KWARGS,
    gaussian_distribution,
    gaussian_mixture_distribution,
    generate_two_dimensional_sample,
    make_two_dimensional_cases,
    nearest_neighbor_distance_report,
    nearest_neighbor_distance_summary,
    plot_nearest_neighbor_distance_histogram,
    plot_two_dimensional_suite,
    run_two_dimensional_suite,
    spiral_distribution,
    two_dimensional_summary,
)


def test_two_dimensional_generators_return_xy_frames():
    gaussian = gaussian_distribution(rows=20, random_state=1)
    spiral = spiral_distribution(rows=20, random_state=1)

    assert list(gaussian.columns) == ["x", "y"]
    assert list(spiral.columns) == ["x", "y"]
    assert gaussian.shape == (20, 2)
    assert spiral.shape == (20, 2)


def test_gaussian_mixture_has_components_with_different_density_scales():
    mixture = gaussian_mixture_distribution(rows=1200, random_state=12)
    centers = np.array([[-3.2, -1.2], [3.0, 0.3], [0.0, 3.8]])
    points = mixture[["x", "y"]].to_numpy()
    assigned = np.argmin(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    radial_std = [
        np.linalg.norm(points[assigned == component] - centers[component], axis=1).std()
        for component in range(3)
    ]

    assert max(radial_std) / min(radial_std) > 2.0


def test_generate_two_dimensional_sample_uses_dataframe_sampler():
    real = gaussian_distribution(rows=80, random_state=2)

    generated = generate_two_dimensional_sample(
        real,
        n_samples=25,
        random_state=3,
        sampler_kwargs={"n_neighbours": 4},
    )

    assert generated.shape == (25, 2)
    assert list(generated.columns) == ["x", "y"]


def test_default_sampler_kwargs_expose_dataframe_sampler_parameters():
    assert {
        "n_neighbours",
        "n_components",
        "n_iterations",
        "random_state",
        "knn_backend",
        "knn_backend_kwargs",
    } == set(DEFAULT_TWO_DIMENSIONAL_SAMPLER_KWARGS)


def test_two_dimensional_suite_summarises_and_plots():
    results = run_two_dimensional_suite(
        random_state=4,
        n_samples=30,
        sampler_kwargs={"n_neighbours": 4},
    )
    summary = two_dimensional_summary(results)
    fig = plot_two_dimensional_suite(results)

    assert len(results) == len(make_two_dimensional_cases(random_state=4))
    assert set(summary["generated_rows"]) == {30}
    assert {"case", "rows", "generated_rows"}.issubset(summary.columns)
    assert len(fig.axes) == len(results) * 3
    assert all("original" in fig.axes[idx].get_title() for idx in range(0, len(fig.axes), 3))
    assert all("generated" in fig.axes[idx].get_title() for idx in range(1, len(fig.axes), 3))
    assert all("superimposed" in fig.axes[idx].get_title() for idx in range(2, len(fig.axes), 3))
    plt.close(fig)


def test_nearest_neighbor_distance_report_compares_generated_and_real_distances():
    real = gaussian_distribution(rows=40, random_state=5)
    generated = generate_two_dimensional_sample(
        real,
        n_samples=20,
        random_state=6,
        sampler_kwargs={"n_neighbours": 4},
    )

    report = nearest_neighbor_distance_report(real, generated)
    fig = plot_nearest_neighbor_distance_histogram(real, generated, bins=12)

    assert set(report["kind"]) == {
        "generated_to_original_1nn",
        "original_to_original_1nn",
    }
    assert len(report.loc[report["kind"] == "generated_to_original_1nn"]) == len(generated)
    assert len(report.loc[report["kind"] == "original_to_original_1nn"]) == len(real)
    assert report["distance"].min() >= 0
    assert len(fig.axes[0].lines) == 2
    assert all(line.get_linewidth() == 3 for line in fig.axes[0].lines)
    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[0].get_ylabel() == "log1p(density)"
    assert fig.axes
    plt.close(fig)


def test_nearest_neighbor_distance_histogram_can_disable_log_scale():
    real = gaussian_distribution(rows=40, random_state=8)
    generated = generate_two_dimensional_sample(
        real,
        n_samples=20,
        random_state=9,
        sampler_kwargs={"n_neighbours": 4},
    )

    fig = plot_nearest_neighbor_distance_histogram(real, generated, bins=12, log=False)

    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[0].get_ylabel() == "density"
    plt.close(fig)


def test_nearest_neighbor_distance_summary_reports_each_suite_case():
    results = run_two_dimensional_suite(
        random_state=7,
        n_samples=20,
        sampler_kwargs={"n_neighbours": 4},
    )

    summary = nearest_neighbor_distance_summary(results)

    assert set(summary["case"]) == {case.key for case in make_two_dimensional_cases(random_state=7)}
    assert set(summary["kind"]) == {
        "generated_to_original_1nn",
        "original_to_original_1nn",
    }
