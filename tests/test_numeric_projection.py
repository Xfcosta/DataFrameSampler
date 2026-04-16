import pandas as pd
import matplotlib.pyplot as plt

from dataframe_sampler import ConcreteDataFrameSampler
from experiments.datasets import DatasetExperimentConfig
from experiments.numeric_projection import (
    numeric_view,
    numeric_view_from_config,
    plot_numeric_projection_triptych,
    project_numeric_views,
)


def make_mixed_dataframe():
    return pd.DataFrame(
        {
            "age": [21, 22, 23, 35, 36, 37, 48, 49, 50, 62, 63, 64],
            "band": [
                "young",
                "young",
                "young",
                "adult",
                "adult",
                "adult",
                "middle",
                "middle",
                "middle",
                "senior",
                "senior",
                "senior",
            ],
            "score": [3, 4, 5, 10, 11, 12, 17, 18, 19, 24, 25, 26],
        }
    )


def test_numeric_view_uses_fitted_sampler_transform():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)
    sampler.fit(df)

    numeric = numeric_view(df.head(4), sampler)

    assert numeric.shape == (4, 3)
    assert all(dtype.kind in {"i", "u", "f"} for dtype in numeric.dtypes)


def test_numeric_view_from_config_fits_configured_sampler():
    df = make_mixed_dataframe()
    config = DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column=None,
        random_state=1,
        manual_sampler_config={
            "n_bins": 4,
            "n_neighbours": 3,
            "knn_backend": "sklearn",
            "embedding_method": "pca",
        },
    )

    numeric = numeric_view_from_config(df, config)

    assert numeric.shape == df.shape
    assert all(dtype.kind in {"i", "u", "f"} for dtype in numeric.dtypes)


def test_project_numeric_views_returns_shared_two_dimensional_frames():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)
    sampler.fit(df)
    generated = sampler.sample(n_samples=5)

    original_projection, generated_projection, reducer_name = project_numeric_views(
        numeric_view(df, sampler),
        numeric_view(generated, sampler),
        reducer="pca",
        random_state=2,
    )

    assert original_projection.shape == (len(df), 2)
    assert generated_projection.shape == (len(generated), 2)
    assert reducer_name == "PCA"


def test_plot_numeric_projection_triptych_returns_three_panel_figure():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)
    sampler.fit(df)
    generated = sampler.sample(n_samples=5)

    fig = plot_numeric_projection_triptych(
        sampler,
        df,
        generated,
        title="Toy",
        reducer="pca",
        random_state=2,
    )

    assert len(fig.axes) == 3
    assert [ax.get_title().split(": ")[-1] for ax in fig.axes] == [
        "original",
        "generated",
        "superimposed",
    ]
    plt.close(fig)


def test_plot_numeric_projection_triptych_uses_binary_target_markers():
    df = make_mixed_dataframe()
    df["target"] = [0, 1] * 6
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)
    sampler.fit(df)
    generated = sampler.sample(n_samples=6)

    fig = plot_numeric_projection_triptych(
        sampler,
        df,
        generated,
        target_column="target",
        reducer="pca",
        random_state=2,
    )

    assert len(fig.axes[0].collections) == 2
    assert len(fig.axes[1].collections) == 2
    assert len(fig.axes[2].collections) == 4
    plt.close(fig)


def test_plot_numeric_projection_triptych_uses_regression_target_sizes():
    df = make_mixed_dataframe()
    df["target"] = [float(value) for value in range(len(df))]
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)
    sampler.fit(df)
    generated = sampler.sample(n_samples=6)

    fig = plot_numeric_projection_triptych(
        sampler,
        df,
        generated,
        target_column="target",
        reducer="pca",
        random_state=2,
    )

    sizes = fig.axes[0].collections[0].get_sizes()
    assert sizes.max() > sizes.min()
    assert len(fig.axes[2].collections) == 2
    plt.close(fig)
