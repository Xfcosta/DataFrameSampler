import pandas as pd
import matplotlib.pyplot as plt

from dataframe_sampler import ConcreteDataFrameSampler
from experiments.numeric_projection import (
    numeric_view,
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
