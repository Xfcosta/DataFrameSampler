import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.exploration import (
    column_distribution_summary,
    plot_column_distributions,
    plot_pairwise_features,
)


def make_exploration_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 24, 29, 35, 41, 50, 57, 64],
            "income": [10.0, 14.0, 18.0, 25.0, 32.0, 44.0, 51.0, 62.0],
            "score": [0.1, 0.18, 0.26, 0.35, 0.49, 0.63, 0.76, 0.88],
            "segment": ["a", "a", "b", "c", "a", "b", "c", "c"],
        }
    )


def test_column_distribution_summary_reports_numeric_and_categorical_columns():
    summary = column_distribution_summary(make_exploration_dataframe())

    assert set(summary["column"]) == {"age", "income", "score", "segment"}
    assert set(summary["kind"]) == {"numeric", "categorical"}
    assert summary.loc[summary["column"] == "segment", "top_values"].iloc[0]


def test_plot_column_distributions_returns_figure():
    fig = plot_column_distributions(make_exploration_dataframe(), title="Toy")

    assert fig.axes
    plt.close(fig)


def test_plot_pairwise_features_returns_figure():
    fig = plot_pairwise_features(make_exploration_dataframe(), target_column=None, title="Toy")
    axes = np.asarray(fig.axes).reshape(3, 3)

    assert fig.axes
    assert len(axes[1, 0].collections) > 1
    assert len(axes[0, 1].collections) == 1
    plt.close(fig)


def test_plot_pairwise_features_can_include_all_numeric_columns():
    df = make_exploration_dataframe()
    df["extra"] = np.linspace(10.0, 20.0, len(df))

    fig = plot_pairwise_features(df, target_column=None, max_numeric=None)
    axes = np.asarray(fig.axes).reshape(4, 4)

    assert axes[-1, -1].get_xlabel() == "extra"
    plt.close(fig)
