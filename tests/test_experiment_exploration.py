import matplotlib.pyplot as plt
import pandas as pd

from experiments.exploration import (
    column_distribution_summary,
    plot_column_distributions,
    plot_pairwise_features,
)


def make_exploration_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [10.0, 20.0, 30.0, 40.0],
            "score": [0.1, 0.3, 0.5, 0.8],
            "segment": ["a", "a", "b", "c"],
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

    assert fig.axes
    plt.close(fig)
