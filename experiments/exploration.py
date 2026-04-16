from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def column_distribution_summary(dataframe: pd.DataFrame, *, max_categories: int = 8) -> pd.DataFrame:
    """Return compact per-column distribution summaries for notebook EDA."""
    rows: list[dict[str, Any]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        base = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
        }
        if _is_plottable_numeric(series):
            rows.append(
                {
                    **base,
                    "kind": "numeric",
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "median": series.median(),
                    "max": series.max(),
                    "top_values": "",
                }
            )
        else:
            top = series.dropna().astype(str).value_counts(normalize=True).head(max_categories)
            rows.append(
                {
                    **base,
                    "kind": "categorical",
                    "mean": pd.NA,
                    "std": pd.NA,
                    "min": pd.NA,
                    "median": pd.NA,
                    "max": pd.NA,
                    "top_values": "; ".join(f"{idx}: {value:.2f}" for idx, value in top.items()),
                }
            )
    return pd.DataFrame(rows)


def plot_column_distributions(
    dataframe: pd.DataFrame,
    *,
    title: str | None = None,
    max_categories: int = 10,
    max_columns: int | None = None,
):
    """Plot a small-multiple distribution view for numeric and categorical columns."""
    columns = list(dataframe.columns[:max_columns]) if max_columns is not None else list(dataframe.columns)
    if not columns:
        raise ValueError("dataframe must have at least one column")

    ncols = min(3, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes_list = list(pd.Series(axes.ravel() if hasattr(axes, "ravel") else [axes]))

    for ax, column in zip(axes_list, columns):
        series = dataframe[column]
        if _is_plottable_numeric(series):
            ax.hist(pd.to_numeric(series, errors="coerce").dropna(), bins=24, color="#4C78A8", alpha=0.85)
            ax.set_ylabel("count")
        else:
            counts = series.dropna().astype(str).value_counts().head(max_categories)
            ax.barh(counts.index[::-1], counts.to_numpy()[::-1], color="#59A14F", alpha=0.85)
            ax.set_xlabel("count")
        ax.set_title(str(column))
        ax.grid(alpha=0.2)

    for ax in axes_list[len(columns) :]:
        ax.axis("off")
    if title:
        fig.suptitle(f"{title}: column distributions")
    fig.tight_layout()
    return fig


def plot_pairwise_features(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    title: str | None = None,
    max_numeric: int = 5,
    sample_size: int = 1000,
    random_state: int = 42,
):
    """Plot pairwise relationships for an already numeric feature view."""
    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if column != target_column and _is_plottable_numeric(dataframe[column])
    ][:max_numeric]
    if len(numeric_columns) < 2:
        raise ValueError("At least two numeric feature columns are required for pairwise plotting.")

    plot_data = dataframe[numeric_columns].copy()
    if len(plot_data) > sample_size:
        plot_data = plot_data.sample(n=sample_size, random_state=random_state)
    fig, axes = plt.subplots(
        len(numeric_columns),
        len(numeric_columns),
        figsize=(2.8 * len(numeric_columns), 2.8 * len(numeric_columns)),
    )
    axes_array = np.asarray(axes).reshape(len(numeric_columns), len(numeric_columns))
    for row_idx, y_column in enumerate(numeric_columns):
        for col_idx, x_column in enumerate(numeric_columns):
            ax = axes_array[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(pd.to_numeric(plot_data[x_column], errors="coerce").dropna(), bins=24, color="#4C78A8", alpha=0.85)
            else:
                x = pd.to_numeric(plot_data[x_column], errors="coerce")
                y = pd.to_numeric(plot_data[y_column], errors="coerce")
                pair = pd.DataFrame({"x": x, "y": y}).dropna()
                if row_idx > col_idx:
                    _plot_pair_density(ax, pair["x"].to_numpy(), pair["y"].to_numpy())
                else:
                    ax.scatter(pair["x"], pair["y"], s=9, alpha=0.28, color="#1f77b4", linewidths=0)
            if row_idx == len(numeric_columns) - 1:
                ax.set_xlabel(str(x_column))
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(str(y_column))
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.15)
    if title:
        fig.suptitle(f"{title}: pairwise numeric feature plots")
        fig.subplots_adjust(top=0.92)
    fig.tight_layout()
    return fig


def _is_plottable_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _plot_pair_density(ax, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return
    try:
        density = gaussian_kde(np.vstack([x, y]))
    except (np.linalg.LinAlgError, ValueError):
        return
    x_min, x_max = np.nanpercentile(x, [1, 99])
    y_min, y_max = np.nanpercentile(y, [1, 99])
    if x_min == x_max or y_min == y_max:
        return
    x_grid, y_grid = np.mgrid[x_min:x_max:40j, y_min:y_max:40j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = density(positions).reshape(x_grid.shape)
    ax.contourf(x_grid, y_grid, z, levels=8, cmap="Blues", alpha=0.32)
    ax.contour(x_grid, y_grid, z, levels=5, colors="#4C78A8", linewidths=0.6, alpha=0.55)
