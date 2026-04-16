from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

from dataframe_sampler import ConcreteDataFrameSampler


DEFAULT_TWO_DIMENSIONAL_SAMPLER_KWARGS = {
    "n_bins": 60,
    "n_neighbours": 8,
    "vectorizing_columns_dict": None,
    "sampled_columns": None,
    "random_state": None,
    "knn_backend": "sklearn",
    "knn_backend_kwargs": None,
    "embedding_method": "mds",
    "embedding_kwargs": None,
    "numeric_decode_strategy": "continuous",
}


@dataclass(frozen=True)
class TwoDimensionalCase:
    key: str
    title: str
    dataframe: pd.DataFrame


def gaussian_distribution(
    *,
    rows: int = 500,
    mean: tuple[float, float] = (0.0, 0.0),
    covariance: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0)),
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    points = rng.multivariate_normal(mean=mean, cov=np.asarray(covariance, dtype=float), size=rows)
    return _to_dataframe(points)


def gaussian_mixture_distribution(*, rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    component = rng.choice([0, 1, 2], size=rows, p=[0.42, 0.33, 0.25])
    means = np.array([[-3.2, -1.2], [3.0, 0.3], [0.0, 3.8]])
    covariances = np.array(
        [
            [[0.04, 0.01], [0.01, 0.05]],
            [[0.42, -0.16], [-0.16, 0.30]],
            [[2.20, 0.80], [0.80, 1.55]],
        ]
    )
    points = np.vstack(
        [
            rng.multivariate_normal(means[idx], covariances[idx])
            for idx in component
        ]
    )
    return _to_dataframe(points)


def spiral_distribution(*, rows: int = 700, turns: float = 2.6, noise: float = 0.08, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    theta = np.linspace(0.25, turns * 2 * np.pi, rows)
    radius = np.linspace(0.15, 2.4, rows)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    points += rng.normal(0, noise, size=points.shape)
    rng.shuffle(points)
    return _to_dataframe(points)


def make_two_dimensional_cases(random_state: int = 42) -> list[TwoDimensionalCase]:
    return [
        TwoDimensionalCase(
            key="gaussian_isotropic",
            title="Isotropic Gaussian",
            dataframe=gaussian_distribution(
                covariance=((1.0, 0.0), (0.0, 1.0)),
                random_state=random_state,
            ),
        ),
        TwoDimensionalCase(
            key="gaussian_correlated",
            title="Correlated Gaussian",
            dataframe=gaussian_distribution(
                covariance=((1.0, 0.82), (0.82, 1.15)),
                random_state=random_state + 1,
            ),
        ),
        TwoDimensionalCase(
            key="gaussian_mixture",
            title="Gaussian Mixture",
            dataframe=gaussian_mixture_distribution(random_state=random_state + 2),
        ),
        TwoDimensionalCase(
            key="spiral",
            title="Spiral",
            dataframe=spiral_distribution(random_state=random_state + 3),
        ),
    ]


def generate_two_dimensional_sample(
    dataframe: pd.DataFrame,
    *,
    n_samples: int | None = None,
    random_state: int = 42,
    sampler_factory: Callable[..., object] = ConcreteDataFrameSampler,
    sampler_kwargs: dict | None = None,
) -> pd.DataFrame:
    kwargs = {**DEFAULT_TWO_DIMENSIONAL_SAMPLER_KWARGS, "random_state": random_state}
    kwargs.update(sampler_kwargs or {})
    sampler = sampler_factory(**kwargs)
    sampler.fit(dataframe)
    return sampler.sample(n_samples=n_samples or len(dataframe))


def run_two_dimensional_suite(
    *,
    random_state: int = 42,
    n_samples: int | None = None,
    sampler_kwargs: dict | None = None,
) -> list[dict[str, object]]:
    rows = []
    for offset, case in enumerate(make_two_dimensional_cases(random_state=random_state)):
        generated = generate_two_dimensional_sample(
            case.dataframe,
            n_samples=n_samples,
            random_state=random_state + 100 + offset,
            sampler_kwargs=sampler_kwargs,
        )
        rows.append({"case": case, "generated": generated})
    return rows


def plot_real_vs_generated_2d(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    title: str | None = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(real["x"], real["y"], s=12, alpha=0.45, color="#1f77b4", label="original")
    ax.scatter(generated["x"], generated["y"], s=12, alpha=0.45, color="#d62728", label="generated")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    return ax.figure


def plot_distribution_panel(
    dataframe: pd.DataFrame,
    *,
    title: str,
    color: str,
    label: str,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(dataframe["x"], dataframe["y"], s=12, alpha=0.55, color=color, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    return ax.figure


def plot_original_generated_triptych(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    title: str | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    prefix = f"{title}: " if title else ""
    plot_distribution_panel(
        real,
        title=f"{prefix}original",
        color="#1f77b4",
        label="original",
        ax=axes[0],
    )
    plot_distribution_panel(
        generated,
        title=f"{prefix}generated",
        color="#d62728",
        label="generated",
        ax=axes[1],
    )
    plot_real_vs_generated_2d(real, generated, title=f"{prefix}superimposed", ax=axes[2])
    fig.tight_layout()
    return fig


def plot_two_dimensional_suite(results: list[dict[str, object]], *, output_path: str | Path | None = None):
    nrows = len(results)
    fig, axes = plt.subplots(nrows, 3, figsize=(15.6, 4.6 * nrows))
    axes_array = np.asarray(axes).reshape(nrows, 3)
    for row_axes, result in zip(axes_array, results):
        case = result["case"]
        generated = result["generated"]
        plot_distribution_panel(
            case.dataframe,
            title=f"{case.title}: original",
            color="#1f77b4",
            label="original",
            ax=row_axes[0],
        )
        plot_distribution_panel(
            generated,
            title=f"{case.title}: generated",
            color="#d62728",
            label="generated",
            ax=row_axes[1],
        )
        plot_real_vs_generated_2d(
            case.dataframe,
            generated,
            title=f"{case.title}: superimposed",
            ax=row_axes[2],
        )
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def nearest_neighbor_distance_report(real: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    """Compare generated-to-real 1NN distances with natural real-to-real 1NN distances."""
    real_points = _xy_values(real)
    generated_points = _xy_values(generated)
    if len(real_points) < 2:
        raise ValueError("At least two original points are required for real-to-real 1NN distances.")

    generated_to_real = NearestNeighbors(n_neighbors=1).fit(real_points)
    generated_distances = generated_to_real.kneighbors(generated_points, return_distance=True)[0][:, 0]

    real_to_real = NearestNeighbors(n_neighbors=2).fit(real_points)
    real_distances = real_to_real.kneighbors(real_points, return_distance=True)[0][:, 1]

    return pd.DataFrame(
        [
            {"distance": float(distance), "kind": "generated_to_original_1nn"}
            for distance in generated_distances
        ]
        + [
            {"distance": float(distance), "kind": "original_to_original_1nn"}
            for distance in real_distances
        ]
    )


def plot_nearest_neighbor_distance_histogram(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    title: str | None = None,
    bins: int = 30,
    log: bool = True,
    ax=None,
):
    report = nearest_neighbor_distance_report(real, generated)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    generated_distances = report.loc[report["kind"] == "generated_to_original_1nn", "distance"]
    real_distances = report.loc[report["kind"] == "original_to_original_1nn", "distance"]
    max_distance = max(float(report["distance"].max()), 1e-12)
    bin_edges = np.linspace(0, max_distance, bins + 1)

    _plot_distance_histogram(
        ax,
        real_distances.to_numpy(),
        bin_edges=bin_edges,
        color="#1f77b4",
        label="original vs original 1NN",
        log=log,
    )
    _plot_distance_histogram(
        ax,
        generated_distances.to_numpy(),
        bin_edges=bin_edges,
        color="#d62728",
        label="generated vs original 1NN",
        log=log,
    )
    _plot_smoothed_distance_line(
        ax,
        real_distances.to_numpy(),
        color="#1f77b4",
        max_distance=max_distance,
        label="original vs original smoothed",
        log=log,
    )
    _plot_smoothed_distance_line(
        ax,
        generated_distances.to_numpy(),
        color="#d62728",
        max_distance=max_distance,
        label="generated vs original smoothed",
        log=log,
    )
    ax.set_xlabel("1NN distance")
    ax.set_ylabel("log1p(density)" if log else "density")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return ax.figure


def plot_nearest_neighbor_distance_suite(
    results: list[dict[str, object]],
    *,
    bins: int = 30,
    log: bool = True,
    output_path: str | Path | None = None,
):
    ncols = 2
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 4.4 * nrows))
    axes_list = list(np.ravel(axes))
    for ax, result in zip(axes_list, results):
        case = result["case"]
        generated = result["generated"]
        plot_nearest_neighbor_distance_histogram(
            case.dataframe,
            generated,
            title=case.title,
            bins=bins,
            log=log,
            ax=ax,
        )
    for ax in axes_list[len(results) :]:
        ax.axis("off")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def nearest_neighbor_distance_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        case = result["case"]
        report = nearest_neighbor_distance_report(case.dataframe, result["generated"])
        for kind, group in report.groupby("kind"):
            rows.append(
                {
                    "case": case.key,
                    "kind": kind,
                    "mean": group["distance"].mean(),
                    "median": group["distance"].median(),
                    "p05": group["distance"].quantile(0.05),
                    "p95": group["distance"].quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def two_dimensional_summary(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        case = result["case"]
        real = case.dataframe
        generated = result["generated"]
        rows.append(
            {
                "case": case.key,
                "rows": len(real),
                "generated_rows": len(generated),
                "real_x_mean": real["x"].mean(),
                "generated_x_mean": generated["x"].mean(),
                "real_y_mean": real["y"].mean(),
                "generated_y_mean": generated["y"].mean(),
                "real_x_std": real["x"].std(),
                "generated_x_std": generated["x"].std(),
                "real_y_std": real["y"].std(),
                "generated_y_std": generated["y"].std(),
            }
        )
    return pd.DataFrame(rows)


def _to_dataframe(points: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["x", "y"])


def _xy_values(dataframe: pd.DataFrame) -> np.ndarray:
    missing = [column for column in ("x", "y") if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Expected x and y columns; missing {missing}.")
    return dataframe.loc[:, ["x", "y"]].to_numpy(dtype=float)


def _plot_distance_histogram(
    ax,
    distances: np.ndarray,
    *,
    bin_edges: np.ndarray,
    color: str,
    label: str,
    log: bool,
):
    density, edges = np.histogram(distances, bins=bin_edges, density=True)
    if log:
        density = np.log1p(density)
    ax.bar(
        edges[:-1],
        density,
        width=np.diff(edges),
        align="edge",
        alpha=0.45,
        color=color,
        label=label,
    )


def _plot_smoothed_distance_line(
    ax,
    distances: np.ndarray,
    *,
    color: str,
    max_distance: float,
    label: str,
    linewidth: float = 3,
    log: bool = False,
):
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return

    x_grid = np.linspace(0, max_distance, 256)
    if len(np.unique(distances)) > 1:
        density = gaussian_kde(distances)(x_grid)
    else:
        counts, edges = np.histogram(distances, bins=min(10, max(1, len(distances))), range=(0, max_distance), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        density = np.interp(x_grid, centers, counts, left=0, right=0)
    if log:
        density = np.log1p(density)
    ax.plot(x_grid, density, color=color, linewidth=linewidth, label=label)
