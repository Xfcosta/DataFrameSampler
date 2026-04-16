from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dataframe_sampler import ConcreteDataFrameSampler

from .datasets import DatasetExperimentConfig
from .workflow import sampler_config_with_random_state


def numeric_view(dataframe, sampler) -> pd.DataFrame:
    """Return the sampler's pure numeric representation for dataframe rows."""
    numeric = sampler.transform(dataframe)
    return numeric.apply(pd.to_numeric, errors="coerce")


def numeric_view_from_config(dataframe: pd.DataFrame, config: DatasetExperimentConfig) -> pd.DataFrame:
    """Fit the configured sampler and return its numeric representation."""
    sampler = ConcreteDataFrameSampler(
        **sampler_config_with_random_state(config.manual_sampler_config, config.random_state)
    )
    sampler.fit(dataframe)
    return numeric_view(dataframe, sampler)


def project_numeric_views(
    original: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    reducer: str = "umap",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Project original and generated numeric views to two shared dimensions."""
    original = original.reset_index(drop=True)
    generated = generated.reset_index(drop=True)
    combined = pd.concat([original, generated], ignore_index=True)
    values = SimpleImputer(strategy="median").fit_transform(combined)
    values = StandardScaler().fit_transform(values)
    projection, reducer_name = _fit_project(values, reducer=reducer, random_state=random_state)
    original_projection = _projection_frame(projection[: len(original)])
    generated_projection = _projection_frame(projection[len(original) :])
    return original_projection, generated_projection, reducer_name


def plot_numeric_projection_triptych(
    sampler,
    original: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    title: str | None = None,
    reducer: str = "umap",
    random_state: int = 42,
    output_path: str | Path | None = None,
):
    original = original.reset_index(drop=True)
    generated = generated.reset_index(drop=True)
    combined = pd.concat([original, generated], ignore_index=True)
    combined_numeric = numeric_view(combined, sampler)
    original_numeric = combined_numeric.iloc[: len(original)].reset_index(drop=True)
    generated_numeric = combined_numeric.iloc[len(original) :].reset_index(drop=True)
    original_projection, generated_projection, reducer_name = project_numeric_views(
        original_numeric,
        generated_numeric,
        reducer=reducer,
        random_state=random_state,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    prefix = f"{title}: " if title else ""
    _scatter_projection(axes[0], original_projection, color="#1f77b4", label="original", title=f"{prefix}original")
    _scatter_projection(axes[1], generated_projection, color="#d62728", label="generated", title=f"{prefix}generated")
    _scatter_projection(axes[2], original_projection, color="#1f77b4", label="original", title=f"{prefix}superimposed")
    _scatter_projection(axes[2], generated_projection, color="#d62728", label="generated", title=f"{prefix}superimposed")
    axes[0].set_ylabel(f"{reducer_name} 2")
    for ax in axes:
        ax.set_xlabel(f"{reducer_name} 1")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _fit_project(values: np.ndarray, *, reducer: str, random_state: int) -> tuple[np.ndarray, str]:
    reducer_key = reducer.lower()
    if reducer_key == "umap":
        try:
            from umap import UMAP

            return UMAP(n_components=2, random_state=random_state).fit_transform(values), "UMAP"
        except ImportError:
            pass
    if reducer_key not in {"umap", "pca"}:
        raise ValueError("reducer must be 'umap' or 'pca'.")
    return PCA(n_components=2, random_state=random_state).fit_transform(values), "PCA"


def _projection_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=["dim1", "dim2"])


def _scatter_projection(ax, projection: pd.DataFrame, *, color: str, label: str, title: str):
    ax.scatter(projection["dim1"], projection["dim2"], s=12, alpha=0.45, color=color, label=label)
    ax.set_title(title)
