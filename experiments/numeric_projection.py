from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dataframe_sampler import DataFrameSampler

from .datasets import DatasetExperimentConfig
from .workflow import sampler_config_with_random_state


def numeric_view(dataframe, sampler) -> pd.DataFrame:
    """Return the sampler's latent numeric representation for dataframe rows."""
    numeric = sampler.transform(dataframe)
    columns = [f"latent_{idx}" for idx in range(numeric.shape[1])]
    return pd.DataFrame(numeric, columns=columns, index=dataframe.index)


def numeric_view_from_config(dataframe: pd.DataFrame, config: DatasetExperimentConfig) -> pd.DataFrame:
    """Fit the configured sampler and return its numeric representation."""
    sampler = DataFrameSampler(
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
    target_column: str | None = None,
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
    original_style = _target_marker_style(original, target_column=target_column)
    generated_style = _target_marker_style(generated, target_column=target_column)
    _scatter_projection(
        axes[0],
        original_projection,
        color="#1f77b4",
        label="original",
        title=f"{prefix}original",
        style=original_style,
    )
    _scatter_projection(
        axes[1],
        generated_projection,
        color="#d62728",
        label="generated",
        title=f"{prefix}generated",
        style=generated_style,
    )
    _scatter_projection(
        axes[2],
        original_projection,
        color="#1f77b4",
        label="original",
        title=f"{prefix}superimposed",
        style=original_style,
    )
    _scatter_projection(
        axes[2],
        generated_projection,
        color="#d62728",
        label="generated",
        title=f"{prefix}superimposed",
        style=generated_style,
    )
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


def _scatter_projection(ax, projection: pd.DataFrame, *, color: str, label: str, title: str, style: dict):
    if style["task"] == "classification":
        for marker_label, marker in style["markers"].items():
            mask = style["labels"] == marker_label
            if mask.any():
                ax.scatter(
                    projection.loc[mask, "dim1"],
                    projection.loc[mask, "dim2"],
                    s=18,
                    alpha=0.52,
                    color=color,
                    marker=marker,
                    linewidths=0.8 if marker == "x" else 0,
                    label=f"{label} {marker_label}",
                )
    else:
        ax.scatter(
            projection["dim1"],
            projection["dim2"],
            s=style["sizes"],
            alpha=0.45,
            color=color,
            marker="o",
            linewidths=0,
            label=label,
        )
    ax.set_title(title)


def _target_marker_style(dataframe: pd.DataFrame, *, target_column: str | None) -> dict:
    if target_column is None or target_column not in dataframe:
        return {"task": "none", "sizes": 12}

    target = dataframe[target_column]
    clean = target.dropna()
    unique_values = list(pd.unique(clean))
    if len(unique_values) == 2:
        ordered = sorted(unique_values, key=lambda value: str(value))
        return {
            "task": "classification",
            "labels": target.map({ordered[0]: str(ordered[0]), ordered[1]: str(ordered[1])}).fillna("missing"),
            "markers": {str(ordered[0]): "o", str(ordered[1]): "x", "missing": "o"},
        }
    if pd.api.types.is_numeric_dtype(target):
        return {"task": "regression", "sizes": _regression_marker_sizes(target)}
    return {"task": "none", "sizes": 12}


def _regression_marker_sizes(target: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(12.0, index=target.index)
    fill_value = numeric.median()
    numeric = numeric.fillna(fill_value)
    minimum = numeric.min()
    maximum = numeric.max()
    if minimum == maximum:
        return pd.Series(18.0, index=target.index)
    scaled = (numeric - minimum) / (maximum - minimum)
    return 10.0 + 34.0 * scaled
