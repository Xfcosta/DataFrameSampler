from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dataframe_sampler import DataFrameSampler

from .baselines import SdvCtganBaseline, simple_baselines
from .datasets import DatasetExperimentConfig
from .workflow import sampler_config_with_random_state


NUMERIC_PROJECTION_METHOD_LABELS = {
    "dataframe_sampler": "DataFrameSampler",
    "row_bootstrap": "Row bootstrap",
    "independent_columns": "Independent columns",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified columns",
    "ctgan": "CTGAN",
}


def numeric_view(dataframe, sampler) -> pd.DataFrame:
    """Return the sampler's latent numeric representation for dataframe rows."""
    numeric = sampler.transform(dataframe)
    columns = [f"latent_{idx}" for idx in range(numeric.shape[1])]
    return pd.DataFrame(numeric, columns=columns, index=dataframe.index)


def numeric_view_from_config(dataframe: pd.DataFrame, config: DatasetExperimentConfig) -> pd.DataFrame:
    """Fit the configured sampler and return its numeric representation."""
    sampler = DataFrameSampler(
        **sampler_config_with_random_state(config.sampler_config, config.random_state)
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


def plot_numeric_projection_triptychs_for_methods(
    dataframe: pd.DataFrame,
    config: DatasetExperimentConfig,
    *,
    n_samples: int | None = None,
    methods: list[str] | None = None,
    include_ctgan: bool = False,
    reducer: str = "pca",
    random_state: int | None = None,
    figures_dir: str | Path | None = None,
) -> tuple[dict[str, plt.Figure], pd.DataFrame]:
    """Plot original/generated triptychs for DataFrameSampler and baselines.

    A single fitted DataFrameSampler defines the numeric view for all methods.
    Baselines generate mixed-type rows, and those rows are projected through the
    same fitted sampler before plotting so the triptychs are visually comparable.
    """
    if dataframe.empty:
        raise ValueError("dataframe must contain at least one row.")
    random_state = config.random_state if random_state is None else random_state
    n_samples = n_samples or min(config.n_generated, len(dataframe))
    selected_methods = methods or _default_numeric_projection_methods(
        target_column=config.target_column,
        include_ctgan=include_ctgan,
    )
    projection_sampler = DataFrameSampler(
        **sampler_config_with_random_state(config.sampler_config, random_state)
    ).fit(dataframe)
    generators = _numeric_projection_generators(
        selected_methods,
        target_column=config.target_column,
        random_state=random_state,
    )
    figures: dict[str, plt.Figure] = {}
    rows = []
    for method_name, generator in generators.items():
        try:
            if method_name == "dataframe_sampler":
                generated = projection_sampler.generate(n_samples=n_samples)
            else:
                generated = generator.fit(dataframe, target_column=config.target_column).sample(n_samples)
            label = NUMERIC_PROJECTION_METHOD_LABELS.get(method_name, method_name)
            output_path = None
            if figures_dir is not None:
                output_path = Path(figures_dir) / f"{config.dataset_name}_{method_name}_numeric_projection.pdf"
            figures[method_name] = plot_numeric_projection_triptych(
                projection_sampler,
                dataframe,
                generated,
                target_column=config.target_column,
                title=f"{config.title} - {label}",
                reducer=reducer,
                random_state=random_state,
                output_path=output_path,
            )
            rows.append(
                {
                    "method": method_name,
                    "method_label": label,
                    "n_rows": len(generated),
                    "status": "ok",
                    "reason": "",
                }
            )
        except Exception as exc:  # pragma: no cover - notebook diagnostic path
            rows.append(
                {
                    "method": method_name,
                    "method_label": NUMERIC_PROJECTION_METHOD_LABELS.get(method_name, method_name),
                    "n_rows": 0,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
    return figures, pd.DataFrame(rows)


def _default_numeric_projection_methods(*, target_column: str | None, include_ctgan: bool) -> list[str]:
    methods = ["dataframe_sampler", "row_bootstrap", "independent_columns", "gaussian_copula_empirical"]
    if target_column is not None:
        methods.append("stratified_columns")
    if include_ctgan:
        methods.append("ctgan")
    return methods


def _numeric_projection_generators(
    methods: list[str],
    *,
    target_column: str | None,
    random_state: int,
) -> dict[str, object]:
    available = {spec.name: spec.estimator for spec in simple_baselines(target_column, random_state=random_state)}
    available["dataframe_sampler"] = None
    if "ctgan" in methods:
        available["ctgan"] = SdvCtganBaseline(random_state=random_state)
    missing = [method for method in methods if method not in available]
    if missing:
        raise ValueError(f"Unknown numeric projection method(s): {missing}")
    return {method: available[method] for method in methods}


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
        for marker_label, marker_style in style["markers"].items():
            mask = style["labels"] == marker_label
            if mask.any():
                filled = marker_style["filled"]
                ax.scatter(
                    projection.loc[mask, "dim1"],
                    projection.loc[mask, "dim2"],
                    s=18,
                    alpha=0.52,
                    facecolors=color if filled else "white",
                    edgecolors=color,
                    marker="o",
                    linewidths=0.8 if not filled else 0,
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
        negative, positive = _negative_positive_values(unique_values)
        return {
            "task": "classification",
            "labels": target.map({negative: str(negative), positive: str(positive)}).fillna("missing"),
            "markers": {
                str(negative): {"filled": False},
                str(positive): {"filled": True},
                "missing": {"filled": False},
            },
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


def _negative_positive_values(values: list) -> tuple:
    if all(isinstance(value, (bool, np.bool_)) for value in values):
        return False, True
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        order = np.argsort(numeric.to_numpy(dtype=float))
        return values[int(order[0])], values[int(order[-1])]

    positive_tokens = ("1", "true", "yes", "y", "positive", "pos", "case", "malignant")
    negative_tokens = ("0", "false", "no", "n", "negative", "neg", "control", "benign")
    by_token = {str(value).strip().lower(): value for value in values}
    positive_matches = [by_token[token] for token in positive_tokens if token in by_token]
    negative_matches = [by_token[token] for token in negative_tokens if token in by_token]
    if positive_matches and negative_matches:
        return negative_matches[0], positive_matches[0]

    ordered = sorted(values, key=lambda value: str(value))
    return ordered[0], ordered[1]
