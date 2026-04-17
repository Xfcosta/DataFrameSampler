from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from dataframe_sampler import DataFrameSampler

from .datasets import DatasetExperimentConfig


MANIFOLD_VALIDATION_COLUMNS = [
    "dataset",
    "method",
    "sample_type",
    "point_index",
    "stress",
    "raw_stress",
    "in_hull",
    "out_hull",
    "train_rows",
    "latent_dim",
    "isomap_neighbors",
    "insertion_neighbors",
    "max_eval_points",
]


def run_manifold_validation_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    max_eval_points: int = 250,
    max_train_rows: int = 800,
    test_size: float = 0.3,
    isomap_neighbors: int = 10,
    insertion_neighbors: int | None = None,
) -> pd.DataFrame:
    """Run frozen-Isomap insertion-stress validation for one configured dataset."""
    if len(dataframe) < 6:
        return _empty_validation_frame(config.dataset_name)

    train, real_test = _train_test_dataframe_split(
        dataframe,
        target_column=config.target_column,
        test_size=test_size,
        random_state=config.random_state,
    )
    if len(train) < 3 or real_test.empty:
        return _empty_validation_frame(config.dataset_name)
    train = deterministic_dataframe_sample(
        train,
        max_rows=max_train_rows,
        random_state=config.random_state,
    )

    sampler_kwargs = dict(sampler_config or config.manual_sampler_config)
    sampler_kwargs.setdefault("random_state", config.random_state)
    sampler = DataFrameSampler(**sampler_kwargs).fit(train)

    train_latent = sampler.transform(train)
    real_test_latent = sampler.transform(real_test)
    real_test_latent = deterministic_row_sample(
        real_test_latent,
        max_rows=max_eval_points,
        random_state=config.random_state,
    )

    n_generated = min(max_eval_points, max(1, config.n_generated))
    generated = sampler.generate(n_samples=n_generated)
    generated_latent = sampler.transform(generated)
    interpolation_latent = latent_interpolation_baseline(
        train_latent,
        n_samples=n_generated,
        n_neighbors=isomap_neighbors,
        random_state=config.random_state,
    )
    bootstrap_latent = latent_bootstrap_baseline(
        train_latent,
        n_samples=n_generated,
        random_state=config.random_state,
    )

    pointwise = manifold_validation_report(
        train_latent=train_latent,
        real_test_latent=real_test_latent,
        generated_latents={
            "dataframe_sampler_manual": generated_latent,
            "latent_interpolation": interpolation_latent,
            "latent_bootstrap": bootstrap_latent,
        },
        dataset_name=config.dataset_name,
        max_eval_points=max_eval_points,
        random_state=config.random_state,
        isomap_neighbors=isomap_neighbors,
        insertion_neighbors=insertion_neighbors,
    )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    pointwise.to_csv(results_path / f"{config.dataset_name}_manifold_validation.csv", index=False)
    return pointwise


def manifold_validation_report(
    *,
    train_latent: np.ndarray,
    real_test_latent: np.ndarray,
    generated_latents: Mapping[str, np.ndarray],
    dataset_name: str,
    max_eval_points: int = 250,
    random_state: int = 42,
    isomap_neighbors: int = 10,
    insertion_neighbors: int | None = None,
) -> pd.DataFrame:
    train_latent = _validate_matrix(train_latent, "train_latent")
    if train_latent.shape[0] < 3:
        return _empty_validation_frame(dataset_name)

    real_test_latent = deterministic_row_sample(
        _validate_matrix(real_test_latent, "real_test_latent"),
        max_rows=max_eval_points,
        random_state=random_state,
    )
    geometry = FrozenIsomapGeometry.fit(
        train_latent,
        n_neighbors=isomap_neighbors,
        insertion_neighbors=insertion_neighbors,
    )

    rows = []
    if len(real_test_latent):
        rows.extend(
            _point_rows(
                dataset_name=dataset_name,
                method="held_out_real",
                sample_type="real_test",
                points=real_test_latent,
                geometry=geometry,
                max_eval_points=max_eval_points,
            )
        )
    for method, values in generated_latents.items():
        points = deterministic_row_sample(
            _validate_matrix(values, method),
            max_rows=max_eval_points,
            random_state=random_state,
        )
        rows.extend(
            _point_rows(
                dataset_name=dataset_name,
                method=method,
                sample_type="generated",
                points=points,
                geometry=geometry,
                max_eval_points=max_eval_points,
            )
        )
    return pd.DataFrame(rows, columns=MANIFOLD_VALIDATION_COLUMNS)


class FrozenIsomapGeometry:
    def __init__(
        self,
        *,
        train_latent: np.ndarray,
        train_embedding: np.ndarray,
        geodesic_distances: np.ndarray,
        isomap: Isomap,
        n_neighbors: int,
        insertion_neighbors: int,
    ):
        self.train_latent = train_latent
        self.train_embedding = train_embedding
        self.geodesic_distances = geodesic_distances
        self.isomap = isomap
        self.n_neighbors = n_neighbors
        self.insertion_neighbors = insertion_neighbors
        self.insertion_neighbours_ = NearestNeighbors(n_neighbors=insertion_neighbors).fit(train_latent)

    @classmethod
    def fit(
        cls,
        train_latent: np.ndarray,
        *,
        n_neighbors: int = 10,
        insertion_neighbors: int | None = None,
    ) -> "FrozenIsomapGeometry":
        train_latent = _validate_matrix(train_latent, "train_latent")
        if train_latent.shape[0] < 3:
            raise ValueError("At least three training rows are required for Isomap validation.")
        neighbour_count = min(max(1, int(n_neighbors)), train_latent.shape[0] - 1)
        insertion_count = min(
            max(1, int(insertion_neighbors or neighbour_count)),
            train_latent.shape[0],
        )
        embedding_dim = min(2, train_latent.shape[0] - 1, max(1, train_latent.shape[1]))
        isomap = Isomap(n_neighbors=neighbour_count, n_components=embedding_dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            train_embedding = isomap.fit_transform(train_latent)
        if train_embedding.ndim == 1:
            train_embedding = train_embedding.reshape(-1, 1)
        graph = _symmetric_neighbour_graph(train_latent, neighbour_count)
        geodesic_distances = shortest_path(graph, directed=False, unweighted=False)
        return cls(
            train_latent=train_latent,
            train_embedding=np.asarray(train_embedding, dtype=float),
            geodesic_distances=np.asarray(geodesic_distances, dtype=float),
            isomap=isomap,
            n_neighbors=neighbour_count,
            insertion_neighbors=insertion_count,
        )

    def insertion_stress(self, points: np.ndarray) -> pd.DataFrame:
        points = _validate_matrix(points, "points")
        if len(points) == 0:
            return pd.DataFrame(columns=["stress", "raw_stress"])
        distances, indices = self.insertion_neighbours_.kneighbors(points)
        embedding = self.isomap.transform(points)
        if embedding.ndim == 1:
            embedding = embedding.reshape(-1, 1)

        rows = []
        for row_idx in range(len(points)):
            geodesic = np.min(
                distances[row_idx, :, None] + self.geodesic_distances[indices[row_idx]],
                axis=0,
            )
            embedded = np.linalg.norm(self.train_embedding - embedding[row_idx], axis=1)
            finite = np.isfinite(geodesic) & np.isfinite(embedded)
            if not finite.any():
                rows.append({"stress": np.nan, "raw_stress": np.nan})
                continue
            residual = geodesic[finite] - embedded[finite]
            raw = float(np.sqrt(np.mean(residual**2)))
            denom = float(np.sum(embedded[finite] ** 2))
            normalized = float(np.sqrt(np.sum(residual**2) / denom)) if denom > 0 else raw
            rows.append({"stress": normalized, "raw_stress": raw})
        return pd.DataFrame(rows)

    def in_convex_hull(self, points: np.ndarray) -> np.ndarray:
        return convex_hull_membership(self.train_latent, points)


def convex_hull_membership(
    train_points: np.ndarray,
    query_points: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return whether each query point is a convex combination of train points."""
    train_points = _validate_matrix(train_points, "train_points")
    query_points = _validate_matrix(query_points, "query_points")
    if train_points.shape[1] != query_points.shape[1]:
        raise ValueError("train_points and query_points must have the same width.")
    train_reduced, query_reduced = affine_rank_reduction(train_points, query_points, tolerance=tolerance)
    if train_reduced.shape[1] == 0:
        center = train_points.mean(axis=0)
        return np.linalg.norm(query_points - center, axis=1) <= tolerance

    n_train = train_reduced.shape[0]
    a_eq = np.vstack([train_reduced.T, np.ones(n_train)])
    c = np.zeros(n_train, dtype=float)
    bounds = [(0.0, None)] * n_train
    inside = []
    for point in query_reduced:
        b_eq = np.concatenate([point, [1.0]])
        result = linprog(
            c,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        inside.append(bool(result.success))
    return np.asarray(inside, dtype=bool)


def affine_rank_reduction(
    train_points: np.ndarray,
    query_points: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    train_points = _validate_matrix(train_points, "train_points")
    query_points = _validate_matrix(query_points, "query_points")
    center = train_points.mean(axis=0)
    shifted_train = train_points - center
    shifted_query = query_points - center
    _, singular_values, vt = np.linalg.svd(shifted_train, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((len(train_points), 0)), np.empty((len(query_points), 0))
    rank_tolerance = max(shifted_train.shape) * np.finfo(float).eps * singular_values[0]
    rank_tolerance = max(rank_tolerance, tolerance)
    rank = int(np.sum(singular_values > rank_tolerance))
    if rank == 0:
        return np.empty((len(train_points), 0)), np.empty((len(query_points), 0))
    components = vt[:rank]
    return shifted_train @ components.T, shifted_query @ components.T


def latent_interpolation_baseline(
    train_latent: np.ndarray,
    *,
    n_samples: int,
    n_neighbors: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Generate SMOTE-style latent interpolation points inside training line segments."""
    train_latent = _validate_matrix(train_latent, "train_latent")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if n_samples == 0:
        return np.empty((0, train_latent.shape[1]), dtype=float)
    if len(train_latent) == 1:
        return np.repeat(train_latent, n_samples, axis=0)

    rng = np.random.default_rng(random_state)
    neighbour_count = min(max(1, int(n_neighbors)), len(train_latent) - 1)
    neighbours = NearestNeighbors(n_neighbors=neighbour_count + 1).fit(train_latent)
    _, indices = neighbours.kneighbors(train_latent)

    samples = []
    for _ in range(n_samples):
        anchor = int(rng.integers(0, len(train_latent)))
        candidate_indices = indices[anchor][indices[anchor] != anchor]
        if len(candidate_indices) == 0:
            neighbour = int(rng.integers(0, len(train_latent)))
        else:
            neighbour = int(rng.choice(candidate_indices))
        weight = float(rng.random())
        samples.append(train_latent[anchor] + weight * (train_latent[neighbour] - train_latent[anchor]))
    return np.asarray(samples, dtype=float)


def latent_bootstrap_baseline(
    train_latent: np.ndarray,
    *,
    n_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    """Sample fitted latent rows with replacement for transport validation."""
    train_latent = _validate_matrix(train_latent, "train_latent")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if n_samples == 0:
        return np.empty((0, train_latent.shape[1]), dtype=float)
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(train_latent), size=n_samples)
    return train_latent[indices].copy()


def summarize_manifold_validation(pointwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pointwise.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "out_hull_rate",
                "real_stress_median",
                "real_stress_q95",
                "generated_stress_median",
                "out_hull_stress_median",
                "out_hull_acceptance_at_real_q95",
            ]
        )
    for dataset, dataset_frame in pointwise.groupby("dataset", dropna=False):
        real = dataset_frame[dataset_frame["sample_type"] == "real_test"]["stress"].dropna()
        real_median = float(real.median()) if len(real) else np.nan
        real_q95 = float(real.quantile(0.95)) if len(real) else np.nan
        generated_frame = dataset_frame[dataset_frame["sample_type"] == "generated"]
        for method, method_frame in generated_frame.groupby("method", dropna=False):
            stress = method_frame["stress"].dropna()
            out_hull = method_frame[method_frame["out_hull"]]
            out_stress = out_hull["stress"].dropna()
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "out_hull_rate": float(method_frame["out_hull"].mean()) if len(method_frame) else np.nan,
                    "real_stress_median": real_median,
                    "real_stress_q95": real_q95,
                    "generated_stress_median": float(stress.median()) if len(stress) else np.nan,
                    "out_hull_stress_median": float(out_stress.median()) if len(out_stress) else np.nan,
                    "out_hull_acceptance_at_real_q95": (
                        float((out_stress <= real_q95).mean())
                        if len(out_stress) and np.isfinite(real_q95)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def deterministic_row_sample(values: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    values = _validate_matrix(values, "values")
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")
    if len(values) <= max_rows:
        return values.copy()
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(values), size=max_rows, replace=False))
    return values[indices].copy()


def deterministic_dataframe_sample(dataframe: pd.DataFrame, *, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")
    if len(dataframe) <= max_rows:
        return dataframe.reset_index(drop=True).copy()
    return dataframe.sample(n=max_rows, random_state=random_state).sort_index().reset_index(drop=True)


def _point_rows(
    *,
    dataset_name: str,
    method: str,
    sample_type: str,
    points: np.ndarray,
    geometry: FrozenIsomapGeometry,
    max_eval_points: int,
) -> list[dict[str, Any]]:
    stress = geometry.insertion_stress(points)
    in_hull = geometry.in_convex_hull(points)
    rows = []
    for idx, inside in enumerate(in_hull):
        rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "sample_type": sample_type,
                "point_index": idx,
                "stress": stress.loc[idx, "stress"],
                "raw_stress": stress.loc[idx, "raw_stress"],
                "in_hull": bool(inside),
                "out_hull": bool(not inside),
                "train_rows": len(geometry.train_latent),
                "latent_dim": geometry.train_latent.shape[1],
                "isomap_neighbors": geometry.n_neighbors,
                "insertion_neighbors": geometry.insertion_neighbors,
                "max_eval_points": max_eval_points,
            }
        )
    return rows


def _train_test_dataframe_split(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if target_column is not None and target_column in dataframe:
        target = dataframe[target_column]
        counts = target.value_counts(dropna=True)
        if len(counts) > 1 and counts.min() >= 2:
            stratify = target
    try:
        train, test = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train, test = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _symmetric_neighbour_graph(values: np.ndarray, n_neighbors: int) -> csr_matrix:
    values = _validate_matrix(values, "values")
    neighbours = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(values))).fit(values)
    distances, indices = neighbours.kneighbors(values)
    row_indices = []
    col_indices = []
    data = []
    for row, (row_distances, row_indices_values) in enumerate(zip(distances, indices)):
        added = 0
        for distance, col in zip(row_distances, row_indices_values):
            if col == row:
                continue
            row_indices.append(row)
            col_indices.append(int(col))
            data.append(float(distance))
            added += 1
            if added >= n_neighbors:
                break
    graph = csr_matrix((data, (row_indices, col_indices)), shape=(len(values), len(values)))
    return graph.maximum(graph.T)


def _validate_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D numeric array.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _empty_validation_frame(dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=MANIFOLD_VALIDATION_COLUMNS)
    if dataset_name:
        frame["dataset"] = frame.get("dataset", pd.Series(dtype=object))
    return frame
