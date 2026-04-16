from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_similarity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    bins: int = 20,
) -> pd.DataFrame:
    """Compute numeric distributional similarity metrics column by column."""
    selected = list(columns) if columns is not None else _common_numeric_columns(real, synthetic)
    rows = []
    for column in selected:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna().to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna().to_numpy(dtype=float)
        row = {
            "column": column,
            "real_count": len(real_values),
            "synthetic_count": len(synthetic_values),
            "real_mean": _safe_mean(real_values),
            "synthetic_mean": _safe_mean(synthetic_values),
            "mean_abs_error": _safe_abs_delta(_safe_mean(real_values), _safe_mean(synthetic_values)),
            "real_std": _safe_std(real_values),
            "synthetic_std": _safe_std(synthetic_values),
            "std_abs_error": _safe_abs_delta(_safe_std(real_values), _safe_std(synthetic_values)),
            "ks_statistic": _safe_ks(real_values, synthetic_values),
            "wasserstein_distance": _safe_wasserstein(real_values, synthetic_values),
            "histogram_overlap": histogram_overlap(real_values, synthetic_values, bins=bins),
            "real_missing_rate": real[column].isna().mean(),
            "synthetic_missing_rate": synthetic[column].isna().mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_similarity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    rare_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute categorical distributional similarity metrics column by column."""
    selected = list(columns) if columns is not None else _common_categorical_columns(real, synthetic)
    rows = []
    for column in selected:
        real_series = _normalized_category_series(real[column])
        synthetic_series = _normalized_category_series(synthetic[column])
        real_probs, synthetic_probs = _aligned_probabilities(real_series, synthetic_series)
        real_categories = set(real_probs.index[real_probs > 0])
        synthetic_categories = set(synthetic_probs.index[synthetic_probs > 0])
        rare_categories = set(real_probs.index[(real_probs > 0) & (real_probs <= rare_threshold)])
        rows.append(
            {
                "column": column,
                "real_unique": len(real_categories),
                "synthetic_unique": len(synthetic_categories),
                "category_coverage": _safe_ratio(len(real_categories & synthetic_categories), len(real_categories)),
                "rare_category_count": len(rare_categories),
                "rare_category_preservation": _safe_ratio(
                    len(rare_categories & synthetic_categories),
                    len(rare_categories),
                ),
                "total_variation_distance": float(0.5 * np.abs(real_probs - synthetic_probs).sum()),
                "jensen_shannon_divergence": float(jensenshannon(real_probs, synthetic_probs, base=2.0) ** 2),
                "real_missing_rate": real[column].isna().mean(),
                "synthetic_missing_rate": synthetic[column].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def dependence_similarity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: str = "pearson",
) -> dict[str, Any]:
    """Compare mixed-type association matrices between real and synthetic data."""
    selected = list(columns) if columns is not None else [c for c in real.columns if c in synthetic.columns]
    real_matrix = mixed_association_matrix(real[selected], method=method)
    synthetic_matrix = mixed_association_matrix(synthetic[selected], method=method)
    diff = (real_matrix - synthetic_matrix).abs()
    upper = _upper_triangle_values(diff.to_numpy(dtype=float))
    return {
        "real_association": real_matrix,
        "synthetic_association": synthetic_matrix,
        "absolute_difference": diff,
        "mean_abs_association_difference": float(np.nanmean(upper)) if upper.size else 0.0,
        "max_abs_association_difference": float(np.nanmax(upper)) if upper.size else 0.0,
    }


def mixed_association_matrix(dataframe: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    """Build a mixed-type association matrix.

    Numeric/numeric pairs use Pearson or Spearman correlation. Categorical pairs
    use Cramer's V. Mixed numeric/categorical pairs use the correlation ratio.
    """
    columns = list(dataframe.columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = mixed_association(dataframe[left], dataframe[right], method=method)
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def mixed_association(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    left_is_numeric = pd.api.types.is_numeric_dtype(left)
    right_is_numeric = pd.api.types.is_numeric_dtype(right)
    if left_is_numeric and right_is_numeric:
        return _numeric_correlation(left, right, method=method)
    if not left_is_numeric and not right_is_numeric:
        return cramers_v(left, right)
    if left_is_numeric:
        return correlation_ratio(categories=right, measurements=left)
    return correlation_ratio(categories=left, measurements=right)


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(_normalized_category_series(left), _normalized_category_series(right))
    if table.empty:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    rows, cols = table.shape
    denom = min(cols - 1, rows - 1)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(phi2 / denom))


def theils_u(x: pd.Series, y: pd.Series) -> float:
    """Compute Theil's U, the uncertainty coefficient U(X|Y)."""
    x_values = _normalized_category_series(x)
    y_values = _normalized_category_series(y)
    entropy_x = _entropy(x_values)
    if entropy_x == 0:
        return 1.0
    conditional = 0.0
    for y_value, group in x_values.groupby(y_values):
        weight = len(group) / len(x_values)
        conditional += weight * _entropy(group)
    return float((entropy_x - conditional) / entropy_x)


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    clean = pd.DataFrame({"category": categories, "measurement": pd.to_numeric(measurements, errors="coerce")})
    clean = clean.dropna()
    if clean.empty:
        return np.nan
    grand_mean = clean["measurement"].mean()
    between = 0.0
    total = ((clean["measurement"] - grand_mean) ** 2).sum()
    if total == 0:
        return 0.0
    for _, group in clean.groupby("category", dropna=True):
        between += len(group) * (group["measurement"].mean() - grand_mean) ** 2
    return float(math.sqrt(between / total))


def histogram_overlap(real_values: np.ndarray, synthetic_values: np.ndarray, *, bins: int = 20) -> float:
    real_values = np.asarray(real_values, dtype=float)
    synthetic_values = np.asarray(synthetic_values, dtype=float)
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    combined = np.concatenate([real_values, synthetic_values])
    if np.nanmin(combined) == np.nanmax(combined):
        return 1.0
    real_hist, edges = np.histogram(real_values, bins=bins, range=(np.nanmin(combined), np.nanmax(combined)))
    synthetic_hist, _ = np.histogram(synthetic_values, bins=edges)
    real_probs = real_hist / max(real_hist.sum(), 1)
    synthetic_probs = synthetic_hist / max(synthetic_hist.sum(), 1)
    return float(np.minimum(real_probs, synthetic_probs).sum())


def nearest_neighbor_distance_test(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> dict[str, float]:
    """Compare synthetic-to-real nearest distances with natural real-real distances."""
    feature_real, feature_synthetic = _aligned_features(real, synthetic, target_column=target_column)
    if len(feature_real) < 2 or len(feature_synthetic) == 0:
        return {
            "nn_synthetic_to_real_mean": np.nan,
            "nn_real_to_real_mean": np.nan,
            "nn_distance_ratio": np.nan,
            "nn_suspiciously_close_rate": np.nan,
        }

    transformed = _fit_transform_features(feature_real, feature_synthetic)
    real_matrix = transformed[: len(feature_real)]
    synthetic_matrix = transformed[len(feature_real) :]

    real_neighbours = NearestNeighbors(n_neighbors=2).fit(real_matrix)
    real_distances, _ = real_neighbours.kneighbors(real_matrix)
    natural_distances = real_distances[:, 1]

    synthetic_neighbours = NearestNeighbors(n_neighbors=1).fit(real_matrix)
    synthetic_distances, _ = synthetic_neighbours.kneighbors(synthetic_matrix)
    synthetic_min = synthetic_distances[:, 0]

    natural_mean = float(np.mean(natural_distances))
    synthetic_mean = float(np.mean(synthetic_min))
    threshold = float(np.quantile(natural_distances, 0.05))
    return {
        "nn_synthetic_to_real_mean": synthetic_mean,
        "nn_real_to_real_mean": natural_mean,
        "nn_distance_ratio": _safe_ratio_float(synthetic_mean, natural_mean),
        "nn_suspiciously_close_rate": float(np.mean(synthetic_min < threshold)),
    }


def discrimination_test(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Train a classifier to distinguish real rows from synthetic rows."""
    feature_real, feature_synthetic = _aligned_features(real, synthetic, target_column=target_column)
    if len(feature_real) < 4 or len(feature_synthetic) < 4:
        return {
            "discrimination_accuracy": np.nan,
            "discrimination_roc_auc": np.nan,
            "discrimination_privacy_score": np.nan,
        }
    data = pd.concat([feature_real, feature_synthetic], ignore_index=True)
    labels = np.r_[np.ones(len(feature_real)), np.zeros(len(feature_synthetic))]
    try:
        train_x, test_x, train_y, test_y = train_test_split(
            data,
            labels,
            test_size=0.3,
            random_state=random_state,
            stratify=labels,
        )
        model = Pipeline(
            [
                ("features", make_feature_preprocessor(train_x)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        min_samples_leaf=3,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        probabilities = model.predict_proba(test_x)[:, 1]
        accuracy = float(accuracy_score(test_y, predictions))
        roc_auc = float(roc_auc_score(test_y, probabilities))
    except ValueError:
        return {
            "discrimination_accuracy": np.nan,
            "discrimination_roc_auc": np.nan,
            "discrimination_privacy_score": np.nan,
        }
    return {
        "discrimination_accuracy": accuracy,
        "discrimination_roc_auc": roc_auc,
        "discrimination_privacy_score": float(max(0.0, 1.0 - 2.0 * abs(accuracy - 0.5))),
    }


def utility_lift_test(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str | None,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Measure whether adding synthetic rows to real training data improves utility."""
    if target_column is None or target_column not in real or target_column not in synthetic:
        return {
            "utility_task": "none",
            "utility_real_score": np.nan,
            "utility_augmented_score": np.nan,
            "utility_lift": np.nan,
        }
    clean_real = real.dropna(subset=[target_column]).reset_index(drop=True)
    clean_synthetic = synthetic.dropna(subset=[target_column]).reset_index(drop=True)
    if len(clean_real) < 10 or len(clean_synthetic) == 0:
        return {
            "utility_task": "insufficient_data",
            "utility_real_score": np.nan,
            "utility_augmented_score": np.nan,
            "utility_lift": np.nan,
        }
    target = clean_real[target_column]
    task = _infer_prediction_task(target)
    stratify = target if task == "classification" and target.nunique(dropna=True) > 1 else None
    try:
        real_train, real_test = train_test_split(
            clean_real,
            test_size=0.3,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        real_train, real_test = train_test_split(clean_real, test_size=0.3, random_state=random_state)

    base_model = _make_predictive_model(task, real_train.drop(columns=[target_column]), random_state)
    base_model.fit(real_train.drop(columns=[target_column]), real_train[target_column])
    base_score = _score_predictive_model(
        base_model,
        real_test.drop(columns=[target_column]),
        real_test[target_column],
        task,
    )

    synthetic_aligned = clean_synthetic[real_train.columns.intersection(clean_synthetic.columns)].copy()
    missing_columns = [column for column in real_train.columns if column not in synthetic_aligned]
    for column in missing_columns:
        synthetic_aligned[column] = pd.NA
    synthetic_aligned = synthetic_aligned[real_train.columns]
    synthetic_aligned[target_column] = _align_synthetic_target_for_task(
        synthetic_aligned[target_column],
        real_train[target_column],
        task,
    )
    synthetic_aligned = synthetic_aligned.dropna(subset=[target_column])
    if synthetic_aligned.empty:
        return {
            "utility_task": "insufficient_synthetic_target",
            "utility_real_score": base_score,
            "utility_augmented_score": np.nan,
            "utility_lift": np.nan,
        }
    augmented_train = pd.concat([real_train, synthetic_aligned], ignore_index=True)
    augmented_model = _make_predictive_model(task, augmented_train.drop(columns=[target_column]), random_state)
    augmented_model.fit(augmented_train.drop(columns=[target_column]), augmented_train[target_column])
    augmented_score = _score_predictive_model(
        augmented_model,
        real_test.drop(columns=[target_column]),
        real_test[target_column],
        task,
    )
    return {
        "utility_task": task,
        "utility_real_score": base_score,
        "utility_augmented_score": augmented_score,
        "utility_lift": augmented_score - base_score,
    }


def distribution_similarity_test(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """Aggregate feature-distribution similarity measures."""
    numeric = numeric_similarity(real, synthetic)
    categorical = categorical_similarity(real, synthetic)
    numeric_overlap = _safe_dataframe_mean(numeric, "histogram_overlap")
    numeric_kl = _mean_numeric_kl(real, synthetic)
    categorical_jsd = _safe_dataframe_mean(categorical, "jensen_shannon_divergence")
    categorical_tv = _safe_dataframe_mean(categorical, "total_variation_distance")
    components = [
        numeric_overlap,
        1.0 - categorical_jsd if not np.isnan(categorical_jsd) else np.nan,
        1.0 - categorical_tv if not np.isnan(categorical_tv) else np.nan,
    ]
    present = [value for value in components if not np.isnan(value)]
    return {
        "distribution_histogram_overlap": numeric_overlap,
        "distribution_numeric_kl": numeric_kl,
        "distribution_categorical_jsd": categorical_jsd,
        "distribution_categorical_tv": categorical_tv,
        "distribution_similarity_score": float(np.mean(present)) if present else np.nan,
    }


def main_measure_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str | None = None,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Compute the four primary measures used by the paper."""
    report: dict[str, float | str] = {}
    report.update(nearest_neighbor_distance_test(real, synthetic, target_column=target_column))
    report.update(discrimination_test(real, synthetic, target_column=target_column, random_state=random_state))
    report.update(utility_lift_test(real, synthetic, target_column=target_column, random_state=random_state))
    report.update(distribution_similarity_test(real, synthetic))
    return report


def classification_scores(y_true, y_pred, y_proba=None, *, average: str = "weighted") -> dict[str, float]:
    """Return classification metrics used by downstream utility experiments."""
    scores = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None:
        scores["roc_auc"] = _safe_roc_auc(y_true, y_proba)
        scores["brier_score"] = _safe_brier(y_true, y_proba)
    return scores


def regression_scores(y_true, y_pred) -> dict[str, float]:
    """Return regression metrics used by downstream utility experiments."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def downstream_utility_scores(
    *,
    train_on_synthetic_test_on_real: Mapping[str, float],
    train_on_real_test_on_real: Mapping[str, float] | None = None,
    train_on_bootstrap_test_on_real: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Create a tidy downstream-utility table for the three TODO references."""
    rows = [_named_score_row("train_on_synthetic_test_on_real", train_on_synthetic_test_on_real)]
    if train_on_real_test_on_real is not None:
        rows.append(_named_score_row("train_on_real_test_on_real", train_on_real_test_on_real))
    if train_on_bootstrap_test_on_real is not None:
        rows.append(_named_score_row("train_on_bootstrap_test_on_real", train_on_bootstrap_test_on_real))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PracticalityMetrics:
    configuration_choices: int | None = None
    lines_of_python: int | None = None
    cli_command_length: int | None = None
    fit_seconds: float | None = None
    sample_seconds: float | None = None
    peak_memory_mb: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "configuration_choices": self.configuration_choices,
            "lines_of_python": self.lines_of_python,
            "cli_command_length": self.cli_command_length,
            "fit_seconds": self.fit_seconds,
            "sample_seconds": self.sample_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


def inspectability_metrics(traces: Sequence[Mapping[str, Any] | Any], required_fields: Sequence[str] | None = None):
    """Summarize generated-row trace completeness and trace size."""
    required = list(required_fields) if required_fields is not None else [
        "anchor",
        "neighbour",
        "second_neighbour",
        "anchor_bins",
        "neighbour_bins",
        "second_neighbour_bins",
        "generated_bins",
        "decoded_bins",
    ]
    if len(traces) == 0:
        return {"trace_count": 0, "trace_completeness_rate": np.nan, "average_trace_size": np.nan}

    complete = 0
    sizes = []
    for trace in traces:
        keys = _trace_keys(trace)
        complete += int(all(field in keys for field in required))
        sizes.append(len(keys))
    return {
        "trace_count": len(traces),
        "trace_completeness_rate": complete / len(traces),
        "average_trace_size": float(np.mean(sizes)),
    }


def practicality_metrics(
    *,
    configuration: Mapping[str, Any] | None = None,
    python_code: str | None = None,
    cli_command: str | None = None,
    fit_seconds: float | None = None,
    sample_seconds: float | None = None,
    peak_memory_mb: float | None = None,
) -> PracticalityMetrics:
    return PracticalityMetrics(
        configuration_choices=_count_configuration_choices(configuration),
        lines_of_python=_count_code_lines(python_code),
        cli_command_length=len(cli_command.split()) if cli_command else None,
        fit_seconds=fit_seconds,
        sample_seconds=sample_seconds,
        peak_memory_mb=peak_memory_mb,
    )


def anonymization_safeguard_metrics(
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Sequence[str],
    *,
    replacement_report: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Measure exact/normalized overlap and optional replacement consistency."""
    rows = []
    mappings = _extract_report_mappings(replacement_report)
    for column in columns:
        source_values = set(source[column].dropna())
        candidate_values = set(candidate[column].dropna())
        normalized_source = {_normalize_value(value) for value in source_values}
        normalized_candidate = {_normalize_value(value) for value in candidate_values}
        mapping = mappings.get(column, {})
        rows.append(
            {
                "column": column,
                "exact_source_overlap_count": len(source_values & candidate_values),
                "normalized_source_overlap_count": len(normalized_source & normalized_candidate),
                "replacement_collision_count": _replacement_collision_count(mapping, normalized_source),
                "repeated_value_consistency_rate": _repeated_value_consistency(mapping),
                "manual_review_required": True,
            }
        )
    return pd.DataFrame(rows)


def distributional_similarity_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Convenience wrapper returning all distributional metric families."""
    return {
        "numeric": numeric_similarity(real, synthetic),
        "categorical": categorical_similarity(real, synthetic),
        "dependence": dependence_similarity(real, synthetic),
    }


def make_feature_preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    categorical_columns = [column for column in dataframe.columns if column not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _common_numeric_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    return [
        column
        for column in real.columns
        if column in synthetic.columns
        and pd.api.types.is_numeric_dtype(real[column])
        and pd.api.types.is_numeric_dtype(synthetic[column])
    ]


def _aligned_features(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [column for column in real.columns if column in synthetic.columns and column != target_column]
    return real[columns].reset_index(drop=True), synthetic[columns].reset_index(drop=True)


def _fit_transform_features(real: pd.DataFrame, synthetic: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([real, synthetic], ignore_index=True)
    if combined.shape[1] == 0:
        return np.zeros((len(combined), 1))
    transformed = make_feature_preprocessor(combined).fit_transform(combined)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def _infer_prediction_task(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    unique_count = target.nunique(dropna=True)
    if unique_count <= max(20, int(len(target) * 0.05)):
        return "classification"
    return "regression"


def _align_synthetic_target_for_task(synthetic_target: pd.Series, real_target: pd.Series, task: str) -> pd.Series:
    if task != "classification":
        return synthetic_target

    real_classes = pd.Series(real_target.dropna().unique())
    if pd.api.types.is_numeric_dtype(real_target):
        real_numeric = pd.to_numeric(real_classes, errors="coerce")
        if real_numeric.notna().all():
            synthetic_numeric = pd.to_numeric(synthetic_target, errors="coerce")
            class_numeric = real_numeric.to_numpy(dtype=float)
            aligned = pd.Series(np.nan, index=synthetic_target.index, dtype=float)
            valid = synthetic_numeric.notna()
            if valid.any():
                distances = np.abs(synthetic_numeric.loc[valid].to_numpy(dtype=float)[:, None] - class_numeric[None, :])
                aligned.loc[valid] = class_numeric[np.argmin(distances, axis=1)]
            return aligned

    valid_classes = set(real_classes)
    return synthetic_target.where(synthetic_target.isin(valid_classes))


def _make_predictive_model(task: str, features: pd.DataFrame, random_state: int) -> Pipeline:
    estimator = (
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        )
        if task == "classification"
        else RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        )
    )
    return Pipeline([("features", make_feature_preprocessor(features)), ("model", estimator)])


def _score_predictive_model(model: Pipeline, features: pd.DataFrame, target: pd.Series, task: str) -> float:
    predictions = model.predict(features)
    if task == "classification":
        return float(f1_score(target, predictions, average="weighted", zero_division=0))
    return float(r2_score(target, predictions))


def _safe_dataframe_mean(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe.empty or column not in dataframe:
        return np.nan
    return float(dataframe[column].mean(skipna=True))


def _safe_ratio_float(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    return float(numerator / denominator)


def _mean_numeric_kl(real: pd.DataFrame, synthetic: pd.DataFrame, *, bins: int = 20) -> float:
    values = []
    for column in _common_numeric_columns(real, synthetic):
        real_values = pd.to_numeric(real[column], errors="coerce").dropna().to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna().to_numpy(dtype=float)
        value = _histogram_kl(real_values, synthetic_values, bins=bins)
        if not np.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _histogram_kl(real_values: np.ndarray, synthetic_values: np.ndarray, *, bins: int) -> float:
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    combined = np.concatenate([real_values, synthetic_values])
    if np.nanmin(combined) == np.nanmax(combined):
        return 0.0
    real_hist, edges = np.histogram(real_values, bins=bins, range=(np.nanmin(combined), np.nanmax(combined)))
    synthetic_hist, _ = np.histogram(synthetic_values, bins=edges)
    epsilon = 1e-12
    real_probs = (real_hist + epsilon) / (real_hist.sum() + epsilon * len(real_hist))
    synthetic_probs = (synthetic_hist + epsilon) / (synthetic_hist.sum() + epsilon * len(synthetic_hist))
    return float(np.sum(real_probs * np.log(real_probs / synthetic_probs)))


def _common_categorical_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    numeric = set(_common_numeric_columns(real, synthetic))
    return [column for column in real.columns if column in synthetic.columns and column not in numeric]


def _normalized_category_series(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str)


def _aligned_probabilities(real: pd.Series, synthetic: pd.Series) -> tuple[pd.Series, pd.Series]:
    real_probs = real.value_counts(normalize=True, dropna=False)
    synthetic_probs = synthetic.value_counts(normalize=True, dropna=False)
    index = real_probs.index.union(synthetic_probs.index)
    return real_probs.reindex(index, fill_value=0.0), synthetic_probs.reindex(index, fill_value=0.0)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def _safe_abs_delta(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return float(abs(left - right))


def _safe_ks(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    return float(ks_2samp(real_values, synthetic_values).statistic)


def _safe_wasserstein(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    return float(wasserstein_distance(real_values, synthetic_values))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _numeric_correlation(left: pd.Series, right: pd.Series, *, method: str) -> float:
    clean = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(clean) < 2:
        return np.nan
    value = clean["left"].corr(clean["right"], method=method)
    return float(abs(value)) if pd.notna(value) else np.nan


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([])
    return matrix[np.triu_indices_from(matrix, k=1)]


def _entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True, dropna=False).to_numpy(dtype=float)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _safe_roc_auc(y_true, y_proba) -> float:
    try:
        y_proba_array = np.asarray(y_proba)
        if y_proba_array.ndim == 2 and y_proba_array.shape[1] == 2:
            y_proba_array = y_proba_array[:, 1]
        return float(roc_auc_score(y_true, y_proba_array))
    except ValueError:
        return np.nan


def _safe_brier(y_true, y_proba) -> float:
    y_proba_array = np.asarray(y_proba)
    if y_proba_array.ndim == 2 and y_proba_array.shape[1] == 2:
        y_proba_array = y_proba_array[:, 1]
    labels = pd.Series(y_true).dropna().unique()
    if len(labels) != 2:
        return np.nan
    try:
        return float(brier_score_loss(y_true, y_proba_array))
    except ValueError:
        return np.nan


def _named_score_row(name: str, scores: Mapping[str, float]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"evaluation": name}
    row.update({key: float(value) for key, value in scores.items()})
    return row


def _trace_keys(trace: Mapping[str, Any] | Any) -> set[str]:
    if isinstance(trace, Mapping):
        return set(trace.keys())
    if hasattr(trace, "__dict__"):
        return set(vars(trace).keys())
    return set()


def _count_configuration_choices(configuration: Mapping[str, Any] | None) -> int | None:
    if configuration is None:
        return None
    return sum(1 for value in configuration.values() if value not in (None, {}, [], ""))


def _count_code_lines(python_code: str | None) -> int | None:
    if python_code is None:
        return None
    return sum(1 for line in python_code.splitlines() if line.strip() and not line.strip().startswith("#"))


def _extract_report_mappings(replacement_report: Mapping[str, Any] | None) -> dict[str, dict[Any, Any]]:
    if not replacement_report:
        return {}
    mappings = replacement_report.get("mappings", replacement_report)
    if not isinstance(mappings, Mapping):
        return {}
    return {
        column: dict(mapping)
        for column, mapping in mappings.items()
        if isinstance(mapping, Mapping)
    }


def _replacement_collision_count(mapping: Mapping[Any, Any], normalized_source: set[str]) -> int:
    return sum(1 for replacement in mapping.values() if _normalize_value(replacement) in normalized_source)


def _repeated_value_consistency(mapping: Mapping[Any, Any]) -> float:
    if not mapping:
        return np.nan
    replacements_by_source = {}
    for source, replacement in mapping.items():
        replacements_by_source.setdefault(source, set()).add(replacement)
    consistent = sum(1 for replacements in replacements_by_source.values() if len(replacements) == 1)
    return consistent / len(replacements_by_source)


def _normalize_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())
