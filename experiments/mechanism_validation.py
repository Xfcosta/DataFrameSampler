from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from dataframe_sampler import DataFrameSampler

from .datasets import DatasetExperimentConfig
from .manifold_validation import _train_test_dataframe_split, deterministic_dataframe_sample


MECHANISM_VALIDATION_COLUMNS = [
    "dataset",
    "column",
    "block_width",
    "cardinality",
    "missing_rate",
    "n_train",
    "n_test",
    "majority_accuracy",
    "nca_accuracy",
    "pca_accuracy",
    "raw_context_accuracy",
    "nca_balanced_accuracy",
    "pca_balanced_accuracy",
    "raw_context_balanced_accuracy",
    "nca_macro_f1",
    "pca_macro_f1",
    "raw_context_macro_f1",
    "nca_accuracy_lift_over_majority",
    "nca_accuracy_lift_over_pca",
    "reason",
]

DECODER_CALIBRATION_COLUMNS = [
    "dataset",
    "column",
    "block_width",
    "cardinality",
    "cardinality_bucket",
    "missing_rate",
    "n_train",
    "n_test",
    "accuracy",
    "mean_top_confidence",
    "confidence_gap",
    "negative_log_loss",
    "brier_score",
    "expected_calibration_error",
    "reason",
]


def run_mechanism_validation_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    test_size: float = 0.3,
    max_train_rows: int = 800,
    max_test_rows: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataframe) < 4:
        mechanism = _empty_mechanism_frame(config.dataset_name)
        calibration = _empty_calibration_frame(config.dataset_name)
    else:
        train, test = _train_test_dataframe_split(
            dataframe,
            target_column=config.target_column,
            test_size=test_size,
            random_state=config.random_state,
        )
        train = deterministic_dataframe_sample(
            train,
            max_rows=max_train_rows,
            random_state=config.random_state,
        )
        test = deterministic_dataframe_sample(
            test,
            max_rows=max_test_rows,
            random_state=config.random_state + 1,
        )
        mechanism, calibration = mechanism_validation_report(
            train=train,
            test=test,
            dataset_name=config.dataset_name,
            sampler_config=sampler_config or config.sampler_config,
            random_state=config.random_state,
        )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    mechanism.to_csv(results_path / f"{config.dataset_name}_mechanism_validation.csv", index=False)
    calibration.to_csv(results_path / f"{config.dataset_name}_decoder_calibration.csv", index=False)
    return mechanism, calibration


def mechanism_validation_report(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    sampler_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train.empty or test.empty:
        return _empty_mechanism_frame(dataset_name), _empty_calibration_frame(dataset_name)

    sampler_kwargs = dict(sampler_config or {})
    sampler_kwargs.setdefault("random_state", random_state)
    sampler = DataFrameSampler(**sampler_kwargs).fit(train)

    train_latent = sampler.transform(train)
    test_latent = sampler.transform(test)
    train_offsets = _categorical_block_offsets(sampler)

    mechanism_rows = []
    calibration_rows = []
    for column in sampler.categorical_columns_:
        start, stop = train_offsets[column]
        labels_train = sampler._categorical_keys(train[column]).to_numpy(dtype=str)
        labels_test = sampler._categorical_keys(test[column]).to_numpy(dtype=str)
        train_block = train_latent[:, start:stop]
        test_block = test_latent[:, start:stop]
        train_context = np.hstack([train_latent[:, :start], train_latent[:, stop:]])
        test_context = np.hstack([test_latent[:, :start], test_latent[:, stop:]])
        metadata = _column_metadata(dataset_name, column, train[column], labels_train, len(train), len(test), stop - start)
        mechanism_rows.append(
            {
                **metadata,
                **_mechanism_metrics(
                    labels_train=labels_train,
                    labels_test=labels_test,
                    train_block=train_block,
                    test_block=test_block,
                    train_context=train_context,
                    test_context=test_context,
                    width=stop - start,
                    random_state=random_state,
                ),
            }
        )
        decoder = sampler.decoders_[column]
        probabilities = decoder.predict_proba(test_block)
        calibration_rows.append(
            {
                **_calibration_metadata(metadata),
                **_calibration_metrics(
                    labels_test=labels_test,
                    classes=np.asarray(decoder.classes_, dtype=str),
                    probabilities=probabilities,
                ),
            }
        )

    return (
        pd.DataFrame(mechanism_rows, columns=MECHANISM_VALIDATION_COLUMNS),
        pd.DataFrame(calibration_rows, columns=DECODER_CALIBRATION_COLUMNS),
    )


def summarize_mechanism_validation(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "columns_evaluated",
                "mean_cardinality",
                "mean_nca_accuracy",
                "mean_majority_accuracy",
                "mean_pca_accuracy",
                "mean_raw_context_accuracy",
                "mean_lift_over_majority",
                "mean_lift_over_pca",
            ]
        )
    grouped = rows.groupby("dataset", dropna=False)
    return grouped.agg(
        columns_evaluated=("column", "count"),
        mean_cardinality=("cardinality", "mean"),
        mean_nca_accuracy=("nca_accuracy", "mean"),
        mean_majority_accuracy=("majority_accuracy", "mean"),
        mean_pca_accuracy=("pca_accuracy", "mean"),
        mean_raw_context_accuracy=("raw_context_accuracy", "mean"),
        mean_lift_over_majority=("nca_accuracy_lift_over_majority", "mean"),
        mean_lift_over_pca=("nca_accuracy_lift_over_pca", "mean"),
    ).reset_index()


def summarize_decoder_calibration(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "cardinality_bucket",
                "columns_evaluated",
                "mean_accuracy",
                "mean_top_confidence",
                "mean_confidence_gap",
                "mean_negative_log_loss",
                "mean_brier_score",
                "mean_expected_calibration_error",
            ]
        )
    grouped = rows.groupby(["dataset", "cardinality_bucket"], dropna=False)
    return grouped.agg(
        columns_evaluated=("column", "count"),
        mean_accuracy=("accuracy", "mean"),
        mean_top_confidence=("mean_top_confidence", "mean"),
        mean_confidence_gap=("confidence_gap", "mean"),
        mean_negative_log_loss=("negative_log_loss", "mean"),
        mean_brier_score=("brier_score", "mean"),
        mean_expected_calibration_error=("expected_calibration_error", "mean"),
    ).reset_index()


def _mechanism_metrics(
    *,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    train_block: np.ndarray,
    test_block: np.ndarray,
    train_context: np.ndarray,
    test_context: np.ndarray,
    width: int,
    random_state: int,
) -> dict[str, float | str]:
    if len(np.unique(labels_train)) < 2:
        return {
            "majority_accuracy": np.nan,
            "nca_accuracy": np.nan,
            "pca_accuracy": np.nan,
            "raw_context_accuracy": np.nan,
            "nca_balanced_accuracy": np.nan,
            "pca_balanced_accuracy": np.nan,
            "raw_context_balanced_accuracy": np.nan,
            "nca_macro_f1": np.nan,
            "pca_macro_f1": np.nan,
            "raw_context_macro_f1": np.nan,
            "nca_accuracy_lift_over_majority": np.nan,
            "nca_accuracy_lift_over_pca": np.nan,
            "reason": "one_train_class",
        }

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(labels_train), 1)), labels_train)
    majority_pred = majority.predict(np.zeros((len(labels_test), 1)))
    majority_accuracy = float(accuracy_score(labels_test, majority_pred))

    nca_scores = _classifier_scores(
        train_block,
        labels_train,
        test_block,
        labels_test,
        random_state=random_state,
    )
    pca_train, pca_test = _pca_projection(train_context, test_context, width=width, random_state=random_state)
    pca_scores = _classifier_scores(
        pca_train,
        labels_train,
        pca_test,
        labels_test,
        random_state=random_state,
    )
    raw_scores = _classifier_scores(
        train_context,
        labels_train,
        test_context,
        labels_test,
        random_state=random_state,
    )
    return {
        "majority_accuracy": majority_accuracy,
        "nca_accuracy": nca_scores["accuracy"],
        "pca_accuracy": pca_scores["accuracy"],
        "raw_context_accuracy": raw_scores["accuracy"],
        "nca_balanced_accuracy": nca_scores["balanced_accuracy"],
        "pca_balanced_accuracy": pca_scores["balanced_accuracy"],
        "raw_context_balanced_accuracy": raw_scores["balanced_accuracy"],
        "nca_macro_f1": nca_scores["macro_f1"],
        "pca_macro_f1": pca_scores["macro_f1"],
        "raw_context_macro_f1": raw_scores["macro_f1"],
        "nca_accuracy_lift_over_majority": nca_scores["accuracy"] - majority_accuracy,
        "nca_accuracy_lift_over_pca": nca_scores["accuracy"] - pca_scores["accuracy"],
        "reason": "ok",
    }


def _classifier_scores(
    train_values: np.ndarray,
    labels_train: np.ndarray,
    test_values: np.ndarray,
    labels_test: np.ndarray,
    *,
    random_state: int,
) -> dict[str, float]:
    if train_values.shape[1] == 0:
        train_values = np.zeros((len(train_values), 1), dtype=float)
        test_values = np.zeros((len(test_values), 1), dtype=float)
    classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(train_values, labels_train)
    predictions = classifier.predict(test_values)
    return {
        "accuracy": float(accuracy_score(labels_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_test, predictions)),
        "macro_f1": float(f1_score(labels_test, predictions, average="macro", zero_division=0)),
    }


def _pca_projection(
    train_context: np.ndarray,
    test_context: np.ndarray,
    *,
    width: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_context.shape[1] == 0:
        return (
            np.zeros((len(train_context), width), dtype=float),
            np.zeros((len(test_context), width), dtype=float),
        )
    effective_width = min(width, train_context.shape[1], max(1, len(train_context) - 1))
    projector = PCA(n_components=effective_width, random_state=random_state)
    train_projected = projector.fit_transform(train_context)
    test_projected = projector.transform(test_context)
    if effective_width == width:
        return train_projected, test_projected
    padded_train = np.zeros((len(train_context), width), dtype=float)
    padded_test = np.zeros((len(test_context), width), dtype=float)
    padded_train[:, :effective_width] = train_projected
    padded_test[:, :effective_width] = test_projected
    return padded_train, padded_test


def _calibration_metrics(
    *,
    labels_test: np.ndarray,
    classes: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | str]:
    if len(classes) < 2:
        return {
            "accuracy": np.nan,
            "mean_top_confidence": np.nan,
            "confidence_gap": np.nan,
            "negative_log_loss": np.nan,
            "brier_score": np.nan,
            "expected_calibration_error": np.nan,
            "reason": "one_decoder_class",
        }
    probabilities = np.asarray(probabilities, dtype=float)
    row_sums = probabilities.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    probabilities = probabilities / row_sums[:, None]
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    true_indices = np.asarray([class_to_index.get(label, -1) for label in labels_test], dtype=int)
    top_indices = np.argmax(probabilities, axis=1)
    top_confidence = probabilities[np.arange(len(probabilities)), top_indices]
    correct = top_indices == true_indices
    clipped_true_probabilities = np.full(len(labels_test), 1e-15, dtype=float)
    known = true_indices >= 0
    clipped_true_probabilities[known] = probabilities[np.arange(len(probabilities))[known], true_indices[known]]
    clipped_true_probabilities = np.clip(clipped_true_probabilities, 1e-15, 1.0)
    one_hot = np.zeros_like(probabilities)
    if known.any():
        one_hot[np.arange(len(probabilities))[known], true_indices[known]] = 1.0
    accuracy = float(np.mean(correct)) if len(correct) else np.nan
    mean_confidence = float(np.mean(top_confidence)) if len(top_confidence) else np.nan
    return {
        "accuracy": accuracy,
        "mean_top_confidence": mean_confidence,
        "confidence_gap": mean_confidence - accuracy if np.isfinite(mean_confidence) else np.nan,
        "negative_log_loss": float(-np.mean(np.log(clipped_true_probabilities))),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": _expected_calibration_error(top_confidence, correct, n_bins=n_bins),
        "reason": "ok",
    }


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, *, n_bins: int) -> float:
    if len(confidence) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def _categorical_block_offsets(sampler: DataFrameSampler) -> dict[str, tuple[int, int]]:
    offsets = {}
    offset = len(sampler.numeric_columns_)
    for column in sampler.categorical_columns_:
        width = sampler._components_for_column(column)
        offsets[column] = (offset, offset + width)
        offset += width
    return offsets


def _column_metadata(
    dataset_name: str,
    column: str,
    train_series: pd.Series,
    labels_train: np.ndarray,
    n_train: int,
    n_test: int,
    block_width: int,
) -> dict[str, Any]:
    cardinality = int(pd.Series(labels_train).nunique(dropna=False))
    return {
        "dataset": dataset_name,
        "column": column,
        "block_width": block_width,
        "cardinality": cardinality,
        "missing_rate": float(train_series.isna().mean()),
        "n_train": n_train,
        "n_test": n_test,
    }


def _calibration_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cardinality = int(metadata["cardinality"])
    return {
        "dataset": metadata["dataset"],
        "column": metadata["column"],
        "block_width": metadata["block_width"],
        "cardinality": cardinality,
        "cardinality_bucket": _cardinality_bucket(cardinality),
        "missing_rate": metadata["missing_rate"],
        "n_train": metadata["n_train"],
        "n_test": metadata["n_test"],
    }


def _cardinality_bucket(cardinality: int) -> str:
    if cardinality <= 2:
        return "binary"
    if cardinality <= 10:
        return "low"
    if cardinality <= 50:
        return "medium"
    return "high"


def _empty_mechanism_frame(dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=MECHANISM_VALIDATION_COLUMNS)
    if dataset_name:
        frame["dataset"] = frame.get("dataset", pd.Series(dtype=object))
    return frame


def _empty_calibration_frame(dataset_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=DECODER_CALIBRATION_COLUMNS)
    if dataset_name:
        frame["dataset"] = frame.get("dataset", pd.Series(dtype=object))
    return frame
