from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataframe_sampler import DataFrameSampler

from .baselines import StratifiedColumnBaseline
from .datasets import DatasetExperimentConfig
from .instrumentation import measure_call
from .metrics import _infer_prediction_task, make_feature_preprocessor


IMBALANCE_VALIDATION_DATASETS = {"adult", "bank_marketing", "pima_diabetes", "heart_disease"}

IMBALANCE_VALIDATION_COLUMNS = [
    "dataset",
    "method",
    "target_column",
    "minority_class",
    "majority_class",
    "train_rows",
    "test_rows",
    "synthetic_rows",
    "train_minority_rate",
    "augmented_minority_rate",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "pr_auc",
    "fit_seconds",
    "sample_seconds",
    "peak_memory_mb",
    "reason",
]


def run_imbalance_validation_for_config(
    config: DatasetExperimentConfig,
    dataframe: pd.DataFrame,
    *,
    results_dir: str | Path,
    sampler_config: Mapping[str, Any] | None = None,
    test_size: float = 0.3,
    max_train_rows: int = 1200,
    max_test_rows: int = 500,
) -> pd.DataFrame:
    """Run a capped secondary target-rebalancing diagnostic.

    The diagnostic is intentionally limited to selected binary-classification
    datasets. It is not a general imbalance-learning benchmark.
    """
    if config.dataset_name not in IMBALANCE_VALIDATION_DATASETS:
        return _empty_imbalance_frame(config.dataset_name, reason="dataset_not_selected")
    if config.target_column is None or config.target_column not in dataframe.columns:
        return _empty_imbalance_frame(config.dataset_name, reason="missing_target")

    clean = dataframe.dropna(subset=[config.target_column]).reset_index(drop=True)
    if len(clean) < 20:
        return _empty_imbalance_frame(config.dataset_name, reason="insufficient_rows")
    if _infer_prediction_task(clean[config.target_column]) != "classification":
        return _empty_imbalance_frame(config.dataset_name, reason="target_not_classification")
    if clean[config.target_column].nunique(dropna=True) != 2:
        return _empty_imbalance_frame(config.dataset_name, reason="target_not_binary")

    train, test = _stratified_train_test_split(
        clean,
        target_column=config.target_column,
        test_size=test_size,
        random_state=config.random_state,
    )
    train = _cap_stratified(train, config.target_column, max_train_rows, config.random_state)
    test = _cap_stratified(test, config.target_column, max_test_rows, config.random_state + 1)

    counts = train[config.target_column].value_counts(dropna=False)
    if len(counts) != 2 or counts.min() < 2:
        report = _empty_imbalance_frame(config.dataset_name, reason="insufficient_minority_rows")
    else:
        minority_class = counts.idxmin()
        majority_class = counts.idxmax()
        n_to_generate = int(counts.max() - counts.min())
        report = imbalance_validation_report(
            train=train,
            test=test,
            dataset_name=config.dataset_name,
            target_column=config.target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            n_to_generate=n_to_generate,
            sampler_config=sampler_config or _sampler_config_with_random_state(
                config.sampler_config,
                config.random_state,
            ),
            random_state=config.random_state,
        )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    report.to_csv(results_path / f"{config.dataset_name}_imbalance_validation.csv", index=False)
    return report


def imbalance_validation_report(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    minority_class: Any,
    majority_class: Any,
    n_to_generate: int,
    sampler_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = [
        _evaluate_augmented_training_set(
            train=train,
            test=test,
            augmented=train,
            dataset_name=dataset_name,
            method="real_train",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            synthetic_rows=0,
            fit_seconds=0.0,
            sample_seconds=0.0,
            peak_memory_mb=0.0,
            reason="",
            random_state=random_state,
        )
    ]

    rows.append(
        _dataframe_sampler_row(
            train=train,
            test=test,
            dataset_name=dataset_name,
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            n_to_generate=n_to_generate,
            sampler_config=sampler_config or {},
            random_state=random_state,
        )
    )
    rows.append(
        _smotenc_row(
            train=train,
            test=test,
            dataset_name=dataset_name,
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            random_state=random_state,
        )
    )
    rows.append(
        _stratified_columns_row(
            train=train,
            test=test,
            dataset_name=dataset_name,
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            n_to_generate=n_to_generate,
            random_state=random_state,
        )
    )
    return pd.DataFrame(rows, columns=IMBALANCE_VALIDATION_COLUMNS)


def summarize_imbalance_validation(validations: pd.DataFrame) -> pd.DataFrame:
    valid = validations[validations["reason"].fillna("") == ""].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["dataset", "method"], dropna=False)
        .agg(
            target_column=("target_column", "first"),
            minority_class=("minority_class", "first"),
            train_minority_rate=("train_minority_rate", "first"),
            augmented_minority_rate=("augmented_minority_rate", "first"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            minority_recall=("minority_recall", "mean"),
            pr_auc=("pr_auc", "mean"),
            synthetic_rows=("synthetic_rows", "first"),
        )
        .reset_index()
    )


def _dataframe_sampler_row(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    minority_class: Any,
    majority_class: Any,
    n_to_generate: int,
    sampler_config: Mapping[str, Any],
    random_state: int,
) -> dict[str, Any]:
    if n_to_generate <= 0:
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="dataframe_sampler_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason="already_balanced",
        )
    minority = train[train[target_column] == minority_class].reset_index(drop=True)
    features = minority.drop(columns=[target_column])
    if len(features) < 2:
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="dataframe_sampler_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason="insufficient_minority_rows",
        )
    config = dict(sampler_config)
    config.setdefault("random_state", random_state)
    config["n_neighbours"] = min(int(config.get("n_neighbours", 10)), max(1, len(features) - 1))
    try:
        sampler = DataFrameSampler(**config)
        fit = measure_call(lambda: sampler.fit(features))
        sample = measure_call(lambda: sampler.generate(n_samples=n_to_generate))
        synthetic = sample.value
        synthetic[target_column] = minority_class
        synthetic = synthetic[train.columns]
        augmented = pd.concat([train, synthetic], ignore_index=True)
        return _evaluate_augmented_training_set(
            train=train,
            test=test,
            augmented=augmented,
            dataset_name=dataset_name,
            method="dataframe_sampler_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            synthetic_rows=len(synthetic),
            fit_seconds=fit.seconds,
            sample_seconds=sample.seconds,
            peak_memory_mb=max(fit.peak_memory_mb, sample.peak_memory_mb),
            reason="",
            random_state=random_state,
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="dataframe_sampler_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason=f"failed: {exc}",
        )


def _smotenc_row(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    minority_class: Any,
    majority_class: Any,
    random_state: int,
) -> dict[str, Any]:
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
    except ImportError:
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="smotenc_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason="imbalanced_learn_missing",
        )
    try:
        fit = measure_call(
            lambda: _fit_resample_smotenc(
                train,
                target_column=target_column,
                random_state=random_state,
                smote_cls=SMOTE,
                smotenc_cls=SMOTENC,
            )
        )
        augmented = fit.value
        synthetic_rows = max(0, len(augmented) - len(train))
        return _evaluate_augmented_training_set(
            train=train,
            test=test,
            augmented=augmented,
            dataset_name=dataset_name,
            method="smotenc_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            synthetic_rows=synthetic_rows,
            fit_seconds=fit.seconds,
            sample_seconds=0.0,
            peak_memory_mb=fit.peak_memory_mb,
            reason="",
            random_state=random_state,
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="smotenc_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason=f"failed: {exc}",
        )


def _stratified_columns_row(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    minority_class: Any,
    majority_class: Any,
    n_to_generate: int,
    random_state: int,
) -> dict[str, Any]:
    if n_to_generate <= 0:
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="stratified_columns_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason="already_balanced",
        )
    minority = train[train[target_column] == minority_class].reset_index(drop=True)
    try:
        baseline = StratifiedColumnBaseline(target_column, random_state=random_state)
        fit = measure_call(lambda: baseline.fit(minority, target_column=target_column))
        sample = measure_call(lambda: baseline.sample(n_to_generate))
        synthetic = sample.value
        synthetic[target_column] = minority_class
        augmented = pd.concat([train, synthetic[train.columns]], ignore_index=True)
        return _evaluate_augmented_training_set(
            train=train,
            test=test,
            augmented=augmented,
            dataset_name=dataset_name,
            method="stratified_columns_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            synthetic_rows=len(synthetic),
            fit_seconds=fit.seconds,
            sample_seconds=sample.seconds,
            peak_memory_mb=max(fit.peak_memory_mb, sample.peak_memory_mb),
            reason="",
            random_state=random_state,
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method="stratified_columns_balanced",
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason=f"failed: {exc}",
        )


def _fit_resample_smotenc(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    random_state: int,
    smote_cls,
    smotenc_cls,
) -> pd.DataFrame:
    features = dataframe.drop(columns=[target_column]).copy()
    target = dataframe[target_column].copy()
    encoded, categorical_features, metadata = _encode_smotenc_features(features)
    if categorical_features:
        sampler = smotenc_cls(
            categorical_features=categorical_features,
            random_state=random_state,
            sampling_strategy="auto",
        )
    else:
        sampler = smote_cls(random_state=random_state, sampling_strategy="auto")
    resampled_x, resampled_y = sampler.fit_resample(encoded, target)
    resampled = pd.DataFrame(resampled_x, columns=features.columns)
    for column, inverse in metadata["inverse_category_maps"].items():
        resampled[column] = (
            pd.to_numeric(resampled[column], errors="coerce")
            .round()
            .astype("Int64")
            .map(inverse)
            .replace("__MISSING__", pd.NA)
        )
    for column, fill_value in metadata["numeric_fill_values"].items():
        resampled[column] = pd.to_numeric(resampled[column], errors="coerce").fillna(fill_value)
    resampled[target_column] = resampled_y
    return _restore_columns_and_dtypes(resampled[dataframe.columns], dataframe)


def _encode_smotenc_features(features: pd.DataFrame):
    encoded = pd.DataFrame(index=features.index)
    categorical_features: list[int] = []
    inverse_category_maps = {}
    numeric_fill_values = {}
    for idx, column in enumerate(features.columns):
        series = features[column]
        if _is_smotenc_categorical(series):
            categorical_features.append(idx)
            labels = series.astype("object").where(series.notna(), "__MISSING__").map(str)
            categories = pd.Index(labels.unique())
            mapping = {value: code for code, value in enumerate(categories)}
            inverse_category_maps[column] = {code: value for value, code in mapping.items()}
            encoded[column] = labels.map(mapping).astype(float)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
            numeric_fill_values[column] = fill_value
            encoded[column] = numeric.fillna(fill_value).astype(float)
    return encoded, categorical_features, {
        "inverse_category_maps": inverse_category_maps,
        "numeric_fill_values": numeric_fill_values,
    }


def _is_smotenc_categorical(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    return series.dropna().nunique() <= 2


def _evaluate_augmented_training_set(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    augmented: pd.DataFrame,
    dataset_name: str,
    method: str,
    target_column: str,
    minority_class: Any,
    majority_class: Any,
    synthetic_rows: int,
    fit_seconds: float,
    sample_seconds: float,
    peak_memory_mb: float,
    reason: str,
    random_state: int,
) -> dict[str, Any]:
    try:
        model = _classification_model(augmented.drop(columns=[target_column]), random_state)
        model.fit(augmented.drop(columns=[target_column]), augmented[target_column])
        predictions = model.predict(test.drop(columns=[target_column]))
        probabilities = _minority_probabilities(model, test.drop(columns=[target_column]), minority_class)
        target = test[target_column]
        return {
            "dataset": dataset_name,
            "method": method,
            "target_column": target_column,
            "minority_class": minority_class,
            "majority_class": majority_class,
            "train_rows": len(train),
            "test_rows": len(test),
            "synthetic_rows": synthetic_rows,
            "train_minority_rate": _minority_rate(train[target_column], minority_class),
            "augmented_minority_rate": _minority_rate(augmented[target_column], minority_class),
            "accuracy": float(accuracy_score(target, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
            "macro_f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
            "minority_recall": float(recall_score(target, predictions, labels=[minority_class], average="macro", zero_division=0)),
            "pr_auc": _binary_average_precision(target, probabilities, minority_class),
            "fit_seconds": fit_seconds,
            "sample_seconds": sample_seconds,
            "peak_memory_mb": peak_memory_mb,
            "reason": reason,
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return _skipped_row(
            train,
            test,
            dataset_name=dataset_name,
            method=method,
            target_column=target_column,
            minority_class=minority_class,
            majority_class=majority_class,
            reason=f"evaluation_failed: {exc}",
        )


def _classification_model(features: pd.DataFrame, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _minority_probabilities(model: Pipeline, features: pd.DataFrame, minority_class: Any) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    classifier = model.named_steps["model"]
    classes = list(classifier.classes_)
    if minority_class not in classes:
        return None
    return model.predict_proba(features)[:, classes.index(minority_class)]


def _binary_average_precision(target: pd.Series, probabilities: np.ndarray | None, minority_class: Any) -> float:
    if probabilities is None or target.nunique(dropna=True) != 2:
        return np.nan
    binary = (target == minority_class).astype(int)
    try:
        return float(average_precision_score(binary, probabilities))
    except ValueError:
        return np.nan


def _skipped_row(
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    *,
    dataset_name: str,
    method: str = "",
    target_column: str = "",
    minority_class: Any = "",
    majority_class: Any = "",
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "method": method,
        "target_column": target_column,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "train_rows": 0 if train is None else len(train),
        "test_rows": 0 if test is None else len(test),
        "synthetic_rows": 0,
        "train_minority_rate": np.nan if train is None or target_column not in train else _minority_rate(train[target_column], minority_class),
        "augmented_minority_rate": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "minority_recall": np.nan,
        "pr_auc": np.nan,
        "fit_seconds": np.nan,
        "sample_seconds": np.nan,
        "peak_memory_mb": np.nan,
        "reason": reason,
    }


def _empty_imbalance_frame(dataset_name: str, *, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{column: np.nan for column in IMBALANCE_VALIDATION_COLUMNS},
                "dataset": dataset_name,
                "reason": reason,
            }
        ],
        columns=IMBALANCE_VALIDATION_COLUMNS,
    )


def _stratified_train_test_split(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train, test = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=dataframe[target_column],
        )
    except ValueError:
        train, test = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _cap_stratified(
    dataframe: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(dataframe) <= max_rows:
        return dataframe.reset_index(drop=True)
    fractions = dataframe[target_column].value_counts(normalize=True, dropna=False)
    parts = []
    remaining = max_rows
    for idx, (value, fraction) in enumerate(fractions.items()):
        group = dataframe[dataframe[target_column] == value]
        if idx == len(fractions) - 1:
            n_rows = min(len(group), remaining)
        else:
            n_rows = min(len(group), max(1, int(round(max_rows * fraction))))
            remaining -= n_rows
        parts.append(group.sample(n=n_rows, random_state=random_state + idx))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _minority_rate(target: pd.Series, minority_class: Any) -> float:
    if len(target) == 0:
        return np.nan
    return float((target == minority_class).mean())


def _restore_columns_and_dtypes(dataframe: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    restored = dataframe.copy()
    for column in reference.columns:
        if column not in restored.columns:
            restored[column] = pd.NA
        try:
            if pd.api.types.is_integer_dtype(reference[column].dtype) and restored[column].isna().any():
                continue
            restored[column] = restored[column].astype(reference[column].dtype)
        except (TypeError, ValueError):
            pass
    return restored[reference.columns]


def _sampler_config_with_random_state(config: Mapping[str, Any], random_state: int) -> dict[str, Any]:
    sampler_config = dict(config)
    sampler_config.setdefault("random_state", random_state)
    return sampler_config
