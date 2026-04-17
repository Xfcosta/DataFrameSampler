from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataframe_sampler import DataFrameSampler

from .datasets import DatasetExperimentConfig
from .metrics import _infer_prediction_task
from .numeric_projection import numeric_view
from .workflow import sampler_config_with_random_state


def target_column_choice(config: DatasetExperimentConfig, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the explicit target-column choice shown in dataset notebooks."""
    if config.target_column is None:
        return pd.DataFrame(
            [
                {
                    "target_column": "",
                    "task": "none",
                    "available": False,
                    "unique": 0,
                    "missing": 0,
                    "note": "No predictive target is configured for this dataset.",
                }
            ]
        )
    available = config.target_column in dataframe.columns
    target = dataframe[config.target_column] if available else pd.Series(dtype=float)
    task = _infer_prediction_task(target) if available and len(target.dropna()) else "unavailable"
    return pd.DataFrame(
        [
            {
                "target_column": config.target_column,
                "task": task,
                "available": available,
                "unique": int(target.nunique(dropna=True)) if available else 0,
                "missing": int(target.isna().sum()) if available else 0,
                "note": "Configured in experiments/datasets.py and used for the prediction task.",
            }
        ]
    )


def predictive_performance_report(
    dataframe: pd.DataFrame,
    config: DatasetExperimentConfig,
    *,
    test_size: float = 0.3,
    n_synthetic: int | None = None,
) -> pd.DataFrame:
    """Compare real-trained and synthetic-trained predictors on a real test split.

    The split is made on prepared real rows. The sampler is fit only on the real
    training split, then the fitted sampler transforms train, test, and
    generated synthetic rows into the same latent numeric coordinate system.
    """
    if config.target_column is None or config.target_column not in dataframe.columns:
        raise ValueError("A configured target column is required for predictive evaluation.")

    clean = dataframe.dropna(subset=[config.target_column]).reset_index(drop=True)
    if len(clean) < 10:
        raise ValueError("At least 10 rows with non-missing target values are required.")

    task = _infer_prediction_task(clean[config.target_column])
    stratify = _stratification_target(clean[config.target_column], task)
    train, test = train_test_split(
        clean,
        test_size=test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    sampler = DataFrameSampler(
        **sampler_config_with_random_state(config.manual_sampler_config, config.random_state)
    )
    sampler.fit(train)
    synthetic = sampler.generate(n_samples=n_synthetic or len(train)).reset_index(drop=True)

    train_numeric = predictive_numeric_view(train, sampler, config.target_column)
    test_numeric = predictive_numeric_view(test, sampler, config.target_column)
    synthetic_numeric = predictive_numeric_view(synthetic, sampler, config.target_column)

    rows = []
    rows.append(
        _evaluate_numeric_predictor(
            train_numeric,
            test_numeric,
            target_column=config.target_column,
            task=task,
            training_source="real_train",
            random_state=config.random_state,
        )
    )
    rows.append(
        _evaluate_numeric_predictor(
            synthetic_numeric,
            test_numeric,
            target_column=config.target_column,
            task=task,
            training_source="synthetic_from_real_train",
            random_state=config.random_state,
        )
    )
    report = pd.DataFrame(rows)
    report.insert(0, "dataset", config.dataset_name)
    report.insert(1, "target_column", config.target_column)
    report.insert(2, "task", task)
    report["train_rows"] = [len(train_numeric), len(synthetic_numeric)]
    report["test_rows"] = len(test_numeric)
    return report


def predictive_numeric_view(dataframe: pd.DataFrame, sampler: DataFrameSampler, target_column: str) -> pd.DataFrame:
    """Return latent features without the target block, plus the original target."""
    latent = sampler.transform(dataframe)
    columns = [f"latent_{idx}" for idx in range(latent.shape[1])]
    view = pd.DataFrame(latent, columns=columns, index=dataframe.index)
    drop_columns = _latent_columns_for_source_column(sampler, target_column)
    view = view.drop(columns=drop_columns, errors="ignore")
    view[target_column] = dataframe[target_column].to_numpy()
    return view.reset_index(drop=True)


def _latent_columns_for_source_column(sampler: DataFrameSampler, source_column: str) -> list[str]:
    offset = 0
    if source_column in sampler.numeric_columns_:
        idx = sampler.numeric_columns_.index(source_column)
        return [f"latent_{idx}"]
    offset += len(sampler.numeric_columns_)
    for column in sampler.categorical_columns_:
        width = sampler._components_for_column(column)
        if column == source_column:
            return [f"latent_{idx}" for idx in range(offset, offset + width)]
        offset += width
    return []


def _evaluate_numeric_predictor(
    train_numeric: pd.DataFrame,
    test_numeric: pd.DataFrame,
    *,
    target_column: str,
    task: str,
    training_source: str,
    random_state: int,
) -> dict[str, Any]:
    x_train = train_numeric.drop(columns=[target_column])
    y_train = train_numeric[target_column]
    x_test = test_numeric.drop(columns=[target_column])
    y_test = test_numeric[target_column]

    model = _numeric_model(task, random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    row: dict[str, Any] = {"training_source": training_source}
    if task == "classification":
        row.update(
            {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                "roc_auc": _binary_roc_auc(model, x_test, y_test),
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
            }
        )
    else:
        row.update(
            {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "f1_weighted": np.nan,
                "roc_auc": np.nan,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
        )
    return row


def _numeric_model(task: str, *, random_state: int) -> Pipeline:
    estimator = (
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        if task == "classification"
        else RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def _stratification_target(target: pd.Series, task: str) -> pd.Series | None:
    if task != "classification":
        return None
    counts = target.value_counts(dropna=True)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return target


def _binary_roc_auc(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    if y_test.nunique(dropna=True) != 2:
        return np.nan
    classifier = model.named_steps["model"]
    if not hasattr(classifier, "predict_proba"):
        return np.nan
    probabilities = model.predict_proba(x_test)
    if probabilities.shape[1] != 2:
        return np.nan
    try:
        return float(roc_auc_score(y_test, probabilities[:, 1]))
    except ValueError:
        return np.nan
