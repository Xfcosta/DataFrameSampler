from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


class RowBootstrapBaseline:
    """Sample complete source rows with replacement."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataframe: pd.DataFrame, target_column: str | None = None):
        self.columns_ = list(dataframe.columns)
        self.dataframe_ = dataframe.reset_index(drop=True).copy()
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        indexes = self.rng.integers(0, len(self.dataframe_), size=n_samples)
        return self.dataframe_.iloc[indexes].reset_index(drop=True).copy()


class IndependentColumnBaseline:
    """Sample each column independently from its empirical values."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataframe: pd.DataFrame, target_column: str | None = None):
        self.columns_ = list(dataframe.columns)
        self.values_ = {column: dataframe[column].to_numpy(copy=True) for column in self.columns_}
        self.dtypes_ = dataframe.dtypes.to_dict()
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        data = {}
        for column in self.columns_:
            values = self.values_[column]
            sampled = values[self.rng.integers(0, len(values), size=n_samples)]
            data[column] = sampled
        return _restore_dtypes(pd.DataFrame(data, columns=self.columns_), self.dtypes_)


class StratifiedColumnBaseline:
    """Sample target values empirically, then sample other columns within target strata."""

    def __init__(self, target_column: str, random_state: int | None = None):
        self.target_column = target_column
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataframe: pd.DataFrame, target_column: str | None = None):
        target = target_column or self.target_column
        if target not in dataframe.columns:
            raise ValueError(f"Unknown target column: {target}")
        self.target_column_ = target
        self.columns_ = list(dataframe.columns)
        self.dtypes_ = dataframe.dtypes.to_dict()
        self.target_values_ = dataframe[target].dropna().to_numpy(copy=True)
        self.target_probabilities_ = dataframe[target].value_counts(normalize=True, dropna=False)
        self.global_values_ = {column: dataframe[column].to_numpy(copy=True) for column in self.columns_}
        self.strata_ = {
            target_value: group.reset_index(drop=True)
            for target_value, group in dataframe.groupby(target, dropna=False)
        }
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        target_levels = self.target_probabilities_.index.to_numpy()
        probabilities = self.target_probabilities_.to_numpy(dtype=float)
        sampled_targets = self.rng.choice(target_levels, size=n_samples, replace=True, p=probabilities)
        rows = []
        for target_value in sampled_targets:
            row = {self.target_column_: target_value}
            stratum = self.strata_.get(target_value)
            for column in self.columns_:
                if column == self.target_column_:
                    continue
                values = (
                    stratum[column].to_numpy(copy=True)
                    if stratum is not None and len(stratum) > 0
                    else self.global_values_[column]
                )
                row[column] = values[self.rng.integers(0, len(values))]
            rows.append(row)
        return _restore_dtypes(pd.DataFrame(rows, columns=self.columns_), self.dtypes_)


class GaussianCopulaEmpiricalBaseline:
    """Gaussian-copula numeric sampler plus empirical categorical sampling."""

    def __init__(self, random_state: int | None = None, regularization: float = 1e-6):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.regularization = regularization

    def fit(self, dataframe: pd.DataFrame, target_column: str | None = None):
        self.columns_ = list(dataframe.columns)
        self.dtypes_ = dataframe.dtypes.to_dict()
        self.numeric_columns_ = list(dataframe.select_dtypes(include="number").columns)
        self.categorical_columns_ = [column for column in self.columns_ if column not in self.numeric_columns_]
        self.numeric_values_ = {
            column: pd.to_numeric(dataframe[column], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
            for column in self.numeric_columns_
        }
        self.categorical_values_ = {
            column: dataframe[column].to_numpy(copy=True)
            for column in self.categorical_columns_
        }
        self.numeric_missing_rates_ = {column: dataframe[column].isna().mean() for column in self.numeric_columns_}

        if self.numeric_columns_:
            z_columns = []
            for column in self.numeric_columns_:
                series = pd.to_numeric(dataframe[column], errors="coerce")
                ranks = series.rank(method="average", na_option="keep")
                probs = (ranks - 0.5) / series.notna().sum()
                z = probs.map(_normal_ppf).astype(float)
                z_columns.append(z.fillna(0.0).to_numpy())
            z_matrix = np.column_stack(z_columns)
            self.mean_ = z_matrix.mean(axis=0)
            covariance = np.cov(z_matrix, rowvar=False)
            if covariance.ndim == 0:
                covariance = np.array([[float(covariance)]])
            self.covariance_ = covariance + np.eye(len(self.numeric_columns_)) * self.regularization
        else:
            self.mean_ = np.array([])
            self.covariance_ = np.empty((0, 0))
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        data: dict[str, Any] = {}
        if self.numeric_columns_:
            z = self.rng.multivariate_normal(self.mean_, self.covariance_, size=n_samples)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            for idx, column in enumerate(self.numeric_columns_):
                values = self.numeric_values_[column]
                if len(values) == 0:
                    sampled = np.repeat(np.nan, n_samples)
                else:
                    probs = _normal_cdf(z[:, idx])
                    sampled = np.quantile(values, probs)
                    missing_mask = self.rng.random(n_samples) < self.numeric_missing_rates_[column]
                    sampled[missing_mask] = np.nan
                data[column] = sampled

        for column in self.categorical_columns_:
            values = self.categorical_values_[column]
            data[column] = values[self.rng.integers(0, len(values), size=n_samples)]

        return _restore_dtypes(pd.DataFrame(data, columns=self.columns_), self.dtypes_)


class SmoteNcBaseline:
    """Optional SMOTENC baseline for supervised tabular augmentation."""

    def __init__(
        self,
        target_column: str,
        random_state: int | None = None,
        sampling_strategy: str | float | dict = "auto",
        **kwargs,
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.kwargs = kwargs

    def fit(self, dataframe: pd.DataFrame, target_column: str | None = None):
        try:
            from imblearn.over_sampling import SMOTENC
        except ImportError as exc:
            raise ImportError(
                "SMOTENC baseline requires imbalanced-learn. Install it with: pip install imbalanced-learn"
            ) from exc

        target = target_column or self.target_column
        if target not in dataframe.columns:
            raise ValueError(f"Unknown target column: {target}")
        self.target_column_ = target
        self.columns_ = list(dataframe.columns)
        self.feature_columns_ = [column for column in self.columns_ if column != target]
        self.dtypes_ = dataframe.dtypes.to_dict()

        X = dataframe[self.feature_columns_].copy()
        y = dataframe[target].copy()
        self.category_maps_ = {}
        self.inverse_category_maps_ = {}
        categorical_features = []
        for idx, column in enumerate(self.feature_columns_):
            if not pd.api.types.is_numeric_dtype(X[column]):
                categorical_features.append(idx)
                categories = pd.Series(X[column].astype("string").fillna("__MISSING__")).unique()
                mapping = {value: code for code, value in enumerate(categories)}
                inverse = {code: value for value, code in mapping.items()}
                self.category_maps_[column] = mapping
                self.inverse_category_maps_[column] = inverse
                X[column] = X[column].astype("string").fillna("__MISSING__").map(mapping)
            else:
                X[column] = pd.to_numeric(X[column], errors="coerce").fillna(X[column].median())

        self.smote_ = SMOTENC(
            categorical_features=categorical_features,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            **self.kwargs,
        )
        X_resampled, y_resampled = self.smote_.fit_resample(X, y)
        self.resampled_ = pd.DataFrame(X_resampled, columns=self.feature_columns_)
        self.resampled_[target] = y_resampled
        for column, inverse in self.inverse_category_maps_.items():
            self.resampled_[column] = self.resampled_[column].round().astype(int).map(inverse).replace("__MISSING__", pd.NA)
        self.resampled_ = self.resampled_[self.columns_]
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        indexes = rng.integers(0, len(self.resampled_), size=n_samples)
        return _restore_dtypes(self.resampled_.iloc[indexes].reset_index(drop=True).copy(), self.dtypes_)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    estimator: Any


def simple_baselines(target_column: str | None = None, random_state: int | None = None) -> list[BaselineSpec]:
    baselines = [
        BaselineSpec("row_bootstrap", RowBootstrapBaseline(random_state=random_state)),
        BaselineSpec("independent_columns", IndependentColumnBaseline(random_state=random_state)),
        BaselineSpec("gaussian_copula_empirical", GaussianCopulaEmpiricalBaseline(random_state=random_state)),
    ]
    if target_column is not None:
        baselines.append(
            BaselineSpec("stratified_columns", StratifiedColumnBaseline(target_column, random_state=random_state))
        )
    return baselines


def _restore_dtypes(dataframe: pd.DataFrame, dtypes: Mapping[str, Any]) -> pd.DataFrame:
    restored = dataframe.copy()
    for column, dtype in dtypes.items():
        if column not in restored.columns:
            continue
        try:
            if pd.api.types.is_integer_dtype(dtype) and restored[column].isna().any():
                continue
            restored[column] = restored[column].astype(dtype)
        except (TypeError, ValueError):
            pass
    return restored


def _normal_ppf(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1 - 1e-6)
    return NormalDist().inv_cdf(clipped)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    normal = NormalDist()
    return np.array([normal.cdf(float(value)) for value in values])
