from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .datasets import DatasetExperimentConfig


DEFAULT_HIGH_CARDINALITY_FRACTION = 0.3
DEFAULT_HIGH_CARDINALITY_UNIQUE = 50


NCA_RATIONALES: dict[str, dict[str, str]] = {
    "adult": {
        "sex": "Binary columns are categorical targets in DataFrameSampler 2.0 and receive a supervised NCA latent block.",
        "income": "The binary income label is treated as a categorical target and receives a supervised NCA latent block.",
    },
    "synthetic_sensitive_identifier": {
        "patient_id": "High-cardinality identifiers are warned about and still used unless the user preprocesses them.",
    },
}


def vectorization_plan(
    dataframe: pd.DataFrame,
    config: DatasetExperimentConfig,
    *,
    config_key: str = "manual_sampler_config",
) -> pd.DataFrame:
    """Return a human-readable plan for DataFrameSampler 2.0 latent construction."""
    sampler_config = _sampler_config(config, config_key)
    n_components = sampler_config.get("n_components", 2)
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        dtype = str(series.dtype)
        unique = int(series.nunique(dropna=True))
        alphabet_fraction = unique / len(dataframe) if len(dataframe) else 0.0
        high_cardinality = _is_high_cardinality(unique, len(dataframe))

        if _is_numeric_non_binary(series):
            strategy = "standardized_numeric"
            latent_components = 1
            decision = "Non-binary numeric column is median-imputed and standardized into the latent matrix."
        else:
            strategy = "categorical_nca"
            latent_components = _components_for_column(n_components, column)
            decision = (
                "Column is treated as categorical, one-hot encoded for context, "
                "and represented by a supervised NCA latent block."
            )
            if high_cardinality:
                decision += " It is high-cardinality, so the sampler warns but proceeds."

        rationale = NCA_RATIONALES.get(config.dataset_name, {}).get(column, decision)
        rows.append(
            {
                "column": column,
                "dtype": dtype,
                "unique": unique,
                "missing": int(series.isna().sum()),
                "alphabet_fraction": alphabet_fraction,
                "strategy": strategy,
                "latent_components": latent_components,
                "high_cardinality_warning": bool(high_cardinality and strategy == "categorical_nca"),
                "rationale": rationale,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def preprocessing_plan(config: DatasetExperimentConfig) -> pd.DataFrame:
    """Return configured column drops before DataFrameSampler fitting."""
    rows: list[dict[str, Any]] = []
    for column in config.drop_columns:
        rows.append(
            {
                "column": column,
                "action": "drop",
                "mapping": "",
                "reason": "Redundant alias or duplicate target representation removed before sampling.",
            }
        )
    return pd.DataFrame(rows, columns=["column", "action", "mapping", "reason"])


def columns_requiring_vectorization(dataframe: pd.DataFrame) -> list[str]:
    """Return columns treated as categorical NCA targets by DataFrameSampler 2.0."""
    return [column for column in dataframe.columns if not _is_numeric_non_binary(dataframe[column])]


def _sampler_config(config: DatasetExperimentConfig, key: str) -> Mapping[str, Any]:
    if key != "manual_sampler_config":
        raise ValueError("config_key must be 'manual_sampler_config'.")
    return config.manual_sampler_config


def _components_for_column(n_components: Any, column: str) -> int:
    if isinstance(n_components, Mapping):
        return int(n_components.get(column, 2))
    return int(n_components)


def _is_numeric_non_binary(series: pd.Series) -> bool:
    if is_bool_dtype(series) or not is_numeric_dtype(series):
        return False
    return len(series.dropna().unique()) > 2


def _is_high_cardinality(unique: int, row_count: int) -> bool:
    limit = max(DEFAULT_HIGH_CARDINALITY_UNIQUE, int(row_count * DEFAULT_HIGH_CARDINALITY_FRACTION))
    return unique > limit
