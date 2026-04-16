from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .datasets import DatasetExperimentConfig


DEFAULT_MAX_CATEGORICAL_FRACTION = 0.3
DEFAULT_MAX_CATEGORICAL_UNIQUE = 50


VECTORIZATION_RATIONALES: dict[str, dict[str, str]] = {
    "adult": {
        "workclass": "Employment sector has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "education": "Education has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "occupation": "Occupation has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "marital_status": "Marital status has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "relationship": "Household relationship has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "race": "Race has a compact alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "sex": "Sex is directly encoded as a binary numeric column before sampling.",
        "native_country": "Native country has a moderate alphabet, so it is one-hot encoded and embedded to one numeric coordinate.",
        "income": "Income is directly encoded as a binary target column before sampling.",
    },
    "synthetic_sensitive_identifier": {
        "patient_id": "Patient IDs are unique identifiers and should be discarded by the high-cardinality rule.",
    },
}


def vectorization_plan(
    dataframe: pd.DataFrame,
    config: DatasetExperimentConfig,
    *,
    config_key: str = "manual_sampler_config",
) -> pd.DataFrame:
    """Return a human-readable plan for dataframe column vectorization."""
    sampler_config = _sampler_config(config, config_key)
    embedding_method = sampler_config.get("embedding_method", "pca")
    max_categorical_fraction = sampler_config.get("max_categorical_fraction", DEFAULT_MAX_CATEGORICAL_FRACTION)
    max_categorical_unique = sampler_config.get("max_categorical_unique", DEFAULT_MAX_CATEGORICAL_UNIQUE)
    direct_mappings = config.direct_numeric_mappings or {}
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        dtype = str(dataframe[column].dtype)
        unique = int(dataframe[column].nunique(dropna=True))
        alphabet_fraction = unique / len(dataframe) if len(dataframe) else 0.0
        direct_mapping = column in direct_mappings

        if direct_mapping:
            strategy = "direct_mapping"
            effective_embedding = ""
            decision = "Column is converted with an explicit configured numeric mapping before binning."
        elif is_numeric_dtype(dataframe[column]):
            strategy = "direct_numeric"
            effective_embedding = ""
            decision = "Column is already numeric in the implementation and is discretized directly."
        elif _is_high_cardinality(unique, len(dataframe), max_categorical_fraction, max_categorical_unique):
            strategy = "drop_high_cardinality"
            effective_embedding = ""
            decision = "Non-numeric alphabet is large relative to row count, so the column is discarded before sampling."
        elif unique == 2:
            strategy = "binary_mapping"
            effective_embedding = ""
            decision = "Binary non-numeric column is mapped to 0/1 before binning."
        else:
            strategy = "one_hot_embedding"
            effective_embedding = _embedding_for_column(embedding_method, column)
            decision = (
                "Non-numeric column has a manageable alphabet, so categories are one-hot encoded "
                f"and reduced to one dimension with {effective_embedding} before binning."
            )

        rationale = VECTORIZATION_RATIONALES.get(config.dataset_name, {}).get(column, decision)
        rows.append(
            {
                "column": column,
                "dtype": dtype,
                "unique": unique,
                "missing": int(dataframe[column].isna().sum()),
                "alphabet_fraction": alphabet_fraction,
                "strategy": strategy,
                "embedding_method": effective_embedding,
                "direct_mapping": _mapping_summary(direct_mappings.get(column, {})),
                "rationale": rationale,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def preprocessing_plan(config: DatasetExperimentConfig) -> pd.DataFrame:
    """Return configured column drops and direct numeric encodings."""
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
    for column, mapping in (config.direct_numeric_mappings or {}).items():
        rows.append(
            {
                "column": column,
                "action": "direct_numeric_mapping",
                "mapping": _mapping_summary(mapping),
                "reason": "Column has an explicit binary, ordinal, or human-defined numeric scale.",
            }
        )
    return pd.DataFrame(rows, columns=["column", "action", "mapping", "reason"])


def columns_requiring_vectorization(dataframe: pd.DataFrame) -> list[str]:
    """Return columns that are non-numeric under the package vectorizer rules."""
    return [column for column in dataframe.columns if not is_numeric_dtype(dataframe[column])]


def _sampler_config(config: DatasetExperimentConfig, key: str) -> Mapping[str, Any]:
    if key == "manual_sampler_config":
        return config.manual_sampler_config
    if key == "llm_assisted_config":
        return config.llm_assisted_config or {}
    raise ValueError("config_key must be 'manual_sampler_config' or 'llm_assisted_config'.")


def _embedding_for_column(embedding_method: Any, column: str) -> str:
    if isinstance(embedding_method, Mapping):
        return str(embedding_method.get(column, "pca"))
    return str(embedding_method)


def _is_high_cardinality(unique: int, row_count: int, fraction: float, max_unique: int) -> bool:
    limit = max(max_unique, int(row_count * fraction))
    return unique > limit


def _mapping_summary(mapping: Mapping[Any, float]) -> str:
    if not mapping:
        return ""
    items = list(mapping.items())
    return ", ".join(f"{key!r}->{value:g}" for key, value in items[:8])
