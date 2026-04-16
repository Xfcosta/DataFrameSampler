from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .datasets import DatasetExperimentConfig


VECTORIZATION_RATIONALES: dict[str, dict[str, str]] = {
    "adult": {
        "workclass": "Employment sector is represented through age, education level, and weekly work hours because these describe career stage, qualification, and labour intensity.",
        "education": "Education category is represented through age, numeric education level, and work hours so nearby categories reflect ordered attainment and employment context.",
        "occupation": "Occupation is represented through age, education level, and work hours because these columns describe plausible career and workload context.",
        "marital_status": "Marital status is represented through age, education level, and work hours as broad life-stage and socioeconomic context.",
        "relationship": "Household relationship is represented through age, education level, and work hours because these help distinguish family and household roles.",
        "race": "Race is left on frequency encoding in the manual configuration because assigning a numeric helper geometry to this sensitive demographic category would require a stronger governance and modelling justification.",
        "sex": "Sex is directly encoded as a binary numeric column before sampling.",
        "native_country": "Native-country categories are represented through age, education level, and work hours as weak demographic and socioeconomic context; this is a convenience representation rather than geography.",
        "income": "Income is directly encoded as a binary target column before sampling.",
    },
    "titanic": {
        "sex": "Sex is directly encoded as a binary numeric column before sampling.",
        "embarked": "The short embarkation code is removed because embark_town carries the same information in a more readable form.",
        "class": "Passenger class is removed because pclass already preserves the ordered class information numerically.",
        "embark_town": "Embarkation town is represented through age, fare, and passenger class as coarse travel and ticket context.",
        "who": "The derived woman/child/man category is directly encoded on the ordered scale 0, 0.5, 1 before sampling.",
        "adult_male": "The boolean adult-male indicator is directly encoded as 0/1 before sampling.",
        "deck": "Deck is represented through survival, class, age, and fare because cabin location is most naturally related to ticket and passenger context.",
        "alive": "Alive is removed because survived already preserves the same survival label numerically.",
        "alone": "The boolean travelling-alone indicator is directly encoded as 0/1 before sampling.",
    },
    "breast_cancer": {
        "diagnosis": "Diagnosis is directly encoded as a binary benign/malignant target before sampling.",
    },
    "pima_diabetes": {
        "diabetes": "Diabetes status is directly encoded as a binary negative/positive target before sampling.",
    },
    "bank_marketing": {
        "job": "Occupation is represented through age, call duration, campaign contact counts, previous contacts, and economic rates as client lifecycle and campaign context.",
        "marital": "Marital status is represented through age and campaign/economic context as a broad demographic and contact-profile approximation.",
        "education": "Education is represented through age and campaign/economic context because these are the available numeric socioeconomic and contact variables.",
        "default": "Default status is directly encoded as a no/yes binary column before sampling.",
        "housing": "Housing-loan status is directly encoded as a no/yes binary column before sampling.",
        "loan": "Personal-loan status is directly encoded as a no/yes binary column before sampling.",
        "contact": "Contact channel is represented through age, campaign intensity, previous contacts, and macroeconomic context because these reflect campaign execution conditions.",
        "month": "Contact month is represented through campaign and macroeconomic variables because timing is related to the campaign period and economic context.",
        "day_of_week": "Contact weekday is represented through campaign and macroeconomic variables as a weak operational timing context.",
        "poutcome": "Previous campaign outcome is represented through contact history and macroeconomic variables because these summarize prior engagement and campaign conditions.",
        "subscribed": "Subscription outcome is directly encoded as a no/yes binary target before sampling.",
    },
    "heart_disease": {
        "sex": "Sex is directly encoded as a binary numeric column before sampling.",
        "chest_pain": "Chest-pain category is represented through age and cardiac measurements because these describe symptom and test context.",
        "fasting_blood_sugar": "Fasting-blood-sugar category is directly encoded as a binary threshold indicator before sampling.",
        "resting_ecg": "Resting ECG category is represented through age and cardiovascular measurements because these summarize related clinical state.",
        "exercise_angina": "Exercise-induced angina is directly encoded as a no/yes binary column before sampling.",
        "slope": "ST-segment slope is represented through exercise and cardiovascular measurements because it is part of the same stress-test context.",
        "thal": "Thal category is represented through age, cardiovascular measurements, oldpeak, and major vessels as available diagnostic-test context.",
        "heart_disease": "Heart-disease label is directly encoded as a binary absent/present target before sampling.",
    },
    "synthetic_correlated_helpers": {
        "helper_band": "The helper band is deliberately derived from the latent numeric structure summarized by spend, visit, and risk scores.",
        "segment": "Segment is derived from spend, visit, and risk scores, so these helpers expose the known categorical dependency structure.",
        "target": "The binary target is generated from spend, visit, and risk scores, so these helpers reflect the controlled data-generating rule.",
    },
    "synthetic_high_cardinality": {
        "region": "Region is generated from the latent numeric structure reflected in account value, tenure, and activity.",
        "sku_code": "SKU code is high-cardinality and tied to the same latent numeric structure, so account value, tenure, and activity are used as helper context.",
        "plan_tier": "Plan tier is derived partly from account value and account activity context.",
        "target": "The target depends on activity, tier, and tenure; the numeric helpers expose the controlled signal used to generate it.",
    },
    "synthetic_rare_categories": {
        "rare_signal": "Rare signal categories are evaluated against recency, frequency, and monetary context to test rare-category preservation.",
        "lifecycle": "Lifecycle is derived from recency, frequency, and monetary value, so these helpers encode its known dependency structure.",
        "target": "The target is generated from frequency, rare-category membership, recency, and noise; recency, frequency, and monetary summarize that context.",
    },
    "synthetic_sensitive_identifier": {
        "patient_id": "Patient IDs are unique identifiers; this helper configuration exists to expose memorization risk and should be replaced or excluded in governed workflows.",
        "condition": "Condition is represented through age, lab score, and visits as synthetic clinical context.",
        "ward": "Ward is represented through age, lab score, and visits as weak operational context.",
        "risk_flag": "Risk flag is generated from lab score, visits, and condition; age, lab score, and visits expose most of the controlled risk context.",
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
    vectorizing_columns = dict(sampler_config.get("vectorizing_columns_dict") or {})
    embedding_method = sampler_config.get("embedding_method", "mds")
    direct_mappings = config.direct_numeric_mappings or {}
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        dtype = str(dataframe[column].dtype)
        helpers = list(vectorizing_columns.get(column, []))
        helper_status = _helper_status(dataframe, helpers)
        direct_mapping = column in direct_mappings
        if direct_mapping:
            strategy = "direct_mapping"
            decision = "Column is converted with an explicit configured numeric mapping before binning."
            used_helpers = []
            effective_embedding = ""
            if helpers:
                decision += " Configured helper columns are not used because explicit mappings take precedence."
        elif is_numeric_dtype(dataframe[column]):
            strategy = "direct_numeric"
            decision = "Column is already numeric in the implementation and is discretized directly."
            used_helpers: list[str] = []
            effective_embedding = ""
            if helpers:
                decision += " Configured helper columns are not used for this column because numeric columns bypass categorical vectorization."
        elif helpers:
            strategy = "helper_embedding"
            used_helpers = helpers
            effective_embedding = _embedding_for_column(embedding_method, column)
            decision = (
                "Map this non-numeric column through the listed numeric helper columns, "
                f"then reduce the helper space to one dimension with {effective_embedding} before binning."
            )
        else:
            strategy = "frequency_encoding"
            used_helpers = []
            effective_embedding = ""
            decision = (
                "No semantic helper columns are configured, so categories are ordered by empirical frequency before binning."
            )
        rationale = VECTORIZATION_RATIONALES.get(config.dataset_name, {}).get(column, decision)

        rows.append(
            {
                "column": column,
                "dtype": dtype,
                "unique": int(dataframe[column].nunique(dropna=True)),
                "missing": int(dataframe[column].isna().sum()),
                "strategy": strategy,
                "embedding_method": effective_embedding,
                "helper_columns": ", ".join(used_helpers),
                "helper_status": helper_status,
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
        return str(embedding_method.get(column, "mds"))
    return str(embedding_method)


def _helper_status(dataframe: pd.DataFrame, helpers: list[str]) -> str:
    if not helpers:
        return ""
    missing = [column for column in helpers if column not in dataframe.columns]
    non_numeric = [
        column
        for column in helpers
        if column in dataframe.columns and not is_numeric_dtype(dataframe[column])
    ]
    if missing:
        return "missing helpers: " + ", ".join(missing)
    if non_numeric:
        return "non-numeric helpers: " + ", ".join(non_numeric)
    return "all helpers numeric"


def _mapping_summary(mapping: Mapping[Any, float]) -> str:
    if not mapping:
        return ""
    items = list(mapping.items())
    return ", ".join(f"{key!r}->{value:g}" for key, value in items[:8])
