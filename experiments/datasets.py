from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetExperimentConfig:
    """Reusable parameters for one dataset experiment workflow."""

    dataset_name: str
    title: str
    data_filename: str
    target_column: str | None
    manual_sampler_config: dict[str, Any]
    llm_assisted_config: dict[str, Any] | None = None
    working_sample_size: int | None = None
    n_generated: int = 1000
    random_state: int = 42


ADULT_VECTOR_COLUMNS = {
    "workclass": ["age", "education_num", "hours_per_week"],
    "education": ["age", "education_num", "hours_per_week"],
    "occupation": ["age", "education_num", "hours_per_week"],
    "marital_status": ["age", "education_num", "hours_per_week"],
    "relationship": ["age", "education_num", "hours_per_week"],
    "native_country": ["age", "education_num", "hours_per_week"],
}

ADULT_LLM_VECTOR_COLUMNS = {
    "workclass": ["age", "education_num", "hours_per_week"],
    "education": ["age", "education_num", "hours_per_week"],
    "marital_status": ["age", "education_num", "hours_per_week", "capital_gain"],
    "occupation": ["age", "education_num", "hours_per_week", "capital_gain"],
    "relationship": ["age", "education_num", "hours_per_week"],
    "race": ["age", "education_num", "hours_per_week"],
    "sex": ["age", "education_num", "hours_per_week"],
    "native_country": ["age", "education_num", "hours_per_week"],
    "income": ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"],
}

TITANIC_VECTOR_COLUMNS = {
    "sex": ["age", "fare", "pclass"],
    "class": ["age", "fare", "pclass"],
    "embark_town": ["age", "fare", "pclass"],
    "who": ["age", "fare", "pclass"],
    "adult_male": ["age", "fare", "pclass"],
    "alone": ["age", "fare", "pclass"],
}

TITANIC_LLM_VECTOR_COLUMNS = {
    "sex": ["survived", "pclass", "age", "fare"],
    "class": ["survived", "pclass", "age", "fare"],
    "who": ["survived", "pclass", "age", "fare"],
    "adult_male": ["survived", "pclass", "age", "fare"],
    "deck": ["survived", "pclass", "age", "fare"],
    "embark_town": ["survived", "pclass", "age", "fare"],
    "alive": ["pclass", "age", "fare"],
    "alone": ["survived", "pclass", "age", "fare"],
}


DATASET_CONFIGS: dict[str, DatasetExperimentConfig] = {
    "adult": DatasetExperimentConfig(
        dataset_name="adult",
        title="Adult Census Income",
        data_filename="adult.csv",
        target_column="income",
        working_sample_size=2500,
        manual_sampler_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "vectorizing_columns_dict": ADULT_VECTOR_COLUMNS,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 12,
            "n_neighbours": 8,
            "vectorizing_columns_dict": ADULT_LLM_VECTOR_COLUMNS,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "titanic": DatasetExperimentConfig(
        dataset_name="titanic",
        title="Titanic",
        data_filename="titanic.csv",
        target_column="survived",
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "vectorizing_columns_dict": TITANIC_VECTOR_COLUMNS,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "vectorizing_columns_dict": TITANIC_LLM_VECTOR_COLUMNS,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "synthetic_correlated_helpers": DatasetExperimentConfig(
        dataset_name="synthetic_correlated_helpers",
        title="Controlled correlated helpers",
        data_filename="synthetic_correlated_helpers.csv",
        target_column="target",
        n_generated=500,
        random_state=101,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "vectorizing_columns_dict": {
                "helper_band": ["spend_score", "visit_rate", "risk_score"],
                "segment": ["spend_score", "visit_rate", "risk_score"],
                "target": ["spend_score", "visit_rate", "risk_score"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "vectorizing_columns_dict": {
                "helper_band": ["spend_score", "visit_rate", "risk_score"],
                "segment": ["spend_score", "visit_rate", "risk_score"],
                "target": ["spend_score", "visit_rate", "risk_score"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "synthetic_high_cardinality": DatasetExperimentConfig(
        dataset_name="synthetic_high_cardinality",
        title="Controlled high cardinality",
        data_filename="synthetic_high_cardinality.csv",
        target_column="target",
        n_generated=500,
        random_state=102,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "vectorizing_columns_dict": {
                "region": ["account_value", "tenure_months", "activity_score"],
                "sku_code": ["account_value", "tenure_months", "activity_score"],
                "plan_tier": ["account_value", "tenure_months", "activity_score"],
                "target": ["account_value", "tenure_months", "activity_score"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "vectorizing_columns_dict": {
                "region": ["account_value", "tenure_months", "activity_score"],
                "sku_code": ["account_value", "tenure_months", "activity_score"],
                "plan_tier": ["account_value", "tenure_months", "activity_score"],
                "target": ["account_value", "tenure_months", "activity_score"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "synthetic_rare_categories": DatasetExperimentConfig(
        dataset_name="synthetic_rare_categories",
        title="Controlled rare categories",
        data_filename="synthetic_rare_categories.csv",
        target_column="target",
        n_generated=500,
        random_state=103,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "vectorizing_columns_dict": {
                "rare_signal": ["recency_days", "frequency", "monetary"],
                "lifecycle": ["recency_days", "frequency", "monetary"],
                "target": ["recency_days", "frequency", "monetary"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "vectorizing_columns_dict": {
                "rare_signal": ["recency_days", "frequency", "monetary"],
                "lifecycle": ["recency_days", "frequency", "monetary"],
                "target": ["recency_days", "frequency", "monetary"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "synthetic_sensitive_identifier": DatasetExperimentConfig(
        dataset_name="synthetic_sensitive_identifier",
        title="Controlled sensitive identifier",
        data_filename="synthetic_sensitive_identifier.csv",
        target_column="risk_flag",
        n_generated=500,
        random_state=104,
        manual_sampler_config={
            "n_bins": 10,
            "n_neighbours": 6,
            "vectorizing_columns_dict": {
                "patient_id": ["age", "lab_score", "visits"],
                "condition": ["age", "lab_score", "visits"],
                "ward": ["age", "lab_score", "visits"],
                "risk_flag": ["age", "lab_score", "visits"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "vectorizing_columns_dict": {
                "patient_id": ["age", "lab_score", "visits"],
                "condition": ["age", "lab_score", "visits"],
                "ward": ["age", "lab_score", "visits"],
                "risk_flag": ["age", "lab_score", "visits"],
            },
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
}
