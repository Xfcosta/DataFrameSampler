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
    drop_columns: tuple[str, ...] = ()
    direct_numeric_mappings: dict[str, dict[Any, float]] | None = None


BINARY_NO_YES_MAPPING = {"no": 0.0, "yes": 1.0}

ADULT_DIRECT_NUMERIC_MAPPINGS = {
    "sex": {"Female": 0.0, "Male": 1.0},
    "income": {"<=50K": 0.0, ">50K": 1.0},
}

TITANIC_DIRECT_NUMERIC_MAPPINGS = {
    "sex": {"female": 0.0, "male": 1.0},
    "who": {"woman": 0.0, "child": 0.5, "man": 1.0},
    "adult_male": {False: 0.0, True: 1.0, "False": 0.0, "True": 1.0},
    "alone": {False: 0.0, True: 1.0, "False": 0.0, "True": 1.0},
}

BREAST_CANCER_DIRECT_NUMERIC_MAPPINGS = {
    "diagnosis": {"benign": 0.0, "malignant": 1.0},
}

PIMA_DIABETES_DIRECT_NUMERIC_MAPPINGS = {
    "diabetes": {"negative": 0.0, "positive": 1.0},
}

BANK_MARKETING_DIRECT_NUMERIC_MAPPINGS = {
    "default": BINARY_NO_YES_MAPPING,
    "housing": BINARY_NO_YES_MAPPING,
    "loan": BINARY_NO_YES_MAPPING,
    "subscribed": BINARY_NO_YES_MAPPING,
}

HEART_DISEASE_DIRECT_NUMERIC_MAPPINGS = {
    "sex": {"female": 0.0, "male": 1.0},
    "fasting_blood_sugar": {"not_above_120": 0.0, "above_120": 1.0},
    "exercise_angina": BINARY_NO_YES_MAPPING,
    "heart_disease": {"absent": 0.0, "present": 1.0},
}


DATASET_CONFIGS: dict[str, DatasetExperimentConfig] = {
    "adult": DatasetExperimentConfig(
        dataset_name="adult",
        title="Adult Census Income",
        data_filename="adult.csv",
        target_column="income",
        working_sample_size=2500,
        direct_numeric_mappings=ADULT_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 12,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "titanic": DatasetExperimentConfig(
        dataset_name="titanic",
        title="Titanic",
        data_filename="titanic.csv",
        target_column="survived",
        drop_columns=("class", "embarked", "alive"),
        direct_numeric_mappings=TITANIC_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "breast_cancer": DatasetExperimentConfig(
        dataset_name="breast_cancer",
        title="Wisconsin Diagnostic Breast Cancer",
        data_filename="breast_cancer.csv",
        target_column="diagnosis",
        n_generated=569,
        random_state=52,
        direct_numeric_mappings=BREAST_CANCER_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "pima_diabetes": DatasetExperimentConfig(
        dataset_name="pima_diabetes",
        title="Pima Indians Diabetes",
        data_filename="pima_diabetes.csv",
        target_column="diabetes",
        n_generated=768,
        random_state=53,
        direct_numeric_mappings=PIMA_DIABETES_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "bank_marketing": DatasetExperimentConfig(
        dataset_name="bank_marketing",
        title="Bank Marketing",
        data_filename="bank_marketing.csv",
        target_column="subscribed",
        working_sample_size=3000,
        n_generated=1000,
        random_state=54,
        direct_numeric_mappings=BANK_MARKETING_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 12,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
    "heart_disease": DatasetExperimentConfig(
        dataset_name="heart_disease",
        title="UCI Heart Disease",
        data_filename="heart_disease.csv",
        target_column="heart_disease",
        n_generated=303,
        random_state=55,
        direct_numeric_mappings=HEART_DISEASE_DIRECT_NUMERIC_MAPPINGS,
        manual_sampler_config={
            "n_bins": 8,
            "n_neighbours": 6,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
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
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
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
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
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
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
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
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
        llm_assisted_config={
            "n_bins": 10,
            "n_neighbours": 8,
            "embedding_method": "pca",
            "knn_backend": "sklearn",
        },
    ),
}
