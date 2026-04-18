from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SAMPLER_CONFIG = {
    "n_components": 1,
    "n_iterations": 0,
    "nca_fit_sample_size": 0.5,
    "lambda_": 0.25,
}


@dataclass(frozen=True)
class DatasetExperimentConfig:
    """Reusable parameters for one dataset experiment workflow."""

    dataset_name: str
    title: str
    data_filename: str
    target_column: str | None
    sampler_config: dict[str, Any]
    working_sample_size: int | None = None
    n_generated: int = 1000
    random_state: int = 42
    drop_columns: tuple[str, ...] = ()


DATASET_CONFIGS: dict[str, DatasetExperimentConfig] = {
    "adult": DatasetExperimentConfig(
        dataset_name="adult",
        title="Adult Census Income",
        data_filename="adult.csv",
        target_column="income",
        drop_columns=("fnlwgt",),
        working_sample_size=2500,
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 8,
            "knn_backend": "sklearn",
        },
    ),
    "titanic": DatasetExperimentConfig(
        dataset_name="titanic",
        title="Titanic",
        data_filename="titanic.csv",
        target_column="survived",
        drop_columns=("class", "embarked", "alive"),
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 8,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
            "knn_backend": "sklearn",
        },
    ),
    "covertype": DatasetExperimentConfig(
        dataset_name="covertype",
        title="Forest Covertype",
        data_filename="covertype.csv",
        target_column="cover_type",
        working_sample_size=1500,
        n_generated=1000,
        random_state=56,
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 8,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
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
        sampler_config={
            **DEFAULT_SAMPLER_CONFIG,
            "n_neighbours": 6,
            "knn_backend": "sklearn",
        },
    ),
}
