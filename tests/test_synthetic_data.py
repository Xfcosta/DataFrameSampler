import pandas as pd

from experiments.datasets import DATASET_CONFIGS
from experiments.synthetic_data import SYNTHETIC_DATASETS, materialize_synthetic_datasets


def test_materialize_synthetic_datasets_writes_all_controlled_regimes(tmp_path):
    paths = materialize_synthetic_datasets(tmp_path)

    assert set(paths) == {spec.key for spec in SYNTHETIC_DATASETS}
    for spec in SYNTHETIC_DATASETS:
        dataframe = pd.read_csv(paths[spec.key])
        assert len(dataframe) == spec.rows
        assert spec.target_column in dataframe.columns
        for column in spec.sensitive_columns:
            assert column in dataframe.columns


def test_synthetic_dataset_configs_cover_all_controlled_regimes():
    for spec in SYNTHETIC_DATASETS:
        config = DATASET_CONFIGS[spec.key]
        assert config.data_filename == f"{spec.key}.csv"
        assert config.target_column == spec.target_column
        assert config.sampler_config["knn_backend"] == "sklearn"


def test_sensitive_identifier_dataset_has_unique_source_ids(tmp_path):
    paths = materialize_synthetic_datasets(tmp_path)
    dataframe = pd.read_csv(paths["synthetic_sensitive_identifier"])

    assert dataframe["patient_id"].is_unique
    assert dataframe["patient_id"].str.startswith("PAT-").all()
