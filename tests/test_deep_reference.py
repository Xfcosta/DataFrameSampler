import builtins
import sys
import types

import pandas as pd
import pytest

from experiments.baselines import SdvCtganBaseline
from experiments.datasets import DatasetExperimentConfig
from experiments.deep_reference import run_deep_reference_comparison_for_config
from experiments.make_tables import DatasetTableMetadata, generate_all_tables, write_deep_reference_table


def make_deep_dataframe(n_rows=24):
    return pd.DataFrame(
        {
            "age": list(range(20, 20 + n_rows)),
            "hours": [35 + idx % 12 for idx in range(n_rows)],
            "education": [["HS", "Bachelors", "Masters"][idx % 3] for idx in range(n_rows)],
            "income": [idx % 2 for idx in range(n_rows)],
        }
    )


def make_deep_config(**overrides):
    config = DatasetExperimentConfig(
        dataset_name="adult",
        title="Adult",
        data_filename="adult.csv",
        target_column="income",
        sampler_config={"n_neighbours": 2, "knn_backend": "sklearn"},
        n_generated=8,
        random_state=3,
    )
    return DatasetExperimentConfig(**{**config.__dict__, **overrides})


def test_sdv_ctgan_baseline_has_actionable_optional_dependency_message(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sdv"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="deep-baselines"):
        SdvCtganBaseline().fit(make_deep_dataframe())


def test_sdv_ctgan_baseline_uses_current_sdv_single_table_api(monkeypatch):
    calls = {}

    class FakeMetadata:
        @staticmethod
        def detect_from_dataframe(*, data, table_name):
            calls["metadata_data"] = data.copy()
            calls["table_name"] = table_name
            return "metadata"

    class FakeCtganSynthesizer:
        def __init__(self, metadata, **kwargs):
            calls["metadata"] = metadata
            calls["kwargs"] = kwargs

        def fit(self, data):
            calls["fit_data"] = data.copy()

        def sample(self, *, num_rows):
            calls["num_rows"] = num_rows
            return pd.DataFrame(
                {
                    "income": [1] * num_rows,
                    "education": ["HS"] * num_rows,
                    "hours": [40] * num_rows,
                    "age": [42] * num_rows,
                    "extra": ["ignored"] * num_rows,
                }
            )

    sdv_module = types.ModuleType("sdv")
    metadata_module = types.ModuleType("sdv.metadata")
    metadata_module.Metadata = FakeMetadata
    single_table_module = types.ModuleType("sdv.single_table")
    single_table_module.CTGANSynthesizer = FakeCtganSynthesizer
    monkeypatch.setitem(sys.modules, "sdv", sdv_module)
    monkeypatch.setitem(sys.modules, "sdv.metadata", metadata_module)
    monkeypatch.setitem(sys.modules, "sdv.single_table", single_table_module)

    df = make_deep_dataframe()
    baseline = SdvCtganBaseline(synthesizer_kwargs={"epochs": 2}).fit(df)
    sample = baseline.sample(5)

    assert calls["table_name"] == "table"
    assert calls["metadata"] == "metadata"
    assert calls["kwargs"]["epochs"] == 2
    assert calls["kwargs"]["enable_gpu"] is False
    assert calls["fit_data"].equals(df)
    assert calls["num_rows"] == 5
    assert list(sample.columns) == list(df.columns)


def test_run_deep_reference_comparison_writes_expected_outputs(tmp_path):
    class FakeBaseline:
        def fit(self, dataframe):
            self.dataframe_ = dataframe.reset_index(drop=True)
            return self

        def sample(self, n_samples):
            return self.dataframe_.sample(n=n_samples, replace=True, random_state=1).reset_index(drop=True)

    report = run_deep_reference_comparison_for_config(
        make_deep_config(),
        make_deep_dataframe(),
        results_dir=tmp_path,
        n_samples=8,
        max_train_rows=16,
        baseline_factory=FakeBaseline,
    )

    assert list(report["method"]) == ["ctgan"]
    assert {"distribution_similarity_score", "discrimination_accuracy", "utility_lift"}.issubset(report.columns)
    assert (tmp_path / "adult_ctgan_generated.csv").exists()
    assert (tmp_path / "adult_deep_reference_comparison.csv").exists()


def test_deep_reference_is_adult_only(tmp_path):
    report = run_deep_reference_comparison_for_config(
        make_deep_config(dataset_name="toy"),
        make_deep_dataframe(),
        results_dir=tmp_path,
    )

    assert report.empty
    assert not (tmp_path / "toy_deep_reference_comparison.csv").exists()


def test_deep_reference_table_helpers_write_outputs(tmp_path):
    rows = pd.DataFrame(
        {
            "dataset": ["adult"],
            "method": ["ctgan"],
            "method_label": ["CTGAN"],
            "distribution_similarity_score": [0.8],
            "discrimination_accuracy": [0.6],
            "utility_lift": [0.01],
            "fit_seconds": [2.0],
            "sample_seconds": [0.5],
            "peak_memory_mb": [3.0],
        }
    )

    assert write_deep_reference_table(rows, tables_dir=tmp_path).exists()


def test_generate_all_tables_includes_deep_reference_when_csv_exists(tmp_path):
    results_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    results_dir.mkdir()
    processed_dir.mkdir()
    make_deep_dataframe(8).to_csv(processed_dir / "adult.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["adult"],
            "method": ["dataframe_sampler"],
            "n_real": [8],
            "n_synthetic": [8],
            "numeric_ks_statistic": [0.0],
            "categorical_total_variation": [0.0],
            "mean_abs_association_difference": [0.0],
            "numeric_histogram_overlap": [1.0],
            "categorical_coverage": [1.0],
            "rare_category_preservation": [1.0],
            "nn_distance_ratio": [1.0],
            "nn_suspiciously_close_rate": [0.0],
            "discrimination_accuracy": [0.5],
            "discrimination_privacy_score": [1.0],
            "utility_task": ["classification"],
            "utility_real_score": [1.0],
            "utility_augmented_score": [1.0],
            "utility_lift": [0.0],
            "distribution_histogram_overlap": [1.0],
            "distribution_numeric_kl": [0.0],
            "distribution_categorical_jsd": [0.0],
            "distribution_similarity_score": [1.0],
            "fit_seconds": [0.1],
            "sample_seconds": [0.1],
            "fit_peak_memory_mb": [0.0],
            "sample_peak_memory_mb": [0.0],
            "peak_memory_mb": [0.0],
        }
    ).to_csv(results_dir / "adult_baseline_comparison.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["adult"],
            "method": ["ctgan"],
            "method_label": ["CTGAN"],
            "distribution_similarity_score": [0.8],
            "discrimination_accuracy": [0.6],
            "utility_lift": [0.01],
            "fit_seconds": [2.0],
            "sample_seconds": [0.5],
            "peak_memory_mb": [3.0],
        }
    ).to_csv(results_dir / "adult_deep_reference_comparison.csv", index=False)

    outputs = generate_all_tables(
        results_dir=results_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        dataset_metadata=[DatasetTableMetadata("adult", "Adult", "Census", "None", "Test")],
    )

    assert tables_dir / "deep_reference_comparison.tex" in outputs
