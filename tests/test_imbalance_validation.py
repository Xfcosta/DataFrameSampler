import sys
import types
from dataclasses import replace

import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.imbalance_validation import (
    imbalance_validation_report,
    run_imbalance_validation_for_config,
)
from experiments.make_tables import write_imbalance_validation_table


def make_imbalance_dataframe(n_majority=24, n_minority=8):
    rows = []
    for idx in range(n_majority):
        rows.append(
            {
                "age": 35 + idx % 10,
                "hours": 35 + idx % 8,
                "segment": "major_a" if idx % 2 else "major_b",
                "target": "negative",
            }
        )
    for idx in range(n_minority):
        rows.append(
            {
                "age": 55 + idx % 7,
                "hours": 20 + idx % 6,
                "segment": "minor_a" if idx % 2 else "minor_b",
                "target": "positive",
            }
        )
    return pd.DataFrame(rows)


def make_imbalance_config(**overrides):
    config = DatasetExperimentConfig(
        dataset_name="adult",
        title="Toy imbalance",
        data_filename="toy.csv",
        target_column="target",
        working_sample_size=None,
        n_generated=12,
        random_state=3,
        sampler_config={
            "n_neighbours": 3,
            "knn_backend": "sklearn",
            "n_iterations": 0,
        },
    )
    return replace(config, **overrides)


def test_imbalance_validation_report_includes_rebalancing_methods():
    dataframe = make_imbalance_dataframe()
    train = dataframe.reset_index(drop=True)
    test = dataframe.reset_index(drop=True)
    minority = "positive"
    majority = "negative"

    report = imbalance_validation_report(
        train=train,
        test=test,
        dataset_name="adult",
        target_column="target",
        minority_class=minority,
        majority_class=majority,
        n_to_generate=16,
        sampler_config={"n_neighbours": 2, "knn_backend": "sklearn", "n_iterations": 0, "random_state": 4},
        random_state=4,
    )

    assert {"real_train", "dataframe_sampler_balanced", "smotenc_balanced", "stratified_columns_balanced"}.issubset(
        set(report["method"])
    )
    dfs = report[report["method"] == "dataframe_sampler_balanced"].iloc[0]
    assert dfs["synthetic_rows"] == 16
    assert 0.0 <= dfs["balanced_accuracy"] <= 1.0
    assert 0.0 <= dfs["minority_recall"] <= 1.0


def test_run_imbalance_validation_writes_expected_columns(tmp_path):
    dataframe = make_imbalance_dataframe(n_majority=30, n_minority=10)
    config = make_imbalance_config()

    report = run_imbalance_validation_for_config(
        config,
        dataframe,
        results_dir=tmp_path,
        sampler_config={"n_neighbours": 2, "knn_backend": "sklearn", "n_iterations": 0, "random_state": 3},
        max_train_rows=30,
        max_test_rows=10,
    )

    assert (tmp_path / "adult_imbalance_validation.csv").exists()
    assert {
        "dataset",
        "method",
        "minority_class",
        "balanced_accuracy",
        "macro_f1",
        "minority_recall",
        "pr_auc",
        "reason",
    }.issubset(report.columns)


def test_imbalance_validation_skips_unselected_dataset(tmp_path):
    dataframe = make_imbalance_dataframe()
    config = make_imbalance_config(dataset_name="toy")

    report = run_imbalance_validation_for_config(config, dataframe, results_dir=tmp_path)

    assert report["reason"].iloc[0] == "dataset_not_selected"
    assert not (tmp_path / "toy_imbalance_validation.csv").exists()


def test_smotenc_path_uses_optional_dependency_when_available(monkeypatch):
    class FakeSMOTENC:
        def __init__(self, categorical_features, random_state=None, sampling_strategy="auto"):
            self.categorical_features = categorical_features

        def fit_resample(self, x, y):
            minority = y.value_counts().idxmin()
            n_needed = y.value_counts().max() - y.value_counts().min()
            extras = x[y == minority].iloc[:1].copy()
            extras = pd.concat([extras] * int(n_needed), ignore_index=True)
            return pd.concat([x, extras], ignore_index=True), pd.concat(
                [y.reset_index(drop=True), pd.Series([minority] * int(n_needed))],
                ignore_index=True,
            )

    fake_module = types.ModuleType("imblearn.over_sampling")
    fake_module.SMOTE = FakeSMOTENC
    fake_module.SMOTENC = FakeSMOTENC
    fake_parent = types.ModuleType("imblearn")
    fake_parent.over_sampling = fake_module
    monkeypatch.setitem(sys.modules, "imblearn", fake_parent)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", fake_module)

    dataframe = make_imbalance_dataframe(n_majority=20, n_minority=6)
    report = imbalance_validation_report(
        train=dataframe,
        test=dataframe,
        dataset_name="adult",
        target_column="target",
        minority_class="positive",
        majority_class="negative",
        n_to_generate=14,
        sampler_config={"n_neighbours": 2, "knn_backend": "sklearn", "n_iterations": 0, "random_state": 5},
        random_state=5,
    )

    smotenc = report[report["method"] == "smotenc_balanced"].iloc[0]
    assert smotenc["reason"] == ""
    assert smotenc["synthetic_rows"] == 14


def test_imbalance_validation_table_writer(tmp_path):
    rows = pd.DataFrame(
        {
            "dataset": ["adult"],
            "method": ["dataframe_sampler_balanced"],
            "target_column": ["target"],
            "minority_class": ["positive"],
            "majority_class": ["negative"],
            "train_rows": [30],
            "test_rows": [10],
            "synthetic_rows": [12],
            "train_minority_rate": [0.25],
            "augmented_minority_rate": [0.5],
            "accuracy": [0.8],
            "balanced_accuracy": [0.75],
            "macro_f1": [0.7],
            "minority_recall": [0.6],
            "pr_auc": [0.5],
            "fit_seconds": [1.0],
            "sample_seconds": [0.5],
            "peak_memory_mb": [2.0],
            "reason": [""],
        }
    )

    path = write_imbalance_validation_table(rows, tables_dir=tmp_path)

    assert path.exists()
    assert "Takeaway:" in path.read_text()
