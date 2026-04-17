import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.predictive import predictive_performance_report, target_column_choice


def make_predictive_dataframe():
    rows = []
    for i in range(40):
        rows.append(
            {
                "age": 20 + i,
                "score": float(i % 10),
                "segment": "high" if i % 4 in {0, 1} else "low",
                "target": float(i % 2),
            }
        )
    return pd.DataFrame(rows)


def make_predictive_config():
    return DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column="target",
        random_state=1,
        manual_sampler_config={
            "n_neighbours": 3,
            "knn_backend": "sklearn",
            },
    )


def test_target_column_choice_reports_configured_task():
    df = make_predictive_dataframe()
    choice = target_column_choice(make_predictive_config(), df)

    assert choice.loc[0, "target_column"] == "target"
    assert choice.loc[0, "task"] == "classification"
    assert choice.loc[0, "available"]


def test_predictive_performance_report_compares_real_and_synthetic_training():
    df = make_predictive_dataframe()
    report = predictive_performance_report(df, make_predictive_config(), n_synthetic=20)

    assert set(report["training_source"]) == {"real_train", "synthetic_from_real_train"}
    assert set(["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]).issubset(report.columns)
    assert report["target_column"].eq("target").all()
    assert report["test_rows"].iloc[0] == 12
