from dataclasses import replace

import pandas as pd

from experiments.datasets import DatasetExperimentConfig
from experiments.workflow import (
    dataset_profile,
    prepare_dataframe_for_experiment,
    quick_similarity_report,
    run_starter_sampler,
    working_dataframe,
)
from experiments.vectorization_plan import (
    columns_requiring_vectorization,
    preprocessing_plan,
    vectorization_plan,
)


def make_workflow_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "fare": [10.0, 12.5, 20.0, 25.0, 30.0],
            "group": ["a", "a", "b", "b", "c"],
        }
    )


def make_workflow_config(**overrides):
    config = DatasetExperimentConfig(
        dataset_name="toy",
        title="Toy",
        data_filename="toy.csv",
        target_column=None,
        working_sample_size=3,
        n_generated=6,
        random_state=1,
        manual_sampler_config={
            "n_bins": 3,
            "n_neighbours": 2,
            "knn_backend": "sklearn",
        },
    )
    return replace(config, **overrides)


def test_dataset_profile_reports_dtype_missing_and_unique_counts():
    df = make_workflow_dataframe()

    profile = dataset_profile(df)

    assert list(profile.columns) == ["dtype", "missing", "unique"]
    assert profile.loc["age", "unique"] == 5
    assert profile.loc["group", "missing"] == 0


def test_working_dataframe_applies_reproducible_sample_size():
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=3)

    sampled = working_dataframe(df, config)

    assert sampled.shape == (3, 3)
    assert sampled.equals(working_dataframe(df, config))


def test_working_dataframe_uses_full_frame_when_sample_size_is_none():
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=None)

    sampled = working_dataframe(df, config)

    assert sampled.equals(df.reset_index(drop=True))


def test_prepare_dataframe_applies_direct_mappings_and_drops_aliases():
    df = pd.DataFrame(
        {
            "pclass": [1, 3],
            "class": ["First", "Third"],
            "sex": ["female", "male"],
            "alone": [False, True],
        }
    )
    config = make_workflow_config(
        drop_columns=("class",),
        direct_numeric_mappings={
            "sex": {"female": 0.0, "male": 1.0},
            "alone": {False: 0.0, True: 1.0},
        },
    )

    prepared = prepare_dataframe_for_experiment(df, config)

    assert list(prepared.columns) == ["pclass", "sex", "alone"]
    assert prepared["sex"].tolist() == [0.0, 1.0]
    assert prepared["alone"].tolist() == [0.0, 1.0]


def test_quick_similarity_report_handles_numeric_and_categorical_columns():
    real = make_workflow_dataframe()
    synthetic = real.sample(n=5, replace=True, random_state=2).reset_index(drop=True)

    report = quick_similarity_report(real, synthetic)

    assert set(report["kind"]) == {"numeric", "categorical"}
    assert {"age", "fare", "group"} == set(report["column"])
    assert report.loc[report["column"] == "group", "category_coverage"].iloc[0] > 0


def test_run_starter_sampler_writes_reusable_outputs(tmp_path):
    df = make_workflow_dataframe()
    config = make_workflow_config(working_sample_size=None, n_generated=6)

    run = run_starter_sampler(df, config, results_dir=tmp_path)

    assert run.generated.shape == (6, 3)
    assert run.peak_memory_mb >= 0
    assert {
        "fit_peak_memory_mb",
        "sample_peak_memory_mb",
        "peak_memory_mb",
    }.issubset(run.runtime.columns)
    assert (tmp_path / "toy_generated_start.csv").exists()
    assert (tmp_path / "toy_similarity_start.csv").exists()
    assert (tmp_path / "toy_runtime_start.csv").exists()


def test_vectorization_plan_reports_helper_embedding_and_frequency_fallback():
    df = make_workflow_dataframe()
    config = make_workflow_config(
        manual_sampler_config={
            "n_bins": 3,
            "n_neighbours": 2,
            "knn_backend": "sklearn",
            "embedding_method": "pca",
            "vectorizing_columns_dict": {"group": ["age", "fare"]},
        }
    )

    plan = vectorization_plan(df, config)

    assert columns_requiring_vectorization(df) == ["group"]
    group = plan.loc[plan["column"] == "group"].iloc[0]
    assert group["strategy"] == "helper_embedding"
    assert group["embedding_method"] == "pca"
    assert group["helper_columns"] == "age, fare"
    assert group["helper_status"] == "all helpers numeric"


def test_vectorization_plan_marks_unconfigured_categoricals_as_frequency_encoded():
    df = make_workflow_dataframe()
    plan = vectorization_plan(df, make_workflow_config())

    group = plan.loc[plan["column"] == "group"].iloc[0]

    assert group["strategy"] == "frequency_encoding"
    assert "empirical frequency" in group["decision"]


def test_vectorization_plan_reports_configured_direct_mapping():
    df = pd.DataFrame({"sex": [0.0, 1.0], "age": [20, 30]})
    config = make_workflow_config(
        direct_numeric_mappings={"sex": {"female": 0.0, "male": 1.0}},
        manual_sampler_config={
            "n_bins": 3,
            "n_neighbours": 2,
            "knn_backend": "sklearn",
            "vectorizing_columns_dict": {"sex": ["age"]},
        },
    )

    plan = vectorization_plan(df, config)
    row = plan.loc[plan["column"] == "sex"].iloc[0]

    assert row["strategy"] == "direct_mapping"
    assert row["helper_columns"] == ""
    assert "'female'->0" in row["direct_mapping"]


def test_preprocessing_plan_reports_drops_and_direct_mappings():
    config = make_workflow_config(
        drop_columns=("class",),
        direct_numeric_mappings={"sex": {"female": 0.0, "male": 1.0}},
    )

    plan = preprocessing_plan(config)

    assert set(plan["action"]) == {"drop", "direct_numeric_mapping"}
    assert "class" in set(plan["column"])
    assert "sex" in set(plan["column"])
