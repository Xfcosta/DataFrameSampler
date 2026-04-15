import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import dataframe_sampler.cli as cli_module
from dataframe_sampler import (
    ColumnDataFrameEncoderDecoder,
    ConcreteDataFrameSampler,
    DataFrameEncoderDecoder,
    DataFrameVectorizer,
    NearestMutualNeighboursEstimator,
    anonymize_columns_with_openai,
    assert_no_value_overlap,
    dataframe_sampler_main,
    find_nearest_neighbours,
    profile_dataframe_for_llm,
    read_dataframe,
    suggest_sampler_config_with_openai,
    write_dataframe,
)


def make_mixed_dataframe():
    return pd.DataFrame(
        {
            "age": [21, 22, 23, 35, 36, 37, 48, 49, 50, 62, 63, 64],
            "band": [
                "young",
                "young",
                "young",
                "adult",
                "adult",
                "adult",
                "middle",
                "middle",
                "middle",
                "senior",
                "senior",
                "senior",
            ],
            "score": [3, 4, 5, 10, 11, 12, 17, 18, 19, 24, 25, 26],
        }
    )


def make_sensitive_dataframe():
    return pd.DataFrame(
        {
            "personName": ["Alice Smith", "Bob Jones", "Alice Smith", "Dana Ray"],
            "age": [21, 35, 21, 48],
            "score": [3, 10, 3, 17],
        }
    )


def test_vectorizer_keeps_numeric_columns_and_frequency_encodes_categories():
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "a", "c"],
        }
    )

    transformed = DataFrameVectorizer().fit_transform(df)

    assert transformed["value"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert transformed["label"].tolist() == [2, 1, 2, 1]


def test_column_encoder_decoder_round_trips_through_known_bins():
    encoder = ColumnDataFrameEncoderDecoder(n_bins=2)
    column_values = np.array(["low", "low", "high", "high"])
    vectorizing_values = np.array([1, 2, 9, 10])

    encoder.fit(column_values, vectorizing_values)
    encoded = encoder.encode(vectorizing_values)
    decoded = encoder.decode(encoded)

    assert encoded.tolist() == [0, 0, 1, 1]
    assert decoded[:2] == ["low", "low"]
    assert decoded[2:] == ["high", "high"]


def test_dataframe_encoder_decoder_sample_uses_valid_bin_indexes():
    np.random.seed(0)
    df = make_mixed_dataframe()[["age", "score"]]
    encoder = DataFrameEncoderDecoder(n_bins=3)
    encoder.fit(df, df)

    sampled_bins = encoder.sample(n_samples=200)

    assert sampled_bins.shape == (200, 2)
    for col_idx, column_encoder in enumerate(encoder.column_dataframe_encoder_decoders):
        n_bins = column_encoder.discretizer.n_bins_[0]
        assert sampled_bins[:, col_idx].min() >= 0
        assert sampled_bins[:, col_idx].max() < n_bins


def test_nearest_mutual_neighbours_are_symmetric():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])

    neighbours = NearestMutualNeighboursEstimator(n_neighbours=1).fit_predict(X)

    assert [n.tolist() for n in neighbours] == [[1], [0], [3], [2]]


def test_nearest_mutual_neighbours_fall_back_to_nearest_when_empty():
    X = np.array([[0.0], [1.0], [2.0]])

    neighbours = NearestMutualNeighboursEstimator(n_neighbours=1).fit_predict(X)

    assert all(len(row) >= 1 for row in neighbours)
    assert neighbours[2].tolist() == [1]


def test_exact_and_sklearn_knn_backends_return_neighbours_without_self():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])

    exact = find_nearest_neighbours(X, n_neighbours=1, backend="exact")
    sklearn = find_nearest_neighbours(X, n_neighbours=1, backend="sklearn")

    assert exact.tolist() == [[1], [0], [3], [2]]
    assert sklearn.tolist() == [[1], [0], [3], [2]]


def test_unknown_knn_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown KNN backend"):
        find_nearest_neighbours(np.array([[0.0], [1.0]]), backend="unknown")


def test_optional_knn_backend_has_actionable_install_message():
    pytest.importorskip("pynndescent")
    neighbours = find_nearest_neighbours(
        np.array([[0.0], [1.0], [10.0], [11.0]]),
        n_neighbours=1,
        backend="pynndescent",
        backend_kwargs={"random_state": 42},
    )

    assert neighbours.shape == (4, 1)
    assert all(row[0] != idx for idx, row in enumerate(neighbours))


def test_concrete_dataframe_sampler_generates_requested_shape_and_columns():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=1)

    sampler.fit(df)
    generated = sampler.sample(n_samples=8)

    assert list(generated.columns) == list(df.columns)
    assert len(generated) == 8
    assert set(generated["band"]).issubset(set(df["band"]))


def test_concrete_dataframe_sampler_accepts_sklearn_knn_backend():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(
        n_bins=4,
        n_neighbours=3,
        random_state=1,
        knn_backend="sklearn",
    )

    sampler.fit(df)
    generated = sampler.sample(n_samples=8)

    assert list(generated.columns) == list(df.columns)
    assert len(generated) == 8


def test_concrete_dataframe_sampler_respects_sampled_columns():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(
        n_bins=4,
        n_neighbours=3,
        sampled_columns=["age", "band"],
        random_state=2,
    )

    sampler.fit(df)
    generated = sampler.sample(n_samples=5)

    assert list(generated.columns) == ["age", "band"]
    assert len(generated) == 5


def test_cli_fits_and_generates_csv(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "generated.csv"
    model_file = tmp_path / "model.obj"
    make_mixed_dataframe().to_csv(input_csv, index=False)

    result = CliRunner().invoke(
        dataframe_sampler_main,
        [
            "-i",
            str(input_csv),
            "-o",
            str(output_csv),
            "-d",
            str(model_file),
            "-n",
            "6",
            "--n_bins",
            "4",
            "--n_neighbours",
            "3",
            "--random_state",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = pd.read_csv(output_csv)
    assert len(generated) == 6
    assert list(generated.columns) == ["age", "band", "score"]
    assert model_file.exists()


def test_anonymize_columns_with_openai_replaces_before_sampling_values():
    class FakeResponse:
        output_text = """
        {
          "replacements": [
            {"original": "Alice Smith", "replacement": "Nora Vale"},
            {"original": "Bob Jones", "replacement": "Milo Stone"},
            {"original": "Dana Ray", "replacement": "Iris Lane"}
          ]
        }
        """

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["text"]["format"]["type"] == "json_schema"
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    df = make_sensitive_dataframe()
    anonymized, report = anonymize_columns_with_openai(
        df,
        columns=["personName"],
        client=FakeClient(),
    )

    assert report["columns"] == ["personName"]
    assert anonymized["personName"].tolist() == ["Nora Vale", "Milo Stone", "Nora Vale", "Iris Lane"]
    assert_no_value_overlap(df, anonymized, ["personName"])


def test_assert_no_value_overlap_rejects_original_values():
    df = make_sensitive_dataframe()

    with pytest.raises(ValueError, match="overlap"):
        assert_no_value_overlap(df, df.copy(), ["personName"])


def test_cli_anonymizes_before_fit_and_checks_generated_output(tmp_path, monkeypatch):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "generated.csv"
    make_sensitive_dataframe().to_csv(input_csv, index=False)

    def fake_anonymize(dataframe, source_dataframe, columns):
        anonymized = dataframe.copy()
        anonymized["personName"] = anonymized["personName"].map(
            {
                "Alice Smith": "Nora Vale",
                "Bob Jones": "Milo Stone",
                "Dana Ray": "Iris Lane",
            }
        )
        return anonymized, {"columns": columns}

    monkeypatch.setattr(cli_module, "anonymize_columns_with_openai", fake_anonymize)
    result = CliRunner().invoke(
        dataframe_sampler_main,
        [
            "-i",
            str(input_csv),
            "-o",
            str(output_csv),
            "--anonymize_columns",
            "personName",
            "-n",
            "4",
            "--n_bins",
            "3",
            "--n_neighbours",
            "2",
            "--random_state",
            "14",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = pd.read_csv(output_csv)
    assert_no_value_overlap(make_sensitive_dataframe(), generated, ["personName"])


def test_sample_to_file_uses_extension_based_csv_writer(tmp_path):
    output_csv = tmp_path / "generated.csv"
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=10).fit(make_mixed_dataframe())

    generated = sampler.sample_to_file(n_samples=4, filename=output_csv)
    loaded = read_dataframe(output_csv)

    assert len(generated) == 4
    assert list(loaded.columns) == ["age", "band", "score"]
    assert len(loaded) == 4


def test_read_write_dataframe_rejects_unknown_extension(tmp_path):
    path = tmp_path / "generated.json"

    with pytest.raises(ValueError, match="Unsupported dataframe file extension"):
        write_dataframe(make_mixed_dataframe(), path)

    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported dataframe file extension"):
        read_dataframe(path)


def test_read_write_dataframe_supports_parquet_when_engine_is_available(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    df = make_mixed_dataframe()

    write_dataframe(df, path)
    loaded = read_dataframe(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_cli_accepts_parquet_input_and_output_when_engine_is_available(tmp_path):
    pytest.importorskip("pyarrow")
    input_parquet = tmp_path / "input.parquet"
    output_parquet = tmp_path / "generated.parquet"
    make_mixed_dataframe().to_parquet(input_parquet, index=False)

    result = CliRunner().invoke(
        dataframe_sampler_main,
        [
            "-i",
            str(input_parquet),
            "-o",
            str(output_parquet),
            "-n",
            "5",
            "--n_bins",
            "4",
            "--n_neighbours",
            "3",
            "--random_state",
            "11",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = pd.read_parquet(output_parquet)
    assert len(generated) == 5


def test_sampler_handles_constant_columns():
    df = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "label": ["a", "a", "a", "a"],
        }
    )

    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=2, random_state=4)
    sampler.fit(df)
    generated = sampler.sample(n_samples=3)

    assert generated["constant"].tolist() == [7, 7, 7]
    assert generated["label"].tolist() == ["a", "a", "a"]


def test_sampler_handles_missing_values_in_fit_data():
    df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 22.0, 40.0, 41.0, np.nan],
            "city": ["a", "a", None, "b", "b", None],
        }
    )

    sampler = ConcreteDataFrameSampler(n_bins=3, n_neighbours=2, random_state=5)
    sampler.fit(df)
    generated = sampler.sample(n_samples=4)

    assert list(generated.columns) == ["age", "city"]
    assert len(generated) == 4


def test_vectorizing_columns_dict_embeds_configured_category():
    df = pd.DataFrame(
        {
            "name": ["ann", "bob", "cam", "dan"],
            "age": [20, 30, 40, 50],
            "country_id": [1, 1, 2, 2],
        }
    )

    transformed = DataFrameVectorizer(
        vectorizing_columns_dict={"name": ["age", "country_id"]},
        random_state=6,
    ).fit_transform(df)

    assert set(transformed.columns) == {"name", "age", "country_id"}
    assert transformed["name"].nunique() > 1


def test_vectorizer_supports_pca_embedding_by_string():
    df = pd.DataFrame(
        {
            "name": ["ann", "bob", "cam", "dan"],
            "age": [20, 30, 40, 50],
            "country_id": [1, 1, 2, 2],
        }
    )

    transformed = DataFrameVectorizer(
        vectorizing_columns_dict={"name": ["age", "country_id"]},
        embedding_method="pca",
    ).fit_transform(df)

    assert transformed["name"].nunique() == 4


def test_vectorizer_supports_custom_transform_embedding_object():
    class FirstColumnProjector:
        def transform(self, X):
            return X[:, :1]

    df = pd.DataFrame(
        {
            "name": ["ann", "bob", "cam"],
            "age": [20, 30, 40],
            "country_id": [1, 2, 3],
        }
    )

    transformed = DataFrameVectorizer(
        vectorizing_columns_dict={"name": ["age", "country_id"]},
        embedding_method=FirstColumnProjector(),
    ).fit_transform(df)

    assert transformed["name"].tolist() == [20.0, 30.0, 40.0]


def test_vectorizer_rejects_unknown_embedding_string():
    df = pd.DataFrame(
        {
            "name": ["ann", "bob"],
            "age": [20, 30],
        }
    )

    vectorizer = DataFrameVectorizer(
        vectorizing_columns_dict={"name": ["age"]},
        embedding_method="not_real",
    )

    with pytest.raises(ValueError, match="Unknown embedding_method"):
        vectorizer.fit_transform(df)


def test_profile_dataframe_for_llm_describes_columns():
    df = make_mixed_dataframe()

    profile = profile_dataframe_for_llm(df)

    assert profile["row_count"] == len(df)
    assert "age" in profile["numeric_columns"]
    assert "band" in profile["categorical_columns"]
    assert any(column["name"] == "band" for column in profile["columns"])


def test_suggest_sampler_config_with_openai_uses_structured_response():
    class FakeResponse:
        output_text = """
        {
          "recommendations": [
            {
              "column": "band",
              "helper_columns": ["age", "score", "not_a_column"],
              "rationale": "Age and score define the band semantics.",
              "confidence": 0.9
            }
          ],
          "sampled_columns": ["age", "band", "score"],
          "embedding_method": "pca",
          "knn_backend": "sklearn",
          "notes": "Use PCA for the numeric helper space."
        }
        """

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "fake-model"
            assert kwargs["text"]["format"]["type"] == "json_schema"
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    config = suggest_sampler_config_with_openai(
        make_mixed_dataframe(),
        model="fake-model",
        client=FakeClient(),
    )

    assert config["vectorizing_columns_dict"] == {"band": ["age", "score"]}
    assert config["sampled_columns"] == ["age", "band", "score"]
    assert config["embedding_method"] == "pca"
    assert config["knn_backend"] == "sklearn"


def test_concrete_dataframe_sampler_accepts_pca_embedding_method():
    df = make_mixed_dataframe()
    sampler = ConcreteDataFrameSampler(
        n_bins=4,
        n_neighbours=3,
        random_state=9,
        vectorizing_columns_dict={"band": ["age", "score"]},
        embedding_method="pca",
    )

    sampler.fit(df)
    generated = sampler.sample(n_samples=5)

    assert list(generated.columns) == list(df.columns)
    assert len(generated) == 5


def test_sampler_is_reproducible_with_random_state():
    df = make_mixed_dataframe()
    sampler1 = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=7).fit(df)
    sampler2 = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=7).fit(df)

    pd.testing.assert_frame_equal(
        sampler1.sample(n_samples=6),
        sampler2.sample(n_samples=6),
    )


def test_save_and_load_model_round_trip(tmp_path):
    model_file = tmp_path / "model.obj"
    sampler = ConcreteDataFrameSampler(n_bins=4, n_neighbours=3, random_state=8).fit(make_mixed_dataframe())

    loaded = sampler.save(model_file).load(model_file)
    generated = loaded.sample(n_samples=4)

    assert len(generated) == 4
    assert list(generated.columns) == ["age", "band", "score"]


def test_sampler_rejects_single_row_dataframes():
    sampler = ConcreteDataFrameSampler(n_bins=2, n_neighbours=1)

    with pytest.raises(ValueError, match="At least two rows"):
        sampler.fit(pd.DataFrame({"value": [1]}))


def test_cli_requires_input_or_model():
    result = CliRunner().invoke(dataframe_sampler_main, ["-o", "ignored.csv"])

    assert result.exit_code != 0
    assert "Provide --input_filename" in result.output


def test_cli_auto_config_fills_omitted_sampler_options(tmp_path, monkeypatch):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "generated.csv"
    make_mixed_dataframe().to_csv(input_csv, index=False)

    def fake_auto_config(df):
        assert list(df.columns) == ["age", "band", "score"]
        return {
            "vectorizing_columns_dict": {"band": ["age", "score"]},
            "sampled_columns": ["age", "band", "score"],
            "embedding_method": "pca",
            "knn_backend": "sklearn",
            "recommendations": [],
            "notes": "Fake auto config.",
        }

    monkeypatch.setattr(cli_module, "suggest_sampler_config_with_openai", fake_auto_config)
    result = CliRunner().invoke(
        dataframe_sampler_main,
        [
            "-A",
            "-i",
            str(input_csv),
            "-o",
            str(output_csv),
            "-n",
            "5",
            "--random_state",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Fake auto config." in result.output
    assert len(pd.read_csv(output_csv)) == 5


def test_cli_auto_config_preserves_user_specified_options(tmp_path, monkeypatch):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "generated.csv"
    make_mixed_dataframe().to_csv(input_csv, index=False)

    vectorizing_yaml = tmp_path / "vectorizing.yaml"
    vectorizing_yaml.write_text("band:\n- age\n")

    def fake_auto_config(df):
        return {
            "vectorizing_columns_dict": {"band": ["score"]},
            "sampled_columns": ["band"],
            "embedding_method": "pca",
            "knn_backend": "sklearn",
            "recommendations": [],
            "notes": "Fake auto config.",
        }

    monkeypatch.setattr(cli_module, "suggest_sampler_config_with_openai", fake_auto_config)
    result = CliRunner().invoke(
        dataframe_sampler_main,
        [
            "-A",
            "-i",
            str(input_csv),
            "-o",
            str(output_csv),
            "-f",
            str(vectorizing_yaml),
            "-c",
            "age",
            "-c",
            "band",
            "--embedding_method",
            "mds",
            "--knn_backend",
            "exact",
            "-n",
            "5",
            "--random_state",
            "13",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = pd.read_csv(output_csv)
    assert list(generated.columns) == ["age", "band"]
