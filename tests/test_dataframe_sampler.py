import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from dataframe_sampler import DataFrameSampler
from dataframe_sampler.cli import dataframe_sampler_main
from dataframe_sampler.io import read_dataframe, write_dataframe
from dataframe_sampler.knn import find_nearest_neighbours
from dataframe_sampler.neighbours import NearestMutualNeighboursEstimator
from dataframe_sampler.sampler import NearestMutualNeighboursSampler


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
            "flag": [0, 1] * 6,
            "score": [3, 4, 5, 10, 11, 12, 17, 18, 19, 24, 25, 26],
        }
    )


def test_dataframe_sampler_fit_transform_inverse_and_generate_contract():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(n_components=2, n_iterations=1, n_neighbours=3, random_state=1)

    returned = sampler.fit(df)
    latent = sampler.transform(df.head(4))
    decoded = sampler.inverse_transform(latent, sample=False)
    generated = sampler.generate(n_samples=5)

    assert returned is sampler
    assert isinstance(latent, np.ndarray)
    assert latent.shape == (4, 6)
    assert isinstance(decoded, pd.DataFrame)
    assert list(decoded.columns) == list(df.columns)
    assert list(generated.columns) == list(df.columns)
    assert len(generated) == 5
    assert not hasattr(sampler, "sample")


def test_generate_none_uses_fit_row_count():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=2).fit(df)

    generated = sampler.generate()

    assert len(generated) == len(df)


def test_fit_rejects_non_dataframe_and_empty_dataframe():
    sampler = DataFrameSampler()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        sampler.fit(np.array([[1, 2]]))
    with pytest.raises(ValueError, match="at least one row"):
        sampler.fit(pd.DataFrame())


def test_fit_accepts_single_row_dataframe_and_generate_replays_latent_shape():
    df = pd.DataFrame({"flag": [1], "label": ["only"]})
    sampler = DataFrameSampler(n_iterations=1, random_state=21).fit(df)

    latent = sampler.transform(df)
    generated = sampler.generate()

    assert latent.shape == (1, 4)
    assert len(generated) == 1
    assert list(generated.columns) == ["flag", "label"]
    assert generated["flag"].isin([1]).all()
    assert generated["label"].isin(["only"]).all()


def test_binary_columns_are_categorical_regardless_of_dtype():
    df = pd.DataFrame(
        {
            "binary_object": ["yes", "no", "yes", "no", "yes", "no"],
            "binary_bool": [True, False, True, False, True, False],
            "binary_numeric": [0, 1, 0, 1, 0, 1],
            "two_value_numeric": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "continuous": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    sampler = DataFrameSampler(n_iterations=1, n_neighbours=2, random_state=3).fit(df)

    assert sampler.numeric_columns_ == ["continuous"]
    assert sampler.categorical_columns_ == [
        "binary_object",
        "binary_bool",
        "binary_numeric",
        "two_value_numeric",
    ]
    assert sampler.latent_dim_ == 1 + 4 * 2


def test_per_column_n_components_controls_latent_width():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(
        n_components={"band": 3, "flag": 1},
        n_iterations=1,
        n_neighbours=3,
        random_state=4,
    ).fit(df)

    latent = sampler.transform(df)

    assert sampler.numeric_columns_ == ["age", "score"]
    assert latent.shape[1] == 2 + 3 + 1


def test_zero_iterations_keeps_one_hot_blocks_and_excludes_target_from_decoders():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(
        n_iterations=0,
        n_neighbours=3,
        random_state=4,
        calibrate_decoders=False,
    ).fit(df)

    latent = sampler.transform(df)

    assert sampler.projector_steps_ == []
    assert latent.shape[1] == 2 + 4 + 2
    assert sampler.decoders_["band"].n_features_in_ == 2 + 2
    assert sampler.decoders_["flag"].n_features_in_ == 2 + 4


def test_nca_decoder_context_excludes_target_block():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(
        n_components=2,
        n_iterations=1,
        n_neighbours=3,
        random_state=4,
        calibrate_decoders=False,
    ).fit(df)

    assert sampler.latent_data_mtx_.shape[1] == 2 + 2 + 2
    assert sampler.decoders_["band"].n_features_in_ == 2 + 2
    assert sampler.decoders_["flag"].n_features_in_ == 2 + 2


def test_nca_fit_sample_size_fraction_fits_projectors_on_row_fraction():
    df = pd.DataFrame(
        {
            "value": np.linspace(0.0, 1.0, 30),
            "label": ["a", "b"] * 15,
        }
    )
    sampler = DataFrameSampler(
        n_components=2,
        n_iterations=1,
        n_neighbours=3,
        nca_fit_sample_size=0.1,
        random_state=4,
        calibrate_decoders=False,
    ).fit(df)

    assert {len(step["fit_indices"]) for step in sampler.projector_steps_} == {3}
    assert sampler.transform(df).shape == (len(df), sampler.latent_dim_)


def test_nca_fit_sample_size_integer_fits_projectors_on_row_cap():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(
        n_components=2,
        n_iterations=1,
        n_neighbours=3,
        nca_fit_sample_size=5,
        random_state=4,
        calibrate_decoders=False,
    ).fit(df)

    assert {len(step["fit_indices"]) for step in sampler.projector_steps_} == {5}
    assert sampler.latent_data_mtx_.shape[0] == len(df)


def test_nca_fit_sample_size_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 1"):
        DataFrameSampler(nca_fit_sample_size=0)
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        DataFrameSampler(nca_fit_sample_size=1.5)
    with pytest.raises(TypeError, match="nca_fit_sample_size"):
        DataFrameSampler(nca_fit_sample_size="10%")


def test_high_cardinality_categorical_warns_but_proceeds():
    df = pd.DataFrame(
        {
            "value": np.arange(80, dtype=float),
            "code": [f"code-{idx}" for idx in range(80)],
        }
    )

    with pytest.warns(UserWarning, match="high-cardinality"):
        sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=5).fit(df)

    assert "code" in sampler.categorical_columns_


def test_unknown_categories_transform_without_crashing():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=6).fit(df)
    new_df = df.head(2).copy()
    new_df.loc[:, "band"] = ["new-a", "new-b"]

    latent = sampler.transform(new_df)

    assert latent.shape == (2, sampler.latent_dim_)


def test_inverse_transform_sample_false_is_deterministic_and_valid():
    df = make_mixed_dataframe()
    sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=7).fit(df)
    latent = sampler.transform(df.head(3))

    first = sampler.inverse_transform(latent, sample=False)
    second = sampler.inverse_transform(latent, sample=False)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["band"]).issubset(set(df["band"]))
    assert set(first["flag"]).issubset(set(df["flag"]))


def test_random_forest_decoders_are_uncalibrated_by_default_and_use_all_cores():
    df = make_mixed_dataframe()

    default_sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=17).fit(df)
    assert all(isinstance(decoder, RandomForestClassifier) for decoder in default_sampler.decoders_.values())
    assert {decoder.n_jobs for decoder in default_sampler.decoders_.values()} == {-1}
    assert set(default_sampler.decoder_calibration_status_.values()) == {"disabled"}


def test_random_forest_decoder_calibration_can_be_enabled():
    df = make_mixed_dataframe()

    calibrated_sampler = DataFrameSampler(
        n_iterations=1,
        n_neighbours=3,
        random_state=17,
        calibrate_decoders=True,
    ).fit(df)

    assert all(isinstance(decoder, CalibratedClassifierCV) for decoder in calibrated_sampler.decoders_.values())
    assert {decoder.n_jobs for decoder in calibrated_sampler.decoders_.values()} == {-1}
    assert set(calibrated_sampler.decoder_calibration_status_.values()) == {"calibrated"}


def test_random_forest_decoder_n_jobs_can_be_overridden():
    df = make_mixed_dataframe()

    override_sampler = DataFrameSampler(
        n_iterations=1,
        n_neighbours=3,
        random_state=17,
        decoder_kwargs={"n_jobs": 2},
    ).fit(df)

    assert all(isinstance(decoder, RandomForestClassifier) for decoder in override_sampler.decoders_.values())
    assert {decoder.n_jobs for decoder in override_sampler.decoders_.values()} == {2}
    assert set(override_sampler.decoder_calibration_status_.values()) == {"disabled"}


def test_decoder_calibration_skips_when_class_counts_are_too_small():
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "label": ["a", "b", "b"],
        }
    )

    sampler = DataFrameSampler(
        n_iterations=1,
        n_neighbours=2,
        random_state=18,
        calibrate_decoders=True,
    ).fit(df)

    assert isinstance(sampler.decoders_["label"], RandomForestClassifier)
    assert sampler.decoder_calibration_status_["label"] == "skipped_insufficient_class_count"


def test_fixed_random_state_reproducible_generation():
    df = make_mixed_dataframe()
    sampler1 = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=8).fit(df)
    sampler2 = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=8).fit(df)

    pd.testing.assert_frame_equal(
        sampler1.generate(n_samples=6),
        sampler2.generate(n_samples=6),
    )


def test_save_and_load_round_trip(tmp_path):
    df = make_mixed_dataframe()
    model = tmp_path / "model.obj"
    sampler = DataFrameSampler(n_iterations=1, n_neighbours=3, random_state=9).fit(df)

    sampler.save(model)
    loaded = DataFrameSampler().load(model)
    generated = loaded.generate(n_samples=4)

    assert list(generated.columns) == list(df.columns)
    assert len(generated) == 4


def test_nearest_mutual_neighbours_are_symmetric():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])

    neighbours = NearestMutualNeighboursEstimator(n_neighbours=1).fit_predict(X)

    assert [n.tolist() for n in neighbours] == [[1], [0], [3], [2]]


def test_nearest_mutual_neighbours_fall_back_to_nearest_when_empty():
    X = np.array([[0.0], [1.0], [2.0]])

    neighbours = NearestMutualNeighboursEstimator(n_neighbours=1).fit_predict(X)

    assert all(len(row) >= 1 for row in neighbours)
    assert neighbours[2].tolist() == [1]


class _FixedProbabilityEstimator:
    def fit_predict_proba(self, X, y=None):
        return np.ones(len(X), dtype=float) / len(X)


class _FixedNeighbourEstimator:
    def fit_predict(self, X, y=None):
        return np.array([np.array([1]), np.array([0])], dtype=object)


def test_neighbour_sampler_retries_out_of_range_candidates_without_clipping():
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator=_FixedNeighbourEstimator(),
        probability_estimator=_FixedProbabilityEstimator(),
        use_min_max_constraints=True,
        max_constraint_retries=3,
        random_state=1,
    ).fit(np.array([[0.0], [1.0]]))
    candidates = iter([np.array([2.0]), np.array([0.5])])
    sampler._sample_anchor_index = lambda sampling_probability: 0
    sampler._generate_from_anchor = lambda *args, **kwargs: next(candidates)

    generated = sampler.sample(1)

    assert generated.tolist() == [[0.5]]
    assert sampler.constraint_retry_count_ == 1
    assert sampler.constraint_violation_count_ == 0


def test_neighbour_sampler_accepts_out_of_range_candidate_after_retry_limit():
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator=_FixedNeighbourEstimator(),
        probability_estimator=_FixedProbabilityEstimator(),
        use_min_max_constraints=True,
        max_constraint_retries=3,
        random_state=1,
    ).fit(np.array([[0.0], [1.0]]))
    sampler._sample_anchor_index = lambda sampling_probability: 0
    sampler._generate_from_anchor = lambda *args, **kwargs: np.array([2.0])

    generated = sampler.sample(1)

    assert generated.tolist() == [[2.0]]
    assert sampler.constraint_retry_count_ == 3
    assert sampler.constraint_violation_count_ == 1


def test_neighbour_sampler_retries_numeric_zscore_outliers():
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator=_FixedNeighbourEstimator(),
        probability_estimator=_FixedProbabilityEstimator(),
        use_min_max_constraints=False,
        use_numeric_std_constraints=True,
        numeric_std_threshold=1.0,
        numeric_constraint_indices=[0],
        max_constraint_retries=3,
        random_state=1,
    ).fit(np.array([[0.0], [1.0]]))
    candidates = iter([np.array([1.2]), np.array([0.75])])
    sampler._sample_anchor_index = lambda sampling_probability: 0
    sampler._generate_from_anchor = lambda *args, **kwargs: next(candidates)

    generated = sampler.sample(1)

    assert generated.tolist() == [[0.75]]
    assert sampler.constraint_retry_count_ == 1
    assert sampler.constraint_violation_count_ == 0


def test_exact_and_sklearn_knn_backends_return_neighbours_without_self():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])

    exact = find_nearest_neighbours(X, n_neighbours=1, backend="exact")
    sklearn = find_nearest_neighbours(X, n_neighbours=1, backend="sklearn")

    assert exact.tolist() == [[1], [0], [3], [2]]
    assert sklearn.tolist() == [[1], [0], [3], [2]]


def test_unknown_knn_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown KNN backend"):
        find_nearest_neighbours(np.array([[0.0], [1.0]]), backend="unknown")


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
            "--n_components",
            "2",
            "--n_iterations",
            "1",
            "--n_neighbours",
            "3",
            "--random_state",
            "3",
            "--no_calibrate_decoders",
        ],
    )

    assert result.exit_code == 0, result.output
    generated = pd.read_csv(output_csv)
    assert len(generated) == 6
    assert list(generated.columns) == list(make_mixed_dataframe().columns)
    assert model_file.exists()


def test_cli_help_shows_new_flags_and_hides_removed_flags():
    result = CliRunner().invoke(dataframe_sampler_main, ["--help"])

    assert result.exit_code == 0
    assert "--n_components" in result.output
    assert "--n_iterations" in result.output
    assert "--nca_fit_sample_size" in result.output
    assert "--calibrate_decoders" in result.output
    assert "--no_calibrate_decoders" in result.output
    assert "--enforce_min_max_constraints" in result.output
    assert "--no_enforce_min_max_constraints" in result.output
    assert "--enforce_numeric_std_constraints" in result.output
    assert "--no_enforce_numeric_std_constraints" in result.output
    assert "--numeric_std_threshold" in result.output
    assert "--max_constraint_retries" in result.output
    assert "--n_bins" not in result.output
    assert "--embedding_method" not in result.output
    assert "--auto_config" not in result.output


def test_read_and_write_dataframe_csv(tmp_path):
    path = tmp_path / "frame.csv"
    df = make_mixed_dataframe()

    write_dataframe(df, path)
    loaded = read_dataframe(path)

    assert loaded.shape == df.shape
