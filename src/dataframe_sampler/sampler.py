import pickle
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import write_dataframe
from .neighbours import NearestMutualNeighboursEstimator, NearestMutualNeighboursProbabilityEstimator
from .utils import make_random_state, random_choice, random_uniform


MISSING_CATEGORY = "__MISSING__"
HIGH_CARDINALITY_UNIQUE = 50
HIGH_CARDINALITY_FRACTION = 0.3


class NearestMutualNeighboursSampler(object):
    def __init__(
        self,
        nearest_mutual_neighbours_estimator=None,
        probability_estimator=None,
        interpolation_factor=1,
        min_interpolation_factor=1,
        use_min_max_constraints=False,
        random_state=None,
        max_attempts_factor=20,
    ):
        if interpolation_factor < min_interpolation_factor:
            raise ValueError("interpolation_factor must be >= min_interpolation_factor.")
        self.nearest_mutual_neighbours_estimator = nearest_mutual_neighbours_estimator
        self.probability_estimator = probability_estimator
        self.interpolation_factor = interpolation_factor
        self.min_interpolation_factor = min_interpolation_factor
        self.use_min_max_constraints = use_min_max_constraints
        self.random_state = random_state
        self.rng = make_random_state(random_state)
        self.max_attempts_factor = max_attempts_factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] < 2:
            raise ValueError("At least two rows are required for neighbour sampling.")

        self.data_mtx = X.copy()
        self.targets = None if y is None else np.asarray(y).copy()
        self.sampling_probability = self.probability_estimator.fit_predict_proba(X, y)
        self.k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X, y)
        self.valid_anchor_indices = self._valid_anchor_indices(self.k_nearest_mutual_neighbours)
        if len(self.valid_anchor_indices) == 0:
            raise ValueError("No valid mutual-neighbour chains were found for sampling.")
        return self

    def generate(self, data_mtx, k_nearest_mutual_neighbours, sampling_probability, interpolation_factor, min_interpolation_factor):
        eligible_probabilities = sampling_probability[self.valid_anchor_indices].astype(float)
        if np.sum(eligible_probabilities) == 0:
            eligible_probabilities = np.ones(len(self.valid_anchor_indices)) / len(self.valid_anchor_indices)
        else:
            eligible_probabilities = eligible_probabilities / np.sum(eligible_probabilities)

        idx1 = random_choice(self.rng, self.valid_anchor_indices, p=eligible_probabilities)
        idx2_candidates = [
            idx for idx in k_nearest_mutual_neighbours[idx1] if len(k_nearest_mutual_neighbours[idx]) > 0
        ]
        idx2 = random_choice(self.rng, idx2_candidates)
        idx3 = random_choice(self.rng, k_nearest_mutual_neighbours[idx2])

        alpha = random_uniform(self.rng) * (interpolation_factor - min_interpolation_factor) + min_interpolation_factor
        return data_mtx[idx1] + alpha * (data_mtx[idx3] - data_mtx[idx2])

    def min_max_constraints(self, X, Xp):
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)
        return np.clip(Xp, mn, mx)

    def sample(self, n_samples, target=None):
        if not hasattr(self, "data_mtx"):
            raise ValueError("Sampler is not fit.")
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        if n_samples == 0:
            return np.empty((0, self.data_mtx.shape[1]))

        sampling_probability = self._target_sampling_probability(target)
        sampled_data_mtx = []
        max_attempts = max(n_samples * self.max_attempts_factor, n_samples)
        attempts = 0
        while len(sampled_data_mtx) < n_samples and attempts < max_attempts:
            attempts += 1
            sampled_data_mtx.append(
                self.generate(
                    self.data_mtx,
                    self.k_nearest_mutual_neighbours,
                    sampling_probability,
                    self.interpolation_factor,
                    self.min_interpolation_factor,
                )
            )

        if len(sampled_data_mtx) != n_samples:
            raise RuntimeError("Generated %d of %d requested samples." % (len(sampled_data_mtx), n_samples))

        sampled_data_mtx = np.array(sampled_data_mtx)
        if self.use_min_max_constraints:
            sampled_data_mtx = self.min_max_constraints(self.data_mtx, sampled_data_mtx)
        return sampled_data_mtx

    def _target_sampling_probability(self, target):
        if target is None:
            return self.sampling_probability
        if self.targets is None:
            raise ValueError("Sampler was not fit with targets.")
        sampling_probability = self.sampling_probability.copy()
        sampling_probability[self.targets != target] = 0
        if np.sum(sampling_probability) == 0:
            raise ValueError("No fitted rows match target %r." % target)
        return sampling_probability / np.sum(sampling_probability)

    @staticmethod
    def _valid_anchor_indices(k_nearest_mutual_neighbours):
        return np.array(
            [
                i
                for i, neighbours in enumerate(k_nearest_mutual_neighbours)
                if len(neighbours) > 0
                and any(len(k_nearest_mutual_neighbours[neighbour]) > 0 for neighbour in neighbours)
            ],
            dtype=int,
        )


def ConcreteNearestMutualNeighboursSampler(
    n_neighbours=10,
    interpolation_factor=1,
    min_interpolation_factor=1,
    metric="euclidean",
    use_min_max_constraints=False,
    random_state=None,
    knn_backend="exact",
    knn_backend_kwargs=None,
):
    knn_backend_kwargs = dict(knn_backend_kwargs or {})
    if random_state is not None and knn_backend == "pynndescent":
        knn_backend_kwargs.setdefault("random_state", random_state)
    if random_state is not None and knn_backend == "hnswlib":
        init_kwargs = dict(knn_backend_kwargs.get("init_kwargs", {}))
        init_kwargs.setdefault("random_seed", random_state)
        knn_backend_kwargs["init_kwargs"] = init_kwargs
    if random_state is not None and knn_backend == "annoy":
        knn_backend_kwargs.setdefault("random_state", random_state)

    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(
        n_neighbours,
        metric,
        knn_backend=knn_backend,
        knn_backend_kwargs=knn_backend_kwargs,
    )
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(
        n_neighbours,
        metric,
        knn_backend=knn_backend,
        knn_backend_kwargs=knn_backend_kwargs,
    )
    return NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints,
        random_state=random_state,
    )


class ConstantProjector(object):
    def __init__(self, n_components):
        self.n_components = int(n_components)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.zeros((X.shape[0], self.n_components), dtype=float)


class PaddedProjector(object):
    def __init__(self, estimator, n_components):
        self.estimator = estimator
        self.n_components = int(n_components)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        projected = np.asarray(self.estimator.transform(X), dtype=float)
        if projected.ndim == 1:
            projected = projected.reshape(-1, 1)
        if projected.shape[1] == self.n_components:
            return projected
        padded = np.zeros((projected.shape[0], self.n_components), dtype=float)
        padded[:, : projected.shape[1]] = projected
        return padded


class LatentBootstrapSampler(object):
    def __init__(self, random_state=None):
        self.rng = make_random_state(random_state)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] < 1:
            raise ValueError("At least one row is required for bootstrap sampling.")
        self.data_mtx = X.copy()
        return self

    def sample(self, n_samples, target=None):
        if target is not None:
            raise ValueError("Latent bootstrap sampling does not support targets.")
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        if n_samples == 0:
            return np.empty((0, self.data_mtx.shape[1]))
        indices = self.rng.choice(self.data_mtx.shape[0], size=n_samples, replace=True)
        return self.data_mtx[indices].copy()


class DataFrameSampler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        n_iterations=2,
        n_neighbours=10,
        lambda_=1.0,
        knn_backend="exact",
        knn_backend_kwargs=None,
        random_state=None,
        nca_kwargs=None,
        decoder_kwargs=None,
    ):
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1.")
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1.")
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.n_neighbours = n_neighbours
        self.lambda_ = lambda_
        self.knn_backend = knn_backend
        self.knn_backend_kwargs = dict(knn_backend_kwargs or {})
        self.random_state = random_state
        self.nca_kwargs = dict(nca_kwargs or {})
        self.decoder_kwargs = dict(decoder_kwargs or {})
        self.rng = make_random_state(random_state)

    def fit(self, X, y=None):
        dataframe = self._validate_dataframe(X, name="X")
        self.input_columns_ = list(dataframe.columns)
        self.dtypes_ = dataframe.dtypes.to_dict()
        self.n_fit_rows_ = len(dataframe)
        self.numeric_columns_, self.categorical_columns_ = self._split_columns(dataframe)
        self._warn_high_cardinality(dataframe)

        self._fit_numeric(dataframe)
        self.category_value_maps_ = {}
        initial_blocks = self._fit_initial_blocks(dataframe)
        self._fit_iterative_projectors(initial_blocks)
        self.latent_data_mtx_ = self._blocks_to_latent(self._fit_blocks_)
        self.latent_dim_ = int(self.latent_data_mtx_.shape[1])
        self._fit_decoders(dataframe)
        self._fit_generator()
        return self

    def transform(self, X):
        self._ensure_fit()
        dataframe = self._validate_dataframe(X, name="X")
        missing = [column for column in self.input_columns_ if column not in dataframe.columns]
        if missing:
            raise ValueError("Missing columns for fitted sampler: %s" % missing)
        blocks = self._initial_blocks_for(dataframe[self.input_columns_])
        blocks = self._apply_projector_steps(blocks)
        return self._blocks_to_latent(blocks)

    def inverse_transform(self, Z, sample=True):
        self._ensure_fit()
        latent = self._validate_latent(Z)
        data = {}
        offset = 0

        if self.numeric_columns_:
            n_numeric = len(self.numeric_columns_)
            numeric_scaled = latent[:, offset : offset + n_numeric]
            numeric_values = self.numeric_scaler_.inverse_transform(numeric_scaled)
            for idx, column in enumerate(self.numeric_columns_):
                data[column] = numeric_values[:, idx]
            offset += n_numeric

        for column in self.categorical_columns_:
            width = self._components_for_column(column)
            block = latent[:, offset : offset + width]
            decoder = self.decoders_[column]
            labels = self._decode_categorical_block(column, decoder, block, sample=sample)
            data[column] = labels
            offset += width

        result = pd.DataFrame(data, columns=self.input_columns_)
        return self._restore_dtypes(result)

    def generate(self, n_samples=None):
        self._ensure_fit()
        if n_samples is None:
            n_samples = self.n_fit_rows_
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        generated_latent = self.generator_.sample(n_samples=n_samples)
        self.generated_latent_data_mtx_ = generated_latent
        return self.inverse_transform(generated_latent, sample=True)

    def generate_to_file(self, n_samples=None, filename="data.csv"):
        generated_df = self.generate(n_samples=n_samples)
        return write_dataframe(generated_df, filename)

    def save(self, filename="model.obj"):
        with open(filename, "wb") as filehandler:
            pickle.dump(self, filehandler)
        return self

    def load(self, filename="model.obj"):
        with open(filename, "rb") as filehandler:
            return pickle.load(filehandler)

    def _fit_numeric(self, dataframe):
        if not self.numeric_columns_:
            self.numeric_imputer_ = None
            self.numeric_scaler_ = None
            return
        numeric = dataframe[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        imputed = self.numeric_imputer_.fit_transform(numeric)
        self.numeric_scaler_ = StandardScaler()
        self.numeric_scaler_.fit(imputed)

    def _fit_initial_blocks(self, dataframe):
        self.categorical_encoders_ = {}
        self.category_labels_ = {}
        blocks = {}
        if self.numeric_columns_:
            blocks["__numeric__"] = self._numeric_block_for(dataframe)
        for column in self.categorical_columns_:
            labels = self._categorical_keys(dataframe[column])
            self.category_value_maps_[column] = self._category_value_map(dataframe[column], labels)
            self.category_labels_[column] = labels.to_numpy(dtype=str)
            encoder = _one_hot_encoder()
            encoded = _as_float_array(encoder.fit_transform(labels.to_frame(name=column)))
            self.categorical_encoders_[column] = encoder
            blocks[column] = encoded
        return blocks

    def _initial_blocks_for(self, dataframe):
        blocks = {}
        if self.numeric_columns_:
            blocks["__numeric__"] = self._numeric_block_for(dataframe)
        for column in self.categorical_columns_:
            labels = self._categorical_keys(dataframe[column])
            encoded = self.categorical_encoders_[column].transform(labels.to_frame(name=column))
            blocks[column] = _as_float_array(encoded)
        return blocks

    def _numeric_block_for(self, dataframe):
        numeric = dataframe[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
        imputed = self.numeric_imputer_.transform(numeric)
        return np.asarray(self.numeric_scaler_.transform(imputed), dtype=float)

    def _fit_iterative_projectors(self, blocks):
        self.projector_steps_ = []
        current_blocks = {key: value.copy() for key, value in blocks.items()}
        for iteration in range(self.n_iterations):
            for column in self.categorical_columns_:
                input_keys = [key for key in current_blocks.keys() if key != column]
                context = self._concat_blocks(current_blocks, input_keys, n_rows=self.n_fit_rows_)
                labels = self.category_labels_[column]
                projector = self._fit_projector(column, context, labels)
                projected = projector.transform(context)
                current_blocks[column] = projected
                self.projector_steps_.append(
                    {
                        "iteration": iteration,
                        "column": column,
                        "input_keys": input_keys,
                        "projector": projector,
                    }
                )
        self._fit_blocks_ = current_blocks

    def _apply_projector_steps(self, blocks):
        current_blocks = {key: value.copy() for key, value in blocks.items()}
        n_rows = self._block_row_count(current_blocks)
        for step in self.projector_steps_:
            context = self._concat_blocks(current_blocks, step["input_keys"], n_rows=n_rows)
            current_blocks[step["column"]] = step["projector"].transform(context)
        return current_blocks

    def _fit_projector(self, column, context, labels):
        width = self._components_for_column(column)
        unique_labels = pd.unique(pd.Series(labels))
        if context.shape[1] == 0 or len(unique_labels) < 2:
            return ConstantProjector(width).fit(context, labels)
        effective_width = min(width, context.shape[1])
        kwargs = dict(self.nca_kwargs)
        kwargs.setdefault("max_iter", 100)
        if self.random_state is not None:
            kwargs.setdefault("random_state", self.random_state)
        estimator = NeighborhoodComponentsAnalysis(n_components=effective_width, **kwargs)
        try:
            return PaddedProjector(estimator, width).fit(context, labels)
        except Exception as exc:
            warnings.warn(
                "Falling back to a constant latent block for categorical column %r because NCA failed: %s"
                % (column, exc),
                RuntimeWarning,
                stacklevel=2,
            )
            return ConstantProjector(width).fit(context, labels)

    def _fit_decoders(self, dataframe):
        self.decoders_ = {}
        offset = len(self.numeric_columns_)
        for column in self.categorical_columns_:
            width = self._components_for_column(column)
            block = self.latent_data_mtx_[:, offset : offset + width]
            labels = self._categorical_keys(dataframe[column]).to_numpy(dtype=str)
            kwargs = {
                "n_estimators": 100,
                "min_samples_leaf": 1,
                "random_state": self.random_state,
                "n_jobs": 1,
            }
            kwargs.update(self.decoder_kwargs)
            decoder = RandomForestClassifier(**kwargs)
            decoder.fit(block, labels)
            self.decoders_[column] = decoder
            offset += width

    def _fit_generator(self):
        if self.n_fit_rows_ < 2:
            self.generator_ = LatentBootstrapSampler(random_state=self.random_state).fit(self.latent_data_mtx_)
            return
        generator = ConcreteNearestMutualNeighboursSampler(
            n_neighbours=min(self.n_neighbours, max(1, self.n_fit_rows_ - 1)),
            interpolation_factor=self.lambda_,
            min_interpolation_factor=self.lambda_,
            metric="euclidean",
            use_min_max_constraints=True,
            random_state=self.random_state,
            knn_backend=self.knn_backend,
            knn_backend_kwargs=self.knn_backend_kwargs,
        )
        try:
            self.generator_ = generator.fit(self.latent_data_mtx_)
        except ValueError as exc:
            warnings.warn(
                "Falling back to latent bootstrap generation because mutual-neighbour fitting failed: %s"
                % exc,
                RuntimeWarning,
                stacklevel=2,
            )
            self.generator_ = LatentBootstrapSampler(random_state=self.random_state).fit(self.latent_data_mtx_)

    def _decode_categorical_block(self, column, decoder, block, sample):
        probabilities = decoder.predict_proba(block)
        classes = decoder.classes_
        if not sample:
            keys = classes[np.argmax(probabilities, axis=1)]
            return self._labels_from_keys(column, keys)
        keys = []
        for row in probabilities:
            probs = np.asarray(row, dtype=float)
            if probs.sum() == 0:
                probs = np.ones(len(classes), dtype=float) / len(classes)
            else:
                probs = probs / probs.sum()
            keys.append(random_choice(self.rng, classes, p=probs))
        return self._labels_from_keys(column, keys)

    def _blocks_to_latent(self, blocks):
        ordered = []
        if self.numeric_columns_:
            ordered.append(blocks["__numeric__"])
        for column in self.categorical_columns_:
            ordered.append(blocks[column])
        if not ordered:
            return np.empty((self.n_fit_rows_, 0), dtype=float)
        return np.asarray(np.hstack(ordered), dtype=float)

    @staticmethod
    def _concat_blocks(blocks, keys, n_rows):
        arrays = [blocks[key] for key in keys]
        if not arrays:
            return np.empty((n_rows, 0), dtype=float)
        return np.asarray(np.hstack(arrays), dtype=float)

    @staticmethod
    def _block_row_count(blocks):
        if not blocks:
            return 0
        first = next(iter(blocks.values()))
        return first.shape[0]

    def _components_for_column(self, column):
        if isinstance(self.n_components, dict):
            value = self.n_components.get(column, 2)
        else:
            value = self.n_components
        if value < 1:
            raise ValueError("n_components for column %r must be at least 1." % column)
        return int(value)

    def _split_columns(self, dataframe):
        numeric_columns = []
        categorical_columns = []
        for column in dataframe.columns:
            series = dataframe[column]
            if self._is_numeric_non_binary(series):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
        if not numeric_columns and not categorical_columns:
            raise ValueError("X must contain at least one column.")
        return numeric_columns, categorical_columns

    @staticmethod
    def _is_numeric_non_binary(series):
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            return False
        unique = pd.Series(series).dropna().unique()
        return len(unique) > 2

    def _warn_high_cardinality(self, dataframe):
        row_count = len(dataframe)
        limit = max(HIGH_CARDINALITY_UNIQUE, int(row_count * HIGH_CARDINALITY_FRACTION))
        for column in self.categorical_columns_:
            unique_count = self._categorical_keys(dataframe[column]).nunique(dropna=False)
            if unique_count > limit:
                warnings.warn(
                    "Categorical column %r has %d unique values. DataFrameSampler 2.0 will use it, "
                    "but high-cardinality columns should usually be preprocessed deliberately."
                    % (column, unique_count),
                    UserWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _categorical_keys(series):
        values = series.astype("object").where(series.notna(), MISSING_CATEGORY)
        return values.map(lambda value: MISSING_CATEGORY if value == MISSING_CATEGORY else str(value))

    @staticmethod
    def _category_value_map(series, keys):
        values = series.astype("object").where(series.notna(), pd.NA)
        mapping = {}
        for key, value in zip(keys, values):
            if key not in mapping:
                mapping[key] = value
        mapping.setdefault(MISSING_CATEGORY, pd.NA)
        return mapping

    def _labels_from_keys(self, column, keys):
        mapping = self.category_value_maps_[column]
        return np.asarray([mapping.get(key, pd.NA) for key in keys], dtype=object)

    def _restore_dtypes(self, dataframe):
        restored = dataframe.copy()
        for column in self.input_columns_:
            if column not in restored:
                continue
            restored[column] = restored[column].replace(MISSING_CATEGORY, pd.NA)
            dtype = self.dtypes_[column]
            try:
                if pd.api.types.is_integer_dtype(dtype) and restored[column].isna().any():
                    continue
                restored[column] = restored[column].astype(dtype)
            except (TypeError, ValueError):
                pass
        return restored[self.input_columns_]

    def _validate_latent(self, Z):
        latent = np.asarray(Z, dtype=float)
        if latent.ndim != 2:
            raise ValueError("Z must be a 2D array.")
        if latent.shape[1] != self.latent_dim_:
            raise ValueError("Z must have %d columns; got %d." % (self.latent_dim_, latent.shape[1]))
        return latent

    @staticmethod
    def _validate_dataframe(dataframe, name="dataframe"):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("%s must be a pandas DataFrame." % name)
        if dataframe.empty:
            raise ValueError("%s must contain at least one row." % name)
        return dataframe.copy()

    def _ensure_fit(self):
        if not hasattr(self, "latent_data_mtx_"):
            raise ValueError("Sampler is not fit.")


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_float_array(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)
