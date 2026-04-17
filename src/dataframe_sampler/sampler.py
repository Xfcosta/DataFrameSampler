import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
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
        use_min_max_constraints=True,
        use_numeric_std_constraints=True,
        numeric_std_threshold=3.0,
        numeric_constraint_indices=None,
        max_constraint_retries=5,
        random_state=None,
        max_attempts_factor=20,
    ):
        if interpolation_factor < min_interpolation_factor:
            raise ValueError("interpolation_factor must be >= min_interpolation_factor.")
        if numeric_std_threshold <= 0:
            raise ValueError("numeric_std_threshold must be positive.")
        if max_constraint_retries < 0:
            raise ValueError("max_constraint_retries must be non-negative.")
        self.nearest_mutual_neighbours_estimator = nearest_mutual_neighbours_estimator
        self.probability_estimator = probability_estimator
        self.interpolation_factor = interpolation_factor
        self.min_interpolation_factor = min_interpolation_factor
        self.use_min_max_constraints = use_min_max_constraints
        self.use_numeric_std_constraints = use_numeric_std_constraints
        self.numeric_std_threshold = numeric_std_threshold
        self.numeric_constraint_indices = (
            None if numeric_constraint_indices is None else np.asarray(numeric_constraint_indices, dtype=int)
        )
        self.max_constraint_retries = max_constraint_retries
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
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self._fit_numeric_std_constraints(X)
        self.targets = None if y is None else np.asarray(y).copy()
        self.sampling_probability = self.probability_estimator.fit_predict_proba(X, y)
        self.k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X, y)
        self.valid_anchor_indices = self._valid_anchor_indices(self.k_nearest_mutual_neighbours)
        self.constraint_retry_count_ = 0
        self.constraint_violation_count_ = 0
        if len(self.valid_anchor_indices) == 0:
            raise ValueError("No valid mutual-neighbour chains were found for sampling.")
        return self

    def _sample_anchor_index(self, sampling_probability):
        eligible_probabilities = sampling_probability[self.valid_anchor_indices].astype(float)
        if np.sum(eligible_probabilities) == 0:
            eligible_probabilities = np.ones(len(self.valid_anchor_indices)) / len(self.valid_anchor_indices)
        else:
            eligible_probabilities = eligible_probabilities / np.sum(eligible_probabilities)
        return random_choice(self.rng, self.valid_anchor_indices, p=eligible_probabilities)

    def generate(self, data_mtx, k_nearest_mutual_neighbours, sampling_probability, interpolation_factor, min_interpolation_factor):
        idx1 = self._sample_anchor_index(sampling_probability)
        candidate = None
        retries = self.max_constraint_retries if self._uses_rejection_constraints() else 0
        for attempt in range(retries + 1):
            candidate = self._generate_from_anchor(
                idx1,
                data_mtx,
                k_nearest_mutual_neighbours,
                interpolation_factor,
                min_interpolation_factor,
            )
            if not self._uses_rejection_constraints() or self._satisfies_rejection_constraints(candidate):
                self.constraint_retry_count_ += attempt
                return candidate
        self.constraint_retry_count_ += retries
        self.constraint_violation_count_ += 1
        return candidate

    def _generate_from_anchor(
        self,
        idx1,
        data_mtx,
        k_nearest_mutual_neighbours,
        interpolation_factor,
        min_interpolation_factor,
    ):
        idx2_candidates = [
            idx for idx in k_nearest_mutual_neighbours[idx1] if len(k_nearest_mutual_neighbours[idx]) > 0
        ]
        idx2 = random_choice(self.rng, idx2_candidates)
        idx3 = random_choice(self.rng, k_nearest_mutual_neighbours[idx2])

        alpha = random_uniform(self.rng) * (interpolation_factor - min_interpolation_factor) + min_interpolation_factor
        return data_mtx[idx1] + alpha * (data_mtx[idx3] - data_mtx[idx2])

    def _uses_rejection_constraints(self):
        return bool(
            self.use_min_max_constraints
            or (self.use_numeric_std_constraints and self.numeric_constraint_indices is not None)
        )

    def _satisfies_rejection_constraints(self, candidate):
        if self.use_min_max_constraints and not self._satisfies_min_max_constraints(candidate):
            return False
        if self.use_numeric_std_constraints and not self._satisfies_numeric_std_constraints(candidate):
            return False
        return True

    def _satisfies_min_max_constraints(self, candidate):
        return bool(np.all(candidate >= self.data_min_) and np.all(candidate <= self.data_max_))

    def _fit_numeric_std_constraints(self, X):
        if self.numeric_constraint_indices is None or len(self.numeric_constraint_indices) == 0:
            self.numeric_constraint_indices = None
            self.numeric_constraint_mean_ = None
            self.numeric_constraint_std_ = None
            return
        numeric = X[:, self.numeric_constraint_indices]
        self.numeric_constraint_mean_ = np.mean(numeric, axis=0)
        std = np.std(numeric, axis=0)
        self.numeric_constraint_std_ = np.where(std > 0, std, 1.0)

    def _satisfies_numeric_std_constraints(self, candidate):
        if self.numeric_constraint_indices is None:
            return True
        numeric = candidate[self.numeric_constraint_indices]
        z_scores = np.abs((numeric - self.numeric_constraint_mean_) / self.numeric_constraint_std_)
        return bool(np.all(z_scores <= self.numeric_std_threshold))

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

        return np.array(sampled_data_mtx)

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
    use_min_max_constraints=True,
    use_numeric_std_constraints=True,
    numeric_std_threshold=3.0,
    numeric_constraint_indices=None,
    max_constraint_retries=5,
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
        use_numeric_std_constraints=use_numeric_std_constraints,
        numeric_std_threshold=numeric_std_threshold,
        numeric_constraint_indices=numeric_constraint_indices,
        max_constraint_retries=max_constraint_retries,
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


class PriorCategoricalClassifier(object):
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.probabilities_ = counts.astype(float) / counts.sum()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.tile(self.probabilities_, (X.shape[0], 1))


class DataFrameSampler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        n_iterations=1,
        n_neighbours=10,
        lambda_=1.0,
        knn_backend="exact",
        knn_backend_kwargs=None,
        random_state=None,
        nca_kwargs=None,
        decoder_kwargs=None,
        calibrate_decoders=False,
        calibration_kwargs=None,
        enforce_min_max_constraints=True,
        enforce_numeric_std_constraints=True,
        numeric_std_threshold=3.0,
        max_constraint_retries=5,
    ):
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1.")
        if numeric_std_threshold <= 0:
            raise ValueError("numeric_std_threshold must be positive.")
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.n_neighbours = n_neighbours
        self.lambda_ = lambda_
        self.knn_backend = knn_backend
        self.knn_backend_kwargs = dict(knn_backend_kwargs or {})
        self.random_state = random_state
        self.nca_kwargs = dict(nca_kwargs or {})
        self.decoder_kwargs = dict(decoder_kwargs or {})
        self.calibrate_decoders = calibrate_decoders
        self.calibration_kwargs = dict(calibration_kwargs or {})
        self.enforce_min_max_constraints = enforce_min_max_constraints
        self.enforce_numeric_std_constraints = enforce_numeric_std_constraints
        self.numeric_std_threshold = numeric_std_threshold
        self.max_constraint_retries = max_constraint_retries
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
        self.latent_block_slices_ = self._block_slices_for_blocks(self._fit_blocks_)
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
        if self.numeric_columns_:
            numeric_slice = self.latent_block_slices_["__numeric__"]
            numeric_scaled = latent[:, numeric_slice]
            numeric_values = self.numeric_scaler_.inverse_transform(numeric_scaled)
            for idx, column in enumerate(self.numeric_columns_):
                data[column] = numeric_values[:, idx]

        for column in self.categorical_columns_:
            decoder = self.decoders_[column]
            context = self._decoder_input_from_latent(column, latent)
            labels = self._decode_categorical_column(column, decoder, context, sample=sample)
            data[column] = labels

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
        self.decoder_calibration_status_ = {}
        for column in self.categorical_columns_:
            context = self._decoder_input_from_latent(column, self.latent_data_mtx_)
            labels = self._categorical_keys(dataframe[column]).to_numpy(dtype=str)
            decoder = self._fit_decoder(column, context, labels)
            self.decoders_[column] = decoder

    def _fit_decoder(self, column, context, labels):
        if context.shape[1] == 0:
            decoder = PriorCategoricalClassifier().fit(context, labels)
            self.decoder_calibration_status_[column] = "prior_only"
            return decoder

        kwargs = {
            "n_estimators": 100,
            "min_samples_leaf": 1,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        kwargs.update(self.decoder_kwargs)
        decoder = RandomForestClassifier(**kwargs)
        if not self.calibrate_decoders:
            decoder.fit(context, labels)
            self.decoder_calibration_status_[column] = "disabled"
            return decoder

        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            decoder.fit(context, labels)
            self.decoder_calibration_status_[column] = "skipped_one_class"
            return decoder

        calibration_kwargs = dict(self.calibration_kwargs)
        if "cv" not in calibration_kwargs:
            min_class_count = int(np.min(counts))
            if min_class_count < 2:
                decoder.fit(context, labels)
                self.decoder_calibration_status_[column] = "skipped_insufficient_class_count"
                return decoder
            calibration_kwargs["cv"] = min(5, min_class_count)
        calibration_kwargs.setdefault("method", "sigmoid")
        calibration_kwargs.setdefault("n_jobs", kwargs.get("n_jobs", -1))
        calibrated = _calibrated_classifier(decoder, calibration_kwargs)
        try:
            calibrated.fit(context, labels)
        except Exception as exc:
            warnings.warn(
                "Falling back to an uncalibrated random-forest decoder for categorical column %r "
                "because calibration failed: %s" % (column, exc),
                RuntimeWarning,
                stacklevel=2,
            )
            decoder = RandomForestClassifier(**kwargs)
            decoder.fit(context, labels)
            self.decoder_calibration_status_[column] = "failed"
            return decoder
        self.decoder_calibration_status_[column] = "calibrated"
        return calibrated

    def _fit_generator(self):
        if self.n_fit_rows_ < 2:
            self.generator_ = LatentBootstrapSampler(random_state=self.random_state).fit(self.latent_data_mtx_)
            return
        generator = ConcreteNearestMutualNeighboursSampler(
            n_neighbours=min(self.n_neighbours, max(1, self.n_fit_rows_ - 1)),
            interpolation_factor=self.lambda_,
            min_interpolation_factor=self.lambda_,
            metric="euclidean",
            use_min_max_constraints=self.enforce_min_max_constraints,
            use_numeric_std_constraints=self.enforce_numeric_std_constraints,
            numeric_std_threshold=self.numeric_std_threshold,
            numeric_constraint_indices=np.arange(len(self.numeric_columns_), dtype=int)
            if self.numeric_columns_
            else None,
            max_constraint_retries=self.max_constraint_retries,
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

    def _decode_categorical_column(self, column, decoder, context, sample):
        probabilities = decoder.predict_proba(context)
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
        for key in self._ordered_block_keys(blocks):
            ordered.append(blocks[key])
        if not ordered:
            return np.empty((self.n_fit_rows_, 0), dtype=float)
        return np.asarray(np.hstack(ordered), dtype=float)

    def _decoder_input_from_latent(self, column, latent):
        slices = [
            block_slice
            for key, block_slice in self.latent_block_slices_.items()
            if key != column
        ]
        if not slices:
            return np.empty((latent.shape[0], 0), dtype=float)
        return np.hstack([latent[:, block_slice] for block_slice in slices])

    def _block_slices_for_blocks(self, blocks):
        slices = {}
        offset = 0
        for key in self._ordered_block_keys(blocks):
            width = blocks[key].shape[1]
            slices[key] = slice(offset, offset + width)
            offset += width
        return slices

    def _ordered_block_keys(self, blocks):
        keys = []
        if self.numeric_columns_ and "__numeric__" in blocks:
            keys.append("__numeric__")
        keys.extend([column for column in self.categorical_columns_ if column in blocks])
        return keys

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
                    "Categorical column %r has %d unique values. The sampler will use it, "
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


def _calibrated_classifier(estimator, kwargs):
    parameters = inspect.signature(CalibratedClassifierCV).parameters
    if "estimator" in parameters:
        return CalibratedClassifierCV(estimator=estimator, **kwargs)
    return CalibratedClassifierCV(base_estimator=estimator, **kwargs)


def _as_float_array(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)
