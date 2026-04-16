import copy
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from .encoding import DataFrameEncoderDecoder
from .io import write_dataframe
from .metrics import compute_symmetrized_kullback_leibler_divergence, divergence_ttest
from .neighbours import NearestMutualNeighboursEstimator, NearestMutualNeighboursProbabilityEstimator
from .utils import make_random_state, random_choice, random_integers, random_uniform
from .vectorizer import DataFrameVectorizer


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
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] < 2:
            raise ValueError("At least two rows are required for neighbour sampling.")

        self.data_mtx = copy.deepcopy(X)
        self.targets = None if y is None else copy.deepcopy(np.asarray(y))
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
        sampling_probability = copy.deepcopy(self.sampling_probability)
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


class DataFrameSampler(object):
    def __init__(self, dataframe_vectorizer=None, dataframe_encoder_decoder=None, sampler=None, sampled_columns=None):
        self.dataframe_vectorizer = dataframe_vectorizer
        self.dataframe_encoder_decoder = dataframe_encoder_decoder
        self.sampler = sampler
        self.sampled_columns = sampled_columns
        self.latent_data_mtx = None

    def fit(self, orig_dataframe):
        if not isinstance(orig_dataframe, pd.DataFrame):
            raise TypeError("orig_dataframe must be a pandas DataFrame.")
        if orig_dataframe.empty:
            raise ValueError("orig_dataframe must contain at least one row.")

        dataframe = orig_dataframe.copy()
        vectorizing_dataframe = self.dataframe_vectorizer.fit_transform(dataframe)

        if self.sampled_columns is not None:
            missing_columns = [column for column in self.sampled_columns if column not in dataframe.columns]
            if missing_columns:
                raise ValueError("Unknown sampled columns: %s" % missing_columns)
            vectorizing_dataframe = vectorizing_dataframe[self.sampled_columns]
            dataframe = dataframe[self.sampled_columns]

        self.dataframe_encoder_decoder.fit(dataframe, vectorizing_dataframe)
        self.latent_data_mtx = self.dataframe_encoder_decoder.encode(vectorizing_dataframe)
        self.sampler.fit(self.latent_data_mtx)
        return self

    def transform(self, dataframe):
        if self.latent_data_mtx is None:
            raise ValueError("Sampler is not fit.")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        vectorizing_dataframe = self.dataframe_vectorizer.transform(dataframe)
        if self.sampled_columns is not None:
            missing_columns = [column for column in self.sampled_columns if column not in vectorizing_dataframe.columns]
            if missing_columns:
                raise ValueError("Unknown sampled columns: %s" % missing_columns)
            vectorizing_dataframe = vectorizing_dataframe[self.sampled_columns]
        return vectorizing_dataframe

    def sample(self, n_samples):
        if self.latent_data_mtx is None:
            raise ValueError("Sampler is not fit.")
        generated_latent_data_mtx = self.sampler.sample(n_samples=n_samples)
        self.generated_latent_data_mtx = generated_latent_data_mtx
        generated_df = self.dataframe_encoder_decoder.decode(self.generated_latent_data_mtx)
        return generated_df

    def sample_to_file(self, n_samples, filename="data.csv"):
        generated_df = self.sample(n_samples)
        return write_dataframe(generated_df, filename)

    def quality_score_(self, grid_nsteps=20, n_components=2):
        if self.latent_data_mtx is None:
            raise ValueError("Sampler is not fit.")
        generated_latent_data_mtx = self.sampler.sample(n_samples=self.latent_data_mtx.shape[0])
        return compute_symmetrized_kullback_leibler_divergence(
            self.latent_data_mtx,
            generated_latent_data_mtx,
            grid_nsteps=grid_nsteps,
            n_components=n_components,
        )

    def quality_score(self, grid_nsteps=20, n_components=2, n_iter=10):
        if self.latent_data_mtx is None:
            raise ValueError("Sampler is not fit.")
        n_instances = self.latent_data_mtx.shape[0]
        real_divergences = []
        for _ in range(n_iter):
            idxs1 = random_integers(self.sampler.rng, n_instances, size=n_instances // 2)
            idxs2 = random_integers(self.sampler.rng, n_instances, size=n_instances // 2)
            real_divergences.append(
                compute_symmetrized_kullback_leibler_divergence(
                    self.latent_data_mtx[idxs1],
                    self.latent_data_mtx[idxs2],
                    grid_nsteps=grid_nsteps,
                    n_components=n_components,
                )
            )
        generated_divergences = [
            self.quality_score_(grid_nsteps=grid_nsteps, n_components=n_components) for _ in range(n_iter)
        ]
        return divergence_ttest(generated_divergences, real_divergences)

    def plot(self, dataframe, filename="df"):
        sns_figure = sns.pairplot(dataframe, diag_kind="hist", corner=True)
        sns_figure.map_lower(sns.kdeplot, levels=5, color=".8")
        sns_figure.figure.savefig(filename + ".png", format="png")
        sns_figure.figure.savefig(filename + ".svg", format="svg", dpi=1200)

    def save(self, filename="model.obj"):
        with open(filename, "wb") as filehandler:
            pickle.dump(self, filehandler)
        return self

    def load(self, filename="model.obj"):
        with open(filename, "rb") as filehandler:
            return pickle.load(filehandler)


def ConcreteDataFrameSampler(
    n_bins=20,
    n_neighbours=10,
    vectorizing_columns_dict=None,
    sampled_columns=None,
    random_state=None,
    knn_backend="exact",
    knn_backend_kwargs=None,
    embedding_method="mds",
    embedding_kwargs=None,
    numeric_decode_strategy="observed_bin",
):
    dataframe_vectorizer = DataFrameVectorizer(
        vectorizing_columns_dict=vectorizing_columns_dict,
        random_state=random_state,
        embedding_method=embedding_method,
        embedding_kwargs=embedding_kwargs,
    )
    dataframe_encoder_decoder = DataFrameEncoderDecoder(
        n_bins=n_bins,
        strategy="uniform",
        random_state=random_state,
        numeric_decode_strategy=numeric_decode_strategy,
    )
    sampler = ConcreteNearestMutualNeighboursSampler(
        n_neighbours=n_neighbours,
        interpolation_factor=1,
        min_interpolation_factor=1,
        metric="euclidean",
        use_min_max_constraints=True,
        random_state=random_state,
        knn_backend=knn_backend,
        knn_backend_kwargs=knn_backend_kwargs,
    )
    return DataFrameSampler(
        dataframe_vectorizer=dataframe_vectorizer,
        dataframe_encoder_decoder=dataframe_encoder_decoder,
        sampler=sampler,
        sampled_columns=sampled_columns,
    )
