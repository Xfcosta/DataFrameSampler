import numpy as np

from .knn import find_nearest_neighbours


class NearestMutualNeighboursEstimator(object):
    def __init__(self, n_neighbours=10, metric="euclidean", knn_backend="sklearn", knn_backend_kwargs=None):
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1.")
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.knn_backend = knn_backend
        self.knn_backend_kwargs = dict(knn_backend_kwargs or {})

    def dist(self, objects, metric="euclidean", diagval=np.inf):
        objects = self._validate_X(objects)
        from scipy.spatial.distance import pdist, squareform

        out = squareform(pdist(objects, metric=metric)) if objects.shape[0] >= 2 else np.zeros((objects.shape[0], objects.shape[0]))
        np.fill_diagonal(out, diagval)
        return out

    def fit(self, X, y=None):
        self._validate_X(X)
        return self

    def fit_predict_single(self, X):
        X = self._validate_X(X)
        if X.shape[0] < 2:
            return [np.array([], dtype=int) for _ in range(X.shape[0])]

        nearest_neighbours = find_nearest_neighbours(
            X,
            n_neighbours=self.n_neighbours,
            metric=self.metric,
            backend=self.knn_backend,
            backend_kwargs=self.knn_backend_kwargs,
        )

        k_nearest_mutual_neighbours_mask = np.zeros((X.shape[0], X.shape[0]), bool)
        for _mask_row, _neighbours_row in zip(k_nearest_mutual_neighbours_mask, nearest_neighbours):
            _mask_row[_neighbours_row] = True

        k_nearest_mutual_neighbours_mask &= k_nearest_mutual_neighbours_mask.T
        k_nearest_mutual_neighbours = [np.where(row)[0] for row in k_nearest_mutual_neighbours_mask]

        for idx, neighbours in enumerate(k_nearest_mutual_neighbours):
            if len(neighbours) == 0:
                k_nearest_mutual_neighbours[idx] = np.array([nearest_neighbours[idx, 0]], dtype=int)

        return k_nearest_mutual_neighbours

    def fit_predict(self, X, y=None):
        X = self._validate_X(X)
        if y is None:
            return self.fit_predict_single(X)

        targets = np.asarray(y)
        if len(targets) != X.shape[0]:
            raise ValueError("y must have the same number of rows as X.")

        k_nearest_mutual_neighbours = [[] for _ in range(len(targets))]
        for target in sorted(set(targets)):
            targets_mask = targets == target
            idxs = np.where(targets_mask)[0]
            local_neighbours = self.fit_predict_single(X[targets_mask])
            for idx, ngbs in zip(idxs, local_neighbours):
                k_nearest_mutual_neighbours[idx] = np.array([idxs[ngb] for ngb in ngbs], dtype=int)

        return k_nearest_mutual_neighbours

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) == 0:
            raise ValueError("X must contain at least one row.")
        return X


class NearestMutualNeighboursProbabilityEstimator(object):
    def __init__(self, n_neighbours=10, metric="euclidean", knn_backend="sklearn", knn_backend_kwargs=None):
        self.nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(
            n_neighbours,
            metric,
            knn_backend=knn_backend,
            knn_backend_kwargs=knn_backend_kwargs,
        )

    def fit(self, X, y=None):
        self.nearest_mutual_neighbours_estimator.fit(X, y)
        return self

    def fit_predict_proba_single(self, X):
        k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)
        p = np.array(
            [
                len(neighbours) / self.nearest_mutual_neighbours_estimator.n_neighbours
                for neighbours in k_nearest_mutual_neighbours
            ],
            dtype=float,
        )
        if np.sum(p) == 0:
            return np.ones(len(p)) / len(p)
        return p / np.sum(p)

    def fit_predict_proba(self, X, y=None):
        if y is None:
            return self.fit_predict_proba_single(X)

        targets = np.asarray(y)
        sampling_probability = np.zeros(len(y))
        for target in sorted(set(targets)):
            targets_mask = targets == target
            sampling_probability[targets_mask] = self.fit_predict_proba_single(np.asarray(X)[targets_mask])

        if np.sum(sampling_probability) == 0:
            return np.ones(len(targets)) / len(targets)
        return sampling_probability / np.sum(sampling_probability)


class ProbabilityEstimator(object):
    def __init__(self, probability_estimators=None):
        self.probability_estimators = list(probability_estimators or [])

    def fit(self, X, y=None):
        self.probability_estimators = [
            probability_estimator.fit(X, y) for probability_estimator in self.probability_estimators
        ]
        return self

    def fit_predict_proba(self, X, y=None):
        if not self.probability_estimators:
            return np.ones(len(X)) / len(X)
        probs_mtx = np.array(
            [probability_estimator.fit_predict_proba(X, y) for probability_estimator in self.probability_estimators]
        ).T
        p = np.prod(probs_mtx, axis=1)
        if np.sum(p) == 0:
            return np.ones(len(p)) / len(p)
        return p / np.sum(p)
