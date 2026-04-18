import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


SUPPORTED_KNN_BACKENDS = ("exact", "sklearn", "pynndescent", "hnswlib", "annoy")


def find_nearest_neighbours(X, n_neighbours=10, metric="euclidean", backend="sklearn", backend_kwargs=None):
    """
    Return nearest-neighbor indexes excluding each row itself.

    Parameters
    ----------
    X:
        Two-dimensional numeric matrix.
    n_neighbours:
        Maximum number of neighbors per row.
    metric:
        Distance metric understood by the selected backend.
    backend:
        One of: exact, sklearn, pynndescent, hnswlib, annoy.
    backend_kwargs:
        Optional backend-specific parameters.
    """
    X = _validate_X(X)
    if n_neighbours < 1:
        raise ValueError("n_neighbours must be at least 1.")
    if X.shape[0] < 2:
        return np.empty((X.shape[0], 0), dtype=int)

    backend_kwargs = dict(backend_kwargs or {})
    k = min(n_neighbours, X.shape[0] - 1)
    backend = backend.lower()

    if backend == "exact":
        return _exact_neighbours(X, k, metric)
    if backend == "sklearn":
        return _sklearn_neighbours(X, k, metric, backend_kwargs)
    if backend == "pynndescent":
        return _pynndescent_neighbours(X, k, metric, backend_kwargs)
    if backend == "hnswlib":
        return _hnswlib_neighbours(X, k, metric, backend_kwargs)
    if backend == "annoy":
        return _annoy_neighbours(X, k, metric, backend_kwargs)

    raise ValueError("Unknown KNN backend %r. Expected one of %s." % (backend, SUPPORTED_KNN_BACKENDS))


def _exact_neighbours(X, k, metric):
    distvec = pdist(X, metric=metric)
    distances = squareform(distvec)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances)[:, :k].astype(int)


def _sklearn_neighbours(X, k, metric, backend_kwargs):
    from sklearn.neighbors import NearestNeighbors

    kwargs = {"metric": metric}
    kwargs.update(backend_kwargs)
    estimator = NearestNeighbors(n_neighbors=k + 1, **kwargs).fit(X)
    indices = estimator.kneighbors(X, return_distance=False)
    return _drop_self_neighbours(indices, k, X, metric)


def _pynndescent_neighbours(X, k, metric, backend_kwargs):
    min_approx_size = backend_kwargs.pop("min_approx_size", 32)
    if X.shape[0] < min_approx_size:
        return _sklearn_neighbours(X, k, metric, {})

    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError(
            "knn_backend='pynndescent' requires the optional dependency. "
            "Install it with: pip install 'dataframe-sampler[pynndescent]'"
        ) from exc

    query_kwargs = dict(backend_kwargs.pop("query_kwargs", {}))
    index = NNDescent(X, metric=metric, n_neighbors=k + 1, **backend_kwargs)
    indices, _ = index.query(X, k=k + 1, **query_kwargs)
    return _drop_self_neighbours(indices, k, X, metric)


def _hnswlib_neighbours(X, k, metric, backend_kwargs):
    min_approx_size = backend_kwargs.pop("min_approx_size", 32)
    if X.shape[0] < min_approx_size:
        return _sklearn_neighbours(X, k, metric, {})

    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "knn_backend='hnswlib' requires the optional dependency. "
            "Install it with: pip install 'dataframe-sampler[hnswlib]'"
        ) from exc

    space = _hnswlib_space(metric)
    dim = X.shape[1]
    init_kwargs = dict(backend_kwargs.pop("init_kwargs", {}))
    index_kwargs = dict(backend_kwargs.pop("index_kwargs", {}))
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=X.shape[0], **{"ef_construction": 200, "M": 16, **init_kwargs})
    index.add_items(X.astype(np.float32), np.arange(X.shape[0]))
    index.set_ef(max(k + 1, backend_kwargs.pop("ef", 50)))
    labels, _ = index.knn_query(X.astype(np.float32), k=k + 1, **index_kwargs)
    return _drop_self_neighbours(labels, k, X, metric)


def _annoy_neighbours(X, k, metric, backend_kwargs):
    min_approx_size = backend_kwargs.pop("min_approx_size", 32)
    if X.shape[0] < min_approx_size:
        return _sklearn_neighbours(X, k, metric, {})

    try:
        from annoy import AnnoyIndex
    except ImportError as exc:
        raise ImportError(
            "knn_backend='annoy' requires the optional dependency. "
            "Install it with: pip install 'dataframe-sampler[annoy]'"
        ) from exc

    annoy_metric = _annoy_metric(metric)
    n_trees = backend_kwargs.pop("n_trees", 10)
    search_k = backend_kwargs.pop("search_k", -1)
    random_state = backend_kwargs.pop("random_state", None)
    index = AnnoyIndex(X.shape[1], annoy_metric)
    if random_state is not None:
        index.set_seed(random_state)
    for idx, row in enumerate(X):
        index.add_item(idx, row.astype(float).tolist())
    index.build(n_trees, **backend_kwargs)
    neighbours = [
        index.get_nns_by_item(idx, k + 1, search_k=search_k, include_distances=False)
        for idx in range(X.shape[0])
    ]
    return _drop_self_neighbours(np.array(neighbours), k, X, metric)


def _drop_self_neighbours(indices, k, X, metric):
    cleaned = []
    for row_idx, row in enumerate(indices):
        row_without_self = [int(idx) for idx in row if int(idx) != row_idx]
        cleaned.append(row_without_self[:k])

    if not cleaned:
        return np.empty((0, 0), dtype=int)

    # Some approximate indexes may return fewer than k non-self items. Pad with
    # the last available neighbor so callers still receive a rectangular array.
    padded = []
    for row_idx, row in enumerate(cleaned):
        if not row:
            row = _exact_neighbours_for_row(X, row_idx, k, metric)
        while len(row) < k:
            row.append(row[-1])
        padded.append(row)

    return np.array(padded, dtype=int)


def _exact_neighbours_for_row(X, row_idx, k, metric):
    distances = cdist(X[row_idx].reshape(1, -1), X, metric=metric).flatten()
    distances[row_idx] = np.inf
    return np.argsort(distances)[:k].astype(int).tolist()


def _hnswlib_space(metric):
    if metric in ("euclidean", "l2"):
        return "l2"
    if metric in ("cosine", "angular"):
        return "cosine"
    if metric in ("ip", "inner_product"):
        return "ip"
    raise ValueError("hnswlib backend supports only euclidean/l2, cosine/angular, and ip metrics.")


def _annoy_metric(metric):
    aliases = {
        "euclidean": "euclidean",
        "l2": "euclidean",
        "cosine": "angular",
        "angular": "angular",
        "manhattan": "manhattan",
        "hamming": "hamming",
        "dot": "dot",
    }
    if metric not in aliases:
        raise ValueError("annoy backend does not support metric %r." % metric)
    return aliases[metric]


def _validate_X(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(X) == 0:
        raise ValueError("X must contain at least one row.")
    return X
