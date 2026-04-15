import copy
import inspect

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


EMBEDDING_METHODS = (
    "mds",
    "pca",
    "incremental_pca",
    "kernel_pca",
    "sparse_pca",
    "truncated_svd",
    "factor_analysis",
    "fast_ica",
    "isomap",
    "lle",
    "spectral_embedding",
    "tsne",
)


def is_numerical_column_type(series):
    return is_numeric_dtype(series)


class DataFrameVectorizer(object):
    """
    Vectorizes a dataframe for discretization.

    Numeric columns are copied after numeric missing values are imputed with the
    column median. Non-numeric columns are mapped either through a configurable
    1D embedding of configured helper columns or through category frequencies.
    """

    def __init__(
        self,
        vectorizing_columns_dict=None,
        random_state=None,
        embedding_method="mds",
        embedding_kwargs=None,
    ):
        self.vectorizing_columns_dict = vectorizing_columns_dict
        self.random_state = random_state
        self.embedding_method = embedding_method
        self.embedding_kwargs = dict(embedding_kwargs or {})

    def fit(self, dataframe):
        self._validate_columns(dataframe)
        return self

    def transform(self, dataframe):
        self._validate_columns(dataframe)
        vectorizing_dataframe = pd.DataFrame(index=dataframe.index)

        for column in dataframe.columns:
            series = dataframe[column]
            if is_numerical_column_type(series):
                vectorizing_dataframe[column] = self._numeric_values(series)
            elif self.vectorizing_columns_dict is not None and column in self.vectorizing_columns_dict:
                selected_cols = self.vectorizing_columns_dict[column]
                X = dataframe.loc[:, selected_cols].apply(self._numeric_values).values
                if X.shape[0] == 1:
                    vectorizing_dataframe[column] = np.zeros(1)
                else:
                    vectorizing_dataframe[column] = self._embed(X, column).flatten()
            else:
                values = series.astype("object").where(series.notna(), "__MISSING__")
                counts = values.value_counts(dropna=False).to_dict()
                vectorizing_dataframe[column] = values.map(counts).astype(float)

        return vectorizing_dataframe

    def fit_transform(self, dataframe):
        return self.fit(dataframe).transform(dataframe)

    def _embed(self, X, column):
        embedding = self._embedding_for_column(column)
        values = self._apply_embedding(embedding, X)
        values = np.asarray(values)
        if values.ndim == 1:
            return values
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("Embedding method for column %r must return at least one component." % column)
        return values[:, 0]

    def _embedding_for_column(self, column):
        method = self._embedding_method_for_column(column)
        kwargs = self._embedding_kwargs_for_column(column)
        if isinstance(kwargs, dict):
            kwargs = dict(kwargs)
        else:
            raise TypeError("embedding_kwargs must be a dictionary or a dictionary keyed by column name.")

        if isinstance(method, str):
            return make_embedding_method(method, random_state=self.random_state, **kwargs)
        return copy.deepcopy(method)

    @staticmethod
    def _apply_embedding(embedding, X):
        if hasattr(embedding, "fit_transform"):
            return embedding.fit_transform(X)
        if hasattr(embedding, "fit") and hasattr(embedding, "transform"):
            return embedding.fit(X).transform(X)
        if hasattr(embedding, "transform"):
            return embedding.transform(X)
        raise TypeError("Embedding object must implement fit_transform, fit+transform, or transform.")

    def _embedding_method_for_column(self, column):
        if isinstance(self.embedding_method, dict):
            return self.embedding_method.get(column, "mds")
        return self.embedding_method

    def _embedding_kwargs_for_column(self, column):
        if column in self.embedding_kwargs and isinstance(self.embedding_kwargs[column], dict):
            return dict(self.embedding_kwargs[column])
        return dict(self.embedding_kwargs)

    def _validate_columns(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("dataframe must contain at least one row.")
        if self.vectorizing_columns_dict is None:
            return
        missing = []
        for column, selected_cols in self.vectorizing_columns_dict.items():
            if column not in dataframe.columns:
                missing.append(column)
            missing.extend([col for col in selected_cols if col not in dataframe.columns])
        if missing:
            raise ValueError("Unknown vectorizing columns: %s" % sorted(set(missing)))

    @staticmethod
    def _numeric_values(series):
        numeric = pd.to_numeric(series, errors="coerce")
        fill_value = numeric.median()
        if pd.isna(fill_value):
            fill_value = 0
        return numeric.fillna(fill_value).astype(float)


def make_embedding_method(method, random_state=None, **kwargs):
    method = method.lower()
    embedding_class = _embedding_class(method)
    params = dict(kwargs)
    params.setdefault("n_components", 1)
    params = _maybe_add_random_state(embedding_class, params, random_state)
    return embedding_class(**params)


def _embedding_class(method):
    if method == "mds":
        from sklearn.manifold import MDS

        return MDS
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA
    if method == "incremental_pca":
        from sklearn.decomposition import IncrementalPCA

        return IncrementalPCA
    if method == "kernel_pca":
        from sklearn.decomposition import KernelPCA

        return KernelPCA
    if method == "sparse_pca":
        from sklearn.decomposition import SparsePCA

        return SparsePCA
    if method == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD
    if method == "factor_analysis":
        from sklearn.decomposition import FactorAnalysis

        return FactorAnalysis
    if method == "fast_ica":
        from sklearn.decomposition import FastICA

        return FastICA
    if method == "isomap":
        from sklearn.manifold import Isomap

        return Isomap
    if method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding

        return LocallyLinearEmbedding
    if method == "spectral_embedding":
        from sklearn.manifold import SpectralEmbedding

        return SpectralEmbedding
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE
    raise ValueError("Unknown embedding_method %r. Expected one of %s." % (method, EMBEDDING_METHODS))


def _maybe_add_random_state(embedding_class, params, random_state):
    if random_state is None or "random_state" in params:
        return params
    signature = inspect.signature(embedding_class.__init__)
    if "random_state" in signature.parameters:
        params["random_state"] = random_state
    return params
