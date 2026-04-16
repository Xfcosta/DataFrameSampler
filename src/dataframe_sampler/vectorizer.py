import copy
import inspect

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


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

MISSING_CATEGORY = "__MISSING__"


def is_numerical_column_type(series):
    return is_numeric_dtype(series)


class DataFrameVectorizer(object):
    """
    Vectorizes a dataframe for discretization.

    Numeric columns are copied after missing values are imputed with the column
    median. Non-numeric columns follow one policy learned during fit:
    high-cardinality identifier-like columns are discarded, binary columns are
    mapped to 0/1, and remaining categorical columns are one-hot encoded and
    reduced to a one-dimensional embedding.
    """

    def __init__(
        self,
        random_state=None,
        embedding_method="pca",
        embedding_kwargs=None,
        max_categorical_fraction=0.3,
        max_categorical_unique=50,
    ):
        self.random_state = random_state
        self.embedding_method = embedding_method
        self.embedding_kwargs = dict(embedding_kwargs or {})
        self.max_categorical_fraction = max_categorical_fraction
        self.max_categorical_unique = max_categorical_unique

    def fit(self, dataframe):
        self._validate_dataframe(dataframe)
        self.input_columns_ = list(dataframe.columns)
        self.column_plans_ = {}
        self.output_columns_ = []
        self.dropped_columns_ = []

        for column in dataframe.columns:
            series = dataframe[column]
            if is_numerical_column_type(series):
                self.column_plans_[column] = self._fit_numeric(series)
                self.output_columns_.append(column)
                continue

            values = self._category_values(series)
            unique_values = list(pd.unique(values))
            if self._should_drop_categorical(unique_values, len(series)):
                self.column_plans_[column] = {"strategy": "drop_high_cardinality"}
                self.dropped_columns_.append(column)
            elif len(unique_values) == 2:
                self.column_plans_[column] = self._fit_binary(unique_values)
                self.output_columns_.append(column)
            else:
                self.column_plans_[column] = self._fit_categorical_embedding(column, values, unique_values)
                self.output_columns_.append(column)

        if not self.output_columns_:
            raise ValueError("No usable columns remain after vectorization.")
        return self

    def transform(self, dataframe):
        self._ensure_fit()
        self._validate_dataframe(dataframe)
        missing = [column for column in self.input_columns_ if column not in dataframe.columns]
        if missing:
            raise ValueError("Missing columns for fitted vectorizer: %s" % missing)

        vectorizing_dataframe = pd.DataFrame(index=dataframe.index)
        for column in self.input_columns_:
            plan = self.column_plans_[column]
            strategy = plan["strategy"]
            if strategy == "numeric":
                vectorizing_dataframe[column] = self._numeric_values(dataframe[column], fill_value=plan["fill_value"])
            elif strategy == "binary":
                vectorizing_dataframe[column] = self._transform_binary(dataframe[column], plan)
            elif strategy == "categorical_embedding":
                vectorizing_dataframe[column] = self._transform_categorical_embedding(dataframe[column], plan)
            elif strategy == "drop_high_cardinality":
                continue
            else:
                raise ValueError("Unknown vectorization strategy %r." % strategy)
        return vectorizing_dataframe[self.output_columns_]

    def fit_transform(self, dataframe):
        return self.fit(dataframe).transform(dataframe)

    def _fit_numeric(self, series):
        numeric = pd.to_numeric(series, errors="coerce")
        fill_value = numeric.median()
        if pd.isna(fill_value):
            fill_value = 0.0
        return {"strategy": "numeric", "fill_value": float(fill_value)}

    @staticmethod
    def _fit_binary(unique_values):
        ordered = sorted(unique_values, key=lambda value: str(value))
        return {
            "strategy": "binary",
            "mapping": {ordered[0]: 0.0, ordered[1]: 1.0},
            "fallback": 0.5,
        }

    def _fit_categorical_embedding(self, column, values, unique_values):
        categories = sorted(unique_values, key=lambda value: str(value))
        category_frame = pd.DataFrame({column: categories})
        one_hot_encoder = _one_hot_encoder()
        one_hot = one_hot_encoder.fit_transform(category_frame[[column]])
        one_hot = _as_float_array(one_hot)

        embedding = self._embedding_for_column(column)
        embedded = self._embed_training_categories(embedding, one_hot, column)
        embedded = np.asarray(embedded, dtype=float).reshape(-1)

        approximator = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=1,
        )
        approximator.fit(one_hot, embedded)

        return {
            "strategy": "categorical_embedding",
            "categories": categories,
            "mapping": dict(zip(categories, embedded)),
            "one_hot_encoder": one_hot_encoder,
            "embedding": embedding if hasattr(embedding, "transform") else None,
            "approximator": approximator,
            "fallback": float(np.mean(embedded)),
        }

    def _transform_binary(self, series, plan):
        values = self._category_values(series)
        return values.map(plan["mapping"]).fillna(plan["fallback"]).astype(float)

    def _transform_categorical_embedding(self, series, plan):
        values = self._category_values(series)
        mapped = values.map(plan["mapping"])
        missing = mapped.isna()
        if not missing.any():
            return mapped.astype(float)

        category_frame = pd.DataFrame({series.name: values.loc[missing]})
        try:
            one_hot = plan["one_hot_encoder"].transform(category_frame[[series.name]])
        except ValueError:
            one_hot = plan["one_hot_encoder"].transform(category_frame.iloc[:, [0]])
        one_hot = _as_float_array(one_hot)

        if plan["embedding"] is not None:
            predicted = self._apply_transform(plan["embedding"], one_hot)
        else:
            predicted = plan["approximator"].predict(one_hot)
        predicted = np.asarray(predicted, dtype=float).reshape(-1)
        mapped.loc[missing] = predicted if len(predicted) else plan["fallback"]
        return mapped.fillna(plan["fallback"]).astype(float)

    def _embed_training_categories(self, embedding, one_hot, column):
        values = self._apply_embedding(embedding, one_hot)
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

    @staticmethod
    def _apply_transform(embedding, X):
        if hasattr(embedding, "transform"):
            return embedding.transform(X)
        raise TypeError("Embedding object must implement transform.")

    def _embedding_method_for_column(self, column):
        if isinstance(self.embedding_method, dict):
            return self.embedding_method.get(column, "pca")
        return self.embedding_method

    def _embedding_kwargs_for_column(self, column):
        if column in self.embedding_kwargs and isinstance(self.embedding_kwargs[column], dict):
            return dict(self.embedding_kwargs[column])
        return dict(self.embedding_kwargs)

    def _should_drop_categorical(self, unique_values, row_count):
        unique_count = len(unique_values)
        limit = max(self.max_categorical_unique, int(row_count * self.max_categorical_fraction))
        return unique_count > limit

    @staticmethod
    def _category_values(series):
        return series.astype("object").where(series.notna(), MISSING_CATEGORY)

    @staticmethod
    def _numeric_values(series, fill_value):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.fillna(fill_value).astype(float)

    @staticmethod
    def _validate_dataframe(dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("dataframe must contain at least one row.")

    def _ensure_fit(self):
        if not hasattr(self, "column_plans_"):
            raise ValueError("Vectorizer is not fit.")


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


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_float_array(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)
