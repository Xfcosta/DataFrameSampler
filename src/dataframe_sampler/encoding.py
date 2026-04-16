from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .utils import make_random_state, random_choice, random_integers


class ConstantDiscretizer(object):
    """
    Minimal discretizer for constant columns.
    """

    def __init__(self):
        self.n_bins_ = np.array([1])

    def fit(self, X):
        return self

    def transform(self, X):
        return np.zeros((len(X), 1))


class ColumnDataFrameEncoderDecoder(object):
    def __init__(self, n_bins=5, strategy="uniform", random_state=None, decode_strategy="observed_bin"):
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")
        if decode_strategy not in {"observed_bin", "continuous"}:
            raise ValueError("decode_strategy must be 'observed_bin' or 'continuous'.")
        self.n_bins = n_bins
        self.strategy = strategy
        self.decode_strategy = decode_strategy
        self.default = -1
        self.max_n_attempts = 100
        self.discretizer = None
        self.random_state = random_state
        self.rng = make_random_state(random_state)

    def fit(self, column_values, vectorizing_column_values):
        column_values = np.asarray(column_values, dtype=object)
        vectorizing_column_values = np.asarray(vectorizing_column_values, dtype=float)
        if len(column_values) == 0:
            raise ValueError("Cannot fit an encoder on an empty column.")
        if len(column_values) != len(vectorizing_column_values):
            raise ValueError("column_values and vectorizing_column_values must have the same length.")

        unique = np.unique(vectorizing_column_values)
        n_bins = min(len(unique), self.n_bins)
        if n_bins < 2:
            self.discretizer = ConstantDiscretizer()
        else:
            self.discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=self.strategy, encode="ordinal")

        self.discretizer.fit(vectorizing_column_values.reshape(-1, 1))
        discretized_column_values = self.encode(vectorizing_column_values)

        self.histogram = defaultdict(list)
        for discretized_column_value, column_value in zip(discretized_column_values, column_values):
            self.histogram[int(discretized_column_value)].append(column_value)

        return self

    def encode(self, vectorizing_column_values):
        if self.discretizer is None:
            raise ValueError("Encoder is not fit.")
        vectorizing_column_values = np.asarray(vectorizing_column_values, dtype=float)
        if self.decode_strategy == "continuous":
            return vectorizing_column_values
        Xt = self.discretizer.transform(vectorizing_column_values.reshape(-1, 1))
        return Xt[:, 0].flatten().astype(int)

    def decode(self, discretized_column_values):
        if self.discretizer is None:
            raise ValueError("Encoder is not fit.")
        if self.decode_strategy == "continuous":
            return np.asarray(discretized_column_values, dtype=float).tolist()

        results = []
        for discretized_column_value in np.asarray(discretized_column_values).astype(int):
            bin_key = self._nearest_non_empty_bin(discretized_column_value)
            if bin_key is None:
                selected = self.default
            else:
                selected = random_choice(self.rng, self.histogram[bin_key])
            results.append(selected)

        return results

    def _nearest_non_empty_bin(self, discretized_column_value):
        available = sorted(key for key, values in self.histogram.items() if len(values) > 0)
        if not available:
            return None
        return min(available, key=lambda key: abs(key - discretized_column_value))


class DataFrameEncoderDecoder(object):
    def __init__(self, n_bins=5, strategy="uniform", random_state=None, numeric_decode_strategy="observed_bin"):
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")
        if numeric_decode_strategy not in {"observed_bin", "continuous"}:
            raise ValueError("numeric_decode_strategy must be 'observed_bin' or 'continuous'.")
        self.n_bins = n_bins
        self.strategy = strategy
        self.numeric_decode_strategy = numeric_decode_strategy
        self.random_state = random_state
        self.rng = make_random_state(random_state)

    def fit(self, dataframe, vectorizing_dataframe):
        if dataframe.empty:
            raise ValueError("dataframe must contain at least one row.")
        if list(dataframe.columns) != list(vectorizing_dataframe.columns):
            raise ValueError("dataframe and vectorizing_dataframe must have the same columns.")

        self.columns = dataframe.columns
        self.column_dataframe_encoder_decoders = [
            ColumnDataFrameEncoderDecoder(
                n_bins=self.n_bins,
                strategy=self.strategy,
                random_state=self.rng,
                decode_strategy=self._column_decode_strategy(dataframe[column]),
            ).fit(dataframe[column].values, vectorizing_dataframe[column].values)
            for column in dataframe.columns
        ]
        return self

    def encode(self, vectorizing_dataframe):
        self._ensure_fit()
        data_mtx = [
            column_dataframe_encoder_decoder.encode(vectorizing_dataframe[self.columns[i]].values)
            for i, column_dataframe_encoder_decoder in enumerate(self.column_dataframe_encoder_decoders)
        ]
        return np.array(data_mtx).T

    def decode(self, data_mtx):
        self._ensure_fit()
        res = pd.DataFrame()
        for i, col in enumerate(np.asarray(data_mtx).T):
            res[self.columns[i]] = self.column_dataframe_encoder_decoders[i].decode(col)
        return res

    def sample(self, n_samples):
        self._ensure_fit()
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")

        all_values = []
        for column_dataframe_encoder_decoder in self.column_dataframe_encoder_decoders:
            n_bins = int(column_dataframe_encoder_decoder.discretizer.n_bins_[0])
            values = random_integers(self.rng, 0, n_bins, size=n_samples)
            all_values.append(values)
        return np.array(all_values).T

    def info(self, k=1):
        self._ensure_fit()
        for i in range(len(self.columns)):
            print()
            print("%2d   %s" % (i, self.columns[i]))
            for key in sorted(self.column_dataframe_encoder_decoders[i].histogram.keys()):
                elements_in_bin = self.column_dataframe_encoder_decoders[i].histogram[key]
                n_elements_in_bin = len(elements_in_bin)
                unique_elements_in_bin = sorted(list(np.unique(elements_in_bin)))
                size = min(len(unique_elements_in_bin), k)
                if size > 0:
                    selected_elements = "|"
                    if size > 2:
                        selected_elements = random_choice(self.rng, unique_elements_in_bin, size=size)
                    last_element = "|" if size == 1 else unique_elements_in_bin[-1]
                    print(
                        "%d  #%3d   <%s  %s  %s>"
                        % (key, n_elements_in_bin, unique_elements_in_bin[0], selected_elements, last_element)
                    )

    def _ensure_fit(self):
        if not hasattr(self, "column_dataframe_encoder_decoders"):
            raise ValueError("Encoder is not fit.")

    def _column_decode_strategy(self, series):
        if self.numeric_decode_strategy == "continuous" and pd.api.types.is_numeric_dtype(series):
            return "continuous"
        return "observed_bin"
