from pathlib import Path

import pandas as pd


PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".txt"}
CSV_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip", ".zst"}


def read_dataframe(filename):
    path = Path(filename)
    suffix = _dataframe_suffix(path)
    if suffix in PARQUET_SUFFIXES:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet input requires an optional parquet engine. "
                "Install one with: pip install 'dataframe-sampler[parquet]'"
            ) from exc
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path)
    raise ValueError("Unsupported dataframe file extension for %s. Use CSV or Parquet." % path)


def write_dataframe(dataframe, filename):
    path = Path(filename)
    suffix = _dataframe_suffix(path)
    if suffix in PARQUET_SUFFIXES:
        try:
            dataframe.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet output requires an optional parquet engine. "
                "Install one with: pip install 'dataframe-sampler[parquet]'"
            ) from exc
    elif suffix in CSV_SUFFIXES:
        dataframe.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported dataframe file extension for %s. Use CSV or Parquet." % path)
    return dataframe


def _dataframe_suffix(path):
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes:
        raise ValueError("File extension is required for %s." % path)
    if suffixes[-1] in CSV_COMPRESSED_SUFFIXES and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1]
