from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import pandas as pd

from experiments.synthetic_data import materialize_synthetic_datasets


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

URLS = {
    "adult_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def clean_adult_value(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text in {"?", ""}:
        return pd.NA
    return text.rstrip(".")


def prepare_adult() -> pd.DataFrame:
    adult_raw = RAW / "adult"
    download(URLS["adult_train"], adult_raw / "adult.data")
    download(URLS["adult_test"], adult_raw / "adult.test")
    download(URLS["adult_names"], adult_raw / "adult.names")

    train = pd.read_csv(
        adult_raw / "adult.data",
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )
    test = pd.read_csv(
        adult_raw / "adult.test",
        names=ADULT_COLUMNS,
        skiprows=1,
        skipinitialspace=True,
        na_values=["?"],
    )
    adult = pd.concat([train, test], ignore_index=True)
    adult = adult.map(clean_adult_value)

    numeric_columns = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for column in numeric_columns:
        adult[column] = pd.to_numeric(adult[column], errors="coerce")

    adult["income"] = adult["income"].astype("string")
    output = PROCESSED / "adult.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    adult.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {adult.shape}")
    return adult


def prepare_titanic() -> pd.DataFrame:
    titanic_raw = RAW / "titanic" / "titanic.csv"
    download(URLS["titanic"], titanic_raw)
    titanic = pd.read_csv(titanic_raw)
    output = PROCESSED / "titanic.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    titanic.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {titanic.shape}")
    return titanic


def main() -> None:
    prepare_adult()
    prepare_titanic()
    materialize_synthetic_datasets(PROCESSED)


if __name__ == "__main__":
    main()
