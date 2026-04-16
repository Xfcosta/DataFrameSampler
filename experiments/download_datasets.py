from __future__ import annotations

import csv
import urllib.request
import zipfile
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

BREAST_CANCER_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

BREAST_CANCER_COLUMNS = (
    ["id", "diagnosis"]
    + [f"mean_{feature}" for feature in BREAST_CANCER_FEATURES]
    + [f"se_{feature}" for feature in BREAST_CANCER_FEATURES]
    + [f"worst_{feature}" for feature in BREAST_CANCER_FEATURES]
)

PIMA_DIABETES_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "diabetes",
]

HEART_DISEASE_COLUMNS = [
    "age",
    "sex",
    "chest_pain",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "major_vessels",
    "thal",
    "heart_disease_raw",
]

URLS = {
    "adult_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "pima_diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "bank_marketing": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
    "heart_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
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


def prepare_breast_cancer() -> pd.DataFrame:
    breast_raw = RAW / "breast_cancer" / "wdbc.data"
    download(URLS["breast_cancer"], breast_raw)
    breast_cancer = pd.read_csv(breast_raw, names=BREAST_CANCER_COLUMNS)
    breast_cancer = breast_cancer.drop(columns=["id"])
    breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map(
        {"M": "malignant", "B": "benign"}
    )
    output = PROCESSED / "breast_cancer.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    breast_cancer.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {breast_cancer.shape}")
    return breast_cancer


def prepare_pima_diabetes() -> pd.DataFrame:
    diabetes_raw = RAW / "pima_diabetes" / "pima-indians-diabetes.data.csv"
    download(URLS["pima_diabetes"], diabetes_raw)
    diabetes = pd.read_csv(diabetes_raw, names=PIMA_DIABETES_COLUMNS)
    zero_missing_columns = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for column in zero_missing_columns:
        diabetes[column] = pd.to_numeric(diabetes[column], errors="coerce").replace(0, pd.NA)
    diabetes["diabetes"] = diabetes["diabetes"].map({0: "negative", 1: "positive"})
    output = PROCESSED / "pima_diabetes.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    diabetes.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {diabetes.shape}")
    return diabetes


def prepare_bank_marketing() -> pd.DataFrame:
    bank_raw = RAW / "bank_marketing" / "bank-additional.zip"
    download(URLS["bank_marketing"], bank_raw)
    extract_dir = bank_raw.parent
    with zipfile.ZipFile(bank_raw) as archive:
        member = "bank-additional/bank-additional-full.csv"
        if not (extract_dir / member).exists():
            archive.extract(member, extract_dir)
    bank = pd.read_csv(extract_dir / member, sep=";")
    bank = bank.rename(
        columns={
            "emp.var.rate": "emp_var_rate",
            "cons.price.idx": "cons_price_idx",
            "cons.conf.idx": "cons_conf_idx",
            "nr.employed": "nr_employed",
            "y": "subscribed",
        }
    )
    bank = bank.replace("unknown", pd.NA)
    numeric_columns = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]
    for column in numeric_columns:
        bank[column] = pd.to_numeric(bank[column], errors="coerce")
    output = PROCESSED / "bank_marketing.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    bank.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {bank.shape}")
    return bank


def prepare_heart_disease() -> pd.DataFrame:
    heart_raw = RAW / "heart_disease" / "processed.cleveland.data"
    download(URLS["heart_disease"], heart_raw)
    heart = pd.read_csv(heart_raw, names=HEART_DISEASE_COLUMNS, na_values=["?"])
    for column in heart.columns:
        heart[column] = pd.to_numeric(heart[column], errors="coerce")
    heart["sex"] = heart["sex"].map({0: "female", 1: "male"})
    heart["chest_pain"] = heart["chest_pain"].map(
        {
            1: "typical_angina",
            2: "atypical_angina",
            3: "non_anginal_pain",
            4: "asymptomatic",
        }
    )
    heart["fasting_blood_sugar"] = heart["fasting_blood_sugar"].map(
        {0: "not_above_120", 1: "above_120"}
    )
    heart["resting_ecg"] = heart["resting_ecg"].map(
        {
            0: "normal",
            1: "st_t_wave_abnormality",
            2: "left_ventricular_hypertrophy",
        }
    )
    heart["exercise_angina"] = heart["exercise_angina"].map({0: "no", 1: "yes"})
    heart["slope"] = heart["slope"].map({1: "upsloping", 2: "flat", 3: "downsloping"})
    heart["thal"] = heart["thal"].map({3: "normal", 6: "fixed_defect", 7: "reversible_defect"})
    heart["heart_disease"] = heart["heart_disease_raw"].map(lambda value: "present" if value > 0 else "absent")
    heart = heart.drop(columns=["heart_disease_raw"])
    output = PROCESSED / "heart_disease.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    heart.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output} with shape {heart.shape}")
    return heart


def main() -> None:
    prepare_adult()
    prepare_titanic()
    prepare_breast_cancer()
    prepare_pima_diabetes()
    prepare_bank_marketing()
    prepare_heart_disease()
    materialize_synthetic_datasets(PROCESSED)


if __name__ == "__main__":
    main()
