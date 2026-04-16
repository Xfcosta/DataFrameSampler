from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


@dataclass(frozen=True)
class SyntheticDatasetMetadata:
    key: str
    name: str
    regime: str
    rows: int
    target_column: str
    sensitive_columns: tuple[str, ...]
    rationale: str


SYNTHETIC_DATASETS = [
    SyntheticDatasetMetadata(
        key="synthetic_correlated_helpers",
        name="Controlled correlated helpers",
        regime="Correlated numeric plus categorical embeddings",
        rows=600,
        target_column="target",
        sensitive_columns=(),
        rationale="Known numeric/categorical dependency structure",
    ),
    SyntheticDatasetMetadata(
        key="synthetic_high_cardinality",
        name="Controlled high cardinality",
        regime="High-cardinality categorical column",
        rows=600,
        target_column="target",
        sensitive_columns=(),
        rationale="Stress category coverage and categorical embeddings",
    ),
    SyntheticDatasetMetadata(
        key="synthetic_rare_categories",
        name="Controlled rare categories",
        regime="Known rare-category structure",
        rows=600,
        target_column="target",
        sensitive_columns=(),
        rationale="Measure rare-category preservation boundaries",
    ),
    SyntheticDatasetMetadata(
        key="synthetic_sensitive_identifier",
        name="Controlled sensitive identifier",
        regime="Known sensitive identifier column",
        rows=600,
        target_column="risk_flag",
        sensitive_columns=("patient_id",),
        rationale="Expose exact source-value reuse risk",
    ),
]


def materialize_synthetic_datasets(processed_dir: str | Path = PROCESSED) -> dict[str, Path]:
    """Write all deterministic controlled datasets and return their paths."""
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "synthetic_correlated_helpers": make_correlated_helpers(rows=600, random_state=101),
        "synthetic_high_cardinality": make_high_cardinality(rows=600, random_state=102),
        "synthetic_rare_categories": make_rare_categories(rows=600, random_state=103),
        "synthetic_sensitive_identifier": make_sensitive_identifier(rows=600, random_state=104),
    }
    paths: dict[str, Path] = {}
    for key, dataframe in datasets.items():
        path = output_dir / f"{key}.csv"
        dataframe.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        paths[key] = path
    return paths


def make_correlated_helpers(*, rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    latent = rng.normal(size=rows)
    seasonal = rng.normal(size=rows)
    spend_score = 50 + 11 * latent + rng.normal(0, 3.0, size=rows)
    visit_rate = 20 + 6 * latent + 2 * seasonal + rng.normal(0, 1.5, size=rows)
    risk_score = 40 - 7 * latent + 4 * seasonal + rng.normal(0, 2.0, size=rows)
    helper_band = pd.cut(
        latent,
        bins=[-np.inf, -0.75, 0.75, np.inf],
        labels=["low_context", "mid_context", "high_context"],
    ).astype(str)
    segment = np.select(
        [spend_score > 58, risk_score > 48, visit_rate < 17],
        ["premium", "watch", "low_touch"],
        default="standard",
    )
    target = ((spend_score + visit_rate - risk_score + rng.normal(0, 5, size=rows)) > 32).astype(int)
    return pd.DataFrame(
        {
            "spend_score": np.round(spend_score, 2),
            "visit_rate": np.round(visit_rate, 2),
            "risk_score": np.round(risk_score, 2),
            "helper_band": helper_band,
            "segment": segment,
            "target": target,
        }
    )


def make_high_cardinality(*, rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    latent = rng.normal(size=rows)
    account_value = np.exp(3.6 + 0.35 * latent + rng.normal(0, 0.25, size=rows))
    tenure_months = np.clip(np.rint(26 + 9 * latent + rng.normal(0, 8, size=rows)), 1, 72).astype(int)
    activity_score = np.clip(55 + 14 * latent + rng.normal(0, 8, size=rows), 0, 100)
    regions = np.array(["north", "south", "east", "west", "central"])
    region = regions[np.digitize(latent + rng.normal(0, 0.4, size=rows), [-1.2, -0.3, 0.3, 1.2])]
    sku_index = np.mod(np.floor((latent + 3.5) * 17 + rng.integers(0, 9, size=rows)).astype(int), 120)
    sku_code = np.array([f"SKU-{idx:03d}" for idx in sku_index])
    plan_tier = np.select(
        [account_value > np.quantile(account_value, 0.8), account_value < np.quantile(account_value, 0.25)],
        ["enterprise", "basic"],
        default="team",
    )
    target = ((activity_score > 62) | ((plan_tier == "enterprise") & (tenure_months > 24))).astype(int)
    return pd.DataFrame(
        {
            "account_value": np.round(account_value, 2),
            "tenure_months": tenure_months,
            "activity_score": np.round(activity_score, 2),
            "region": region,
            "sku_code": sku_code,
            "plan_tier": plan_tier,
            "target": target,
        }
    )


def make_rare_categories(*, rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    latent = rng.normal(size=rows)
    recency_days = np.clip(np.rint(45 - 11 * latent + rng.normal(0, 12, size=rows)), 1, 180).astype(int)
    frequency = np.clip(np.rint(5 + 2.2 * latent + rng.normal(0, 2, size=rows)), 0, 20).astype(int)
    monetary = np.exp(3.2 + 0.28 * latent + rng.normal(0, 0.3, size=rows))
    rare_signal = rng.choice(
        ["common_a", "common_b", "common_c", "rare_gold", "rare_silver", "rare_bronze"],
        size=rows,
        p=[0.45, 0.35, 0.18, 0.008, 0.007, 0.005],
    )
    rare_boost = np.isin(rare_signal, ["rare_gold", "rare_silver", "rare_bronze"]).astype(int)
    lifecycle = np.select(
        [recency_days < 20, frequency < 3, monetary > np.quantile(monetary, 0.85)],
        ["active", "cold", "high_value"],
        default="steady",
    )
    target = ((0.35 * frequency + 1.4 * rare_boost - 0.02 * recency_days + rng.normal(0, 0.8, size=rows)) > 0.9).astype(int)
    return pd.DataFrame(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary": np.round(monetary, 2),
            "rare_signal": rare_signal,
            "lifecycle": lifecycle,
            "target": target,
        }
    )


def make_sensitive_identifier(*, rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = np.clip(np.rint(rng.normal(58, 14, size=rows)), 18, 92).astype(int)
    lab_score = np.clip(rng.normal(70, 12, size=rows) + 0.12 * (age - 58), 20, 120)
    visits = np.clip(rng.poisson(2.4 + (lab_score > 78) * 1.2, size=rows), 0, 12)
    condition = rng.choice(
        ["routine", "cardio", "endocrine", "respiratory", "oncology_review"],
        size=rows,
        p=[0.48, 0.2, 0.18, 0.11, 0.03],
    )
    ward = rng.choice(["north", "south", "east", "west"], size=rows, p=[0.28, 0.27, 0.23, 0.22])
    risk_flag = ((lab_score > 77) | (visits > 4) | (condition == "oncology_review")).astype(int)
    patient_id = [f"PAT-{random_state}-{idx:05d}" for idx in range(rows)]
    return pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "lab_score": np.round(lab_score, 2),
            "visits": visits,
            "condition": condition,
            "ward": ward,
            "risk_flag": risk_flag,
        }
    )


def main() -> None:
    paths = materialize_synthetic_datasets()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
