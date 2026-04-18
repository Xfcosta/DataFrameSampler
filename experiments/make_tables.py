from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.manifold_validation import summarize_manifold_validation
from experiments.mechanism_validation import summarize_decoder_calibration, summarize_mechanism_validation
from experiments.imbalance_validation import summarize_imbalance_validation
from experiments.primary_uncertainty import summarize_primary_uncertainty
from experiments.proposed_setups import PROPOSED_SAMPLER_SETUPS
from experiments.sensitivity_validation import summarize_sensitivity_validation
from experiments.synthetic_data import SYNTHETIC_DATASETS, materialize_synthetic_datasets


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
PROCESSED = EXPERIMENTS / "data" / "processed"
TABLES = ROOT / "publication" / "tables"
UP = chr(8593)
DOWN = chr(8595)
TARGET = chr(8594)


@dataclass(frozen=True)
class DatasetTableMetadata:
    key: str
    name: str
    domain: str
    sensitive: str
    rationale: str

METHOD_METADATA = [
    *[
        {
            "method": setup.label,
            "package": "DataFrameSampler",
            "family": (
                f"Proposed setup: n_iter={setup.n_iterations}, "
                "k=1, sample=0.5, lambda=0.25, "
                f"retries={setup.max_constraint_retries}, "
                f"calibration={'yes' if setup.calibrate_decoders else 'no'}"
            ),
            "mixed": "Yes",
            "setup": "Low" if setup.key != "accurate" else "Medium",
            "inspectability": "High",
            "optional": "None",
        }
        for setup in PROPOSED_SAMPLER_SETUPS
    ],
    {
        "method": "DataFrameSampler",
        "package": "DataFrameSampler",
        "family": "Default comparison setup: n_iter=0, k=1, sample=0.5, lambda=0.25, retries=5",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "High",
        "optional": "None",
    },
    {
        "method": "Row bootstrap",
        "package": "Experiment helper",
        "family": "Resampling",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "Medium",
        "optional": "None",
    },
    {
        "method": "Independent columns",
        "package": "Experiment helper",
        "family": "Empirical marginals",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "Medium",
        "optional": "None",
    },
    {
        "method": "Gaussian copula",
        "package": "Experiment helper",
        "family": "Copula + empirical categorical",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "Medium",
        "optional": "None",
    },
    {
        "method": "Stratified columns",
        "package": "Experiment helper",
        "family": "Target-stratified empirical",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "Medium",
        "optional": "Target column",
    },
    {
        "method": "CTGAN",
        "package": "SDV",
        "family": "High-capacity adversarial reference",
        "mixed": "Yes",
        "setup": "Medium",
        "inspectability": "Low",
        "optional": "sdv",
    },
    {
        "method": "SMOTENC",
        "package": "imbalanced-learn",
        "family": "Supervised oversampling",
        "mixed": "Yes",
        "setup": "Medium",
        "inspectability": "Medium",
        "optional": "imbalanced-learn",
    },
]

METHOD_LABELS = {
    "dataframe_sampler": "DataFrameSampler",
    "ctgan": "CTGAN",
    "row_bootstrap": "Row bootstrap",
    "independent_columns": "Independent columns",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified columns",
    "latent_interpolation": "Latent interpolation",
    "latent_bootstrap": "Latent bootstrap",
    "real_train": "Real train",
    "dataframe_sampler_balanced": "DataFrameSampler balanced",
    "smotenc_balanced": "SMOTE/SMOTENC balanced",
    "stratified_columns_balanced": "Stratified columns balanced",
}

DEFAULT_DATASETS = [
    DatasetTableMetadata(
        key="adult",
        name="Adult Census Income",
        domain="Census / income",
        sensitive="None selected",
        rationale="Generic mixed-type benchmark",
    ),
    DatasetTableMetadata(
        key="titanic",
        name="Titanic",
        domain="Passenger survival",
        sensitive="None selected",
        rationale="Small mixed-type smoke benchmark",
    ),
    DatasetTableMetadata(
        key="breast_cancer",
        name="Wisconsin Diagnostic Breast Cancer",
        domain="Medical diagnosis",
        sensitive="Diagnosis label",
        rationale="Small medical binary-classification benchmark",
    ),
    DatasetTableMetadata(
        key="pima_diabetes",
        name="Pima Indians Diabetes",
        domain="Medical risk classification",
        sensitive="Diabetes label",
        rationale="Small medical classification benchmark with missing clinical measurements",
    ),
    DatasetTableMetadata(
        key="bank_marketing",
        name="Bank Marketing",
        domain="Marketing / subscription",
        sensitive="None selected",
        rationale="Mixed-type business classification benchmark",
    ),
    DatasetTableMetadata(
        key="heart_disease",
        name="UCI Heart Disease",
        domain="Medical diagnosis",
        sensitive="Heart-disease label",
        rationale="Small medical classification benchmark with mixed clinical variables",
    ),
    DatasetTableMetadata(
        key="covertype",
        name="Forest Covertype",
        domain="Ecology / forest cover",
        sensitive="None selected",
        rationale="Large-scale public benchmark with collapsed categorical terrain indicators",
    ),
    *[
        DatasetTableMetadata(
            key=spec.key,
            name=spec.name,
            domain="Synthetic controlled",
            sensitive=", ".join(spec.sensitive_columns) if spec.sensitive_columns else "None selected",
            rationale=spec.rationale,
        )
        for spec in SYNTHETIC_DATASETS
    ],
]


def load_comparisons(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(Path(results_dir).glob("*_baseline_comparison.csv"))]
    if not frames:
        raise FileNotFoundError("No baseline comparison files found. Run the notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    for column in [
        "fit_peak_memory_mb",
        "sample_peak_memory_mb",
        "peak_memory_mb",
        "nn_distance_ratio",
        "nn_suspiciously_close_rate",
        "discrimination_accuracy",
        "discrimination_privacy_score",
        "utility_task",
        "utility_real_score",
        "utility_augmented_score",
        "utility_lift",
        "distribution_histogram_overlap",
        "distribution_numeric_kl",
        "distribution_categorical_jsd",
        "distribution_similarity_score",
    ]:
        if column not in data:
            data[column] = pd.NA
    return data


def load_manifold_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_manifold_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No manifold validation files found. Run the notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def load_mechanism_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_mechanism_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No mechanism validation files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_decoder_calibrations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_decoder_calibration.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No decoder calibration files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_sensitivity_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_sensitivity_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No sensitivity validation files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_primary_uncertainty(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_primary_uncertainty.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No primary uncertainty files found. Run the Adult repeated-seed diagnostic first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def load_imbalance_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_imbalance_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No imbalance validation files found. Run the selected dataset notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def load_deep_reference_comparisons(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_deep_reference_comparison.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No deep reference comparison files found. Run the Adult CTGAN reference first.")
    data = pd.concat(frames, ignore_index=True)

    reference_rows = []
    for dataset in sorted(data["dataset"].dropna().unique()):
        baseline_path = results_path / f"{dataset}_baseline_comparison.csv"
        if not baseline_path.exists():
            continue
        baseline = pd.read_csv(baseline_path)
        reference_rows.append(
            baseline[
                baseline["method"].isin(["dataframe_sampler", "gaussian_copula_empirical"])
            ]
        )
    if reference_rows:
        data = pd.concat([*reference_rows, data], ignore_index=True, sort=False)

    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data.get("method_label", data["method"]))
    return data


def write_dataset_table(
    *,
    processed_dir: str | Path = PROCESSED,
    tables_dir: str | Path = TABLES,
    dataset_metadata: list[DatasetTableMetadata] | None = None,
) -> Path:
    rows = []
    processed_path = Path(processed_dir)
    for meta in dataset_metadata or DEFAULT_DATASETS:
        df = pd.read_csv(processed_path / f"{meta.key}.csv")
        numeric = len(df.select_dtypes(include="number").columns)
        categorical = len(df.columns) - numeric
        rows.append(
            {
                "Dataset": meta.name,
                "Domain": meta.domain,
                "Rows": len(df),
                "Numeric": numeric,
                "Categorical": categorical,
                "Missing": int(df.isna().sum().sum()),
                "Sensitive": meta.sensitive,
                "Rationale": meta.rationale,
            }
        )
    return write_latex(
        pd.DataFrame(rows),
        Path(tables_dir) / "datasets.tex",
        "Datasets used in the starter experiments. Takeaway: the evidence spans mixed public benchmarks and controlled regimes, so conclusions are practical and diagnostic rather than universal.",
        "tab:datasets",
        full_width=True,
    )


def write_method_table(*, tables_dir: str | Path = TABLES) -> Path:
    df = pd.DataFrame(METHOD_METADATA)
    df = df.rename(
        columns={
            "method": "Method",
            "package": "Package",
            "family": "Family",
            "mixed": "Mixed",
            "setup": "Setup",
            "inspectability": "Inspect.",
            "optional": "Optional deps",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "methods.tex",
        "Baseline and competitor method metadata. Takeaway: the comparisons emphasise low-setup, inspectable baselines that match the paper's practical-use claim.",
        "tab:methods",
        full_width=True,
    )


def write_distribution_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    df = comparisons[
        [
            "dataset",
            "method_label",
            "numeric_ks_statistic",
            "categorical_total_variation",
            "mean_abs_association_difference",
            "numeric_histogram_overlap",
        ]
    ].copy()
    df["Caveat"] = df["method_label"].map(
        {
            "Row bootstrap": "Reuses source rows",
            "Independent columns": "Breaks dependencies",
            "Gaussian copula": "Numeric copula only",
        }
    ).fillna("Generated sample")
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "numeric_ks_statistic": f"{DOWN} KS",
            "categorical_total_variation": f"{DOWN} Cat. TV",
            "mean_abs_association_difference": f"{DOWN} Assoc. diff.",
            "numeric_histogram_overlap": f"{UP} Hist. overlap",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "distributional_similarity.tex",
        "Starter distributional similarity results. Lower is better for KS, categorical TV, and association difference; higher is better for histogram overlap. Takeaway: marginal similarity alone is not decisive, because simple baselines can look strong while breaking dependencies or reusing rows.",
        "tab:distributional-similarity",
        float_format="%.3f",
    )


def write_main_measure_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    selected_methods = {
        "dataframe_sampler",
        "row_bootstrap",
        "independent_columns",
    }
    df = comparisons[comparisons["method"].isin(selected_methods)][
        [
            "dataset",
            "method_label",
            "nn_distance_ratio",
            "nn_suspiciously_close_rate",
            "discrimination_accuracy",
            "discrimination_privacy_score",
            "utility_lift",
            "distribution_histogram_overlap",
            "distribution_categorical_jsd",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "nn_distance_ratio": f"{TARGET} NN ratio",
            "nn_suspiciously_close_rate": f"{DOWN} NN close rate",
            "discrimination_accuracy": f"{TARGET} Disc. acc.",
            "discrimination_privacy_score": f"{UP} Disc. privacy",
            "utility_lift": f"{UP} Utility lift",
            "distribution_histogram_overlap": f"{UP} Hist. overlap",
            "distribution_categorical_jsd": f"{DOWN} Cat. JSD",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "main_measures.tex",
        "Primary experiment measures. Arrows mark the preferred direction for each quality measure: higher, lower, or target value. NN ratio compares synthetic-to-real nearest-neighbour distance with natural real-to-real nearest-neighbour distance; values below one indicate closer-than-natural synthetic rows. Discrimination accuracy near 0.5 is better. Utility lift is the change from adding synthetic rows to the real training set. Takeaway: DataFrameSampler is best read as a balanced example generator, not as a single-metric winner.",
        "tab:main-measures",
        float_format="%.3f",
        full_width=True,
    )


def write_primary_uncertainty_table(rows: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    summary = summarize_primary_uncertainty(rows)
    if summary.empty:
        raise FileNotFoundError("No valid primary uncertainty rows found.")
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    for metric in ["distribution_similarity", "discrimination_accuracy", "utility_lift", "nn_distance_ratio"]:
        summary[metric] = summary.apply(
            lambda row: _mean_pm_std(row[f"{metric}_mean"], row[f"{metric}_std"]),
            axis=1,
        )
    df = summary[
        [
            "dataset",
            "method_label",
            "runs",
            "distribution_similarity",
            "discrimination_accuracy",
            "utility_lift",
            "nn_distance_ratio",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "runs": "Runs",
            "distribution_similarity": f"{UP} Dist. score",
            "discrimination_accuracy": f"{TARGET} Disc. acc.",
            "utility_lift": f"{UP} Utility lift",
            "nn_distance_ratio": f"{TARGET} NN ratio",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "primary_uncertainty.tex",
        "Repeated-seed uncertainty for the primary metrics on the Adult running-example dataset, reported as mean plus/minus standard deviation. Takeaway: variability is reported descriptively to avoid treating point estimates as statistical conclusions.",
        "tab:primary-uncertainty",
        full_width=True,
    )


def write_cross_dataset_aggregation_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    selected_methods = ["dataframe_sampler", "row_bootstrap", "independent_columns", "gaussian_copula_empirical", "stratified_columns"]
    data = comparisons[comparisons["method"].isin(selected_methods)].copy()
    if data.empty:
        raise FileNotFoundError("No comparison rows available for aggregation.")
    rank_specs = [
        ("distribution_similarity_score", False),
        ("utility_lift", False),
        ("discrimination_gap", True),
    ]
    data["discrimination_gap"] = (data["discrimination_accuracy"] - 0.5).abs()
    rank_rows = []
    for dataset, subset in data.groupby("dataset"):
        for metric, ascending in rank_specs:
            metric_subset = subset[["method", "method_label", metric]].dropna()
            if metric_subset.empty:
                continue
            ranks = metric_subset[metric].rank(method="min", ascending=ascending)
            best = ranks == ranks.min()
            worst = ranks == ranks.max()
            for idx, row in metric_subset.iterrows():
                rank_rows.append(
                    {
                        "dataset": dataset,
                        "method": row["method"],
                        "method_label": row["method_label"],
                        "rank": float(ranks.loc[idx]),
                        "win": bool(best.loc[idx]),
                        "loss": bool(worst.loc[idx]),
                    }
                )
    ranks = pd.DataFrame(rank_rows)
    summary = (
        ranks.groupby(["method", "method_label"], dropna=False)
        .agg(avg_rank=("rank", "mean"), wins=("win", "sum"), losses=("loss", "sum"), comparisons=("rank", "count"))
        .reset_index()
        .sort_values(["avg_rank", "method_label"])
    )
    df = summary.rename(
        columns={
            "method_label": "Method",
            "avg_rank": f"{DOWN} Avg. rank",
            "wins": f"{UP} Wins",
            "losses": f"{DOWN} Losses",
            "comparisons": "Metric-datasets",
        }
    )[["Method", f"{DOWN} Avg. rank", f"{UP} Wins", f"{DOWN} Losses", "Metric-datasets"]]
    return write_latex(
        df,
        Path(tables_dir) / "cross_dataset_aggregation.tex",
        "Descriptive cross-dataset aggregation over distribution score, utility lift, and real-versus-synthetic discrimination gap. Lower average rank is better; wins and losses count metric-dataset best and worst cases. Takeaway: aggregation supports only weak descriptive pattern claims, not formal statistical superiority.",
        "tab:cross-dataset-aggregation",
        float_format="%.3f",
    )
def write_downstream_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    df = comparisons[
        [
            "dataset",
            "method_label",
            "utility_task",
            "utility_real_score",
            "utility_augmented_score",
            "utility_lift",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "utility_task": "Task",
            "utility_real_score": f"{UP} Real train score",
            "utility_augmented_score": f"{UP} Augmented score",
            "utility_lift": f"{UP} Utility lift",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "downstream_utility.tex",
        "Utility lift test. A baseline model is trained on real training data, then compared with a model trained on real plus generated rows and evaluated on held-out real rows. Takeaway: utility gains are dataset-specific, so the paper supports regime-specific usefulness rather than broad predictive superiority.",
        "tab:downstream-utility",
        float_format="%.3f",
        full_width=True,
    )


def write_runtime_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    df = comparisons[
        [
            "dataset",
            "method_label",
            "fit_seconds",
            "sample_seconds",
            "fit_peak_memory_mb",
            "sample_peak_memory_mb",
            "peak_memory_mb",
        ]
    ].copy()
    setup_steps = {
        "DataFrameSampler": "Fixed sampler config",
        "Row bootstrap": "None",
        "Independent columns": "None",
        "Gaussian copula": "None",
        "Stratified columns": "Target column",
    }
    df["Commands/LOC"] = "Notebook cell"
    df["Tuning/config"] = df["method_label"].map(setup_steps).fillna("Not recorded")
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "fit_seconds": f"{DOWN} Fit s",
            "sample_seconds": f"{DOWN} Sample s",
            "fit_peak_memory_mb": f"{DOWN} Fit peak MB",
            "sample_peak_memory_mb": f"{DOWN} Sample peak MB",
            "peak_memory_mb": f"{DOWN} Peak MB",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "usability_runtime.tex",
        "Starter usability, runtime, and traced Python allocation measurements. Peak MB is the maximum traced allocation peak observed during fit or sample, not total process RSS. Takeaway: the method remains notebook-friendly, but the NCA, decoding, and diagnostic steps carry real computational cost.",
        "tab:usability-runtime",
        float_format="%.3f",
        full_width=True,
    )


def write_synthetic_dataset_table(
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    rows = [
        {
            "Dataset": spec.key.replace("synthetic_", ""),
            "Controlled regime": spec.regime,
            "Rows": spec.rows,
            "Target": spec.target_column,
            "Sensitive columns": ", ".join(spec.sensitive_columns) if spec.sensitive_columns else "None",
            "Evidence role": spec.rationale,
        }
        for spec in SYNTHETIC_DATASETS
    ]
    return write_latex(
        pd.DataFrame(rows),
        Path(tables_dir) / "synthetic_controlled_datasets.tex",
        "Controlled synthetic datasets used to isolate specific boundary regimes. Takeaway: the controlled suite makes failure modes observable instead of relying only on aggregate benchmark scores.",
        "tab:synthetic-controlled-datasets",
        full_width=True,
    )


def write_synthetic_results_table(
    comparisons: pd.DataFrame,
    *,
    processed_dir: str | Path = PROCESSED,
    results_dir: str | Path = RESULTS,
    tables_dir: str | Path = TABLES,
) -> Path:
    synthetic_keys = {spec.key for spec in SYNTHETIC_DATASETS}
    method_subset = {
        "dataframe_sampler",
        "independent_columns",
        "row_bootstrap",
    }
    df = comparisons[
        comparisons["dataset"].isin(synthetic_keys) & comparisons["method"].isin(method_subset)
    ][
        [
            "dataset",
            "method_label",
            "numeric_ks_statistic",
            "categorical_total_variation",
            "categorical_coverage",
            "rare_category_preservation",
            "mean_abs_association_difference",
        ]
    ].copy()
    df["Sensitive overlap"] = df.apply(
        lambda row: sensitive_overlap_count(
            dataset_name=str(row["dataset"]),
            method_label=str(row["method_label"]),
            processed_dir=processed_dir,
            results_dir=results_dir,
        ),
        axis=1,
    )
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "numeric_ks_statistic": f"{DOWN} KS",
            "categorical_total_variation": f"{DOWN} Cat. TV",
            "categorical_coverage": f"{UP} Cat. coverage",
            "rare_category_preservation": f"{UP} Rare preserve",
            "mean_abs_association_difference": f"{DOWN} Assoc. diff.",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "synthetic_controlled_results.tex",
        "Focused results for controlled synthetic regimes. Sensitive overlap counts exact reuse of patient identifiers in the controlled identifier dataset. Takeaway: controlled regimes show where the sampler preserves useful structure and where it still needs explicit safeguards.",
        "tab:synthetic-controlled-results",
        float_format="%.3f",
        full_width=True,
    )


def sensitive_overlap_count(
    *,
    dataset_name: str,
    method_label: str,
    processed_dir: str | Path = PROCESSED,
    results_dir: str | Path = RESULTS,
) -> str:
    metadata = {spec.key: spec for spec in SYNTHETIC_DATASETS}
    spec = metadata.get(dataset_name)
    if spec is None or not spec.sensitive_columns:
        return "n/a"
    method_lookup = {value: key for key, value in METHOD_LABELS.items()}
    method = method_lookup.get(method_label)
    if method is None:
        return "n/a"
    real_path = Path(processed_dir) / f"{dataset_name}.csv"
    generated_path = Path(results_dir) / f"{dataset_name}_{method}_generated.csv"
    if not real_path.exists() or not generated_path.exists():
        return "missing"
    real = pd.read_csv(real_path)
    generated = pd.read_csv(generated_path)
    count = 0
    for column in spec.sensitive_columns:
        if column in real and column in generated:
            source_values = set(real[column].dropna().astype(str))
            count += int(generated[column].dropna().astype(str).isin(source_values).sum())
    return str(count)


def write_manifold_validation_table(
    validations: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    summary = summarize_manifold_validation(validations)
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    df = summary[
        [
            "dataset",
            "method_label",
            "out_hull_rate",
            "real_stress_median",
            "real_stress_q95",
            "generated_stress_median",
            "out_hull_stress_median",
            "out_hull_acceptance_at_real_q95",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "out_hull_rate": f"{UP} Out-hull rate",
            "real_stress_median": f"{DOWN} Real stress med.",
            "real_stress_q95": f"{DOWN} Real stress q95",
            "generated_stress_median": f"{DOWN} Gen. stress med.",
            "out_hull_stress_median": f"{DOWN} Out-hull stress med.",
            "out_hull_acceptance_at_real_q95": f"{UP} Out-hull accept.",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "manifold_validation.tex",
        "Frozen-Isomap manifold validation in DataFrameSampler latent space. Out-hull rate is the fraction of generated points outside the training convex hull. Out-hull accept is the fraction of out-of-hull generated points with insertion stress no larger than the held-out real 95th percentile. The validation CSV records the training and evaluated-point caps used for each run. Takeaway: displacement transport can produce out-of-hull points that often remain close to the held-out manifold-stress baseline, but this is diagnostic evidence only.",
        "tab:manifold-validation",
        float_format="%.3f",
        full_width=True,
    )


def write_mechanism_validation_table(
    validations: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    summary = summarize_mechanism_validation(validations)
    df = summary[
        [
            "dataset",
            "columns_evaluated",
            "mean_cardinality",
            "mean_nca_accuracy",
            "mean_majority_accuracy",
            "mean_pca_accuracy",
            "mean_raw_context_accuracy",
            "mean_lift_over_majority",
            "mean_lift_over_pca",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "columns_evaluated": "Columns",
            "mean_cardinality": "Mean card.",
            "mean_nca_accuracy": f"{UP} NCA acc.",
            "mean_majority_accuracy": f"{UP} Majority acc.",
            "mean_pca_accuracy": f"{UP} PCA acc.",
            "mean_raw_context_accuracy": f"{UP} Raw ctx acc.",
            "mean_lift_over_majority": f"{UP} NCA-majority",
            "mean_lift_over_pca": f"{UP} NCA-PCA",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "mechanism_validation.tex",
        "Mechanism validation for categorical NCA blocks. Each row aggregates capped held-out categorical prediction tests over categorical columns. NCA-majority and NCA-PCA are accuracy differences against majority and same-width PCA baselines. Takeaway: the supervised NCA blocks provide mechanism evidence where they beat majority and PCA references, and failures mark claim boundaries.",
        "tab:mechanism-validation",
        float_format="%.3f",
        full_width=True,
    )


def write_decoder_calibration_table(
    calibrations: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    summary = summarize_decoder_calibration(calibrations)
    df = summary[
        [
            "dataset",
            "cardinality_bucket",
            "columns_evaluated",
            "mean_accuracy",
            "mean_top_confidence",
            "mean_confidence_gap",
            "mean_negative_log_loss",
            "mean_brier_score",
            "mean_expected_calibration_error",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "cardinality_bucket": "Cardinality",
            "columns_evaluated": "Columns",
            "mean_accuracy": f"{UP} Acc.",
            "mean_top_confidence": f"{TARGET} Top conf.",
            "mean_confidence_gap": f"{DOWN} Conf.-acc.",
            "mean_negative_log_loss": f"{DOWN} NLL",
            "mean_brier_score": f"{DOWN} Brier",
            "mean_expected_calibration_error": f"{DOWN} ECE",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "decoder_calibration.tex",
        "Random-forest decoder calibration diagnostics by dataset and categorical-cardinality bucket. These are boundary diagnostics for probabilistic decoding rather than guarantees of calibrated sampling. Takeaway: probabilistic decoding is usable as an empirical inverse map, but uncertainty quality varies by dataset and cardinality.",
        "tab:decoder-calibration",
        float_format="%.3f",
        full_width=True,
    )


def write_sensitivity_validation_table(
    validations: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    summary = summarize_sensitivity_validation(validations)
    df = summary[
        [
            "parameter",
            "value",
            "setup_label",
            "n_components",
            "n_iterations",
            "nca_fit_sample_size",
            "lambda_",
            "max_constraint_retries",
            "calibrate_decoders",
            "datasets_evaluated",
            "mean_nn_distance_ratio",
            "mean_discrimination_accuracy",
            "mean_utility_lift",
            "mean_distribution_similarity_score",
            "mean_fit_seconds",
            "mean_sample_seconds",
        ]
    ].copy()
    setup_order = {
        "DataFrameSampler fast": 0,
        "DataFrameSampler default": 1,
        "DataFrameSampler accurate": 2,
    }
    parameter_order = {"setup": 0, "lambda": 1, "n_components": 2, "n_iterations": 3}
    df["_parameter_order"] = df["parameter"].map(parameter_order).fillna(99)
    df["_setup_order"] = df["setup_label"].map(setup_order).fillna(99)
    df = df.sort_values(["_parameter_order", "_setup_order", "parameter", "value"]).drop(columns=["_parameter_order", "_setup_order"])
    for column in ["n_components", "nca_fit_sample_size", "lambda_"]:
        df[column] = df[column].where(df[column].notna(), "")
    df = df.rename(
        columns={
            "parameter": "Parameter",
            "value": "Value",
            "setup_label": "Setup",
            "n_components": "NCA dim.",
            "n_iterations": "NCA iter.",
            "nca_fit_sample_size": "NCA sample",
            "lambda_": "Lambda",
            "max_constraint_retries": "Retries",
            "calibrate_decoders": "Calibration",
            "datasets_evaluated": "Datasets",
            "mean_nn_distance_ratio": f"{TARGET} NN ratio",
            "mean_discrimination_accuracy": f"{TARGET} Disc. acc.",
            "mean_utility_lift": f"{UP} Utility lift",
            "mean_distribution_similarity_score": f"{UP} Dist. score",
            "mean_fit_seconds": f"{DOWN} Fit s",
            "mean_sample_seconds": f"{DOWN} Sample s",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "sensitivity_validation.tex",
        "Capped setup and default-parameter sensitivity on the representative Adult Census Income dataset. The setup rows compare fast, default, and accurate operating points; the parameter rows perturb lambda, NCA dimension, and NCA iteration count around the default configuration. NCA sample reports the row fraction or cap used to estimate each NCA block. Takeaway: sensitivity is used to test default robustness under small perturbations, not to claim global hyperparameter optimality.",
        "tab:sensitivity-validation",
        float_format="%.3f",
        full_width=True,
    )


def write_imbalance_validation_table(
    validations: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    summary = summarize_imbalance_validation(validations)
    if summary.empty:
        summary = validations.copy()
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    df = summary[
        [
            "dataset",
            "method_label",
            "minority_class",
            "train_minority_rate",
            "augmented_minority_rate",
            "balanced_accuracy",
            "macro_f1",
            "minority_recall",
            "pr_auc",
            "synthetic_rows",
        ]
    ].copy()
    method_order = {
        "Real train": 0,
        "DataFrameSampler balanced": 1,
        "SMOTE/SMOTENC balanced": 2,
        "Stratified columns balanced": 3,
    }
    df["_method_order"] = df["method_label"].map(method_order).fillna(99)
    df = df.sort_values(["dataset", "_method_order", "method_label"]).drop(columns="_method_order")
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "minority_class": "Minority",
            "train_minority_rate": "Train min. rate",
            "augmented_minority_rate": f"{TARGET} Aug. min. rate",
            "balanced_accuracy": f"{UP} Bal. acc.",
            "macro_f1": f"{UP} Macro F1",
            "minority_recall": f"{UP} Min. recall",
            "pr_auc": f"{UP} PR AUC",
            "synthetic_rows": "Synth. rows",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "imbalance_validation.tex",
        "Secondary class-rebalancing diagnostic on selected binary target datasets. DataFrameSampler is fit only on minority-class feature rows, then generated rows are labelled as the minority class to rebalance the real training split; SMOTE is used for numeric-only data and SMOTENC when categorical features are present. Takeaway: imbalance results are boundary evidence for a secondary augmentation use case, not a claim that the method is a general imbalance-learning optimiser.",
        "tab:imbalance-validation",
        float_format="%.3f",
        full_width=True,
    )


def write_deep_reference_table(
    comparisons: pd.DataFrame,
    *,
    tables_dir: str | Path = TABLES,
) -> Path:
    df = comparisons[
        [
            "dataset",
            "method_label",
            "distribution_similarity_score",
            "discrimination_accuracy",
            "utility_lift",
            "fit_seconds",
            "sample_seconds",
            "peak_memory_mb",
        ]
    ].copy()
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "distribution_similarity_score": f"{UP} Dist. score",
            "discrimination_accuracy": f"{TARGET} Disc. acc.",
            "utility_lift": f"{UP} Utility lift",
            "fit_seconds": f"{DOWN} Fit s",
            "sample_seconds": f"{DOWN} Sample s",
            "peak_memory_mb": f"{DOWN} Peak MB",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "deep_reference_comparison.tex",
        "Adult high-capacity reference comparison. CTGAN is included as an optional SDV-based reference model with a global adversarial objective, not as a leaderboard target; DataFrameSampler and Gaussian copula rows are included from the same Adult baseline artefact for scale. Takeaway: the comparison locates DataFrameSampler against a modern deep generator while preserving the paper's emphasis on inspectability and setup cost.",
        "tab:deep-reference-comparison",
        float_format="%.3f",
        full_width=True,
    )


def write_ablation_table(*, tables_dir: str | Path = TABLES) -> Path:
    rows = [
        {
            "Component": "Per-column NCA blocks",
            "Expected effect": "Supervised categorical geometry",
            "Observed effect": "Mechanism validation table",
            "Claim status": "Supported only where lift is positive",
        },
        {
            "Component": "RF categorical decoders",
            "Expected effect": "Probabilistic inverse categories",
            "Observed effect": "Calibration diagnostics",
            "Claim status": "Boundary diagnostic",
        },
        {
            "Component": "Frozen-Isomap validation",
            "Expected effect": "Extrapolation evidence",
            "Observed effect": "Implemented",
            "Claim status": "Empirical diagnostic",
        },
        {
            "Component": "Mutual-neighbour fallback",
            "Expected effect": "Preserve sampleability",
            "Observed effect": "Unit tested",
            "Claim status": "Mechanism plausible",
        },
        {
            "Component": "Approximate KNN",
            "Expected effect": "Lower runtime on larger data",
            "Observed effect": "Planned",
            "Claim status": "Not yet tested",
        },
    ]
    return write_latex(
        pd.DataFrame(rows),
        Path(tables_dir) / "ablations.tex",
        "Ablation plan and current claim status. Takeaway: each mechanism component is treated as evidence to be checked, not as an assumed source of correctness.",
        "tab:ablations",
    )


def write_limitations_table(*, tables_dir: str | Path = TABLES) -> Path:
    rows = [
        {
            "Area": "Similarity",
            "Known": "Public and controlled metrics",
            "Unknown": "Stability on broader task collections",
            "Allowable scope": "Regime-specific mixed-type evidence",
        },
        {
            "Area": "Utility",
            "Known": "Held-out real utility tests",
            "Unknown": "General predictive advantage",
            "Allowable scope": "Utility only where lift is observed",
        },
        {
            "Area": "Privacy",
            "Known": "Nearest-neighbour and overlap diagnostics",
            "Unknown": "Linkage risk under a threat model",
            "Allowable scope": "No formal privacy claim",
        },
        {
            "Area": "Usability",
            "Known": "Notebook runtime and configs",
            "Unknown": "User study or setup-time distribution",
            "Allowable scope": "Practical workflow claim only",
        },
    ]
    return write_latex(
        pd.DataFrame(rows),
        Path(tables_dir) / "limitations_scope.tex",
        "Limitations and allowable conclusion scope under the current evidence. Takeaway: the paper's strongest defensible conclusion is practical, inspectable example generation under explicit boundaries.",
        "tab:limitations-scope",
        full_width=True,
    )


def write_latex(
    dataframe: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    *,
    float_format: str | None = None,
    full_width: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = dataframe.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        float_format=float_format,
    )
    latex = latex.replace(UP, r"$\uparrow$")
    latex = latex.replace(DOWN, r"$\downarrow$")
    latex = latex.replace(TARGET, r"$\rightarrow$")
    environment = "table*" if full_width else "table"
    width = r"\textwidth" if full_width else r"\columnwidth"
    latex = latex.replace(r"\begin{table}", rf"\begin{{{environment}}}", 1)
    latex = latex.replace(r"\end{table}", rf"\end{{{environment}}}", 1)
    latex = latex.replace(r"\begin{tabular}", rf"\begin{{adjustbox}}{{max width={width}}}" "\n" r"\begin{tabular}", 1)
    latex = latex.replace(r"\end{tabular}", r"\end{tabular}" "\n" r"\end{adjustbox}", 1)
    path.write_text(latex)
    dataframe.to_csv(path.with_suffix(".csv"), index=False)
    print(path)
    return path


def _mean_pm_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def generate_all_tables(
    *,
    results_dir: str | Path = RESULTS,
    processed_dir: str | Path = PROCESSED,
    tables_dir: str | Path = TABLES,
    dataset_metadata: list[DatasetTableMetadata] | None = None,
) -> list[Path]:
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    materialize_synthetic_datasets(processed_dir)
    comparisons = load_comparisons(results_dir)
    outputs = [
        write_dataset_table(
            processed_dir=processed_dir,
            tables_dir=tables_dir,
            dataset_metadata=dataset_metadata,
        ),
        write_method_table(tables_dir=tables_dir),
        write_main_measure_table(comparisons, tables_dir=tables_dir),
        write_cross_dataset_aggregation_table(comparisons, tables_dir=tables_dir),
        write_distribution_table(comparisons, tables_dir=tables_dir),
        write_downstream_table(comparisons, tables_dir=tables_dir),
        write_runtime_table(comparisons, tables_dir=tables_dir),
        write_synthetic_dataset_table(tables_dir=tables_dir),
        write_synthetic_results_table(
            comparisons,
            processed_dir=processed_dir,
            results_dir=results_dir,
            tables_dir=tables_dir,
        ),
        write_ablation_table(tables_dir=tables_dir),
        write_limitations_table(tables_dir=tables_dir),
    ]
    try:
        outputs.insert(
            -2,
            write_primary_uncertainty_table(
                load_primary_uncertainty(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_manifold_validation_table(
                load_manifold_validations(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_mechanism_validation_table(
                load_mechanism_validations(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_decoder_calibration_table(
                load_decoder_calibrations(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_sensitivity_validation_table(
                load_sensitivity_validations(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_imbalance_validation_table(
                load_imbalance_validations(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    try:
        outputs.insert(
            -2,
            write_deep_reference_table(
                load_deep_reference_comparisons(results_dir),
                tables_dir=tables_dir,
            ),
        )
    except FileNotFoundError:
        pass
    return outputs


def main() -> None:
    generate_all_tables()


if __name__ == "__main__":
    main()
