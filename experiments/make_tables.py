from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.manifold_validation import summarize_manifold_validation
from experiments.mechanism_validation import summarize_decoder_calibration, summarize_mechanism_validation
from experiments.proposed_setups import PROPOSED_SAMPLER_SETUPS
from experiments.sensitivity_validation import summarize_sensitivity_validation
from experiments.synthetic_data import SYNTHETIC_DATASETS, materialize_synthetic_datasets


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
PROCESSED = EXPERIMENTS / "data" / "processed"
TABLES = ROOT / "publication" / "tables"


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
        "family": "Default setup used in baseline comparisons",
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


def load_deep_reference_comparisons(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(Path(results_dir).glob("*_deep_reference_comparison.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No deep reference comparison files found. Run the Adult CTGAN reference first.")
    data = pd.concat(frames, ignore_index=True)
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
        "Baseline and competitor method metadata. Takeaway: the comparisons emphasize low-setup, inspectable baselines that match the paper's practical-use claim.",
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
            "numeric_ks_statistic": "KS",
            "categorical_total_variation": "Cat. TV",
            "mean_abs_association_difference": "Assoc. diff.",
            "numeric_histogram_overlap": "Hist. overlap",
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
            "nn_distance_ratio": "NN ratio",
            "nn_suspiciously_close_rate": "NN close rate",
            "discrimination_accuracy": "Disc. acc.",
            "discrimination_privacy_score": "Disc. privacy",
            "utility_lift": "Utility lift",
            "distribution_histogram_overlap": "Hist. overlap",
            "distribution_categorical_jsd": "Cat. JSD",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "main_measures.tex",
        "Primary experiment measures. NN ratio compares synthetic-to-real nearest-neighbour distance with natural real-to-real nearest-neighbour distance; values below one indicate closer-than-natural synthetic rows. Discrimination accuracy near 0.5 is better. Utility lift is the change from adding synthetic rows to the real training set. Histogram overlap is higher-is-better and categorical JSD is lower-is-better. Takeaway: DataFrameSampler is best read as a balanced example generator, not as a single-metric winner.",
        "tab:main-measures",
        float_format="%.3f",
        full_width=True,
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
            "utility_real_score": "Real train score",
            "utility_augmented_score": "Augmented score",
            "utility_lift": "Utility lift",
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
        "DataFrameSampler": "NCA config",
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
            "fit_seconds": "Fit s",
            "sample_seconds": "Sample s",
            "fit_peak_memory_mb": "Fit peak MB",
            "sample_peak_memory_mb": "Sample peak MB",
            "peak_memory_mb": "Peak MB",
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
            "numeric_ks_statistic": "KS",
            "categorical_total_variation": "Cat. TV",
            "categorical_coverage": "Cat. coverage",
            "rare_category_preservation": "Rare preserve",
            "mean_abs_association_difference": "Assoc. diff.",
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
            "out_hull_rate": "Out-hull rate",
            "real_stress_median": "Real stress med.",
            "real_stress_q95": "Real stress q95",
            "generated_stress_median": "Gen. stress med.",
            "out_hull_stress_median": "Out-hull stress med.",
            "out_hull_acceptance_at_real_q95": "Out-hull accept.",
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
            "mean_nca_accuracy": "NCA acc.",
            "mean_majority_accuracy": "Majority acc.",
            "mean_pca_accuracy": "PCA acc.",
            "mean_raw_context_accuracy": "Raw ctx acc.",
            "mean_lift_over_majority": "NCA-majority",
            "mean_lift_over_pca": "NCA-PCA",
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
            "mean_accuracy": "Acc.",
            "mean_top_confidence": "Top conf.",
            "mean_confidence_gap": "Conf.-acc.",
            "mean_negative_log_loss": "NLL",
            "mean_brier_score": "Brier",
            "mean_expected_calibration_error": "ECE",
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
            "setup_label",
            "n_iterations",
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
    df["_setup_order"] = df["setup_label"].map(setup_order).fillna(99)
    df = df.sort_values(["_setup_order", "setup_label"]).drop(columns="_setup_order")
    df = df.rename(
        columns={
            "setup_label": "Setup",
            "n_iterations": "NCA iter.",
            "max_constraint_retries": "Retries",
            "calibrate_decoders": "Calibration",
            "datasets_evaluated": "Datasets",
            "mean_nn_distance_ratio": "NN ratio",
            "mean_discrimination_accuracy": "Disc. acc.",
            "mean_utility_lift": "Utility lift",
            "mean_distribution_similarity_score": "Dist. score",
            "mean_fit_seconds": "Fit s",
            "mean_sample_seconds": "Sample s",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "sensitivity_validation.tex",
        "Capped comparison of the three proposed DataFrameSampler setups on the representative Adult Census Income dataset. Fast uses no NCA iteration, no retry budget, and no calibration; default uses one NCA iteration, five retries, and no calibration; accurate uses two NCA iterations, twenty retries, and calibrated decoders. Takeaway: setup choice is presented as a practical speed--accuracy tradeoff illustration, not as a new full benchmark claim.",
        "tab:sensitivity-validation",
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
            "distribution_similarity_score": "Dist. score",
            "discrimination_accuracy": "Disc. acc.",
            "utility_lift": "Utility lift",
            "fit_seconds": "Fit s",
            "sample_seconds": "Sample s",
            "peak_memory_mb": "Peak MB",
        }
    )
    return write_latex(
        df,
        Path(tables_dir) / "deep_reference_comparison.tex",
        "Adult high-capacity reference comparison. CTGAN is included as an optional SDV-based reference model with a global adversarial objective, not as a leaderboard target. Takeaway: the comparison locates DataFrameSampler against a modern deep generator while preserving the paper's emphasis on inspectability and setup cost.",
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
