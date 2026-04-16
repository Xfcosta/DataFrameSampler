from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    {
        "method": "DFS default",
        "package": "DataFrameSampler",
        "family": "Latent bin-space sampler",
        "mixed": "Yes",
        "setup": "Low",
        "inspectability": "High",
        "optional": "None",
    },
    {
        "method": "DFS manual",
        "package": "DataFrameSampler",
        "family": "Latent bin-space sampler",
        "mixed": "Yes",
        "setup": "Medium",
        "inspectability": "High",
        "optional": "None",
    },
    {
        "method": "DFS LLM-style",
        "package": "DataFrameSampler",
        "family": "Latent bin-space sampler",
        "mixed": "Yes",
        "setup": "Low/medium",
        "inspectability": "High",
        "optional": "OpenAI for live mode",
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
    "dataframe_sampler_default": "DFS default",
    "dataframe_sampler_manual": "DFS manual",
    "dataframe_sampler_llm_assisted": "DFS LLM-style",
    "row_bootstrap": "Row bootstrap",
    "independent_columns": "Independent columns",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified columns",
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
        "Datasets used in the starter experiments.",
        "tab:datasets",
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
    return write_latex(df, Path(tables_dir) / "methods.tex", "Baseline and competitor method metadata.", "tab:methods")


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
        "Starter distributional similarity results. Lower is better for KS, categorical TV, and association difference; higher is better for histogram overlap.",
        "tab:distributional-similarity",
        float_format="%.3f",
    )


def write_main_measure_table(comparisons: pd.DataFrame, *, tables_dir: str | Path = TABLES) -> Path:
    selected_methods = {
        "dataframe_sampler_manual",
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
        "Primary experiment measures. NN ratio compares synthetic-to-real nearest-neighbour distance with natural real-to-real nearest-neighbour distance; values below one indicate closer-than-natural synthetic rows. Discrimination accuracy near 0.5 is better. Utility lift is the change from adding synthetic rows to the real training set. Histogram overlap is higher-is-better and categorical JSD is lower-is-better.",
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
        "Utility lift test. A baseline model is trained on real training data, then compared with a model trained on real plus generated rows and evaluated on held-out real rows.",
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
        "DFS default": "1 config",
        "DFS manual": "Category embedding",
        "DFS LLM-style": "LLM-style dict",
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
        "Starter usability, runtime, and traced Python allocation measurements. Peak MB is the maximum traced allocation peak observed during fit or sample, not total process RSS.",
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
        "Controlled synthetic datasets used to isolate specific boundary regimes.",
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
        "dataframe_sampler_manual",
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
        "Focused results for controlled synthetic regimes. Sensitive overlap counts exact reuse of patient identifiers in the controlled identifier dataset.",
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


def write_ablation_table(*, tables_dir: str | Path = TABLES) -> Path:
    rows = [
        {
            "Component": "Helper-column vectorization",
            "Expected effect": "Better categorical context",
            "Observed effect": "Planned",
            "Claim status": "Not yet tested",
        },
        {
            "Component": "LLM-style configuration",
            "Expected effect": "Lower setup burden",
            "Observed effect": "Starter metrics only",
            "Claim status": "Exploratory",
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
        "Ablation plan and current claim status.",
        "tab:ablations",
    )


def write_limitations_table(*, tables_dir: str | Path = TABLES) -> Path:
    rows = [
        {
            "Area": "Similarity",
            "Known": "Starter Adult/Titanic metrics",
            "Unknown": "Multi-dataset stability",
            "Allowable scope": "Exploratory mixed-type evidence",
        },
        {
            "Area": "Utility",
            "Known": "Metric code exists",
            "Unknown": "Train-synthetic/test-real results",
            "Allowable scope": "No utility claim yet",
        },
        {
            "Area": "Privacy",
            "Known": "Exact selected-value checks",
            "Unknown": "Linkage risk",
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
        "Limitations and allowable conclusion scope under the current evidence.",
        "tab:limitations-scope",
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
    return [
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


def main() -> None:
    generate_all_tables()


if __name__ == "__main__":
    main()
