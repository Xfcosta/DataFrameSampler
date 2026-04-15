from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
PROCESSED = EXPERIMENTS / "data" / "processed"
TABLES = ROOT / "publication" / "tables"

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


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    comparisons = load_comparisons()
    write_dataset_table()
    write_method_table()
    write_distribution_table(comparisons)
    write_downstream_table()
    write_runtime_table(comparisons)
    write_ablation_table()
    write_limitations_table()


def load_comparisons() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(RESULTS.glob("*_baseline_comparison.csv"))]
    if not frames:
        raise FileNotFoundError("No baseline comparison files found. Run the notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def write_dataset_table() -> None:
    rows = []
    dataset_meta = {
        "adult": {
            "name": "Adult Census Income",
            "domain": "Census / income",
            "sensitive": "None selected",
            "rationale": "Generic mixed-type benchmark",
        },
        "titanic": {
            "name": "Titanic",
            "domain": "Passenger survival",
            "sensitive": "None selected",
            "rationale": "Small mixed-type smoke benchmark",
        },
    }
    for key, meta in dataset_meta.items():
        df = pd.read_csv(PROCESSED / f"{key}.csv")
        numeric = len(df.select_dtypes(include="number").columns)
        categorical = len(df.columns) - numeric
        rows.append(
            {
                "Dataset": meta["name"],
                "Domain": meta["domain"],
                "Rows": len(df),
                "Numeric": numeric,
                "Categorical": categorical,
                "Missing": int(df.isna().sum().sum()),
                "Sensitive": meta["sensitive"],
                "Rationale": meta["rationale"],
            }
        )
    write_latex(pd.DataFrame(rows), TABLES / "datasets.tex", "Datasets used in the starter experiments.", "tab:datasets")


def write_method_table() -> None:
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
    write_latex(df, TABLES / "methods.tex", "Baseline and competitor method metadata.", "tab:methods")


def write_distribution_table(comparisons: pd.DataFrame) -> None:
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
    write_latex(
        df,
        TABLES / "distributional_similarity.tex",
        "Starter distributional similarity results. Lower is better for KS, categorical TV, and association difference; higher is better for histogram overlap.",
        "tab:distributional-similarity",
        float_format="%.3f",
    )


def write_downstream_table() -> None:
    rows = [
        {
            "Evaluation": "Train synthetic, test real",
            "Adult": "Planned",
            "Titanic": "Planned",
            "Baseline": "Compared with generated samples",
            "Uncertainty": "Repeated splits planned",
        },
        {
            "Evaluation": "Train real, test real",
            "Adult": "Planned",
            "Titanic": "Planned",
            "Baseline": "Upper reference",
            "Uncertainty": "Repeated splits planned",
        },
        {
            "Evaluation": "Train bootstrap, test real",
            "Adult": "Planned",
            "Titanic": "Planned",
            "Baseline": "Row bootstrap reference",
            "Uncertainty": "Repeated splits planned",
        },
    ]
    write_latex(
        pd.DataFrame(rows),
        TABLES / "downstream_utility.tex",
        "Downstream utility table scaffold. Values remain planned until supervised evaluations are run.",
        "tab:downstream-utility",
    )


def write_runtime_table(comparisons: pd.DataFrame) -> None:
    df = comparisons[["dataset", "method_label", "fit_seconds", "sample_seconds"]].copy()
    setup_steps = {
        "DFS default": "1 config",
        "DFS manual": "Manual helpers",
        "DFS LLM-style": "LLM-style dict",
        "Row bootstrap": "None",
        "Independent columns": "None",
        "Gaussian copula": "None",
        "Stratified columns": "Target column",
    }
    df["Commands/LOC"] = "Notebook cell"
    df["Memory"] = "Not measured"
    df["Tuning/config"] = df["method_label"].map(setup_steps).fillna("Not recorded")
    df = df.rename(
        columns={
            "dataset": "Dataset",
            "method_label": "Method",
            "fit_seconds": "Fit s",
            "sample_seconds": "Sample s",
        }
    )
    write_latex(
        df,
        TABLES / "usability_runtime.tex",
        "Starter usability and runtime measurements. Memory is not yet instrumented.",
        "tab:usability-runtime",
        float_format="%.3f",
    )


def write_ablation_table() -> None:
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
    write_latex(pd.DataFrame(rows), TABLES / "ablations.tex", "Ablation plan and current claim status.", "tab:ablations")


def write_limitations_table() -> None:
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
    write_latex(
        pd.DataFrame(rows),
        TABLES / "limitations_scope.tex",
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
) -> None:
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


if __name__ == "__main__":
    main()
