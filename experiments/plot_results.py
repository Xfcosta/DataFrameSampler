from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.mechanism_validation import summarize_decoder_calibration, summarize_mechanism_validation
from experiments.imbalance_validation import summarize_imbalance_validation
from experiments.sensitivity_validation import summarize_sensitivity_validation
from experiments.synthetic_data import SYNTHETIC_DATASETS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

METHOD_LABELS = {
    "dataframe_sampler": "DataFrameSampler",
    "latent_interpolation": "Latent interpolation",
    "row_bootstrap": "Bootstrap",
    "independent_columns": "Independent",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified",
    "latent_bootstrap": "Latent bootstrap",
    "real_train": "Real train",
    "dataframe_sampler_balanced": "DataFrameSampler",
    "smotenc_balanced": "SMOTE/SMOTENC",
    "stratified_columns_balanced": "Stratified columns",
}

METHOD_ORDER = [
    "dataframe_sampler",
    "row_bootstrap",
    "independent_columns",
    "gaussian_copula_empirical",
    "stratified_columns",
]


@dataclass(frozen=True)
class DistributionDashboardSpec:
    dataset_name: str
    generated_method: str
    numeric_column: str
    categorical_column: str
    correlation_columns: list[str]
    dataset_label: str | None = None
    generated_label: str | None = None


DEFAULT_DASHBOARD = DistributionDashboardSpec(
    dataset_name="titanic",
    generated_method="dataframe_sampler",
    numeric_column="age",
    categorical_column="sex",
    correlation_columns=["survived", "pclass", "age", "sibsp", "parch", "fare"],
    dataset_label="Titanic",
    generated_label="DataFrameSampler",
)


def load_comparisons(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = []
    for path in sorted(results_path.glob("*_baseline_comparison.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No *_baseline_comparison.csv files found. Run the notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    data["method_label"] = pd.Categorical(
        data["method_label"],
        categories=[METHOD_LABELS[m] for m in METHOD_ORDER],
        ordered=True,
    )
    return data.sort_values(["dataset", "method_label"])


def load_manifold_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_manifold_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No *_manifold_validation.csv files found. Run the notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def load_mechanism_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_mechanism_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No *_mechanism_validation.csv files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_decoder_calibrations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_decoder_calibration.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No *_decoder_calibration.csv files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_sensitivity_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_sensitivity_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No *_sensitivity_validation.csv files found. Run the notebooks first.")
    return pd.concat(frames, ignore_index=True)


def load_imbalance_validations(results_dir: str | Path = RESULTS) -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = [
        pd.read_csv(path)
        for path in sorted(results_path.glob("*_imbalance_validation.csv"))
    ]
    if not frames:
        raise FileNotFoundError("No *_imbalance_validation.csv files found. Run the selected notebooks first.")
    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = data["method"].map(METHOD_LABELS).fillna(data["method"])
    return data


def plot_baseline_similarity(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    metrics = [
        ("nn_distance_ratio", "NN distance ratio (higher)"),
        ("discrimination_accuracy", "Discrimination accuracy (near 0.5)"),
        ("utility_lift", "Utility lift (higher)"),
        ("distribution_similarity_score", "Distribution score (higher)"),
    ]
    datasets = list(data["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(14, max(6.5, 2.1 * len(datasets))), sharex=False)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, dataset in enumerate(datasets):
        subset = data[data["dataset"] == dataset]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.barh(subset["method_label"].astype(str), subset[metric], color="#4C78A8")
            ax.set_title(f"{dataset}: {title}")
            ax.grid(axis="x", alpha=0.25)
            if col_idx > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Primary four-measure summary for DataFrameSampler and simple baselines")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "baseline_similarity.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_distribution_dashboard(
    *,
    data_dir: str | Path = ROOT / "data" / "processed",
    results_dir: str | Path = RESULTS,
    figures_dir: str | Path = FIGURES,
    spec: DistributionDashboardSpec = DEFAULT_DASHBOARD,
) -> Path:
    real = pd.read_csv(Path(data_dir) / f"{spec.dataset_name}.csv")
    generated = pd.read_csv(
        Path(results_dir) / f"{spec.dataset_name}_{spec.generated_method}_generated.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(real[spec.numeric_column].dropna(), bins=20, alpha=0.55, label="real", density=True)
    axes[0].hist(
        generated[spec.numeric_column].dropna(),
        bins=20,
        alpha=0.55,
        label="generated",
        density=True,
    )
    axes[0].set_title(f"Numeric: {spec.numeric_column}")
    axes[0].legend()

    real_counts = real[spec.categorical_column].value_counts(normalize=True)
    gen_counts = generated[spec.categorical_column].value_counts(normalize=True)
    categories = real_counts.index.union(gen_counts.index)
    x = range(len(categories))
    axes[1].bar([i - 0.18 for i in x], real_counts.reindex(categories, fill_value=0), width=0.36, label="real")
    axes[1].bar([i + 0.18 for i in x], gen_counts.reindex(categories, fill_value=0), width=0.36, label="generated")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(categories, rotation=25)
    axes[1].set_title(f"Categorical: {spec.categorical_column}")
    axes[1].legend()

    numeric_cols = spec.correlation_columns
    diff = real[numeric_cols].corr(numeric_only=True) - generated[numeric_cols].corr(numeric_only=True)
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(numeric_cols)))
    axes[2].set_xticklabels(numeric_cols, rotation=45, ha="right")
    axes[2].set_yticks(range(len(numeric_cols)))
    axes[2].set_yticklabels(numeric_cols)
    axes[2].set_title("Correlation difference")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    dataset_label = spec.dataset_label or spec.dataset_name
    generated_label = spec.generated_label or spec.generated_method
    fig.suptitle(f"Distributional similarity dashboard: {dataset_label}, {generated_label}")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "distribution_dashboard.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_utility_cost_frontier(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    frontier = data.copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for dataset, subset in frontier.groupby("dataset"):
        ax.scatter(subset["sample_seconds"], subset["utility_lift"], label=dataset, s=70)
        for _, row in subset.iterrows():
            ax.annotate(str(row["method_label"]), (row["sample_seconds"], row["utility_lift"]), fontsize=7, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Sample time in seconds, log scale")
    ax.set_ylabel("Utility lift on held-out real data")
    ax.set_title("Utility-lift versus cost frontier for starter comparisons")
    ax.grid(alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "utility_cost_frontier.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_synthetic_controlled_similarity(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    synthetic_keys = [spec.key for spec in SYNTHETIC_DATASETS]
    subset = data[
        data["dataset"].isin(synthetic_keys)
        & data["method"].isin(["dataframe_sampler", "independent_columns", "row_bootstrap"])
    ].copy()
    if subset.empty:
        raise FileNotFoundError("No synthetic controlled comparison rows found. Run the synthetic notebook first.")

    metrics = [
        ("nn_distance_ratio", "NN distance ratio"),
        ("discrimination_accuracy", "Discrimination accuracy"),
        ("utility_lift", "Utility lift"),
    ]
    datasets = list(subset["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(13.5, 9.0), sharex=False)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, dataset in enumerate(datasets):
        ds = subset[subset["dataset"] == dataset]
        label = dataset.replace("synthetic_", "").replace("_", " ")
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.bar(ds["method_label"].astype(str), ds[metric], color="#4C78A8")
            ax.set_title(f"{label}: {title}")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Controlled synthetic regimes on primary measures")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "synthetic_controlled_similarity.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_synthetic_category_stress(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    synthetic_keys = [spec.key for spec in SYNTHETIC_DATASETS]
    subset = data[
        data["dataset"].isin(synthetic_keys)
        & data["method"].isin(
            [
                "dataframe_sampler",
                "independent_columns",
            ]
        )
    ].copy()
    if subset.empty:
        raise FileNotFoundError("No synthetic controlled comparison rows found. Run the synthetic notebook first.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, metric, title in [
        (axes[0], "categorical_coverage", "Category coverage"),
        (axes[1], "rare_category_preservation", "Rare-category preservation"),
    ]:
        pivot = subset.pivot(index="dataset", columns="method_label", values=metric)
        pivot = pivot.rename(index=lambda value: value.replace("synthetic_", "").replace("_", " "))
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.suptitle("Controlled categorical stress tests")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "synthetic_category_stress.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_manifold_validation_stress(
    data: pd.DataFrame,
    figures_dir: str | Path = FIGURES,
    *,
    max_datasets: int = 6,
) -> Path:
    data = data.copy()
    data["group"] = data.apply(_manifold_group_label, axis=1)
    selected = [
        dataset
        for dataset in data["dataset"].drop_duplicates().tolist()
        if not data[(data["dataset"] == dataset) & (data["group"] == "DFS out-hull")].empty
    ]
    if not selected:
        selected = data["dataset"].drop_duplicates().tolist()
    selected = selected[:max_datasets]
    subset = data[data["dataset"].isin(selected) & data["stress"].notna()].copy()
    if subset.empty:
        raise FileNotFoundError("No finite manifold validation stress values found.")

    groups = ["Held-out real", "DFS generated", "DFS out-hull", "Latent interpolation"]
    if "latent_bootstrap" in set(subset["method"]):
        groups.append("Latent bootstrap")
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, max(3.2, 2.2 * len(selected))), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, selected):
        frame = subset[subset["dataset"] == dataset]
        values = [frame.loc[frame["group"] == group, "stress"].dropna().to_numpy() for group in groups]
        labels = [group for group, group_values in zip(groups, values) if len(group_values)]
        values = [group_values for group_values in values if len(group_values)]
        ax.boxplot(values, tick_labels=labels, orientation="horizontal", showfliers=False)
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Normalized frozen-Isomap insertion stress")
    fig.suptitle("Manifold validation: held-out real stress versus generated stress")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "manifold_validation_stress.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _manifold_group_label(row: pd.Series) -> str:
    if row["sample_type"] == "real_test":
        return "Held-out real"
    if row["method"] == "dataframe_sampler" and bool(row.get("out_hull", False)):
        return "DFS out-hull"
    if row["method"] == "dataframe_sampler":
        return "DFS generated"
    if row["method"] == "latent_interpolation":
        return "Latent interpolation"
    if row["method"] == "latent_bootstrap":
        return "Latent bootstrap"
    return str(row.get("method_label", row["method"]))


def plot_mechanism_validation(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    summary = summarize_mechanism_validation(data)
    if summary.empty:
        raise FileNotFoundError("No mechanism validation rows found.")
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.42 * len(summary))))
    y = range(len(summary))
    ax.barh(
        [idx - 0.18 for idx in y],
        summary["mean_lift_over_majority"],
        height=0.34,
        label="NCA - majority",
        color="#4C78A8",
    )
    ax.barh(
        [idx + 0.18 for idx in y],
        summary["mean_lift_over_pca"],
        height=0.34,
        label="NCA - PCA",
        color="#F58518",
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["dataset"])
    ax.set_xlabel("Held-out categorical accuracy difference")
    ax.set_title("Mechanism validation: supervised NCA block lift")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "mechanism_validation.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_decoder_calibration(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    summary = summarize_decoder_calibration(data)
    if summary.empty:
        raise FileNotFoundError("No decoder calibration rows found.")
    summary["label"] = summary["dataset"].astype(str) + " / " + summary["cardinality_bucket"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.32 * len(summary))))
    axes[0].barh(summary["label"], summary["mean_expected_calibration_error"], color="#E45756")
    axes[0].set_title("Expected calibration error")
    axes[0].set_xlabel("ECE")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(summary["label"], summary["mean_negative_log_loss"], color="#72B7B2")
    axes[1].set_title("Negative log loss")
    axes[1].set_xlabel("NLL")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Decoder calibration diagnostics by dataset and categorical-cardinality bucket")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "decoder_calibration.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sensitivity_validation(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    summary = summarize_sensitivity_validation(data)
    if summary.empty:
        raise FileNotFoundError("No sensitivity validation rows found.")
    summary = summary[summary["parameter"].isin(["setup", "lambda", "n_components", "n_iterations"])].copy()
    summary["label"] = summary.apply(
        lambda row: str(row["setup_label"])
        if row["parameter"] == "setup"
        else f"{row['parameter']}={row['value']}",
        axis=1,
    )
    parameter_order = {"setup": 0, "lambda": 1, "n_components": 2, "n_iterations": 3}
    setup_order = {
        "DataFrameSampler fast": 0,
        "DataFrameSampler default": 1,
        "DataFrameSampler accurate": 2,
    }
    summary["_parameter_order"] = summary["parameter"].map(parameter_order).fillna(99)
    summary["_setup_order"] = summary["setup_label"].map(setup_order).fillna(99)
    subset = summary.sort_values(["_parameter_order", "_setup_order", "value"]).reset_index(drop=True)
    metrics = [
        ("mean_distribution_similarity_score", "Distribution score"),
        ("mean_utility_lift", "Utility lift"),
        ("mean_discrimination_accuracy", "Discrimination accuracy"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, max(4.8, 0.35 * len(subset))))
    for ax, (metric, title) in zip(axes, metrics):
        ax.barh(subset["label"].astype(str), subset[metric], color="#59A14F")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Representative DataFrameSampler setup and default-parameter sensitivity")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "sensitivity_validation.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_imbalance_validation(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    summary = summarize_imbalance_validation(data)
    if summary.empty:
        raise FileNotFoundError("No valid imbalance validation rows found.")
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    datasets = list(summary["dataset"].drop_duplicates())
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("minority_recall", "Minority recall"),
    ]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(13, max(4.0, 2.3 * len(datasets))), sharex=False)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    for row_idx, dataset in enumerate(datasets):
        subset = summary[summary["dataset"] == dataset]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.barh(subset["method_label"].astype(str), subset[metric], color="#B279A2")
            ax.set_title(f"{dataset}: {title}")
            ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Secondary class-rebalancing diagnostic")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "imbalance_validation.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_all_figures(
    *,
    results_dir: str | Path = RESULTS,
    figures_dir: str | Path = FIGURES,
    data_dir: str | Path = ROOT / "data" / "processed",
    dashboard_spec: DistributionDashboardSpec = DEFAULT_DASHBOARD,
) -> list[Path]:
    data = load_comparisons(results_dir)
    outputs = [
        plot_distribution_dashboard(
            data_dir=data_dir,
            results_dir=results_dir,
            figures_dir=figures_dir,
            spec=dashboard_spec,
        ),
        plot_baseline_similarity(data, figures_dir),
        plot_utility_cost_frontier(data, figures_dir),
    ]
    if any(data["dataset"].isin([spec.key for spec in SYNTHETIC_DATASETS])):
        outputs.extend(
            [
                plot_synthetic_controlled_similarity(data, figures_dir),
                plot_synthetic_category_stress(data, figures_dir),
            ]
        )
    try:
        outputs.append(plot_manifold_validation_stress(load_manifold_validations(results_dir), figures_dir))
    except FileNotFoundError:
        pass
    try:
        outputs.append(plot_mechanism_validation(load_mechanism_validations(results_dir), figures_dir))
    except FileNotFoundError:
        pass
    try:
        outputs.append(plot_decoder_calibration(load_decoder_calibrations(results_dir), figures_dir))
    except FileNotFoundError:
        pass
    try:
        outputs.append(plot_sensitivity_validation(load_sensitivity_validations(results_dir), figures_dir))
    except FileNotFoundError:
        pass
    try:
        outputs.append(plot_imbalance_validation(load_imbalance_validations(results_dir), figures_dir))
    except FileNotFoundError:
        pass
    return outputs


def main() -> None:
    outputs = generate_all_figures()
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
