from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments.synthetic_data import SYNTHETIC_DATASETS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

METHOD_LABELS = {
    "dataframe_sampler_default": "DFS default",
    "dataframe_sampler_manual": "DFS manual",
    "latent_interpolation": "Latent interpolation",
    "row_bootstrap": "Bootstrap",
    "independent_columns": "Independent",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified",
}

METHOD_ORDER = [
    "dataframe_sampler_default",
    "dataframe_sampler_manual",
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
    generated_method="dataframe_sampler_manual",
    numeric_column="age",
    categorical_column="class",
    correlation_columns=["survived", "pclass", "age", "sibsp", "parch", "fare"],
    dataset_label="Titanic",
    generated_label="manual DataFrameSampler",
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
    fig.suptitle("Primary four-measure summary for DataFrameSampler configurations and simple baselines")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "baseline_similarity.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_configuration_competitors(data: pd.DataFrame, figures_dir: str | Path = FIGURES) -> Path:
    subset = data[data["method"].str.startswith("dataframe_sampler_")].copy()
    metrics = [
        ("nn_distance_ratio", "NN distance ratio (higher)"),
        ("discrimination_accuracy", "Discrimination accuracy (near 0.5)"),
        ("utility_lift", "Utility lift (higher)"),
        ("sample_seconds", "Sample seconds (lower)"),
    ]
    datasets = list(subset["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(max(10, 3.0 * len(datasets)), 8), sharex=False)
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (metric, title) in enumerate(metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            ds = subset[subset["dataset"] == dataset]
            ax.bar(ds["method_label"].astype(str), ds[metric], color="#59A14F")
            ax.set_title(f"{dataset}: {title}")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=25)
    fig.suptitle("DataFrameSampler configuration competitors on primary measures")
    fig.tight_layout()
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "configuration_competitors.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_what_context(figures_dir: str | Path = FIGURES) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    boxes = [
        ("Source table", "Adult / Titanic\\nMixed numeric + categorical\\nGovernance constraint", 0.04, 0.55),
        ("Sampler", "DataFrameSampler\\ninspectable NCA latent-space\\nconfiguration choices", 0.38, 0.55),
        ("Example data", "Generated table\\nsame schema\\nfor non-production use", 0.72, 0.55),
        ("Uses", "Tests\\nDashboards\\nDemos\\nNotebooks", 0.72, 0.15),
        ("Review", "Explicit caveat:\\nnot formal privacy\\nnot clinical simulator", 0.38, 0.15),
    ]
    for title, body, x, y in boxes:
        add_box(ax, x, y, title, body)
    add_arrow(ax, (0.28, 0.70), (0.38, 0.70))
    add_arrow(ax, (0.62, 0.70), (0.72, 0.70))
    add_arrow(ax, (0.86, 0.55), (0.86, 0.37))
    add_arrow(ax, (0.72, 0.24), (0.62, 0.24))
    ax.set_title("WHAT: intended use context for simple, inspectable tabular example generation", pad=14)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "what_context.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_how_pipeline(figures_dir: str | Path = FIGURES) -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    steps = [
        ("Dataframe", "pandas / CSV\\nAdult, Titanic"),
        ("Encode context", "standardize numerics\\none-hot categoricals"),
        ("NCA blocks", "per-categorical\\nsupervised latent"),
        ("Neighbour chain", "anchor -> neighbour\\n-> neighbour"),
        ("Decode", "inverse scale\\nRF categorical draw"),
        ("Output + trace", "same schema\\nexplainable path"),
    ]
    xs = [0.03, 0.21, 0.39, 0.57, 0.75, 0.90]
    for idx, ((title, body), x) in enumerate(zip(steps, xs)):
        add_box(ax, x, 0.52, title, body, width=0.11, height=0.28)
        if idx < len(steps) - 1:
            add_arrow(ax, (x + 0.11, 0.66), (xs[idx + 1], 0.66))
    ax.text(
        0.5,
        0.18,
        "Inspectable generation: anchor row + neighbour chain + latent difference + decoded values",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("HOW: DataFrameSampler pipeline and explanation trace", pad=14)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    output = figures_path / "how_pipeline.pdf"
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
        & data["method"].isin(["dataframe_sampler_manual", "independent_columns", "row_bootstrap"])
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
                "dataframe_sampler_default",
                "dataframe_sampler_manual",
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
    if row["method"] == "dataframe_sampler_manual" and bool(row.get("out_hull", False)):
        return "DFS out-hull"
    if row["method"] == "dataframe_sampler_manual":
        return "DFS generated"
    if row["method"] == "latent_interpolation":
        return "Latent interpolation"
    return str(row.get("method_label", row["method"]))


def add_box(ax, x, y, title, body, width=0.24, height=0.25):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1,
        edgecolor="#333333",
        facecolor="#F4F6F8",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.70, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + width / 2, y + height * 0.35, body, ha="center", va="center", fontsize=9)


def add_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.3, color="#333333"))


def generate_all_figures(
    *,
    results_dir: str | Path = RESULTS,
    figures_dir: str | Path = FIGURES,
    data_dir: str | Path = ROOT / "data" / "processed",
    dashboard_spec: DistributionDashboardSpec = DEFAULT_DASHBOARD,
) -> list[Path]:
    data = load_comparisons(results_dir)
    outputs = [
        plot_what_context(figures_dir),
        plot_how_pipeline(figures_dir),
        plot_distribution_dashboard(
            data_dir=data_dir,
            results_dir=results_dir,
            figures_dir=figures_dir,
            spec=dashboard_spec,
        ),
        plot_baseline_similarity(data, figures_dir),
        plot_configuration_competitors(data, figures_dir),
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
    return outputs


def main() -> None:
    outputs = generate_all_figures()
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
