from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

METHOD_LABELS = {
    "dataframe_sampler_default": "DFS default",
    "dataframe_sampler_manual": "DFS manual",
    "dataframe_sampler_llm_assisted": "DFS LLM-style",
    "row_bootstrap": "Bootstrap",
    "independent_columns": "Independent",
    "gaussian_copula_empirical": "Gaussian copula",
    "stratified_columns": "Stratified",
}

METHOD_ORDER = [
    "dataframe_sampler_default",
    "dataframe_sampler_manual",
    "dataframe_sampler_llm_assisted",
    "row_bootstrap",
    "independent_columns",
    "gaussian_copula_empirical",
    "stratified_columns",
]


def load_comparisons() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS.glob("*_baseline_comparison.csv")):
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


def plot_baseline_similarity(data: pd.DataFrame) -> Path:
    metrics = [
        ("numeric_ks_statistic", "Numeric KS (lower)"),
        ("categorical_total_variation", "Categorical TV (lower)"),
        ("mean_abs_association_difference", "Association diff. (lower)"),
        ("numeric_histogram_overlap", "Histogram overlap (higher)"),
    ]
    datasets = list(data["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(14, 6.5), sharex=False)
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
    fig.suptitle("Distributional and association metrics for DataFrameSampler configurations and simple baselines")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "baseline_similarity.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_configuration_competitors(data: pd.DataFrame) -> Path:
    subset = data[data["method"].str.startswith("dataframe_sampler_")].copy()
    metrics = [
        ("numeric_ks_statistic", "Numeric KS (lower)"),
        ("categorical_total_variation", "Categorical TV (lower)"),
        ("mean_abs_association_difference", "Association diff. (lower)"),
        ("sample_seconds", "Sample seconds (lower)"),
    ]
    datasets = list(subset["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(10, 8), sharex=False)
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
    fig.suptitle("DataFrameSampler configuration competitors")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "configuration_competitors.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_what_context() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    boxes = [
        ("Source table", "Adult / Titanic\\nMixed numeric + categorical\\nGovernance constraint", 0.04, 0.55),
        ("Sampler", "DataFrameSampler\\ninspectable bin-space\\nconfiguration choices", 0.38, 0.55),
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
    output = FIGURES / "what_context.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_how_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    steps = [
        ("Dataframe", "pandas / CSV\\nAdult, Titanic"),
        ("Optional\\nanonymization", "selected columns\\nsurrogate values"),
        ("Vectorize", "numeric + categorical\\nhelper columns"),
        ("Encode bins", "latent integer\\nbin-space"),
        ("Neighbour chain", "anchor -> neighbour\\n-> neighbour"),
        ("Decode", "bins back to\\ncolumn values"),
        ("Output + trace", "same schema\\nexplainable path"),
    ]
    xs = [0.02, 0.17, 0.32, 0.47, 0.62, 0.77, 0.90]
    for idx, ((title, body), x) in enumerate(zip(steps, xs)):
        add_box(ax, x, 0.52, title, body, width=0.11, height=0.28)
        if idx < len(steps) - 1:
            add_arrow(ax, (x + 0.11, 0.66), (xs[idx + 1], 0.66))
    ax.text(
        0.5,
        0.18,
        "Inspectable generation: anchor row + neighbour chain + latent difference + decoded bins",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("HOW: DataFrameSampler pipeline and explanation trace", pad=14)
    output = FIGURES / "how_pipeline.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_llm_configuration_flow() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    boxes = [
        ("Dataframe profile", "names, dtypes\\nmissingness\\nexamples", 0.05, 0.56),
        ("LLM recommendation", "vectorizing dict\\nsampled columns\\nembedding + KNN", 0.36, 0.56),
        ("User overrides", "explicit CLI/API\\nchoices win", 0.67, 0.56),
        ("Sampler config", "fixed dictionary\\nrecorded in notebook", 0.36, 0.16),
    ]
    for title, body, x, y in boxes:
        add_box(ax, x, y, title, body, width=0.22, height=0.25)
    add_arrow(ax, (0.27, 0.69), (0.36, 0.69))
    add_arrow(ax, (0.58, 0.69), (0.67, 0.69))
    add_arrow(ax, (0.78, 0.56), (0.58, 0.29))
    add_arrow(ax, (0.47, 0.56), (0.47, 0.41))
    ax.set_title("LLM-assisted configuration flow", pad=14)
    output = FIGURES / "llm_configuration_flow.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_distribution_dashboard() -> Path:
    real = pd.read_csv(ROOT / "data" / "processed" / "titanic.csv")
    generated = pd.read_csv(RESULTS / "titanic_dataframe_sampler_manual_generated.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(real["age"].dropna(), bins=20, alpha=0.55, label="real", density=True)
    axes[0].hist(generated["age"].dropna(), bins=20, alpha=0.55, label="generated", density=True)
    axes[0].set_title("Numeric: age")
    axes[0].legend()

    real_counts = real["class"].value_counts(normalize=True)
    gen_counts = generated["class"].value_counts(normalize=True)
    categories = real_counts.index.union(gen_counts.index)
    x = range(len(categories))
    axes[1].bar([i - 0.18 for i in x], real_counts.reindex(categories, fill_value=0), width=0.36, label="real")
    axes[1].bar([i + 0.18 for i in x], gen_counts.reindex(categories, fill_value=0), width=0.36, label="generated")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(categories, rotation=25)
    axes[1].set_title("Categorical: class")
    axes[1].legend()

    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    diff = real[numeric_cols].corr(numeric_only=True) - generated[numeric_cols].corr(numeric_only=True)
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(numeric_cols)))
    axes[2].set_xticklabels(numeric_cols, rotation=45, ha="right")
    axes[2].set_yticks(range(len(numeric_cols)))
    axes[2].set_yticklabels(numeric_cols)
    axes[2].set_title("Correlation difference")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Distributional similarity dashboard: Titanic, manual DataFrameSampler")
    fig.tight_layout()
    output = FIGURES / "distribution_dashboard.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_utility_cost_frontier(data: pd.DataFrame) -> Path:
    frontier = data.copy()
    frontier["similarity_score"] = (
        (1 - frontier["numeric_ks_statistic"].clip(0, 1))
        + frontier["numeric_histogram_overlap"].clip(0, 1)
        + (1 - frontier["categorical_total_variation"].clip(0, 1))
        + (1 - frontier["mean_abs_association_difference"].clip(0, 1))
    ) / 4
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for dataset, subset in frontier.groupby("dataset"):
        ax.scatter(subset["sample_seconds"], subset["similarity_score"], label=dataset, s=70)
        for _, row in subset.iterrows():
            ax.annotate(str(row["method_label"]), (row["sample_seconds"], row["similarity_score"]), fontsize=7, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Sample time in seconds, log scale")
    ax.set_ylabel("Composite similarity score, higher is better")
    ax.set_title("Utility versus cost frontier for starter comparisons")
    ax.grid(alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()
    output = FIGURES / "utility_cost_frontier.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


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


def main() -> None:
    data = load_comparisons()
    outputs = [
        plot_what_context(),
        plot_how_pipeline(),
        plot_distribution_dashboard(),
        plot_baseline_similarity(data),
        plot_configuration_competitors(data),
        plot_utility_cost_frontier(data),
        plot_llm_configuration_flow(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
