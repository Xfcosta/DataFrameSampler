# Numeric Views And 2D Projection

DataFrameSampler can render any fitted dataframe row as the pure numeric view
used internally before bin encoding:

```python
from dataframe_sampler import ConcreteDataFrameSampler

sampler = ConcreteDataFrameSampler(
    n_bins=10,
    n_neighbours=8,
    vectorizing_columns_dict={
        "category_column": ["numeric_helper_1", "numeric_helper_2"],
    },
    embedding_method="pca",
    knn_backend="sklearn",
    random_state=42,
)
sampler.fit(df)

numeric_df = sampler.transform(df)
generated_df = sampler.sample(n_samples=len(df))
generated_numeric_df = sampler.transform(generated_df)
```

`transform` preserves the dataframe columns but converts every column to a
numeric representation:

- numeric columns are converted to floats with median imputation for missing
  values;
- categorical columns with configured helper columns use the selected
  one-dimensional embedding;
- categorical columns without helper columns use frequency encoding;
- `sampled_columns`, when configured, is respected.

This numeric view is useful for inspection, distance calculations, and shared
visualization of original and generated rows.

The experiment helper `experiments.numeric_projection` builds on this interface:

```python
from experiments.numeric_projection import plot_numeric_projection_triptych

fig = plot_numeric_projection_triptych(
    sampler,
    original=df,
    generated=generated_df,
    title="Dataset name",
    reducer="umap",
    random_state=42,
    output_path="experiments/figures/dataset_numeric_projection.pdf",
)
```

The helper transforms original and generated rows together with the fitted
sampler, imputes and standardizes the numeric values, projects them to two
dimensions with UMAP when `umap-learn` is installed, and falls back to PCA
otherwise. The resulting figure has three panels: original only, generated
only, and both superimposed.
