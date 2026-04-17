# Latent Numeric Views And 2D Projection

DataFrameSampler transforms dataframe rows into the fitted NCA latent
matrix:

```python
from dataframe_sampler import DataFrameSampler

sampler = DataFrameSampler(
    n_components=2,
    n_iterations=2,
    n_neighbours=8,
    knn_backend="sklearn",
    random_state=42,
)
sampler.fit(df)

latent = sampler.transform(df)
generated_df = sampler.generate(n_samples=len(df))
generated_latent = sampler.transform(generated_df)
```

`transform` returns a NumPy array, not a DataFrame. Its columns are ordered as:

- standardized non-binary numeric columns;
- one NCA latent block per categorical column.

Binary columns are categorical even when stored as `0/1`, boolean, or another
two-valued numeric dtype. High-cardinality categoricals are warned about but
still used.

The experiment helper `experiments.numeric_projection` wraps this array in a
temporary DataFrame for plotting:

```python
from experiments.numeric_projection import plot_numeric_projection_triptych

fig = plot_numeric_projection_triptych(
    sampler,
    original=df,
    generated=generated_df,
    target_column="target",
    title="Dataset name",
    reducer="umap",
    random_state=42,
)
```

The helper transforms original and generated rows with the fitted sampler,
projects both latent views into a shared two-dimensional space with UMAP when
available or PCA otherwise, and displays original-only, generated-only, and
superimposed panels. Pass `output_path=...` only when you explicitly want a
saved diagnostic figure; numeric projection PDFs are not part of the current
paper artifact set.
