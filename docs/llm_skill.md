# DataFrameSampler LLM Skill

Use this skill when an LLM needs to configure or run DataFrameSampler on a CSV,
Parquet file, or pandas DataFrame.

## Goal

Generate a useful DataFrameSampler configuration:

- optional `sampled_columns`;
- optional `embedding_method`;
- optional KNN backend choice.

Then use either the Python API or CLI to fit the sampler and generate data.

## How To Inspect The Dataset

For every column, identify:

- column name;
- dtype;
- whether it is numeric, binary, categorical, identifier-like, or free text;
- examples of values;
- missing count;
- unique count;
- likely semantic role, such as ID, category, place, person, amount, time, or target.

## Categorical Policy

DataFrameSampler has one fitted categorical conversion policy:

- high-cardinality identifier-like non-numeric columns are discarded;
- binary non-numeric columns are mapped to `0/1`;
- other non-numeric columns are one-hot encoded and embedded to one numeric
  coordinate.

The default high-cardinality threshold discards a non-numeric column only when
its alphabet is larger than both 50 distinct values and 30% of the row count.

## Choosing `sampled_columns`

Include columns that should appear in generated output. Exclude direct
identifiers, free-text fields, names, addresses, emails, phone numbers, and
mostly unique operational IDs unless the user explicitly wants to test
memorization or overlap risk.

## Choosing `embedding_method`

Use:

- `pca` as the default for compact categorical alphabets;
- `mds`, `kernel_pca`, `isomap`, or `lle` only when there is a clear nonlinear
  or manifold reason.

For reducers that cannot transform new data directly, the fitted vectorizer
learns the training category mapping and a regressor fallback from one-hot
indicators to embedded values.

## Choosing KNN Backend

Use:

- `exact` for small datasets or debugging;
- `sklearn` as the safe faster default without optional dependencies;
- `pynndescent` for larger dense tabular data when optional dependencies are installed;
- `hnswlib` for large vector-search-style workloads with euclidean/cosine metrics.

## Python Pattern

```python
from dataframe_sampler import ConcreteDataFrameSampler

sampler = ConcreteDataFrameSampler(
    n_bins=10,
    n_neighbours=5,
    embedding_method="pca",
    knn_backend="sklearn",
    random_state=42,
)
sampler.fit(df)
generated_df = sampler.sample(n_samples=len(df))
```

## Auto Mode With OpenAI

If the `openai` optional dependency is installed and `OPENAI_API_KEY` is set,
use:

```python
from dataframe_sampler import ConcreteDataFrameSampler, suggest_sampler_config_with_openai

config = suggest_sampler_config_with_openai(df)
sampler = ConcreteDataFrameSampler(
    sampled_columns=config["sampled_columns"],
    embedding_method=config["embedding_method"],
    knn_backend=config["knn_backend"],
    random_state=42,
)
```

Review the generated configuration before using it on sensitive or high-stakes
data.

## Sensitive-Value Replacement Before Sampling

When the user asks to anonymize sensitive columns, replace those values before
fitting the sampler:

```python
from dataframe_sampler import anonymize_columns_with_openai, assert_no_value_overlap

anon_df, report = anonymize_columns_with_openai(
    dataframe=df,
    source_dataframe=df,
    columns=["personName", "email"],
)

sampler.fit(anon_df)
generated_df = sampler.sample(n_samples=len(anon_df))
assert_no_value_overlap(df, generated_df, ["personName", "email"])
```
