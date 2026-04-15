# DataFrameSampler LLM Skill

Use this skill when an LLM needs to configure or run DataFrameSampler on a CSV,
Parquet file, or pandas DataFrame.

## Goal

Generate a useful DataFrameSampler configuration:

- `vectorizing_columns_dict`
- optional `sampled_columns`
- optional `embedding_method`
- optional KNN backend choice

Then use either the Python API or CLI to fit the sampler and generate data.

## How To Inspect The Dataset

For every column, identify:

- column name
- dtype
- whether it is numeric or categorical
- examples of values
- missing count
- unique count
- likely semantic role, such as ID, category, place, person, amount, time, target

## Choosing `vectorizing_columns_dict`

Only add entries for categorical/non-numeric columns that have meaningful
numeric helper columns.

Good examples:

- `personName`: use `age`, `country_id`, `income`, or other numeric demographic/context columns.
- `city`: use `country_id`, `region_id`, latitude/longitude, or numeric location encodings.
- `productName`: use price, category_id, rating, or numeric product descriptors.
- `diagnosisLabel`: use numeric clinical measurements only if generating that label is appropriate.

Avoid:

- using the categorical column itself as a helper
- using free-text columns as helpers
- using helper columns that do not exist
- using non-numeric helper columns
- overloading with every numeric column; prefer 1 to 4 meaningful helpers
- using target/outcome columns as helpers when that would leak label information

If no meaningful helper exists, omit the categorical column and let
DataFrameSampler use frequency encoding.

## Choosing `embedding_method`

Use:

- `pca` for mostly linear numeric helper spaces.
- `mds` when the helper columns represent mixed semantic distances.
- `kernel_pca`, `isomap`, or `lle` only when there is a clear nonlinear reason.

Default to `pca` for broad tabular use unless the original MDS behavior is
specifically desired.

## Choosing KNN Backend

Use:

- `exact` for small datasets or debugging.
- `sklearn` as the safe faster default without optional dependencies.
- `pynndescent` for larger dense tabular data when optional dependencies are installed.
- `hnswlib` for large vector-search-style workloads with euclidean/cosine metrics.

## Python Pattern

```python
from dataframe_sampler import ConcreteDataFrameSampler

vectorizing_columns_dict = {
    "personName": ["age", "country_id"],
    "city": ["country_id"],
}

sampler = ConcreteDataFrameSampler(
    n_bins=10,
    n_neighbours=5,
    vectorizing_columns_dict=vectorizing_columns_dict,
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
from dataframe_sampler import suggest_sampler_config_with_openai

config = suggest_sampler_config_with_openai(df)
sampler = ConcreteDataFrameSampler(
    vectorizing_columns_dict=config["vectorizing_columns_dict"],
    sampled_columns=config["sampled_columns"],
    embedding_method=config["embedding_method"],
    knn_backend="sklearn",
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
generated_df = sampler.sample(n_samples=len(df))
assert_no_value_overlap(df, generated_df, ["personName", "email"])
```

CLI:

```bash
dataframe-sampler \
  --input_filename input.csv \
  --output_filename generated.csv \
  --anonymize_columns personName \
  --anonymize_columns email
```

This is surrogate value replacement, not a formal privacy guarantee.

## CLI Auto Mode

The CLI can ask OpenAI to fill omitted configuration values:

```bash
dataframe-sampler \
  -A \
  --input_filename input.csv \
  --output_filename generated.csv \
  --n_samples 100 \
  --random_state 42
```

The user remains in control. Values explicitly supplied by the user are kept:

- `-f/--vectorizing_columns_dict_filename` overrides the LLM mapping.
- `-c/--sampled_columns` overrides LLM sampled columns.
- `--embedding_method` overrides the LLM embedding method.
- `--knn_backend` overrides the LLM KNN backend.
