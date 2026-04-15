# Prompt For LLM Vectorizing Configuration

You are configuring DataFrameSampler for a tabular dataset.

DataFrameSampler generates synthetic rows by:

1. vectorizing categorical columns into numeric values,
2. discretizing every column into bins,
3. sampling in latent bin space,
4. decoding bins back into observed source values.

Your task is to recommend `vectorizing_columns_dict`.

Rules:

- Recommend entries only for categorical/non-numeric columns.
- Helper columns must be numeric and present in the dataframe.
- Helper columns should plausibly explain the categorical value.
- Use 1 to 4 helper columns per categorical column.
- Do not recommend a helper if it leaks a target/outcome column.
- If no useful numeric helpers exist for a categorical column, omit it.
- Also recommend one global `embedding_method`.
- Also recommend one `knn_backend`; prefer `sklearn` unless there is a strong reason otherwise.

Return JSON with:

```json
{
  "recommendations": [
    {
      "column": "personName",
      "helper_columns": ["age", "country_id"],
      "rationale": "Age and country help locate names in a demographic space.",
      "confidence": 0.85
    }
  ],
  "sampled_columns": ["personName", "age", "country"],
  "embedding_method": "pca",
  "knn_backend": "sklearn",
  "notes": "Short explanation of tradeoffs."
}
```
