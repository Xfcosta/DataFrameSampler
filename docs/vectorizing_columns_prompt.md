# Prompt For LLM Sampler Configuration

You are configuring DataFrameSampler for a tabular dataset.

DataFrameSampler generates synthetic rows by:

1. converting every usable column to numeric values,
2. discretizing every converted column into bins,
3. sampling in latent bin space,
4. decoding bins back into observed source values.

Categorical conversion is fixed:

- high-cardinality identifier-like non-numeric columns are discarded;
- binary non-numeric columns are mapped to `0/1`;
- other non-numeric columns are one-hot encoded and embedded to one numeric
  coordinate.

Your task is to recommend sampled columns, one global `embedding_method`, and
one `knn_backend`.

Rules:

- Exclude direct identifiers such as names, emails, phone numbers, addresses,
  free-text IDs, or mostly unique code columns.
- Keep compact categorical columns that should appear in generated output; the
  vectorizer will embed them automatically.
- Avoid leakage-like target columns unless the user explicitly wants to sample
  them.
- Prefer `pca` for `embedding_method` unless there is a clear reason for a
  nonlinear reducer.
- Prefer `sklearn` for `knn_backend` unless there is a strong reason otherwise.

Return JSON with:

```json
{
  "recommendations": [
    {
      "column": "personName",
      "action": "exclude",
      "rationale": "Mostly unique direct identifier.",
      "confidence": 0.95
    }
  ],
  "sampled_columns": ["age", "city", "country"],
  "embedding_method": "pca",
  "knn_backend": "sklearn",
  "notes": "Short explanation of tradeoffs."
}
```
