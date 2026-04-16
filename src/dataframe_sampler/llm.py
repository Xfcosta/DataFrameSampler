import json

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


LLM_VECTORISING_COLUMNS_SYSTEM_PROMPT = """
You configure DataFrameSampler for a tabular dataset.

Goal:
- Choose sampled columns, a categorical embedding method, and a KNN backend.
- DataFrameSampler now uses one fixed categorical policy: high-cardinality
  identifier-like non-numeric columns are discarded by the vectorizer, binary
  columns are mapped to 0/1, and other non-numeric columns are one-hot encoded
  and embedded to one numeric dimension.

Guidelines:
- Avoid leakage-like target columns if the profile marks them as likely target/outcome, unless there is no better choice.
- Exclude direct identifiers such as names, addresses, emails, phone numbers, free text IDs, or mostly unique code columns from sampled_columns.
- Choose a global embedding method. Prefer pca unless there is an explicit nonlinear reason for kernel_pca/isomap/lle.
- Choose a KNN backend. Prefer sklearn as the safe default; use exact only for tiny/debug datasets; recommend optional ANN backends only when the user has said those dependencies are available.
""".strip()


def profile_dataframe_for_llm(dataframe, max_examples=5, max_unique=20):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("dataframe must contain at least one row.")

    columns = []
    for column in dataframe.columns:
        series = dataframe[column]
        entry = {
            "name": str(column),
            "dtype": str(series.dtype),
            "semantic_type": _semantic_type(series),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "examples": _examples(series, max_examples=max_examples),
        }
        if is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            entry["summary"] = {
                "min": _json_number(numeric.min()),
                "max": _json_number(numeric.max()),
                "mean": _json_number(numeric.mean()),
            }
        elif series.nunique(dropna=True) <= max_unique:
            entry["values"] = _examples(series.drop_duplicates(), max_examples=max_unique)
        columns.append(entry)

    return {
        "row_count": int(len(dataframe)),
        "columns": columns,
        "numeric_columns": [entry["name"] for entry in columns if entry["semantic_type"] == "numeric"],
        "categorical_columns": [
            entry["name"] for entry in columns if entry["semantic_type"] in ("categorical", "boolean", "datetime")
        ],
    }


def suggest_sampler_config_with_openai(
    dataframe,
    model="gpt-4o-mini",
    client=None,
    max_examples=5,
    max_unique=20,
):
    profile = profile_dataframe_for_llm(dataframe, max_examples=max_examples, max_unique=max_unique)
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI auto configuration requires the optional dependency. "
                "Install it with: pip install 'dataframe-sampler[llm]'"
            ) from exc
        client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": LLM_VECTORISING_COLUMNS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Given this dataframe profile, return a DataFrameSampler configuration as JSON.\n\n"
                    + json.dumps(profile, indent=2)
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "dataframe_sampler_auto_config",
                "strict": True,
                "schema": _sampler_config_schema(),
            }
        },
    )

    payload = json.loads(_response_text(response))
    return _sanitize_sampler_config(payload, dataframe)


def _sampler_config_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["recommendations", "sampled_columns", "embedding_method", "knn_backend", "notes"],
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["column", "action", "rationale", "confidence"],
                    "properties": {
                        "column": {"type": "string"},
                        "action": {"type": "string", "enum": ["keep", "exclude"]},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "sampled_columns": {"type": "array", "items": {"type": "string"}},
            "embedding_method": {
                "type": "string",
                "enum": [
                    "mds",
                    "pca",
                    "incremental_pca",
                    "kernel_pca",
                    "sparse_pca",
                    "truncated_svd",
                    "factor_analysis",
                    "fast_ica",
                    "isomap",
                    "lle",
                    "spectral_embedding",
                    "tsne",
                ],
            },
            "knn_backend": {
                "type": "string",
                "enum": ["exact", "sklearn", "pynndescent", "hnswlib", "annoy"],
            },
            "notes": {"type": "string"},
        },
    }


def _sanitize_sampler_config(payload, dataframe):
    columns = set(map(str, dataframe.columns))

    recommendations = []
    excluded_columns = set()
    for recommendation in payload.get("recommendations", []):
        column = recommendation.get("column")
        if column not in columns:
            continue
        clean = {
            "column": column,
            "action": recommendation.get("action", "keep"),
            "rationale": recommendation.get("rationale", ""),
            "confidence": recommendation.get("confidence", 0),
        }
        recommendations.append(clean)
        if clean["action"] == "exclude":
            excluded_columns.add(column)

    sampled_columns = [
        column
        for column in payload.get("sampled_columns", [])
        if column in columns and column not in excluded_columns
    ]
    if not sampled_columns:
        sampled_columns = [column for column in map(str, dataframe.columns) if column not in excluded_columns]

    return {
        "sampled_columns": sampled_columns,
        "embedding_method": payload.get("embedding_method", "pca"),
        "knn_backend": payload.get("knn_backend", "sklearn"),
        "recommendations": recommendations,
        "notes": payload.get("notes", ""),
    }


def _response_text(response):
    if hasattr(response, "output_text"):
        return response.output_text
    raise TypeError("OpenAI response did not include output_text.")


def _semantic_type(series):
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _examples(series, max_examples):
    values = series.dropna().head(max_examples).tolist()
    return [_json_value(value) for value in values]


def _json_value(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_number(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value
