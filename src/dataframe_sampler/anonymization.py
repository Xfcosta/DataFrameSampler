import json
import re

import pandas as pd


ANONYMIZATION_SYSTEM_PROMPT = """
You generate surrogate replacement values for sensitive dataframe columns.

Rules:
- Return plausible synthetic values with the same broad semantic type and formatting.
- Do not reuse any original value.
- Do not output real public figures or famous entities.
- Prefer obviously fictional but realistic values where possible.
- Preserve row-level consistency only through the mappings requested by the user.
- Return JSON only.
""".strip()


def anonymize_columns_with_openai(
    dataframe,
    columns,
    source_dataframe=None,
    model="gpt-4o-mini",
    client=None,
    batch_size=100,
    max_retries=3,
    consistent=True,
):
    if source_dataframe is None:
        source_dataframe = dataframe
    _validate_anonymization_columns(dataframe, columns)
    _validate_anonymization_columns(source_dataframe, columns)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI anonymization requires the optional dependency. "
                "Install it with: pip install 'dataframe-sampler[llm]'"
            ) from exc
        client = OpenAI()

    anonymized = dataframe.copy()
    mappings = {}
    for column in columns:
        source_values = _ordered_unique_non_null(source_dataframe[column])
        values_to_replace = _ordered_unique_non_null(dataframe[column]) if consistent else dataframe[column].dropna().tolist()
        mapping = _generate_column_mapping(
            column=column,
            values=values_to_replace,
            source_values=source_values,
            model=model,
            client=client,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        anonymized[column] = anonymized[column].map(lambda value: mapping.get(value, value))
        mappings[column] = mapping

    assert_no_value_overlap(source_dataframe, anonymized, columns)
    return anonymized, {
        "columns": list(columns),
        "mappings": mappings,
        "privacy_note": (
            "This is sensitive-value substitution with overlap checks, not a formal privacy guarantee."
        ),
    }


def assert_no_value_overlap(source_dataframe, candidate_dataframe, columns):
    _validate_anonymization_columns(source_dataframe, columns)
    _validate_anonymization_columns(candidate_dataframe, columns)
    overlaps = {}
    for column in columns:
        source_values = {_normalize_value(value) for value in source_dataframe[column].dropna()}
        candidate_values = {_normalize_value(value) for value in candidate_dataframe[column].dropna()}
        overlap = sorted(value for value in source_values & candidate_values if value)
        if overlap:
            overlaps[column] = overlap
    if overlaps:
        raise ValueError("Anonymized values overlap source values: %s" % overlaps)
    return True


def _generate_column_mapping(column, values, source_values, model, client, batch_size, max_retries):
    mapping = {}
    source_normalized = {_normalize_value(value) for value in source_values}
    used_normalized = set()

    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        remaining = list(batch)
        for attempt in range(max_retries):
            if not remaining:
                break
            replacements = _request_replacements(
                column=column,
                values=remaining,
                source_values=source_values,
                model=model,
                client=client,
            )
            next_remaining = []
            for original in remaining:
                replacement = replacements.get(original)
                normalized = _normalize_value(replacement)
                if not replacement or normalized in source_normalized or normalized in used_normalized:
                    next_remaining.append(original)
                    continue
                mapping[original] = replacement
                used_normalized.add(normalized)
            remaining = next_remaining
        if remaining:
            raise ValueError(
                "Could not generate non-overlapping replacements for column %r values: %s"
                % (column, remaining)
            )

    return mapping


def _request_replacements(column, values, source_values, model, client):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": ANONYMIZATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "column": column,
                        "values_to_replace": [_json_value(value) for value in values],
                        "forbidden_source_values": [_json_value(value) for value in source_values],
                    },
                    indent=2,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "dataframe_sampler_anonymization",
                "strict": True,
                "schema": _anonymization_schema(),
            }
        },
    )
    payload = json.loads(_response_text(response))
    return {
        item["original"]: item["replacement"]
        for item in payload.get("replacements", [])
        if "original" in item and "replacement" in item
    }


def _anonymization_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["replacements"],
        "properties": {
            "replacements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["original", "replacement"],
                    "properties": {
                        "original": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                },
            }
        },
    }


def _ordered_unique_non_null(series):
    values = []
    seen = set()
    for value in series.dropna().tolist():
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _validate_anonymization_columns(dataframe, columns):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Unknown anonymization columns: %s" % missing)


def _normalize_value(value):
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _json_value(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _response_text(response):
    if hasattr(response, "output_text"):
        return response.output_text
    raise TypeError("OpenAI response did not include output_text.")
