# Dataset Descriptions

This folder contains one human-readable `DESCRIPTION.md` file for each dataset configured in `experiments/datasets.py`.

Each description includes:

- a short dataset-level summary;
- a paragraph for every column in the prepared CSV;
- a `POTENTIAL TARGET COLUMNS` section listing columns that can reasonably be used as predictive targets, with the task type marked as classification, regression, or ordinal classification.

## Files

- [Adult Census Income](adult/DESCRIPTION.md)
- [Titanic](titanic/DESCRIPTION.md)
- [Wisconsin Diagnostic Breast Cancer](breast_cancer/DESCRIPTION.md)
- [Pima Indians Diabetes](pima_diabetes/DESCRIPTION.md)
- [Bank Marketing](bank_marketing/DESCRIPTION.md)
- [UCI Heart Disease](heart_disease/DESCRIPTION.md)
- [Controlled correlated helpers](synthetic_correlated_helpers/DESCRIPTION.md)
- [Controlled high cardinality](synthetic_high_cardinality/DESCRIPTION.md)
- [Controlled rare categories](synthetic_rare_categories/DESCRIPTION.md)
- [Controlled sensitive identifier](synthetic_sensitive_identifier/DESCRIPTION.md)
