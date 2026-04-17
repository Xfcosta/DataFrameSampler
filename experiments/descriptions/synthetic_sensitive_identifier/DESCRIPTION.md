# Controlled Sensitive Identifier

This synthetic dataset is designed to expose exact source-value reuse risk. It includes a unique patient-like identifier alongside medical-style measurements and categorical care-context columns.

## Columns

### patient_id

`patient_id` is a unique synthetic patient-like identifier. It is intentionally sensitive in the experiment design and should normally be excluded before generating examples.

### age

`age` represents a synthetic patient age in years. It is generated from a clipped normal distribution.

### lab_score

`lab_score` represents a synthetic clinical laboratory score. It is a numeric measurement related weakly to age and used in the construction of the risk flag.

### visits

`visits` represents a synthetic count of visits. It is an integer utilization feature influenced by the lab-score range.

### condition

`condition` represents a categorical clinical context or condition group, such as routine, cardio, endocrine, respiratory, or oncology review. It is a synthetic medical-style category.

### ward

`ward` represents a categorical care location or ward assignment. It is a synthetic operational context variable.

### risk_flag

`risk_flag` is a binary synthetic outcome derived from lab score, visit count, and condition. It is the primary predictive target for this controlled dataset.

## POTENTIAL TARGET COLUMNS

- `risk_flag`: classification. This is the primary target for controlled medical-style risk prediction.
- `condition`: classification. This can be used to predict condition group from age, lab score, visits, and ward context.
- `ward`: classification. This can be used as a care-location prediction target, although it is generated mostly independently.
- `age`: regression. This can be used as a demographic reconstruction target, though this may be sensitive in real medical settings.
- `lab_score`: regression. This can be used as a clinical-measurement prediction target.
- `visits`: regression or ordinal classification. This can be used as a utilization-count prediction target.
- `patient_id`: high-cardinality classification. This should generally not be used as a predictive target except for leakage or memorization diagnostics.
