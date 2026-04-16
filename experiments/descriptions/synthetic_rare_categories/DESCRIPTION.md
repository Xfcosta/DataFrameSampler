# Controlled Rare Categories

This synthetic dataset is designed to test rare-category preservation. It contains numeric recency-frequency-monetary style variables and a categorical signal with deliberately rare values.

## Columns

### recency_days

`recency_days` represents the number of days since the last synthetic event or purchase. Lower values indicate more recent activity.

### frequency

`frequency` represents a synthetic count of repeated events or purchases. It is an integer behavioural frequency feature.

### monetary

`monetary` represents a synthetic monetary value generated from a log-normal process. It is a skewed numeric value similar to spend or revenue.

### rare_signal

`rare_signal` is a categorical feature with common categories and deliberately rare categories such as rare gold, rare silver, and rare bronze. It is included to measure rare-category coverage and preservation.

### lifecycle

`lifecycle` represents a categorical lifecycle state derived from recency, frequency, and monetary value. It includes states such as active, cold, high-value, and steady.

### target

`target` is a binary synthetic outcome influenced by frequency, rare-category membership, recency, and noise. It is the primary predictive target for this dataset.

## POTENTIAL TARGET COLUMNS

- `target`: classification. This is the primary target for controlled binary prediction.
- `rare_signal`: classification. This is useful for stress testing rare-category prediction and preservation.
- `lifecycle`: classification. This can be used to predict lifecycle state from recency, frequency, and monetary value.
- `recency_days`: regression. This can be used as a time-since-event prediction target.
- `frequency`: regression or ordinal classification. This can be used as a count prediction target.
- `monetary`: regression. This can be used as a skewed value prediction target.
