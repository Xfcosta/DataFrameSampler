# Controlled High Cardinality

This synthetic dataset is designed to stress categorical coverage and helper-column embeddings. It includes normal numeric business-style features plus a high-cardinality SKU-like categorical column.

## Columns

### account_value

`account_value` represents a synthetic account or customer value. It is generated from a log-normal process influenced by a latent factor.

### tenure_months

`tenure_months` represents how long the synthetic account has been active. It is an integer duration feature correlated with the latent factor.

### activity_score

`activity_score` represents synthetic account engagement or activity. It is a bounded numeric score related to the same latent structure as account value and tenure.

### region

`region` represents a categorical geographic or market region. It is generated from the latent factor with noise so that it has a relationship to the numeric columns.

### sku_code

`sku_code` represents a high-cardinality product or plan identifier. It is intentionally included to test whether a sampler preserves many categorical values without collapsing rare or moderately rare categories.

### plan_tier

`plan_tier` represents a categorical service tier such as basic, team, or enterprise. It is derived partly from account value.

### target

`target` is a binary synthetic outcome that depends on activity, plan tier, and tenure. It is the primary predictive target for this dataset.

## POTENTIAL TARGET COLUMNS

- `target`: classification. This is the primary target for controlled binary prediction.
- `plan_tier`: classification. This can be used to test whether tier labels remain predictable from account value and tenure.
- `region`: classification. This can be used to test categorical reconstruction from numeric context.
- `sku_code`: classification. This is a high-cardinality classification target and is useful for stress testing, but it is harder and less stable than lower-cardinality targets.
- `account_value`: regression. This can be used to predict synthetic account value from tenure, activity, and categorical context.
- `tenure_months`: regression or ordinal classification. This can be used to predict customer age or account duration.
- `activity_score`: regression. This can be used to predict synthetic engagement.
