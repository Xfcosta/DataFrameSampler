# Controlled Correlated Helpers

This synthetic dataset is generated from latent numeric factors so that numeric columns and categorical embeddings have known dependencies. It is used to test whether DataFrameSampler preserves simple mixed-type relationships.

## Columns

### spend_score

`spend_score` represents a synthetic spending-intensity score. It is generated from a latent factor plus noise, so it is positively correlated with some behavioural columns and negatively related to the risk-like score.

### visit_rate

`visit_rate` represents a synthetic engagement or visit-frequency score. It depends on both the main latent factor and a secondary seasonal factor.

### risk_score

`risk_score` represents a synthetic risk-like score. It is generated to move partly against `spend_score` while also depending on a secondary latent factor.

### helper_band

`helper_band` is a categorical discretization of the main latent factor into low, mid, and high context bands. It is intentionally designed as a helper category that should be recoverable from numeric context.

### segment

`segment` is a categorical customer-style segment derived from spend, risk, and visit patterns. It contains categories such as premium, watch, low-touch, and standard.

### target

`target` is a binary synthetic outcome derived from a noisy combination of spending, visit rate, and risk score. It is the primary predictive target for this controlled dataset.

## POTENTIAL TARGET COLUMNS

- `target`: classification. This is the primary target for controlled binary prediction.
- `segment`: classification. This can be used to evaluate whether generated data preserves the relationship between numeric behaviour and categorical segment labels.
- `helper_band`: classification. This can be used to test recovery of latent-context bands from numeric features.
- `spend_score`: regression. This can be used to predict a synthetic spending measurement from visit, risk, and category context.
- `visit_rate`: regression. This can be used to predict a synthetic engagement measurement.
- `risk_score`: regression. This can be used to predict the synthetic risk-like measurement.
