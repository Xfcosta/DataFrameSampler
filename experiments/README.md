# DataFrameSampler Experiments

This folder contains reproducible experiment notebooks and local data
preparation helpers for the paper.

## Layout

- `download_datasets.py`: downloads and prepares the datasets used by the
  notebooks.
- `synthetic_data.py`: creates deterministic controlled datasets for correlated
  helpers, high-cardinality categoricals, rare categories, and sensitive
  identifiers.
- `datasets.py`: dataset-specific parameters, including sampler configs,
  redundant columns to drop, target columns, and row limits.
- `workflow.py`: reusable notebook workflow for loading, profiling, starter
  sampling, baseline comparison, manifold validation, mechanism validation,
  decoder calibration, and output writing.
- `make_tables.py`: reusable table-generation functions for publication
  artifacts.
- `plot_results.py`: reusable figure-generation functions for publication
  artifacts.
- `numeric_projection.py`: fitted-sampler numeric transformations and
  original/generated/superimposed 2D projection plots. It uses UMAP when
  available and PCA as a fallback.
- `manifold_validation.py`: convex-hull and frozen-Isomap insertion-stress
  diagnostics in the fitted DataFrameSampler latent space.
- `mechanism_validation.py`: capped NCA-block ablations and random-forest
  decoder calibration diagnostics.
- `predictive.py`: target-choice reporting and real-test predictive
  comparisons between models trained on real rows and models trained on
  synthetic rows generated from the real training split.
- `notebooks/`: one notebook per dataset.
- `data/raw/`: downloaded source files.
- `data/processed/`: cleaned CSV files consumed by notebooks.
- `descriptions/`: human-readable per-dataset column descriptions and
  potential predictive targets.
- `results/`: generated tables and intermediate experiment outputs.
- `figures/`: generated plots for the paper.

The primary comparison CSVs report four main measures: nearest-neighbor
distance against natural real-data neighbor distances, real-versus-synthetic
discrimination, utility lift from adding generated rows to real training data,
and distribution similarity through histogram overlap/divergence summaries.
The manifold validation CSVs report pointwise held-out-real and generated
insertion stress plus convex-hull membership in the sampler latent space. The
mechanism and calibration CSVs report held-out categorical prediction lift for
NCA blocks and probability-quality diagnostics for categorical decoders.

## Current Datasets

- Adult Census Income from the UCI Machine Learning Repository.
- Titanic from the seaborn example datasets repository.
- Wisconsin Diagnostic Breast Cancer from the UCI Machine Learning Repository.
- Pima Indians Diabetes from a public CSV mirror of the classic diabetes
  classification dataset.
- Bank Marketing from the UCI Machine Learning Repository.
- Heart Disease Cleveland from the UCI Machine Learning Repository.
- Four deterministic controlled synthetic datasets for TODO point 5.

Run:

```bash
python experiments/download_datasets.py
```

Then open:

```bash
jupyter notebook experiments/notebooks/adult.ipynb
jupyter notebook experiments/notebooks/titanic.ipynb
jupyter notebook experiments/notebooks/breast_cancer.ipynb
jupyter notebook experiments/notebooks/pima_diabetes.ipynb
jupyter notebook experiments/notebooks/bank_marketing.ipynb
jupyter notebook experiments/notebooks/heart_disease.ipynb
jupyter notebook experiments/notebooks/synthetic_controlled.ipynb
```

The notebooks are intentionally thin. Each notebook selects a dataset key from
`experiments/datasets.py`, displays the preprocessing and vectorization plans,
plots pairwise features only after the configured sampler has reduced every
column to its numeric representation, states the configured target column, calls
`run_configured_dataset_experiment`, and displays the returned profile, starter
sample report, numeric projection triptych, baseline comparison, manifold
validation summary, mechanism validation summary, decoder calibration summary,
and predictive target evaluation.
To add another dataset, add a `DatasetExperimentConfig`, create a small
notebook that changes `DATASET_NAME`, then rerun the table and figure helpers:

```bash
python experiments/make_tables.py
python experiments/plot_results.py
```
