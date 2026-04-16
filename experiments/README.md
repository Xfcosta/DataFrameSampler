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
  vectorizing columns, target columns, and row limits.
- `workflow.py`: reusable notebook workflow for loading, profiling, starter
  sampling, baseline comparison, and output writing.
- `make_tables.py`: reusable table-generation functions for publication
  artifacts.
- `plot_results.py`: reusable figure-generation functions for publication
  artifacts.
- `notebooks/`: one notebook per dataset.
- `data/raw/`: downloaded source files.
- `data/processed/`: cleaned CSV files consumed by notebooks.
- `results/`: generated tables and intermediate experiment outputs.
- `figures/`: generated plots for the paper.

The primary comparison CSVs report four main measures: nearest-neighbor
distance against natural real-data neighbor distances, real-versus-synthetic
discrimination, utility lift from adding generated rows to real training data,
and distribution similarity through histogram overlap/divergence summaries.

## Current Datasets

- Adult Census Income from the UCI Machine Learning Repository.
- Titanic from the seaborn example datasets repository.
- Four deterministic controlled synthetic datasets for TODO point 5.

Run:

```bash
python experiments/download_datasets.py
```

Then open:

```bash
jupyter notebook experiments/notebooks/adult.ipynb
jupyter notebook experiments/notebooks/titanic.ipynb
jupyter notebook experiments/notebooks/synthetic_controlled.ipynb
```

The notebooks are intentionally thin. Each notebook selects a dataset key from
`experiments/datasets.py`, calls `run_configured_dataset_experiment`, and then
displays the returned profile, starter sample report, and baseline comparison.
To add another dataset, add a `DatasetExperimentConfig`, create a small
notebook that changes `DATASET_NAME`, then rerun the table and figure helpers:

```bash
python experiments/make_tables.py
python experiments/plot_results.py
```
