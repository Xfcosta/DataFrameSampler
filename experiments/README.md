# DataFrameSampler Experiments

This folder contains reproducible experiment notebooks and local data
preparation helpers for the paper.

## Layout

- `download_datasets.py`: downloads and prepares the datasets used by the
  notebooks.
- `notebooks/`: one notebook per dataset.
- `data/raw/`: downloaded source files.
- `data/processed/`: cleaned CSV files consumed by notebooks.
- `results/`: generated tables and intermediate experiment outputs.
- `figures/`: generated plots for the paper.

## Current Datasets

- Adult Census Income from the UCI Machine Learning Repository.
- Titanic from the seaborn example datasets repository.

Run:

```bash
python experiments/download_datasets.py
```

Then open:

```bash
jupyter notebook experiments/notebooks/adult.ipynb
jupyter notebook experiments/notebooks/titanic.ipynb
```

The notebooks currently establish the first reproducible dataset-specific
workflow: load prepared data, inspect mixed column types, fit DataFrameSampler,
generate example rows, and compute lightweight schema/similarity checks. They
are intentionally not yet the full benchmark suite from `TODO.md`.

