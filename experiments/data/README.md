# Experiment Data

This directory is populated by:

```bash
python experiments/download_datasets.py
```

Raw downloaded files are stored under `raw/`; cleaned CSVs used by notebooks are
stored under `processed/`.

The current datasets are public benchmark/example datasets:

- Adult Census Income, Wisconsin Diagnostic Breast Cancer, Bank Marketing, and
  Heart Disease from the UCI Machine Learning Repository.
- Titanic from the seaborn example datasets repository.
- Pima Indians Diabetes from a public CSV mirror of the classic benchmark.
- Forest Covertype from scikit-learn's cached UCI Covertype dataset, with
  one-hot wilderness and soil indicators collapsed into categorical columns.
