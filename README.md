# DataFrameSampler

DataFrameSampler 2.0 generates synthetic mixed-type tabular rows by learning a
fully numerical latent representation with supervised per-column categorical
embeddings, then applying a local mutual-neighbor displacement operator in that
latent space.

The package is an experimental tabular sampler. It is not a privacy mechanism:
the model is fit directly on the input dataframe and decoded values can match
values observed during training.

## Method

`DataFrameSampler` treats every non-continuous column as categorical. This
includes object/string columns, booleans, and binary numeric columns such as
`0/1`. Numeric columns are only numeric dtype columns with more than two
non-missing values.

During `fit`:

1. Non-binary numeric columns are median-imputed and standardized.
2. Every categorical column is one-hot encoded for context.
3. For each categorical column `C_j`, the sampler removes that column's own
   current block from the context and fits a
   `sklearn.neighbors.NeighborhoodComponentsAnalysis` projection to predict
   `C_j`.
4. The one-hot block for `C_j` is replaced by its learned latent block.
5. The process repeats for `n_iterations`.
6. A `RandomForestClassifier` decoder is fit for each categorical column from
   its final latent block back to the original categories.

The final latent matrix is:

```text
Z = [standardized numeric columns | NCA block for categorical column 1 | ...]
```

Generation uses the fitted latent matrix. For each synthetic row it picks an
anchor row `A`, a mutual neighbor `B`, and a mutual neighbor `C` of `B`, then
creates:

```text
A' = A + lambda_ * (C - B)
```

`A'` is decoded back to a dataframe row. Numeric columns are inverse-scaled.
Categorical columns are sampled from the decoder's predicted class
probabilities.

This neighbor transport step is a heuristic. It assumes the learned latent space
is locally linear enough for transferred displacements to stay near the
empirical manifold.

## Installation

```bash
pip install .
```

For test dependencies:

```bash
pip install ".[test]"
pytest
```

Parquet data input/output is optional:

```bash
pip install ".[parquet]"
```

Optional approximate-nearest-neighbor backends are installed only when needed:

```bash
pip install ".[pynndescent]"
pip install ".[hnswlib]"
pip install ".[annoy]"
pip install ".[ann]"
```

## Python API

```python
import pandas as pd
from dataframe_sampler import DataFrameSampler

df = pd.DataFrame(
    {
        "age": [21, 22, 35, 36, 48, 49, 63, 64],
        "city": ["Porto", "Porto", "London", "London", "Paris", "Paris", "Rome", "Rome"],
        "spend": [120, 130, 240, 260, 310, 330, 410, 430],
        "member": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

sampler = DataFrameSampler(
    n_components=2,
    n_iterations=2,
    n_neighbours=3,
    lambda_=1.0,
    knn_backend="sklearn",
    random_state=42,
)

sampler.fit(df)
latent = sampler.transform(df)
reconstructed = sampler.inverse_transform(latent, sample=False)
generated = sampler.generate(n_samples=5)
```

Public methods:

- `fit(X, y=None)`: fit on a non-empty pandas DataFrame. `y` is ignored and
  exists only for sklearn compatibility.
- `transform(X)`: transform a pandas DataFrame with the fitted columns into a
  NumPy latent matrix.
- `inverse_transform(Z, sample=True)`: decode a NumPy latent matrix back into a
  pandas DataFrame.
- `generate(n_samples=None)`: generate synthetic rows from the fitted latent
  matrix. If `n_samples=None`, the fitted row count is used.
- `save(filename)` and `load(filename)`: pickle model persistence helpers.

Constructor arguments:

- `n_components`: integer categorical latent width, or a dictionary keyed by
  categorical column. Defaults to `2`.
- `n_iterations`: number of NCA refinement rounds. Defaults to `2`.
- `n_neighbours`: nearest-neighbor count used for mutual-neighbor generation.
- `lambda_`: multiplier for the transferred neighbor displacement.
- `knn_backend`: one of `exact`, `sklearn`, `pynndescent`, `hnswlib`, or
  `annoy`.
- `knn_backend_kwargs`: optional backend-specific options.
- `random_state`: optional seed.
- `nca_kwargs`: optional keyword arguments for
  `NeighborhoodComponentsAnalysis`.
- `decoder_kwargs`: optional keyword arguments for `RandomForestClassifier`.

High-cardinality categorical columns are not dropped automatically.
DataFrameSampler warns and proceeds, assuming such columns have been
preprocessed deliberately.

## CLI

```text
Usage: dataframe-sampler [OPTIONS]

  Generate a dataframe file similar to the input CSV or Parquet file.

Options:
  -i, --input_filename PATH       Path to input CSV or Parquet file.
  -o, --output_filename PATH      Path to CSV or Parquet file to generate.
  -m, --input_model_filename PATH
                                  Path to fit model.
  -d, --output_model_filename PATH
                                  Path to model to save.
  -n, --n_samples INTEGER RANGE   Number of samples to generate. If 0 then
                                  generate the same number of samples as input.
  --n_components INTEGER RANGE    NCA latent dimensions per categorical column.
  --n_iterations INTEGER RANGE    Number of iterative categorical NCA
                                  refinement rounds.
  --n_neighbours INTEGER RANGE    Number of neighbours.
  --lambda FLOAT                  Latent neighbour displacement multiplier.
  --random_state INTEGER          Optional random seed for reproducible output.
  --knn_backend [exact|sklearn|pynndescent|hnswlib|annoy]
                                  KNN backend used for neighbour search.
  --knn_backend_kwargs_filename PATH
                                  Path to backend-specific KNN options
                                  serialized in YAML.
  -v, --version                   Show the version and exit.
  -h, --help                      Show this message and exit.
```

Example:

```bash
dataframe-sampler \
  --input_filename input.csv \
  --output_filename generated.csv \
  --n_samples 100 \
  --n_components 2 \
  --n_iterations 2 \
  --n_neighbours 5 \
  --knn_backend sklearn \
  --random_state 42
```

The legacy source-checkout script path still works:

```bash
python dataframe_sampler.py --help
```

## Repository Layout

- `src/dataframe_sampler/sampler.py`: `DataFrameSampler`, iterative NCA latent
  learning, probabilistic decoding, and neighbor transport generation.
- `src/dataframe_sampler/knn.py` and `src/dataframe_sampler/neighbours.py`:
  exact, sklearn, and optional approximate nearest-neighbor backends.
- `src/dataframe_sampler/cli.py`: CSV/Parquet file workflow.
- `experiments/`: reusable experiment, metric, baseline, plotting, and notebook
  helpers.
- `publication/`: paper draft and generated tables.

## Validation

```bash
python -m pytest tests/test_dataframe_sampler.py
python -m pytest tests/test_numeric_projection.py
python -m pytest tests/test_experiment_workflow.py tests/test_experiment_baselines.py tests/test_experiment_predictive.py
dataframe-sampler --help
python dataframe_sampler.py --help
```
