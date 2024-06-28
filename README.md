# DataFrameSampler


```
Usage: dataframe_sampler.py [OPTIONS]

  The program dataframe_sampler can generate a CSV file similar to the one
  given in input.

Options:
  -i, --input_filename PATH       Path to input CSV file.
  -o, --output_filename PATH      Path to CSV file to generate. By default a
                                  file called "data.csv" will be generated.
  -m, --input_model_filename PATH
                                  Path to fit model.
  -d, --output_model_filename PATH
                                  Path to model to save.
  -f, --vectorizing_columns_dict_filename PATH
                                  Path to vectorizing_columns_dict serialized
                                  in YAML.
  -n, --n_samples INTEGER RANGE   Number of samples to generate. If 0 then
                                  generate the same number of samples as there
                                  are in input. Default is 100.  [x>=0]
  --n_bins INTEGER RANGE          Number of bins.  [x>=2]
  --n_neighbours INTEGER RANGE    Number of neighbours.  [x>=1]
  -c, --sampled_columns TEXT      Selected columns to generate.
  -v, --version                   Show the version and exit.
  -h, --help                      Show this message and exit.
```

See accompanying IPython notebook [here](https://github.com/Xfcosta/DataFrameSampler/blob/main/dataframe_sampler_notebook.ipynb)

### Installation
```
pip install -r requirements.txt

```
