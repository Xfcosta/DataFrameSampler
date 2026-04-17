import sys

import click

from .io import read_dataframe
from .sampler import DataFrameSampler
from .utils import yaml_load


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--input_filename", "-i", type=click.Path(exists=True), help="Path to input CSV or Parquet file.")
@click.option(
    "--output_filename",
    "-o",
    type=click.Path(),
    default="data.csv",
    help='Path to CSV or Parquet file to generate. By default a file called "data.csv" will be generated.',
)
@click.option("--input_model_filename", "-m", type=click.Path(exists=True), help="Path to fit model.")
@click.option(
    "--output_model_filename",
    "-d",
    type=click.Path(),
    default="dataframe_sampler_model.obj",
    help="Path to model to save.",
)
@click.option(
    "--n_samples",
    "-n",
    type=click.IntRange(min=0, max_open=True, clamp=True),
    default=100,
    help="Number of samples to generate. If 0 then generate the same number of samples as there are in input.",
)
@click.option(
    "--n_components",
    type=click.IntRange(min=1, max_open=True, clamp=True),
    default=2,
    show_default=True,
    help="NCA latent dimensions per categorical column.",
)
@click.option(
    "--n_iterations",
    type=click.IntRange(min=0, max_open=True, clamp=True),
    default=1,
    show_default=True,
    help="Number of iterative categorical NCA refinement rounds. Use 0 to keep one-hot categorical blocks.",
)
@click.option(
    "--n_neighbours",
    type=click.IntRange(min=1, max_open=True, clamp=True),
    default=5,
    help="Number of neighbours.",
)
@click.option(
    "--lambda",
    "lambda_",
    type=float,
    default=1.0,
    show_default=True,
    help="Latent neighbour displacement multiplier.",
)
@click.option("--random_state", type=int, help="Optional random seed for reproducible output.")
@click.option(
    "--knn_backend",
    type=click.Choice(["exact", "sklearn", "pynndescent", "hnswlib", "annoy"], case_sensitive=False),
    default="exact",
    show_default=True,
    help="KNN backend used for neighbour search.",
)
@click.option(
    "--knn_backend_kwargs_filename",
    type=click.Path(exists=True),
    help="Path to backend-specific KNN options serialized in YAML.",
)
@click.option(
    "--calibrate_decoders/--no_calibrate_decoders",
    default=False,
    show_default=True,
    help="Calibrate categorical decoder probabilities when feasible.",
)
@click.option(
    "--enforce_min_max_constraints/--no_enforce_min_max_constraints",
    default=True,
    show_default=True,
    help="Reject and retry latent candidates outside fitted columnwise min/max ranges.",
)
@click.option(
    "--enforce_numeric_std_constraints/--no_enforce_numeric_std_constraints",
    default=True,
    show_default=True,
    help="Reject and retry latent candidates with improbable fitted numeric-column z-scores.",
)
@click.option(
    "--numeric_std_threshold",
    type=click.FloatRange(min=0, min_open=True),
    default=3.0,
    show_default=True,
    help="Maximum absolute fitted numeric-column z-score before retrying a generated candidate.",
)
@click.option(
    "--max_constraint_retries",
    type=click.IntRange(min=0, max_open=True, clamp=True),
    default=3,
    show_default=True,
    help="Retries per generated row before accepting an out-of-range latent candidate.",
)
@click.version_option("2.0.0", "--version", "-v")
def dataframe_sampler_main(
    input_filename,
    output_filename,
    input_model_filename,
    output_model_filename,
    n_samples,
    n_components,
    n_iterations,
    n_neighbours,
    lambda_,
    random_state,
    knn_backend,
    knn_backend_kwargs_filename,
    calibrate_decoders,
    enforce_min_max_constraints,
    enforce_numeric_std_constraints,
    numeric_std_threshold,
    max_constraint_retries,
):
    """
    Generate a dataframe file similar to the input CSV or Parquet file.
    """
    if not input_filename and not input_model_filename:
        raise click.UsageError("Provide --input_filename to fit a model or --input_model_filename to load one.")

    knn_backend_kwargs = yaml_load(fname=knn_backend_kwargs_filename) if knn_backend_kwargs_filename else None
    df = read_dataframe(input_filename) if input_filename else None

    if input_model_filename:
        sampler = DataFrameSampler().load(input_model_filename)
    else:
        sampler = DataFrameSampler(
            n_components=n_components,
            n_iterations=n_iterations,
            n_neighbours=n_neighbours,
            lambda_=lambda_,
            random_state=random_state,
            knn_backend=knn_backend,
            knn_backend_kwargs=knn_backend_kwargs,
            calibrate_decoders=calibrate_decoders,
            enforce_min_max_constraints=enforce_min_max_constraints,
            enforce_numeric_std_constraints=enforce_numeric_std_constraints,
            numeric_std_threshold=numeric_std_threshold,
            max_constraint_retries=max_constraint_retries,
        )

    if input_filename:
        sampler.fit(df)
        sampler.save(output_model_filename)
        if n_samples == 0:
            n_samples = len(df)
    elif n_samples == 0:
        raise click.UsageError("--n_samples=0 requires --input_filename so the input row count is known.")

    sampler.generate_to_file(n_samples=n_samples, filename=output_filename)


def main():
    if len(sys.argv) == 1:
        dataframe_sampler_main.main(["--help"])
    else:
        dataframe_sampler_main()
