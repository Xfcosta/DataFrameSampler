import sys

import click

from .anonymization import anonymize_columns_with_openai, assert_no_value_overlap
from .io import read_dataframe
from .llm import suggest_sampler_config_with_openai
from .sampler import ConcreteDataFrameSampler
from .utils import yaml_load
from .vectorizer import EMBEDDING_METHODS


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
    "--vectorizing_columns_dict_filename",
    "-f",
    type=click.Path(exists=True),
    help="Path to vectorizing_columns_dict serialized in YAML.",
)
@click.option(
    "--n_samples",
    "-n",
    type=click.IntRange(min=0, max_open=True, clamp=True),
    default=100,
    help="Number of samples to generate. If 0 then generate the same number of samples as there are in input.",
)
@click.option("--n_bins", type=click.IntRange(min=1, max_open=True, clamp=True), default=9, help="Number of bins.")
@click.option(
    "--n_neighbours",
    type=click.IntRange(min=1, max_open=True, clamp=True),
    default=5,
    help="Number of neighbours.",
)
@click.option("--sampled_columns", "-c", multiple=True, help="Selected columns to generate.")
@click.option(
    "--anonymize_columns",
    multiple=True,
    help="Sensitive columns to replace with OpenAI-generated surrogate values before fitting.",
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
    "--embedding_method",
    type=click.Choice(EMBEDDING_METHODS, case_sensitive=False),
    default="mds",
    show_default=True,
    help="Embedding method for categorical columns with vectorizing helper columns.",
)
@click.option(
    "--embedding_kwargs_filename",
    type=click.Path(exists=True),
    help="Path to embedding-method options serialized in YAML.",
)
@click.option(
    "--auto_config",
    "-A",
    is_flag=True,
    help="Use OpenAI to choose omitted vectorizing columns, sampled columns, embedding method, and KNN backend.",
)
@click.version_option("0.5.0", "--version", "-v")
@click.pass_context
def dataframe_sampler_main(
    ctx,
    input_filename,
    output_filename,
    input_model_filename,
    output_model_filename,
    vectorizing_columns_dict_filename,
    n_samples,
    n_bins,
    n_neighbours,
    sampled_columns,
    anonymize_columns,
    random_state,
    knn_backend,
    knn_backend_kwargs_filename,
    embedding_method,
    embedding_kwargs_filename,
    auto_config,
):
    """
    Generate a dataframe file similar to the input CSV or Parquet file.
    """
    if not input_filename and not input_model_filename:
        raise click.UsageError("Provide --input_filename to fit a model or --input_model_filename to load one.")

    sampled_columns = list(sampled_columns) if len(sampled_columns) else None
    anonymize_columns = list(anonymize_columns)
    vectorizing_columns_dict = (
        yaml_load(fname=vectorizing_columns_dict_filename) if vectorizing_columns_dict_filename else None
    )
    knn_backend_kwargs = yaml_load(fname=knn_backend_kwargs_filename) if knn_backend_kwargs_filename else None
    embedding_kwargs = yaml_load(fname=embedding_kwargs_filename) if embedding_kwargs_filename else None
    df = read_dataframe(input_filename) if input_filename else None

    if auto_config:
        if df is None:
            raise click.UsageError("--auto_config requires --input_filename so the dataframe can be profiled.")
        if input_model_filename:
            raise click.UsageError("--auto_config cannot be combined with --input_model_filename.")

        llm_config = suggest_sampler_config_with_openai(df)
        if vectorizing_columns_dict is None:
            vectorizing_columns_dict = llm_config["vectorizing_columns_dict"]
        if sampled_columns is None:
            sampled_columns = llm_config["sampled_columns"]
        if _is_default_parameter(ctx, "embedding_method"):
            embedding_method = llm_config["embedding_method"]
        if _is_default_parameter(ctx, "knn_backend"):
            knn_backend = llm_config["knn_backend"]
        click.echo("Auto configuration: %s" % llm_config["notes"], err=True)

    if input_model_filename:
        sampler = ConcreteDataFrameSampler().load(input_model_filename)
    else:
        sampler = ConcreteDataFrameSampler(
            n_bins=n_bins,
            n_neighbours=n_neighbours,
            vectorizing_columns_dict=vectorizing_columns_dict,
            sampled_columns=sampled_columns,
            random_state=random_state,
            knn_backend=knn_backend,
            knn_backend_kwargs=knn_backend_kwargs,
            embedding_method=embedding_method,
            embedding_kwargs=embedding_kwargs,
        )

    if input_filename:
        source_df = df
        if anonymize_columns:
            df, anonymization_report = anonymize_columns_with_openai(
                dataframe=df,
                source_dataframe=source_df,
                columns=anonymize_columns,
            )
            click.echo("Anonymized columns before fitting: %s" % ", ".join(anonymization_report["columns"]), err=True)
        sampler.fit(df)
        sampler.save(output_model_filename)
        if n_samples == 0:
            n_samples = len(df)
    elif n_samples == 0:
        raise click.UsageError("--n_samples=0 requires --input_filename so the input row count is known.")

    generated_df = sampler.sample_to_file(n_samples=n_samples, filename=output_filename)
    if input_filename and anonymize_columns:
        assert_no_value_overlap(source_df, generated_df, anonymize_columns)


def main():
    if len(sys.argv) == 1:
        dataframe_sampler_main.main(["--help"])
    else:
        dataframe_sampler_main()


def _is_default_parameter(ctx, parameter_name):
    return ctx.get_parameter_source(parameter_name) == click.core.ParameterSource.DEFAULT
