
import click
import warnings
import glob
from typing import Callable, Sequence


def input_data_path_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "input-data-path",
            type=click.Path(exists=True),
            nargs=-1,
        )(f)

    return decorator


def deskew_param_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "deskew-param-path", type=click.Path(exists=True, file_okay=True), nargs=1
        )(f)
    return decorator


def output_dataset_options(default) -> Callable:
    click_options = [
        click.option(
            "--output-path",
            "-o",
            default=default,
            help="Path to output.zarr",
        )
    ]
    # good place to add chunking, overwrite flag, etc

    def decorator(f: Callable) -> Callable:
        for opt in click_options:
            f = opt(f)
        return f

    return decorator
