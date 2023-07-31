from typing import Callable
from natsort import natsorted
from pathlib import Path
import click
from iohub.ngff import open_ome_zarr, Plate


def _validate_and_process_paths(ctx: click.Context, opt: click.Option, value: str) -> None:
    # Sort and validate the input paths
    input_paths = [Path(path) for path in natsorted(value)]
    for path in input_paths:
        with open_ome_zarr(path, mode='r') as dataset:
            if isinstance(dataset, Plate):
                raise ValueError(
                    "Please supply a single position instead of an HCS plate. Likely fix: replace 'input.zarr' with 'input.zarr/0/0/0'"
                )
    return input_paths


def _convert_to_Path(ctx, param, value):
    if value:
        value = Path(value)
    else:
        value = None
    return value


def input_data_paths_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "input-paths",
            type=click.Path(exists=True),
            nargs=-1,
            callback=_validate_and_process_paths,
        )(f)

    return decorator


def deskew_param_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "deskew-param-path",
            type=click.Path(exists=True, file_okay=True),
            nargs=1,
            callback=_convert_to_Path,
        )(f)

    return decorator


def registration_param_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "registration_param_path",
            type=click.Path(exists=True, file_okay=True),
            nargs=1,
            callback=_convert_to_Path,
        )(f)

    return decorator


def output_dataset_options(default) -> Callable:
    click_options = [
        click.option(
            "--output-path",
            "-o",
            default=default,
            help="Path to output.zarr",
            type=click.Path(),  # Valid Path
            callback=_convert_to_Path,
        )
    ]
    # good place to add chunking, overwrite flag, etc

    def decorator(f: Callable) -> Callable:
        for opt in click_options:
            f = opt(f)
        return f

    return decorator
