from pathlib import Path
from typing import Callable

import click

from iohub.ngff import Plate, open_ome_zarr
from natsort import natsorted

from mantis.cli.option_eat_all import OptionEatAll


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


def _str_to_path(ctx: click.Context, opt: click.Option, value: str) -> Path:
    return Path(value)


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--input-position-dirpaths",
            "-i",
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
        )(f)

    return decorator


def source_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--source-position-dirpaths",
            "-s",
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
        )(f)

    return decorator


def target_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--target-position-dirpaths",
            "-t",
            cls=OptionEatAll,
            type=tuple,
            callback=_validate_and_process_paths,
        )(f)

    return decorator

def config_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-filepath",
            "-c",
            required=True,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            callback=_str_to_path,
            help="Path to YAML configuration file.",
        )(f)

    return decorator


def output_dirpath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-dirpath",
            "-o",
            required=True,
            type=click.Path(exists=False, file_okay=False, dir_okay=True),
            help="Path to output directory",
            callback=_str_to_path,
        )(f)

    return decorator


def output_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-filepath",
            "-o",
            required=True,
            type=click.Path(exists=False, file_okay=True, dir_okay=False),
            help="Path to output file",
        )(f)

    return decorator
