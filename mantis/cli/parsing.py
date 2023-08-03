from mantis.cli.option_eat_all import OptionEatAll
from typing import Callable

import click


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option("--input-position-dirpaths", "-i", cls=OptionEatAll, type=tuple)(f)

    return decorator


def config_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-filepath",
            "-c",
            required=True,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help="Path to YAML configuration file",
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
