from typing import Callable

import click


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "input-position-dirpaths",
            type=click.Path(exists=True, file_okay=False, dir_okay=True),
            nargs=-1,
        )(f)

    return decorator


def config_filepath() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-filepath",
            "-c",
            required=True,
            type=click.Path(exists=True, file_okay=True, dir_okay=False),
            help="Path to configuration file",
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
