"""Reusable click options shared by the per-microscope acquisition commands."""

from __future__ import annotations

from pathlib import Path

import click

mm_config = click.option(
    "--mm-config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to Micro-Manager configuration file",
)

mda_config = click.option(
    "--mda-config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help=(
        "Path to the acquisition configuration YAML file (an MDASequence, with "
        "the microscope settings under 'metadata')"
    ),
)

output_dir = click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory where acquisition data and logs will be saved (must exist)",
)

name = click.option(
    "-n",
    "--name",
    default="acquisition",
    show_default=True,
    help="Name of the acquisition (used for log files and output)",
)
