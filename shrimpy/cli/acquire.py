"""Acquisition commands for shrimpy CLI."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def acquire():
    """Run microscope acquisitions."""
    pass


@acquire.command()
@click.option(
    "--mm-config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to Micro-Manager configuration file",
)
@click.option(
    "--mda-config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to MDA sequence configuration YAML file",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./acquisition_data",
    help="Output directory where acquisition data and logs will be saved",
)
@click.option(
    "-n",
    "--name",
    default="mantis_acquisition",
    help="Name of the acquisition (used for log files and output)",
)
def mantis(
    mm_config: Path,
    mda_config: Path,
    output_dir: Path,
    name: str,
):
    """Run Mantis microscope acquisition.

    Example:

        shrimpy acquire mantis \\
            --mm-config /path/to/mantis.cfg \\
            --mda-config /path/to/sequence.yaml \\
            --output-dir ./data \\
            --name my_experiment
    """
    from shrimpy.mantis.mantis_engine import acquire as acquire_mantis

    acquire_mantis(
        mm_config=mm_config,
        mda_config=mda_config,
        output_dir=output_dir,
        name=name,
    )


@acquire.command()
def isim():
    """Run iSIM microscope acquisition (coming soon).

    Example:

        shrimpy acquire isim \\
            --mm-config /path/to/isim.cfg \\
            --mda-config /path/to/sequence.yaml
    """
    click.echo(
        click.style("iSIM acquisition is not yet implemented. Coming soon!", fg="yellow")
    )
