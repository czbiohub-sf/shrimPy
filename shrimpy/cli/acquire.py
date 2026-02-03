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
    "--mmconfig",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to Micro-Manager configuration file",
)
@click.option(
    "--mda-sequence",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to MDA sequence YAML file",
)
@click.option(
    "--save-dir",
    type=click.Path(path_type=Path),
    default="./acquisition_data",
    help="Directory where acquisition data and logs will be saved",
)
@click.option(
    "--acquisition-name",
    default="mantis_acquisition",
    help="Name of the acquisition (used for log files)",
)
def mantis(
    mmconfig: Path,
    mda_sequence: Path,
    save_dir: Path,
    acquisition_name: str,
):
    """Run Mantis microscope acquisition.

    Example:

        shrimpy acquire mantis \\
            --mmconfig /path/to/mantis.cfg \\
            --mda-sequence /path/to/sequence.yaml \\
            --save-dir ./data \\
            --acquisition-name my_experiment
    """
    from shrimpy.mantis.mantis_engine import acquire as acquire_mantis

    acquire_mantis(
        mmconfig=str(mmconfig),
        mda_sequence=str(mda_sequence),
        save_dir=str(save_dir),
        acquisition_name=acquisition_name,
    )


@acquire.command()
def isim():
    """Run iSIM microscope acquisition (coming soon).

    Example:

        shrimpy acquire isim \\
            --mmconfig /path/to/isim.cfg \\
            --mda-sequence /path/to/sequence.yaml
    """
    click.echo(
        click.style("iSIM acquisition is not yet implemented. Coming soon!", fg="yellow")
    )
