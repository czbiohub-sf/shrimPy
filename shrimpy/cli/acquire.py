"""Acquisition commands for shrimpy CLI."""

from __future__ import annotations

import logging

from pathlib import Path

import click

from shrimpy._logging import log_conda_environment, setup_logging

logger = logging.getLogger(__name__)


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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory where acquisition data and logs will be saved (must exist)",
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
    # Configure logging
    config_file = Path(__file__).parent.parent.parent / "config" / "logging.ini"
    log_file = setup_logging(config_file, output_dir, name)
    if config_file.exists():
        logger.info(f"Logging configured for acquisition: {name}")
        logger.info(f"Log file: {log_file}")
    else:
        logger.warning(f"Logging config not found at {config_file}, using defaults")

    # Log conda environment
    out, err = log_conda_environment(log_file.parent)
    if err is None:
        logger.debug(out)
    else:
        logger.error(err)

    # Initialize core and engine, then run acquisition
    from shrimpy.mantis.mantis_engine import MantisEngine

    core = MantisEngine.initialize_core(mm_config)
    engine = MantisEngine(core)
    engine.acquire(output_dir=output_dir, name=name, mda_config=mda_config)


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
