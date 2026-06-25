"""Acquisition commands for shrimpy CLI."""

from __future__ import annotations

import logging

from pathlib import Path

import click

from shrimpy._logging import configure_logging

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
@click.option(
    "--napari-viewer",
    is_flag=True,
    default=False,
    help="Show acquired data live in a separate-process napari viewer.",
)
@click.option(
    "--napari-cache-mb",
    type=float,
    default=2048.0,
    show_default=True,
    help="Approximate RAM budget (MB) for the viewer's in-memory frame cache.",
)
def mantis(
    mm_config: Path,
    mda_config: Path,
    output_dir: Path,
    name: str,
    napari_viewer: bool,
    napari_cache_mb: float,
):
    """Run Mantis microscope acquisition.

    Example:

        shrimpy acquire mantis \\
            --mm-config /path/to/mantis.cfg \\
            --mda-config /path/to/sequence.yaml \\
            --output-dir ./data \\
            --name my_experiment
    """
    # Import before configure_logging: pymmcore-plus calls configure_logging() at module
    # level, which clears all handlers on the "pymmcore-plus" logger. Importing first
    # ensures that call happens before fileConfig() attaches the shrimpy file handler.
    from shrimpy.mantis.mantis_engine import MantisEngine
    from shrimpy.robust_cmmcore import RobustCMMCore

    # Configure logging
    config_file = Path(__file__).parent.parent.parent / "config" / "logging.ini"
    log_file = configure_logging(config_file, output_dir, name)
    if config_file.exists():
        logger.info(f"Logging configured for acquisition: {name}")
        logger.info(f"Log file: {log_file}")
    else:
        logger.warning(f"Logging config not found at {config_file}, using defaults")

    core = RobustCMMCore()
    logger.info(f"Loading Micro-Manager configuration from {mm_config}")
    core.loadSystemConfiguration(mm_config)
    engine = MantisEngine(core)

    feeder = None
    if napari_viewer:
        from shrimpy.viewer import ViewerFeeder

        feeder = ViewerFeeder(core, cache_mb=napari_cache_mb)
        feeder.start()

    try:
        engine.acquire(output_dir=output_dir, name=name, mda_config=mda_config)
    finally:
        if feeder is not None:
            # Keep the window open after acquisition so the user can inspect the
            # cached data, then release shared memory once they close it.
            logger.info("Acquisition done; viewer window left open until closed.")
            feeder.join()
            feeder.cleanup()


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
