"""Acquisition commands for shrimpy CLI."""

from __future__ import annotations

import logging

from pathlib import Path

import click

from shrimpy._logging import configure_logging
from shrimpy.cli.options import mda_config, mm_config, name, output_dir

logger = logging.getLogger(__name__)


@click.group()
def acquire():
    """Run microscope acquisitions."""
    pass


@acquire.command()
@mm_config
@mda_config
@output_dir
@name
@click.option(
    "--unicore",
    is_flag=True,
    default=False,
    help="Use UniMMCore instead of standard CMMCorePlus",
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
    default=8192.0,
    show_default=True,
    help="Approximate RAM budget (MB) for the viewer's in-memory frame cache.",
)
def mantis(
    mm_config: Path,
    mda_config: Path,
    output_dir: Path,
    name: str,
    unicore: bool,
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
    from shrimpy.engines.mantis_engine import MantisEngine

    # Configure logging
    config_file = Path(__file__).parent.parent.parent / "config" / "logging.ini"
    log_file = configure_logging(config_file, output_dir, name)
    if config_file.exists():
        logger.info(f"Logging configured for acquisition: {name}")
        logger.info(f"Log file: {log_file}")
    else:
        logger.warning(f"Logging config not found at {config_file}, using defaults")

    if unicore:
        from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore

        core = UniMMCore()
    else:
        from shrimpy.robust_cmmcore import RobustCMMCore

        core = RobustCMMCore()

    # Pre-import torch before MM loads its DLLs to avoid DLL conflict on Windows
    # (shm.dll fails with WinError 127 if MM CUDA DLLs are loaded first)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    logger.info(f"Loading Micro-Manager configuration from {mm_config}")
    core.loadSystemConfiguration(mm_config)

    if unicore:
        from shrimpy.replay_camera import ReplayCamera

        cam_label = core.getCameraDevice()
        if cam_label and core.isPyDevice(cam_label):
            device = core._pydevices[cam_label]
            if isinstance(device, ReplayCamera):
                device.connect_z_stage(core)
                device.connect_to_mda(core)
    engine = MantisEngine(core)

    feeder = None
    if napari_viewer:
        from shrimpy.viewer import ViewerFeeder

        # Mantis is an oblique-plane light-sheet microscope, so the deskew widget is
        # shown by default (on, toggleable). Other microscopes pass deskew=False.
        feeder = ViewerFeeder(core, cache_mb=napari_cache_mb, deskew=True)
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
@mm_config
@mda_config
@output_dir
@name
def dragonfly(mm_config: Path, mda_config: Path, output_dir: Path, name: str):
    """Run Dragonfly microscope acquisition.

    Example:

        shrimpy acquire dragonfly \\
            --mm-config /path/to/dragonfly.cfg \\
            --mda-config /path/to/sequence.yaml \\
            --output-dir ./data \\
            --name my_experiment
    """
    # Import before configure_logging: pymmcore-plus calls configure_logging() at module
    # level, which clears all handlers on the "pymmcore-plus" logger. Importing first
    # ensures that call happens before fileConfig() attaches the shrimpy file handler.
    from pymmcore_plus import CMMCorePlus

    from shrimpy.engines.dragonfly_engine import DragonflyEngine

    config_file = Path(__file__).parent.parent.parent / "config" / "logging.ini"
    log_file = configure_logging(config_file, output_dir, name)
    if config_file.exists():
        logger.info(f"Logging configured for acquisition: {name}")
        logger.info(f"Log file: {log_file}")
    else:
        logger.warning(f"Logging config not found at {config_file}, using defaults")

    core = CMMCorePlus()

    logger.info(f"Loading Micro-Manager configuration from {mm_config}")
    core.loadSystemConfiguration(mm_config)

    engine = DragonflyEngine(core)
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
