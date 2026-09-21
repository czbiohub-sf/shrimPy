"""Acquisition commands for shrimpy CLI."""

from __future__ import annotations

import logging

from pathlib import Path

import click

from shrimpy.cli.options import mda_config, mm_config, name, output_dir
from shrimpy.logging import configure_logging

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
    help="Show the acquisition live in a separate-process napari viewer.",
)
def mantis(
    mm_config: Path,
    mda_config: Path,
    output_dir: Path,
    name: str,
    unicore: bool,
    napari_viewer: bool,
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
    log_file = configure_logging(output_dir, name)
    logger.info(f"Logging configured for acquisition: {name}")
    logger.info(f"Log file: {log_file}")

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

    viewer = None
    if napari_viewer:
        from shrimpy.viewer import LiveViewer

        # Mantis is an oblique-plane light-sheet microscope, so the deskew widget is
        # shown by default (on, toggleable). Other microscopes pass deskew=False.
        viewer = LiveViewer(engine, deskew=True)
        viewer.start()

    try:
        engine.acquire(output_dir=output_dir, name=name, mda_config=mda_config)
    finally:
        if viewer is not None:
            # Keep the window open after the acquisition: the store is complete on
            # disk, so the user can go on browsing all of it.
            logger.info("Acquisition done; viewer window left open until closed.")
            viewer.join()
            viewer.cleanup()


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
    from shrimpy.engines.dragonfly_engine import DragonflyEngine
    from shrimpy.robust_cmmcore import RobustCMMCore

    log_file = configure_logging(output_dir, name)
    logger.info(f"Logging configured for acquisition: {name}")
    logger.info(f"Log file: {log_file}")

    core = RobustCMMCore()

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
