"""``shrimpy gui``: pymmcore-gui, plus standalone tool GUIs as subcommands.

``shrimpy gui`` (no subcommand) launches pymmcore-gui; ``shrimpy gui photom``
launches only the photomanipulation GUI, without pymmcore-gui or Micro-Manager.
pymmcore-gui's options belong to the group, so they go before any subcommand
name, and are rejected when a subcommand is given.
"""

from __future__ import annotations

import logging

from pathlib import Path

import click


@click.group(invoke_without_command=True)
@click.option(
    "--mm-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to Micro-Manager configuration file (.cfg)",
)
@click.option(
    "--unicore",
    is_flag=True,
    default=False,
    help="Use UniMMCore instead of standard CMMCorePlus",
)
@click.pass_context
def gui(ctx: click.Context, mm_config: Path | None, unicore: bool):
    """Launch the pymmcore-gui microscope control interface.

    With a subcommand, launch that GUI instead (see Commands below).

    Examples:

        shrimpy gui

        shrimpy gui --mm-config path/to/config.cfg

        shrimpy gui --mm-config path/to/MMConfig_demo_ReplayCamera.cfg --unicore

        shrimpy gui photom --demo
    """
    if ctx.invoked_subcommand is not None:
        if mm_config is not None or unicore:
            raise click.UsageError(
                "--mm-config/--unicore configure pymmcore-gui and do not apply to "
                f"`shrimpy gui {ctx.invoked_subcommand}`."
            )
        return

    try:
        from pymmcore_gui._app import create_mmgui
    except ImportError as exc:  # pragma: no cover - depends on how shrimpy was installed
        raise click.ClickException(
            "pymmcore-gui is not installed. It is an optional dependency; install it "
            "with `uv sync --extra gui` or `pip install 'shrimpy[gui]'`."
        ) from exc

    mmcore = None
    if unicore:
        from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore

        mmcore = UniMMCore()

    win = create_mmgui(
        mm_config=mm_config if mm_config else None,
        mmcore=mmcore,
        install_sentry=False,
        exec_app=False,
    )

    # Connect ReplayCamera to the Z stage if present
    if unicore and mm_config:
        from shrimpy.replay_camera import ReplayCamera

        core = win.mmcore
        cam_label = core.getCameraDevice()
        if cam_label and core.isPyDevice(cam_label):
            device = core._pydevices[cam_label]
            if isinstance(device, ReplayCamera):
                device.connect_z_stage(core)

    from pymmcore_gui._qt.QtWidgets import QApplication

    QApplication.instance().exec()


@gui.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "photom YAML config: lasers, mirrors, optional Arduinos and calibration "
        "settings (see config/photom/photom_VIS.yaml). Optional with --demo."
    ),
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help=(
        "Use simulated lasers, mirrors, Arduino and calibration camera; no hardware is opened."
    ),
)
@click.option(
    "--calibration",
    "calibration_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "Camera-to-mirror calibration YAML to load at startup, as the Load Calibration "
        "button does. Overrides the mirror's affine_matrix_path. Repeat for several "
        "mirrors, in config order."
    ),
)
def photom(config_path: Path | None, demo: bool, calibration_paths: tuple[Path, ...]):
    """Launch the photomanipulation (photom) GUI on its own, without pymmcore-gui.

    Steers the lasers with the galvo mirrors: drag the marker in the Laser Marker
    window to aim, right-click to fire, or draw ROIs and fill them with patterns.

    Examples:

        shrimpy gui photom --demo

        shrimpy gui photom --config config/photom/photom_VIS.yaml

        shrimpy gui photom --config photom_VIS.yaml --calibration affine_VIS.yml
    """
    if config_path is None and not demo:
        raise click.UsageError("--config is required unless --demo is given.")
    try:
        from copylot.assemblies.photom.gui.photom_gui import (
            DEMO_CONFIG,
            PhotomConfig,
            check_calibration_files,
            launch,
        )
    except ImportError as exc:  # pragma: no cover - depends on how shrimpy was installed
        raise click.ClickException(
            f"The photom GUI needs the `photom` extra ({exc}); install it with "
            "`uv sync --extra photom` or `pip install 'shrimpy[photom]'`."
        ) from exc

    _log_copylot_to_console()
    calibration_files = [path.resolve() for path in calibration_paths]
    try:
        config = PhotomConfig.from_yaml(config_path) if config_path is not None else None
        # Fail on a bad calibration before launch() connects any hardware.
        check_calibration_files(config or DEMO_CONFIG, calibration_files)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    raise SystemExit(launch(config, demo=demo, calibration_files=calibration_files))


def _log_copylot_to_console() -> None:
    """Show copylot's INFO messages (laser power, calibration) on the console."""
    copylot_logger = logging.getLogger("copylot")
    if not any(type(h) is logging.StreamHandler for h in copylot_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        )
        copylot_logger.addHandler(handler)
        copylot_logger.setLevel(logging.DEBUG)
