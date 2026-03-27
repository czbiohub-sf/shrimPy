"""GUI command for shrimpy CLI — launches pymmcore-gui."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
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
def gui(mm_config: Path | None, unicore: bool):
    """Launch the pymmcore-gui microscope control interface.

    Examples:

        shrimpy gui

        shrimpy gui --mm-config path/to/config.cfg

        shrimpy gui --mm-config path/to/MMConfig_demo_ReplayCamera.cfg --unicore
    """
    from pymmcore_gui._app import create_mmgui

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
        from shrimpy.mantis.replay_camera import ReplayCamera

        core = win.mmcore
        cam_label = core.getCameraDevice()
        if cam_label and core.isPyDevice(cam_label):
            device = core._pydevices[cam_label]
            if isinstance(device, ReplayCamera):
                device.connect_z_stage(core)
                device.connect_to_mda(core)

    from pymmcore_gui._qt.QtWidgets import QApplication

    QApplication.instance().exec()
