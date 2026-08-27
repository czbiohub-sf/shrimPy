"""Command-line interface for shrimpy microscope control."""

from __future__ import annotations

import os

import click

# Set pymmcore-plus log properties before it gets imported.
# pymmcore-plus reads env vars at import time as a module-level constants,
# so these must be set before any pymmcore-plus import occurs.
if not os.environ.get("PYMM_LOG_LEVEL"):
    os.environ["PYMM_LOG_LEVEL"] = "DEBUG"
if not os.environ.get("PYMM_LOG_RICH"):
    os.environ["PYMM_LOG_RICH"] = "1"
# PYMM_LOG_LEVEL only affects the Python-side "pymmcore-plus" logger. MMCore's
# own device-level trace ("Will start absolute move of ZStage to ...", "Waiting
# for device ...", serial traffic) needs enableDebugLog(True), which
# CMMCorePlus.__init__ gates on PYMM_DEBUG_LOG. Without it the CoreLog has no
# record of which stage was commanded when.
if not os.environ.get("PYMM_DEBUG_LOG"):
    os.environ["PYMM_DEBUG_LOG"] = "1"

from shrimpy.cli.acquire import acquire
from shrimpy.cli.gui import gui


@click.group()
@click.version_option(package_name="shrimpy")
def cli():
    """shrimpy - Custom acquisition engines for optical microscopes.

    High-throughput smart microscopy framework built on pymmcore-plus.
    """
    pass


# Register command groups
cli.add_command(acquire)
cli.add_command(gui)


if __name__ == "__main__":
    cli()
