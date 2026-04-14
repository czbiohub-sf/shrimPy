"""Command-line interface for shrimpy microscope control."""

from __future__ import annotations

import os

import click

# Set pymmcore-plus log level before it gets imported.
# pymmcore-plus reads PYMM_LOG_LEVEL at import time as a module-level constant,
# so this must be set before any pymmcore-plus import occurs.
if not os.environ.get("PYMM_LOG_LEVEL"):
    os.environ["PYMM_LOG_LEVEL"] = "INFO"

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
