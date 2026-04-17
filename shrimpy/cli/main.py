"""Command-line interface for shrimpy microscope control."""

from __future__ import annotations

import os

import click

# Set pymmcore-plus log properties before it gets imported.
# pymmcore-plus reads env vars at import time as a module-level constants,
# so these must be set before any pymmcore-plus import occurs.
if not os.environ.get("PYMM_LOG_LEVEL"):
    os.environ["PYMM_LOG_LEVEL"] = "INFO"
if not os.environ.get("PYMM_LOG_RICH"):
    os.environ["PYMM_LOG_RICH"] = "1"

from shrimpy.cli.acquire import acquire


@click.group()
@click.version_option(package_name="shrimpy")
def cli():
    """shrimpy - Custom acquisition engines for optical microscopes.

    High-throughput smart microscopy framework built on pymmcore-plus.
    """
    pass


# Register command groups
cli.add_command(acquire)


if __name__ == "__main__":
    cli()
