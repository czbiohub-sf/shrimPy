"""Command-line interface for shrimpy microscope control."""

from __future__ import annotations

import click

from shrimpy.cli.acquire import acquire


@click.group()
@click.version_option(package_name='shrimpy')
def cli():
    """shrimpy - Custom acquisition engines for optical microscopes.

    High-throughput smart microscopy framework built on pymmcore-plus.
    """
    pass


# Register command groups
cli.add_command(acquire)


if __name__ == '__main__':
    cli()
