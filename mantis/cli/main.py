import click

from mantis.cli.deskew import deskew
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.run_acquisition import run_acquisition


@click.group()
def cli():
    """command-line tools for mantis"""


cli.add_command(deskew)
cli.add_command(estimate_bleaching)
cli.add_command(estimate_deskew)
cli.add_command(run_acquisition)
