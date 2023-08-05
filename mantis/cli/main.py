import click

from mantis.cli.deskew import deskew
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.run_acquisition import run_acquisition
from mantis.cli.estimate_registration import manual_registration
from mantis.cli.register import register

CONTEXT = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT)
def cli():
    """command-line tools for mantis"""


cli.add_command(deskew)
cli.add_command(estimate_bleaching)
cli.add_command(estimate_deskew)
cli.add_command(run_acquisition)
cli.add_command(manual_registration)
cli.add_command(register)
