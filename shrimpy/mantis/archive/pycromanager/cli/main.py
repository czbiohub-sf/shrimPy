import click

from shrimpy.cli.run_acquisition import run_acquisition
from shrimpy.cli.stir_plate import stir_plate_cli

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `mantis -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for mantis"""


cli.add_command(run_acquisition)
cli.add_command(stir_plate_cli)
