import click

from mantis.cli.apply_affine import apply_affine
from mantis.cli.deskew import deskew
from mantis.cli.estimate_affine import estimate_phase_to_fluor_affine
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.run_acquisition import run_acquisition

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `mantis -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for mantis"""


cli.add_command(run_acquisition)
cli.add_command(estimate_bleaching)
cli.add_command(estimate_deskew)
cli.add_command(deskew)
cli.add_command(estimate_phase_to_fluor_affine)
cli.add_command(apply_affine)
