import click

from mantis.cli.register import apply_affine
from mantis.cli.deskew import deskew
from mantis.cli.estimate_registration import estimate_registration
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.estimate_stabilization import estimate_stabilization
from mantis.cli.optimize_registration import optimize_registration
from mantis.cli.run_acquisition import run_acquisition
from mantis.cli.stabilize import stabilize
from mantis.cli.update_scale_metadata import update_scale_metadata

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
cli.add_command(estimate_registration)
cli.add_command(optimize_registration)
cli.add_command(apply_affine)
cli.add_command(update_scale_metadata)
cli.add_command(estimate_stabilization)
cli.add_command(stabilize)
