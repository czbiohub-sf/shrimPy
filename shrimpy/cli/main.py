import click

from shrimpy.cli.apply_affine import apply_affine
from shrimpy.cli.deskew import deskew
from shrimpy.cli.estimate_affine import estimate_affine
from shrimpy.cli.estimate_bleaching import estimate_bleaching
from shrimpy.cli.estimate_deskew import estimate_deskew
from shrimpy.cli.estimate_stabilization import estimate_stabilization
from shrimpy.cli.optimize_affine import optimize_affine
from shrimpy.cli.run_acquisition import run_acquisition
from shrimpy.cli.stabilize import stabilize
from shrimpy.cli.update_scale_metadata import update_scale_metadata

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `shrimpy -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for shrimpy"""


cli.add_command(run_acquisition)
cli.add_command(estimate_bleaching)
cli.add_command(estimate_deskew)
cli.add_command(deskew)
cli.add_command(estimate_affine)
cli.add_command(optimize_affine)
cli.add_command(apply_affine)
cli.add_command(update_scale_metadata)
cli.add_command(estimate_stabilization)
cli.add_command(stabilize)
