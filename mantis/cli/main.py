import click

from mantis.cli.apply_affine import apply_affine
from mantis.cli.deskew import deskew
from mantis.cli.estimate_affine import estimate_affine
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.optimize_affine import optimize_affine
from mantis.cli.run_acquisition import run_acquisition
from mantis.cli.update_scale_metadata import update_scale_metadata
from mantis.cli.psf_from_beads import psf_from_beads
from mantis.cli.deconvolve import deconvolve

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
cli.add_command(estimate_affine)
cli.add_command(optimize_affine)
cli.add_command(apply_affine)
cli.add_command(update_scale_metadata)
cli.add_command(psf_from_beads)
cli.add_command(deconvolve)
