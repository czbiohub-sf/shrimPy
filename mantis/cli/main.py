import click

from mantis.cli.characterize_psf import characterize_psf
from mantis.cli.concatenate import concatenate
from mantis.cli.deconvolve import deconvolve
from mantis.cli.deskew import deskew
from mantis.cli.estimate_bleaching import estimate_bleaching
from mantis.cli.estimate_deskew import estimate_deskew
from mantis.cli.estimate_psf import estimate_psf
from mantis.cli.estimate_registration import estimate_registration
from mantis.cli.estimate_stabilization import estimate_stabilization
from mantis.cli.estimate_stitch import estimate_stitch
from mantis.cli.optimize_registration import optimize_registration
from mantis.cli.register import register
from mantis.cli.run_acquisition import run_acquisition
from mantis.cli.stabilize import stabilize
from mantis.cli.stitch import stitch
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
cli.add_command(register)
cli.add_command(estimate_stitch)
cli.add_command(stitch)
cli.add_command(update_scale_metadata)
cli.add_command(concatenate)
cli.add_command(estimate_stabilization)
cli.add_command(stabilize)
cli.add_command(estimate_psf)
cli.add_command(deconvolve)
cli.add_command(characterize_psf)
