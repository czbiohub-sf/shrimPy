import click
import yaml

from mantis.cli.parsing import output_dirpath
from mantis.acquisition.acq_engine import MantisAcquisition

# isort: off
from mantis.acquisition.AcquisitionSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
)

# isort: on


@click.command()
@output_dirpath()
@click.option(
    '--name',
    required=True,
    help='Name of the acquisition',
)
@click.option(
    '--settings',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='YAML file containing the acquisition settings',
)
@click.option(
    '--mm-app-path',
    default='C:\\Program Files\\Micro-Manager-nightly',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    show_default=True,
    help='''Path to Micro-manager installation directory
      which will run the light-sheet acquisition''',
)
@click.option(
    '--mm-config-file',
    default='C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help='''Path to Micro-manager config file
      which will run the light-sheet acquisition''',
)
def run_acquisition(
    data_dirpath,
    name,
    settings,
    mm_app_path,
    mm_config_file,
):
    """Acquire data using a settings file."""

    demo_run = True if 'demo' in mm_config_file else False

    with open(settings) as file:
        raw_settings = yaml.safe_load(file)

    # Load and validate YAML settings
    time_settings = TimeSettings(**raw_settings.get('time_settings'))
    position_settings = PositionSettings()
    lf_channel_settings = ChannelSettings(**raw_settings.get('lf_channel_settings'))
    lf_slice_settings = SliceSettings(**raw_settings.get('lf_slice_settings'))
    lf_microscope_settings = MicroscopeSettings(**raw_settings.get('lf_microscope_settings'))
    ls_channel_settings = ChannelSettings(**raw_settings.get('ls_channel_settings'))
    ls_slice_settings = SliceSettings(**raw_settings.get('ls_slice_settings'))
    ls_microscope_settings = MicroscopeSettings(**raw_settings.get('ls_microscope_settings'))

    with MantisAcquisition(
        acquisition_directory=data_dirpath,
        acquisition_name=name,
        mm_app_path=mm_app_path,
        mm_config_file=mm_config_file,
        demo_run=demo_run,
        verbose=False,
    ) as acq:
        acq.time_settings = time_settings
        acq.position_settings = position_settings
        acq.lf_acq.channel_settings = lf_channel_settings
        acq.lf_acq.slice_settings = lf_slice_settings
        acq.lf_acq.microscope_settings = lf_microscope_settings
        acq.ls_acq.channel_settings = ls_channel_settings
        acq.ls_acq.slice_settings = ls_slice_settings
        acq.ls_acq.microscope_settings = ls_microscope_settings

        acq.setup()
        acq.acquire()


if __name__ == '__main__':
    run_acquisition()
