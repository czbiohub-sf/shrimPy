from pathlib import Path

import click
import yaml

from mantis import __mm_version__
from mantis.cli.parsing import config_filepath, output_dirpath

default_mm_app_path = 'C:\\Program Files\\Micro-Manager-2.0_{}_{}_{}_2'.format(
    *__mm_version__.split('-')
)
default_mm_config_filepath = 'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'


@click.command()
@config_filepath()
@output_dirpath()
@click.option(
    "--mm-app-path",
    default=default_mm_app_path,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    show_default=True,
    help='''Path to Micro-manager installation directory
      which will run the light-sheet acquisition''',
)
@click.option(
    "--mm-config-filepath",
    default=default_mm_config_filepath,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help='''Path to Micro-manager config file
      which will run the light-sheet acquisition''',
)
def run_acquisition(
    config_filepath,
    output_dirpath,
    mm_app_path,
    mm_config_filepath,
):
    """Acquire mantis data as specified by a configuration file.

    >> mantis run-acquisition -c path/to/config.yaml -o ./YYYY_MM_DD_experiment_name/acquisition_name
    """

    # These imports are placed here to accelerate CLI help calls
    from mantis.acquisition.acq_engine import MantisAcquisition

    # isort: off
    from mantis.acquisition.AcquisitionSettings import (
        TimeSettings,
        PositionSettings,
        ChannelSettings,
        SliceSettings,
        MicroscopeSettings,
        AutoexposureSettings,
    )

    # isort: on

    demo_run = True if 'demo' in mm_config_filepath else False

    output_dirpath = Path(output_dirpath)
    acq_directory, acq_name = output_dirpath.parent, output_dirpath.name

    with open(config_filepath) as file:
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
    ls_autoexposure_settings = AutoexposureSettings(
        **raw_settings.get('ls_autoexposure_settings')
    )

    with MantisAcquisition(
        acquisition_directory=acq_directory,
        acquisition_name=acq_name,
        mm_app_path=mm_app_path,
        mm_config_file=mm_config_filepath,
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
        acq.ls_acq.autoexposure_settings = ls_autoexposure_settings

        acq.setup()
        acq.acquire()


if __name__ == '__main__':
    run_acquisition()
