# /// script
# requires-python = "==3.10"
# dependencies = [
#   "copylot @ git+https://github.com/czbiohub-sf/coPylot",
#   "ndtiff==2.2.1",
#   "nidaqmx",
#   "numpy<2",
#   "pycromanager==0.28.1",
#   "pydantic>=2.0.0",
#   "pylablib==1.4.1",
#   "tifffile",
#   "waveorder @ git+https://github.com/mehta-lab/waveorder@focus-statistics-api",
#   "viscy"
# ]
# ///

import logging
import os
import re
import time

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Union

import copylot
import nidaqmx
import numpy as np
import tifffile

from nidaqmx.constants import Slope
from pycromanager import Acquisition, Core, Studio, multi_d_acquisition_events, start_headless
from waveorder.focus import focus_from_transverse_band

from mantis import get_console_formatter
from mantis.acquisition import microscope_operations
from mantis.acquisition.autoexposure import load_manual_illumination_settings
from mantis.acquisition.autotracker import Autotracker
from mantis.acquisition.hook_functions import globals
from mantis.acquisition.logger import configure_debug_logger, log_conda_environment

# isort: off
from mantis.acquisition.AcquisitionSettings import (
    TimeSettings,
    PositionSettings,
    ChannelSettings,
    SliceSettings,
    MicroscopeSettings,
    AutoexposureSettings,
    AutotrackerSettings,
)
from mantis.acquisition.hook_functions.pre_hardware_hook_functions import (
    log_preparing_acquisition,
    lf_pre_hardware_hook_function,
    ls_pre_hardware_hook_function,
)
from mantis.acquisition.hook_functions.post_hardware_hook_functions import (
    log_acquisition_start,
    update_ls_hardware,
    update_laser_power,
)
from mantis.acquisition.hook_functions.post_camera_hook_functions import (
    start_daq_counters,
)
from mantis.acquisition.hook_functions.image_saved_hook_functions import (
    check_lf_acq_finished,
    check_ls_acq_finished,
)

# isort: on

# Define constants
LF_ZMQ_PORT = 4827
LS_ZMQ_PORT = 5827  # we need to space out port numbers a bit
LS_POST_READOUT_DELAY = 0.05  # delay before acquiring next frame, in ms
MCL_STEP_TIME = 1.5  # in ms
LC_CHANGE_TIME = 20  # in ms
LS_CHANGE_TIME = 200  # time needed to change LS filter wheel, in ms
LS_KIM101_SN = 74000291
LF_KIM101_SN = 74000565
KIM101_BACKLASH = 0  # backlash correction distance, in steps
VORTRAN_488_COM_PORT = 'COM6'
VORTRAN_561_COM_PORT = 'COM13'
VORTRAN_639_COM_PORT = 'COM12'
LF_ACQ_LABEL = 'labelfree'
LS_ACQ_LABEL = 'lightsheet'

NA_DETECTION = 1.35
LS_PIXEL_SIZE = 6.5 / (40 * 1.4)  # in um

logger = logging.getLogger(__name__)


class BaseChannelSliceAcquisition(object):
    """
    Base class which handles setup of the label-free or light-sheet acquisition

    Parameters
    ----------
    enabled : bool, optional
        Flag if acquisition should be enabled, by default True
    mm_app_path : str, optional
        Path to Micro-manager which will be launched in headless mode, by default None
    mm_config_file : str, optional
        Path to config file for the headless acquisition, by default None
    zmq_port : int, optional
        ZeroMQ port of the acquisition, by default 4827
    core_log_path : str, optional
        Path where the headless acquisition core logs will be saved, by default ''
    """

    def __init__(
        self,
        enabled: bool = True,
        mm_app_path: str = None,
        mm_config_file: str = None,
        zmq_port: int = 4827,
        core_log_path: str = '',
    ):
        self.enabled = enabled
        self._channel_settings = ChannelSettings()
        self._slice_settings = SliceSettings()
        self._microscope_settings = MicroscopeSettings()
        self._autoexposure_settings = AutoexposureSettings()
        self._autotracker_settings = None
        self._z0 = None
        self.headless = False if mm_app_path is None else True
        self.type = 'light-sheet' if self.headless else 'label-free'
        self.mmc = None
        self.mmStudio = None
        self.o3_stage = None

        logger.debug(f'Initializing {self.type} acquisition engine')
        if enabled:
            if self.headless:
                java_loc = None
                if "JAVA_HOME" in os.environ:
                    java_loc = os.environ["JAVA_HOME"]

                logger.debug(f'Starting headless Micro-Manager instance on port {zmq_port}')
                logger.debug(f'Core logs will be saved at: {core_log_path}')
                start_headless(
                    mm_app_path,
                    mm_config_file,
                    java_loc=java_loc,
                    port=zmq_port,
                    core_log_path=core_log_path,
                    buffer_size_mb=2048,
                )

            logger.debug(f'Connecting to Micro-Manager on port {zmq_port}')

            self.mmc = Core(port=zmq_port)

            # headless MM instance doesn't have a studio object
            if not self.headless:
                self.mmStudio = Studio(port=zmq_port)

            logger.debug('Successfully connected to Micro-Manager')
            logger.debug(f'{self.mmc.get_version_info()}')  # MMCore Version

            if not self.headless:
                logger.debug(f'MM Studio version: {self.mmStudio.compat().get_version()}')
        else:
            logger.info(f'{self.type.capitalize()} acquisition is not enabled')

    def _check_num_sequenced_events(self):
        num_sequenced_events = 1
        if self.slice_settings.use_sequencing:
            num_sequenced_events *= self.slice_settings.num_slices
        if self.channel_settings.use_sequencing:
            num_sequenced_events *= self.channel_settings.num_channels
        if num_sequenced_events > 1200:
            raise ValueError(
                f'The number of sequenced events: {num_sequenced_events} exceeds the maximum allowed limit of 1200. '
                'Please reduce the number of slices and channels or disable channel sequencing. '
                'This limitation can be overcome by increasing the length of hardware sequences supported by the '
                'TriggerScope firmware (see NR_DAC_STATES and NR_DO_STATES).'
            )

    @property
    def channel_settings(self):
        return self._channel_settings

    @property
    def slice_settings(self):
        return self._slice_settings

    @property
    def microscope_settings(self):
        return self._microscope_settings

    @property
    def autoexposure_settings(self):
        return self._autoexposure_settings

    @property
    def autotracker_settings(self):
        return self._autotracker_settings

    @channel_settings.setter
    def channel_settings(self, settings: ChannelSettings):
        if settings is None:
            return
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._channel_settings = settings
        self._check_num_sequenced_events()

    @slice_settings.setter
    def slice_settings(self, settings: SliceSettings):
        if settings is None:
            return
        settings_dict = {key: val for key, val in asdict(settings).items() if key != 'z_range'}
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {settings_dict}'
        )
        self._slice_settings = settings
        self._check_num_sequenced_events()

    @microscope_settings.setter
    def microscope_settings(self, settings: MicroscopeSettings):
        if settings is None:
            return
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._microscope_settings = settings

    @autoexposure_settings.setter
    def autoexposure_settings(self, settings: AutoexposureSettings):
        if settings is None:
            return
        logger.debug(
            f"{self.type.capitalize()} acquisition will have the following settings:{asdict(settings)}"
        )
        self._autoexposure_settings = settings

    @autotracker_settings.setter
    def autotracker_settings(self, settings: AutotrackerSettings):
        if settings is None:
            logger.debug('Autotracker settings are not provided')
        else:
            logger.debug(
                f"{self.type.capitalize()} acquisition will have the following settings:{asdict(settings)}"
            )
        self._autotracker_settings = settings

    def setup(self):
        """
        Apply acquisition settings as specified by the class properties
        """
        if self.enabled:
            # Turn off Live mode
            if self.mmStudio:
                snap_live_manager = self.mmStudio.get_snap_live_manager()
                if snap_live_manager.is_live_mode_on():
                    snap_live_manager.set_live_mode_on(False)

            # Apply microscope config settings
            for settings in self.microscope_settings.config_group_settings:
                microscope_operations.set_config(
                    self.mmc, settings.config_group, settings.config_name
                )

            # Apply microscope device property settings
            for settings in self.microscope_settings.device_property_settings:
                microscope_operations.set_property(
                    self.mmc,
                    settings.device_name,
                    settings.property_name,
                    settings.property_value,
                )

            # Apply ROI
            if self.microscope_settings.roi is not None:
                microscope_operations.set_roi(self.mmc, self.microscope_settings.roi)

            # Setup z scan stage
            microscope_operations.set_property(
                self.mmc, 'Core', 'Focus', self.slice_settings.z_stage_name
            )
            self._z0 = round(float(self.mmc.get_position(self.slice_settings.z_stage_name)), 3)

            # Note: sequencing should be turned off by default
            # Setup z sequencing
            if self.slice_settings.use_sequencing:
                for settings in self.microscope_settings.z_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value,
                    )

            # Setup channel sequencing
            if self.channel_settings.use_sequencing:
                for settings in self.microscope_settings.channel_sequencing_settings:
                    microscope_operations.set_property(
                        self.mmc,
                        settings.device_name,
                        settings.property_name,
                        settings.property_value,
                    )

<<<<<<< HEAD
=======
        # Zarr store will be initialized by MantisAcquisition after position_settings are updated

    def initialize_zarr_store(
        self,
        output_path: Union[str, os.PathLike] = None,
        position_settings: PositionSettings = None,
    ):
        if not self.enabled or output_path is None:
            return

        x_size = self.mmc.getImageWidth()
        y_size = self.mmc.getImageHeight()

        # Create dimensions list for the array using ZarrSettings
        dimensions = [
            Dimension(
                name='t',
                array_size_px=0,  # zero denotes the append dimension in acquire
                chunk_size_px=1,  # don't chunk in time dimension
                shard_size_chunks=1,  # don't shard in time dimension
                kind=DimensionType.TIME,
            ),
            Dimension(
                name='z',
                array_size_px=self.slice_settings.num_slices,
                chunk_size_px=min(
                    self.zarr_settings.chunk_sizes['z'],
                    max(1, self.slice_settings.num_slices),
                ),
                shard_size_chunks=self.zarr_settings.shard_sizes['z'],
                kind=DimensionType.SPACE,
            ),
            Dimension(
                name='y',
                array_size_px=y_size,
                chunk_size_px=min(self.zarr_settings.chunk_sizes['y'], max(1, y_size)),
                shard_size_chunks=self.zarr_settings.shard_sizes['y'],
                kind=DimensionType.SPACE,
            ),
            Dimension(
                name='x',
                array_size_px=x_size,
                chunk_size_px=min(self.zarr_settings.chunk_sizes['x'], max(1, x_size)),
                shard_size_chunks=self.zarr_settings.shard_sizes['x'],
                kind=DimensionType.SPACE,
            ),
        ]

        if self.channel_settings.num_channels > 1:
            dimensions.insert(
                1,
                Dimension(
                    name='c',
                    array_size_px=self.channel_settings.num_channels,
                    chunk_size_px=min(
                        self.zarr_settings.chunk_sizes['c'],
                        max(1, self.channel_settings.num_channels),
                    ),
                    shard_size_chunks=self.zarr_settings.shard_sizes['c'],
                    kind=DimensionType.CHANNEL,
                ),
            )

        # Create array settings using ZarrSettings
        array_settings = ArraySettings(
            dimensions=dimensions,
            data_type=self.zarr_settings.get_data_type_enum(),
            compression=self.zarr_settings.get_compression_settings(),
        )

        # Set store path in zarr_settings if not already set
        if self.zarr_settings.store_path is None:
            zarr_path = str(output_path)
            if not zarr_path.endswith('.zarr'):
                zarr_path += '.zarr'
            self.zarr_settings.store_path = zarr_path

        # Create stream settings with the array
        if self.zarr_settings.use_hcs_layout:
            # Create HCS layout with plate and wells
            plate = Plate(
                path=self.zarr_settings.plate_name,
                # description=self.zarr_settings.plate_description or "",
            )

            # Create wells from position settings
            wells = []
            for well_id in set(position_settings.well_ids if position_settings else []):
                well = Well(
                    name=well_id,
                    row=well_id[0] if len(well_id) > 0 else "0",  # Extract row letter
                    column=(well_id[1:] if len(well_id) > 1 else "0"),  # Extract column number
                )
                wells.append(well)
            plate.wells = wells

            stream_settings = StreamSettings(
                store_path=self.zarr_settings.store_path,
                arrays=[array_settings],
                version=self.zarr_settings.get_zarr_version_enum(),
                max_threads=self.zarr_settings.max_threads,
                hcs_plates=[plate],
            )
        else:
            stream_settings = StreamSettings(
                store_path=self.zarr_settings.store_path,
                arrays=[array_settings],
                version=self.zarr_settings.get_zarr_version_enum(),
                max_threads=self.zarr_settings.max_threads,
            )

        self._zarr_writer = ZarrStream(stream_settings)

        self.mmc.mda.events.frameReady.connect(self.write_data)

>>>>>>> 2b2b0f2 (Apply code formatting with black and isort)
    def reset(self):
        """
        Reset the microscope device properties, typically at the end of the acquisition
        """
        if self.enabled:
            # Reset device property settings
            for settings in self.microscope_settings.reset_device_properties:
                microscope_operations.set_property(
                    self.mmc,
                    settings.device_name,
                    settings.property_name,
                    settings.property_value,
                )

            # Reset z stage to initial position
            if self._z0 is not None:
                microscope_operations.set_z_position(
                    self.mmc, self.slice_settings.z_stage_name, self._z0
                )


# TODO: check the enable_ls_acq and enable_lf_acq work independently
class MantisAcquisition(object):
    """
    Acquisition class for simultaneous label-free and light-sheet acquisition on
    the mantis microscope.

    Parameters
    ----------
    acquisition_directory : str or PathLike
        Directory where acquired data will be saved
    acquisition_name : str
        Name of the acquisition
    mm_app_path : str
        Path to Micro-manager installation directory which runs the light-sheet
        acquisition, typically 'C:\\Program Files\\Micro-Manager-2.0_YYYY_MM_DD_2'
    config_file : str
        Path to config file which runs the light-sheet acquisition, typically
        'C:\\CompMicro_MMConfigs\\mantis\\mantis-LS.cfg'
    enable_ls_acq : bool, optional
        Set to False if only acquiring label-free data, by default True
    enable_lf_acq : bool, optional
        Set to False if only acquiring fluorescence light-sheet data, by default
        True
    demo_run : bool, optional
        Set to True if using the MicroManager demo config, by default False
    verbose : bool, optional
        By default False

    Examples
    -----
    This class should be used with a context manager for proper cleanup of the
    acquisition:

    >>> with MantisAcquisition(...) as acq:
            # define acquisition settings
            acq.time_settings = TimeSettings(...)
            acq.lf_acq.channel_settings = ChannelSettings(...)
            ...

            acq.setup()
            acq.acquire()
    """

    def __init__(
        self,
        acquisition_directory: Union[str, os.PathLike],
        acquisition_name: str,
        mm_app_path: str,
        mm_config_file: str,
        enable_ls_acq: bool = True,
        enable_lf_acq: bool = True,
        demo_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self._root_dir = Path(acquisition_directory).resolve()
        self._acq_name = acquisition_name
        self._demo_run = demo_run
        self._verbose = verbose
        self._lf_acq_obj = None
        self._ls_acq_obj = None

        globals.demo_run = demo_run

        if not enable_lf_acq or not enable_ls_acq:
            raise Exception('Disabling LF or LS acquisition is not currently supported')

        # Create acquisition directory and log directory
        self._acq_dir = _create_acquisition_directory(self._root_dir, self._acq_name)
        self._logs_dir = self._acq_dir / 'logs'
        self._logs_dir.mkdir()

        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        acq_log_path = self._logs_dir / f'mantis_acquisition_log_{timestamp}.txt'
        configure_debug_logger(acq_log_path)
        # configure copylot logger with matching stream handler
        # indexing into handlers[0] is a hack
        copylot.enable_logging(acq_log_path, logging.INFO)
        console_format = get_console_formatter()
        logging.getLogger('copylot').handlers[0].setFormatter(console_format)

        if self._demo_run:
            logger.info('NOTE: This is a demo run')
        logger.debug(f'Starting mantis acquisition log at: {acq_log_path}')

        # Log conda environment
        outs, errs = log_conda_environment(
            self._logs_dir / f'conda_environment_log_{timestamp}.txt'
        )
        if errs is None:
            logger.debug(outs.decode('ascii').strip())
        else:
            logger.error(errs.decode('ascii'))

        # initialize time and position settings
        self._time_settings = TimeSettings()
        self._position_settings = PositionSettings()

        # Connect to MM running LF acq
        self.lf_acq = BaseChannelSliceAcquisition(
            enabled=enable_lf_acq,
            zmq_port=LF_ZMQ_PORT,
        )

        # Connect to MM running LS acq
        self.ls_acq = BaseChannelSliceAcquisition(
            enabled=enable_ls_acq,
            mm_app_path=mm_app_path,
            mm_config_file=mm_config_file,
            zmq_port=LS_ZMQ_PORT,
            core_log_path=Path(mm_app_path) / 'CoreLogs' / f'CoreLog{timestamp}_headless.txt',
        )

    @property
    def time_settings(self):
        return self._time_settings

    @property
    def position_settings(self):
        return self._position_settings

    @time_settings.setter
    def time_settings(self, settings: TimeSettings):
        if settings is None:
            return
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._time_settings = settings

    @position_settings.setter
    def position_settings(self, settings: PositionSettings):
        if settings is None:
            return
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._position_settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Log final O3 stage position
        if self.ls_acq.o3_stage:
            _pos = float(self.ls_acq.mmc.get_position(self.ls_acq.o3_stage))
            logger.debug(f'Final O3 stage position: {_pos:.3f} um')

        # Shut down DAQ
        if not self._demo_run:
            self.cleanup_daq()

        # Reset LF and LS acquisitions
        self.lf_acq.reset()
        self.ls_acq.reset()

        # Abort acquisitions if they have not finished, usually after Ctr+C
        if self._lf_acq_obj:
            self._lf_acq_obj.abort()
        if self._ls_acq_obj:
            self._ls_acq_obj.abort()

    def load_position_settings(self):
        """
        Fetch positions defined in the Micro-manager Position List Manager
        """
        autofocus_stage = self.lf_acq.microscope_settings.autofocus_stage

        if self.position_settings.num_positions == 0:
            logger.debug('Fetching position list from Micro-manager')

            xyz_positions, position_labels = microscope_operations.get_position_list(
                self.lf_acq.mmStudio, autofocus_stage
            )

            if not xyz_positions:
                logger.debug('Micro-manager position list is empty. Fetching current position')

                xyz_positions, position_labels = microscope_operations.get_current_position(
                    self.lf_acq.mmc, autofocus_stage
                )

            self.position_settings = PositionSettings(
                xyz_positions=xyz_positions,
                position_labels=position_labels,
            )
            self.position_settings.xyz_positions_shift = deepcopy(xyz_positions)

        else:
            logger.debug('Position list is already populated and will not be updated')

    def update_mm_position_list(self):
        """
        Update the MicroManager position list with the current position settings.
        Currently done between acquisition chunks when using the autotracker.
        """

        microscope_operations.update_position_list(
            self.lf_acq.mmc,
            self.lf_acq.mmStudio,
            self.position_settings.xyz_positions,
            self.position_settings.position_labels,
            z_stage_name=self.lf_acq.microscope_settings.autofocus_stage,
        )

    def update_lf_acquisition_rates(self, lf_exposure_times: list):
        if self._demo_run:
            # Set approximate demo camera acquisition rate for use in await_cz_acq_completion
            self.lf_acq.slice_settings.acquisition_rate = 30
            return

        # Determine label-free acq timing
        oryx_framerate = float(self.lf_acq.mmc.get_property('Oryx', 'Frame Rate'))
        # assumes all channels have the same exposure time
        self.lf_acq.slice_settings.acquisition_rate = np.minimum(
            1000 / (lf_exposure_times[0] + MCL_STEP_TIME),
            np.floor(oryx_framerate),
        )
        self.lf_acq.channel_settings.acquisition_rate = 1 / (
            self.lf_acq.slice_settings.num_slices / self.lf_acq.slice_settings.acquisition_rate
            + LC_CHANGE_TIME / 1000
        )
        logger.debug(f'Maximum Oryx acquisition framerate: {oryx_framerate:.6f}')
        logger.debug(
            f'Label-free slice acquisition rate: {self.lf_acq.slice_settings.acquisition_rate:.6f}'
        )
        logger.debug(
            f'Label-free channel acquisition rate: {self.lf_acq.channel_settings.acquisition_rate:.6f}'
        )

    def update_ls_acquisition_rates(self, ls_exposure_times: list):
        if self._demo_run:
            # Set approximate demo camera acquisition rate for use in await_cz_acq_completion
            self.ls_acq.slice_settings.acquisition_rate = [
                np.minimum(30, 1000 / exp_time) for exp_time in ls_exposure_times
            ]
            self.ls_acq.channel_settings.min_exposure_time = 2  # useful for debugging
            return

        # Determine light-sheet acq timing
        ls_readout_time_ms = np.around(
            float(self.ls_acq.mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))
            * 1e-6,
            decimals=3,
        )
        _cam_max_fps = int(np.around(1000 / ls_readout_time_ms))
        # When using simulated global shutter by modulating the laser excitation time,
        # the exposure time needs to be greater than the sensor readout time
        self.ls_acq.channel_settings.min_exposure_time = ls_readout_time_ms
        for ls_exp_time in ls_exposure_times:
            assert (
                ls_readout_time_ms < ls_exp_time
            ), f'Exposure time needs to be greater than the {ls_readout_time_ms} ms sensor readout time'
        self.ls_acq.slice_settings.acquisition_rate = [
            1000 / (exp + ls_readout_time_ms + LS_POST_READOUT_DELAY)
            for exp in ls_exposure_times
        ]
        # self.ls_acq.channel_settings.acquisition_rate = [
        #     1 / (self.ls_acq.slice_settings.num_slices/acq_rate + LS_CHANGE_TIME/1000)
        #     for acq_rate in self.ls_acq.slice_settings.acquisition_rate
        # ]
        acq_rates = list(np.around(self.ls_acq.slice_settings.acquisition_rate, decimals=6))
        logger.debug(f'Maximum Prime BSI Express acquisition framerate: ~{_cam_max_fps}')
        logger.debug(f'Light-sheet slice acquisition rate: {acq_rates}')
        # logger.debug(f'Current light-sheet channel acquisition rate: ~{self.ls_acq.channel_settings.acquisition_rate}')

    def setup_daq(self):
        """
        Setup the NI DAQ to output trigger pulses for the label-free and
        light-sheet acquisitions. Acquisition can be sequenced across z slices
        and channels
        """
        self.update_lf_acquisition_rates(
            self.lf_acq.channel_settings.default_exposure_times_ms,
        )
        self.update_ls_acquisition_rates(
            self.ls_acq.channel_settings.default_exposure_times_ms,
        )

        if self._demo_run:
            logger.debug('DAQ setup is not supported in demo mode')
            return

        # LF channel trigger - accommodates longer LC switching times
        self._lf_channel_ctr_task = nidaqmx.Task('LF Channel Counter')
        microscope_operations.setup_daq_counter(
            self._lf_channel_ctr_task,
            co_channel='cDAQ1/_ctr0',
            freq=self.lf_acq.channel_settings.acquisition_rate,
            duty_cycle=0.1,
            samples_per_channel=self.lf_acq.channel_settings.num_channels,
            pulse_terminal='/cDAQ1/Ctr0InternalOutput',
        )

        # Sutter filter wheel is not capable of sequencing, so no need for channel sync here
        # # LS channel trigger
        # self._ls_channel_ctr_task = nidaqmx.Task('LS Channel Counter')
        # ls_channel_ctr = microscope_operations.setup_daq_counter(
        #     self._ls_channel_ctr_task,
        #     co_channel='cDAQ1/_ctr1',
        #     freq=self.ls_acq.channel_settings.acquisition_rate,
        #     duty_cycle=0.1,
        #     samples_per_channel=self.ls_acq.channel_settings.num_channels,
        #     pulse_terminal='/cDAQ1/Ctr1InternalOutput')

        # LF Z trigger
        self._lf_z_ctr_task = nidaqmx.Task('LF Z Counter')
        microscope_operations.setup_daq_counter(
            self._lf_z_ctr_task,
            co_channel='cDAQ1/_ctr2',
            freq=self.lf_acq.slice_settings.acquisition_rate,
            duty_cycle=0.1,
            samples_per_channel=self.lf_acq.slice_settings.num_slices,
            pulse_terminal='/cDAQ1/PFI0',
        )

        # LS Z trigger
        # LS Z counter will start with a software command
        # Counter frequency is updated for each channel in post-camera hook fn
        self._ls_z_ctr_task = nidaqmx.Task('LS Z Counter')
        microscope_operations.setup_daq_counter(
            self._ls_z_ctr_task,
            co_channel='cDAQ1/_ctr3',
            freq=self.ls_acq.slice_settings.acquisition_rate[0],
            duty_cycle=0.1,
            samples_per_channel=self.ls_acq.slice_settings.num_slices,
            pulse_terminal='/cDAQ1/PFI1',
        )

        # # The LF Channel counter task serve as a master start trigger
        # # LF Channel counter triggers LS Channel counter
        # logger.debug('Setting up LF Channel counter as start trigger for LS Channel counter')
        # self._ls_channel_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
        #     trigger_source='/cDAQ1/Ctr0InternalOutput',
        #     trigger_edge=Slope.RISING)

        # # LS Channel counter triggers LS Z counter
        # logger.debug('Setting up LS Channel counter as start trigger for LS Z counter')
        # self._ls_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
        #     trigger_source='/cDAQ1/Ctr1InternalOutput',
        #     trigger_edge=Slope.RISING)
        # # will always return is_task_done = False after counter is started
        # logger.debug('Setting up LS Z counter as retriggerable')
        # self._ls_z_ctr_task.triggers.start_trigger.retriggerable = True

        # LF Channel counter triggers LF Z counter
        logger.debug('Setting up LF Channel counter as start trigger for LF Z counter')
        self._lf_z_ctr_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source='/cDAQ1/Ctr0InternalOutput', trigger_edge=Slope.RISING
        )
        # will always return is_task_done = False after counter is started
        logger.debug('Setting up LF Z counter as retriggerable')
        self._lf_z_ctr_task.triggers.start_trigger.retriggerable = True

    def cleanup_daq(self):
        logger.debug('Stopping DAQ counter tasks')
        if self.ls_acq.enabled:
            self._ls_z_ctr_task.stop()
            self._ls_z_ctr_task.close()

        if self.lf_acq.enabled:
            self._lf_z_ctr_task.stop()
            self._lf_z_ctr_task.close()
            self._lf_channel_ctr_task.stop()
            self._lf_channel_ctr_task.close()

    def setup_autofocus(self):
        if self.lf_acq.microscope_settings.use_autofocus:
            autofocus_method = self.lf_acq.microscope_settings.autofocus_method
            logger.debug(f'Setting autofocus method as {autofocus_method}')
            self.lf_acq.mmc.set_auto_focus_device(autofocus_method)
        else:
            logger.debug('Autofocus is not enabled')

        # Connect to LS O3 scan stage
        self.ls_acq.o3_stage = 'O3 Piezo'

    def setup_autoexposure(self):
        # assign exposure_times_per_well and laser_powers_per_well to default values
        for well_id in set(self.position_settings.well_ids):
            self.ls_acq.channel_settings.exposure_times_per_well[well_id] = deepcopy(
                self.ls_acq.channel_settings.default_exposure_times_ms
            )
            self.ls_acq.channel_settings.laser_powers_per_well[well_id] = deepcopy(
                self.ls_acq.channel_settings.default_laser_powers
            )

        if not any(self.ls_acq.channel_settings.use_autoexposure):
            logger.debug(
                'Autoexposure is not enabled for any channels. Using default exposure time and laser power'
            )
            return

        if self._demo_run:
            logger.debug(
                'Autoexposure is not supported in demo mode. Using default exposure time and laser power'
            )
            return

        if self.ls_acq.autoexposure_settings.autoexposure_method is None:
            raise ValueError(
                'Autoexposure is requested, but autoexposure settings are not provided. '
                'Please provide autoexposure settings in the acquisition config file.'
            )

        logger.debug('Setting up autoexposure for light-sheet acquisition')
        if self.ls_acq.autoexposure_settings.autoexposure_method == 'manual':
            # Check that the 'illumination.csv' file exists
            if not (self._root_dir / 'illumination.csv').exists():
                raise FileNotFoundError(
                    f'The illumination.csv file required for manual autoexposure was not found in {self._root_dir}'
                )
            illumination_settings = load_manual_illumination_settings(
                self._root_dir / 'illumination.csv',
            )
            # Check that exposure times are greater than the minimum exposure time
            if not (
                illumination_settings["exposure_time_ms"]
                > self.ls_acq.channel_settings.min_exposure_time
            ).all():
                raise ValueError(
                    f'All exposure times in the illumination.csv file must be greater than the minimum exposure time of {self.ls_acq.channel_settings.min_exposure_time} ms.'
                )
            # Check that illumination settings are provided for all wells
            if not set(illumination_settings.index.values) == set(
                self.position_settings.well_ids
            ):
                raise ValueError(
                    'Well IDs in the illumination.csv file do not match the well IDs in the position settings.'
                )

        # initialize lasers
        for channel_idx, config_name in enumerate(self.ls_acq.channel_settings.channels):
            if self.ls_acq.channel_settings.use_autoexposure[channel_idx]:
                config_group = self.ls_acq.channel_settings.channel_group
                config = self.ls_acq.mmc.get_config_data(config_group, config_name)
                ts2_ttl_state = int(
                    config.get_setting('TS2_TTL1-8', 'State').get_property_value()
                )
                if ts2_ttl_state == 32:
                    # State 32 corresponds to illumination with 488 laser
                    self.ls_acq.channel_settings.light_sources[channel_idx] = (
                        microscope_operations.setup_vortran_laser(VORTRAN_488_COM_PORT)
                    )
                elif ts2_ttl_state == 64:
                    # State 64 corresponds to illumination with 561 laser
                    self.ls_acq.channel_settings.light_sources[channel_idx] = (
                        microscope_operations.setup_vortran_laser(VORTRAN_561_COM_PORT)
                    )
                elif ts2_ttl_state == 128:
                    # State 128 corresponds to illumination with 639 laser
                    self.ls_acq.channel_settings.light_sources[channel_idx] = (
                        microscope_operations.setup_vortran_laser(VORTRAN_639_COM_PORT)
                    )
                else:
                    logger.error(
                        'Unknown TTL state {} for channel {} in config group {}'.format(
                            ts2_ttl_state, config_name, config_group
                        )
                    )

    def update_position_autotracker(self):
        # Update the position list from the backup
        self.position_settings.xyz_positions = deepcopy(
            self.position_settings.xyz_positions_shift
        )

    def setup_o3_refocus(self):
        """
        The position of the O3 refocus stage resets between chunks in a chunked acquisition.
        Here we'll read the final position of the O3 refocus stage from the log file of the
        previous chunk and set the O3 refocus stage to that position at the start of the current chunk.
        """
        if not self.ls_acq.microscope_settings.use_o3_refocus:
            return

        acq_name_parts = self._acq_dir.name.split('_')
        acq_name = '_'.join(acq_name_parts[:-1])
        prev_chunk = int(acq_name_parts[-1]) - 1
        prev_logs_dir = self._root_dir / f'{acq_name}_{prev_chunk}' / 'logs'
        prev_log_file = list(prev_logs_dir.glob('mantis_acquisition_log_*.txt'))
        if not prev_log_file:
            logger.debug(
                'No log files from a previous acquisition found. Will not change O3 stage position.'
            )
            return

        o3_position = None
        with open(prev_log_file[0], 'r') as f:
            for line in reversed(f.readlines()):
                match = re.search(r"Final O3 stage position: ([-\d\.]+) um", line)
                if match:
                    o3_position = float(match.group(1))
                    break
        if o3_position is None:
            logger.debug(
                "The final O3 stage position was not found in the previous log file. Will not change O3 stage position."
            )
            return

        logger.debug(f'Updating O3 stage position to {o3_position:.3f} um')
        microscope_operations.set_z_position(
            self.ls_acq.mmc, self.ls_acq.o3_stage, o3_position
        )

    def go_to_position(self, position_index: int):
        # Move slowly for short distances such that autofocus can stay engaged.
        # Autofocus typically fails when moving long distances, so we can move
        # quickly and re-engage
        slow_speed = 2.0
        fast_speed = 5.75
        short_distance = 2000  # in um
        p_label = self.position_settings.position_labels[position_index]

        # only change stage speed if using autofocus
        if self.lf_acq.microscope_settings.use_autofocus and not self._demo_run:
            current_xy_position = np.asarray(
                [self.lf_acq.mmc.get_x_position(), self.lf_acq.mmc.get_y_position()]
            )
            target_xy_position = np.asarray(
                self.position_settings.xyz_positions[position_index][:2]
            )
            distance = np.linalg.norm(target_xy_position - current_xy_position)

            speed = slow_speed if distance < short_distance else fast_speed
            microscope_operations.set_property(
                self.lf_acq.mmc, 'XYStage:XY:31', 'MotorSpeedX-S(mm/s)', speed
            )
            microscope_operations.set_property(
                self.lf_acq.mmc, 'XYStage:XY:31', 'MotorSpeedY-S(mm/s)', speed
            )

        logger.debug(
            f'Moving to position {p_label} with coordinates {self.position_settings.xyz_positions[position_index]}'
        )
        microscope_operations.set_xy_position(
            self.lf_acq.mmc, self.position_settings.xyz_positions[position_index][:2]
        )
        microscope_operations.wait_for_device(
            self.lf_acq.mmc, self.lf_acq.mmc.get_xy_stage_device()
        )

        # Note: only set the z position if not using autofocus. Calling
        # set_z_position will disengage continuous autofocus. The autofocus
        # algorithm sets the z position independently
        if not self.lf_acq.microscope_settings.use_autofocus:
            if self.lf_acq.microscope_settings.autofocus_stage:
                microscope_operations.set_z_position(
                    self.lf_acq.mmc,
                    self.lf_acq.microscope_settings.autofocus_stage,
                    self.position_settings.xyz_positions[position_index][2],
                )
                microscope_operations.wait_for_device(
                    self.lf_acq.mmc,
                    self.lf_acq.microscope_settings.autofocus_stage,
                )

    def acquire_ls_defocus_stack(
        self,
        z_range: Iterable,
        galvo_range: Iterable,
        use_pycromanager: bool = False,
    ) -> np.ndarray:
        """Acquire defocus stacks at different galvo positions and return image data

        Parameters
        ----------
        z_range : Iterable
            Absolute range when using pycromanager, relative range otherwise
        galvo_range : Iterable
            Absolute galvo range
        use_pycromanager : bool, optional
            Flag to use pycromanager for acquisition, by default False

        Returns
        -------
        data : np.ndarray

        """
        data = []
        mmc = self.ls_acq.mmc
        config_group = self.ls_acq.microscope_settings.o3_refocus_config.config_group
        config_name = self.ls_acq.microscope_settings.o3_refocus_config.config_name
        config_idx = self.ls_acq.channel_settings.channels.index(config_name)
        exposure_time = self.ls_acq.channel_settings.default_exposure_times_ms[config_idx]
        z_stage = self.ls_acq.o3_stage
        galvo = self.ls_acq.slice_settings.z_stage_name

        # Set config
        if config_name is not None:
            mmc.set_config(config_group, config_name)
            mmc.wait_for_config(config_group, config_name)

        # Set exposure time
        if exposure_time is not None:
            mmc.set_exposure(exposure_time)

        # Open shutter
        auto_shutter_state, shutter_state = microscope_operations.get_shutter_state(mmc)
        microscope_operations.open_shutter(mmc)

        # get galvo and z stage starting position
        p0 = float(mmc.get_position(galvo))
        z0 = float(mmc.get_position(z_stage))

        # set camera to internal trigger
        # TODO: do this properly, context manager?
        microscope_operations.set_property(
            mmc, 'Prime BSI Express', 'TriggerMode', 'Internal Trigger'
        )
        if use_pycromanager:
            tempdir = TemporaryDirectory()
            focus_stage = mmc.get_property('Core', 'Focus')
            microscope_operations.set_property(mmc, 'Core', 'Focus', z_stage)

        # acquire stacks at different galvo positions
        for p_idx, p in enumerate(galvo_range):
            # set galvo position
            microscope_operations.set_z_position(mmc, galvo, p)

            # acquire defocus stack
            if use_pycromanager:
                global acq_finished
                acq_finished = False
                acq_fps = 20  # TODO: hardcoded for now
                camera = 'Prime BSI Express'
                num_slices = len(z_range)
                acq_duration = num_slices / acq_fps + 5  # Extra buffer time

                def check_acq_finished(axes, dataset):
                    global acq_finished
                    if axes['z'] == num_slices - 1:
                        acq_finished = True

                microscope_operations.set_z_position(mmc, z_stage, z_range[0])

                logger.debug('Starting pycromanager O3 autofocus acquisition')
                events = multi_d_acquisition_events(
                    z_start=z_range[0],
                    z_end=z_range[-1],
                    z_step=z_range[1] - z_range[0],
                )
                acq = Acquisition(
                    tempdir.name,
                    f'ls_refocus_p{p_idx}',
                    port=LS_ZMQ_PORT,
                    image_saved_fn=check_acq_finished,
                    show_display=False,
                )
                acq.acquire(events)
                acq.mark_finished()
                start_time = time.time()
                while not acq_finished and time.time() - start_time < acq_duration:
                    time.sleep(0.2)
                if acq_finished:
                    acq.await_completion()
                    logger.debug('Pycromanager acquisition finished. Fetching data')
                    ds = acq.get_dataset()
                    data.append(np.asarray(ds.as_array()))
                    logger.debug('Data retrieved. Closing dataset')
                    ds.close()
                else:
                    logger.error('O3 autofocus is taking longer than expected - aborting.')
                    microscope_operations.abort_acquisition_sequence(self.ls_acq.mmc, camera)
                    logger.debug('Would have waited for acquisition to complete')
                    # acq.await_completion()  # Cleanup
                    logger.debug('Closing dataset')
                    acq.get_dataset().close()  # Close dataset
                    logger.debug('Acquisition aborted')
            else:
                z_stack = microscope_operations.acquire_defocus_stack(
                    mmc, z_stage, z_range, backlash_correction_distance=KIM101_BACKLASH
                )
                data.append(z_stack)

        if use_pycromanager:
            microscope_operations.set_property(mmc, 'Core', 'Focus', focus_stage)
            tempdir.cleanup()

        # Reset camera triggering
        microscope_operations.set_property(
            mmc, 'Prime BSI Express', 'TriggerMode', 'Edge Trigger'
        )

        # Reset stages
        microscope_operations.set_z_position(mmc, galvo, p0)
        microscope_operations.set_z_position(mmc, z_stage, z0)

        # Reset shutter
        microscope_operations.reset_shutter(mmc, auto_shutter_state, shutter_state)

        return np.asarray(data)

    def refocus_ls_path(
        self, scan_left: bool = False, scan_right: bool = False
    ) -> tuple[bool, bool, bool]:
        logger.info('Running O3 refocus algorithm on light-sheet arm')
        success = False
        o3_low_limit = 0
        o3_high_limit = 30

        # Define O3 z range
        # The stack starts close to O2 and moves away
        o3_z_stage = self.ls_acq.o3_stage
        o3_position = float(self.ls_acq.mmc.get_property(o3_z_stage, 'Position'))
        logger.debug(f'Starting O3 position: {o3_position} um')

        o3_z_start = -5.0
        o3_z_end = 5.0
        o3_z_step = 0.3
        if scan_left:
            logger.info('O3 refocus will scan further to the left')
            o3_z_start *= 2
        if scan_right:
            logger.info('O3 refocus will scan further to the right')
            o3_z_end *= 2
        o3_range_rel = np.arange(o3_z_start, o3_z_end + o3_z_step, o3_z_step)
        o3_range_abs = o3_range_rel + o3_position

        valid_indices = (o3_range_abs >= o3_low_limit) & (o3_range_abs <= o3_high_limit)
        o3_range_rel = o3_range_rel[valid_indices]
        o3_range_abs = o3_range_abs[valid_indices]
        if not all(valid_indices):
            logger.warning(
                'Some O3 positions are outside the valid range. Truncating O3 travel range.'
            )
        if o3_range_rel.size < 3:
            logger.error('Insufficient O3 travel range. Aborting O3 refocus.')
            return success, scan_left, scan_right

        # Define galvo range, i.e. galvo positions at which O3 defocus stacks
        # are acquired, here at 30%, 50%, and 70% of galvo range. Should be odd number
        galvo_scan_range = self.ls_acq.slice_settings.z_range
        len_galvo_scan_range = len(galvo_scan_range)
        galvo_range_abs = [
            galvo_scan_range[int(0.4 * len_galvo_scan_range)],
            galvo_scan_range[int(0.5 * len_galvo_scan_range)],
            galvo_scan_range[int(0.6 * len_galvo_scan_range)],
        ]

        # Acquire defocus stacks at several galvo positions
        data = self.acquire_ls_defocus_stack(
            z_range=o3_range_abs,
            galvo_range=galvo_range_abs,
            use_pycromanager=False,
        )

        # Abort if the acquisition failed
        if not data.size > 0:
            logger.error('No data was acquired during O3 autofocus - aborting.')
            return success, scan_left, scan_right

        # Save acquired stacks in logs
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        tifffile.imwrite(
            self._logs_dir / f'ls_refocus_data_{timestamp}.ome.tif',
            np.expand_dims(data, -3).astype('uint16'),
        )

        # Find in-focus slice
        wavelength = 0.55  # in um, approx
        # works well to distinguish between noise and sample when using z_step = 15
        # the idea is that true features in the sample will come in focus slowly
        threshold_FWHM = 4.5

        focus_indices = []
        peak_indices = []
        for stack_idx, stack in enumerate(data):
            idx, stats = focus_from_transverse_band(
                stack,
                NA_det=NA_DETECTION,
                lambda_ill=wavelength,
                pixel_size=LS_PIXEL_SIZE,
                threshold_FWHM=threshold_FWHM,
                plot_path=self._logs_dir / f'ls_refocus_plot_{timestamp}_Pos{stack_idx}.png',
                return_statistics=True,
            )
            focus_indices.append(idx)
            peak_indices.append(stats['peak_index'])
        logger.debug(
            'Stacks at galvo positions %s are in focus at slice %s',
            np.round(galvo_range_abs, 3),
            focus_indices,
        )

        # Refocus O3
        # Some focus_indices may be None, e.g. if there is no sample
        valid_focus_indices = [idx for idx in focus_indices if idx is not None]
        if valid_focus_indices:
            focus_idx = int(np.median(valid_focus_indices))
            o3_position_rel = round(o3_range_rel[focus_idx], 2)
            o3_position_abs = o3_range_abs[focus_idx]

            logger.info(f'Moving O3 by {o3_position_rel} um')
            logger.debug(f'Moving O3 to {o3_position_abs} um')
            microscope_operations.set_z_position(self.ls_acq.mmc, o3_z_stage, o3_position_abs)
            success = True
        else:
            logger.error(
                'Could not determine the correct O3 in-focus position. O3 will not move'
            )
            if not any((scan_left, scan_right)):
                # Only do this if we are not already scanning at an extended range
                peak_indices = np.asarray(peak_indices)
                max_idx = len(o3_range_rel) - 1
                if all(peak_indices < 0.2 * max_idx):
                    scan_left = True
                    logger.info(
                        'O3 autofocus will scan further to the left at the next iteration'
                    )
                if all(peak_indices > 0.8 * max_idx):
                    scan_right = True
                    logger.info(
                        'O3 autofocus will scan further to the right at the next iteration'
                    )

        return success, scan_left, scan_right

    def run_autoexposure(
        self,
        acq: BaseChannelSliceAcquisition,
        well_id: str,
        method: str = 'manual',
    ):
        """
        If autoexposure on any channel is requested, this method will switch to
        that channel and call the specified autoexposure method. The optimal
        exposure time and laser power are logged per well in the
        channel_settings object of that acquisition.

        Parameters
        ----------
        acq : BaseChannelSliceAcquisition
            Label-free or light-sheet acquisition object
        well_id : str
            ID or name of the current well, e.g. "0" or "A1"
        method : str, optional
            Autoexposure method
        """
        if not any(acq.channel_settings.use_autoexposure):
            return

        for channel_idx, channel_name in enumerate(acq.channel_settings.channels):
            if acq.channel_settings.use_autoexposure[channel_idx]:
                logger.info(f'Running autoexposure on channel {channel_name}')
                microscope_operations.set_config(
                    acq.mmc,
                    acq.channel_settings.channel_group,
                    channel_name,
                )
                if method == 'manual':
                    illumination_settings_filepath = self._root_dir / 'illumination.csv'
                    if not illumination_settings_filepath.is_file():
                        raise ValueError(
                            'Illumination settings for manual autoexposure cannot be found in '
                            f'{illumination_settings_filepath}'
                        )

                    (
                        acq.channel_settings.exposure_times_per_well[well_id][channel_idx],
                        acq.channel_settings.laser_powers_per_well[well_id][channel_idx],
                    ) = microscope_operations.autoexposure(
                        acq.mmc,
                        acq.channel_settings.light_sources[channel_idx],
                        acq.autoexposure_settings,
                        method,
                        illumination_settings_filepath=illumination_settings_filepath,
                        well_id=well_id,
                    )
                else:
                    raise NotImplementedError(
                        f'Autoexposure method {method} is not yet implemented.'
                    )

    def run_autotracker(
        self,
        acq: BaseChannelSliceAcquisition,
        well_id: str,
        method: str = 'manual',
    ):
        logging.debug('running autotracker')
        if not any(acq.channel_settings.use_autoexposure):
            return
        # TODO: implement autotracker
        microscope_operations.autotracker(acq.mmc, acq.autoexposure_settings)

    def setup(self):
        """
        Setup the mantis acquisition. This method sets up the label-free
        acquisition, the light-sheet acquisition, the NI DAQ, the autofocus, and
        fetches positions defined in the Micro-manager Position List Manager
        """

        logger.info('Setting up acquisition')

        logger.debug('Setting up label-free acquisition')
        self.lf_acq.setup()

        logger.debug('Setting up light-sheet acquisition')
        self.ls_acq.setup()

        logger.debug('Setting up DAQ')
        self.setup_daq()

        logger.debug('Setting up autofocus')
        self.setup_autofocus()

        logger.debug('Updating position settings')
        self.load_position_settings()

<<<<<<< HEAD
=======
        logger.debug('Initializing zarr stores')
        if self.lf_acq.enabled:
            self.lf_acq.initialize_zarr_store(
                f'{self._acq_dir}/{self._acq_name}_{LF_ACQ_LABEL}', self.position_settings
            )
        if self.ls_acq.enabled:
            self.ls_acq.initialize_zarr_store(
                f'{self._acq_dir}/{self._acq_name}_{LS_ACQ_LABEL}', self.position_settings
            )

>>>>>>> 2b2b0f2 (Apply code formatting with black and isort)
        logger.debug('Setting up autoexposure')
        self.setup_autoexposure()

        logger.debug('Setting up O3 refocus')
        self.setup_o3_refocus()

    def acquire(self):
        """
        Simultaneously acquire label-free and light-sheet data over multiple
        positions and time points.
        """

        use_autotracker = any(
            [
                self.lf_acq.microscope_settings.autotracker_config,
                self.ls_acq.microscope_settings.autotracker_config,
            ]
        )

        # define LF hook functions
        if self._demo_run:
            lf_pre_hardware_hook_fn = log_preparing_acquisition
            lf_post_camera_hook_fn = None
        else:
            lf_pre_hardware_hook_fn = partial(
                lf_pre_hardware_hook_function,
                [self._lf_z_ctr_task, self._lf_channel_ctr_task],
            )
            lf_post_camera_hook_fn = partial(
                start_daq_counters, [self._lf_z_ctr_task, self._lf_channel_ctr_task]
            )
        lf_post_hardware_hook_fn = log_acquisition_start

        # TODO: implement logic for the autotracker_img_saved_hook_fn
        if self.lf_acq.microscope_settings.autotracker_config is not None:
            zyx_shape = (self.lf_acq.slice_settings.num_slices,) + tuple(
                self.lf_acq.microscope_settings.roi[-2:][::-1]
            )
            logger.info("Instantiating autotracker on lf arm...")
            autotracker = Autotracker(
                arm='lf',
                zyx_shape=zyx_shape,
                position_settings=self.position_settings,
                channel_config=self.lf_acq.microscope_settings.autotracker_config,
                output_shift_path=self._logs_dir,
                settings=self.lf_acq.autotracker_settings,
            )
            lf_image_saved_fn = autotracker.track

        else:
            logger.info(
                'No autotracker config found for LF acquisition. Using default image saved hook'
            )
            lf_image_saved_fn = check_lf_acq_finished

        # define LF acquisition
        self._lf_acq_obj = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_{LF_ACQ_LABEL}',
            port=LF_ZMQ_PORT,
            pre_hardware_hook_fn=lf_pre_hardware_hook_fn,
            post_hardware_hook_fn=lf_post_hardware_hook_fn,
            post_camera_hook_fn=lf_post_camera_hook_fn,
            image_saved_fn=lf_image_saved_fn,  # data processing and display
            show_display=False,
        )

        # define LS hook functions
        if self._demo_run:
            ls_pre_hardware_hook_fn = None
            ls_post_hardware_hook_fn = None
            ls_post_camera_hook_fn = None
        else:
            ls_pre_hardware_hook_fn = partial(
                ls_pre_hardware_hook_function, [self._ls_z_ctr_task]
            )
            ls_post_hardware_hook_fn = partial(
                update_ls_hardware,
                self._ls_z_ctr_task,
                self.ls_acq.channel_settings.light_sources,
                self.ls_acq.channel_settings.channels,
            )
            ls_post_camera_hook_fn = partial(start_daq_counters, [self._ls_z_ctr_task])

        # TODO: implement logic for the autotracker_img_saved_hook_fn
        if self.ls_acq.microscope_settings.autotracker_config is not None:
            zyx_shape = (self.ls_acq.slice_settings.num_slices,) + tuple(
                self.ls_acq.microscope_settings.roi[-2:][::-1]
            )
            logger.info("Instantiating autotracker on ls arm...")
            autotracker = Autotracker(
                arm='ls',
                zyx_shape=zyx_shape,
                position_settings=self.position_settings,
                channel_config=self.ls_acq.microscope_settings.autotracker_config,
                output_shift_path=self._logs_dir,
                settings=self.ls_acq.autotracker_settings,
            )
            ls_image_saved_fn = autotracker.track

        else:
            logger.info(
                'No autotracker config found for LS acquisition. Using default image saved hook'
            )
            ls_image_saved_fn = check_ls_acq_finished

        # define LS acquisition
        self._ls_acq_obj = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_{LS_ACQ_LABEL}',
            port=LS_ZMQ_PORT,
            pre_hardware_hook_fn=ls_pre_hardware_hook_fn,
            post_hardware_hook_fn=ls_post_hardware_hook_fn,
            post_camera_hook_fn=ls_post_camera_hook_fn,
            image_saved_fn=ls_image_saved_fn,
            saving_queue_size=500,
            show_display=False,
        )

        # Generate LF acquisition events
        lf_cz_events = _generate_channel_slice_acq_events(
            self.lf_acq.channel_settings, self.lf_acq.slice_settings
        )
        # Generate LS acquisition events
        ls_cz_events = _generate_channel_slice_acq_events(
            self.ls_acq.channel_settings, self.ls_acq.slice_settings
        )

        logger.info('Starting acquisition')
        ls_o3_refocus_time = time.time()
        previous_well_id = None
        previous_position_label = None
        for t_idx in range(self.time_settings.num_timepoints):
            timepoint_start_time = time.time()
            for p_idx in range(self.position_settings.num_positions):
                p_label = self.position_settings.position_labels[p_idx]
                well_id = self.position_settings.well_ids[p_idx]

                # move to the given position
                if use_autotracker:
                    # TODO: Should we get the corods from the csv file or the modified xyz_positions_shifts
                    logger.debug('Updating the positions for autotracker')
                    logger.debug(
                        'Previous position: %f,%f ',
                        *self.position_settings.xyz_positions[p_idx][0:2],
                    )
                    self.update_position_autotracker()
                if p_label != previous_position_label or use_autotracker:
                    self.go_to_position(p_idx)

                    # TODO get the delta shifts
                    # read the files here and move separately

                # autofocus
                if self.lf_acq.microscope_settings.use_autofocus:
                    autofocus_success = microscope_operations.autofocus(
                        self.lf_acq.mmc,
                        self.lf_acq.mmStudio,
                        self.lf_acq.microscope_settings.autofocus_stage,
                        self.position_settings.xyz_positions[p_idx][2],
                    )
                    if not autofocus_success:
                        # abort acquisition at this time/position index
                        logger.error(
                            f'Autofocus failed. Aborting acquisition for timepoint {t_idx} at position {p_label}'
                        )
                        continue

                # autoexposure
                if well_id != previous_well_id:
                    globals.new_well = True
                    if t_idx == 0 or self.ls_acq.autoexposure_settings.rerun_each_timepoint:
                        self.run_autoexposure(
                            acq=self.ls_acq,
                            well_id=well_id,
                            method=self.ls_acq.autoexposure_settings.autoexposure_method,
                        )

                        # needs to be set before calling update_laser_power
                        globals.ls_laser_powers = (
                            self.ls_acq.channel_settings.laser_powers_per_well[well_id]
                        )
                        # This is a bit of a hack, laser powers should be set in update_ls_hardware
                        for c_idx in range(self.ls_acq.channel_settings.num_channels):
                            update_laser_power(
                                self.ls_acq.channel_settings.light_sources, c_idx
                            )

                    # Acq rate needs to be updated even if autoexposure was not rerun in this well
                    # Only do that if we are using autoexposure?
                    self.update_ls_acquisition_rates(
                        self.ls_acq.channel_settings.exposure_times_per_well[well_id]
                    )
                    # needs to be set after calling update_ls_acquisition_rates
                    globals.ls_slice_acquisition_rates = (
                        self.ls_acq.slice_settings.acquisition_rate
                    )

                # O3 refocus
                # Failing to refocus O3 will not abort the acquisition at the current PT index
                if self.ls_acq.microscope_settings.use_o3_refocus:
                    current_time = time.time()
                    # Always refocus at the start
                    if (
                        (t_idx == 0 and p_idx == 0)
                        or current_time - ls_o3_refocus_time
                        > self.ls_acq.microscope_settings.o3_refocus_interval_min * 60
                    ):
                        # O3 refocus can be skipped for certain wells
                        if well_id in self.ls_acq.microscope_settings.o3_refocus_skip_wells:
                            logger.debug(
                                f'O3 refocus is due, but will be skipped in well {well_id}.'
                            )
                        else:
                            success, scan_left, scan_right = self.refocus_ls_path()
                            # If autofocus fails, try again with extended range if we know which way to go
                            if not success and any((scan_left, scan_right)):
                                success, _, _ = self.refocus_ls_path(scan_left, scan_right)
                            # If it failed again, retry at the next position
                            if success:
                                ls_o3_refocus_time = current_time

                # update events dictionaries
                lf_events = deepcopy(lf_cz_events)
                for _event in lf_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_label
                    _event['min_start_time'] = 0

                ls_events = deepcopy(ls_cz_events)
                for _event in ls_events:
                    _event['axes']['time'] = t_idx
                    _event['axes']['position'] = p_label
                    _event['min_start_time'] = 0
                    if any(self.ls_acq.channel_settings.use_autoexposure):
                        channel_index = self.ls_acq.channel_settings.channels.index(
                            _event['axes']['channel']
                        )
                        _event['exposure'] = (
                            self.ls_acq.channel_settings.exposure_times_per_well[well_id][
                                channel_index
                            ]
                        )

                globals.lf_last_img_idx = lf_events[-1]['axes']
                globals.ls_last_img_idx = ls_events[-1]['axes']
                globals.lf_acq_finished = False
                globals.lf_acq_aborted = False
                globals.ls_acq_finished = False
                globals.ls_acq_aborted = False

                # start acquisition
                self._ls_acq_obj.acquire(ls_events)
                self._lf_acq_obj.acquire(lf_events)

                # wait for CZYX acquisition to finish
                self.await_cz_acq_completion()
                lf_acq_aborted, ls_acq_aborted = self.abort_stalled_acquisition()
                error_message = (
                    '{} acquisition for timepoint {} at position {} did not complete in time. '
                    'Aborting acquisition'
                )
                if lf_acq_aborted:
                    logger.error(error_message.format('Label-free', t_idx, p_label))
                if ls_acq_aborted:
                    logger.error(error_message.format('Light-sheet', t_idx, p_label))
                previous_well_id = well_id
                previous_position_label = p_label
                globals.new_well = False

            # wait for time interval between time points
            t_wait = self.time_settings.time_interval_s - (time.time() - timepoint_start_time)
            if t_wait > 0 and t_idx < self.time_settings.num_timepoints - 1:
                logger.info(f"Waiting {t_wait/60:.2f} minutes until the next time point")
                time.sleep(t_wait)

        self._ls_acq_obj.mark_finished()
        self._lf_acq_obj.mark_finished()

        logger.debug('Waiting for acquisition to finish')

        self._ls_acq_obj.await_completion()
        logger.debug('Light-sheet acquisition finished')
        self._lf_acq_obj.await_completion()
        logger.debug('Label-free acquisition finished')

        # Close ndtiff dataset - not sure why this is necessary
        self._lf_acq_obj.get_dataset().close()
        self._ls_acq_obj.get_dataset().close()

        # Clean up pycromanager acquisition objects
        self._lf_acq_obj = None
        self._ls_acq_obj = None

        logger.info('Acquisition finished')

        if use_autotracker:
            # update self.position_settings with the latest autotracker positions
            self.update_position_autotracker()
            # update mm position list
            self.update_mm_position_list()

    def await_cz_acq_completion(self):
        # LS acq time
        num_slices = self.ls_acq.slice_settings.num_slices
        slice_acq_rate = self.ls_acq.slice_settings.acquisition_rate  # list
        num_channels = self.ls_acq.channel_settings.num_channels
        ls_acq_time = sum(
            [num_slices / rate for rate in slice_acq_rate]
        ) + LS_CHANGE_TIME / 1000 * (num_channels - 1)

        # LF acq time
        num_slices = self.lf_acq.slice_settings.num_slices
        slice_acq_rate = self.lf_acq.slice_settings.acquisition_rate  # float
        num_channels = self.lf_acq.channel_settings.num_channels
        lf_acq_time = num_slices / slice_acq_rate * num_channels + LC_CHANGE_TIME / 1000 * (
            num_channels - 1
        )

        wait_time = np.ceil(np.maximum(ls_acq_time, lf_acq_time))
        time.sleep(wait_time)

    def abort_stalled_acquisition(self):
        # TODO: buffer_time was bumped up 10x in autotracker_refactor
        buffer_time = 100
        lf_acq_aborted = False
        ls_acq_aborted = False

        t_start = time.time()
        while (
            not all((globals.lf_acq_finished, globals.ls_acq_finished))
            and (time.time() - t_start) < buffer_time
        ):
            time.sleep(0.2)

        # TODO: a lot of hardcoded values here
        if not globals.lf_acq_finished:
            # abort LF acq
            lf_acq_aborted = True
            camera = 'Camera' if self._demo_run else 'Oryx'
            sequenced_stages = []
            if self.lf_acq.slice_settings.use_sequencing:
                sequenced_stages.append(self.lf_acq.slice_settings.z_stage_name)
            if (
                self.lf_acq.channel_settings.use_sequencing
                and self.lf_acq.channel_settings.num_channels > 1
                and not self._demo_run
            ):
                sequenced_stages.extend(['TS1_DAC01', 'TS1_DAC02'])
            microscope_operations.abort_acquisition_sequence(
                self.lf_acq.mmc, camera, sequenced_stages
            )
            # set a flag to clear any remaining events
            globals.lf_acq_aborted = True

        if not globals.ls_acq_finished:
            # abort LS acq
            ls_acq_aborted = True
            camera = 'Camera' if self._demo_run else 'Prime BSI Express'
            sequenced_stages = []
            if self.ls_acq.slice_settings.use_sequencing:
                sequenced_stages.append(self.ls_acq.slice_settings.z_stage_name)
            if self.ls_acq.channel_settings.use_sequencing:
                # for now, we don't do channel sequencing on the LS acquisition
                pass
            microscope_operations.abort_acquisition_sequence(
                self.ls_acq.mmc, camera, sequenced_stages
            )
            # set a flag to clear any remaining events
            globals.ls_acq_aborted = True

        return lf_acq_aborted, ls_acq_aborted


def _generate_channel_slice_acq_events(
    channel_settings: ChannelSettings, slice_settings: SliceSettings
):
    events = multi_d_acquisition_events(
        num_time_points=1,
        time_interval_s=0,
        z_start=slice_settings.z_start,
        z_end=slice_settings.z_end,
        z_step=slice_settings.z_step,
        channel_group=channel_settings.channel_group,
        channels=channel_settings.channels,
        channel_exposures_ms=channel_settings.default_exposure_times_ms,
        order="tpcz",
    )

    return events


def _create_acquisition_directory(root_dir: Path, acq_name: str, idx=1) -> Path:
    acq_dir = Path(root_dir) / f'{acq_name}_{idx}'
    # 10000 4 GB files would be 40 TB, which should be plenty
    _ndtif_filename = (
        acq_dir
        / f'{acq_name}_{LS_ACQ_LABEL}_1'
        / f'{acq_name}_{LS_ACQ_LABEL}_NDTiffStack_9999.tif'
    )
    if len(str(_ndtif_filename)) > 255:
        raise ValueError(
            "Path length cannot exceed 255 characters. Please shorten the acquisition name."
        )
    try:
        acq_dir.mkdir(parents=False, exist_ok=False)
    except OSError:
        return _create_acquisition_directory(root_dir, acq_name, idx + 1)
    return acq_dir
