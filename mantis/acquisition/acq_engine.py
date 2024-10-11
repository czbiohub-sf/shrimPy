import logging
import os
import time

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
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
)
from mantis.acquisition.hook_functions.pre_hardware_hook_functions import (
    log_preparing_acquisition,
    lf_pre_hardware_hook_function,
    ls_pre_hardware_hook_function,
)
from mantis.acquisition.hook_functions.post_hardware_hook_functions import (
    log_acquisition_start,
    update_ls_hardware,
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
KIM101_BACKLASH = 10  # backlash correction distance, in steps
VORTRAN_488_COM_PORT = 'COM6'
VORTRAN_561_COM_PORT = 'COM13'
VORTRAN_639_COM_PORT = 'COM12'

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
        self._autoexposure_settings = None
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

    @channel_settings.setter
    def channel_settings(self, settings: ChannelSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._channel_settings = settings
        self._check_num_sequenced_events()

    @slice_settings.setter
    def slice_settings(self, settings: SliceSettings):
        settings_dict = {key: val for key, val in asdict(settings).items() if key != 'z_range'}
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {settings_dict}'
        )
        self._slice_settings = settings
        self._check_num_sequenced_events()

    @microscope_settings.setter
    def microscope_settings(self, settings: MicroscopeSettings):
        logger.debug(
            f'{self.type.capitalize()} acquisition will have the following settings: {asdict(settings)}'
        )
        self._microscope_settings = settings

    @autoexposure_settings.setter
    def autoexposure_settings(self, settings: AutoexposureSettings):
        logger.debug(
            f"{self.type.capitalize()} acquisition will have the following settings:{asdict(settings)}"
        )
        self._autoexposure_settings = settings

    def setup(self):
        """
        Apply acquisition settings as specified by the class properties
        """
        if self.enabled:
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
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._time_settings = settings

    @position_settings.setter
    def position_settings(self, settings: PositionSettings):
        logger.debug(
            f'Mantis acquisition will have the following settings: {asdict(settings)}'
        )
        self._position_settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
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

    def update_position_settings(self):
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
        else:
            logger.debug('Position list is already populated and will not be updated')

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
            return

        # Determine light-sheet acq timing
        ls_readout_time_ms = np.around(
            float(self.ls_acq.mmc.get_property('Prime BSI Express', 'Timing-ReadoutTimeNs'))
            * 1e-6,
            decimals=3,
        )
        _cam_max_fps = int(np.around(1000 / ls_readout_time_ms))
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
        if self.ls_acq.microscope_settings.use_o3_refocus:
            self.ls_acq.o3_stage = microscope_operations.setup_kim101_stage(LS_KIM101_SN)

    def setup_autoexposure(self):
        # assign exposure_times_per_well and laser_powers_per_well to default values
        for well_id in set(self.position_settings.well_ids):
            self.ls_acq.channel_settings.exposure_times_per_well[well_id] = deepcopy(
                self.ls_acq.channel_settings.default_exposure_times_ms
            )
            self.ls_acq.channel_settings.laser_powers_per_well[well_id] = deepcopy(
                self.ls_acq.channel_settings.default_laser_powers
            )

        if self._demo_run:
            logger.debug(
                'Autoexposure is not supported in demo mode. Using default exposure time and laser power'
            )
            return

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
                    self.ls_acq.channel_settings.light_sources[
                        channel_idx
                    ] = microscope_operations.setup_vortran_laser(VORTRAN_488_COM_PORT)
                elif ts2_ttl_state == 64:
                    # State 64 corresponds to illumination with 561 laser
                    self.ls_acq.channel_settings.light_sources[
                        channel_idx
                    ] = microscope_operations.setup_vortran_laser(VORTRAN_561_COM_PORT)
                elif ts2_ttl_state == 128:
                    # State 128 corresponds to illumination with 639 laser
                    self.ls_acq.channel_settings.light_sources[
                        channel_idx
                    ] = microscope_operations.setup_vortran_laser(VORTRAN_639_COM_PORT)
                else:
                    logger.error(
                        'Unknown TTL state {} for channel {} in config group {}'.format(
                            ts2_ttl_state, config_name, config_group
                        )
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

    @staticmethod
    def acquire_ls_defocus_stack(
        mmc: Core,
        z_stage,
        z_range: Iterable,
        galvo: str,
        galvo_range: Iterable,
        config_group: str = None,
        config_name: str = None,
        exposure_time: float = None,
    ):
        """Acquire defocus stacks at different galvo positions and return image data

        Parameters
        ----------
        mmc : Core
        mmStudio : Studio
        z_stage : str or KinesisPiezoMotor
        z_range : Iterable
        galvo : str
        galvo_range : Iterable
        config_group : str, optional
        config_name : str, optional

        Returns
        -------
        data : np.array

        """
        data = []

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

        # get galvo starting position
        p0 = mmc.get_position(galvo)

        # set camera to internal trigger
        # TODO: do this properly, context manager?
        microscope_operations.set_property(
            mmc, 'Prime BSI Express', 'TriggerMode', 'Internal Trigger'
        )

        # acquire stack at different galvo positions
        for p_idx, p in enumerate(galvo_range):
            # set galvo position
            mmc.set_position(galvo, p0 + p)

            # acquire defocus stack
            z_stack = microscope_operations.acquire_defocus_stack(
                mmc,
                z_stage,
                z_range,
                backlash_correction_distance=KIM101_BACKLASH
            )
            data.append(z_stack)

        # Reset camera triggering
        microscope_operations.set_property(
            mmc, 'Prime BSI Express', 'TriggerMode', 'Edge Trigger'
        )

        # Reset galvo
        mmc.set_position(galvo, p0)

        # Reset shutter
        microscope_operations.reset_shutter(mmc, auto_shutter_state, shutter_state)

        return np.asarray(data)

    def refocus_ls_path(self) -> bool:
        logger.info('Running O3 refocus algorithm on light-sheet arm')
        success = False

        # Define O3 z range
        # 1 step is approx 20 nm, 15 steps are 300 nm which is sub-Nyquist sampling
        # The stack starts away from O2 and moves closer
        o3_z_start = -165
        o3_z_end = 165
        o3_z_step = 15
        o3_z_range = np.arange(o3_z_start, o3_z_end + o3_z_step, o3_z_step)

        # Define relative travel limits, in steps
        o3_z_stage = self.ls_acq.o3_stage
        target_z_position = o3_z_stage.true_position + o3_z_range
        max_z_position = np.inf  # O3 is allowed to travel ~15 um towards O2
        min_z_position = -np.inf  # O3 is allowed to travel ~30 um away from O2
        if np.any(target_z_position > max_z_position) or np.any(
            target_z_position < min_z_position
        ):
            logger.error('O3 relative travel limits will be exceeded. Aborting O3 refocus.')
            return

        # Define galvo range, i.e. galvo positions at which O3 defocus stacks
        # are acquired, here at 30%, 50%, and 70% of galvo range. Should be odd number
        galvo_scan_range = self.ls_acq.slice_settings.z_range
        len_galvo_scan_range = len(galvo_scan_range)
        galvo_range = [
            galvo_scan_range[int(0.3 * len_galvo_scan_range)],
            galvo_scan_range[int(0.5 * len_galvo_scan_range)],
            galvo_scan_range[int(0.7 * len_galvo_scan_range)],
        ]

        # Acquire defocus stacks at several galvo positions
        config_group = self.ls_acq.microscope_settings.o3_refocus_config.config_group
        config_name = self.ls_acq.microscope_settings.o3_refocus_config.config_name
        config_idx = self.ls_acq.channel_settings.channels.index(config_name)
        exposure_time = self.ls_acq.channel_settings.default_exposure_times_ms[config_idx]
        
        data = self.acquire_ls_defocus_stack(
            mmc=self.ls_acq.mmc,
            z_stage=o3_z_stage,
            z_range=o3_z_range,
            galvo=self.ls_acq.slice_settings.z_stage_name,
            galvo_range=galvo_range,
            config_group=config_group,
            config_name=config_name,
            exposure_time=exposure_time,
        )

        # Discount O3 backlash compensation from true position count
        o3_z_stage.true_position -= KIM101_BACKLASH * len(galvo_range)

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
        threshold_FWHM = 3.0

        focus_indices = []
        for stack_idx, stack in enumerate(data):
            idx = focus_from_transverse_band(
                stack,
                NA_det=NA_DETECTION,
                lambda_ill=wavelength,
                pixel_size=LS_PIXEL_SIZE,
                threshold_FWHM=threshold_FWHM,
                plot_path=self._logs_dir / f'ls_refocus_plot_{timestamp}_Pos{stack_idx}.png',
            )
            focus_indices.append(idx)
        logger.debug(
            'Stacks at galvo positions %s are in focus at slice %s',
            np.round(galvo_range, 3),
            focus_indices,
        )

        # Refocus O3
        # Some focus_indices may be None, e.g. if there is no sample
        valid_focus_indices = [idx for idx in focus_indices if idx is not None]
        if valid_focus_indices:
            focus_idx = int(np.median(valid_focus_indices))
            o3_displacement = int(o3_z_range[focus_idx])

            logger.info(f'Moving O3 by {o3_displacement} steps')
            microscope_operations.set_relative_kim101_position(
                self.ls_acq.o3_stage, o3_displacement
            )
            success = True
        else:
            logger.error(
                'Could not determine the correct O3 in-focus position. O3 will not move'
            )

        return success

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
        self.update_position_settings()

        logger.debug('Setting up autoexposure')
        self.setup_autoexposure()

    def acquire(self):
        """
        Simultaneously acquire label-free and light-sheet data over multiple
        positions and time points.
        """

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
        lf_image_saved_fn = check_lf_acq_finished

        # define LF acquisition
        self._lf_acq_obj = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_labelfree',
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
        ls_image_saved_fn = check_ls_acq_finished

        # define LS acquisition
        self._ls_acq_obj = Acquisition(
            directory=self._acq_dir,
            name=f'{self._acq_name}_lightsheet',
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
                if p_label != previous_position_label:
                    self.go_to_position(p_idx)

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
                        success = self.refocus_ls_path()
                        if success:
                            ls_o3_refocus_time = current_time

                # autoexposure
                if well_id != previous_well_id:
                    globals.new_well = True
                    if t_idx == 0 or self.ls_acq.autoexposure_settings.rerun_each_timepoint:
                        self.run_autoexposure(
                            acq=self.ls_acq,
                            well_id=well_id,
                            method=self.ls_acq.autoexposure_settings.autoexposure_method,
                        )
                    # Acq rate needs to be updated even if autoexposure was not rerun in this well
                    # Only do that if we are using autoexposure?
                    self.update_ls_acquisition_rates(
                        self.ls_acq.channel_settings.exposure_times_per_well[well_id]
                    )
                    globals.ls_slice_acquisition_rates = (
                        self.ls_acq.slice_settings.acquisition_rate
                    )
                    globals.ls_laser_powers = (
                        self.ls_acq.channel_settings.laser_powers_per_well[well_id]
                    )

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
                        _event[
                            'exposure'
                        ] = self.ls_acq.channel_settings.exposure_times_per_well[well_id][
                            channel_index
                        ]

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
        buffer_time = 5
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
    acq_dir = root_dir / f'{acq_name}_{idx}'
    try:
        acq_dir.mkdir(parents=False, exist_ok=False)
    except OSError:
        return _create_acquisition_directory(root_dir, acq_name, idx + 1)
    return acq_dir
