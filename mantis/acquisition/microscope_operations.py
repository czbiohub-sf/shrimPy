import logging
import time
import numpy as np
from pycromanager import Core, Studio

from typing import Tuple, Iterable
from functools import partial

import nidaqmx

from nidaqmx.constants import AcquisitionType

from copylot.hardware.stages.abstract_stage import AbstractStage
from copylot.hardware.stages.thorlabs.KIM001 import KCube_PiezoInertia

logger = logging.getLogger(__name__)


def _try_mmc_call(mmc, mmc_call_name, *mmc_carr_args):
    """Wrapper that tries to repeat calls to mmCore if they fail. Largely copied
    from dragonfly_automation MicromanagerInterface

    """
    num_mm_call_tries = 3
    wait_time_between_mm_calls = 5  # in seconds
    call_succeeded = False
    error_occurred = False
    result = None

    for _ in range(num_mm_call_tries):
        try:
            result = getattr(mmc, mmc_call_name)(*mmc_carr_args)
            call_succeeded = True
            break
        except Exception as e:
            error_occurred = True
            msg = str(e).split("\n")[0]
            logger.error(f'An error occurred calling method {mmc_call_name}: {msg}')
            time.sleep(wait_time_between_mm_calls)

    if error_occurred and call_succeeded:
        logger.debug(f'The call to method {mmc_call_name} succeeded on a subsequent attempt')

    if not call_succeeded:
        message = f'Call to method {mmc_call_name} failed after {num_mm_call_tries} tries'
        logger.critical(message)
        raise Exception(message)

    return result


def set_config(mmc, config_group, config_name):
    logger.debug(f'Setting {config_group} config group to {config_name}')

    mmc.set_config(config_group, config_name)


def set_property(mmc, device_name, property_name, property_value):
    logger.debug(f'Setting {device_name} {property_name} to {property_value}')

    mmc.set_property(device_name, property_name, property_value)

    if 'Line Selector' in property_name:
        mmc.update_system_state_cache()


def set_roi(mmc, roi: tuple):
    logger.debug(f'Setting ROI to {roi}')

    mmc.set_roi(*roi)


def get_position_list(mmStudio, z_stage_name):
    mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
    number_of_positions = mm_pos_list.get_number_of_positions()

    xyz_position_list = []
    position_labels = []
    for i in range(number_of_positions):
        _pos = mm_pos_list.get_position(i)
        xyz_position_list.append(
            [
                _pos.get_x(),
                _pos.get_y(),
                _pos.get(z_stage_name).get1_d_position() if z_stage_name else None,
            ]
        )
        position_labels.append(_pos.get_label())

    return xyz_position_list, position_labels


def set_z_position(mmc, z_stage_name: str, z_position: float):
    _try_mmc_call(mmc, 'set_position', str(z_stage_name), float(z_position))


def set_relative_z_position(mmc, z_stage_name: str, z_offset: float):
    _try_mmc_call(mmc, 'set_relative_position', str(z_stage_name), float(z_offset))


def set_xy_position(mmc, xy_position: Tuple[float, float]):
    _try_mmc_call(mmc, 'set_xy_position', float(xy_position[0]), float(xy_position[1]))


def set_relative_xy_position(mmc, xy_offset: Tuple[float, float]):
    _try_mmc_call(mmc, 'set_relative_xy_position', float(xy_offset[0]), float(xy_offset[1]))


def wait_for_device(mmc, device_name: str):
    _try_mmc_call(
        mmc,
        'wait_for_device',
        str(device_name),
    )


def setup_daq_counter(
    task: nidaqmx.Task, co_channel, freq, duty_cycle, samples_per_channel, pulse_terminal
):

    logger.debug(f'Setting up {task.name} on {co_channel}')
    logger.debug(
        f'{co_channel} will output {samples_per_channel} samples with {duty_cycle} duty cycle at {freq:.6f} Hz on terminal {pulse_terminal}'
    )

    ctr = task.co_channels.add_co_pulse_chan_freq(co_channel, freq=freq, duty_cycle=duty_cycle)
    task.timing.cfg_implicit_timing(
        sample_mode=AcquisitionType.FINITE, samps_per_chan=samples_per_channel
    )
    ctr.co_pulse_term = pulse_terminal

    return ctr


def get_daq_counter_names(CtrTask: nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    ctr_names = []
    for _task in CtrTask:
        ctr_names.append(_task.name)

    return ctr_names


def start_daq_counter(CtrTask: nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    for _task in CtrTask:
        if _task.is_task_done():
            _task.stop()  # Counter needs to be stopped before it is restarted
            _task.start()


def get_total_num_daq_counter_samples(CtrTask: nidaqmx.Task or list):
    if not isinstance(CtrTask, list):
        CtrTask = [CtrTask]

    num_counter_samples = 1
    for _task in CtrTask:
        num_counter_samples *= _task.timing.samp_quant_samp_per_chan

    return num_counter_samples


def autofocus(mmc, mmStudio, z_stage_name: str, z_position):
    """
    Attempt to engage Nikon PFS continuous autofocus. This function will log a
    message and continue if continuous autofocus is already engaged. Otherwise,
    it will attempt to engage autofocus, moving the z stage by amounts given in
    `z_offsets`, if necessary.

    Nikon PFS status codes:
        0000000000000000 - Off and out of range?
        0000000100000000 - Off
        0000001100001001 - In Range
        0000001100001010 - In range and engaged?
        0000001000011001 - Out of Range
        0010001000001001 - ?

    Returns
    -------
    bool
        True if continuous autofocus successfully engaged, False otherwise.
    """
    logger.debug('Engaging autofocus')
    autofocus_success = False
    error_occurred = False

    af_method = mmStudio.get_autofocus_manager().get_autofocus_method()
    z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

    # Turn on autofocus if it has been turned off. This call has no effect is
    # continuous autofocus is already engaged
    try:
        af_method.full_focus()
    except Exception:
        logger.debug('Call to full_focus() method failed')
    else:
        logger.debug('Call to full_focus() method succeeded')

    if af_method.is_continuous_focus_locked():  # True if autofocus is engaged
        autofocus_success = True
        logger.debug('Continuous autofocus is already engaged')
    else:
        for z_offset in z_offsets:
            mmc.set_position(z_stage_name, z_position + z_offset)
            mmc.wait_for_device(z_stage_name)
            time.sleep(1)  # wait an extra second

            af_method.enable_continuous_focus(True)  # this call engages autofocus
            if af_method.is_continuous_focus_locked():
                autofocus_success = True
                break
            else:
                error_occurred = True
                logger.debug(f'Autofocus call failed with z offset of {z_offset} um')

    if error_occurred and autofocus_success:
        logger.debug(f'Continuous autofocus call succeeded with z offset of {z_offset} um')

    if not autofocus_success:
        logger.error(f'Autofocus call failed after {len(z_offsets)} tries')

    return autofocus_success

def setup_kim101_stage(serial_number: int, step_rate = 500, step_acceleration = 1000):
    stage = KCube_PiezoInertia(serial_number=str(serial_number))

    # Change the acceleration and step rate
    stage.step_rate = step_rate
    stage.step_acceleration = step_acceleration

    return stage

def create_ram_datastore(
    mmStudio: Studio,
):
    datastore = mmStudio.get_data_manager().create_ram_datastore()
    mmStudio.get_display_manager().create_display(datastore)

    return datastore

def acquire_defocus_stack(
    mmc: Core,
    mmStudio: Studio,
    datastore,
    z_stage,
    z_range: Iterable,
    channel_ind: int=0,
    position_ind: int=0,
):
    """Snap image at every z position and put image in the datastore

    Parameters
    ----------
    mmc : Core
    mmStudio : Studio
    datastore : micromanager.data.Datastore
        Micro-manager datastore object
    z_stage : str or coPylot stage object
    z_range : Iterable
    channel_ind : int, optional
        Channel index of acquired images in the Micro-manager datastore, by default 0
    position_ind : int, optional
        Position index of acquired images in the Micro-manager datastore, by default 0

    Returns
    -------
    data : np.array

    """
    data = []

    if isinstance(z_stage, str):
        # this is a MM stage
        z0 = mmc.get_position(z_stage)
        move_z = partial(mmc.set_position, z_stage)  # test if this works
    elif issubclass(type(z_stage), AbstractStage):
        # this is a copylot stage
        z0 = z_stage.position
        move_z = z_stage.move_absolute
    else:
        raise RuntimeError(f'Unknown z stage: {z_stage}')

    for z_ind, z in enumerate(z_range):
        # set z position
        move_z(z0 + z)

        # snap image
        mmc.snap_image()
        tagged_image = mmc.get_tagged_image()

        # get image data
        image_data = np.reshape(
            tagged_image.pix, 
            (tagged_image.tags['Height'], tagged_image.tags['Width'])
        )
        data.append(image_data.astype('uint16'))

        # set image coordinates and put in datastore
        image = mmStudio.get_data_manager().convert_tagged_image(tagged_image)
        coords_builder = image.get_coords().copy()
        coords_builder = coords_builder.z(z_ind)
        coords_builder = coords_builder.channel(channel_ind)
        coords_builder = coords_builder.stage_position(position_ind)
        mm_coords = coords_builder.build()

        image = image.copy_at_coords(mm_coords)
        datastore.put_image(image)

    # reset z stage
    move_z(z0)
    
    return np.asarray(data)

def acquire_ls_defocus_stack(
    mmc: Core,
    mmStudio: Studio,
    z_stage,
    z_start: float, 
    z_end: float, 
    z_step: float,
    config_group: str=None, 
    config_name: str=None,
):
    """Acquire defocus stacks at different galvo positions

    Parameters
    ----------
    mmc : Core
        _description_
    mmStudio : Studio
        _description_
    z_stage : _type_
        _description_
    z_start : float
        _description_
    z_end : float
        _description_
    z_step : float
        _description_
    config_group : str, optional
        _description_, by default None
    config_name : str, optional
        _description_, by default None

    Returns
    -------
    data : np.array

    """
    datastore = create_ram_datastore(mmStudio)
    z_range = np.arange(z_start, z_end+z_step, z_step)
    data = []

    # Set config
    if config_name is not None:
        mmc.set_config(config_group, config_name)
        mmc.wait_for_config(config_group, config_name)

    # Open shutter
    auto_shutter_state, shutter_state = get_shutter_state(mmc)
    open_shutter(mmc)

    # acquire stack at different galvo positions
    for p_idx, p in enumerate(range(3)):
        # TODO: set galvo position

        # acquire defocus stack
        z_stack = acquire_defocus_stack(
            mmc, mmStudio, datastore, z_stage, z_range, channel_ind=0, position_ind=p_idx
        )
        data.append(z_stack)

    # freeze datastore to indicate that we are finished writing to it
    datastore.freeze()

    # Reset shutter
    reset_shutter(mmc, auto_shutter_state, shutter_state)

    return np.asarray(data)

def get_shutter_state(mmc: Core):
    """Return the current state of the shutter

    Parameters
    ----------
    mmc : Core

    Returns
    -------
    auto_shutter_state : bool
    shutter_state : bool

    """
    auto_shutter_state = mmc.get_auto_shutter()
    shutter_state = mmc.get_shutter_open()

    return auto_shutter_state, shutter_state

def open_shutter(mmc: Core):
    """Open shutter if mechanical shutter exists

    Parameters
    ----------
    mmc : Core

    """

    if mmc.get_shutter_device():
        mmc.set_shutter_open(True)

def reset_shutter(mmc: Core, auto_shutter_state: bool, shutter_state: bool):
    """Reset shutter if mechanical shutter exists

    Parameters
    ----------
    mmc : Core
    auto_shutter_state : bool
    shutter_state : bool

    """

    if mmc.get_shutter_device():
        mmc.set_auto_shutter(auto_shutter_state)
        mmc.set_shutter_open(shutter_state)
