# %% on Dragonfly microscope works with pycromanager==0.19.2 and ndtiff==1.3.9
import logging
import time
from pathlib import Path
import json
from datetime import datetime
import re

import numpy as np

from pycromanager import Acquisition, Core, multi_d_acquisition_events

# from mantis import get_console_handler
# from mantis.acquisition import microscope_operations
# from mantis.acquisition.acq_engine import _create_acquisition_directory
# from mantis.acquisition.logger import configure_debug_logger

def _create_acquisition_directory(root_dir: Path, acq_name: str, idx=1) -> Path:
    assert root_dir.exists(), f'Root directory {root_dir} does not exist'
    acq_dir = root_dir / f'{acq_name}_{idx}'
    try:
        acq_dir.mkdir(parents=False, exist_ok=False)
    except OSError:
        return _create_acquisition_directory(root_dir, acq_name, idx + 1)
    return acq_dir

USE_HW_SEQUENCING = False
DEBUG = False
PIEZO_STEP_TIME_S = 0.05
mmc = Core()

acquisition_directory = Path(r'G:\OPS')
acquisition_name = 'OPS0061'
start_time = '2025-07-18 01:00:00'
# start_time = 'now'
well_diameter = 35000  # in um, 6 well plates have 35 mm diameter wells
min_fov_distance_from_well_edge = 800  # in um
#TODO:uncomment this after this acquisition -EH
well_centers = {
    'A1': (-35214, -20986, 6299),
    'A2': (4126, -20986, 6312),
    'A3': (43466, -20986, 6396),
}  # (x, y, z) in um

phenotyping_magnification = 20
tracking_magnification = 5
percent_overlap = 8

image_size = (2048, 2048)
pixel_size = 6.5  # in um

phenotyping_channel_group = 'Channels'
phenotyping_channel = '5-MultiCam_GFP_mCherry_BF'
# phenotyping_channel = '4-MultiCam_CL488_BF'
# phenotyping_channel = '4-MultiCam_mCherry_BF'
# phenotyping_channel = '1-Zyla_BF'
tracking_channel_group = 'Channels'
tracking_channel = '1-Zyla_BF'

# Define same exposure for all channels.
# Not absolutely necessary, but helpful in debugging issues with BF exposure
exposure_time = 100 # in ms

z_start = 0
z_end = 12
z_step = 2.0

if phenotyping_channel[0] in ('1', '2', '3'):
    num_phenotyping_channel = 1
elif phenotyping_channel[0] == '4':
    num_phenotyping_channel = 2
elif phenotyping_channel[0] == '5':
    num_phenotyping_channel = 3
else:
    if not DEBUG:
        raise ValueError('Unknown number of phenotyping channels')

if start_time == 'now':
    start_time_obj = datetime.now()
else:
    start_time_obj = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

if DEBUG:
    phenotyping_channel_group = 'Channel'
    phenotyping_channel = 'DAPI'
    tracking_channel_group = 'Channel'
    tracking_channel = 'FITC'
    num_phenotyping_channel = 1
    USE_HW_SEQUENCING = False  # HW Sequencing is not tested in debug mode

# Setup Dragonfly microscope
if not DEBUG:
    # Make sure field stop is opened all the way
    field_diaphragm = mmc.get_property('TL-FieldDiaphragm', 'Position')
    if int(field_diaphragm) != 46:
        raise ValueError('Please set the field diaphragm to 46 and adjust the brightfield illumnation as needed.')


def change_magnification_phenotyping():
    mmc.set_config(phenotyping_channel_group, phenotyping_channel)
    mmc.set_exposure(exposure_time)

    if DEBUG:
        mmc.set_config('Objective', '20X')
    else:
        if USE_HW_SEQUENCING:
            mmc.set_property('Core', 'Focus', 'TS_PiezoZ')
        else:
            mmc.set_property('Core', 'Focus', 'PiezoZ')
        mmc.set_property('ObjectiveTurret', 'Label', '3-20x'); time.sleep(5)
        mmc.set_property('TL-ApertureDiaphragm', 'Position', '24')
        # turn AFC back on
        mmc.set_property('Adaptive Focus Control', 'DichroicMirrorIn', '1'); time.sleep(5)
        


def change_magnification_tracking():
    mmc.set_config(tracking_channel_group, tracking_channel)
    mmc.set_exposure(exposure_time)

    if DEBUG:
        mmc.set_config('Objective', '10X')
    else:
        mmc.set_property('Core', 'Focus', 'FocusDrive')
        mmc.set_property('ObjectiveTurret', 'Label', '1-5x'); time.sleep(5)
        mmc.set_property('TL-ApertureDiaphragm', 'Position', '4')


def setup_hw_sequencing():
        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Setting up for hardware sequencing')

        # Setup Z Stage
        # Set PiezoZ to external input; TODO: should we do this here?
        mmc.set_property('XYStage', 'SerialCommand', 'PZ Z=1'); time.sleep(2)
        mmc.set_property('TS_DAC01', 'Sequence', 'On')
        mmc.set_position('TS_PiezoZ', z_start)

        # Setup Zyla camera
        # turn off overlapping readout to be able to set framerate inpedendently
        mmc.set_property('Zyla', 'AuxiliaryOutSource (TTL I/O)', 'FireAny')
        mmc.set_property('Zyla', 'Overlap', 'Off')
        max_framerate_str = mmc.get_property('Zyla', 'FrameRateLimits')
        match = re.search(r'Max:\s*([\d.]+)', max_framerate_str)
        if match:
            max_framerate = float(match.group(1))
        else:
            raise RuntimeError('Could not determine max Zyla framerate')
        framerate = 1 / (1/max_framerate + PIEZO_STEP_TIME_S)
        mmc.set_property('Zyla', 'FrameRate', framerate)

        # Setup Prime BSI Express camera
        mmc.set_property('BSI_Express', 'ExposeOutMode', 'Any Row')
        mmc.set_property('BSI_Express', 'TriggerMode', 'Edge Trigger')

        # Setup Prime BSI camera
        mmc.set_property('Prime', 'ExposeOutMode', 'Any Row')
        mmc.set_property('Prime', 'TriggerMode', 'Edge Trigger')


def reset_hw_sequencing():
        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Resetting after hardware sequencing')

        # Reset Z Stage
        mmc.set_property('XYStage', 'SerialCommand', 'PZ Z=0'); time.sleep(2)
        mmc.set_property('TS_DAC01', 'Sequence', 'Off')

        # Reset Zyla camera
        mmc.set_property('Zyla', 'AuxiliaryOutSource (TTL I/O)', 'FireAll')
        mmc.set_property('Zyla', 'Overlap', 'On')

        # Reset Prime BSI Express camera
        mmc.set_property('BSI_Express', 'ExposeOutMode', 'Rolling Shutter')
        mmc.set_property('BSI_Express', 'TriggerMode', 'Internal Trigger')

        # Reset Prime BSI camera
        mmc.set_property('Prime', 'ExposeOutMode', 'Rolling Shutter')
        mmc.set_property('Prime', 'TriggerMode', 'Internal Trigger')



# %% Setup logger and acquisition directory=
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# acq_dir = _create_acquisition_directory(acquisition_directory, acquisition_name)
# logs_dir = acq_dir / 'logs'
# logs_dir.mkdir()

# # Setup logger
# logger = logging.getLogger('OPS')
# logger.setLevel(logging.DEBUG)

# logger.addHandler(get_console_handler())
# logger.propagate = False

# timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
# configure_debug_logger(logs_dir / f'ops_acquisition_{timestamp}.txt', logger_name='OPS')


# %% Setup position grid
def make_position_grid(
    well_label,
    well_diameter,
    well_center,
    fov_size,
    percent_overlap,
    min_fov_distance_from_well_edge,
):
    """
    TODO: test with non-square FOVs
    """
    x_step = fov_size[-1] * (1 - percent_overlap / 100)
    y_step = fov_size[-2] * (1 - percent_overlap / 100)
    x0 = (well_diameter % x_step) / 2 - fov_size[-1] / 2
    y0 = (well_diameter % y_step) / 2 - fov_size[-2] / 2
    x_coords = np.arange(x0, well_diameter, x_step)
    y_coords = np.arange(y0, well_diameter, y_step)
    x, y = np.meshgrid(x_coords, y_coords, indexing='xy')

    position_list = np.stack(
        (
            x + well_center[0] - well_diameter / 2,
            y + well_center[1] - well_diameter / 2,
            np.ones_like(x) * well_center[2],
        ),
        axis=-1,
    ).round(decimals=2)

    position_labels = np.asarray(
        [
            [f"{well_label}-Site_{_x:03}{_y:03}" for _x in range(x.shape[0])]
            for _y in range(x.shape[1])
        ]
    )

    # make snake pattern
    for i in range(1, position_list.shape[0], 2):
        position_list[i] = position_list[i][::-1]
        position_labels[i] = position_labels[i][::-1]

    # make 1D
    position_list = position_list.reshape(-1, 3)
    position_labels = position_labels.flatten()

    # remove positions that are too close to the well edge
    if min_fov_distance_from_well_edge:
        fov_distance_from_center = np.sqrt(
            ((position_list - np.asarray(well_center)) ** 2).sum(axis=1)
        )
        fovs_to_remove = fov_distance_from_center > (
            well_diameter / 2 - min_fov_distance_from_well_edge
        )
        position_list = position_list[~fovs_to_remove]
        position_labels = position_labels[~fovs_to_remove]

    return position_list.tolist(), position_labels.tolist()


pheno_fov_size = np.asarray(image_size) * pixel_size / phenotyping_magnification
tracking_fov_size = np.asarray(image_size) * pixel_size / tracking_magnification

pheno_position_list = []
pheno_position_labels = []
for well_name, well_coords in well_centers.items():
    _position_list, _position_labels = make_position_grid(
        well_name,
        well_diameter,
        well_coords,
        pheno_fov_size,
        percent_overlap,
        min_fov_distance_from_well_edge,
    )
    pheno_position_list.extend(_position_list)
    pheno_position_labels.extend(_position_labels)

tracking_position_list = []
tracking_position_labels = []
for well_name, well_coords in well_centers.items():
    _position_list, _position_labels = make_position_grid(
        well_name,
        well_diameter,
        well_coords,
        tracking_fov_size,
        percent_overlap,
        min_fov_distance_from_well_edge,
    )
    tracking_position_list.extend(_position_list)
    tracking_position_labels.extend(_position_labels)

# %%
if USE_HW_SEQUENCING:
    logger.warning(
        "WARNING: This acquisition will use hardware sequencing. "
        "This is an experimental feature."
    )

acq_dir = _create_acquisition_directory(acquisition_directory, acquisition_name)
num_wells = len(well_centers)
num_positions_per_well = len(pheno_position_list) // num_wells

def check_acq_finished(axes, dataset):
    global acq_finished
    if axes == last_img_idx:
        acq_finished = True

def autofocus_hook_fn(events):
    if isinstance(events, list):
        _event = events[0]
    else:
        _event = events  # events is a dict

    if _event['axes']['z'] == 0:
        try:
            mmc.full_focus()
        except Exception:
            logger.error('Autofocus failed')
    return events

with open(acq_dir / 'tracking_position_list.json', 'w') as fp:
    json.dump(dict(zip(tracking_position_labels, tracking_position_list)), fp, indent=4)

with open(acq_dir / 'pheno_position_list.json', 'w') as fp:
    json.dump(dict(zip(pheno_position_labels, pheno_position_list)), fp, indent=4)

start_delay_s = (start_time_obj - datetime.now()).total_seconds()
if start_delay_s > 0:
    logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Waiting {int(start_delay_s)} seconds until {start_time}')
    time.sleep(start_delay_s)

tracking_events = multi_d_acquisition_events(
    z_start=-100,
    z_end=100,
    z_step=25,
    xyz_positions=tracking_position_list,
    keep_shutter_open_between_z_steps=True,
    # position_labels=tracking_position_labels,
)
phenotyping_events = []
pheno_position_list = np.asarray(pheno_position_list)
for i in range(len(well_centers)):
    _events = multi_d_acquisition_events(
        z_start=z_start,
        z_end=z_end,
        z_step=z_step,
        xy_positions=pheno_position_list[
            i * num_positions_per_well : (i + 1) * num_positions_per_well, :2
        ],
        keep_shutter_open_between_z_steps=True,
    )
    phenotyping_events.append(_events)

try:
    if USE_HW_SEQUENCING:
        setup_hw_sequencing()
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Starting acquisition')
    pheno_position_list = np.asarray(pheno_position_list)
    for i, well_name in enumerate(well_centers.keys()):
        # Track cells across all wells
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'{timestamp} Changing magnification for tracking')
        change_magnification_tracking()
        n_z_steps = len(np.arange(z_start, z_end+z_step, z_step))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'{timestamp} Acquiring tracking acquisition')
        with Acquisition(directory=str(acq_dir), name='tracking') as acq:
            acq.acquire(tracking_events)

        # Phenotype cells in this well
        acq_finished = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'{timestamp} Changing magnification for phenotyping')
        change_magnification_phenotyping()
        events = phenotyping_events[i]
        last_img_idx = dict(events[-1]['axes'])
        last_img_idx.update({'channel': num_phenotyping_channel-1})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'{timestamp} Acquiring phenotyping at well {well_name}')
        acq = Acquisition(
            directory=str(acq_dir),
            name=f'phenotyping_well_{well_name}',
            post_hardware_hook_fn=autofocus_hook_fn,
            image_saved_fn=check_acq_finished,
        )
        acq.acquire(events)
        acq.mark_finished()
        while not acq_finished:
            time.sleep(1)
        acq.await_completion()

    # Track cells once more at the end
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Changing magnification for tracking')
    change_magnification_tracking()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Acquiring tracking acquisition')
    with Acquisition(directory=str(acq_dir), name='tracking') as acq:
        acq.acquire(tracking_events)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Acquisition finished')

finally:
    if USE_HW_SEQUENCING:
        reset_hw_sequencing()

## %%
