# %%
import logging
import time
from pathlib import Path
import json
from datetime import datetime

import numpy as np

from pycromanager import Acquisition, Core, multi_d_acquisition_events

# from mantis import get_console_handler
# from mantis.acquisition import microscope_operations
# from mantis.acquisition.acq_engine import _create_acquisition_directory
# from mantis.acquisition.logger import configure_debug_logger

def _create_acquisition_directory(root_dir: Path, acq_name: str, idx=1) -> Path:
    acq_dir = root_dir / f'{acq_name}_{idx}'
    try:
        acq_dir.mkdir(parents=False, exist_ok=False)
    except OSError:
        return _create_acquisition_directory(root_dir, acq_name, idx + 1)
    return acq_dir

DEBUG = False
mmc = Core()

acquisition_directory = Path(r'G:\OPS\OPS0003')
acquisition_name = 'OPS0003'

well_diameter = 35000  # in um, 6 well plates have 35 mm diameter wells
min_fov_distance_from_well_edge = 800  # in um
well_centers = {
    'A1': (10100, 11600, 6336),
    'A2': (49400, 11600, 6394),
    'A3': (88600, 11600, 6488),
}  # (x, y, z) in um

phenotyping_magnification = 20
tracking_magnification = 5
percent_overlap = 8

image_size = (2048, 2048)
pixel_size = 6.5  # in um

phenotyping_channel_group = 'Channels'
phenotyping_channel = 'MultiCam_GFP_mCherry_BF_narrow'
tracking_channel_group = 'Channels'
tracking_channel = 'Zyla_BF'

z_start = -2
z_end = 6
z_step = 2

if DEBUG:
    phenotyping_channel_group = 'Channel'
    phenotyping_channel = 'DAPI'
    tracking_channel_group = 'Channel'
    tracking_channel = 'FITC'

# Setup Dragonfly microscope
if not DEBUG:
    # Open field stop all the way
    mmc.set_property('TL-FieldDiaphragm', 'Position', '46')


def change_magnification_phenotyping():
    if DEBUG:
        mmc.set_config('Objective', '20X')
    else:
        mmc.set_property('Core', 'Focus', 'PiezoZ')
        mmc.set_property('Zyla', 'Binning', '2x2')
        mmc.set_property('ObjectiveTurret', 'Label', '3-20x'); time.sleep(5)
        mmc.set_property('TL-ApertureDiaphragm', 'Position', '24')
        # turn AFC back on
        mmc.set_property('Adaptive Focus Control', 'DichroicMirrorIn', '1'); time.sleep(5)
        


def change_magnification_tracking():
    if DEBUG:
        mmc.set_config('Objective', '10X')
    else:
        mmc.set_property('Core', 'Focus', 'FocusDrive')
        mmc.set_property('Zyla', 'Binning', '1x1')
        mmc.set_property('ObjectiveTurret', 'Label', '1-5x'); time.sleep(5)
        mmc.set_property('TL-ApertureDiaphragm', 'Position', '4')


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

    # make snake pattern
    for i in range(1, x.shape[0], 2):
        x[i] = x[i, ::-1]

    position_list = (
        np.vstack(
            [
                x.flatten() + well_center[0] - well_diameter / 2,
                y.flatten() + well_center[1] - well_diameter / 2,
                np.ones(x.size) * well_center[2],
            ]
        )
        .round(decimals=2)
        .T
    )
    position_labels = np.asarray(
        [
            f'{well_label}-Site_{j:03}{i:03}'
            for i in range(x.shape[0])
            for j in range(x.shape[1])
        ]
    )

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

pheno_position_list = np.asarray(pheno_position_list)
for i, well_name in enumerate(well_centers.keys()):
    # Track cells across all wells
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Changing magnification for tracking')
    change_magnification_tracking()
    events = multi_d_acquisition_events(
        z_start=-50,
        z_end=50,
        z_step=25,
        channel_group=tracking_channel_group,
        channels=[tracking_channel],
        xyz_positions=tracking_position_list,
        # position_labels=tracking_position_labels,
    )
    n_z_steps = len(np.arange(z_start, z_end+z_step, z_step))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Acquiring tracking acquisition')
    with Acquisition(directory=str(acq_dir), name='tracking') as acq:
        acq.acquire(events)

    # Phenotype cells in this well
    acq_finished = False
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'{timestamp} Changing magnification for phenotyping')
    change_magnification_phenotyping()
    events = multi_d_acquisition_events(
        z_start=z_start,
        z_end=z_end,
        z_step=z_step,
        channel_group=phenotyping_channel_group,
        channels=[phenotyping_channel],
        xy_positions=pheno_position_list[
            i * num_positions_per_well : (i + 1) * num_positions_per_well, :2
        ],
        keep_shutter_open_between_z_steps=True,
    )
    last_img_idx = dict(events[-1]['axes'])
    last_img_idx.update({'channel': 2})
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
events = multi_d_acquisition_events(
    z_start=-50,
    z_end=50,
    z_step=25,
    channel_group=tracking_channel_group,
    channels=[tracking_channel],
    xyz_positions=tracking_position_list,
    # position_labels=tracking_position_labels,
)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info(f'{timestamp} Acquiring tracking acquisition')
with Acquisition(directory=str(acq_dir), name='tracking') as acq:
    acq.acquire(events)

# %%
