# %%
import logging

from datetime import datetime
from pathlib import Path

import numpy as np

from pycromanager import Acquisition, Core, multi_d_acquisition_events

from mantis import get_console_handler
from mantis.acquisition import microscope_operations
from mantis.acquisition.acq_engine import _create_acquisition_directory
from mantis.acquisition.logger import configure_debug_logger

DEBUG = True
mmc = Core()

acquisition_directory = Path('/Users/ivan.ivanov/Documents/images_local/OPS/')
acquisition_name = 'test_acq'

well_diameter = 35000  # in um, 6 well plates have 35 mm diameter wells
min_fov_distance_from_well_edge = 800  # in um
well_centers = [
    (25000, 23000, 0),
    (64000, 23000, 0),
    (103000, 23000, 0),
]  # (x, y, z) in um

phenotyping_magnification = 20
tracking_magnification = 5
percent_overlap = 5

image_size = (2048, 2048)
pixel_size = 6.5  # in um

phenotyping_channel_group = 'Channel'
phenotyping_channel = 'DAPI'
tracking_channel_group = 'Channel'
tracking_channel = 'FITC'

z_start = 0
z_end = 5
z_step = 1

if DEBUG:
    phenotyping_channel_group = 'Channel'
    phenotyping_channel = 'DAPI'
    tracking_channel_group = 'Channel'
    tracking_channel = 'FITC'


def change_magnification_phenotyping():
    if DEBUG:
        microscope_operations.set_config(mmc, 'Objective', '20X')


def change_magnification_tracking():
    if DEBUG:
        microscope_operations.set_config(mmc, 'Objective', '10X')


# %% Setup acquisition directory
acq_dir = _create_acquisition_directory(acquisition_directory, acquisition_name)
logs_dir = acq_dir / 'logs'
logs_dir.mkdir()

# Setup logger
logger = logging.getLogger('OPS')
logger.setLevel(logging.DEBUG)

logger.addHandler(get_console_handler())
logger.propagate = False

timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
configure_debug_logger(logs_dir / f'ops_acquisition_{timestamp}.txt', logger_name='OPS')


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
            [f"{well_label}-Site_{j:03}{i:03}" for i in range(x.shape[0])]
            for j in range(x.shape[1])
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
for well_number, well_coords in enumerate(well_centers):
    _position_list, _position_labels = make_position_grid(
        str(well_number + 1),
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
for well_number, well_coords in enumerate(well_centers):
    _position_list, _position_labels = make_position_grid(
        str(well_number + 1),
        well_diameter,
        well_coords,
        tracking_fov_size,
        percent_overlap,
        min_fov_distance_from_well_edge,
    )
    tracking_position_list.extend(_position_list)
    tracking_position_labels.extend(_position_labels)

# %%
num_wells = len(well_centers)
num_positions_per_well = len(pheno_position_list) // num_wells

for i in range(num_wells):
    # Track cells across all wells
    logger.info('Changing magnification for tracking')
    change_magnification_tracking()
    events = multi_d_acquisition_events(
        z_start=z_start,
        z_end=z_end,
        z_step=z_step,
        channel_group=tracking_channel_group,
        channels=[tracking_channel],
        xyz_positions=tracking_position_list,
        position_labels=tracking_position_labels,
    )
    for event in events:
        event['axes']['time'] = i
    logger.info('Acquiring tracking acquisition')
    with Acquisition(directory=str(acq_dir), name='tracking', show_display=False) as acq:
        acq.acquire(events)

    # Phenotype cells in this well
    logger.info('Changing magnification for phenotyping')
    change_magnification_phenotyping()
    events = multi_d_acquisition_events(
        z_start=z_start,
        z_end=z_end,
        z_step=z_step,
        channel_group=phenotyping_channel_group,
        channels=[phenotyping_channel],
        xyz_positions=pheno_position_list[
            i * num_positions_per_well : (i + 1) * num_positions_per_well
        ],
        position_labels=pheno_position_labels[
            i * num_positions_per_well : (i + 1) * num_positions_per_well
        ],
    )
    logger.info(f'Acquiring phenotyping at well {i+1}')
    with Acquisition(
        directory=str(acq_dir), name=f'phenotyping_well_{i+1}', show_display=False
    ) as acq:
        acq.acquire(events)

# Track cells once more at the end
logger.info('Changing magnification for tracking')
change_magnification_tracking()
events = multi_d_acquisition_events(
    z_start=z_start,
    z_end=z_end,
    z_step=z_step,
    channel_group=tracking_channel_group,
    channels=[tracking_channel],
    xyz_positions=tracking_position_list,
    position_labels=tracking_position_labels,
)
for event in events:
    event['axes']['time'] = num_wells
logger.info('Acquiring tracking acquisition')
with Acquisition(directory=str(acq_dir), name='tracking', show_display=False) as acq:
    acq.acquire(events)

# %%
