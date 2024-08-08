# %%
import time

from pathlib import Path

import napari
import numpy as np
import torch

from iohub.ngff_meta import TransformationMeta
from iohub.reader import open_ome_zarr
from pycromanager import Acquisition, Core, multi_d_acquisition_events

from mantis.acquisition.microscope_operations import acquire_defocus_stack
from mantis.analysis.AnalysisSettings import CharacterizeSettings
from mantis.analysis.deskew import deskew_data, get_deskewed_data_shape
from mantis.cli.characterize_psf import _characterize_psf

device = "cuda" if torch.cuda.is_available() else "cpu"
epi_bead_detection_settings = {
    "block_size": (8, 8, 8),
    "blur_kernel_size": 3,
    "min_distance": 20,
    "threshold_abs": 200.0,
    "max_num_peaks": 500,
    "exclude_border": (5, 5, 5),
    "device": device,
}

ls_bead_detection_settings = {
    "block_size": (64, 64, 32),
    "blur_kernel_size": 3,
    "nms_distance": 32,
    "min_distance": 50,
    "threshold_abs": 200.0,
    "max_num_peaks": 2000,
    "exclude_border": (5, 10, 5),
    "device": device,
}

deskew_bead_detection_settings = {
    "block_size": (64, 32, 16),
    "blur_kernel_size": 3,
    "nms_distance": 10,
    "min_distance": 50,
    "threshold_abs": 200.0,
    "max_num_peaks": 500,
    "exclude_border": (5, 5, 5),
    "device": device,
}


def check_acquisition_directory(root_dir: Path, acq_name: str, suffix='', idx=1) -> Path:
    acq_dir = root_dir / f'{acq_name}_{idx}{suffix}'
    if acq_dir.exists():
        return check_acquisition_directory(root_dir, acq_name, suffix, idx + 1)
    return acq_dir


mmc = Core()
step_per_um = None

# mmc.set_property('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter')
# mmc.set_property('Oryx2', 'Line Selector', 'Line5')
# mmc.update_system_state_cache()
# mmc.set_property('Oryx2', 'Line Mode', 'Output')
# mmc.set_property('Oryx2', 'Line Source', 'ExposureActive')

# %%
data_dir = Path(r'E:\2024_08_06_A549_TOMM20_SEC61')
date = '2024_08_07'
# dataset = f'{date}_RR_Straight_O3_scan'
dataset = f'{date}_epi_O1_benchmark'
# dataset = f'{date}_LS_Oryx_epi_illum'
# dataset =  f'{date}_LS_Oryx_LS_illum'
# dataset = f'{date}_LS_benchmark'

# epi settings
z_stage = 'PiezoStage:Q:35'
z_step = 0.2  # in um
z_range = (-2, 50)  # in um
pixel_size = 2 * 3.45 / 100  # in um
# pixel_size = 3.45 / 55.7  # in um
axis_labels = ("Z", "Y", "X")

# ls settings
# z_stage = 'AP Galvo'
# z_step = 0.205  # in um
# z_range = (-100, 85)  # in um
# pixel_size = 0.116  # in um
# axis_labels = ("SCAN", "TILT", "COVERSLIP")

# epi illumination rr detection settings
# z_stage = 'AP Galvo'
# # z_step = 0.205  # in um
# # z_range = (-85, 85)  # in um
# z_step = 0.1  # in um, reduced range and smaller step size
# z_range = (-31, 49)  # in um
# # pixel_size = 0.116  # in um
# pixel_size = 6.5 / 40 / 1.4  # in um, no binning
# axis_labels = ("SCAN", "TILT", "COVERSLIP")

# ls straight  settings
# from mantis.acquisition.microscope_operations import setup_kim101_stage
# z_stage = setup_kim101_stage('74000291')
# step_per_um = 35  # matches ~30 nm per step quoted in PIA13 specs
# z_start = 0 / step_per_um  # in um
# z_end = 1000 / step_per_um
# z_step = 5 / step_per_um
# # z_end = 500 / step_per_um
# # z_step = 20 / step_per_um
# z_range = np.arange(z_start, z_end + z_step, z_step)  # in um
# z_step /= 1.4 # count in 1.4x remote volume magnification
# pixel_size = 3.45 / 40 / 1.4  # in um, counting the 1.4x remote volume magnification
# axis_labels = ("Z", "Y", "X")


deskew = True
view = False
scale = (z_step, pixel_size, pixel_size)
data_path = data_dir / dataset

camera = mmc.get_camera_device()
if isinstance(z_stage, str):
    mmc.set_property('Core', 'Focus', z_stage)
    z_pos = mmc.get_position(z_stage)
    events = multi_d_acquisition_events(
        z_start=z_pos + z_range[0],
        z_end=z_pos + z_range[-1],
        z_step=z_step,
    )

    if camera in ('Prime BSI Express', 'Oryx2') and z_stage == 'AP Galvo':
        mmc.set_property('TS2_TTL1-8', 'Blanking', 'On')
        mmc.set_property('TS2_DAC03', 'Sequence', 'On')

    mmc.set_auto_shutter(False)
    mmc.set_shutter_open(True)
    with Acquisition(
        directory=str(data_dir),
        name=dataset,
        show_display=False,
    ) as acq:
        acq.acquire(events)
    mmc.set_shutter_open(False)
    mmc.set_auto_shutter(True)
    mmc.set_position(z_stage, z_pos)

    if camera in ('Prime BSI Express', 'Oryx2') and z_stage == 'AP Galvo':
        mmc.set_property('TS2_TTL1-8', 'Blanking', 'Off')
        mmc.set_property('TS2_DAC03', 'Sequence', 'Off')

    ds = acq.get_dataset()
    zyx_data = np.asarray(ds.as_array())
    channel_names = ['GFP']
    dataset = Path(ds.path).name
    ds.close()

else:
    acq_dir = check_acquisition_directory(data_dir, dataset, suffix='.zarr')
    dataset = acq_dir.stem

    mmc.set_auto_shutter(False)
    mmc.set_shutter_open(True)
    z_range_microsteps = (z_range * step_per_um).astype(int)
    zyx_data = acquire_defocus_stack(mmc, z_stage, z_range_microsteps)
    mmc.set_shutter_open(False)
    mmc.set_auto_shutter(True)

    # save to zarr store
    channel_names = ['GFP']
    with open_ome_zarr(
        data_dir / (dataset + '.zarr'),
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    ) as output_dataset:
        pos = output_dataset.create_position('0', '0', '0')
        pos.create_image(
            name="0",
            data=zyx_data[None, None, ...],
            chunks=(1, 1, 50) + zyx_data.shape[1:],  # may be bigger than 500 MB
        )
    z_stage.close()

raw = False
patch_size = (scale[0] * 15, scale[1] * 18, scale[2] * 18)
if axis_labels == ("SCAN", "TILT", "COVERSLIP"):
    raw = True
    patch_size = (scale[0] * 30, scale[1] * 36, scale[2] * 18)

# %% Characterize peaks

peaks = _characterize_psf(
    zyx_data=zyx_data,
    zyx_scale=scale,
    settings=CharacterizeSettings(
        **epi_bead_detection_settings, axis_labels=axis_labels, patch_size=patch_size
    ),
    output_report_path=data_dir / (dataset + '_psf_analysis'),
    input_dataset_path=data_dir,
    input_dataset_name=dataset,
)

# %% Visualize in napari

if view:
    viewer = napari.Viewer()
    viewer.add_image(zyx_data)
    viewer.add_points(
        peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
    )

# %% Deskew data and analyze
output_zarr_path = data_dir / (dataset + '_deskewed.zarr')

if raw and deskew:
    # chunk data so that it fits in the GPU memory
    # should not be necessary on the mantis GPU
    num_chunks = 4
    chunked_data = np.split(zyx_data, num_chunks, axis=-1)

    deskew_settings = {
        "ls_angle_deg": 30,
        "px_to_scan_ratio": round(scale[-1] / scale[-3], 3),
        "keep_overhang": True,
        "average_n_slices": 3,
    }

    deskewed_shape, deskewed_voxel_size = get_deskewed_data_shape(
        raw_data_shape=zyx_data.shape,
        pixel_size_um=scale[-1],
        **deskew_settings,
    )

    print('Deskewing data...')
    t1 = time.time()
    deskewed_chunks = []
    for chunk in chunked_data:
        deskewed_chunks.append(
            deskew_data(
                chunk,
                device=device,
                **deskew_settings,
            )
        )

    # concatenate arrays in reverse order
    deskewed_data = np.concatenate(deskewed_chunks[::-1], axis=-2)
    print(f'Tike to deskew data: {time.time() - t1}')

    # Characterize deskewed peaks
    deskewed_peaks = _characterize_psf(
        zyx_data=deskewed_data,
        zyx_scale=deskewed_voxel_size,
        settings=CharacterizeSettings(
            **deskew_bead_detection_settings,
            axis_labels=('Z', 'Y', 'X'),
        ),
        output_report_path=data_dir / (dataset + '_deskewed_psf_analysis'),
        input_dataset_path=output_zarr_path,
        input_dataset_name=dataset,
    )

    if view:
        viewer2 = napari.Viewer()
        viewer2.add_image(deskewed_data)
        viewer2.add_points(
            deskewed_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
        )

    # Save to zarr store
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + deskewed_voxel_size,
    )

    with open_ome_zarr(
        output_zarr_path, layout="hcs", mode="w", channel_names=channel_names
    ) as output_dataset:
        pos = output_dataset.create_position('0', '0', '0')
        pos.create_image(
            name="0",
            data=deskewed_data[None, None, ...],
            chunks=(1, 1, 50) + deskewed_shape[1:],  # may be bigger than 500 MB
            transform=[transform],
        )

# %%
