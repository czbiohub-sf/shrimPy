# %%
import gc
import time
import warnings

from pathlib import Path

import cupy as cp
import napari
import numpy as np
import torch

from cupyx.scipy.ndimage import affine_transform
from iohub.ngff_meta import TransformationMeta
from iohub.reader import open_ome_zarr
from pycromanager import Acquisition, Core, multi_d_acquisition_events

from mantis.acquisition.microscope_operations import acquire_defocus_stack
from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.analysis.analyze_psf import (
    analyze_psf,
    detect_peaks,
    extract_beads,
    generate_report,
)
from mantis.analysis.deskew import (
    _average_n_slices,
    _get_transform_matrix,
    get_deskewed_data_shape,
)

epi_bead_detection_settings = {
    "block_size": (8, 8, 8),
    "blur_kernel_size": 3,
    "min_distance": 20,
    "threshold_abs": 200.0,
    "max_num_peaks": 500,
    "exclude_border": (5, 5, 5),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ls_bead_detection_settings = {
    "block_size": (64, 64, 32),
    "blur_kernel_size": 3,
    "nms_distance": 32,
    "min_distance": 50,
    "threshold_abs": 200.0,
    "max_num_peaks": 2000,
    "exclude_border": (5, 10, 5),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

deskew_bead_detection_settings = {
    "block_size": (64, 32, 16),
    "blur_kernel_size": 3,
    "nms_distance": 10,
    "min_distance": 50,
    "threshold_abs": 200.0,
    "max_num_peaks": 500,
    "exclude_border": (5, 5, 5),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def check_acquisition_directory(root_dir: Path, acq_name: str, suffix='', idx=1) -> Path:
    acq_dir = root_dir / f'{acq_name}_{idx}{suffix}'
    if acq_dir.exists():
        return check_acquisition_directory(root_dir, acq_name, suffix, idx + 1)
    return acq_dir


mmc = Core()

# mmc.set_property('Prime BSI Express', 'ExposeOutMode', 'Rolling Shutter')
# mmc.set_property('Oryx2', 'Line Selector', 'Line5')
# mmc.update_system_state_cache()
# mmc.set_property('Oryx2', 'Line Mode', 'Output')
# mmc.set_property('Oryx2', 'Line Source', 'ExposureActive')

# %%
data_dir = Path(r'E:\2024_05_10_A594_CAAX_DRAQ5')
date = '2024_05_07'
# dataset = f'{date}_RR_Straight_O3_scan'
# dataset = f'{date}_epi_O1_benchmark'
# dataset = f'{date}_LS_Oryx_epi_illum'
# dataset =  f'{date}_LS_Oryx_LS_illum'
dataset = f'{date}_LS_benchmark'

# epi settings
# z_stage = 'PiezoStage:Q:35'
# z_step = 0.2  # in um
# z_range = (-2, 50)  # in um
# pixel_size = 2 * 3.45 / 100  # in um
# # pixel_size = 3.45 / 55.7  # in um
# axis_labels = ("Z", "Y", "X")

# ls settings
# z_stage = 'AP Galvo'
# z_step = 0.205  # in um
# z_range = (-100, 85)  # in um
# pixel_size = 0.116  # in um
# axis_labels = ("SCAN", "TILT", "COVERSLIP")

# epi illumination rr detection settings
z_stage = 'AP Galvo'
# z_step = 0.205  # in um
# z_range = (-85, 85)  # in um
z_step = 0.1  # in um, reduced range and smaller step size
z_range = (-31, 49)  # in um
# pixel_size = 0.116  # in um
pixel_size = 6.5 / 40 / 1.4  # in um, no binning
axis_labels = ("SCAN", "TILT", "COVERSLIP")
step_per_um = None

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
if axis_labels == ("SCAN", "TILT", "COVERSLIP"):
    raw = True

# %% Detect peaks

t1 = time.time()
peaks = detect_peaks(
    zyx_data,
    # **epi_bead_detection_settings,
    **ls_bead_detection_settings,
    verbose=True,
)
gc.collect()
torch.cuda.empty_cache()
t2 = time.time()
print(f'Time to detect peaks: {t2-t1}')

# %% Visualize in napari

if view:
    viewer = napari.Viewer()
    viewer.add_image(zyx_data)
    viewer.add_points(
        peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
    )

# %% Extract and analyze bead patches

t1 = time.time()
if raw:
    patch_size = (scale[0] * 30, scale[1] * 36, scale[2] * 18)
else:
    patch_size = (scale[0] * 15, scale[1] * 18, scale[2] * 18)
beads, offsets = extract_beads(
    zyx_data=zyx_data,
    points=peaks,
    scale=scale,
    patch_size=patch_size,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_gaussian_fit, df_1d_peak_width = analyze_psf(
        zyx_patches=beads,
        bead_offsets=offsets,
        scale=scale,
    )
t2 = time.time()
print(f'Time to analyze PSFs: {t2-t1}')

# Generate HTML report

psf_analysis_path = data_dir / (dataset + '_psf_analysis')
generate_report(
    psf_analysis_path,
    data_dir,
    dataset,
    beads,
    peaks,
    df_gaussian_fit,
    df_1d_peak_width,
    scale,
    axis_labels,
)

# %% Deskew data and analyze

if raw and deskew:
    # deskew
    num_chunks = 4
    chunked_data = np.split(zyx_data, num_chunks, axis=-1)
    chunk_shape = chunked_data[0].shape

    settings = DeskewSettings(
        pixel_size_um=scale[-1],
        ls_angle_deg=30,
        scan_step_um=scale[-3],
        keep_overhang=True,
        average_n_slices=3,
    )

    t1 = time.time()
    deskewed_shape, _ = get_deskewed_data_shape(
        chunk_shape,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
    )

    matrix = _get_transform_matrix(
        chunk_shape,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
    )

    matrix_gpu = cp.asarray(matrix)
    deskewed_chunks = []
    for chunk in chunked_data:
        deskewed_data_gpu = affine_transform(
            cp.asarray(chunk),
            matrix_gpu,
            output_shape=deskewed_shape,
            order=1,
            cval=80,
        )
        deskewed_chunks.append(cp.asnumpy(deskewed_data_gpu))
        del deskewed_data_gpu
    cp._default_memory_pool.free_all_blocks()

    # concatenate arrays in reverse order
    # identical to cpu deskew using ndi.affine_transform
    deskewed_data = np.concatenate(deskewed_chunks[::-1], axis=-2)

    averaged_deskewed_data = _average_n_slices(
        deskewed_data, average_window_width=settings.average_n_slices
    )

    deskewed_shape, voxel_size = get_deskewed_data_shape(
        zyx_data.shape,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio,
        settings.keep_overhang,
        settings.average_n_slices,
        settings.pixel_size_um,
    )
    t2 = time.time()
    print(f'Time to deskew: {t2-t1: .2f} seconds')

    # detect peaks again :(
    t1 = time.time()
    deskewed_peaks = detect_peaks(
        averaged_deskewed_data,
        **deskew_bead_detection_settings,
        verbose=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    t2 = time.time()
    print(f'Time to detect deskewed peaks: {t2-t1: .2f} seconds')

    if view:
        viewer2 = napari.Viewer()
        viewer2.add_image(averaged_deskewed_data)
        viewer2.add_points(
            deskewed_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
        )

    deskewed_beads, deskewed_offsets = extract_beads(
        zyx_data=averaged_deskewed_data,
        points=deskewed_peaks,
        scale=scale,
    )

    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_deskewed_gaussian_fit, df_deskewed_1d_peak_width = analyze_psf(
            zyx_patches=deskewed_beads,
            bead_offsets=deskewed_offsets,
            scale=voxel_size,
        )
    t2 = time.time()
    print(f'Time to analyze deskewed PSFs: {t2-t1: .2f} seconds')

    output_zarr_path = data_dir / (dataset + '_deskewed.zarr')
    report_path = data_dir / (dataset + '_deskewed_psf_analysis')
    generate_report(
        report_path,
        output_zarr_path,
        dataset,
        deskewed_beads,
        deskewed_peaks,
        df_deskewed_gaussian_fit,
        df_deskewed_1d_peak_width,
        voxel_size,
        ('Z', 'Y', 'X'),
    )

    # Save to zarr store
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    with open_ome_zarr(
        output_zarr_path, layout="hcs", mode="w", channel_names=channel_names
    ) as output_dataset:
        pos = output_dataset.create_position('0', '0', '0')
        pos.create_image(
            name="0",
            data=averaged_deskewed_data[None, None, ...],
            chunks=(1, 1, 50) + deskewed_shape[1:],  # may be bigger than 500 MB
            transform=[transform],
        )

# %%
