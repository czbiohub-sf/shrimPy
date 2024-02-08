# %%
import gc
import time

from pathlib import Path

import cupy as cp
import napari
import numpy as np
import torch

from cupyx.scipy.ndimage import affine_transform
from iohub.ngff_meta import TransformationMeta
from iohub.reader import open_ome_zarr, read_micromanager

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
    "threshold_abs": 250.0,
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

# %% Load data - swap with data acquisition block

deskew = True
view = False

data_dir = Path(r'E:\temp_2023_03_30_beads')
dataset = 'beads_ip_0.74_1'

scale = (0.1565, 0.116, 0.116)  # in um
axis_labels = ("SCAN", "TILT", "COVERSLIP")

# data_dir = Path(r'E:\temp_2022_12_22_LS_after_SL2')
# dataset = 'epi_beads_100nm_fl_mount_after_SL2_1'

data_path = data_dir / dataset

# zyx_data = tifffile.imread(data_dir / dataset / 'LS_beads_100nm_fl_mount_after_SL2_1_MMStack_Pos0.ome.tif')
# scale = (0.250, 0.069, 0.069)  # in um
# axis_labels = ("Z", "Y", "X")

if str(data_path).endswith('.zarr'):
    ds = open_ome_zarr(data_path / '0/0/0')
    zyx_data = ds.data[0, 0]
    channel_names = ds.channel_names
else:
    ds = read_micromanager(str(data_path))
    zyx_data = ds.get_array(0)[0, 0]
    channel_names = ds.channel_names

raw = False
if axis_labels == ("SCAN", "TILT", "COVERSLIP"):
    raw = True

# %% Detect peaks

t1 = time.time()
peaks = detect_peaks(
    zyx_data,
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
beads, offsets = extract_beads(
    zyx_data=zyx_data,
    points=peaks,
    scale=scale,
)

df_gaussian_fit, df_1d_peak_width = analyze_psf(
    zyx_patches=beads,
    bead_offsets=offsets,
    scale=scale,
)
t2 = time.time()
print(f'Time to analyze PSFs: {t2-t1}')

# %% Generate HTML report

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
    num_chunks = 2
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
