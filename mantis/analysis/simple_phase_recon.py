#%%
import os
import numpy as np

import tifffile
from ndtiff import Dataset
from waveorder.io.reader import WaveorderReader
from recOrder.compute.reconstructions import (
    initialize_reconstructor,
    reconstruct_phase3D,
)
import napari

#%% Load data
raw_data_path = r'D:\2023_03_30_kidney_tissue'
processed_data_path = r'D:\2023_03_30_kidney_tissue'
dataset = 'FOV_grid_1'

lf_dataset_name = '_'.join(dataset.split('_')[:-1]) + '_labelfree_1'
ds = Dataset(os.path.join(raw_data_path, dataset, lf_dataset_name))

data = np.asarray(ds.as_array())
P, T, C, Z, Y, X = data.shape

# %% Init reconstructor

reconstructor_args = {
    "image_dim": (Y, X),
    "mag": 40 * 1.4,  # magnification
    "pixel_size_um": 3.45 * 2,  # pixel size in um
    "n_slices": Z,  # number of slices in z-stack
    "z_step_um": 0.4 / 1.4,  # z-step size in um
    "wavelength_nm": 450,
    "NA_obj": 1.35,  # numerical aperture of objective
    "NA_illu": 0.52,  # numerical aperture of condenser
    "n_obj_media": 1.40,  # refractive index of objective immersion media
    "pad_z": 5,  # slices to pad for phase reconstruction boundary artifacts
    "mode": "3D",  # phase reconstruction mode, "2D" or "3D"
    "use_gpu": False,
    "gpu_id": 0,
}
reconstructor = initialize_reconstructor(pipeline="PhaseFromBF", **reconstructor_args)

#%%
phase3D = reconstruct_phase3D(data[0, 0, 0], reconstructor, method="Tikhonov", reg_re=1e-2)
# %%
napari.view_image(phase3D)
# %%

tifffile.imwrite(
    os.path.join(processed_data_path, dataset, 'phase3D.tif'), phase3D.astype('single')
)
# %%

for p_idx in range(P):
    phase3D = reconstruct_phase3D(
        data[p_idx, 0, 0], reconstructor, method="Tikhonov", reg_re=1e-2
    )

    tifffile.imwrite(
        os.path.join(processed_data_path, dataset, f'phase3D_Pos{p_idx}.tif'),
        phase3D.astype('single'),
    )

# %%
