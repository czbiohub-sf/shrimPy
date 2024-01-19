# %%
import csv

import numpy as np

from iohub import open_ome_zarr, read_micromanager
from iohub.ngff_meta import TransformationMeta
from recOrder.compute.reconstructions import initialize_reconstructor, reconstruct_phase3D

# %% Load data

data_path = r'Z:\rawdata\mantis\2023_04_20 HEK RAC1 PCNA\timelapse_2\timelapse_labelfree_1'
output_path = r'Z:\projects\mantis\2023_04_20 HEK RAC1 PCNA\timelapse_2_phase.zarr'
positions = r'Z:\rawdata\mantis\2023_04_20 HEK RAC1 PCNA\timelapse_2\positions.csv'

channel_names = ['phase3D']

# %%
# Load positions log and generate pos_hcs_idx
if positions is not None:
    with open(positions, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        pos_log = [row for row in reader]

reader = read_micromanager(data_path)
writer = open_ome_zarr(output_path, mode="a", layout="hcs", channel_names=channel_names)

P, T, C, Z, Y, X = reader.get_num_positions(), *reader.shape
if positions is not None:
    pos_hcs_idx = [(row['well_id'][0], row['well_id'][1:], row['site_num']) for row in pos_log]
else:
    pos_hcs_idx = [(0, p, 0) for p in range(P)]

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
voxel_size = (
    reconstructor_args["z_step_um"],
    reconstructor_args["pixel_size_um"] / reconstructor_args["mag"],
    reconstructor_args["pixel_size_um"] / reconstructor_args["mag"],
)

# %%
# Loop through (P, T, C), deskewing and writing as we go
for p in range(P):
    position = writer.create_position(*pos_hcs_idx[p])
    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )
    img = position.create_zeros(
        name="0",
        shape=(T, 1, Z, Y, X),
        chunks=(1, 1, 32) + (Y, X),
        dtype=np.double,
        transform=[transform],
    )
    for t in range(T):
        print(f"Reconstructing t={t}/{T-1}, p={p}/{P-1}")
        data = reader.get_array(p)[t, 0, ...]  # zyx

        # Reconstruct
        phase3D = reconstruct_phase3D(data, reconstructor, method="Tikhonov", reg_re=1e-2)

        img[t, 0, ...] = phase3D  # write to zarr
