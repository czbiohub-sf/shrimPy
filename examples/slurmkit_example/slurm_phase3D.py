import datetime
import os
import glob
import multiprocessing as mp
from slurmkit import SlurmParams, slurm_function, submit_function
from natsort import natsorted
import click

#TODO: these are imports that are needed for now but shouldn't be required
# or are temporarily needed until we find a cleaner solution
from iohub import open_ome_zarr
from mantis.analysis.multiproc_phase_recon import(
get_output_paths,create_empty_zarr,
reconstruct_phase3D_n_save,reconstruct_single_position
)
from waveorder.models import phase_thick_3d


# io parameters
input_paths = "/hpc/instruments/cm.mantis/2023_05_10_PCNA_RAC1/timelapse_1/timelapse_labelfree_1.zarr/*/*/*"
output_data_path = './phase_3D.zarr'

# sbatch and resource parameters
cpus_per_task = 16
mem_per_cpu = "16G"
time = 40  # minutes


# path handling
input_paths = natsorted(glob.glob(input_paths))
output_dir = os.path.dirname(output_data_path)
output_paths = get_output_paths(input_paths, output_data_path)
click.echo(f"in: {input_paths}, out: {output_paths}")
slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))

# ---------- TODO: Parameters that are roughly needed for most recons-----
#TODO: This should be handled within recOrder or passed by reading yaml file
# Parameter file options (mantis)
yx_pixel_size = (3.45 * 2) / (40 * 1.4)
z_pixel_size = 0.4 / 1.4
z_padding = 5
index_of_refraction_media = 1.40
numerical_aperture_illumination = 0.52
numerical_aperture_detection = 1.35
wavelength_illumination = 0.450
regularization_strength = 0.001

# -------TODO: This should be replaced by recOrders implementation eventually-------
#TODO: for now this is what decies what reconstruction to do
recon_params = ("phase", 3)  # (bire | phase |bire_n_phase , ndim)

## Precomputation for transfer functions
print("Computing transfer function...")
tczyx = open_ome_zarr(input_paths[0])
T, C, Z, Y, X = tczyx.data.shape

zyx_data = tczyx[0][0, 0]
(
    real_potential_transfer_function,
    imaginary_potential_transfer_function,
) = phase_thick_3d.calculate_transfer_function(
    zyx_shape=zyx_data.shape,
    yx_pixel_size=yx_pixel_size,
    z_pixel_size=z_pixel_size,
    wavelength_illumination=wavelength_illumination,
    z_padding=z_padding,
    index_of_refraction_media=index_of_refraction_media,
    numerical_aperture_illumination=numerical_aperture_illumination,
    numerical_aperture_detection=numerical_aperture_detection,
    axial_flip=True,
)
tczyx.close()

# Get the reconstruction parameters
reconstruction_parameters = (
    real_potential_transfer_function,
    imaginary_potential_transfer_function,
    z_padding,
    z_pixel_size,
    wavelength_illumination,
    regularization_strength,
)
reconstruction_function = reconstruct_phase3D_n_save
# -------------------------------------------------------

# initialize zarr
create_empty_zarr(input_paths, output_data_path, recon_params)

# prepare slurm parameters
params = SlurmParams(
    partition="cpu",
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our reconstruct_single_position() function with slurmkit
#TODO: the reconstruction_function should be passed based on the future config. file?
slurm_reconstruct_single_position = slurm_function(reconstruct_single_position)
reconstruction_func = slurm_reconstruct_single_position(
    reconstruction_function = reconstruction_function,
    reconstruction_parameters =reconstruction_parameters, 
    num_processes=cpus_per_task
)

# generate an array of jobs by passing the in_path and out_path to slurm wrapped function
deskew_jobs = [
    submit_function(
        reconstruction_func,
        slurm_params=params,
        input_data_path=in_path,
        output_path=out_path,
    )
    for in_path, out_path in zip(input_paths, output_paths)
]
