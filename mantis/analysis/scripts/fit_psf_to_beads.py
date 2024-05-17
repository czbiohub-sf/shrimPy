# %%
import gc
import napari
import numpy as np
import time
import torch

from mantis.analysis.analyze_psf import detect_peaks, extract_beads
from mantis.analysis.deskew import _deskew_matrix
from mantis.analysis.scripts.simulate_psf import _apply_centered_affine
from iohub import read_micromanager
from waveorder import optics
from waveorder.models.isotropic_fluorescent_thick_3d import apply_inverse_transfer_function

# %% Load beads (from ndtiff for now)
data_dir = (
    "/hpc/instruments/cm.mantis/2024_04_23_mantis_alignment/2024_05_05_LS_Oryx_LS_illum_8/"
)
input_dataset = read_micromanager(data_dir, data_type="ndtiff")
stc_data = input_dataset.get_array(position="0")[0, 0]

# manual...pull from zarr later
s_step = 5 / 35 / 1.4
tc_size = 3.45 / 40 / 1.4
stc_scale = (s_step, tc_size, tc_size)


# %% Detect peaks and find an "average PSF"
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

t1 = time.time()
peaks = detect_peaks(
    stc_data,
    **ls_bead_detection_settings,
    verbose=True,
)
gc.collect()
torch.cuda.empty_cache()
t2 = time.time()
print(f'Time to detect peaks: {t2-t1}')

# %% Extract beads
beads, offsets = extract_beads(
    zyx_data=stc_data,
    points=peaks,
    scale=stc_scale,
)
stc_shape = beads[0].shape

# Filter PSFs with different shapes
filtered_beads = [x for x in beads if x.shape == stc_shape]
bzyx_data = np.stack(filtered_beads)
normalized_bzyx_data = bzyx_data / np.max(bzyx_data, axis=(-3, -2, -1))[:, None, None, None]
average_psf = np.mean(normalized_bzyx_data, axis=0)

# %% View PSFs
import napari

v = napari.Viewer()
v.add_image(normalized_bzyx_data)
v.add_image(average_psf)


# %% Generate simulated PSF library
def calculate_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_emission,
    z_padding,
    index_of_refraction_media,
    numerical_aperture_detection,
    coma_strength,
):
    # Modified from waveorder
    fy = torch.fft.fftfreq(zyx_shape[1], yx_pixel_size)
    fx = torch.fft.fftfreq(zyx_shape[2], yx_pixel_size)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    radial_frequencies = torch.sqrt(fyy**2 + fxx**2)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )

    # Custom pupil
    det_pupil = torch.zeros(radial_frequencies.shape, dtype=torch.complex64)
    cutoff = numerical_aperture_detection / wavelength_emission
    det_pupil[radial_frequencies < cutoff] = 1
    # det_pupil[((fxx) ** 2 + (fy)**2) ** 0.5 > cutoff] = 0  # add cutoff lune here
    det_pupil *= np.exp(
        coma_strength
        * 1j
        * ((3 * (radial_frequencies / cutoff) ** 3) - (2 * (radial_frequencies / cutoff)))
        * torch.div(fxx + 1e-15, radial_frequencies + 1e-15)
    )  # coma

    # v.add_image(torch.real(det_pupil).numpy())
    # v.add_image(torch.imag(det_pupil).numpy())

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_emission / index_of_refraction_media,
        z_position_list,
    )

    point_spread_function = torch.abs(torch.fft.ifft2(propagation_kernel, dim=(1, 2))) ** 2
    optical_transfer_function = torch.fft.fftn(point_spread_function, dim=(0, 1, 2))
    optical_transfer_function /= torch.max(torch.abs(optical_transfer_function))  # normalize

    return optical_transfer_function


def generate_psf(numerical_aperture_detection, ls_angle_deg, coma_strength):
    # detection parameters
    wavelength_emission = 0.550  # um
    index_of_refraction_media = 1.404

    # internal simulation parameters
    px_to_scan_ratio = stc_scale[1] / stc_scale[0]
    ct = np.cos(ls_angle_deg * np.pi / 180)
    st = np.sin(ls_angle_deg * np.pi / 180)
    deskew_matrix = _deskew_matrix(px_to_scan_ratio, ct)
    skew_matrix = np.linalg.inv(deskew_matrix)

    zyx_scale = np.array([st * stc_scale[1], stc_scale[1], stc_scale[1]])
    detection_otf_zyx = calculate_transfer_function(
        stc_shape,
        zyx_scale[1],
        zyx_scale[0],
        wavelength_emission,
        0,
        index_of_refraction_media,
        numerical_aperture_detection,
        coma_strength,
    )

    detection_psf_zyx = np.array(
        torch.real(torch.fft.ifftshift(torch.fft.ifftn(detection_otf_zyx, dim=(0, 1, 2))))
    )

    simulated_psf = _apply_centered_affine(detection_psf_zyx, skew_matrix)
    simulated_psf /= np.max(simulated_psf)
    return simulated_psf, zyx_scale, deskew_matrix


# Define grid search
na_det_list = np.array([0.95, 1.05, 1.15, 1.25, 1.35])
ls_angle_deg_list = np.array([30])
coma_strength_list = np.array([-0.2, -0.1, 0, 0.1, 0.2])
params = np.stack(
    np.meshgrid(na_det_list, ls_angle_deg_list, coma_strength_list, indexing="ij"), axis=-1
)

pzyx_array = np.zeros(params.shape[:-1] + stc_shape)
pzyx_deskewed_array = np.zeros(params.shape[:-1] + stc_shape)

for i in np.ndindex(params.shape[:-1]):
    print(f"Simulating PSF with params: {params[i]}")
    pzyx_array[i], zyx_scale, deskew_matrix = generate_psf(*params[i])
    pzyx_deskewed_array[i] = _apply_centered_affine(pzyx_array[i], deskew_matrix)

print("Visualizing")
v = napari.Viewer()
v.add_image(average_psf, scale=stc_scale)
v.add_image(pzyx_array, scale=stc_scale)

v.dims.axis_labels = ["NA", "", "COMA", "Z", "Y", "X"]

# v.add_image(_apply_centered_affine(average_psf, deskew_matrix), scale=zyx_scale)
# v.add_image(pzyx_deskewed_array, scale=zyx_scale)

# Optimize match
diff = np.sum((pzyx_array - average_psf) ** 2, axis=(-3, -2, -1))
min_idx = np.unravel_index(np.argmin(diff), diff.shape)
print(min_idx)
print(params[min_idx])


# %% Crop data for prototyping deconvolution
stc_data = stc_data[:200, :200, :500]

# %% 

# Simple background subtraction and normalization
average_psf -= np.min(average_psf)
average_psf /= np.max(average_psf)

# %%
zyx_padding = np.array(stc_data.shape) - np.array(average_psf.shape)
pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
padded_average_psf = np.pad(average_psf, pad_width=pad_width, mode="constant", constant_values=0)
transfer_function = np.abs(np.fft.fftn(padded_average_psf))
transfer_function /= np.max(transfer_function)
print(transfer_function.shape)

# %%

# %%
stc_data_deconvolved = apply_inverse_transfer_function(torch.tensor(stc_data), torch.tensor(transfer_function), 0, regularization_strength=1e-3)

v = napari.Viewer()
v.add_image(padded_average_psf)
v.add_image(stc_data)
v.add_image(stc_data_deconvolved.numpy())



# %%
