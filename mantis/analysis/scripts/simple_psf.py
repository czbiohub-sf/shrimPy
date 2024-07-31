# %%
import warnings

import napari
import numpy as np
import torch

from waveorder import optics

from mantis.analysis.analyze_psf import analyze_psf, detect_peaks, extract_beads


# %% Generate simulated PSF library
def calculate_transfer_function(
    zyx_shape,
    yx_pixel_size,
    z_pixel_size,
    wavelength_emission,
    z_padding,
    index_of_refraction_media,
    numerical_aperture_detection,
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


def generate_psf(numerical_aperture_detection, zyx_shape, zyx_scale):
    # detection parameters
    wavelength_emission = 0.550  # um
    index_of_refraction_media = 1.404

    # internal simulation parameters
    detection_otf_zyx = calculate_transfer_function(
        zyx_shape,
        zyx_scale[1],
        zyx_scale[0],
        wavelength_emission,
        0,
        index_of_refraction_media,
        numerical_aperture_detection,
    )

    simulated_psf = np.array(
        torch.real(torch.fft.ifftshift(torch.fft.ifftn(detection_otf_zyx, dim=(0, 1, 2))))
    )
    simulated_psf *= 1e7
    return simulated_psf


numerical_apertures = [0.9, 1.1, 1.35]
zyx_shape = np.array([151, 151, 151])
zyx_scale = (0.1, 0.0616, 0.0616)

v = napari.Viewer()
for numerical_aperture in numerical_apertures:
    print(f"Generating NA={numerical_aperture}")

    zyx_data = generate_psf(numerical_aperture, zyx_shape, zyx_scale)
    v.add_image(zyx_data, name=f"{numerical_aperture}", scale=zyx_scale)

    epi_bead_detection_settings = {
        "block_size": (8, 8, 8),
        "blur_kernel_size": 3,
        "min_distance": 0,
        "threshold_abs": 100.0,
        "max_num_peaks": 1,
        "exclude_border": (0, 0, 0),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    peaks = detect_peaks(zyx_data, **epi_bead_detection_settings, verbose=True)

    beads, offsets = extract_beads(
        zyx_data=zyx_data,
        points=peaks,
        scale=zyx_scale,
    )

    print(f"Fitting NA={numerical_aperture}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gaussian_fit, df_1d_peak_width = analyze_psf(
            zyx_patches=beads,
            bead_offsets=offsets,
            scale=zyx_scale,
        )

    print(df_gaussian_fit)
    print(df_1d_peak_width)

# %%
