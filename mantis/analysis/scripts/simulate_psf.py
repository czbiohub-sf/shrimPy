# Variable abbreviations
# stc = scan, tilt, coverslip --- raw data coordinates
# otf = optical transfer function
# psf = point spread function

import napari
import numpy as np
import scipy
import torch

from waveorder.models.isotropic_fluorescent_thick_3d import calculate_transfer_function

from mantis.analysis.deskew import _deskew_matrix


def _apply_centered_affine(zyx_array, M):
    """Applies a translation-free affine transformation to a 3D array while
    maintaining the center coordinate at (zyx_array.shape // 2)

    For mantis - useful for moving PSFs between skewed and deskewed spaces.

    Parameters
    ----------
    zyx_array : NDArray with ndim == 3
        3D input array
    M : NDArray with shape = (3, 3)
        3x3 transformation matrix, the translation-free part of an affine matrix
        Can model reflection, scaling, rotation, and shear.

    Returns
    -------
    NDArray with ndim == 3
        transformed matrix with shape matched to input
    """

    # keep (zyx_array.shape // 2) centered
    offset = np.dot(np.eye(3) - M, np.array(zyx_array.shape) // 2)

    return scipy.ndimage.affine_transform(
        zyx_array,
        M,
        offset=offset,
        output_shape=zyx_array.shape,
        order=1,
        cval=0,
    )


v = napari.Viewer()

# ---- psf input parameters (to be refactored)

# sampling parameters
psf_stc_shape = 3 * (30,)
psf_stc_scale = 3 * (0.116,)  # um
supersample_factor = 5

# illumination and detection parameters
ls_angle_deg = 30

# illumination parameters
ls_scan_waist_fwhm = 1.0  # um

# detection parameters
wavelength_emission = 0.550  # um
numerical_aperture_detection = 1.35
index_of_refraction_media = 1.404

# ----

# internal simulation parameters
px_to_scan_ratio = psf_stc_scale[1] / psf_stc_scale[0]
ct = np.cos(ls_angle_deg * np.pi / 180)
st = np.sin(ls_angle_deg * np.pi / 180)
deskew_matrix = _deskew_matrix(px_to_scan_ratio, ct)
skew_matrix = np.linalg.inv(deskew_matrix)

psf_stc_ss_shape = np.array(psf_stc_shape) * supersample_factor
psf_stc_ss_scale = np.array(psf_stc_scale) / supersample_factor
psf_zyx_ss_scale = np.array(
    [st * psf_stc_ss_scale[0], psf_stc_ss_scale[1], psf_stc_ss_scale[2]]
)

# calculate illumination psf
ls_scan_waist_std = ls_scan_waist_fwhm / (2 * np.sqrt(2 * np.log(2)))
scan_positions = psf_stc_ss_scale[0] * (
    np.arange(psf_stc_ss_shape[0]) - (psf_stc_ss_shape[0] / 2)
)
illumination_psf_scan = np.exp(-(scan_positions**2) / (2 * ls_scan_waist_std**2))

# calculate detection psf in zyx coordinates using waveorder
detection_otf_zyx = calculate_transfer_function(
    psf_stc_ss_shape,
    psf_zyx_ss_scale[1],
    psf_zyx_ss_scale[0],
    wavelength_emission,
    0,
    index_of_refraction_media,
    numerical_aperture_detection,
)

detection_psf_zyx = np.array(
    torch.real(torch.fft.ifftshift(torch.fft.ifftn(detection_otf_zyx, dim=(0, 1, 2))))
)

detection_psf_stc = _apply_centered_affine(detection_psf_zyx, skew_matrix)
psf_stc = np.einsum('i,ijk->ijk', illumination_psf_scan, detection_psf_stc)

# this dense illumination_psf is not necessary, but it's useful for debugging
illumination_psf_stc = np.einsum(
    'i,ijk->ijk', illumination_psf_scan, np.ones_like(detection_psf_stc)
)

# prepare viewer
v.scale_bar.visible = True
v.scale_bar.unit = "um"

v.add_image(illumination_psf_stc, name="raw illumination", scale=psf_stc_ss_scale)
v.add_image(detection_psf_stc, name="raw detection", scale=psf_stc_ss_scale)
v.add_image(psf_stc, name="raw total", scale=psf_stc_ss_scale)

v.add_image(
    _apply_centered_affine(illumination_psf_stc, deskew_matrix),
    name="deskewed illumination",
    scale=psf_stc_ss_scale,
)
v.add_image(
    _apply_centered_affine(detection_psf_stc, deskew_matrix),
    name="deskewed detection",
    scale=psf_stc_ss_scale,
)
v.add_image(
    _apply_centered_affine(psf_stc, deskew_matrix),
    name="deskewed total",
    scale=psf_stc_ss_scale,
)
