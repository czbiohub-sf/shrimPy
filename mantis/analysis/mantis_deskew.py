#%%
import os
import numpy as np
import tifffile
import napari
import scipy


def mantis_deskew(raw_data, xy_px_spacing, scan_spacing, theta):
    """Deskews data from the mantis microscope

    Parameters
    ----------
    raw_data : NDArray with ndim == 3
        raw data from the mantis microscope
        - axis 0 corresponds to the scanning axis
        - axis 1 corresponds to the "tilted" axis
        - axis 2 corresponds to the axis in the plane of the coverslip
    xy_px_spacing : float
        transverse spacing of pixels in the object
        e.g. camera pixels = 6.5 um, magnification = 1.4*40 implies
        xy_px_spacing = 6.5/(1.4*40) = 0.116
    scan_spacing : float
        spacing of samples along the scan dimension in the plane of the
        coverslip. e.g. if the stage moves 0.3 um, scan_spacing = 0.3
        Note: use identical units as `xy_px_spacing`
    theta : float
        angle of light sheet with respect to the optical axis.

    Returns
    -------
    deskewed_data : NDArray with ndim == 3
        axis 0 is the Z axis, normal to the coverslip
        axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        axis 2 is the X axis, the scanning axis
    dimensions : tuple
        For convenience, (scan_spacing, xy_px_spacing, xy_px_spacing)
        is the spacing of the output voxels in each dimension.

    """
    # Non-dimensional parameters
    ar = xy_px_spacing / scan_spacing  # voxel aspect ratio
    st = np.sin(theta * np.pi / 180)  # sin(theta)

    # Prepare transforms
    Z, Y, X = raw_data.shape
    matrix = np.array([[ar * st, 0, ar], [1, 0, 0], [0, 1, 0]])
    offset = (-Y * st * ar, 0, 0)
    output_shape = (Y, X, int(np.ceil(Z / ar + (Y * st))))

    # Apply transforms
    deskewed_data = scipy.ndimage.affine_transform(
        raw_data, matrix, offset=offset, output_shape=output_shape
    )

    # Return transformed data with its dimensions
    return deskewed_data, (scan_spacing, xy_px_spacing, xy_px_spacing)

#%%

raw_data_path = "/hpc/projects/comp_micro/rawdata/mantis/2023_03_27_argolight/"
processed_data_path = "/hpc/projects/comp_micro/projects/mantis/2023_03_31_deskew"
datasets = ['sphere_1', 'ring_stack_1', 'rings_1']

for dataset in datasets:
    os.makedirs(os.path.join(processed_data_path, dataset), exist_ok=True)

    # Load data
    data = tifffile.imread(
        os.path.join(raw_data_path, dataset, dataset + "_MMStack_Pos0.ome.tif")
    )

    #data = data[170:470, 50:251, 900:1102]  # crop for faster testing

    # Deskew and save
    for tilt in [47]:
        for scan_spacing in [0.31]:
            print(tilt, scan_spacing)
            deskewed, dims = mantis_deskew(data, 0.116, scan_spacing, tilt)
            out_path = os.path.join(processed_data_path, dataset, f"{tilt}-{scan_spacing:.2f}.tif")
            tifffile.imwrite(out_path, deskewed)
