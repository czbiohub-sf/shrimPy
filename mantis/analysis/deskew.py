import numpy as np
import scipy
import click
from AnalysisSettings import DeskewSettings
import yaml
import napari
import csv
from dataclasses import asdict
from iohub import read_micromanager, open_ome_zarr
from iohub.ngff_meta import TransformationMeta


@click.command()
@click.argument(
    "data_path",
    type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False, file_okay=False),
)
@click.argument(
    "deskew_params_path",
    type=click.Path(exists=True, file_okay=True),
)
@click.option(
    "--positions",
    type=click.Path(exists=True, file_okay=True),
    required=False,
    help="Path to positions CSV log file",
)
@click.option(
    "--view",
    "-v",
    default=False,
    required=False,
    is_flag=True,
    help="View the correctly scaled result in napari",
)
@click.option(
    "--keep-overhang",
    "-ko",
    default=False,
    is_flag=True,
    help="Keep the overhanging region.",
)
def deskew(data_path, output_path, deskew_params_path, positions, view, keep_overhang):
    """
    Deskews across P, T, C axes using a parameter file generated with estimate_deskew.py
    """

    assert str(output_path).endswith('.zarr'), "Output path must be a zarr store"

    # Load params
    with open(deskew_params_path) as file:
        raw_settings = yaml.safe_load(file)
    settings = DeskewSettings(**raw_settings)
    print(f"Deskewing parameters: {asdict(settings)}")

    # Load positions log and generate pos_hcs_idx
    if positions is not None:
        with open(positions, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            pos_log = [row for row in reader]

    reader = read_micromanager(data_path)
    writer = open_ome_zarr(
        output_path, mode="a", layout="hcs", channel_names=reader.channel_names
    )

    P, T, C, Z, Y, X = reader.get_num_positions(), *reader.shape
    deskewed_shape, voxel_size = get_deskewed_data_shape(
        (Z, Y, X), 
        settings.pixel_size_um,
        settings.ls_angle_deg,
        settings.px_to_scan_ratio
    )

    if positions is not None:
        pos_hcs_idx = [
            (
            row['well_id'][0], row['well_id'][1:], row['site_num']
            ) for row in pos_log
        ]
    else:
        pos_hcs_idx = [(0, p, 0) for p in range(P)]

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
            shape=(
                T,
                C,
            )
            + deskewed_shape,
            dtype=np.int16,
            transform=[transform],
        )
        for t in range(T):
            for c in range(C):
                print(f"Deskewing c={c}/{C-1}, t={t}/{T-1}, p={p}/{P-1}")
                data = reader.get_array(p)[t, c, ...]  # zyx

                # Deskew
                deskewed = mantis_deskew(
                    data, settings.px_to_scan_ratio, settings.ls_angle_deg, keep_overhang
                )

                img[t, c, ...] = deskewed  # write to zarr

    # Write metadata
    writer.zattrs["deskewing"] = asdict(settings)
    writer.zattrs["mm-meta"] = reader.mm_meta["Summary"]

    # Optional view
    if view:
        v = napari.Viewer()
        v.add_image(deskewed)
        v.layers[-1].scale = voxel_size
        napari.run()


def get_deskewed_data_shape(
        raw_data_shape:tuple, 
        pixel_size_um:float, 
        ls_angle_deg:float, 
        px_to_scan_ratio:float, 
        keep_overhang:bool=True
):
    """Get the shape of the deskewed data set and its voxel size

    Parameters
    ----------
    raw_data_shape : tuple
        Shape of the raw data, must be len = 3
    pixel_size_um : float
        Pixel size in micrometers
    ls_angle_deg : float
        Angle of the light sheet relative to the optical axis, in degrees
    px_to_scan_ratio : float
        Ratio of the pixel size to light sheet scan step
    keep_overhang : bool

    Returns
    -------
    output_shape : tuple
        Output shape of the deskewed data in ZYX order
    voxel_size : tuple
        Size of the deskewed voxels in micrometers
    """

    # Trig
    theta = ls_angle_deg * np.pi / 180
    st = np.sin(theta)
    ct = np.cos(theta)

    # Prepare transforms
    Z, Y, X = raw_data_shape

    if keep_overhang:
        Xp = int(np.ceil((Z / px_to_scan_ratio) + (Y * ct)))
    else:
        Xp = int(np.ceil((Z / px_to_scan_ratio) - (Y * ct)))

    output_shape = (Y, X, Xp)
    voxel_size = (st * pixel_size_um, pixel_size_um, pixel_size_um)

    return output_shape, voxel_size


def mantis_deskew(raw_data, px_to_scan_ratio, ls_angle_deg, keep_overhang=True, order=1, cval=None):
    """Deskews fluorescence data from the mantis microscope

    Parameters
    ----------
    raw_data : NDArray with ndim == 3
        raw data from the mantis microscope
        - axis 0 corresponds to the scanning axis
        - axis 1 corresponds to the "tilted" axis
        - axis 2 corresponds to the axis in the plane of the coverslip
    px_to_scan_ratio : float
        (pixel spacing / scan spacing) in object space
        e.g. if camera pixels = 6.5 um and mag = 1.4*40, then the pixel spacing
        is 6.5/(1.4*40) = 0.116 um. If the scan spacing is 0.3 um, then
        px_to_scan_ratio = 0.116 / 0.3 = 0.386
    ls_angle_deg : float
        angle of light sheet with respect to the optical axis in degrees
    keep_overhang : bool
        If true, compute the whole volume within the tilted parallelepiped.
        If false, only compute the deskewed volume within a cuboid region.
    order : int, optional
        interpolation order (default 1 is linear interpolation)
    cval : float, optional
        fill value area outside of the measured volume (default None fills
        with the minimum value of the input array)

    Returns
    -------
    deskewed_data : NDArray with ndim == 3
        axis 0 is the Z axis, normal to the coverslip
        axis 1 is the Y axis, input axis 2 in the plane of the coverslip
        axis 2 is the X axis, the scanning axis
    """
    if cval is None:
        cval = np.min(np.ravel(raw_data))

    # Prepare transforms
    Z, Y, X = raw_data.shape

    ct = np.cos(ls_angle_deg * np.pi / 180)
    Z_shift = 0
    if not keep_overhang:
        Z_shift = int(np.floor(Y * ct * px_to_scan_ratio))
    
    matrix = np.array(
        [
            [
                -px_to_scan_ratio * ct,
                0,
                px_to_scan_ratio,
                Z_shift,
            ],
            [-1, 0, 0, Y - 1],
            [0, -1, 0, X - 1],
        ]
    )
    output_shape, _ = get_deskewed_data_shape(
        raw_data.shape, 1, ls_angle_deg, px_to_scan_ratio, keep_overhang
    )

    # Apply transforms
    deskewed_data = scipy.ndimage.affine_transform(
        raw_data,
        matrix,
        output_shape=output_shape,
        order=order,
        cval=cval,
    )

    # Return transformed data with its voxel dimensions
    return deskewed_data


if __name__ == "__main__":
    deskew()
