import numpy as np
import scipy
import click
import deskew_settings
import yaml
import napari
import os
from iohub import read_micromanager, open_ome_zarr
from iohub.ngff_meta import TransformationMeta


@click.command()
@click.argument(
    "mmfolder",
    type=click.Path(exists=True),
)
@click.argument(
    "deskew_params_file",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    default="./output.zarr",
    required=False,
    help="Output zarrr",
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
def deskew(mmfolder, deskew_params_file, output, view, keep_overhang):
    """
    Deskews across P, T, C axes using a parameter file generated with estimate_deskew.py

    Usage: python deskew.py </path/to/mmfolder/> </path/to/deskew_params.yml>
    """

    # Load params
    with open(deskew_params_file) as file:
        raw_settings = yaml.safe_load(file)
    settings = deskew_settings.DeskewSettings(**raw_settings)
    print(f"Deskewing parameters: {settings}")

    # Prepare reader and writer
    if os.path.isdir(output):
        output = os.path.join(output, mmfolder.split(os.sep)[-2] + ".zarr")

    reader = read_micromanager(mmfolder)
    writer = open_ome_zarr(
        output, mode="w-", layout="hcs", channel_names=reader.channel_names
    )

    # Loop through (P, T, C), deskewing and writing as we go
    P, T, C = reader.get_num_positions(), reader.shape[0], reader.shape[1]
    for p in range(P):
        position = writer.create_position(0, p, 0)
        for t in range(T):
            for c in range(C):
                print(f"Deskewing c={c}/{C-1}, t={t}/{T-1}, p={p}/{P-1}")
                data = reader.get_array(p)[t, c, ...]  # zyx

                # Deskew
                deskewed, dims = mantis_deskew(
                    data, settings.px_to_scan_ratio, settings.theta_deg, keep_overhang
                )

                # Handle transforms and metadata
                transform = TransformationMeta(
                    type="scale",
                    scale=2 * (1,) + dims,
                )
                img = position.create_zeros(
                    name="0",
                    shape=(
                        T,
                        C,
                    )
                    + deskewed.shape,
                    dtype=np.int16,
                    transform=[transform],
                )

                img[t, c, ...] = deskewed  # write to zarr

    # Write metadata
    writer.zattrs["deskewing"] = settings.dict()
    writer.zattrs["mm-meta"] = reader.mm_meta["Summary"]

    # Optional view
    if view:
        v = napari.Viewer()
        v.add_image(deskewed)
        v.layers[-1].scale = dims
        napari.run()


def mantis_deskew(
    raw_data, px_to_scan_ratio, theta_deg, keep_overhang=True, order=1, cval=None
):
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
    keep_overhang : bool
        If true, compute the whole volume within the tilted parallelopipid.
        If false, only compute the deskewed volume within a cuboid region.
    theta_deg : float
        angle of light sheet with respect to the optical axis in degrees
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
    dimensions : tuple
        For convenience, (sin(theta), 1, 1) are the relative lengths of the
        returned voxels. i.e. the Z-axis voxels are a factor of sin(theta)
        shorter than the Y- and X-axis voxels.
    """
    if cval is None:
        cval = np.min(np.ravel(raw_data))

    # Trig
    theta = theta_deg * np.pi / 180
    st = np.sin(theta)
    ct = np.cos(theta)

    # Prepare transforms
    Z, Y, X = raw_data.shape

    if keep_overhang:
        Z_shift = 0
        Xp = int(np.ceil((Z / px_to_scan_ratio) + (Y * ct)))
    else:
        Z_shift = int(np.floor(Y * ct * px_to_scan_ratio))
        Xp = int(np.ceil((Z / px_to_scan_ratio) - (Y * ct)))

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
    output_shape = (Y, X, Xp)

    # Apply transforms
    deskewed_data = scipy.ndimage.affine_transform(
        raw_data,
        matrix,
        output_shape=output_shape,
        order=order,
        cval=cval,
    )

    # Return transformed data with its relative dimensions
    return deskewed_data, (st, 1, 1)


if __name__ == "__main__":
    deskew()
