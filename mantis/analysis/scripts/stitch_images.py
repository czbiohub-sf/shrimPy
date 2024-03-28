# %% This script creates an stitched image of a multi-position dataset given
# average column translation and row translation distances.

import os

from pathlib import Path

import click
import numpy as np

# from skimage.transform import SimilarityTransform, warp
import scipy.ndimage as ndi

from iohub import open_ome_zarr

os.environ["DISPLAY"] = ':1005'


def get_stitch_output_shape(n_rows, n_cols, sizeY, sizeX, col_translation, row_translation):
    # TODO: test with non-square images and non-square grid
    global_translation = (
        np.ceil(np.abs(np.minimum(row_translation[0] * (n_rows - 1), 0))).astype(int),
        np.ceil(np.abs(np.minimum(col_translation[1] * (n_cols - 1), 0))).astype(int),
    )
    xy_output_shape = (
        np.ceil(
            sizeY
            + col_translation[1] * (n_cols - 1)
            + row_translation[1] * (n_rows - 1)
            + global_translation[1]
        ).astype(int),
        np.ceil(
            sizeX
            + col_translation[0] * (n_cols - 1)
            + row_translation[0] * (n_rows - 1)
            + global_translation[0]
        ).astype(int),
    )
    return xy_output_shape, global_translation


def stitch_images(
    data_array: np.ndarray,
    percent_overlap: float = None,
    col_translation: float | tuple[float, float] = None,
    row_translation: float | tuple[float, float] = None,
) -> np.ndarray:
    """
    Stitch an array of 2D images together to create a larger composite image.

    Args:
        data_array (np.ndarray):
            The data array to with shape (ROWS, COLS, Y, X) that will be stitched. Call this function multiple
            times to stitch multiple channels, slices, or time points.
        percent_overlap (float, optional):
            The percentage of overlap between adjacent images. Must be between 0 and 1. Defaults to None.
        col_translation (float | tuple[float, float], optional):
            The translation distance in pixels in the column direction. Can be a single value or a tuple
            of (x_translation, y_translation) when moving across columns. Defaults to None.
        row_translation (float | tuple[float, float], optional):
            See col_translation. Defaults to None.

    Returns:
        np.ndarray: The stitched composite 2D image

    Raises:
        AssertionError: If percent_overlap is not between 0 and 1.

    """

    n_rows, n_cols, sizeY, sizeX = data_array.shape

    if percent_overlap is not None:
        assert 0 <= percent_overlap <= 1, "percent_overlap must be between 0 and 1"
        col_translation = sizeX * (1 - percent_overlap)
        row_translation = sizeY * (1 - percent_overlap)
    if not isinstance(col_translation, tuple):
        col_translation = (col_translation, 0)
    if not isinstance(row_translation, tuple):
        row_translation = (0, row_translation)

    xy_output_shape, global_translation = get_stitch_output_shape(
        n_rows, n_cols, sizeY, sizeX, col_translation, row_translation
    )
    stitched_array = np.zeros(xy_output_shape, dtype=np.float32)
    input_image = np.zeros(xy_output_shape, dtype=np.float32)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            image = data_array[row_idx, col_idx]
            # #v1 - doesn't work as well with very large arrays
            # transform = SimilarityTransform(
            #     translation=(
            #         col_translation[0] * col_idx
            #         + row_translation[0] * row_idx
            #         + global_translation[0],
            #         col_translation[1] * col_idx
            #         + row_translation[1] * row_idx
            #         + global_translation[1],
            #     )
            # )
            # warped_image = warp(
            #     image, transform.inverse, output_shape=xy_output_shape, order=0
            # )
            shift = (
                col_translation[1] * col_idx
                + row_translation[1] * row_idx
                + global_translation[1],
                col_translation[0] * col_idx
                + row_translation[0] * row_idx
                + global_translation[0],
            )
            input_image[:sizeY, :sizeX] = image
            warped_image = ndi.shift(input_image, shift, order=0)
            overlap = np.logical_and(stitched_array, warped_image)
            stitched_array[:, :] += warped_image
            stitched_array[overlap] /= 2  # average blending in the overlapping region

    return stitched_array


@click.command()
@click.option(
    "--input_data_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path to the input Zarr store.",
)
@click.option(
    "--output_data_path",
    "-o",
    type=click.Path(),
    required=True,
    help="The path to the output Zarr store. Channels will be appended is the store already exists.",
)
@click.option(
    "--col_translation",
    type=(float, float),
    default=(967.9, -7.45),
    required=False,
    help="The translation distance in pixels in the column direction.",
)
@click.option(
    "--row_translation",
    type=(float, float),
    default=(7.78, 969),
    required=False,
    help="The translation distance in pixels in the row direction.",
)
@click.option(
    "--channels",
    "-c",
    multiple=True,
    help="List of channels to stitch. If not provided all channels will be stitched.",
)
def stitch_zarr_store(
    input_data_path: str,
    output_data_path: str,
    col_translation: float | tuple[float, float],
    row_translation: float | tuple[float, float],
    channels: list[str] = None,
) -> None:
    """
    Stitch a Zarr store of multi-position data.

    Args:
        input_data_path (Path):
            The path to the input Zarr store.
        output_data_path (Path):
            The path to the output Zarr store. Channels will be appended is the store already exists.
        col_translation (float | tuple[float, float], optional):
            The translation distance in pixels in the column direction. Can be a single value or a tuple
            of (x_translation, y_translation) when moving across columns.
        row_translation (float | tuple[float, float], optional):
            See col_translation.
        channels (list[str], optional):
            List of channels to stitch. If not provided all channels will be stitched.

    """

    input_dataset = open_ome_zarr(input_data_path)
    input_dataset_channels = input_dataset.channel_names
    if channels is None:
        channels = input_dataset_channels

    assert all(
        channel in input_dataset_channels for channel in channels
    ), "Invalid channel(s) provided."

    grid_rows = set()
    grid_cols = set()
    well_name, well = next(input_dataset.wells())
    for position_name, input_position in well.positions():
        fov_name = Path(position_name).parts[-1]
        grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
        grid_cols.add(fov_name[:3])
    sizeT, sizeC, sizeZ, sizeY, sizeX = input_position.data.shape
    dtype = input_position.data.dtype
    grid_rows = sorted(grid_rows)
    grid_cols = sorted(grid_cols)
    n_rows = len(grid_rows)
    n_cols = len(grid_cols)

    output_shape, _ = get_stitch_output_shape(
        n_rows, n_cols, sizeY, sizeX, col_translation, row_translation
    )

    if not Path(output_data_path).exists():
        output_dataset = open_ome_zarr(
            output_data_path,
            layout='hcs',
            mode='w-',
            channel_names=channels,
        )
        output_position = output_dataset.create_position(*well_name.split('/'), '0')
        output_position.create_zeros(
            name="0",
            shape=(sizeT, len(channels), sizeZ) + output_shape,
            dtype=np.float32,
            chunks=(1, 1, 1, 4096, 4096),
        )
        num_existing_output_channels = 0
    else:
        output_dataset = open_ome_zarr(output_data_path, mode='r+')
        num_existing_output_channels = len(output_dataset.channel_names)
        _, output_position = next(output_dataset.positions())
        for channel_name in channels:
            output_position.append_channel(channel_name)

    for t_idx in range(sizeT):
        for _c_idx, channel_name in enumerate(channels):
            input_c_idx = input_dataset_channels.index(channel_name)
            output_c_idx = num_existing_output_channels + _c_idx
            for z_idx in range(sizeZ):
                data_array = np.zeros((n_rows, n_cols, sizeY, sizeX), dtype=dtype)
                for row_idx, row_name in enumerate(grid_rows):
                    for col_idx, col_name in enumerate(grid_cols):
                        data_array[row_idx, col_idx] = input_dataset[
                            Path(well_name, col_name + row_name)
                        ].data[t_idx, input_c_idx, z_idx]
                stitched_array = stitch_images(
                    data_array,
                    col_translation=col_translation,
                    row_translation=row_translation,
                )
                output_position.data[t_idx, output_c_idx, z_idx] = stitched_array

    input_dataset.close()
    output_dataset.close()


if __name__ == '__main__':
    stitch_zarr_store()


# %%
# input_data_dir = Path(
#     '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/1-reconstruct'
# )
# output_data_dir = Path(
#     '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/live/3-stitch'
# )

# dataset = 'A3_600k_38x38_1.zarr'
# stitch_channels = ['Phase3D']

# stitch_zarr_store(
#     input_data_dir / dataset,
#     output_data_dir / dataset,
#     channels=stitch_channels,
#     col_translation=(967.9, -7.45),
#     row_translation=(7.78, 969),
# )

# %%
# import napari
# viewer = napari.Viewer()
# viewer.add_image(stitched_array == 0)
# viewer.add_image(stitched_array)
