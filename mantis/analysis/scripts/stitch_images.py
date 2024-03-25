# %% This script creates an stitched image of a multi-position dataset given
# average column translation and row translation distances.

import os

from pathlib import Path

import napari
import numpy as np

from iohub import open_ome_zarr
from skimage.transform import SimilarityTransform, warp

os.environ["DISPLAY"] = ':1005'


def stitch_images(
    data_path: Path,
    percent_overlap: float = None,
    col_translation: float | tuple[float, float] = None,
    row_translation: float | tuple[float, float] = None,
    channels: list[str] = None,
) -> np.ndarray:
    """
    Stitch multiple images together to create a larger composite image.

    Args:
        data_path (Path):
            The path to the data.
        percent_overlap (float, optional):
            The percentage of overlap between adjacent images. Must be between 0 and 1. Defaults to None.
        col_translation (float | tuple[float, float], optional):
            The translation distance in pixels in the column direction. Can be a single value or a tuple
            of (x_translation, y_translation) when moving across columns. Defaults to None.
        row_translation (float | tuple[float, float], optional):
            See col_translation. Defaults to None.
        channels (list[str], optional):
            List of channels to stitch. If not provided all channels will be stitched.

    Returns:
        np.ndarray: The stitched composite image.

    Raises:
        AssertionError: If percent_overlap is not between 0 and 1.

    """
    dataset = open_ome_zarr(data_path)
    dataset_channels = dataset.channel_names
    if channels is None:
        channels = dataset_channels

    assert all(
        channel in dataset_channels for channel in channels
    ), "Invalid channel(s) provided."

    # TODO: expand to multichannel stitching
    num_channels = len(dataset_channels)
    assert num_channels == 1, "Only single channel stitching is currently supported."

    grid_rows = set()
    grid_cols = set()
    well_name, well = next(dataset.wells())
    for position_name, position in well.positions():
        fov_name = Path(position_name).parts[-1]
        grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
        grid_cols.add(fov_name[:3])
    sizeT, sizeC, sizeZ, sizeY, sizeX = position.data.shape
    grid_rows = sorted(grid_rows)
    grid_cols = sorted(grid_cols)
    n_rows = len(grid_rows)
    n_cols = len(grid_cols)

    # TODO: expand to ND images
    assert (sizeT, sizeC, sizeZ) == (1, 1, 1), "Only 2D images are currently supported."

    if percent_overlap is not None:
        assert 0 <= percent_overlap <= 1, "percent_overlap must be between 0 and 1"
        col_translation = sizeX * (1 - percent_overlap)
        row_translation = sizeY * (1 - percent_overlap)
    if not isinstance(col_translation, tuple):
        col_translation = (col_translation, 0)
    if not isinstance(row_translation, tuple):
        row_translation = (0, row_translation)

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
    # output_shape = (sizeY * len(grid_rows), sizeX * len(grid_cols))
    stitched_array = np.zeros((1, 1, 1) + xy_output_shape, dtype=np.float32)

    for channel in channels:
        channel_index = dataset_channels.index(channel)
        for row_idx, row_name in enumerate(grid_rows):
            for col_idx, col_name in enumerate(grid_cols):
                image = dataset[Path(well_name, col_name + row_name)].data[0, channel_index, 0]
                transform = SimilarityTransform(
                    translation=(
                        col_translation[0] * col_idx
                        + row_translation[0] * row_idx
                        + global_translation[0],
                        col_translation[1] * col_idx
                        + row_translation[1] * row_idx
                        + global_translation[1],
                    )
                )
                # transform = SimilarityTransform(
                #     translation=(964*col_idx + 7.00*row_idx, -6.73*col_idx + 966*row_idx)
                # )
                warped_image = warp(
                    image, transform.inverse, output_shape=xy_output_shape, order=0
                )
                overlap = np.logical_and(stitched_array, warped_image)
                stitched_array[..., :, :] += warped_image
                stitched_array[..., overlap] /= 2  # average blending in the overlapping region

    return stitched_array


# %%
data_dir = Path(
    '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/0-convert/'
)
dataset = 'grid_test_3.zarr'
data_path = data_dir / dataset
stitch_channels = ['Default']

# stitched_array = stitch_images(data_path, percent_overlap=0.05)
# stitched_array = stitch_images(
#     data_path, col_translation=(964, 7.00), row_translation=(-6.73, 966)
# )
# stitched_array = stitch_images(
#     data_path, col_translation=(964, -6.73), row_translation=(7.00, 966)
# )
stitched_array = stitch_images(
    data_path,
    col_translation=(967.9, -7.45),
    row_translation=(7.78, 969),
    channels=stitch_channels,
)

# %%
viewer = napari.Viewer()
viewer.add_image(stitched_array == 0)
viewer.add_image(stitched_array)

# %% Save as zarr file

save_dir = Path(
    '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/test-register'
)
save_path = save_dir / dataset

with open_ome_zarr(data_path) as ds:
    well_name, _ = next(ds.wells())

with open_ome_zarr(
    save_path,
    layout='hcs',
    mode='w-',
    channel_names=stitch_channels,
) as ds:
    position = ds.create_position(*well_name.split('/'), '0')
    position.create_image(
        name="0",
        data=stitched_array,
        chunks=(1, 1, 1, 4096, 4096),
    )

# %%
