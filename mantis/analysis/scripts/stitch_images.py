# %%
from pathlib import Path

import napari
import numpy as np

from iohub import open_ome_zarr
from skimage.transform import SimilarityTransform, warp


def stitch_images(
    data_path: Path,
    percent_overlap: float = None,
    col_translation: float | tuple[float, float] = None,
    row_translation: float | tuple[float, float] = None,
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

    Returns:
        np.ndarray: The stitched composite image.

    Raises:
        AssertionError: If percent_overlap is not between 0 and 1.

    """
    dataset = open_ome_zarr(data_path)

    grid_rows = set()
    grid_cols = set()
    well_name, well = next(dataset.wells())
    for position_name, position in well.positions():
        fov_name = Path(position_name).parts[-1]
        grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
        grid_cols.add(fov_name[:3])
    sizeY, sizeX = position.data.shape[-2:]
    grid_rows = sorted(grid_rows)
    grid_cols = sorted(grid_cols)

    if percent_overlap is not None:
        assert 0 <= percent_overlap <= 1, "percent_overlap must be between 0 and 1"
        col_translation = sizeX * (1 - percent_overlap)
        row_translation = sizeY * (1 - percent_overlap)
    if not isinstance(col_translation, tuple):
        col_translation = (col_translation, 0)
    if not isinstance(row_translation, tuple):
        row_translation = (0, row_translation)

    # TODO: calculate the output shape based on col and row translations
    output_shape = (sizeY * len(grid_rows), sizeX * len(grid_cols))
    stitched_array = np.zeros(output_shape, dtype=np.float32)

    for row_idx, row_name in enumerate(grid_rows):
        for col_idx, col_name in enumerate(grid_cols):
            image = dataset[Path(well_name, col_name + row_name)].data[0, 0, 0]
            transform = SimilarityTransform(
                translation=(
                    col_translation[0] * col_idx + col_translation[1] * row_idx,
                    row_translation[0] * col_idx + row_translation[1] * row_idx,
                )
            )
            # transform = SimilarityTransform(
            #     translation=(964*col_idx + 7.00*row_idx, -6.73*col_idx + 966*row_idx)
            # )
            warped_image = warp(image, transform.inverse, output_shape=output_shape, order=0)
            overlap = np.logical_and(stitched_array, warped_image)
            stitched_array += warped_image
            stitched_array[overlap] /= 2  # average blending in the overlapping region

    return stitched_array


# %%
data_dir = Path(
    '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/fixed/0-convert/'
)
dataset = 'grid_test_3.zarr'
data_path = data_dir / dataset

# stitched_array = stitch_images(data_path, percent_overlap=0.05)
stitched_array = stitch_images(
    data_path, col_translation=(964, 7.00), row_translation=(-6.73, 966)
)

# %%
viewer = napari.Viewer()
viewer.add_image(stitched_array)
# %%
