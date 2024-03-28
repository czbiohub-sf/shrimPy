import numpy as np
import scipy.ndimage as ndi


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
