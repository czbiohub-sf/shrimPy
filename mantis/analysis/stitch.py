from pathlib import Path
from typing import Literal

import click
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from iohub import open_ome_zarr
from skimage.registration import phase_cross_correlation

from mantis.analysis.AnalysisSettings import ProcessingSettings


def estimate_shift(
    im0: np.ndarray, im1: np.ndarray, percent_overlap: float, direction: Literal["row", "col"]
):
    assert 0 <= percent_overlap <= 1, "percent_overlap must be between 0 and 1"
    assert direction in ["row", "col"], "direction must be either 'row' or 'col'"
    assert im0.shape == im1.shape, "Images must have the same shape"

    sizeY, sizeX = im0.shape[-2:]

    # TODO: there may be a one pixel error in the estimated shift
    if direction == "row":
        y_roi = int(sizeY * np.minimum(percent_overlap + 0.05, 1))
        shift, _, _ = phase_cross_correlation(
            im0[-y_roi:, :], im1[:y_roi, :], upsample_factor=10
        )
        shift[0] += sizeY - y_roi
    elif direction == "col":
        x_roi = int(sizeX * np.minimum(percent_overlap + 0.05, 1))
        shift, _, _ = phase_cross_correlation(
            im0[:, -x_roi:], im1[:, :x_roi], upsample_factor=10
        )
        shift[1] += sizeX - x_roi

    # TODO: we shouldn't need to flip the order
    return shift[::-1]


def get_grid_rows_cols(dataset_path: str):
    grid_rows = set()
    grid_cols = set()

    with open_ome_zarr(dataset_path) as dataset:

        _, well = next(dataset.wells())
        for position_name, _ in well.positions():
            fov_name = Path(position_name).parts[-1]
            grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
            grid_cols.add(fov_name[:3])

    return sorted(grid_rows), sorted(grid_cols)


def get_stitch_output_shape(n_rows, n_cols, sizeY, sizeX, col_translation, row_translation):
    """
    Compute the output shape of the stitched image and the global translation when only col and row translation are given
    """
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


def get_image_shift(col_idx, row_idx, col_translation, row_translation, global_translation):
    """
    Compute total translation when only col and row translation are given
    """
    total_translation = (
        col_translation[1] * col_idx + row_translation[1] * row_idx + global_translation[1],
        col_translation[0] * col_idx + row_translation[0] * row_idx + global_translation[0],
    )

    return total_translation


def shift_image(
    czyx_data: np.ndarray,
    yx_output_shape: tuple[float, float],
    yx_shift: tuple[float, float],
    verbose: bool = False,
) -> np.ndarray:
    assert czyx_data.ndim == 4, "Input data must be a CZYX array"
    C, Z, Y, X = czyx_data.shape

    if verbose:
        print(f"Shifting image by {yx_shift}")
    # Create array of output_shape and put input data at (0, 0)
    output = np.zeros((C, Z) + yx_output_shape, dtype=np.float32)
    output[..., :Y, :X] = czyx_data

    return ndi.shift(output, (0, 0) + tuple(yx_shift), order=0)


def _stitch_images(
    data_array: np.ndarray,
    total_translation: dict[str : tuple[float, float]] = None,
    percent_overlap: float = None,
    col_translation: float | tuple[float, float] = None,
    row_translation: float | tuple[float, float] = None,
) -> np.ndarray:
    """
    Deprecated method to stitch an array of 2D images together to create a larger composite image.

    Args:
        data_array (np.ndarray):
            The data array to with shape (ROWS, COLS, Y, X) that will be stitched. Call this function multiple
            times to stitch multiple channels, slices, or time points.
        total_translation (dict[str: tuple[float, float]], optional):
            Shift to be applied to each fov, given as {fov: (y_shift, x_shift)}. Defaults to None.
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

    if total_translation is None:
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
    else:
        df = pd.DataFrame.from_dict(
            total_translation, orient="index", columns=["shift-y", "shift-x"]
        )
        xy_output_shape = (
            np.ceil(df["shift-y"].max() + sizeY).astype(int),
            np.ceil(df["shift-x"].max() + sizeX).astype(int),
        )
    stitched_array = np.zeros(xy_output_shape, dtype=np.float32)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            image = data_array[row_idx, col_idx]

            if total_translation is None:
                shift = get_image_shift(
                    col_idx, row_idx, col_translation, row_translation, global_translation
                )
            else:
                shift = total_translation[f"{col_idx:03d}{row_idx:03d}"]

            warped_image = shift_image(image, xy_output_shape, shift)
            overlap = np.logical_and(stitched_array, warped_image)
            stitched_array[:, :] += warped_image
            stitched_array[overlap] /= 2  # average blending in the overlapping region

    return stitched_array


def process_dataset(
    data_array: np.ndarray | da.Array,
    settings: ProcessingSettings,
    verbose: bool = True,
) -> np.ndarray:
    flip = np.flip
    if isinstance(data_array, da.Array):
        flip = da.flip

    if settings:
        if settings.flipud:
            if verbose:
                click.echo("Flipping data array up-down")
            data_array = flip(data_array, axis=-2)

        if settings.fliplr:
            if verbose:
                click.echo("Flipping data array left-right")
            data_array = flip(data_array, axis=-1)

    return data_array


def preprocess_and_shift(
    image,
    settings: ProcessingSettings,
    output_shape: tuple[int, int],
    shift_x: float,
    shift_y: float,
    verbose=True,
):
    return shift_image(
        process_dataset(image, settings, verbose), output_shape, (shift_y, shift_x), verbose
    )


def blend(array: da.Array, method: Literal["average"] = "average"):
    """
    Blend array of pre-shifted images stacked across axis=0

    Args:
        array (da.Array): Input dask array
        method (str, optional): Blending method. Defaults to "average".

    Raises:
        NotImplementedError: Raise error is blending method is not implemented.

    Returns:
        da.Array: Stitched array
    """
    if method == "average":
        # Sum up all images
        array_sum = array.sum(axis=0)
        # Count how many images contribute to each pixel in the stitched image
        array_bool_sum = (array != 0).sum(axis=0)
        # Replace 0s with 1s to avoid division by zero
        array_bool_sum[array_bool_sum == 0] = 1
        # Divide the sum of images by the number of images contributing to each pixel
        stitched_array = array_sum / array_bool_sum
    else:
        raise NotImplementedError(f"Blending method {method} is not implemented")

    return stitched_array


def stitch_shifted_store(
    input_data_path: str,
    output_data_path: str,
    settings: ProcessingSettings,
    blending="average",
    verbose=True,
):
    """
    Stitch a zarr store of pre-shifted images.

    Args:
        input_data_path (str): Path to the input zarr store
        output_data_path (str): Path to the output zarr store
        settings (ProcessingSettings): Postprocessing settings
        blending (str, optional): Blending method. Defaults to "average".
        verbose (bool, optional): Defaults to True.
    """
    click.echo(f'Stitching zarr store: {input_data_path}')
    with open_ome_zarr(input_data_path, mode="r") as input_dataset:
        for well_name, well in input_dataset.wells():
            if verbose:
                click.echo(f'Processing well {well_name}')

            # Stack images along axis=0
            dask_array = da.stack(
                [da.from_zarr(pos.data) for _, pos in well.positions()], axis=0
            )

            # Blend images
            stitched_array = blend(dask_array, method=blending)

            # Postprocessing
            stitched_array = process_dataset(stitched_array, settings, verbose)

            # Save stitched array
            click.echo('Computing and writing data')
            with open_ome_zarr(
                Path(output_data_path, well_name, '0'), mode="a"
            ) as output_image:
                da.to_zarr(stitched_array, output_image['0'])
            click.echo(f'Finishing writing data for well {well_name}')


def estimate_zarr_fov_shifts(
    fov0_zarr_path: str,
    fov1_zarr_path: str,
    tcz_index: tuple[int, int, int],
    percent_overlap: float,
    fliplr: bool,
    flipud: bool,
    direction: Literal["row", "col"],
    output_dirname: str = None,
):
    fov0_zarr_path = Path(fov0_zarr_path)
    fov1_zarr_path = Path(fov1_zarr_path)
    well_name = Path(*fov0_zarr_path.parts[-3:-1])
    fov0 = fov0_zarr_path.name
    fov1 = fov1_zarr_path.name
    click.echo(f'Estimating shift between FOVs {fov0} and {fov1} in well {well_name}...')

    T, C, Z = tcz_index
    im0 = open_ome_zarr(fov0_zarr_path).data[T, C, Z]
    im1 = open_ome_zarr(fov1_zarr_path).data[T, C, Z]

    if fliplr:
        im0 = np.fliplr(im0)
        im1 = np.fliplr(im1)
    if flipud:
        im0 = np.flipud(im0)
        im1 = np.flipud(im1)

    shift = estimate_shift(im0, im1, percent_overlap, direction)

    df = pd.DataFrame(
        {
            "well": str(well_name),
            "fov0": fov0,
            "fov1": fov1,
            "shift-x": shift[0],
            "shift-y": shift[1],
            "direction": direction,
        },
        index=[0],
    )
    click.echo(f'Estimated shift:\n {df.to_string(index=False)}')

    if output_dirname:
        df.to_csv(
            Path(output_dirname, f"{'_'.join(well_name.parts + (fov0, fov1))}_shift.csv"),
            index=False,
        )
    else:
        return df


def consolidate_zarr_fov_shifts(
    input_dirname: str,
    output_filepath: str,
):
    # read all csv files in input_dirname and combine into a single dataframe
    csv_files = Path(input_dirname).rglob("*_shift.csv")
    df = pd.concat(
        [pd.read_csv(csv_file, dtype={'fov0': str, 'fov1': str}) for csv_file in csv_files],
        ignore_index=True,
    )
    df.to_csv(output_filepath, index=False)


def cleanup_shifts(csv_filepath: str):
    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})
    df['shift-x-raw'] = df['shift-x']
    df['shift-y-raw'] = df['shift-y']

    # replace row shifts with median value calculated across all columns
    _df = df[df['direction'] == 'row']
    # group by well and last three characters of fov0
    groupby = _df.groupby(['well', _df['fov0'].str[-3:]])
    df.loc[df['direction'] == 'row', 'shift-x'] = groupby['shift-x-raw'].transform('median')
    df.loc[df['direction'] == 'row', 'shift-y'] = groupby['shift-y-raw'].transform('median')

    df.to_csv(csv_filepath, index=False)


def compute_total_translation(csv_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(csv_filepath, dtype={'fov0': str, 'fov1': str})

    df['row'] = df['fov1'].str[-3:].astype(int)
    df['col'] = df['fov1'].str[:3].astype(int)
    df.set_index('fov1', inplace=True)
    df.sort_index(inplace=True)

    total_shift = []
    for well in df['well'].unique():
        _df = df[(df['direction'] == 'col') & (df['well'] == well)]
        col_shifts = _df.groupby('row')[['shift-x', 'shift-y']].cumsum()
        _df = df[(df['direction'] == 'row') & (df['well'] == well)]
        row_shifts = _df.groupby('col')[['shift-x', 'shift-y']].cumsum()
        _total_shift = col_shifts.add(row_shifts, fill_value=0)

        # add row 000000
        _total_shift = pd.concat(
            [pd.DataFrame({'shift-x': 0, 'shift-y': 0}, index=['000000']), _total_shift]
        )

        # add global offset to remove negative values
        _total_shift['shift-x'] += -np.minimum(_total_shift['shift-x'].min(), 0)
        _total_shift['shift-y'] += -np.minimum(_total_shift['shift-y'].min(), 0)
        _total_shift.set_index(well + '/' + _total_shift.index, inplace=True)
        total_shift.append(_total_shift)

    return pd.concat(total_shift)
