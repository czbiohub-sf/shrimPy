from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.stitch import get_grid_rows_cols, get_stitch_output_shape, stitch_images


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
    Stitch a Zarr store of multi-position data. Works well on grids with ~10 positions, but is rather slow
    on grids with ~1000 positions.

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
    well_name, _ = next(input_dataset.wells())
    _, sample_position = next(input_dataset.positions())
    sizeT, sizeC, sizeZ, sizeY, sizeX = sample_position.data.shape
    dtype = sample_position.data.dtype

    if channels is None:
        channels = input_dataset_channels

    assert all(
        channel in input_dataset_channels for channel in channels
    ), "Invalid channel(s) provided."

    grid_rows, grid_cols = get_grid_rows_cols(input_data_path)
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
