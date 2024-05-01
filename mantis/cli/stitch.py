from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import ProcessingSettings, StitchSettings
from mantis.analysis.stitch import (
    get_grid_rows_cols,
    get_stitch_output_shape,
    shift_image,
    stitch_images,
)
from mantis.cli.parsing import config_filepath, output_dirpath
from mantis.cli.utils import yaml_to_model


def process_dataset(
    data_array: np.ndarray,
    settings: ProcessingSettings,
    verbose: bool = True,
) -> np.ndarray:
    if settings:
        if settings.flipud:
            if verbose:
                click.echo("Flipping data array up-down")
            data_array = np.flip(data_array, axis=-2)
        if settings.fliplr:
            if verbose:
                click.echo("Flipping data array left-right")
            data_array = np.flip(data_array, axis=-1)

    return data_array


def _preprocess_and_shift(
    image, settings: ProcessingSettings, output_shape, shift, verbose=True
):
    return shift_image(process_dataset(image, settings, verbose), output_shape, shift, verbose)


def _stitch_shifted_store(
    input_data_path, output_data_path, settings: ProcessingSettings, verbose=True
):
    click.echo(f'Stitching zarr store: {input_data_path}')
    with open_ome_zarr(input_data_path, mode="r") as input_dataset:
        well_name, _ = next(input_dataset.wells())
        _, sample_position = next(input_dataset.positions())
        array_shape = sample_position.data.shape
        channels = input_dataset.channel_names

        stitched_array = np.zeros(array_shape, dtype=np.float32)
        denominator = np.zeros(array_shape, dtype=np.uint8)

        j = 0
        for _, position in input_dataset.positions():
            if verbose:
                click.echo(f'Processing position {j}')
            stitched_array += position.data
            denominator += np.bool_(position.data)
            j += 1

    denominator[denominator == 0] = 1
    stitched_array /= denominator

    stitched_array = process_dataset(stitched_array, settings, verbose)

    click.echo(f'Saving stitched array in :{output_data_path}')
    with open_ome_zarr(
        output_data_path, layout='hcs', channel_names=channels, mode="w-"
    ) as output_dataset:
        position = output_dataset.create_position(*Path(well_name, '0').parts)
        position.create_image('0', stitched_array, chunks=(1, 1, 1, 4096, 4096))


@click.command()
@click.option(
    "--input_dirpath",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path to the input Zarr store.",
)
@output_dirpath()
@config_filepath()
def stitch_zarr_store(
    input_dirpath: str,
    output_dirpath: str,
    config_filepath: str,
) -> None:
    """
    Stitch a Zarr store of multi-position data. Works well on grids with ~10 positions, but is rather slow
    on grids with ~1000 positions.

    Args:
        input_dirpath (str):
            The path to the input Zarr store.
        output_dirpath (str):
            The path to the output Zarr store. Channels will be appended is the store already exists.
        config_filepath (str):
            The path to the YAML file containing the stitching settings.
    """

    input_dataset = open_ome_zarr(input_dirpath)
    input_dataset_channels = input_dataset.channel_names
    well_name, _ = next(input_dataset.wells())
    _, sample_position = next(input_dataset.positions())
    sizeT, sizeC, sizeZ, sizeY, sizeX = sample_position.data.shape
    dtype = sample_position.data.dtype

    settings = yaml_to_model(config_filepath, StitchSettings)

    if settings.channels is None:
        settings.channels = input_dataset_channels

    assert all(
        channel in input_dataset_channels for channel in settings.channels
    ), "Invalid channel(s) provided."

    grid_rows, grid_cols = get_grid_rows_cols(input_dirpath)
    n_rows = len(grid_rows)
    n_cols = len(grid_cols)

    output_shape, _ = get_stitch_output_shape(
        n_rows, n_cols, sizeY, sizeX, settings.column_translation, settings.row_translation
    )

    if not Path(output_dirpath).exists():
        output_dataset = open_ome_zarr(
            output_dirpath,
            layout='hcs',
            mode='w-',
            channel_names=settings.channels,
        )
        output_position = output_dataset.create_position(*well_name.split('/'), '0')
        output_position.create_zeros(
            name="0",
            shape=(sizeT, len(settings.channels), sizeZ) + output_shape,
            dtype=np.float32,
            chunks=(1, 1, 1, 4096, 4096),
        )
        num_existing_output_channels = 0
    else:
        output_dataset = open_ome_zarr(output_dirpath, mode='r+')
        num_existing_output_channels = len(output_dataset.channel_names)
        _, output_position = next(output_dataset.positions())
        for channel_name in settings.channels:
            output_position.append_channel(channel_name)

    for t_idx in range(sizeT):
        for _c_idx, channel_name in enumerate(settings.channels):
            input_c_idx = input_dataset_channels.index(channel_name)
            output_c_idx = num_existing_output_channels + _c_idx
            for z_idx in range(sizeZ):
                data_array = np.zeros((n_rows, n_cols, sizeY, sizeX), dtype=dtype)
                for row_idx, row_name in enumerate(grid_rows):
                    for col_idx, col_name in enumerate(grid_cols):
                        data_array[row_idx, col_idx] = input_dataset[
                            Path(well_name, col_name + row_name)
                        ].data[t_idx, input_c_idx, z_idx]

                data_array = process_dataset(data_array, settings.preprocessing)
                stitched_array = stitch_images(
                    data_array,
                    col_translation=settings.column_translation,
                    row_translation=settings.row_translation,
                )
                stitched_array = process_dataset(stitched_array, settings.postprocessing)

                output_position.data[t_idx, output_c_idx, z_idx] = stitched_array

    input_dataset.close()
    output_dataset.zgroup.attrs.update({'stitching': settings.dict()})
    output_dataset.close()


if __name__ == '__main__':
    stitch_zarr_store()
