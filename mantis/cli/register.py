from pathlib import Path
from typing import List

import click
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.analysis.register import apply_affine_transform, find_overlapping_volume
from mantis.cli.parsing import (
    config_filepath,
    output_dirpath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from mantis.cli.utils import (
    copy_n_paste_czyx,
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)


def rescale_voxel_size(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of parallel processes",
    required=False,
    type=int,
)
def register(
    source_position_dirpaths: List[str],
    target_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Apply an affine transformation to a single position across T and C axes based on a registration config file.

    Start by generating an initial affine transform with `estimate-register`. Optionally, refine this transform with `optimize-register`. Finally, use `register`.

    >> mantis register -s source.zarr/*/*/* -t target.zarr/*/*/* -c config.yaml -o ./acq_name_registerred.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Handle single position or wildcard filepath
    click.echo(f"\nInput positions: {[str(path) for path in source_position_dirpaths]}")
    click.echo(f"Output position: {output_dirpath}")

    # Parse from the yaml file
    settings = yaml_to_model(config_filepath, RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)
    keep_overhang = settings.keep_overhang

    # Calculate the output voxel size from the input scale and affine transform
    with open_ome_zarr(source_position_dirpaths[0]) as source_dataset:
        T, C, Z, Y, X = source_dataset.data.shape
        source_channel_names = source_dataset.channel_names
        source_shape_zyx = source_dataset.data.shape[-3:]
        source_voxel_size = source_dataset.scale[-3:]
        output_voxel_size = rescale_voxel_size(matrix[:3, :3], source_voxel_size)

    with open_ome_zarr(target_position_dirpaths[0]) as target_dataset:
        target_channel_names = target_dataset.channel_names
        target_shape_zyx = target_dataset.data.shape[-3:]

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Transformation matrix:\n{matrix}')
    click.echo(f'Voxel size: {output_voxel_size}')

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    output_channel_names = target_channel_names
    if target_position_dirpaths != source_position_dirpaths:
        output_channel_names += source_channel_names

    if not keep_overhang:
        # Find the largest interior rectangle
        click.echo('\nFinding largest overlapping volume between source and target datasets')
        Z_slice, Y_slice, X_slice = find_overlapping_volume(
            source_shape_zyx, target_shape_zyx, matrix
        )
        # TODO: start or stop may be None
        # Overwrite the previous target shape
        cropped_shape_zyx = (
            Z_slice.stop - Z_slice.start,
            Y_slice.stop - Y_slice.start,
            X_slice.stop - X_slice.start,
        )
        click.echo(f'Shape of cropped output dataset: {cropped_shape_zyx}\n')
    else:
        cropped_shape_zyx = target_shape_zyx
        Z_slice, Y_slice, X_slice = (
            slice(0, cropped_shape_zyx[-3]),
            slice(0, cropped_shape_zyx[-2]),
            slice(0, cropped_shape_zyx[-1]),
        )

    output_metadata = {
        "shape": (len(time_indices), len(output_channel_names)) + tuple(cropped_shape_zyx),
        "chunks": None,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": output_channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring source_position_dirpaths
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in source_position_dirpaths],
        **output_metadata,
    )

    # Get the affine transformation matrix
    # NOTE: add any extra metadata if needed:
    extra_metadata = {
        'affine_transformation': {
            'transform_matrix': matrix.tolist(),
        }
    }

    affine_transform_args = {
        'matrix': matrix,
        'output_shape_zyx': target_shape_zyx,  # NOTE: this should be the shape of the original target dataset
        'crop_output_slicing': ([Z_slice, Y_slice, X_slice] if not keep_overhang else None),
        'extra_metadata': extra_metadata,
    }

    copy_n_paste_kwargs = {"czyx_slicing_params": ([Z_slice, Y_slice, X_slice])}

    # NOTE: channels will not be processed in parallel
    # NOTE: the the source and target datastores may be the same (e.g. Hummingbird datasets)

    # apply affine transform to channels in the source datastore that should be registered
    # as given in the config file (i.e. settings.source_channel_names)
    for input_position_path in source_position_dirpaths:
        for channel_name in source_channel_names:
            if channel_name in settings.source_channel_names:
                process_single_position_v2(
                    apply_affine_transform,
                    input_data_path=input_position_path,  # source store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[source_channel_names.index(channel_name)],
                    output_channel_idx=[output_channel_names.index(channel_name)],
                    num_processes=num_processes,  # parallel processing over time
                    **affine_transform_args,
                )

    # crop all channels that are not being registered and save them in the output zarr store
    # Note: when target and source datastores are the same we don't process channels which
    # were already registered in the previous step
    for input_position_path in target_position_dirpaths:
        for channel_name in target_channel_names:
            if channel_name not in settings.source_channel_names:
                process_single_position_v2(
                    copy_n_paste_czyx,
                    input_data_path=input_position_path,  # target store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=[target_channel_names.index(channel_name)],
                    output_channel_idx=[output_channel_names.index(channel_name)],
                    num_processes=num_processes,
                    **copy_n_paste_kwargs,
                )


if __name__ == "__main__":
    register()
