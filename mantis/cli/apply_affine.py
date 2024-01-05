from pathlib import Path
from typing import List

import click
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import (
    config_filepath,
    output_dirpath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from mantis.cli.utils import yaml_to_model


def apply_affine_to_scale(affine_matrix, input_scale):
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
def apply_affine(
    source_position_dirpaths: List[str],
    target_position_dirpaths: List[str],
    config_filepath: str,
    output_dirpath: str,
    num_processes: int,
):
    """
    Apply an affine transformation to a single position across T and C axes based on a registration config file

    >> mantis apply_affine -i ./acq_name_lightsheet_deskewed.zarr/*/*/* -c ./register.yml -o ./acq_name_registerred.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    # Handle single position or wildcard filepath
    click.echo(f"List of input_pos:{source_position_dirpaths} output_pos:{output_dirpath}")

    # Parse from the yaml file
    settings = yaml_to_model(config_filepath, RegistrationSettings)
    matrix = np.array(settings.affine_transform_zyx)
    source_shape_zyx = tuple(settings.source_shape_zyx)
    target_shape_zyx = tuple(settings.target_shape_zyx)
    keep_overhang = settings.keep_overhang

    # Calculate the output voxel size from the input scale and affine transform
    with open_ome_zarr(source_position_dirpaths[0]) as source_dataset:
        output_voxel_size = apply_affine_to_scale(matrix[:3, :3], source_dataset.scale[-3:])
        T, C, Z, Y, X = source_dataset.data.shape
        source_channel_names = source_dataset.channel_names

    with open_ome_zarr(target_position_dirpaths[0]) as target_dataset:
        target_channel_names = target_dataset.channel_names

    click.echo('\nREGISTRATION PARAMETERS:')
    click.echo(f'Affine transform: {matrix}')
    click.echo(f'Voxel size: {output_voxel_size}')

    # Logic to parse time indices
    if settings.time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    output_channel_names = target_channel_names
    if target_position_dirpaths == source_position_dirpaths:
        output_channel_names += source_channel_names

    # # Logic to parse channels
    # if settings.channels_to_register == 'all':
    #     input_channel_indeces = list(range(len(source_channel_names)))
    #     output_channel_indeces = list(range(len(input_channel_indeces)))
    # elif isinstance(settings.channels_to_register, list):
    #     # check list is integers or strings
    #     if all(isinstance(item, int) for item in settings.channels_to_register):
    #         assert len(settings.channels_to_register) <= len(source_channel_names)
    #         input_channel_indeces = settings.channels_to_register
    #         output_channel_indeces = input_channel_indeces
    #     else:
    #         # get channel indeces
    #         input_channel_indeces = [
    #             source_channel_names.index(c) for c in settings.channels_to_register
    #         ]
    #         output_channel_indeces = input_channel_indeces
    # elif isinstance(settings.channels_to_register, int):
    #     input_channel_indeces = list(settings.channels_to_register)
    #     output_channel_indeces = input_channel_indeces

    # # Logic to parse channels
    # if target_position_dirpaths == source_position_dirpaths:
    #     all_channel_names = source_channel_names
    #     tmp_non_processed_source_c_idx = [
    #         i
    #         for i in range(len(all_channel_names))
    #         if i not in input_channel_indeces and i < len(source_channel_names)
    #     ]
    #     non_processed_input_c_idx = [
    #         tmp_non_processed_source_c_idx.copy() for _ in range(len(source_position_dirpaths))
    #     ]
    #     non_processed_output_c_idx = non_processed_input_c_idx
    # else:
    #     all_channel_names = source_channel_names + target_channel_names
    #     print(f'all_channel_names: {all_channel_names}')
    #     # Check which channels are copyied over without processing
    #     tmp_non_processed_source_c_idx_input = [
    #         i
    #         for i in range(len(all_channel_names))
    #         if i not in input_channel_indeces and i <= len(source_channel_names) - 1
    #     ]
    #     tmp_non_processed_target_c_idx_input = [
    #         i - len(source_channel_names)
    #         for i in range(len(all_channel_names))
    #         if i not in input_channel_indeces and i > len(source_channel_names) - 1
    #     ]

    #     tmp_non_processed_source_c_idx_output = [
    #         i
    #         for i in range(len(all_channel_names))
    #         if i not in input_channel_indeces and i <= len(source_channel_names) - 1
    #     ]
    #     tmp_non_processed_target_c_idx_output = [
    #         i
    #         for i in range(len(all_channel_names))
    #         if i not in input_channel_indeces and i > len(source_channel_names) - 1
    #     ]

        # non_processed_source_c_idx_input = [
        #     tmp_non_processed_source_c_idx_input.copy()
        #     for _ in range(len(source_position_dirpaths))
        #     if len(tmp_non_processed_source_c_idx_input) > 0
        # ]
        # non_procesed_target_c_idx_input = [
        #     tmp_non_processed_target_c_idx_input.copy()
        #     for _ in range(len(target_position_dirpaths))
        #     if len(tmp_non_processed_target_c_idx_input) > 0
        # ]

        # non_processed_source_c_idx_output = [tmp_non_processed_source_c_idx_output.copy() for _ in range(len(source_position_dirpaths)) if len(tmp_non_processed_source_c_idx_output)>0]
        # non_procesed_target_c_idx_output = [tmp_non_processed_target_c_idx_output.copy() for _ in range(len(target_position_dirpaths)) if len(tmp_non_processed_target_c_idx_output)>0]

        # non_processed_input_c_idx = []
        # non_processed_output_c_idx = []
        # if len(tmp_non_processed_source_c_idx_input) > 0:
        #     non_processed_input_c_idx.extend(non_processed_source_c_idx_input)
        #     non_processed_output_c_idx.extend(non_processed_source_c_idx_output)
        # if len(tmp_non_processed_target_c_idx_input) > 0:
        #     non_processed_input_c_idx.extend(non_procesed_target_c_idx_input)
        #     non_processed_output_c_idx.extend(non_procesed_target_c_idx_output)

        # # Get matching paths to the non processed channels to copy over
        # non_proc_paths = []
        # if len(non_processed_source_c_idx_input) > 0:
        #     non_proc_paths.extend(source_position_dirpaths)
        # if len(non_procesed_target_c_idx_input) > 0:
        #     non_proc_paths.extend(target_position_dirpaths)

    # Find the largest interior rectangle
    print(
        'Find the cropping parameters for the largest interior rectangle of the affine transform'
    )
    if not keep_overhang:
        Z_slice, Y_slice, X_slice = utils.find_lir_slicing_params(
            source_shape_zyx, target_shape_zyx, matrix
        )
        # TODO: start or stop may be None
        target_shape_zyx = (
            Z_slice.stop - Z_slice.start,
            Y_slice.stop - Y_slice.start,
            X_slice.stop - X_slice.start,
        )
        Z, Y, X = target_shape_zyx[-3:]
        print(f'New target shape: {target_shape_zyx}')

    output_metadata = {
        "shape": (len(time_indices), len(output_channel_names), Z, Y, X),
        "chunks": None,
        "scale": (1,) * 2 + tuple(output_voxel_size),
        "channel_names": output_channel_names,
        "dtype": np.float32,
    }

    # Create the output zarr mirroring source_position_dirpaths
    utils.create_empty_hcs_zarr(
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
        'output_shape_zyx': settings.target_shape_zyx,
        'crop_output_slicing': ([Z_slice, Y_slice, X_slice] if not keep_overhang else None),
        'extra_metadata': extra_metadata,
    }

    copy_n_pase_kwargs = {
        "czyx_slicing_params": ([Z_slice, Y_slice, X_slice] if not keep_overhang else None),
    }

    # # # Loop over positions
    # for input_position_path in source_position_dirpaths:
    #     utils.process_single_position_v2(
    #         utils.affine_transform,
    #         input_data_path=input_position_path,
    #         output_path=output_dirpath,
    #         time_indices=time_indices,
    #         input_channel_idx=input_channel_indeces,
    #         output_channel_idx=output_channel_indeces,
    #         num_processes=num_processes,
    #         **affine_transform_args,
    #     )

    # # Crop and copy the data that did not need to be processed
    # for input_position_path, input_c_idx, output_c_idx in zip(
    #     non_proc_paths, non_processed_input_c_idx, non_processed_output_c_idx
    # ):
    #     utils.process_single_position_v2(
    #         utils.copy_n_paste_czyx,
    #         input_data_path=input_position_path,
    #         output_path=output_dirpath,
    #         time_indices=time_indices,
    #         input_channel_idx=input_c_idx,
    #         output_channel_idx=output_c_idx,
    #         num_processes=num_processes,
    #         **copy_n_pase_kwargs,
    #     )

    # NOTE: channels will not be processed in parallel
    # apply affine transform to all channels that need to be registered
    for input_position_path in source_position_dirpaths:
        for channel_name in source_channel_names:
            if channel_name in settings.source_channel_names:
                # processes single channel at a time
                utils.process_single_position_v2(
                    utils.affine_transform,
                    input_data_path=input_position_path, # source store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=source_channel_names.index(channel_name),
                    output_channel_idx=output_channel_names.index(channel_name),
                    num_processes=num_processes, # parallel processing over time
                    **affine_transform_args
                )
            else:
                utils.process_single_position_v2(
                    utils.copy_n_paste_czyx,
                    input_data_path=input_position_path, # target store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=source_channel_names.index(channel_name),
                    output_channel_idx=output_channel_names.index(channel_name), 
                    num_processes=num_processes,
                    **copy_n_pase_kwargs
                )

    for input_position_path in target_position_dirpaths:
        for channel_name in target_channel_names:
            if channel_name not in settings.source_channel_names:
                utils.process_single_position_v2(
                    utils.copy_n_paste_czyx,
                    input_data_path=input_position_path, # target store
                    output_path=output_dirpath,
                    time_indices=time_indices,
                    input_channel_idx=target_channel_names.index(channel_name),
                    output_channel_idx=output_channel_names.index(channel_name), 
                    num_processes=num_processes,
                    **copy_n_pase_kwargs,
                )

    # if target_position_dirpaths != source_position_dirpaths:
    #     for input_position_path in target_position_dirpaths:
    #         for channel_name in settings.target_channel_names:
    #             utils.process_single_position_v2(
    #                 utils.copy_n_paste_czyx,
    #                 input_data_path=input_position_path, # target store
    #                 output_path=output_dirpath,
    #                 time_indices=time_indices,
    #                 input_channel_idx=target_channel_names.index(channel_name),
    #                 output_channel_idx=output_channel_names.index(channel_name), 
    #                 num_processes=num_processes,
    #                 **copy_n_pase_kwargs, #fix
    #             )
    # else:
    #     for input_position_path in target_position_dirpaths:
    #         for channel_name in settings.target_channel_names:
    #             if channel_name not in settings.source_channel_names:
    #                 utils.process_single_position_v2(
    #                     utils.copy_n_paste_czyx,
    #                     input_data_path=input_position_path, # target store
    #                     output_path=output_dirpath,
    #                     time_indices=time_indices,
    #                     input_channel_idx=target_channel_names.index(channel_name),
    #                     output_channel_idx=output_channel_names.index(channel_name), 
    #                     num_processes=num_processes,
    #                     **copy_n_pase_kwargs, #fix
    #                 )


    # # copy over the channels that do not need to be processed


    #     for path, channel_name in zip(paths, settings.target_channel_names):
    #         # maybe apply only is cropping is enabled
    #         utils.process_single_position_v2(
    #             utils.copy_n_paste_czyx,
    #             input_data_path=input_position_path, # target store
    #             output_path=output_dirpath,
    #             time_indices=time_indices,
    #             input_channel_idx=input_c_idx,  #fix 
    #             output_channel_idx=output_c_idx, #fix
    #             num_processes=num_processes,
    #             **copy_n_pase_kwargs,
    #         )
            
    # else:
    #     for path, channel_name in zip(paths, settings.target_channel_names):
    #         if  channel_name is not in settings.source_channel_names:
    #             # maybe apply only is cropping is enabled
    #             utils.process_single_position_v2(
    #                 utils.copy_n_paste_czyx,
    #                 input_data_path=input_position_path, # target store
    #                 output_path=output_dirpath,
    #                 time_indices=time_indices,
    #                 input_channel_idx=input_c_idx,  #fix 
    #                 output_channel_idx=output_c_idx, #fix
    #                 num_processes=num_processes,
    #                 **copy_n_pase_kwargs,
    #             )

    # for path, channel_name in zip(paths, output_channel_names):
    #     if channel_name in settings.source_channel_names:
    #         # processes single channel at a time
    #         utils.process_single_position_v2(
    #             utils.affine_transform,
    #             input_data_path=path,
    #             output_path=output_dirpath,
    #             time_indices=time_indices,
    #             input_channel_idx=target_channel_names.index(channel_name), # fix
    #             output_channel_idx=output_channel_index, # fix
    #             num_processes=num_processes,
    #             **affine_transform_args,
    #         )

    #     else:
    #         # maybe apply only is cropping is enabled
    #         utils.process_single_position_v2(
    #             utils.copy_n_paste_czyx,
    #             input_data_path=input_position_path,
    #             output_path=output_dirpath,
    #             time_indices=time_indices,
    #             input_channel_idx=input_c_idx,
    #             output_channel_idx=output_c_idx,
    #             num_processes=num_processes,
    #             **copy_n_pase_kwargs,
    #         )
