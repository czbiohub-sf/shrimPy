import os

import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import (
    config_filepath,
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from mantis.cli.utils import model_to_yaml, yaml_to_model

# TODO: maybe a CLI call?
T_IDX = 0


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@config_filepath()
@output_filepath()
@click.option(
    "--display-viewer",
    "-d",
    is_flag=True,
    help="Display the registered channels in a napari viewer",
)
@click.option(
    "--optimizer-verbose",
    "-v",
    is_flag=True,
    help="Show verbose output of optimizer",
)
def optimize_affine(
    source_position_dirpaths,
    target_position_dirpaths,
    config_filepath,
    output_filepath,
    display_viewer,
    optimizer_verbose,
):
    """
    Optimize the affine transform between source and target channels using ANTs library.

    mantis optimize-affine -s ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./transform.yml -o ./optimized_transform.yml -d -v
    """

    print("Getting dataset info")
    print("\n source channel INFO:")
    os.system(f"iohub info {source_position_dirpaths[0]}")
    print("\n target channel INFO:")
    os.system(f"iohub info {target_position_dirpaths[0]} ")

    settings = yaml_to_model(config_filepath, RegistrationSettings)

    click.echo("Loading source and target")

    # Load the source volume
    source_position = open_ome_zarr(source_position_dirpaths[0])
    source_channel_names = source_position.channel_names
    #NOTE: using the first channel in the config to register
    source_channel_index = source_channel_names.index(settings.source_channel_names[0])        
    source_data_zyx = source_position[0][T_IDX, source_channel_index].astype(np.float32)
    source_zyx_ants = ants.from_numpy(source_data_zyx)
    click.echo(f"Using source channel {source_position.channel_names[source_channel_index]}")

    # Load the target volume
    target_position = open_ome_zarr(target_position_dirpaths[0])
    target_channel_names = target_position.channel_names
    target_channel_index = target_channel_names.index(settings.target_channel_name)
    target_channel_zyx = target_position[0][T_IDX, target_channel_index]
    target_zyx_ants = ants.from_numpy(target_channel_zyx.astype(np.float32))
    click.echo(f"Using target channel {target_position.channel_names[target_channel_index]}")

    # Affine Transforms
    # numpy to ants
    T_pre_optimize_numpy = np.array(settings.affine_transform_zyx)
    T_pre_optimize = utils.numpy_to_ants_transform_zyx(T_pre_optimize_numpy)

    # Apply transformation to source prior to optimization of the matrix
    source_zyx_pre_optim = T_pre_optimize.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    click.echo("Running the optimizer...")
    # Optimization
    tx_opt = ants.registration(
        fixed=target_zyx_ants,
        moving=source_zyx_pre_optim,
        type_of_transform="Similarity",
        verbose=optimizer_verbose,
    )

    tx_opt_mat = ants.read_transform(tx_opt["fwdtransforms"][0])
    tx_opt_numpy = utils.ants_to_numpy_transform_zyx(tx_opt_mat)

    composed_matrix = T_pre_optimize_numpy @ tx_opt_numpy
    composed_matrix_ants = utils.numpy_to_ants_transform_zyx(composed_matrix)

    source_registered = composed_matrix_ants.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    # Saving the parameters
    click.echo(f"Writing registration parameters to {output_filepath}")

    # TODO: should this be model_to_yaml() from recOrder? Should it override the previous config?
    model = RegistrationSettings(
        source_channel_names=settings.source_channel_names,
        target_channel_name=settings.target_channel_name,
        affine_transform_zyx=composed_matrix.tolist(),
        source_shape_zyx=list(source_zyx_ants.numpy().shape),
        target_shape_zyx=list(target_zyx_ants.numpy().shape),
    )
    model_to_yaml(model, output_filepath)

    if display_viewer:
        viewer = napari.Viewer()
        source_pre_opt_layer = viewer.add_image(
            source_zyx_pre_optim.numpy(),
            name="source_pre_optimization",
            colormap="cyan",
            opacity=0.5,
        )
        source_pre_opt_layer.visible = False

        viewer.add_image(
            source_registered.numpy(),
            name="source_post_optimization",
            colormap="cyan",
            blending="additive",
        )
        viewer.add_image(
            target_position[0][0, target_channel_index],
            name="target",
            colormap="magenta",
            blending="additive",
        )
    
        input("\n Displaying registered channels. Press <enter> to close...")

    source_position.close()
    target_position.close()
