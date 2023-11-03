import ants
import numpy as np
from iohub import open_ome_zarr
import napari
import click
from mantis.cli.utils import yaml_to_model, model_to_yaml
from mantis.analysis.AnalysisSettings import EstimateTransformSettings, RegistrationSettings
import yaml
from mantis.cli.parsing import (
    config_filepath,
    lightsheet_position_dirpaths,
    virtual_staining_position_dirpaths,
    output_filepath,
)
from dataclasses import asdict

from mantis.cli import utils

# TODO: maybe in config?
T_IDX = 0
OPTIMIZER_VERBOSE = False
# TODO: should the cli calls be source/target?


@click.command()
@virtual_staining_position_dirpaths()
@lightsheet_position_dirpaths()
@config_filepath()
@output_filepath()
def optimize_affine(
    virtual_staining_position_dirpaths,
    lightsheet_position_dirpaths,
    config_filepath,
    output_filepath,
):
    """
    Optimize the affine transform between two channels (source channel and target channel) by manual inputs.

    mantis optimize_affine -vs ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -ls ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./config.yml -o ./output.yml
    """
    settings = yaml_to_model(config_filepath, EstimateTransformSettings)
    target_channel_idx = settings.target_channel_idx
    source_channel_idx = settings.source_channel_idx

    # manual_registered_lf_layer.visible = False
    # target_target_layer.visible = False
    click.echo("Running optimizer between virtual staining and target")

    # Load the virtual stained volume [SOURCE]
    source_position = open_ome_zarr(virtual_staining_position_dirpaths[0])
    source_data_zyx = source_position[0][T_IDX, source_channel_idx].astype(np.float32)
    source_zyx_ants = ants.from_numpy(source_data_zyx)

    # Load the target volume [TARGET]
    target_channel_position = open_ome_zarr(lightsheet_position_dirpaths[0])
    target_channel_zyx = target_channel_position[0][T_IDX, target_channel_idx]
    target_zyx_ants = ants.from_numpy(target_channel_zyx.astype(np.float32))

    # Affine Transforms
    # numpy to ants
    T_pre_optimize_numpy = np.array(settings.affine_transform_zyx)
    T_pre_optimize = utils.numpy_to_ants_transform_zyx(T_pre_optimize_numpy)

    # Apply transformation to source prior to optimization of the matrix
    source_zyx_pre_optim = T_pre_optimize.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    click.echo("RUNNING THE OPTIMIZER")
    # Optimization
    tx_opt = ants.registration(
        fixed=target_zyx_ants,
        moving=source_zyx_pre_optim,
        type_of_transform="Similarity",
        verbose=OPTIMIZER_VERBOSE,
    )

    tx_opt_mat = ants.read_transform(tx_opt["fwdtransforms"][0])
    tx_opt_numpy = utils.ants_to_numpy_transform_zyx(tx_opt_mat)

    composed_matrix = tx_opt_numpy @ T_pre_optimize_numpy
    composed_matrix_ants = utils.numpy_to_ants_transform_zyx(composed_matrix)

    source_registered = composed_matrix_ants.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )
    source_registered = ants.apply_transforms(
        fixed=target_zyx_ants,
        moving=source_zyx_pre_optim,
        transformlist=tx_opt["fwdtransforms"],
    )

    # Saving the parameters
    click.echo(f"Writing registration parameters to {output_filepath}")

    # TODO: should this be model_to_yaml() from recOrder? Should it override the previous config?
    model = RegistrationSettings(
        affine_transform_zyx=composed_matrix.tolist(),
        output_shape_zyx=list(target_zyx_ants.numpy().shape),
    )
    model_to_yaml(model, output_filepath)

    if settings.display_viewer is True:
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
            target_channel_position[0][0, target_channel_idx],
            name="target",
            colormap="magenta",
            blending="additive",
        )

        input("\n Displaying registered channels. Press <enter> to close...")
