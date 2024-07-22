import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.analysis.register import convert_transform_to_ants, convert_transform_to_numpy
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
def optimize_registration(
    source_position_dirpaths,
    target_position_dirpaths,
    config_filepath,
    output_filepath,
    display_viewer,
    optimizer_verbose,
):
    """
    Optimize the affine transform between source and target channels using ANTs library.

    Start by generating an initial affine transform with `estimate-registration`.

    >> mantis optimize-registration -s ./acq_name_virtual_staining_reconstructed.zarr/0/0/0 -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./transform.yml -o ./optimized_transform.yml -d -v
    """

    settings = yaml_to_model(config_filepath, RegistrationSettings)

    # Load the source volume
    with open_ome_zarr(source_position_dirpaths[0]) as source_position:
        source_channel_names = source_position.channel_names
        # NOTE: using the first channel in the config to register
        source_channel_index = source_channel_names.index(settings.source_channel_names[0])
        source_channel_name = source_channel_names[source_channel_index]
        source_data_zyx = source_position[0][T_IDX, source_channel_index].astype(np.float32)

    # Load the target volume
    with open_ome_zarr(target_position_dirpaths[0]) as target_position:
        target_channel_names = target_position.channel_names
        target_channel_index = target_channel_names.index(settings.target_channel_name)
        target_channel_name = target_channel_names[target_channel_index]
        target_channel_zyx = target_position[0][T_IDX, target_channel_index]

    source_zyx_ants = ants.from_numpy(source_data_zyx)
    target_zyx_ants = ants.from_numpy(target_channel_zyx.astype(np.float32))
    click.echo(
        f"\nOptimizing registration using source channel {source_channel_name} and target channel {target_channel_name}"
    )

    # Affine Transforms
    # numpy to ants
    T_pre_optimize_numpy = np.array(settings.affine_transform_zyx)
    T_pre_optimize = convert_transform_to_ants(T_pre_optimize_numpy)

    # Apply transformation to source prior to optimization of the matrix
    source_zyx_pre_optim = T_pre_optimize.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    click.echo("Running ANTS optimizer...")
    # Optimization
    tx_opt = ants.registration(
        fixed=target_zyx_ants,
        moving=source_zyx_pre_optim,
        type_of_transform="Similarity",
        verbose=optimizer_verbose,
    )

    tx_opt_mat = ants.read_transform(tx_opt["fwdtransforms"][0])
    tx_opt_numpy = convert_transform_to_numpy(tx_opt_mat)
    composed_matrix = T_pre_optimize_numpy @ tx_opt_numpy

    # Saving the parameters
    click.echo(f"Writing registration parameters to {output_filepath}")
    # copy config settings and modify only ones that change
    output_settings = settings.copy()
    output_settings.affine_transform_zyx = composed_matrix.tolist()
    model_to_yaml(output_settings, output_filepath)

    if display_viewer:
        composed_matrix_ants = convert_transform_to_ants(composed_matrix)
        source_registered = composed_matrix_ants.apply_to_image(
            source_zyx_ants, reference=target_zyx_ants
        )

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


if __name__ == "__main__":
    optimize_registration()
