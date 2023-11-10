import os

import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr
from skimage.transform import EuclideanTransform
from waveorder.focus import focus_from_transverse_band

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import (
    output_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from mantis.cli.utils import model_to_yaml

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_SOURCE = 1.35
NA_DETECTION_TARGET = 1.35
WAVELENGTH_EMISSION_SOURCE_CHANNEL = 0.45  # [um]
WAVELENGTH_EMISSION_TARGET_CHANNEL = 0.6  # [um]
FOCUS_SLICE_ROI_WIDTH = 150  # size of central ROI used to find focal slice


@click.command()
@source_position_dirpaths()
@target_position_dirpaths()
@output_filepath()
def estimate_affine(source_position_dirpaths, target_position_dirpaths, output_filepath):
    """
    Estimate the affine transform between a source channel and a target with manual inputs.

    mantis estimate-affine
    -s ./acq_name_labelfree_reconstructed.zarr/0/0/0
    -t ./acq_name_lightsheet_deskewed.zarr/0/0/0
    -o ./output.yml
    """

    print("Getting dataset info")
    print("\n Target channel INFO:")
    os.system(f"iohub info {target_position_dirpaths[0]} ")
    print("\n Source channel INFO:")
    os.system(f"iohub info {source_position_dirpaths[0]}")

    target_channel_index = int(input("Enter target channel index to process: "))
    source_channel_index = int(input("Enter source channel index to process: "))
    pre_affine_90degree_rotations_about_z = int(
        input("Rotate the source channel by 90 degrees? (0, 1, or -1): ")
    )

    click.echo(f"Estimating registration with labelfree channel index {source_channel_index}")
    click.echo("Loading data and estimating best focus plane...")

    # Display volumes rescaled
    with open_ome_zarr(source_position_dirpaths[0], mode="r") as source_channel_position:
        source_channel_str = source_channel_position.channel_names[source_channel_index]

        source_channel_Z, source_channel_Y, source_channel_X = source_channel_position[
            0
        ].shape[-3:]
        source_channel_volume = source_channel_position[0][0, source_channel_index]

        source_channel_Z, source_channel_Y, source_channel_X = source_channel_volume.shape[-3:]

        # Get the voxel dimensions in sample space
        (
            z_sample_space_source_channel,
            y_sample_space_source_channel,
            x_sample_space_source_channel,
        ) = source_channel_position.scale[-3:]

        # Find the infocus slice
        focus_source_channel_idx = focus_from_transverse_band(
            source_channel_position[0][
                0,
                source_channel_index,
                :,
                source_channel_position[0].shape[-2] // 2
                - FOCUS_SLICE_ROI_WIDTH : source_channel_position[0].shape[-2] // 2
                + FOCUS_SLICE_ROI_WIDTH,
                source_channel_position[0].shape[-1] // 2
                - FOCUS_SLICE_ROI_WIDTH : source_channel_position[0].shape[-1] // 2
                + FOCUS_SLICE_ROI_WIDTH,
            ],
            NA_det=NA_DETECTION_SOURCE,
            lambda_ill=WAVELENGTH_EMISSION_SOURCE_CHANNEL,
            pixel_size=x_sample_space_source_channel,
        )
    click.echo(f"Best focus source z_idx: {focus_source_channel_idx}")

    with open_ome_zarr(target_position_dirpaths[0], mode="r") as target_channel_position:
        target_channel_str = target_channel_position.channel_names[target_channel_index]
        target_channel_Z, target_channel_Y, target_channel_X = target_channel_position[
            0
        ].shape[-3:]
        target_channel_volume = target_channel_position[0][0, target_channel_index]
        target_channel_Z, target_channel_Y, target_channel_X = target_channel_volume.shape[-3:]

        # Get the voxel dimension in sample space
        (
            z_sample_space_target_channel,
            y_sample_space_target_channel,
            x_sample_space_target_channel,
        ) = target_channel_position.scale[-3:]

        # Finding the infocus plane
        focus_target_channel_idx = focus_from_transverse_band(
            target_channel_position[0][
                0,
                target_channel_index,
                :,
                target_channel_position[0].shape[-2] // 2
                - FOCUS_SLICE_ROI_WIDTH : target_channel_position[0].shape[-2] // 2
                + FOCUS_SLICE_ROI_WIDTH,
                target_channel_position[0].shape[-1] // 2
                - FOCUS_SLICE_ROI_WIDTH : target_channel_position[0].shape[-1] // 2
                + FOCUS_SLICE_ROI_WIDTH,
            ],
            NA_det=NA_DETECTION_TARGET,
            lambda_ill=WAVELENGTH_EMISSION_TARGET_CHANNEL,
            pixel_size=x_sample_space_target_channel,
        )
    click.echo(f"Best focus target z_idx: {focus_target_channel_idx}")

    # Calculate scaling factors for displaying data
    scaling_factor_z = z_sample_space_source_channel / z_sample_space_target_channel
    scaling_factor_yx = x_sample_space_source_channel / x_sample_space_target_channel
    print(f"scaling factor z {scaling_factor_z}, scaling factor xy {scaling_factor_yx}")
    # Add layers to napari with and transform
    # Rotate the image if needed here

    # Convert to ants objects
    source_zyx_ants = ants.from_numpy(source_channel_volume.astype(np.float32))
    target_zyx_ants = ants.from_numpy(target_channel_volume.astype(np.float32))

    scaling_affine = utils.scale_affine(
        (target_channel_Z, target_channel_Y, target_channel_X),
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx),
        (target_channel_Z, target_channel_Y, target_channel_X),
    )
    rotate90_affine = utils.rotate_affine(
        (source_channel_Z, source_channel_Y, source_channel_X),
        90 * pre_affine_90degree_rotations_about_z,
        (target_channel_Z, target_channel_Y, target_channel_X),
    )
    compound_affine = scaling_affine @ rotate90_affine

    # NOTE: these two functions are key to pass the function properly to ANTs
    compound_affine_ants_style = compound_affine[:, :-1].ravel()
    compound_affine_ants_style[-3:] = compound_affine[:3, -1]

    # Ants affine transforms
    tx_manual = ants.new_ants_transform()
    tx_manual.set_parameters(compound_affine_ants_style)
    tx_manual = tx_manual.invert()
    source_zxy_pre_reg = tx_manual.apply_to_image(source_zyx_ants, reference=target_zyx_ants)

    # Get a napari viewer
    viewer = napari.Viewer()

    COLOR_CYCLE = [
        "white",
        "cyan",
        "lime",
        "orchid",
        "blue",
        "orange",
        "yellow",
        "magenta",
    ]

    viewer.add_image(target_channel_volume, name=target_channel_str)
    points_target_channel = viewer.add_points(
        ndim=3, name=f"pts_{target_channel_str}", size=50, face_color=COLOR_CYCLE[0]
    )

    viewer.add_image(
        source_zxy_pre_reg.numpy(),
        name=source_channel_str,
        blending='additive',
        colormap='bop blue',
    )
    points_source_channel = viewer.add_points(
        ndim=3, name=f"pts_{source_channel_str}", size=50, face_color=COLOR_CYCLE[0]
    )

    # setup viewer
    viewer.layers.selection.active = points_source_channel
    viewer.grid.enabled = False
    viewer.grid.stride = 2
    viewer.grid.shape = (-1, 2)
    points_source_channel.mode = "add"
    points_target_channel.mode = "add"

    # Manual annotation of features
    def next_on_click(layer, event, in_focus):
        if layer.mode == "add":
            if layer is points_source_channel:
                next_layer = points_target_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_target_channel = (
                        in_focus[1],
                        0,
                        0,
                    )
                else:
                    prev_step_target_channel = (next_layer.data[-1][0], 0, 0)
                # Add a point to the active layer
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color

                # Switch to the next layer
                next_layer.mode = "add"
                layer.selected_data = {}
                viewer.layers.selection.active = next_layer
                viewer.dims.current_step = prev_step_target_channel

            else:
                next_layer = points_source_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_source_channel = (
                        in_focus[0] * scaling_factor_z,
                        0,
                        0,
                    )
                else:
                    # TODO: this +1 is not clear to me?
                    prev_step_source_channel = (next_layer.data[-1][0], 0, 0)
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)
                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index + 1) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color

                # Switch to the next layer
                next_layer.mode = "add"
                layer.selected_data = {}
                viewer.layers.selection.active = next_layer
                viewer.dims.current_step = prev_step_source_channel

    # Bind the mouse click callback to both point layers
    in_focus = (focus_source_channel_idx, focus_target_channel_idx)

    def lambda_callback(layer, event):
        return next_on_click(layer=layer, event=event, in_focus=in_focus)

    viewer.dims.current_step = (
        in_focus[0] * scaling_factor_z,
        0,
        0,
    )
    points_source_channel.mouse_drag_callbacks.append(lambda_callback)
    points_target_channel.mouse_drag_callbacks.append(lambda_callback)

    input(
        "\nAdd at least three points in the two channels by sequentially clicking "
        + "on a feature in the source channel and its corresponding feature in target channel. "
        + "Select grid mode if you prefer side-by-side view. "
        + "Press <enter> when done..."
    )

    # Get the data from the layers
    pts_source_channel = points_source_channel.data
    pts_target_channel = points_target_channel.data

    # Estimate the affine transform between the points xy to make sure registration is good
    transform = EuclideanTransform()
    transform.estimate(pts_source_channel[:, 1:], pts_target_channel[:, 1:])
    yx_points_transformation_matrix = transform.params

    z_translation = pts_target_channel[0, 0] - pts_source_channel[0, 0]

    z_scale_translate_matrix = np.array([[1, 0, 0, z_translation]])

    # 2D to 3D matrix
    euclidian_transform = np.vstack(
        (z_scale_translate_matrix, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
    )  # Insert 0 in the third entry of each row

    scaling_affine = utils.scale_affine(
        (1, target_channel_Y, target_channel_X),
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx),
    )
    manual_estimated_transform = euclidian_transform @ compound_affine

    # NOTE: these two functions are key to pass the function properly to ANTs
    manual_estimated_transform_ants_style = manual_estimated_transform[:, :-1].ravel()
    manual_estimated_transform_ants_style[-3:] = manual_estimated_transform[:3, -1]

    # Ants affine transforms
    tx_manual = ants.new_ants_transform()
    tx_manual.set_parameters(manual_estimated_transform_ants_style)
    tx_manual = tx_manual.invert()

    source_zxy_manual_reg = tx_manual.apply_to_image(
        source_zyx_ants, reference=target_zyx_ants
    )

    print("Showing registered source image in magenta")
    viewer.add_image(
        source_zxy_manual_reg.numpy(),
        name=f"registered_{source_channel_str}",
        colormap="magenta",
        blending='additive',
    )
    viewer.layers.remove(f"pts_{source_channel_str}")
    viewer.layers.remove(f"pts_{target_channel_str}")
    viewer.layers[source_channel_str].visible = False

    click.echo("Saving affine transforms")
    # Ants affine transforms
    T_manual_numpy = utils.ants_to_numpy_transform_zyx(tx_manual)
    print(T_manual_numpy)

    # TODO: should this be model_to_yaml() from recOrder? Should it override the previous config?
    model = RegistrationSettings(
        source_channel_index=source_channel_index,
        target_channel_index=target_channel_index,
        affine_transform_zyx=T_manual_numpy.tolist(),
        output_shape_zyx=list(target_zyx_ants.numpy().shape),
    )
    click.echo(f"Writing registration parameters to {output_filepath}")
    model_to_yaml(model, output_filepath)

    input("\n Displaying registered channels. Press <enter> to close...")
