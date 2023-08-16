import os

from dataclasses import asdict

import click
import napari
import numpy as np
import scipy
import yaml

from iohub import open_ome_zarr
from skimage.transform import SimilarityTransform
from waveorder.focus import focus_from_transverse_band

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli.parsing import (
    labelfree_position_dirpaths,
    lightsheet_position_dirpaths,
    output_filepath,
)

# TODO: see if at some point these globals should be hidden or exposed.
FOCUS_SLICE_ROI_SIDE = 150
NA_DETECTION_PHASE = 1.35
NA_DETECTION_FLUOR = 1.35
WAVELENGTH_EMISSION_PHASE_CHANNEL = 0.45  # [um]
WAVELENGTH_EMISSION_FLUOR_CHANNEL = 0.6  # [um]

# TODO:the current pipeline always assumes we register to fluoresence mcherry/mScarlet channel so it will change the colormaps to magenta


@click.command()
@labelfree_position_dirpaths()
@lightsheet_position_dirpaths()
@output_filepath()
@click.option(
    "--pre-affine-90degree-rotations-about-z",
    "-k",
    default=1,
    help="Pre-affine 90degree rotations about z",
    required=False,
    type=int,
)
def estimate_phase_to_fluor_affine(
    labelfree_position_dirpaths,
    lightsheet_position_dirpaths,
    output_filepath,
    pre_affine_90degree_rotations_about_z,
):
    """
    Estimate the affine transform between two channels (source channel and target channel) by manual inputs.

    mantis estimate-phase-to-fluor-affine -lf ./acq_name_labelfree_reconstructed.zarr/0/0/0 -ls ./acq_name_lightsheet_deskewed.zarr/0/0/0 -o ./register.yml
    """
    assert str(output_filepath).endswith(('.yaml', '.yml')), "Output file must be a YAML file."

    # Get a napari viewer()
    viewer = napari.Viewer()

    print("Getting dataset info")
    print("\n phase channel INFO:")
    os.system(f"iohub info {labelfree_position_dirpaths[0]}")
    print("\n fluorescence channel INFO:")
    os.system(f"iohub info {lightsheet_position_dirpaths[0]} ")

    phase_channel_idx = int(input("Enter phase_channel index to process: "))
    fluor_channel_idx = int(input("Enter fluor_channel index to process: "))

    click.echo("Loading data and estimating best focus plane...")

    # Display volumes rescaled
    with open_ome_zarr(labelfree_position_dirpaths[0], mode="r") as phase_channel_position:
        phase_channel_str = phase_channel_position.channel_names[phase_channel_idx]
        phase_channel_volume = phase_channel_position[0][0, phase_channel_idx]
        phase_channel_Z, phase_channel_Y, phase_channel_X = phase_channel_volume.shape
        # Get the voxel dimensions in sample space
        (
            z_sample_space_phase_channel,
            y_sample_space_phase_channel,
            x_sample_space_phase_channel,
        ) = phase_channel_position.scale[-3:]

        # Find the infocus slice
        focus_phase_channel_idx = focus_from_transverse_band(
            phase_channel_position[0][
                0,
                phase_channel_idx,
                :,
                phase_channel_Y // 2
                - FOCUS_SLICE_ROI_SIDE : phase_channel_Y // 2
                + FOCUS_SLICE_ROI_SIDE,
                phase_channel_X // 2
                - FOCUS_SLICE_ROI_SIDE : phase_channel_X // 2
                + FOCUS_SLICE_ROI_SIDE,
            ],
            NA_det=NA_DETECTION_PHASE,
            lambda_ill=WAVELENGTH_EMISSION_PHASE_CHANNEL,
            pixel_size=x_sample_space_phase_channel,
            plot_path="./best_focus_phase.svg",
        )
    click.echo(f"Best focus phase z_idx: {focus_phase_channel_idx}")

    with open_ome_zarr(lightsheet_position_dirpaths[0], mode="r") as fluor_channel_position:
        fluor_channel_str = fluor_channel_position.channel_names[fluor_channel_idx]
        fluor_channel_volume = fluor_channel_position[0][0, fluor_channel_idx]
        fluor_channel_Z, fluor_channel_Y, fluor_channel_X = fluor_channel_volume.shape
        # Get the voxel dimension in sample space
        (
            z_sample_space_fluor_channel,
            y_sample_space_fluor_channel,
            x_sample_space_fluor_channel,
        ) = fluor_channel_position.scale[-3:]

        # Finding the infocus plane
        focus_fluor_channel_idx = focus_from_transverse_band(
            fluor_channel_position[0][
                0,
                fluor_channel_idx,
                :,
                fluor_channel_Y // 2
                - FOCUS_SLICE_ROI_SIDE : fluor_channel_Y // 2
                + FOCUS_SLICE_ROI_SIDE,
                fluor_channel_X // 2
                - FOCUS_SLICE_ROI_SIDE : fluor_channel_X // 2
                + FOCUS_SLICE_ROI_SIDE,
            ],
            NA_det=NA_DETECTION_FLUOR,
            lambda_ill=WAVELENGTH_EMISSION_FLUOR_CHANNEL,
            pixel_size=x_sample_space_fluor_channel,
            plot_path="./best_focus_fluor.svg",
        )
    click.echo(f"Best focus fluor z_idx: {focus_fluor_channel_idx}")

    # Calculate scaling factors for displaying data
    scaling_factor_z = z_sample_space_phase_channel / z_sample_space_fluor_channel
    scaling_factor_yx = x_sample_space_phase_channel / x_sample_space_fluor_channel

    # Add layers to napari with and transform
    # Rotate the image if needed here
    phase_channel_volume_rotated = np.rot90(
        phase_channel_volume, k=pre_affine_90degree_rotations_about_z, axes=(1, 2)
    )
    layer_phase_channel = viewer.add_image(
        phase_channel_volume_rotated, name=phase_channel_str
    )
    layer_fluor_channel = viewer.add_image(fluor_channel_volume, name=fluor_channel_str)
    layer_phase_channel.scale = (scaling_factor_z, scaling_factor_yx, scaling_factor_yx)
    Z_rot, Y_rot, X_rot = phase_channel_volume_rotated.shape
    layer_fluor_channel.translate = (
        0,
        0,
        phase_channel_Y * scaling_factor_yx,
    )

    # %%
    # Manual annotation of features
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

    def next_on_click(layer, event, in_focus):
        if layer.mode == "add":
            if layer is points_phase_channel:
                next_layer = points_fluor_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_fluor_channel = (
                        in_focus[1],
                        0,
                        0,
                    )
                else:
                    prev_step_fluor_channel = (next_layer.data[-1][0] + 1, 0, 0)
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
                viewer.dims.current_step = prev_step_fluor_channel

            else:
                next_layer = points_phase_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_phase_channel = (
                        in_focus[0] * scaling_factor_z,
                        0,
                        0,
                    )
                else:
                    # TODO: this +1 is not clear to me?
                    prev_step_phase_channel = (next_layer.data[-1][0] + 1, 0, 0)
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
                viewer.dims.current_step = prev_step_phase_channel

    # Create the first points layer
    points_phase_channel = viewer.add_points(
        ndim=3, name=f"pts_{phase_channel_str}", size=50, face_color=COLOR_CYCLE[0]
    )
    points_fluor_channel = viewer.add_points(
        ndim=3, name=f"pts_{fluor_channel_str}", size=50, face_color=COLOR_CYCLE[0]
    )

    # Create the second points layer
    viewer.layers.selection.active = points_phase_channel
    points_phase_channel.mode = "add"
    points_fluor_channel.mode = "add"
    # Bind the mouse click callback to both point layers
    in_focus = (focus_phase_channel_idx, focus_fluor_channel_idx)

    def lambda_callback(layer, event):
        return next_on_click(layer=layer, event=event, in_focus=in_focus)

    viewer.dims.current_step = (
        in_focus[0] * scaling_factor_z,
        0,
        0,
    )
    points_phase_channel.mouse_drag_callbacks.append(lambda_callback)
    points_fluor_channel.mouse_drag_callbacks.append(lambda_callback)

    input(
        "\n Add at least three points in the two channels by sequentially clicking a feature on phase channel and its corresponding feature in fluorescence channel.  Press <enter> when done..."
    )

    # Get the data from the layers
    pts_phase_channel = points_phase_channel.data
    pts_fluor_channel = points_fluor_channel.data

    # De-apply the scaling and translation that was applied in the viewer
    pts_phase_channel[:, 1:] /= scaling_factor_yx
    pts_fluor_channel[:, 2] -= (
        phase_channel_Y * scaling_factor_yx
    )  # subtract the translation offset for display

    # Estimate the affine transform between the points xy to make sure registration is good
    transform = SimilarityTransform()
    transform.estimate(pts_phase_channel[:, 1:], pts_fluor_channel[:, 1:])
    yx_points_transformation_matrix = transform.params

    z_shift = np.array([1, 0, 0, 0])

    # 2D to 3D matrix
    zyx_affine_transform = np.vstack(
        (z_shift, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
    )  # Insert 0 in the third entry of each row
    zyx_affine_transform = np.linalg.inv(zyx_affine_transform)
    # Get the transformation matrix
    output_shape_zyx = (fluor_channel_Z, fluor_channel_Y, fluor_channel_X)

    # Demo: apply the affine transform to the image at the z-slice where all the points are located
    aligned_image = scipy.ndimage.affine_transform(
        phase_channel_volume_rotated[
            int(np.ceil(pts_phase_channel[0, 0])) : int(np.ceil(pts_phase_channel[0, 0])) + 1
        ],
        zyx_affine_transform,
        output_shape=(1, fluor_channel_Y, fluor_channel_X),
    )
    viewer.add_image(
        fluor_channel_position[0][
            0,
            fluor_channel_idx,
            int(np.ceil(pts_fluor_channel[0, 0])) : int(np.ceil(pts_fluor_channel[0, 0])) + 1,
        ],
        name=f"middle_plane_{fluor_channel_str}",
        colormap="magenta",
    )
    print(
        'Showing registered pair (phase and fluorescence) with pseudo colored fluorescence in magenta'
    )
    viewer.add_image(aligned_image, name=f"registered_{phase_channel_str}", opacity=0.5)
    viewer.layers.remove(f"pts_{phase_channel_str}")
    viewer.layers.remove(f"pts_{fluor_channel_str}")
    viewer.layers[fluor_channel_str].visible = False
    viewer.layers[phase_channel_str].visible = False
    viewer.dims.current_step = (0, 0, 0)  # Return to slice 0

    # NOTE: This assumes within a channel will lie in the same plane
    # Compute the 3D registration
    # Estimate the Similarity Transform (rotation,scaling,translation)
    transform = SimilarityTransform()
    transform.estimate(pts_phase_channel[:, 1:], pts_fluor_channel[:, 1:])
    yx_points_transformation_matrix = transform.params
    z_translation = pts_fluor_channel[0, 0] - pts_phase_channel[0, 0]

    z_scale_translate_matrix = np.array([[scaling_factor_z, 0, 0, z_translation]])
    zyx_affine_transform = np.vstack(
        (z_scale_translate_matrix, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
    )

    # Composite of all transforms
    zyx_affine_transform = np.linalg.inv(zyx_affine_transform)  # phase to fluorescence mapping
    print(f"Affine Transform Matrix:\n {zyx_affine_transform}\n")
    settings = RegistrationSettings(
        affine_transform_zyx=zyx_affine_transform.tolist(),  # phase to fluorescence mapping
        output_shape_zyx=list(output_shape_zyx),
        pre_affine_90degree_rotations_about_z=pre_affine_90degree_rotations_about_z,
    )

    print(f"Writing registration parameters to {output_filepath}")
    with open(output_filepath, "w") as f:
        yaml.dump(asdict(settings), f)

    # Apply the transformation to 3D volume
    flag_apply_3D_transform = input("\n Apply 3D registration *this make some time* (Y/N) :")
    if flag_apply_3D_transform == "Y" or flag_apply_3D_transform == "y":
        print("Applying 3D Affine Transform...")
        # Rotate the image first

        phase_volume_rotated = np.rot90(
            phase_channel_position[0][0, phase_channel_idx],
            k=pre_affine_90degree_rotations_about_z,
            axes=(1, 2),
        )

        registered_3D_volume = scipy.ndimage.affine_transform(
            phase_volume_rotated,
            zyx_affine_transform,
            output_shape=output_shape_zyx,
        )
        viewer.add_image(
            registered_3D_volume,
            name=f"registered_volume_{phase_channel_str}",
            opacity=1.0,
        )

        viewer.add_image(
            fluor_channel_position[0][0, fluor_channel_idx],
            name=f"{fluor_channel_str}",
            opacity=0.5,
            colormap="magenta",
        )

        viewer.layers[f"registered_{phase_channel_str}"].visible = False
        viewer.layers[f"{phase_channel_str}"].visible = False
        viewer.layers[f"middle_plane_{fluor_channel_str}"].visible = False

    input("\n Displaying registered channels. Press <enter> to close...")
