import glob
import os

from dataclasses import asdict
from pathlib import Path

import ants
import click
import napari
import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from skimage.transform import EuclideanTransform, SimilarityTransform
from waveorder.focus import focus_from_transverse_band

from mantis.analysis.AnalysisSettings import EstimateTransformSettings
from mantis.cli import utils
from mantis.cli.parsing import (
    config_filepath,
    labelfree_position_dirpaths,
    lightsheet_position_dirpaths,
    output_dirpath,
)
from mantis.cli.utils import yaml_to_model

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_PHASE = 1.35
NA_DETECTION_FLUOR = 1.35
WAVELENGTH_EMISSION_PHASE_CHANNEL = 0.45  # [um]
WAVELENGTH_EMISSION_FLUOR_CHANNEL = 0.6  # [um]
FOCUS_SLICE_ROI_SIDE = 150


@click.command()
@labelfree_position_dirpaths()
@lightsheet_position_dirpaths()
@config_filepath()
@output_dirpath()
def estimate_phase_to_fluor_affine(
    labelfree_position_dirpaths, lightsheet_position_dirpaths, config_filepath, output_dirpath
):
    """
    Estimate the affine transform between two channels (source channel and target channel) by manual inputs.

    mantis estimate-phase-to-fluor-affine -lf ./acq_name_labelfree_reconstructed.zarr/0/0/0 -ls ./acq_name_lightsheet_deskewed.zarr/0/0/0 -c ./config.yml -o ./output_folder
    """

    # # Get a napari viewer()
    viewer = napari.Viewer()

    settings = yaml_to_model(config_filepath, EstimateTransformSettings)
    phase_channel_idx = settings.label_free_channel_idx
    fluor_channel_idx = settings.light_sheet_channel_idx
    pre_affine_90degree_rotations_about_z = settings.light_sheet_channel_idx

    click.echo(f"Estimating registration with labelfree_channel idx {phase_channel_idx}")

    click.echo("Loading data and estimating best focus plane...")

    # Display volumes rescaled
    with open_ome_zarr(labelfree_position_dirpaths[0], mode="r") as phase_channel_position:
        phase_channel_str = phase_channel_position.channel_names[phase_channel_idx]

        phase_channel_Z, phase_channel_Y, phase_channel_X = phase_channel_position[0].shape[
            -3:
        ]
        phase_channel_volume = phase_channel_position[0][0, phase_channel_idx]

        phase_channel_Z, phase_channel_Y, phase_channel_X = phase_channel_volume.shape[-3:]

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
                phase_channel_position[0].shape[-2] // 2
                - FOCUS_SLICE_ROI_SIDE : phase_channel_position[0].shape[-2] // 2
                + FOCUS_SLICE_ROI_SIDE,
                phase_channel_position[0].shape[-1] // 2
                - FOCUS_SLICE_ROI_SIDE : phase_channel_position[0].shape[-1] // 2
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
        fluor_channel_Z, fluor_channel_Y, fluor_channel_X = fluor_channel_position[0].shape[
            -3:
        ]
        fluor_channel_volume = fluor_channel_position[0][0, fluor_channel_idx]
        fluor_channel_Z, fluor_channel_Y, fluor_channel_X = fluor_channel_volume.shape[-3:]

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
                fluor_channel_position[0].shape[-2] // 2
                - FOCUS_SLICE_ROI_SIDE : fluor_channel_position[0].shape[-2] // 2
                + FOCUS_SLICE_ROI_SIDE,
                fluor_channel_position[0].shape[-1] // 2
                - FOCUS_SLICE_ROI_SIDE : fluor_channel_position[0].shape[-1] // 2
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
    print(f"scaling factor z {scaling_factor_z}, scaling factor xy {scaling_factor_yx}")
    # Add layers to napari with and transform
    # Rotate the image if needed here

    # Convert to ants objects
    phase_zyx_ants = ants.from_numpy(phase_channel_volume.astype(np.float32))
    fluor_zyx_ants = ants.from_numpy(fluor_channel_volume.astype(np.float32))

    scaling_affine = utils.scale_affine(
        (fluor_channel_Z, fluor_channel_Y, fluor_channel_X),
        (scaling_factor_z, scaling_factor_yx, scaling_factor_yx),
        (fluor_channel_Z, fluor_channel_Y, fluor_channel_X),
    )
    rotate90_affine = utils.rotate_affine(
        (phase_channel_Z, phase_channel_Y, phase_channel_X),
        90 * pre_affine_90degree_rotations_about_z,
        (fluor_channel_Z, fluor_channel_Y, fluor_channel_X),
    )
    compound_affine = scaling_affine @ rotate90_affine

    # NOTE: these two functions are key to pass the function properly to ANTs
    compound_affine_ants_style = compound_affine[:, :-1].ravel()
    compound_affine_ants_style[-3:] = compound_affine[:3, -1]

    # Ants affine transforms
    tx_manual = ants.new_ants_transform()
    tx_manual.set_parameters(compound_affine_ants_style)
    tx_manual = tx_manual.invert()
    phase_zxy_pre_reg = tx_manual.apply_to_image(phase_zyx_ants, reference=fluor_zyx_ants)

    layer_phase_channel = viewer.add_image(phase_zxy_pre_reg.numpy(), name=phase_channel_str)
    layer_fluor_channel = viewer.add_image(fluor_channel_volume, name=fluor_channel_str)
    Z_phase_pre_reg, Y_phase_pre_reg, X_phase_pre_reg = layer_phase_channel.data.shape[-3:]

    layer_fluor_channel.translate = (
        0,
        0,
        X_phase_pre_reg,
    )
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
                    prev_step_fluor_channel = (next_layer.data[-1][0], 0, 0)
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
                    prev_step_phase_channel = (next_layer.data[-1][0], 0, 0)
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
    # subtract the translation offset for display
    pts_fluor_channel[:, 2] -= X_phase_pre_reg

    # Estimate the affine transform between the points xy to make sure registration is good
    transform = EuclideanTransform()
    transform.estimate(pts_phase_channel[:, 1:], pts_fluor_channel[:, 1:])
    yx_points_transformation_matrix = transform.params

    z_translation = pts_fluor_channel[0, 0] - pts_phase_channel[0, 0]

    z_scale_translate_matrix = np.array([[1, 0, 0, z_translation]])

    # 2D to 3D matrix
    euclidian_transform = np.vstack(
        (z_scale_translate_matrix, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
    )  # Insert 0 in the third entry of each row

    scaling_affine = utils.scale_affine(
        (1, fluor_channel_Y, fluor_channel_X),
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

    phase_zxy_manual_reg = tx_manual.apply_to_image(phase_zyx_ants, reference=fluor_zyx_ants)

    print(
        "Showing registered pair (phase and fluorescence) with pseudo colored fluorescence in magenta"
    )
    viewer.add_image(
        fluor_zyx_ants.numpy(),
        name=f"target_fluor_{fluor_channel_str}",
        colormap="magenta",
    )
    viewer.add_image(
        phase_zxy_manual_reg.numpy(), name=f"registered_{phase_channel_str}", opacity=0.5
    )
    viewer.layers.remove(f"pts_{phase_channel_str}")
    viewer.layers.remove(f"pts_{fluor_channel_str}")
    viewer.layers[fluor_channel_str].visible = False
    viewer.layers[phase_channel_str].visible = False

    if settings.virtual_staining_path is not None:
        click.echo("Running optimizer between virtual staining and fluorescence")
        # Load the virtual stained volume
        VS_position = open_ome_zarr(settings.virtual_staining_path)

        click.echo(f"Using virtual staining channel_idx {settings.virtual_staining_channel}")
        VS_data_zyx = VS_position[0][0, settings.virtual_staining_channel].astype(np.float32)
        VS_zyx_ants = ants.from_numpy(VS_data_zyx)

        # Affine Transforms
        VS_zyx_prep = tx_manual.apply_to_image(VS_zyx_ants, reference=fluor_zyx_ants)
        viewer.add_image(
            VS_zyx_prep.numpy(), name="VS_pre_opt", colormap="cyan", opacity="0.5"
        )

        print("RUNNING THE OPTIMIZER")
        # Optimization
        tx_opt = ants.registration(
            fixed=fluor_zyx_ants,
            moving=VS_zyx_prep,
            type_of_transform="Similarity",
            verbose=settings.optimizer_verbose,
        )
        print(f"optimized {tx_opt}")

        VS_registered = ants.apply_transforms(
            fixed=fluor_zyx_ants, moving=VS_zyx_prep, transformlist=tx_opt["fwdtransforms"]
        )
        viewer.add_image(
            VS_registered.numpy(),
            name="VS_registered",
            colormap="cyan",
            blending="additive",
        )
        viewer.add_image(
            fluor_channel_position[0][0, fluor_channel_idx],
            name="Fluroescence Target",
            colormap="magenta",
            blending="additive",
        )
        tx_opt_mat = ants.read_transform(tx_opt["fwdtransforms"][0])
        tx_composed = ants.compose_ants_transforms(
            [tx_opt_mat, tx_manual]
        )  # this works with their recon
        VS_test = tx_composed.apply_to_image(VS_zyx_ants, reference=fluor_zyx_ants)
        viewer.add_image(VS_test.numpy(), name="vs test")

        # Save the transformation
        output_dirpath.mkdir(parents=True, exist_ok=True)
        ants.write_transform(tx_opt_mat, output_dirpath / "tx_opt.mat")
        ants.write_transform(tx_manual, output_dirpath / "tx_manual.mat")
    else:
        click.echo("Saving affine transforms")
        identity_T = np.eye(4)
        identity_T_ants_style = identity_T[:, :-1].ravel()

        # Ants affine transforms
        tx_opt = ants.new_ants_transform()
        tx_opt.set_parameters(identity_T_ants_style)
        output_dirpath.mkdir(parents=True, exist_ok=True)
        ants.write_transform(tx_opt, output_dirpath / "tx_opt.mat")
        ants.write_transform(tx_manual, output_dirpath / "tx_manual.mat")

    input("\n Displaying registered channels. Press <enter> to close...")
