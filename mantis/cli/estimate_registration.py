import os

from dataclasses import asdict
from pathlib import Path

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

# TODO: remove this for non mantis datasets
ROTATE_90deg_CCW = True
# TODO: add option to use the autofocused slice


def find_focus_channel_pairs(
    phase_channel_data_path: Path,
    phase_channel_idx: int,
    fluor_channel_data_path: Path,
    fluor_channel_idx: int,
):
    """
    Find the focus sliced for input channels

    Parameters
    ----------
    phase_channel_data_path : Path
    phase_channel_idx : int
    fluor_channel_data_path : Path
    fluor_channel_idx : int
    """
    # TODO: get the microscope parameters through a yaml file? Currently hard-coded
    # Find focus phase_channel
    print("Finding the phase_channel best focus position at t=0")
    with open_ome_zarr(phase_channel_data_path, mode="r") as phase_channel_position:
        phase_channel_str = phase_channel_position.channel_names[phase_channel_idx]
        focus_phase_channel_idx = focus_from_transverse_band(
            phase_channel_position[0][0, phase_channel_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=3.45 / (46.2 * 1.4),
        )
        phase_channel_Z, phase_channel_Y, phase_channel_X = phase_channel_position[0][
            0, phase_channel_idx
        ].shape
        print(f"{phase_channel_str} focus idx: {focus_phase_channel_idx}")

    # Find focus fluor_channel
    print("Finding the fluor_channel best focus position at t=0")
    with open_ome_zarr(fluor_channel_data_path, mode="r") as fluor_channel_position:
        fluor_channel_position.channel_names[fluor_channel_idx]
        focus_fluor_channel_idx = focus_from_transverse_band(
            fluor_channel_position[0][0, fluor_channel_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=6.5 / (40 * 1.4),
        )
        fluor_channel_Z, fluor_channel_Y, fluor_channel_X = fluor_channel_position[0][
            0, fluor_channel_idx
        ].shape
        print(f"fluor_channel focus idx: {focus_fluor_channel_idx}")


@click.command()
@labelfree_position_dirpaths()
@lightsheet_position_dirpaths()
@output_filepath()
def estimate_registration(
    labelfree_position_dirpaths, lightsheet_position_dirpaths, output_filepath
):
    """
    Estimate the affine transform between two channels (source channel and target channel) by manual inputs.

    mantis estimate-registration -lf ./acq_name_labelfree_reconstructed.zarr/0/0/0 -ls ./acq_name_lightsheet_deskewed.zarr/0/0/0 -o ./register.yml
    """
    assert str(output_filepath).endswith(('.yaml', '.yml')), "Output file must be a YAML file."

    # Get a napari viewer()
    viewer = napari.Viewer()

    print("Getting dataset info")
    print("\n phase channel INFO:")
    os.system(f"iohub info {labelfree_position_dirpaths} ")
    print("\n fluorescence channel INFO:")
    os.system(f"iohub info {lightsheet_position_dirpaths} ")

    phase_channel_idx = int(input("Enter phase_channel index to process: "))
    fluor_channel_idx = int(input("Enter fluor_channel index to process: "))

    # Display volumes rescaled
    with open_ome_zarr(labelfree_position_dirpaths, mode="r") as phase_channel_position:
        phase_channel_str = phase_channel_position.channel_names[phase_channel_idx]
        phase_channel_volume = phase_channel_position[0][0, phase_channel_idx]
        phase_channel_Z, phase_channel_Y, phase_channel_X = phase_channel_volume.shape
    with open_ome_zarr(lightsheet_position_dirpaths, mode="r") as fluor_channel_position:
        fluor_channel_str = fluor_channel_position.channel_names[fluor_channel_idx]
        fluor_channel_volume = fluor_channel_position[0][0, fluor_channel_idx]
        fluor_channel_Z, fluor_channel_Y, fluor_channel_X = fluor_channel_volume.shape

    # Find the z-scaling and apply it for display
    # TODO: Get these values from the yaml file
    # Use deskew metadta
    try:
        (
            _,
            _,
            z_sampling_fluor_channel,
            y_sampling_fluor_channel,
            x_sampling_fluor_channel,
        ) = fluor_channel_position.zattrs["multiscales"][0]["coordinateTransformations"][0][
            "scale"
        ]
        voxel_size = (
            z_sampling_fluor_channel,
            y_sampling_fluor_channel,
            x_sampling_fluor_channel,
        )
    except LookupError("Couldn't find z_sampling in metadata"):
        z_sampling_fluor_channel = 1
        voxel_size = (1, 1, 1)
    magnification_LF_remote_volume = 1.4
    step_size_LF_remote_volume = 0.4  # [um]
    phase_remote_volume_sample_size = (
        step_size_LF_remote_volume / magnification_LF_remote_volume
    )
    scaling_factor_z = phase_remote_volume_sample_size / z_sampling_fluor_channel

    # Add layers to napari with and transform
    # TODO: these rotations and scaling are not required for all datasets
    layer_phase_channel = viewer.add_image(phase_channel_volume, name=phase_channel_str)
    layer_fluor_channel = viewer.add_image(fluor_channel_volume, name=fluor_channel_str)
    layer_phase_channel.scale = (scaling_factor_z, 1, 1)
    layer_fluor_channel.rotate = 90
    layer_fluor_channel.translate = (0, fluor_channel_X, phase_channel_X)

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

    def next_on_click(layer, event):
        if layer.mode == "add":
            if layer is points_phase_channel:
                next_layer = points_fluor_channel
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_fluor_channel = viewer.dims.current_step
                else:
                    prev_step_fluor_channel = (next_layer.data[-1][0] + 2, 0, 0)
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
                    prev_step_phase_channel = viewer.dims.current_step
                else:
                    # TODO: this +2 is not clear to me?
                    prev_step_phase_channel = (next_layer.data[-1][0] + 2, 0, 0)
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
    points_phase_channel.mouse_drag_callbacks.append(next_on_click)
    points_fluor_channel.mouse_drag_callbacks.append(next_on_click)

    input(
        "\n Add at least three points in the two channels by sequentially clicking a feature on phase channel and its corresponding feature in fluorescence channel.  Press <enter> when done..."
    )

    # Get the data from the layers
    pts_phase_channel = points_phase_channel.data
    pts_fluor_channel = points_fluor_channel.data

    pts_fluor_channel[:, 2] -= phase_channel_X  # subtract the translation offset for display

    # Estimate the affine transform between the points xy to make sure registration is good
    transform = SimilarityTransform()
    transform.estimate(pts_phase_channel[:, 1:], pts_fluor_channel[:, 1:])
    yx_points_transformation_matrix = transform.params

    # Add rotation matrix
    # TODO: make this optional: return the identity matrix if no rotation is needed.
    if ROTATE_90deg_CCW:
        rotation_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, fluor_channel_X - 1], [0, 0, 0, 1]]
        )
    else:
        rotation_matrix = np.eye(4)

    z_shift = np.array([1, 0, 0, 0])

    # 2D to 3D matrix
    zyx_points_transformation_matrix = np.vstack(
        (z_shift, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
    )  # Insert 0 in the third entry of each row

    zyx_affine_transform = rotation_matrix @ zyx_points_transformation_matrix

    # Get the transformation matrix
    print(f"Affine Transform Matrix:\n {zyx_affine_transform}\n")
    output_shape_volume = (fluor_channel_Z, fluor_channel_Y, fluor_channel_X)

    # Demo: apply the affine transform to the image at the middle of the stack
    aligned_image = scipy.ndimage.affine_transform(
        fluor_channel_position[0][
            0, fluor_channel_idx, fluor_channel_Z // 2 : fluor_channel_Z // 2 + 1
        ],
        zyx_affine_transform,
        output_shape=(1, phase_channel_Y, phase_channel_X),
    )
    viewer.add_image(
        phase_channel_position[0][
            0, phase_channel_idx, phase_channel_Z // 2 : phase_channel_Z // 2 + 1
        ],
        name=f"middle_plane_{phase_channel_str}",
    )
    viewer.add_image(aligned_image, name=f"registered_{fluor_channel_str}", opacity=0.5)
    viewer.layers[f"registered_{fluor_channel_str}"].colormap = "magenta"
    viewer.layers.remove(f"pts_{phase_channel_str}")
    viewer.layers.remove(f"pts_{fluor_channel_str}")
    viewer.layers[fluor_channel_str].visible = False

    # Compute the 3D registration
    # Use deskew metadta
    try:
        _, _, z_sampling_fluor_channel, _, _ = fluor_channel_position.zattrs["multiscales"][0][
            "coordinateTransformations"
        ][0]["scale"]
    except LookupError("Could not find coordinateTransformation scale in metadata"):
        z_sampling_fluor_channel = 1

    print(z_sampling_fluor_channel)
    # Estimate the Similarity Transform (rotation,scaling,translation)
    transform = SimilarityTransform()
    transform.estimate(pts_phase_channel, pts_fluor_channel)
    zyx_points_transformation_matrix = transform.params

    z_scaling_matrix = np.array(
        [[scaling_factor_z, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    # Composite of all transforms
    zyx_affine_transform = (
        rotation_matrix @ zyx_points_transformation_matrix @ z_scaling_matrix
    )
    print(f"Affine Transform Matrix:\n {zyx_affine_transform}\n")
    settings = RegistrationSettings(
        affine_transform_zyx=zyx_affine_transform.tolist(),
        voxel_size=list(voxel_size),
        output_shape=list(output_shape_volume),
        pts_phase_registration=pts_phase_channel.tolist(),
        pts_fluor_registration=pts_fluor_channel.tolist(),
        phase_channel_path=str(labelfree_position_dirpaths),
        fluor_channel_path=str(lightsheet_position_dirpaths),
        fluor_channel_90deg_CCW_rotation=ROTATE_90deg_CCW,
    )
    print(f"Writing deskewing parameters to {output_filepath}")
    with open(output_filepath, "w") as f:
        yaml.dump(asdict(settings), f)

    # Apply the transformation to 3D volume
    flag_apply_3D_transform = input("\n Apply 3D registration (Y/N):")
    if flag_apply_3D_transform == "Y" or flag_apply_3D_transform == "y":
        print("Applying 3D Affine Transform...")
        registered_3D_volume = scipy.ndimage.affine_transform(
            phase_channel_position[0][0, phase_channel_idx],
            np.linalg.inv(zyx_affine_transform),
            output_shape=output_shape_volume,
        )
        viewer.add_image(
            registered_3D_volume, name=f"registered_volume_{phase_channel_str}", opacity=1.0
        )

        viewer.add_image(
            fluor_channel_position[0][0, fluor_channel_idx],
            name=f"{fluor_channel_str}",
            opacity=0.5,
            colormap="magenta",
        )
        viewer.layers[f"registered_{fluor_channel_str}"].visible = False
        viewer.layers[f"{phase_channel_str}"].visible = False
        viewer.layers[f"middle_plane_{phase_channel_str}"].visible = False

    input("\n Displaying registered channels. Press <enter> to close...")
