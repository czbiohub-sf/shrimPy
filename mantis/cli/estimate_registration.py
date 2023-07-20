import os


import click
import napari
import numpy as np
import scipy

from iohub import open_ome_zarr
from pathlib import Path
from skimage.transform import SimilarityTransform
from waveorder.focus import focus_from_transverse_band

# TODO: remove this for non mantis datasets
ROTATE_90deg_CCW = True
# TODO: add option to use the autofocused slice


def find_focus_channel_pairs(
    channel_1_data_path: Path,
    channel_1_idx: int,
    channel_2_data_path: Path,
    channel_2_idx: int,
):
    """
    Find the focus sliced for input channels

    Parameters
    ----------
    channel_1_data_path : Path
    channel_1_idx : int
    channel_2_data_path : Path
    channel_2_idx : int
    """
    # TODO: get the microscope parameters through a yaml file? Currently hard-coded
    # Find focus channel_1
    print("Finding the channel_1 best focus position at t=0")
    with open_ome_zarr(channel_1_data_path, mode="r") as channel_1_position:
        channel_1_str = channel_1_position.channel_names[channel_1_idx]
        focus_channel_1_idx = focus_from_transverse_band(
            channel_1_position[0][0, channel_1_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=3.45 / (46.2 * 1.4),
        )
        channel_1_Z, channel_1_Y, channel_1_X = channel_1_position[0][
            0, channel_1_idx
        ].shape
        print(f"{channel_1_str} focus idx: {focus_channel_1_idx}")

    # Find focus channel_2
    print("Finding the channel_2 best focus position at t=0")
    with open_ome_zarr(channel_2_data_path, mode="r") as channel_2_position:
        channel_2_position.channel_names[channel_2_idx]
        focus_channel_2_idx = focus_from_transverse_band(
            channel_2_position[0][0, channel_2_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=6.5 / (40 * 1.4),
        )
        channel_2_Z, channel_2_Y, channel_2_X = channel_2_position[0][
            0, channel_2_idx
        ].shape
        print(f"channel_2 focus idx: {focus_channel_2_idx}")


@click.command()
@click.argument("channel_1_data_path", type=click.Path(exists=True))
@click.argument("channel_2_data_path", type=click.Path(exists=True))
@click.option(
    "--output-file",
    "-o",
    default="./registration_parameters.zarr",
    required=False,
    help="Path to saved registration",
)
def manual_registration(channel_1_data_path, channel_2_data_path, output_file):
    """
    Estimate the affine transform between two channels (source channel and target channel) by manual inputs.

    python estimate_registration.py  <path/to/channel_1.zarr/0/0/0> <path/to/channel_2.zarr/0/0/0>
    """
    # Get a napari viewer()
    viewer = napari.Viewer()

    print("Getting dataset info")
    print("\n Channel 1 INFO:")
    os.system(f"iohub info {channel_1_data_path} ")
    print("\n Channel 2 INFO:")
    os.system(f"iohub info {channel_2_data_path} ")

    channel_1_idx = int(input("Enter channel_1 index to process: "))
    channel_2_idx = int(input("Enter channel_2 index to process: "))

    # Display volumes rescaled
    with open_ome_zarr(channel_1_data_path, mode="r") as channel_1_position:
        channel_1_str = channel_1_position.channel_names[channel_1_idx]
        channel_1_volume = channel_1_position[0][0, channel_1_idx]
        channel_1_Z, channel_1_Y, channel_1_X = channel_1_volume.shape
    with open_ome_zarr(channel_2_data_path, mode="r") as channel_2_position:
        channel_2_str = channel_2_position.channel_names[channel_2_idx]
        channel_2_volume = channel_2_position[0][0, channel_2_idx]
        channel_2_Z, channel_2_Y, channel_2_X = channel_2_volume.shape

    ## Find the z-scaling and apply it for display
    # TODO: Get these values from the yaml file
    # Use deskew metadta
    try:
        (
            _,
            _,
            z_sampling_channel_2,
            y_sampling_channel_2,
            x_sampling_channel_2,
        ) = channel_2_position.zattrs["multiscales"][0]["coordinateTransformations"][0][
            "scale"
        ]
        voxel_size = (z_sampling_channel_2, y_sampling_channel_2, x_sampling_channel_2)
    except LookupError("Couldn't find z_sampling in metadata"):
        z_sampling_channel_2 = 1
        voxel_size = (1, 1, 1)
    magnification_LF_remote_volume = 1.4
    step_size_LF_remote_volume = 0.4  # [um]
    phase_remote_volume_sample_size = (
        step_size_LF_remote_volume / magnification_LF_remote_volume
    )
    scaling_factor_z = phase_remote_volume_sample_size / z_sampling_channel_2

    # Add layers to napari with and transform
    # TODO: these rotations and scaling are not required for all datasets
    layer_channel_1 = viewer.add_image(channel_1_volume, name=channel_1_str)
    layer_channel_2 = viewer.add_image(channel_2_volume, name=channel_2_str)
    layer_channel_1.scale = (scaling_factor_z, 1, 1)
    layer_channel_2.rotate = 90
    layer_channel_2.translate = (0, channel_2_X, channel_1_X)

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
            if layer is points_channel_1:
                next_layer = points_channel_2
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_channel_2 = viewer.dims.current_step
                else:
                    prev_step_channel_2 = (next_layer.data[-1][0] + 2, 0, 0)
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
                viewer.dims.current_step = prev_step_channel_2

            else:
                next_layer = points_channel_1
                # Change slider value
                if len(next_layer.data) < 1:
                    prev_step_channel_1 = viewer.dims.current_step
                else:
                    # TODO: this +2 is not clear to me?
                    prev_step_channel_1 = (next_layer.data[-1][0] + 2, 0, 0)
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
                viewer.dims.current_step = prev_step_channel_1

    # Create the first points layer
    points_channel_1 = viewer.add_points(
        ndim=3, name=f"pts_{channel_1_str}", size=50, face_color=COLOR_CYCLE[0]
    )
    points_channel_2 = viewer.add_points(
        ndim=3, name=f"pts_{channel_2_str}", size=50, face_color=COLOR_CYCLE[0]
    )

    # Create the second points layer
    viewer.layers.selection.active = points_channel_1
    points_channel_1.mode = "add"
    points_channel_2.mode = "add"
    # Bind the mouse click callback to both point layers
    points_channel_1.mouse_drag_callbacks.append(next_on_click)
    points_channel_2.mouse_drag_callbacks.append(next_on_click)

    input(
        "\n Add at least three points in the two channels by sequentially clicking a feature on channel 1 and its corresponding feature in channel 2.  Press <enter> when done..."
    )

    # Get the data from the layers
    pts_channel_1 = points_channel_1.data
    pts_channel_2 = points_channel_2.data

    pts_channel_2[:, 2] -= channel_1_X  # subtract the translation offset for display

    # Estimate the affine transform between the points xy to make sure registration is good
    transform = SimilarityTransform()
    transform.estimate(pts_channel_1[:, 1:], pts_channel_2[:, 1:])
    yx_points_transformation_matrix = transform.params

    # Add rotation matrix
    # TODO: make this optional: return the identity matrix if no rotation is needed.
    if ROTATE_90deg_CCW:
        rotation_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, channel_2_X - 1], [0, 0, 0, 1]]
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
    output_shape_volume = (1, channel_1_Y, channel_1_X)

    # Demo: apply the affine transform to the image at the middle of the stack
    aligned_image = scipy.ndimage.affine_transform(
        channel_2_position[0][
            0, channel_2_idx, channel_2_Z // 2 : channel_2_Z // 2 + 1
        ],
        zyx_affine_transform,
        output_shape=output_shape_volume,
    )
    viewer.add_image(
        channel_1_position[0][
            0, channel_1_idx, channel_1_Z // 2 : channel_1_Z // 2 + 1
        ],
        name=f"middle_plane_{channel_1_str}",
    )
    viewer.add_image(aligned_image, name=f"registered_{channel_2_str}", opacity=0.5)
    viewer.layers[f"registered_{channel_2_str}"].colormap = "magenta"
    viewer.layers.remove(f"pts_{channel_1_str}")
    viewer.layers.remove(f"pts_{channel_2_str}")
    viewer.layers[channel_2_str].visible = False

    # Compute the 3D registration
    flag_estimate_3D_transform = input("\n Estimate and apply 3D registration (Y/N):")
    if flag_estimate_3D_transform == "Y" or flag_estimate_3D_transform == "y":
        print("Applying 3D Affine Transform...")
        # Use deskew metadta
        try:
            _, _, z_sampling_channel_2, _, _ = channel_2_position.zattrs["multiscales"][
                0
            ]["coordinateTransformations"][0]["scale"]
        except LookupError("Could not find coordinateTransformation scale in metadata"):
            z_sampling_channel_2 = 1

        print(z_sampling_channel_2)
        # Estimate the Similarity Transform (rotation,scaling,translation)
        transform = SimilarityTransform()
        transform.estimate(pts_channel_1, pts_channel_2)
        zyx_points_transformation_matrix = transform.params

        z_scaling_matrix = np.array(
            [[scaling_factor_z, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Composite of all transforms
        zyx_affine_transform = (
            rotation_matrix @ zyx_points_transformation_matrix @ z_scaling_matrix
        )
        print(f"Affine Transform Matrix:\n {zyx_affine_transform}\n")

        output_shape_volume = (channel_2_Z, channel_2_Y, channel_2_X)
        registered_3D_volume = scipy.ndimage.affine_transform(
            channel_1_position[0][0, channel_1_idx],
            np.linalg.inv(zyx_affine_transform),
            output_shape=output_shape_volume,
        )
        viewer.add_image(
            registered_3D_volume, name=f"registered_volume_{channel_1_str}", opacity=1.0
        )

        viewer.add_image(
            channel_2_position[0][0, channel_2_idx],
            name=f"{channel_2_str}",
            opacity=0.5,
            colormap="magenta",
        )
        viewer.layers[f"registered_{channel_2_str}"].visible = False
        viewer.layers[f"{channel_1_str}"].visible = False
        viewer.layers[f"middle_plane_{channel_1_str}"].visible = False

    # Write and Save the matrix
    with open_ome_zarr(
        output_file, layout="fov", mode="w", channel_names=["None"]
    ) as output_dataset:
        output_dataset["affine_transform_zyx"] = zyx_affine_transform[
            None, None, None, ...
        ]
        output_dataset["pts_channel_1"] = pts_channel_1[None, None, None, ...]
        output_dataset["pts_channel_2"] = pts_channel_2[None, None, None, ...]

        # Write extra registration metadata
        registration_params = {
            "channel_1": channel_1_data_path,
            "channel_1_name": channel_1_str,
            "channel_2": channel_2_data_path,
            "channel_2_name": channel_2_str,
            "channel_2_90deg_CCW_rotation": ROTATE_90deg_CCW,
            "channel_1_shape": list((channel_1_Z, channel_1_Y, channel_1_X)),
            "channel_2_shape": list((channel_2_Z, channel_2_Y, channel_2_X)),
            "voxel_size": list(voxel_size),
        }
        output_dataset.zattrs["registration"] = registration_params

    print(f"Finished saving registration output in: {output_file}")

    input("\n Displaying registered channels. Press <enter> to close...")


if __name__ == "__main__":
    manual_registration()
