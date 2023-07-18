# %%
import os

from dataclasses import asdict

import click
import napari
import numpy as np
import scipy
import yaml

from iohub import open_ome_zarr, read_micromanager
from waveorder.focus import focus_from_transverse_band

from mantis.analysis.registration import estimate_transformation_matrix

# TODO: remove this for non mantis datasets
ROTATE_90deg_CCW = True


@click.command()
@click.argument("channel_1_data_path", type=click.Path(exists=True))
@click.argument("channel_2_data_path", type=click.Path(exists=True))
@click.option(
    "--output-file",
    "-o",
    default="./registration_output.zarr",
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
    print('\n Channel 1 INFO:')
    os.system(f"iohub info {channel_1_data_path} ")
    print('\n Channel 2 INFO:')
    os.system(f"iohub info {channel_2_data_path} ")

    channel_1_idx = int(input("Enter channel_1 index to process: "))
    channel_2_idx = int(input("Enter channel_2 index to process: "))
    #%%
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
        channel_1_Z, channel_1_Y, channel_1_X = channel_1_position[0][0, channel_1_idx].shape
        print(f"{channel_1_str} focus idx: {focus_channel_1_idx}")

    # Find focus channel_2
    print("Finding the channel_2 best focus position at t=0")
    with open_ome_zarr(channel_2_data_path, mode="r") as channel_2_position:
        channel_2_str = channel_2_position.channel_names[channel_2_idx]
        focus_channel_2_idx = focus_from_transverse_band(
            channel_2_position[0][0, channel_2_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=6.5/ (40 * 1.4),
        )
        channel_2_Z, channel_2_Y, channel_2_X = channel_2_position[0][0, channel_2_idx].shape
        print(f"channel_2 focus idx: {focus_channel_2_idx}")

    # Display only the in-focus slices
    channel_1_img = channel_1_position[0][
        0, channel_1_idx, focus_channel_1_idx : focus_channel_1_idx + 1
    ]
    channel_2_img = channel_2_position[0][
        0, channel_2_idx, focus_channel_2_idx : focus_channel_2_idx + 1
    ]
    layer_channel_1 = viewer.add_image(channel_1_img, name=channel_1_str)

    # TODO: Need to find a better way to handle rotations or just assume data is passed in the right orientation
    if ROTATE_90deg_CCW:
        channel_2_img = scipy.ndimage.rotate(
            channel_2_img, angle=90, axes=(1, 2), reshape=True
        )
        _, channel_2_Y, channel_2_X = channel_2_img.shape

    layer_channel_2 = viewer.add_image(channel_2_img, name=channel_2_str)
    layer_channel_2.translate = (0, 0, channel_1_X)

    #%%
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
                # Add a point to the active layer
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color
            else:
                next_layer = points_channel_1
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

    pts_channel_1 = np.array([[   0.        ,  299.27558608,  575.06444763],
       [   0.        ,  712.51585715, 1161.51920358],
       [   0.        ,  752.71660219,  380.36647624],
       [   0.        ,  460.72647054,  311.08068229]])

    pts_channel_2 = np.array([[   0.        ,  321.54601985, 2200.80611286],
       [   0.        ,  852.31940309, 2980.5870999 ],
       [   0.        ,  920.98210179, 1973.93973714],
       [   0.        ,  544.85922033, 1867.53655357]])

    pts_channel_2[:, 2] -= channel_1_X  # subtract the translation offset for display
    
    # Estimate the affine transform between the points in-focus
    yx_points_transformation_matrix = estimate_transformation_matrix(
        pts_channel_1[:, 1:], pts_channel_2[:, 1:]
    )
    # Add rotation matrix
    # TODO: make this optional: return the identity matrix if no rotation is needed.
    if ROTATE_90deg_CCW:
        rotation_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, channel_2_Y - 1], [0, 0, 0, 1]]
        )
        # rotated_image = scipy.ndimage.affine_transform(
        #     channel_2_position[0][0, channel_2_idx, focus_channel_2_idx : focus_channel_2_idx + 1],
        #     rotation_matrix,
        #     output_shape=(1, channel_2_Y, channel_2_X),
        # )
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

    # Apply the affine transform to the image
    aligned_image = scipy.ndimage.affine_transform(
        channel_2_position[0][0, channel_2_idx, focus_channel_2_idx : focus_channel_2_idx + 1],
        zyx_affine_transform,
        output_shape=output_shape_volume,
    )

    viewer.add_image(aligned_image, name=f"registered_{channel_2_str}", opacity=0.5)
    viewer.layers["aligned"].colormap = "magenta"
    viewer.layers.remove(f"pts_{channel_1_str}")
    viewer.layers.remove(f"pts_{channel_2_str}")
    viewer.layers[channel_2_str].visible = False

    # Compute the 3D registration
    flag_estimate_3D_transform = input("\n Estimate and apply 3D registration (Y/N):")
    if flag_estimate_3D_transform == "Y" or flag_estimate_3D_transform == "y":
        print('Applying 3D Affine Transform...')
        # Use deskew metadta
        try:
            _, _, z_sampling_channel_2, _, _ = channel_2_position.zattrs["multiscales"][0][
                "coordinateTransformations"
            ][0]["scale"]
        except:
            z_sampling_channel_2 = 1

        print(z_sampling_channel_2)
        # Scaling and translation factor based on metadata.
        magnification_LF_remote_volume = 1.4
        step_size_LF_remote_volume = 0.4  # [um]
        phase_remote_volume_sample_size = (
            step_size_LF_remote_volume / magnification_LF_remote_volume
        )
        scaling_factor_z =  phase_remote_volume_sample_size/z_sampling_channel_2
        translation_z = (focus_channel_1_idx * scaling_factor_z) - focus_channel_2_idx
        # 2D to 3D matrix
        z_shift = np.array([scaling_factor_z, 0, 0, -translation_z])
        print(f'Z-transform: {z_shift}')
        zyx_points_transformation_matrix = np.vstack(
            (z_shift, np.insert(yx_points_transformation_matrix, 0, 0, axis=1))
        )
        zyx_affine_transform = rotation_matrix @ zyx_points_transformation_matrix
        print(f"Affine Transform Matrix:\n {zyx_affine_transform}\n")

        output_shape_volume = (channel_2_Z, channel_2_X, channel_2_Y)
        registered_3D_volume = scipy.ndimage.affine_transform(
            channel_1_position[0][0, channel_1_idx],
            np.linalg.inv(zyx_affine_transform),
            output_shape=output_shape_volume,
        )
        viewer.add_image(registered_3D_volume,name=f'registered_volume_{channel_1_str}', opacity=1.0)

        viewer.add_image(channel_2_position[0][0, channel_2_idx],name=f'channel_2_str', opacity=0.5,colormap='magenta')
        viewer.layers["aligned"].visible = False



    # Write and Save the matrix
    with open_ome_zarr(
        output_file, layout="fov", mode='w', channel_names=["None"]
    ) as output_dataset:
        output_dataset["affine_transform_zyx"] = zyx_affine_transform[None, None, None, ...]
        output_dataset["pts_channel_1"] = pts_channel_1[None, None, None, ...]
        output_dataset["pts_channel_2"] = pts_channel_2[None, None, None, ...]

        # Write extra registration metadata
        registration_params = {
            "channel_1": channel_1_data_path,
            "channel_1_name": channel_1_str,
            "channel_1_focused_slice": focus_channel_1_idx,
            "channel_2": channel_2_data_path,
            "channel_2_name": channel_2_str,
            "channel_2_focused_slice": focus_channel_2_idx,
            "channel_2_90deg_CCW_rotation": ROTATE_90deg_CCW,
        }
        output_dataset.zattrs["registration"] = registration_params

    print(f"Finished saving registration output in: {output_file}")

    input("\n Displaying registered channels. Press <enter> to close...")

if __name__ == "__main__":
    manual_registration()

# %%
