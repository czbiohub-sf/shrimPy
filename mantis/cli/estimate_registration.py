# %%
from dataclasses import asdict
import click
import napari
import numpy as np
import yaml

from iohub import read_micromanager
from iohub import open_ome_zarr
import os
from waveorder.focus import focus_from_transverse_band
from skimage import transform


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
    Estimate the affine transform between two channels by manual inputs.

    python estimate_registration.py  <path/to/channel_1.zarr/0/0/0> <path/to/channel_2.zarr/0/0/0>
    """
    # Get a napari viewer()
    viewer = napari.Viewer()

    print("Getting dataset info")
    os.system(f"iohub info {channel_1_data_path} ")
    os.system(f"iohub info {channel_2_data_path} ")

    channel_1_idx = int(input("Enter channel_1 index to process: "))
    channel_2_idx = int(input("Enter channel_2 index to process: "))

    # TODO: get the microscope parameters through a yaml file? Currently hard-coded

    # Find focus channel_1
    print('Finding the channel_1 best focus position at t=0')
    with open_ome_zarr(channel_1_data_path, mode="r") as channel_1_position:
        channel_1_str = channel_1_position.channel_names[channel_1_idx]
        focus_channel_1_idx = focus_from_transverse_band(
            channel_1_position[0][0, channel_1_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=6.5 / (40 * 1.4),
        )
        print(f"{channel_1_str} focus idx: {focus_channel_1_idx}")

    # Find focus channel_2
    print('Finding the channel_2 best focus position at t=0')
    with open_ome_zarr(channel_2_data_path, mode="r") as channel_2_position:
        channel_2_str = channel_2_position.channel_names[channel_2_idx]
        focus_channel_2_idx = focus_from_transverse_band(
            channel_2_position[0][0, channel_2_idx],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=3.45 / (46.2 * 1.4),
        )
        print(f"channel_2 focus idx: {focus_channel_2_idx}")

    # Display only the in-focus slices
    channel_1_img = channel_1_position[0][0, 1, focus_channel_1_idx]
    #TODO: this rotation might not be necessary for all datasets
    channel_1_img = transform.rotate(channel_1_img, 90, resize=True, preserve_range=True)
    channel_2_img = channel_2_position[0][0, 0, focus_channel_2_idx]
    Y, X = channel_2_img.shape
    viewer.add_image(channel_1_img, name=channel_1_str)
    viewer.layers[-1].translate = (0, X)
    viewer.add_image(channel_2_img, name=channel_2_str)

    # Manual annotation of features
    COLOR_CYCLE = ["white", "cyan", "lime", "orchid", "blue", "orange", "yellow", "magenta"]

    def next_on_click(layer, event):
        if layer.mode == "add":
            if layer is points_channel_1:
                next_layer = points_channel_2
                # Add a point to the active layer
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index + 1) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                next_layer.current_face_color = next_color
            else:
                next_layer = points_channel_1
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

    # Create the first points layer
    points_channel_1 = viewer.add_points(
        ndim=2, name=f"pts_{channel_1_str}", size=50, face_color=COLOR_CYCLE[0]
    )
    points_channel_2 = viewer.add_points(
        ndim=2, name=f"pts_{channel_2_str}", size=50, face_color=COLOR_CYCLE[0]
    )

    # Create the second points layer
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

    pts_channel_1[:, 1] -= X  # subtract the translation offset for display
    pts_channel_1 = pts_channel_1[:, ::-1]  # Flip the axis (x,y)
    pts_channel_2 = pts_channel_2[:, ::-1]  # Flip the axis (x,y)

    # Find the affine transform
    affine_transform = transform.AffineTransform()
    affine_transform.estimate(pts_channel_1, pts_channel_2)

    # Get the transformation matrix
    matrix = affine_transform.params
    print(f"Affine Transform Matrix:\n {matrix}\n")

    # Apply the affine transform to the image
    Y, X = channel_2_img.shape
    aligned_image = transform.warp(channel_1_img, inverse_map=affine_transform.inverse, clip=False)
    viewer.add_image(aligned_image, name="aligned", opacity=0.5)
    viewer.layers.remove(f"pts_{channel_1_str}")
    viewer.layers.remove(f"pts_{channel_2_str}")
    viewer.layers[channel_1_str].visible = False
    viewer.layers["aligned"].colormap = "magenta"

    # %%
    # Write and Save the matrix
    with open_ome_zarr(
        output_file, layout="fov", mode="w", channel_names=["None"]
    ) as output_dataset:
        output_dataset["affine_transform"] = matrix[None, None, None, ...]
        output_dataset["pts_channel_1"] = pts_channel_1[None, None, None, ...]
        output_dataset["pts_channel_2"] = pts_channel_2[None, None, None, ...]
        
        # Write extra registration metadata
        registration_params = {
            "channel_1": channel_1_data_path,
            "channel_1_name": channel_1_str,
            "channel_1_focused_slice": focus_channel_1_idx,
            "channel_2": channel_2_data_path,
            "channel_2_name": channel_2_str,
            "channel_2_focused_slice": focus_channel_2_idx
        }
        output_dataset.zattrs["registration"] = registration_params

    print(f"Finished saving registration parameters in: {output_file}")

    input("\n Displaying registered channels. Press <enter> to close...")


if __name__ == '__main__':
    manual_registration()
