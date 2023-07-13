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
@click.argument(
    "fluor_data_path", type=click.Path(exists=True), help="Path to fluorescence dataset"
)
@click.argument("phase_data_path", type=click.Path(exists=True), help="Path to phase dataset")
@click.option(
    "--output-file",
    "-o",
    default="./registration_output.yml",
    required=False,
    help="Path to saved registration",
)
def manual_registration(fluor_data_path, phase_data_path, output_file):
    # Get a napari viewer()
    viewer = napari.Viewer()

    # TODO: get the microscope parameters through a yaml file? Currently hard-coded

    # Find focus fluor channel
    print('Finding the fluorescence best focus position at t=0')
    with open_ome_zarr(fluor_data_path, phase_data_path, mode="r") as fluor_position:
        focus_fluor_idx = focus_from_transverse_band(
            fluor_position[0][0, 0],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=6.5 / (40 * 1.4),
        )
        print(f"Fluor focus idx: {focus_fluor_idx}")

    # Find focus Phase
    print('Finding the phase best focus position at t=0')
    with open_ome_zarr(phase_data_path, mode="r") as phase_position:
        focus_phase_idx = focus_from_transverse_band(
            phase_position[0][0, 0],
            NA_det=1.35,
            lambda_ill=0.55,
            pixel_size=3.45 / (46.2 * 1.4),
        )
        print(f"Phase focus idx: {focus_phase_idx}")

    # Display only the in-focus slices
    fluor_img = fluor_position[0][0, 1, focus_fluor_idx]
    fluor_img = transform.rotate(fluor_img, 90, resize=True, preserve_range=True)
    phase_img = phase_position[0][0, 0, focus_phase_idx]
    Y, X = phase_img.shape
    viewer.add_image(fluor_img, name="focus_fluor")
    viewer.layers[-1].translate = (0, X)
    viewer.add_image(phase_img, name="focus_phase")

    # Manual annotation of features
    COLOR_CYCLE = ["white", "cyan", "lime", "orchid", "blue", "orange", "yellow", "magenta"]

    def next_on_click(layer, event):
        if layer.mode == "add":
            if layer is points_fluor:
                next_layer = points_phase
                # Add a point to the active layer
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)

                print(f"curr lay1 color {layer.current_face_color}")

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index + 1) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                print(f"next color {next_color}")
                next_layer.current_face_color = next_color
            else:
                print(f"curr lay2 color {layer.current_face_color}")
                next_layer = points_fluor
                cursor_position = np.array(viewer.cursor.position)
                layer.add(cursor_position)

                # Change the colors
                current_index = COLOR_CYCLE.index(layer.current_face_color)
                next_index = (current_index) % len(COLOR_CYCLE)
                next_color = COLOR_CYCLE[next_index]
                print(f"next color {next_color}")
                next_layer.current_face_color = next_color

            # Switch to the next layer
            next_layer.mode = "add"
            layer.selected_data = {}
            viewer.layers.selection.active = next_layer

    # Create the first points layer
    points_fluor = viewer.add_points(
        ndim=2, name="pts_fluor", size=50, face_color=COLOR_CYCLE[0]
    )
    points_phase = viewer.add_points(
        ndim=2, name="pts_phase", size=50, face_color=COLOR_CYCLE[0]
    )

    # Create the second points layer
    points_fluor.mode = "add"
    points_phase.mode = "add"
    # Bind the mouse click callback to both point layers
    points_fluor.mouse_drag_callbacks.append(next_on_click)
    points_phase.mouse_drag_callbacks.append(next_on_click)

    input(
        "\n Add at least three points in the two channels by sequentially clicking a feature on channel 1 and its corresponding feature in channel 2.  Press <enter> when done..."
    )

    # Get the data from the layers
    pts_fluor = points_fluor.data
    pts_phase = points_phase.data

    pts_fluor[:, 1] -= X  # subtract the translation offset for display
    pts_phase = pts_phase[:, ::-1]  # Flip the axis (x,y)
    pts_fluor = pts_fluor[:, ::-1]  # Flip the axis (x,y)

    # Find the affine transform
    affine_transform = transform.AffineTransform()
    affine_transform.estimate(pts_fluor, pts_phase)

    # Get the transformation matrix
    matrix = affine_transform.params
    print(f"Affine Transform Matrix: {matrix}")

    # Apply the affine transform to the image
    Y, X = phase_img.shape
    aligned_image = transform.warp(fluor_img, inverse_map=affine_transform.inverse, clip=False)
    viewer.add_image(aligned_image, name="aligned", opacity=0.5)

    # %%
    # Write and Save the matrix
    with open_ome_zarr(
        output_file, layout="fov", mode="w", channel_names=["None"]
    ) as output_dataset:
        output_dataset["affine_transform"] = matrix[None, None, None, ...]
        output_dataset["pts_phase"] = pts_phase[None, None, None, ...]
        output_dataset["pts_fluor"] = pts_fluor[None, None, None, ...]
        output_dataset["focus_fluor_idx"] = focus_fluor_idx[None, None, None, None, None, ...]
        output_dataset["focus_phase_idx"] = focus_phase_idx[None, None, None, None, None, ...]


if __name__ == '__main__':
    manual_registration()
