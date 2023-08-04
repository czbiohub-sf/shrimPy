from dataclasses import asdict

import click
import napari
import numpy as np
import yaml

from iohub.ngff import open_ome_zarr

from mantis.analysis.AnalysisSettings import DeskewSettings
from mantis.cli.parsing import input_position_dirpaths, output_filepath


@click.command()
@input_position_dirpaths()
@output_filepath()
def estimate_deskew(input_position_dirpaths, output_filepath):
    """
    Routine for estimating deskewing parameters from calibration data.

    >> mantis estimate-deskew -i ./input.zarr/0/0/0 -o ./deskew_params.yml
    """
    assert str(output_filepath).endswith(('.yaml', '.yml')), "Output file must be a YAML file."

    # Read p, t, c = (0, 0, 0) into an array
    with open_ome_zarr(input_position_dirpaths[0]) as reader:
        data = reader["0"][0, 0]  # zyx

    pixel_size_um = float(input("Enter image pixel size in micrometers: "))
    scan_step_um = float(input("Enter the estimated galvo scan step in micrometers: "))
    approx_theta_deg = float(input("Enter the approximate light sheet angle in degrees: "))
    approx_px_to_scan_ratio = pixel_size_um / scan_step_um

    v = napari.Viewer()
    v.add_image(data)

    # Estimate px_to_scan_ratio
    v.add_shapes(ndim=3, name="rect")
    v.layers[-1].mode = "add_rectangle"
    v.dims.order = (1, 2, 0)
    input(
        "\nDraw a rectangle around an object that you expect to be square after deskewing, then press <enter>..."
    )
    rect = v.layers["rect"].data[0]
    px_to_scan_ratio = (rect[2, 0] - rect[0, 0]) / (rect[2, 2] - rect[0, 2])
    print(f"Measured px_to_scan_ratio : {px_to_scan_ratio:.3f}\n")

    factor = np.abs(1 - approx_px_to_scan_ratio / px_to_scan_ratio) * 100
    print(f"The measured px_to_scan_ratio is within {round(factor)}% from your estimate")

    # Estimate theta
    v.layers.remove("data")
    v.layers.remove("rect")
    x_proj = np.sum(data, axis=2, dtype=np.float32)
    v.add_image(x_proj, name="x_proj")
    v.add_shapes(ndim=2, name="coverslip-normal")
    v.layers[-1].mode = "add_line"
    input(
        "Draw a line parallel to an object perpendicular to the coverslip, then press <enter>..."
    )
    line = v.layers[-1].data[0]
    r = line[1] - line[0]
    r_hat = r / np.linalg.norm(r)
    theta = np.arccos(r_hat[0] / r_hat[1] / px_to_scan_ratio)
    theta_deg = (theta % np.pi) * 180 / np.pi
    print(f"Measured light-sheet angle : {theta_deg:.2f}\n")

    factor = np.abs(1 - approx_theta_deg / theta_deg) * 100
    print(f"The measured light-sheet angle is within {round(factor)}% from your estimate")

    # Create validated object
    settings = DeskewSettings(
        pixel_size_um=pixel_size_um,
        ls_angle_deg=theta_deg,
        px_to_scan_ratio=px_to_scan_ratio,
        scan_step_um=scan_step_um,
    )

    # Write result
    print(f"Writing deskewing parameters to {output_filepath}")
    with open(output_filepath, "w") as f:
        yaml.dump(asdict(settings), f)


if __name__ == "__main__":
    estimate_deskew()
