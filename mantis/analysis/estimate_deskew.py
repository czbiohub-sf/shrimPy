import click
from iohub import read_micromanager
import napari
import numpy as np
import yaml
import deskew_settings


@click.command()
@click.argument(
    "mmfolder",
    type=click.Path(exists=True),
)
@click.option(
    "--output-file",
    "-o",
    default="./deskew_params.yml",
    required=False,
    help="Path to saved parameters",
)
def estimate_deskew(mmfolder, output_file):
    """
    Routine for estimating deskewing parameters from calibration data.

    >> python estimate_deskew.py </path/to/mmfolder>

    """
    # Read p, t, c = (0, 0, 0) into an array
    reader = read_micromanager(mmfolder)
    data = reader.get_array(0)[0, 0, ...]  # zyx

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
    print(f"px_to_scan_ratio : {px_to_scan_ratio:.3f}\n")

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
    print(f"theta_deg : {theta_deg:.2f}\n")

    # Create validated object
    settings = deskew_settings.DeskewSettings(
        px_to_scan_ratio=px_to_scan_ratio, theta_deg=theta_deg
    )

    # Write result
    print(f"Writing deskewing parameters to {output_file}")
    with open(output_file, "w") as f:
        yaml.dump(settings.dict(), f)


if __name__ == "__main__":
    estimate_deskew()
