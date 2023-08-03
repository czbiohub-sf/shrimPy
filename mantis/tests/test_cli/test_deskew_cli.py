import pathlib

import numpy as np

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from mantis.cli.main import cli


def test_deskew_cli(tmp_path):
    input_path = tmp_path / "input.zarr"
    output_path = input_path.with_name("output.zarr")

    # Generate input dataset
    dataset = open_ome_zarr(
        input_path,
        layout="fov",
        mode="w",
        channel_names=["GFP", "RFP"],
    )

    dataset.create_zeros("0", (3, 2, 4, 5, 6), dtype=np.uint16)
    assert input_path.exists()

    # Check config
    config_path = pathlib.Path("./mantis/analysis/settings/example_deskew_settings.yml")
    assert config_path.exists()

    # Test deskew cli
    runner = CliRunner()
    runner.invoke(
        cli, ["deskew", "-i", str(input_path), "-c", str(config_path), "-o", str(output_path)]
    )

    assert output_path.exists()
