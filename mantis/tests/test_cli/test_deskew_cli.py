from click.testing import CliRunner

from mantis.cli.main import cli


def test_deskew_cli(tmp_path, example_fov, example_deskew_settings):
    fov_path, _ = example_fov
    config_path, _ = example_deskew_settings

    # Test deskew cli
    output_path = tmp_path / "output.zarr"
    runner = CliRunner()
    runner.invoke(
        cli, ["deskew", "-i", str(fov_path), "-c", str(config_path), "-o", str(output_path)]
    )

    assert output_path.exists()
