from click.testing import CliRunner

from mantis.cli.main import cli


def test_deskew_cli(tmp_path, example_fov, example_deskew_settings):
    fov_path, _ = example_fov
    config_path, _ = example_deskew_settings
    output_path = tmp_path / "output.zarr"

    # Test deskew cli
    runner = CliRunner()
    result = runner.invoke(
        cli, ["deskew", "-i", str(fov_path), "-c", str(config_path), "-o", str(output_path)]
    )

    assert output_path.exists()
    assert result.exit_code == 0
