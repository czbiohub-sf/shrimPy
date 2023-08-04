from click.testing import CliRunner

from mantis.cli.main import cli


def test_deskew_cli(tmp_path, example_plate, example_deskew_settings):
    plate_path, _ = example_plate
    config_path, _ = example_deskew_settings
    output_path = tmp_path / "output.zarr"

    # Test deskew cli
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "deskew",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0
