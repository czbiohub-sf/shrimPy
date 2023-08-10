from click.testing import CliRunner

from mantis.cli.main import cli


def test_apply_affine_cli(tmp_path, example_plate, example_apply_affine_settings):
    plate_path, _ = example_plate
    config_path, _ = example_apply_affine_settings
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "apply-affine",
            "-lf",
            str(plate_path) + "/A/1/0",
            "-ls",
            str(plate_path) + "/B/1/0",  # test could be improved with different stores
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0
