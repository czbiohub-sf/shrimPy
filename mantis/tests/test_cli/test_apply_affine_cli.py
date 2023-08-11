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
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    assert output_path.exists()
    assert result.exit_code == 0
