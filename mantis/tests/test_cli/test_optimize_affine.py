from click.testing import CliRunner

from mantis.cli.main import cli


def test_optimize_affine_cli(tmp_path, example_plate, example_register_settings):
    plate_path, _ = example_plate
    config_path, _ = example_register_settings
    output_path = tmp_path / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "optimize-registration",
            "-s",
            str(plate_path) + "/A/1/0",
            "-t",
            str(plate_path) + "/B/1/0",  # test could be improved with different stores
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    # Weak test
    # NOTE: we changed the output of the function so this is no longer printed. Do we need to compare with something?
    # assert "Getting dataset info" in result.output
    assert result.exit_code == 0
    assert output_path.exists()
