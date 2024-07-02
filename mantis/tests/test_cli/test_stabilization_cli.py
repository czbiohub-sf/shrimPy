from click.testing import CliRunner

from mantis.cli.main import cli


def test_estimate_stabilization(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "config.yml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-stabilization",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-j",
            "1",
            "-c",
            "0",
            "-y",
            "-z",
            "-v",
            "--crop-size-xy",
            "200",
            "200",
            '--stabilization-channel-indices',
            '0',
        ],
    )

    # Weak test
    assert "Estimating z stabilization parameters" in result.output
    assert output_path.exists()
    assert result.exit_code == 0


def test_apply_stabilization(tmp_path, example_plate, example_stabilize_timelapse_settings):
    plate_path, _ = example_plate
    config_path, _ = example_stabilize_timelapse_settings
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stabilize",
            "-i",
            str(plate_path) + "/A/1/0",
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "-j",
            "1",
        ],
    )

    # Weak test
    assert output_path.exists()
    assert result.exit_code == 0
