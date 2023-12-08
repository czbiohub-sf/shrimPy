from click.testing import CliRunner

from mantis.cli.main import cli


def test_estimate_stabilization_affine_list(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-stabilization-affine",
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
            "-s",
            "200",
            "200",
        ],
    )

    # Weak test
    assert "Getting dataset info" in result.output


def test_stabilize_timelapse(tmp_path, example_plate, example_stabilize_timelapse_settings):
    plate_path, _ = example_plate
    config_path, _ = example_stabilize_timelapse_settings
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stabilize-timelapse",
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
    assert "Getting dataset info" in result.output
