from click.testing import CliRunner

from mantis.cli.main import cli


def test_estimate_bleaching_cli(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "output/"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-bleaching",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-o",
            str(output_path),
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0
