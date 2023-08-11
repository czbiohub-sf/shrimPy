from click.testing import CliRunner

from mantis.cli.main import cli


def test_estimate_phase_to_fluor_affine_cli(tmp_path, example_plate):
    plate_path, _ = example_plate
    output_path = tmp_path / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-phase-to-fluor-affine",
            "-lf",
            str(plate_path) + "/A/1/0",
            "-ls",
            str(plate_path) + "/B/1/0",  # test could be improved with different stores
            "-o",
            str(output_path),
        ],
    )

    # Weak test
    assert "Enter phase_channel index to process" in result.output
