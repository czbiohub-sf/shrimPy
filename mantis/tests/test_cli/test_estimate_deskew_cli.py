from click.testing import CliRunner

from mantis.cli.main import cli


def test_estimate_deskew_cli(tmp_path, example_fov):
    fov_path, _ = example_fov
    output_path = tmp_path / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "estimate-deskew",
            "-i",
            str(fov_path),
            "-o",
            str(output_path),
        ],
        "1\n",
    )

    # Tough to test hand-drawn things in napari, but at least this tests that it starts and loads data
    assert "Enter image pixel size" in result.output
