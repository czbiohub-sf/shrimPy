import yaml

from click.testing import CliRunner

from mantis.cli.main import cli


def test_concatenate_cli(example_plate, tmp_path, example_concatenate_settings):
    plate_path_1, _ = example_plate
    settings_path, settings = example_concatenate_settings
    config_path = tmp_path / "concat.yml"

    # Load the YAML settings
    with open(settings_path) as file:
        settings = yaml.safe_load(file)

    # Update the 'concat_data_paths' key in the settings
    settings["concat_data_paths"] = [
        str(plate_path_1) + '/*/*/*',
        str(plate_path_1) + '/*/*/*',
    ]

    # Save the modified settings back to a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(settings, file)

    output_path = tmp_path / "output.zarr"

    # Test deskew cli
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "concatenate",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )

    assert output_path.exists()
    assert result.exit_code == 0
