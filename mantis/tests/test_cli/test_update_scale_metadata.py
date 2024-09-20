import pytest
import os

from click.testing import CliRunner

from mantis.cli.main import cli


def test_update_scale_metadata(example_plate):
    plate_path, _ = example_plate
    with pytest.warns(DeprecationWarning) as record:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update-scale-metadata",
                "-i",
                os.path.join(str(plate_path), "A", "1", "0"),
                os.path.join(str(plate_path), "B", "1", "0"),
            ],
        )
    # Weak test
    assert "The first dataset" in result.output
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."
