import pytest

from click.testing import CliRunner

from mantis.cli.main import cli


def test_main():
    with pytest.warns(DeprecationWarning) as record:
        runner = CliRunner()
        result = runner.invoke(cli)

    assert result.exit_code == 0
    assert "tools for mantis" in result.output
    assert "biahub" in str(record.list[0].message), "Deprecation warning was not found."
