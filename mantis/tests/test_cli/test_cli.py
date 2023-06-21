from mantis.cli.main import cli
from click.testing import CliRunner


def test_main():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0
    assert "tools for mantis" in result.output
