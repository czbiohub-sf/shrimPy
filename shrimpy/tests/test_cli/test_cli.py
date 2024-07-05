from click.testing import CliRunner

from shrimpy.cli.main import cli


def test_main():
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0
    assert "tools for shrimpy" in result.output
