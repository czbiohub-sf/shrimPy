"""Tests for the CLI module."""

import pytest

from click.testing import CliRunner

from shrimpy.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_cli_help(runner):
    """Test that the main CLI shows help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "shrimpy - Custom acquisition engines" in result.output
    assert "acquire" in result.output


def test_cli_version(runner):
    """Test that --version shows version information."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_acquire_help(runner):
    """Test that the acquire subcommand shows help message."""
    result = runner.invoke(cli, ["acquire", "--help"])
    assert result.exit_code == 0
    assert "Run microscope acquisitions" in result.output
    assert "mantis" in result.output
    assert "isim" in result.output


def test_acquire_mantis_help(runner):
    """Test that the mantis command shows help message."""
    result = runner.invoke(cli, ["acquire", "mantis", "--help"])
    assert result.exit_code == 0
    assert "Run Mantis microscope acquisition" in result.output
    assert "--mm-config" in result.output
    assert "--mda-config" in result.output
    assert "--output-dir" in result.output
    assert "--name" in result.output
