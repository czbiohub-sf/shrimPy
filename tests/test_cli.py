"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch

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


def test_acquire_isim_help(runner):
    """Test that the isim command shows help message."""
    result = runner.invoke(cli, ["acquire", "isim", "--help"])
    assert result.exit_code == 0
    assert "Run iSIM microscope acquisition" in result.output
    for option in ("--mm-config", "--mda-config", "--output-dir", "--name", "--viewer"):
        assert option in result.output


def test_acquire_mantis_help(runner):
    """Test that the mantis command shows help message."""
    result = runner.invoke(cli, ["acquire", "mantis", "--help"])
    assert result.exit_code == 0
    assert "Run Mantis microscope acquisition" in result.output
    assert "--mm-config" in result.output
    assert "--mda-config" in result.output
    assert "--output-dir" in result.output
    assert "--name" in result.output


def test_acquire_mantis_help_lists_viewer_not_deprecated_alias(runner):
    result = runner.invoke(cli, ["acquire", "mantis", "--help"])
    assert result.exit_code == 0
    assert "--viewer" in result.output
    assert "--napari-viewer" not in result.output


def _invoke_acquire(runner, tmp_path, microscope, engine_path, *flags):
    """Invoke ``shrimpy acquire <microscope>`` with the core, engine and run mocked."""
    mm_cfg = tmp_path / "mm.cfg"
    mda_cfg = tmp_path / "mda.yaml"
    mm_cfg.touch()
    mda_cfg.touch()
    args = ["acquire", microscope, "--mm-config", mm_cfg, "--mda-config", mda_cfg]
    args += ["--output-dir", tmp_path, *flags]
    with (
        patch("shrimpy.cli.acquire.configure_logging", return_value=tmp_path / "log"),
        patch("shrimpy.robust_cmmcore.RobustCMMCore", MagicMock()),
        patch(engine_path, MagicMock()),
        patch("shrimpy.cli.acquire._acquire") as run,
    ):
        result = runner.invoke(cli, [str(a) for a in args])
    return result, run


@pytest.mark.parametrize(
    "microscope, engine_path, deskew",
    [
        ("mantis", "shrimpy.engines.mantis_engine.MantisEngine", True),
        ("isim", "shrimpy.engines.isim_engine.ISIMEngine", False),
        ("dragonfly", "shrimpy.engines.dragonfly_engine.DragonflyEngine", False),
    ],
)
@pytest.mark.parametrize("flags, expected", [((), False), (("--viewer",), True)])
def test_acquire_viewer_flag(
    runner, tmp_path, microscope, engine_path, deskew, flags, expected
):
    result, run = _invoke_acquire(runner, tmp_path, microscope, engine_path, *flags)
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["viewer"] is expected
    assert run.call_args.kwargs["deskew"] is deskew
    assert "DeprecationWarning" not in result.output


def test_acquire_mantis_napari_viewer_is_deprecated_alias(runner, tmp_path):
    result, run = _invoke_acquire(
        runner,
        tmp_path,
        "mantis",
        "shrimpy.engines.mantis_engine.MantisEngine",
        "--napari-viewer",
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["viewer"] is True
    assert "DeprecationWarning" in result.output
    assert "--viewer" in result.output
