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
        patch("shrimpy.engines.isim_engine.ISIMCore", MagicMock()),
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


def test_gui_help_lists_photom(runner):
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0
    assert "pymmcore-gui" in result.output
    assert "photom" in result.output


def test_gui_photom_requires_config_or_demo(runner):
    result = runner.invoke(cli, ["gui", "photom"])
    assert result.exit_code == 2
    assert "--config is required unless --demo" in result.output


def test_gui_rejects_pymmcore_gui_options_with_subcommand(runner):
    result = runner.invoke(cli, ["gui", "--unicore", "photom", "--demo"])
    assert result.exit_code == 2
    assert "do not apply to `shrimpy gui photom`" in result.output


def test_gui_photom_demo_launches_only_photom(runner):
    """`shrimpy gui photom` runs the photom GUI and never touches pymmcore-gui."""
    pytest.importorskip("copylot.assemblies.photom.gui.photom_gui")
    with (
        patch("copylot.assemblies.photom.gui.photom_gui.launch", return_value=0) as launch,
        patch("pymmcore_gui._app.create_mmgui", create=True) as create_mmgui,
    ):
        result = runner.invoke(cli, ["gui", "photom", "--demo"])
    assert result.exit_code == 0, result.output
    launch.assert_called_once_with(None, demo=True, calibration_files=[])
    create_mmgui.assert_not_called()


def test_gui_photom_config_is_loaded(runner):
    pytest.importorskip("copylot.assemblies.photom.gui.photom_gui")
    from pathlib import Path

    example = Path(__file__).parent.parent / "config" / "photom" / "photom_VIS.yaml"
    with patch("copylot.assemblies.photom.gui.photom_gui.launch", return_value=0) as launch:
        result = runner.invoke(cli, ["gui", "photom", "--config", str(example)])
    assert result.exit_code == 0, result.output
    (config,), kwargs = launch.call_args
    assert kwargs == {"demo": False, "calibration_files": []}
    assert [m.name for m in config.mirrors] == ["mirror_1_VIS"]


def _calibration_file(tmp_path, name="affine.yml"):
    path = tmp_path / name
    path.write_text(
        "affine_transform_yx:\n- [1.0, 0.0, 0.0]\n- [0.0, 1.0, 0.0]\n- [0.0, 0.0, 1.0]\n"
    )
    return path


def test_gui_photom_calibration_is_passed_to_launch(runner, tmp_path):
    pytest.importorskip("copylot.assemblies.photom.gui.photom_gui")
    cal = _calibration_file(tmp_path)
    with patch("copylot.assemblies.photom.gui.photom_gui.launch", return_value=0) as launch:
        result = runner.invoke(cli, ["gui", "photom", "--demo", "--calibration", str(cal)])
    assert result.exit_code == 0, result.output
    assert launch.call_args.kwargs["calibration_files"] == [cal.resolve()]


def test_gui_photom_bad_calibration_fails_before_launch(runner, tmp_path):
    pytest.importorskip("copylot.assemblies.photom.gui.photom_gui")
    bad = tmp_path / "bad.yml"
    bad.write_text("not_a_calibration: 1\n")
    cal = _calibration_file(tmp_path)
    with patch("copylot.assemblies.photom.gui.photom_gui.launch") as launch:
        malformed = runner.invoke(cli, ["gui", "photom", "--demo", "--calibration", str(bad)])
        too_many = runner.invoke(
            cli,
            ["gui", "photom", "--demo", "--calibration", str(cal), "--calibration", str(cal)],
        )
    launch.assert_not_called()
    assert malformed.exit_code == 1 and "affine_transform_yx" in malformed.output
    assert too_many.exit_code == 1 and "pass at most one per mirror" in too_many.output


class _Recorder:
    """Stand-in engine/core/viewer that logs the order calls arrive in."""

    def __init__(self, *, acquire_error=None, join_error=None, unload_error=None):
        self.calls = []
        self._acquire_error = acquire_error
        self._join_error = join_error
        self._unload_error = unload_error
        self.mmcore = self

    # engine
    def acquire(self, **kwargs):
        self.calls.append("acquire")
        if self._acquire_error is not None:
            raise self._acquire_error

    # core
    def waitForSystem(self):
        self.calls.append("waitForSystem")

    def unloadAllDevices(self):
        self.calls.append("unloadAllDevices")
        if self._unload_error is not None:
            raise self._unload_error

    # viewer
    def disconnect(self):
        self.calls.append("disconnect")

    def join(self):
        self.calls.append("join")
        if self._join_error is not None:
            raise self._join_error

    def terminate(self):
        self.calls.append("terminate")


_RELEASE_THEN_WAIT = [
    "acquire",
    "disconnect",
    "waitForSystem",
    "unloadAllDevices",
    "join",
    "terminate",
]


def test_viewer_outlives_the_core():
    """The core is unloaded before the CLI waits on the viewer window."""
    from shrimpy.cli.acquire import _acquire_with_viewer

    rec = _Recorder()
    _acquire_with_viewer(rec, rec, output_dir="x", name="y", mda_config="z")
    assert rec.calls == _RELEASE_THEN_WAIT


def test_failed_acquisition_still_releases_the_core():
    from shrimpy.cli.acquire import _acquire_with_viewer

    rec = _Recorder(acquire_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _acquire_with_viewer(rec, rec)
    assert rec.calls == _RELEASE_THEN_WAIT


def test_unload_failure_does_not_close_the_viewer():
    from shrimpy.cli.acquire import _acquire_with_viewer

    rec = _Recorder(unload_error=RuntimeError("device stuck"))
    _acquire_with_viewer(rec, rec)
    assert rec.calls == _RELEASE_THEN_WAIT


def test_ctrl_c_while_viewing_closes_the_viewer():
    from shrimpy.cli.acquire import _acquire_with_viewer

    rec = _Recorder(join_error=KeyboardInterrupt())
    _acquire_with_viewer(rec, rec)
    assert rec.calls == _RELEASE_THEN_WAIT
