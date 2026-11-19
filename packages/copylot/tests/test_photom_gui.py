"""The photom Qt GUI with simulated devices, offscreen (no event loop is run)."""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# qtpy imports fine without a binding and then raises QtBindingsNotFoundError on use,
# which importorskip("qtpy") would not catch.
try:
    from qtpy import QtWidgets
    from qtpy.QtCore import QPoint
except Exception:  # pragma: no cover - depends on the environment
    pytest.skip("no Qt bindings", allow_module_level=True)

from copylot.assemblies.photom.gui import photom_gui  # noqa: E402
from copylot.assemblies.photom.gui.photom_gui import (  # noqa: E402
    DEMO_CONFIG,
    PhotomConfig,
    demo_camera_factory,
    make_arduinos,
    make_photom_assembly,
)
from copylot.assemblies.photom.utils.affine_transform import AffineTransform  # noqa: E402
from copylot.assemblies.photom.utils.pattern_tracing import ShapeTrace  # noqa: E402

SENSOR_YX = (300, 400)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def config(tmp_path):
    return DEMO_CONFIG.model_copy(
        update={
            "camera_sensor_size_yx": SENSOR_YX,
            "calibration": DEMO_CONFIG.calibration.model_copy(
                update={"grid_n_points": 3, "save_dir": tmp_path / "calib"}
            ),
        },
        deep=True,
    )


@pytest.fixture
def app(qapp, config, monkeypatch):
    from copylot.assemblies.photom.gui import windows

    # Never block on a modal dialog
    monkeypatch.setattr(windows.QMessageBox, "critical", lambda *a, **k: None)

    assembly = make_photom_assembly(config, demo=True)
    window = windows.PhotomApp(
        photom_assembly=assembly,
        photom_sensor_size_yx=config.camera_sensor_size_yx,
        photom_window_size_x=200,
        demo_mode=True,
        arduino=make_arduinos(config, demo=True),
        camera_factory=demo_camera_factory(assembly, config),
        calibration=windows.CalibrationSettings(
            mirror_roi=[list(p) for p in config.calibration.mirror_roi],
            grid_n_points=config.calibration.grid_n_points,
            save_dir=config.calibration.save_dir,
        ),
    )
    yield window
    window.photom_window.close()
    window.close()


def _window_to_sensor(app, xy):
    # mirrors PhotomApp._window_scaling_matrix
    return np.array([xy[0] * app.scaling_factor_x + 1, xy[1] * app.scaling_factor_y + 1])


def test_config_loads_upstream_keys_and_resolves_paths(tmp_path):
    (tmp_path / "cfg").mkdir()
    path = tmp_path / "cfg" / "photom.yml"
    path.write_text(
        "lasers:\n  - {name: laser_405, COM_port: COM9}\n"
        "mirrors:\n  - {name: m1, COM_port: COM8, x_position: 0, y_position: 0,"
        " affine_matrix_path: affine.yml}\n"
        "calibration: {save_dir: out}\n"
    )
    config = PhotomConfig.from_yaml(path)
    assert config.lasers[0].com_port == "COM9"
    assert config.mirrors[0].affine_matrix_path == tmp_path / "cfg" / "affine.yml"
    assert config.calibration.save_dir == tmp_path / "cfg" / "out"
    assert config.arduinos == []


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "photom.yml"
    path.write_text("lasers: [{name: l, COM_prot: COM9}]\nmirrors: [{name: m}]\n")
    with pytest.raises(ValueError, match="COM_prot"):
        PhotomConfig.from_yaml(path)


def test_launch_requires_config_without_demo():
    with pytest.raises(ValueError, match="config is required"):
        photom_gui.launch(None, demo=False)


def test_example_config_validates():
    root = Path(__file__).resolve().parents[3]
    example = root / "config" / "photom" / "photom_VIS.yaml"
    if not example.exists():  # pragma: no cover - copylot checked out on its own
        pytest.skip("shrimPy example config not present")
    config = PhotomConfig.from_yaml(example)
    assert [laser.name for laser in config.lasers] == ["laser_405"]


def test_app_builds_windows_and_widgets(app, config):
    assert app.photom_window.isVisible()
    assert len(app.laser_widgets) == len(config.lasers)
    assert len(app.mirror_widgets) == len(config.mirrors)
    assert len(app.arduino_pwm_widgets) == 1
    assert app.calibrate_button.isEnabled()
    # the marker window keeps the sensor aspect ratio
    width, height = app.photom_window.window_geometry[-2:]
    assert width == 200 and height == int(200 * SENSOR_YX[0] / SENSOR_YX[1])


def test_calibrate_disabled_without_camera(qapp, config):
    from copylot.assemblies.photom.gui.windows import PhotomApp

    window = PhotomApp(
        photom_assembly=make_photom_assembly(config, demo=True),
        photom_sensor_size_yx=SENSOR_YX,
        photom_window_size_x=200,
    )
    try:
        assert not window.calibrate_button.isEnabled()
    finally:
        window.photom_window.close()
        window.close()


def test_marker_moves_mirror_through_window_affine(app):
    mirror = app.mirrors[0]
    app.update_laser_window_affine()  # identity calibration: window px -> sensor px
    app.photom_window.marker.setPos(50, 30)
    app.photom_window._move_marker_and_update_sliders()

    center = app.photom_window.get_marker_center(app.photom_window.marker)
    np.testing.assert_allclose(mirror.position, _window_to_sensor(app, center), atol=1e-4)


def test_laser_widget_power_and_pulse_mode(app):
    widget = app.laser_widgets[0]
    widget.power_edit.setText("42")
    widget.edit_power()
    assert widget.laser.power == 42
    widget.toggle_laser()
    assert widget.laser.toggle_emission == 1
    widget.laser_pulse_mode()
    assert widget.laser.pulse_mode == 1 and widget.laser.pulse_power == 42


def test_demo_calibration_end_to_end(app, tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    pytest.importorskip("tifffile")
    from copylot.assemblies.photom.gui import windows

    out = tmp_path / "affine.yml"
    monkeypatch.setattr(windows.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    app.calibrate_w_camera()
    app.calibration_w_cam_thread.wait()  # started as a real QThread
    assert app.calibration_w_cam_thread.isFinished()
    # its signals were queued to the GUI thread; deliver them
    QtWidgets.QApplication.processEvents()

    assert out.exists(), "done_calibration should have saved the matrix"
    assert app.photom_assembly.camera == []
    assert app.calibrate_button.isVisibleTo(app)
    saved = AffineTransform(config_file=out).get_affine_matrix()
    np.testing.assert_allclose(saved, app.T_mirror_cam_matrix)

    # After calibration, aiming at a window pixel puts the (simulated) spot on the
    # matching sensor pixel.
    camera = app.camera_factory()
    width, height = app.photom_window.window_geometry[-2:]
    for target in ([width / 2, height / 2], [width / 3, height / 4]):
        app.photom_assembly.set_position(0, target)
        np.testing.assert_allclose(camera.spot_xy(), _window_to_sensor(app, target), atol=2)


def test_calibration_failure_restores_buttons(app):
    class BrokenCamera:
        image_size = (400, 300, 0, 0)

        def snap(self):
            raise RuntimeError("camera unplugged")

    app.camera_factory = BrokenCamera
    app.calibrate_w_camera()
    app.calibration_w_cam_thread.wait()
    QtWidgets.QApplication.processEvents()

    assert app.calibrate_button.isVisibleTo(app)
    assert not app.cancel_calibration_button.isVisibleTo(app)
    assert app.photom_assembly.laser[0].toggle_emission is False


def test_shape_patterns(qapp):
    square = [QPoint(0, 0), QPoint(100, 0), QPoint(100, 100), QPoint(0, 100), QPoint(0, 0)]

    shape = ShapeTrace(square)
    shape._pattern_bidirectional(vertical_spacing=20, horizontal_spacing=20, num_points=10)
    assert len(shape.ablation_points) == 10
    assert shape.ablation_points[0] == (50, 50)

    spiral = ShapeTrace(square)
    spiral._pattern_spiral()  # num_points=None: as many as fit
    assert spiral.ablation_points
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in spiral.ablation_points)


# --- calibration files (CLI --calibration, config affine_matrix_path, Load Calibration)

CAL = np.array([[0.5, 0.0, -10.0], [0.0, 0.25, 5.0], [0.0, 0.0, 1.0]])


def _write_calibration(path, matrix=CAL):
    AffineTransform(tx_matrix=matrix).save_matrix(config_file=path)
    return path


def _two_mirror_config(config, affine_path=None):
    mirrors = [
        config.mirrors[0].model_copy(update={"affine_matrix_path": affine_path}),
        config.mirrors[0].model_copy(update={"name": "mirror_2_IR"}),
    ]
    return config.model_copy(update={"mirrors": mirrors})


def _make_app(config, **kwargs):
    from copylot.assemblies.photom.gui.windows import PhotomApp

    return PhotomApp(
        photom_assembly=make_photom_assembly(config, demo=True),
        photom_sensor_size_yx=config.camera_sensor_size_yx,
        photom_window_size_x=200,
        **kwargs,
    )


@pytest.fixture
def closing():
    windows = []
    yield windows.append
    for window in windows:
        window.photom_window.close()
        window.close()


def test_calibration_file_is_applied_like_load_calibration_and_shown(
    qapp, config, tmp_path, closing
):
    path = _write_calibration(tmp_path / "affine_VIS.yml")
    app = _make_app(config, calibration_files=[path])
    closing(app)

    np.testing.assert_allclose(app.T_mirror_cam_matrix, CAL)
    # the mirror's affine maps *window* pixels: calibration @ window -> sensor scaling
    np.testing.assert_allclose(
        app.mirrors[0].affine_transform_obj.get_affine_matrix(), CAL @ app.scaling_matrix
    )
    assert app.calibration_label.text() == "Calibration: affine_VIS.yml"
    assert app.calibration_label.toolTip() == str(path)
    assert app.photom_window.windowTitle() == "Laser Marker - affine_VIS.yml"


def test_uncalibrated_is_shown(qapp, config, closing):
    app = _make_app(config)
    closing(app)
    assert app.calibration_label.text() == "Calibration: none (uncalibrated)"
    assert app.photom_window.windowTitle() == "Laser Marker - uncalibrated"


def test_config_affine_matrix_path_gets_window_scaling(qapp, config, tmp_path, closing):
    """Regression: upstream left the startup matrix on the mirror unscaled."""
    path = _write_calibration(tmp_path / "from_config.yml")
    app = _make_app(_two_mirror_config(config, affine_path=path))
    closing(app)
    np.testing.assert_allclose(
        app.mirrors[0].affine_transform_obj.get_affine_matrix(), CAL @ app.scaling_matrix
    )
    assert app.calibration_label.text() == "Calibration: from_config.yml"


def test_calibrations_are_per_mirror(qapp, config, tmp_path, closing):
    first = _write_calibration(tmp_path / "vis.yml")
    second = _write_calibration(tmp_path / "ir.yml", matrix=np.diag([2.0, 3.0, 1.0]))
    app = _make_app(_two_mirror_config(config), calibration_files=[first, second])
    closing(app)

    assert app.calibration_label.text() == "Calibration: vis.yml"
    app.mirror_dropdown.setCurrentIndex(1)
    assert app.calibration_label.text() == "Calibration: ir.yml"

    # resizing the marker window rescales *both* mirrors, each with its own matrix
    app.resize_spinbox.setValue(150)
    for mirror, matrix in zip(app.mirrors, (CAL, np.diag([2.0, 3.0, 1.0])), strict=True):
        np.testing.assert_allclose(
            mirror.affine_transform_obj.get_affine_matrix(), matrix @ app.scaling_matrix
        )


def test_load_calibration_button_updates_display(qapp, config, tmp_path, closing, monkeypatch):
    from copylot.assemblies.photom.gui import windows

    path = _write_calibration(tmp_path / "picked.yaml")
    monkeypatch.setattr(
        windows.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "")
    )
    app = _make_app(config)
    closing(app)
    app.load_calibration()
    assert app.calibration_label.text() == "Calibration: picked.yaml"
    np.testing.assert_allclose(app.T_mirror_cam_matrix, CAL)


def test_unsaved_calibration_is_shown(app, monkeypatch):
    pytest.importorskip("skimage")
    from copylot.assemblies.photom.gui import windows

    monkeypatch.setattr(windows.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    app.calibrate_w_camera()
    app.calibration_w_cam_thread.wait()
    QtWidgets.QApplication.processEvents()
    assert app.calibration_label.text() == "Calibration: unsaved"


def test_check_calibration_files(config, tmp_path):
    good = _write_calibration(tmp_path / "good.yml")
    assert photom_gui.check_calibration_files(config, [good]) == [good]

    with pytest.raises(ValueError, match="2 calibration files but the config has 1 mirror"):
        photom_gui.check_calibration_files(config, [good, good])

    bad = tmp_path / "bad.yml"
    bad.write_text("affine_transform_yx: [[1, 0], [0, 1]]\n")
    with pytest.raises(ValueError, match="3x3"):
        photom_gui.check_calibration_files(config, [bad])

    with pytest.raises(FileNotFoundError):
        photom_gui.check_calibration_files(config, [tmp_path / "missing.yml"])
