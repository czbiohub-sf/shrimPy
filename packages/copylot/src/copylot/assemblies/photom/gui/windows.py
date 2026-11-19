"""Photom GUI windows: the control panel and the laser-marker overlay.

Vendored from coPylot ``copylot/assemblies/photom/gui/windows.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE. Ported from PyQt5 to
qtpy (fully qualified enums, ``QScreen`` instead of the removed
``QApplication.desktop()``, ``QEvent.Type`` instead of ``QMouseEvent`` event
types).

Changes from upstream:

- the calibration camera comes from a ``camera_factory`` (upstream opened
  coPylot's FLIR camera, or its mock in demo mode); without one the Calibrate
  button is disabled. The calibration ROI, grid and output directory come from
  :class:`CalibrationSettings` instead of being hardcoded
- one :meth:`PhotomApp.update_laser_window_affine` composes the camera
  calibration with the window scaling everywhere. Upstream computed the scaling
  three ways; after a calibration it scaled y by the window *width* and ignored
  the Resize setting, so the marker was off in y for non-square sensors
- one camera calibration per mirror (upstream kept a single matrix for all
  mirrors), loaded through :meth:`PhotomApp.load_calibration_file` by the Load
  Calibration button, by ``calibration_files`` (``shrimpy gui photom
  --calibration``), and from each mirror's ``affine_matrix_path`` at startup.
  Upstream put the startup matrix on the mirror raw, without the window scaling
  that Load Calibration applies. The control panel and the Laser Marker title
  show the current mirror's calibration file
- after a calibration the window affine is applied even if the save dialog is
  cancelled (upstream left the raw camera-to-mirror matrix in place, so the
  marker no longer mapped to the mirror), and the mirror is recentered on the
  window center (upstream passed sensor (y, x) coordinates through the window
  affine)
- a left-press anywhere in the Laser Marker window moves the marker there and
  drags it; upstream only dragged on a direct hit on the 40 px marker, and a
  press elsewhere snapped the laser to the marker and then ignored the drag
- the marker shows where the laser is: at startup, after loading a calibration,
  on mirror switch and on resize it is placed at the mirror's actual position
  (upstream centred it, so the first press jumped the laser). Switching mirrors
  no longer resets the selected mirror to (0, 0), Recenter Marker also steers the
  laser to the centre, and the mirror sliders start at the mirror's position
- dragging the marker steers the mirror to where the marker *is*: upstream
  steered before the view moved the marker, so the laser trailed one mouse event
  behind. Mirror updates are also coalesced, and a target outside the mirror's
  range (every target when uncalibrated, which pins the laser at the edge) is
  reported in the log and the status bar instead of being clamped silently
- a calibration that raises shows an error and restores the buttons (upstream
  left the GUI stuck in "Cancel Calibration")
- unused imports removed (including ``networkx``, which was not a dependency);
  logging instead of ``print``
"""

from __future__ import annotations

import logging
import time

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from qtpy.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
)
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from copylot.assemblies.photom.gui.utils import (
    CalibrationWithCameraThread,
    ClickablePixmapItem,
)
from copylot.assemblies.photom.gui.widgets import (
    ArduinoPWMWidget,
    LaserWidget,
    MirrorWidget,
)
from copylot.assemblies.photom.photom import Camera, PhotomAssembly
from copylot.assemblies.photom.utils.pattern_tracing import ShapeTrace
from copylot.assemblies.photom.utils.scanning_algorithms import calculate_rectangle_corners

logger = logging.getLogger(__name__)

MARKER_IMAGE = Path(__file__).parent.parent / "utils" / "images" / "hit_marker_red.png"

#: Mouse events written to the debug log (``shrimpy gui photom --debug``).
_TRACED_MOUSE_EVENTS = (
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseMove,
    QEvent.Type.MouseButtonRelease,
)


def _event_pos(event) -> QPoint:
    """Mouse event position as a QPoint (Qt6 ``position()``, Qt5 ``pos()``)."""
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


@dataclass
class CalibrationSettings:
    """How the Calibrate button runs ``PhotomAssembly.calibrate_w_camera``."""

    #: ``[[x0, y0], [x1, y1]]`` mirror-coordinate extent of the calibration grid.
    mirror_roi: list[list[float]] = field(
        default_factory=lambda: [[-0.01, -0.03], [0.015, 0.0]]
    )
    #: Grid points per row and per column.
    grid_n_points: int = 5
    #: Where calibration stacks, plots and points are written (default: cwd).
    save_dir: Path | None = None


class PhotomApp(QMainWindow):
    """Control panel: lasers, mirrors, Arduino PWM, drawing mode, calibration.

    Parameters
    ----------
    photom_assembly : PhotomAssembly
        The lasers and mirrors to control.
    photom_sensor_size_yx : (int, int)
        Camera sensor shape the laser-marker window is scaled to.
    photom_window_size_x : int
        Width of the control and laser-marker windows, in pixels.
    photom_window_pos : (int, int)
        Screen position of the control window; the marker window sits to its right.
    demo_mode : bool
        Only labels the window title.
    arduino : list
        Arduino PWM controllers, one widget each.
    camera_factory : callable, optional
        Returns an opened camera for calibration (closed again afterwards). The
        Calibrate button is disabled without one.
    calibration : CalibrationSettings, optional
        Calibration ROI, grid size and output directory.
    calibration_files : list of (Path or None), optional
        Camera-to-mirror calibration YAML per mirror, loaded as if by Load
        Calibration. ``None`` (or a shorter list) leaves a mirror with the
        calibration from its ``affine_matrix_path``, if any.
    """

    def __init__(
        self,
        photom_assembly: PhotomAssembly,
        photom_sensor_size_yx: tuple[int, int] = (2048, 2448),
        photom_window_size_x: int = 800,
        photom_window_pos: tuple[int, int] = (100, 100),
        demo_mode: bool = False,
        arduino: list | None = None,
        camera_factory: Callable[[], Camera] | None = None,
        calibration: CalibrationSettings | None = None,
        calibration_files: list[Path | None] | None = None,
    ):
        super().__init__()
        # TODO (upstream): temporary for the Arduino; remove when replaced with a DAC
        self.arduino_pwm = list(arduino or [])
        self.camera_factory = camera_factory
        self.calibration = calibration or CalibrationSettings()

        self.photom_window = None
        self.photom_controls_window = None

        self.photom_assembly = photom_assembly
        self.lasers = self.photom_assembly.laser
        self.mirrors = self.photom_assembly.mirror
        self.photom_window_size_x = photom_window_size_x
        self.photom_sensor_size_yx = photom_sensor_size_yx
        self.photom_window_pos = photom_window_pos
        self._current_mirror_idx = 0
        self._laser_window_transparency = 0.7
        self.scaling_matrix = np.eye(3)
        # Per mirror: camera -> mirror calibration, and the file it came from
        # (None: uncalibrated, or calibrated but not saved).
        self.T_mirror_cam_matrices = [np.eye(3) for _ in self.mirrors]
        self.calibration_sources: list[str | None] = [None] * len(self.mirrors)
        self.calibration_w_cam_thread = CalibrationWithCameraThread(
            self.photom_assembly,
            self._current_mirror_idx,
            mirror_roi=self.calibration.mirror_roi,
            grid_n_points=self.calibration.grid_n_points,
            save_dir=self.calibration.save_dir,
        )
        self.calibration_w_cam_thread.finished.connect(self.done_calibration)
        self.calibration_w_cam_thread.failed.connect(self.calibration_failed)
        self.imageWindows = []
        self.drawingTraces = []
        self.currentTrace = []

        self.demo_mode = demo_mode

        self.initializer_laser_marker_window()
        self.initialize_UI()
        self._load_initial_calibrations(calibration_files or [])

    @property
    def T_mirror_cam_matrix(self) -> np.ndarray:
        """Camera -> mirror calibration of the current mirror."""
        return self.T_mirror_cam_matrices[self._current_mirror_idx]

    @T_mirror_cam_matrix.setter
    def T_mirror_cam_matrix(self, matrix) -> None:
        self.T_mirror_cam_matrices[self._current_mirror_idx] = np.asarray(matrix)

    def _load_initial_calibrations(self, calibration_files: list[Path | None]) -> None:
        if len(calibration_files) > len(self.mirrors):
            raise ValueError(
                f"Got {len(calibration_files)} calibration files for "
                f"{len(self.mirrors)} mirrors"
            )
        for idx, mirror in enumerate(self.mirrors):
            path = calibration_files[idx] if idx < len(calibration_files) else None
            if path is None:
                # The assembly loaded the mirror's affine_matrix_path, if any, as a raw
                # camera -> mirror matrix; adopt it as that mirror's calibration.
                path = getattr(mirror.affine_transform_obj, "config_file", None)
            if path is not None:
                self.load_calibration_file(path, mirror_idx=idx)
        self.update_laser_window_affine()
        self._update_calibration_display()
        self.sync_marker_to_mirror()

    def sync_marker_to_mirror(self, warn: bool = True) -> None:
        """Place the marker where the current mirror points; the mirror does not move.

        Inverts the mirror's window affine (calibration @ window scaling) at its
        actual position, so the marker shows the laser from startup (upstream
        centred it, and the first press jumped the laser to the centre). A position
        that lands outside the window, e.g. when uncalibrated, centres the marker
        with a warning instead.
        """
        if self.photom_window is None:
            return
        idx = self._current_mirror_idx
        mirror = self.mirrors[idx]
        marker_window = self.photom_window
        width, height = marker_window.window_geometry[-2:]
        mirror_xy = [float(v) for v in mirror.position]
        try:
            window_to_mirror = np.asarray(
                mirror.affine_transform_obj.get_affine_matrix(), float
            )
            wx, wy, w = np.linalg.inv(window_to_mirror) @ [mirror_xy[0], mirror_xy[1], 1.0]
            wx, wy = wx / w, wy / w
        except np.linalg.LinAlgError:
            wx = wy = np.nan
        if np.isfinite(wx) and np.isfinite(wy) and 0 <= wx <= width and 0 <= wy <= height:
            marker_window.display_marker_center(marker_window.marker, (wx, wy))
            logger.debug(
                f"Marker placed at {mirror.name}'s position {mirror_xy} -> window ({wx:.0f}, {wy:.0f})"
            )
        else:
            marker_window.recenter_marker()
            if not warn:
                return
            uncalibrated = self.calibration_sources[idx] is None and np.allclose(
                self.T_mirror_cam_matrices[idx], np.eye(3)
            )
            message = (
                f"{mirror.name} points at {[round(v, 5) for v in mirror_xy]}, outside the "
                f"Laser Marker window{' (uncalibrated)' if uncalibrated else ''}; the "
                "marker is centred and does not show the laser until you drag it."
            )
            logger.warning(message)
            self.statusBar().showMessage(message, 10000)
        for widget in getattr(self, "mirror_widgets", [])[idx : idx + 1]:
            widget.show_position(mirror_xy)

    def load_calibration_file(self, path: Path | str, mirror_idx: int | None = None) -> None:
        """Load a camera -> mirror calibration YAML for a mirror (default: the current one)."""
        idx = self._current_mirror_idx if mirror_idx is None else mirror_idx
        affine = self.photom_assembly.mirror[idx].affine_transform_obj
        affine.load_matrix(config_file=path)
        logger.info(f"Loaded calibration for {self.mirrors[idx].name} from {path}")
        logger.debug(f"Calibration matrix: {affine.T_affine}")
        self.T_mirror_cam_matrices[idx] = affine.get_affine_matrix()
        self.calibration_sources[idx] = str(path)
        self.update_laser_window_affine()
        self._update_calibration_display()
        if idx == self._current_mirror_idx:
            self.sync_marker_to_mirror()

    def _update_calibration_display(self) -> None:
        source = self.calibration_sources[self._current_mirror_idx]
        if source is not None:
            name = Path(source).name
            text = f"Calibration: {name}"
        elif not np.allclose(self.T_mirror_cam_matrix, np.eye(3)):
            text = name = "Calibration: unsaved"
        else:
            text, name = "Calibration: none (uncalibrated)", "uncalibrated"
        self.calibration_label.setText(text)
        self.calibration_label.setToolTip(
            source
            or (
                "Uncalibrated: the marker does not map onto the mirror, so dragging it "
                "pins the laser at the edge of the mirror's range. Load a calibration."
                if name == "uncalibrated"
                else ""
            )
        )
        self.photom_window.setWindowTitle(f"{self.photom_window.window_name} - {name}")

    def initializer_laser_marker_window(self):
        # The marker window has the sensor's aspect ratio and sits right of the controls.
        window_pos = (
            self.photom_window_size_x + self.photom_window_pos[0],
            self.photom_window_pos[1],
        )
        self.photom_window = LaserMarkerWindow(
            photom_controls=self,
            name="Laser Marker",
            sensor_size_yx=self.photom_sensor_size_yx,
            fixed_width=self.photom_window_size_x,
            window_pos=window_pos,
        )
        self.aspect_ratio = self.photom_sensor_size_yx[1] / self.photom_sensor_size_yx[0]
        self.scaling_matrix = self._window_scaling_matrix()
        self.photom_window.windowClosed.connect(self.closeAllWindows)

    def _window_scaling_matrix(self) -> np.ndarray:
        """Laser-marker window pixels (x, y) -> camera sensor pixels (x, y)."""
        width, height = self.photom_window.window_geometry[-2:]
        self.scaling_factor_x = self.photom_sensor_size_yx[1] / width
        self.scaling_factor_y = self.photom_sensor_size_yx[0] / height
        # TODO (upstream): the affine transforms are in XY coordinates. Need to change to YX
        return np.array(
            [[self.scaling_factor_x, 0, 1], [0, self.scaling_factor_y, 1], [0, 0, 1]]
        )

    def initialize_UI(self):
        self.setGeometry(
            self.photom_window_pos[0],
            self.photom_window_pos[1],
            self.photom_window_size_x,
            self.photom_window_size_x,
        )
        title = "Laser and Mirror Control App"
        self.setWindowTitle(f"{title} (demo)" if self.demo_mode else title)

        # Transparency and size of the laser-marker window
        transparency_group = QGroupBox("Photom Transparency")
        transparency_layout = QVBoxLayout()
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setMinimum(0)
        self.transparency_slider.setMaximum(100)
        self.transparency_slider.setValue(int(self._laser_window_transparency * 100))
        self.transparency_slider.valueChanged.connect(self.update_transparency)
        transparency_layout.addWidget(self.transparency_slider)

        self.transparency_label = QLabel("Transparency: 100%")
        transparency_layout.addWidget(self.transparency_label)

        self.resize_spinbox = QSpinBox()
        self.resize_spinbox.setRange(50, 400)
        self.resize_spinbox.setSuffix("%")
        self.resize_spinbox.setValue(100)
        self.resize_spinbox.valueChanged.connect(self.resize_laser_marker_window)
        self.resize_spinbox.editingFinished.connect(self.resize_laser_marker_window)
        transparency_layout.addWidget(QLabel("Resize Window:"))
        transparency_layout.addWidget(self.resize_spinbox)
        transparency_group.setLayout(transparency_layout)

        laser_group = QGroupBox("Lasers")
        laser_layout = QVBoxLayout()
        self.laser_widgets = []
        for laser in self.lasers:
            laser_widget = LaserWidget(laser)
            self.laser_widgets.append(laser_widget)
            laser_layout.addWidget(laser_widget)
        laser_group.setLayout(laser_layout)

        mirror_group = QGroupBox("Mirror")
        mirror_layout = QVBoxLayout()
        self.mirror_widgets = []
        for mirror in self.mirrors:
            mirror_widget = MirrorWidget(mirror)
            self.mirror_widgets.append(mirror_widget)
            mirror_layout.addWidget(mirror_widget)
        mirror_group.setLayout(mirror_layout)

        # TODO (upstream): remove if the Arduino is removed
        arduino_group = QGroupBox("Arduino PWM")
        arduino_layout = QVBoxLayout()
        self.arduino_pwm_widgets = []
        for arduino in self.arduino_pwm:
            arduino_pwm_widget = ArduinoPWMWidget(self.photom_assembly, arduino, self)
            self.arduino_pwm_widgets.append(arduino_pwm_widget)
            arduino_layout.addWidget(arduino_pwm_widget)
        arduino_group.setLayout(arduino_layout)
        if not self.arduino_pwm:
            arduino_group.hide()

        main_layout = QVBoxLayout()

        self.game_mode_button = QPushButton("Game Mode: OFF", self)
        self.game_mode_button.setCheckable(True)
        self.game_mode_button.clicked.connect(self.toggle_game_mode)

        self.toggle_drawing_mode_button = QPushButton("Toggle Drawing Mode", self)
        self.toggle_drawing_mode_button.clicked.connect(self.toggleDrawingMode)
        self.playButton = QPushButton("Play", self)
        self.playButton.clicked.connect(self.play_drawing)
        self.playButton.hide()

        # Drawing mode: ROI and pattern selection
        self.drawing_dropdowns = QHBoxLayout()
        self.drawing_dropdowns_widget = QWidget()

        self.roi_dropdown = QComboBox(self)
        self.roi_dropdown.addItem("Select ROI")
        self.photom_window.shapesUpdated.connect(self.updateRoiDropdown)
        self.roi_dropdown.currentIndexChanged.connect(self.onRoiSelected)
        self.drawing_dropdowns.addWidget(self.roi_dropdown)

        self.patterns = ["Bidirectional", "Spiral"]
        self.pattern_dropdown = QComboBox(self)
        self.pattern_dropdown.addItem("Select Pattern")
        self.addPatternDropdownItems()
        self.pattern_dropdown.currentIndexChanged.connect(self.onPatternSelected)
        self.drawing_dropdowns.addWidget(self.pattern_dropdown)
        self.drawing_dropdowns_widget.setLayout(self.drawing_dropdowns)

        # Bidirectional pattern parameters
        self.bidirectional_params = QHBoxLayout()
        self.horizontal_spacing_label = QLabel("H Spacing:", self)
        self.bidirectional_params.addWidget(self.horizontal_spacing_label)
        self.horizontal_spacing_input = QLineEdit(self)
        self.bidirectional_params.addWidget(self.horizontal_spacing_input)

        self.vertical_spacing_label = QLabel("V Spacing:", self)
        self.bidirectional_params.addWidget(self.vertical_spacing_label)
        self.vertical_spacing_input = QLineEdit(self)
        self.bidirectional_params.addWidget(self.vertical_spacing_input)

        self.num_points_label = QLabel("No. Points:", self)
        self.bidirectional_params.addWidget(self.num_points_label)
        self.num_points_input = QLineEdit(self)
        self.bidirectional_params.addWidget(self.num_points_input)

        self.bidirectional_params_widget = QWidget()
        self.bidirectional_params_widget.setLayout(self.bidirectional_params)
        self.bidirectional_params_widget.hide()

        # Spiral pattern parameters
        self.spiral_params = QHBoxLayout()
        self.spiral_nums_label = QLabel("Number of Points:", self)
        self.spiral_params.addWidget(self.spiral_nums_label)
        self.spiral_nums_input = QLineEdit(self)
        self.spiral_params.addWidget(self.spiral_nums_input)
        self.spiral_params_widget = QWidget()
        self.spiral_params_widget.setLayout(self.spiral_params)
        self.spiral_params_widget.hide()

        # Delay between pattern points
        self.delay_time_widget = QWidget()
        self.delay_label = QLabel("Delay (s):", self)
        self.delay_input = QLineEdit(self)
        self.delay_time_layout = QHBoxLayout()
        self.delay_time_layout.addWidget(self.delay_label)
        self.delay_time_layout.addWidget(self.delay_input)
        self.delay_time_widget.setLayout(self.delay_time_layout)
        self.delay_time_widget.hide()

        self.pattern_and_delete_buttons = QHBoxLayout()
        self.apply_pattern_button = QPushButton("Apply Pattern", self)
        self.apply_pattern_button.setMinimumSize(200, 50)
        self.apply_pattern_button.clicked.connect(self.onApplyPatternClick)
        self.pattern_and_delete_buttons.addWidget(self.apply_pattern_button)

        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setMinimumSize(200, 50)
        self.delete_button.clicked.connect(self.onDeleteClick)
        self.pattern_and_delete_buttons.addWidget(self.delete_button)
        self.pattern_and_delete_widget = QWidget()
        self.pattern_and_delete_widget.setLayout(self.pattern_and_delete_buttons)

        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(lambda: self.photom_window._roi_tracing())

        self.drawing_mode_layout = QVBoxLayout()
        self.drawing_mode_layout.addWidget(self.playButton)
        self.drawing_mode_layout.addWidget(self.drawing_dropdowns_widget)
        self.drawing_mode_layout.addWidget(self.bidirectional_params_widget)
        self.drawing_mode_layout.addWidget(self.spiral_params_widget)
        self.drawing_mode_layout.addWidget(self.delay_time_widget)
        self.drawing_mode_layout.addWidget(self.pattern_and_delete_widget)
        self.drawing_mode_layout.addWidget(self.run_button)

        self.drawing_mode_group = QGroupBox("Drawing Mode")
        self.drawing_mode_group.setLayout(self.drawing_mode_layout)
        self.drawing_mode_group.hide()

        main_layout.addWidget(transparency_group)
        main_layout.addWidget(self.game_mode_button)
        main_layout.addWidget(self.toggle_drawing_mode_button)
        main_layout.addWidget(self.drawing_mode_group)
        main_layout.addWidget(laser_group)
        main_layout.addWidget(mirror_group)
        main_layout.addWidget(arduino_group)

        self.mirror_dropdown = QComboBox()
        self.mirror_dropdown.addItems([mirror.name for mirror in self.mirrors])
        main_layout.addWidget(self.mirror_dropdown)
        self.mirror_dropdown.setCurrentIndex(self._current_mirror_idx)
        self.mirror_dropdown.currentIndexChanged.connect(self.mirror_dropdown_changed)

        self.recenter_marker_button = QPushButton("Recenter Marker")
        self.recenter_marker_button.clicked.connect(self.recenter_marker)
        main_layout.addWidget(self.recenter_marker_button)

        self.calibration_label = QLabel("Calibration: none (uncalibrated)")
        self.calibration_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        main_layout.addWidget(self.calibration_label)

        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.calibrate_w_camera)
        if self.camera_factory is None:
            self.calibrate_button.setEnabled(False)
            self.calibrate_button.setToolTip(
                "No calibration camera configured; use Load Calibration"
            )
        main_layout.addWidget(self.calibrate_button)

        self.load_calibration_button = QPushButton("Load Calibration")
        self.load_calibration_button.clicked.connect(self.load_calibration)
        main_layout.addWidget(self.load_calibration_button)

        self.cancel_calibration_button = QPushButton("Cancel Calibration")
        self.cancel_calibration_button.clicked.connect(self.cancel_calibration)
        self.cancel_calibration_button.hide()
        main_layout.addWidget(self.cancel_calibration_button)
        main_widget = QWidget(self)
        main_widget.setLayout(main_layout)

        scroll_area = QScrollArea(self)
        scroll_area.setWidget(main_widget)
        scroll_area.setWidgetResizable(True)

        self.setCentralWidget(scroll_area)
        self.show()

    @property
    def _current_affine(self):
        return self.photom_assembly.mirror[self._current_mirror_idx].affine_transform_obj

    def toggle_game_mode(self, checked):
        self.game_mode_button.setText(f"Game Mode: {'ON' if checked else 'OFF'}")
        self.photom_window.game_mode = bool(checked)

    def toggleDrawingMode(self):
        if (
            self.photom_window.stacked_widget.currentWidget()
            == self.photom_window.drawing_view
        ):
            # Leaving drawing mode clears the drawing
            self.photom_window.clearDrawing()
            self.drawing_mode_group.hide()
        else:
            self.drawing_mode_group.show()
        self.photom_window.toggleDrawingScene()

    def resize_laser_marker_window(self):
        percentage = self.resize_spinbox.value() / 100.0
        new_width = int(self.photom_window_size_x * percentage)
        new_height = int(new_width / self.photom_window.aspect_ratio)
        self.photom_window.update_window_geometry(new_width, new_height)
        self.update_laser_window_affine()
        self.sync_marker_to_mirror()  # same laser spot, new window scale

    def mirror_dropdown_changed(self, index):
        # Upstream reset the newly selected mirror to (0, 0); it now stays where it
        # is (e.g. its configured start position) and the marker moves to it.
        logger.info(f"Mirror dropdown changed to index {index}")
        self._current_mirror_idx = index
        self._update_calibration_display()
        self.sync_marker_to_mirror()

    def recenter_marker(self):
        """Centre the marker and steer the laser there (upstream moved only the marker)."""
        self.photom_window.recenter_marker()
        self.photom_window._move_marker_and_update_sliders()

    def calibrate_w_camera(self):
        if self.camera_factory is None:
            return
        logger.info("Calibrating with camera...")
        self.calibrate_button.hide()
        self.load_calibration_button.hide()
        self.cancel_calibration_button.show()

        self.setup_calibration()
        self._current_affine.reset_T_affine()
        self.calibration_w_cam_thread.current_mirror_idx = self._current_mirror_idx
        self.calibration_w_cam_thread.start()

    def setup_calibration(self):
        """Open the calibration camera and turn on the first laser at 30%."""
        self.photom_assembly.camera = [self.camera_factory()]

        self.photom_assembly.laser[0].power = 0.0
        self.photom_assembly.laser[0].toggle_emission = True
        self.photom_assembly.laser[0].power = 30.0

    def _release_calibration_camera(self):
        for cam in self.photom_assembly.camera or []:
            close = getattr(cam, "close", None)
            if close is not None:
                close()
        self.photom_assembly.camera = []

    def load_calibration(self):
        typed_filename, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration File", "", "YAML Files (*.yml *.yaml)"
        )
        if typed_filename:
            self.load_calibration_file(typed_filename)
            self.photom_window.switch_to_shooting_scene()
            self.photom_window.marker.show()

    def cancel_calibration(self):
        logger.info("Canceling calibration...")
        self.calibrate_button.show()
        self.load_calibration_button.show()
        self.photom_window.marker.show()
        self.cancel_calibration_button.hide()
        self.photom_window.switch_to_shooting_scene()

    def display_saved_plot(self, plot_path):
        image_window = ImageWindow(plot_path)
        image_window.show()
        self.imageWindows.append(image_window)

    def calibration_failed(self, message: str):
        self._release_calibration_camera()
        self.photom_assembly.laser[0].toggle_emission = False
        self.cancel_calibration()
        QMessageBox.critical(self, "Calibration failed", message)

    def done_calibration(self, t_affine, plot_save_path):
        self._release_calibration_camera()

        if plot_save_path and plot_save_path != "None":
            self.display_saved_plot(plot_save_path)
        self.T_mirror_cam_matrix = t_affine

        typed_filename, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "YAML Files (*.yml)"
        )
        if typed_filename:
            if not typed_filename.endswith((".yml", ".yaml")):
                typed_filename += ".yml"
            logger.info(f"Saving calibration to {typed_filename}")
            self._current_affine.save_matrix(
                matrix=self.T_mirror_cam_matrix, config_file=typed_filename
            )
            self.calibration_sources[self._current_mirror_idx] = typed_filename
        else:
            logger.warning("No file selected; the calibration matrix was not saved.")
            self.calibration_sources[self._current_mirror_idx] = None

        self.photom_window.switch_to_shooting_scene()
        self.photom_window.marker.show()
        self.calibrate_button.show()
        self.load_calibration_button.show()
        self.cancel_calibration_button.hide()

        self.update_laser_window_affine()
        self._update_calibration_display()
        logger.info("Calibration done")

        width, height = self.photom_window.window_geometry[-2:]
        self.photom_window.recenter_marker()
        self.photom_assembly.set_position(self._current_mirror_idx, [width / 2, height / 2])

    def update_laser_window_affine(self):
        """Each mirror's affine = its camera calibration @ window -> sensor scaling."""
        self.scaling_matrix = self._window_scaling_matrix()
        for mirror, T_mirror_cam in zip(self.mirrors, self.T_mirror_cam_matrices, strict=True):
            T_compose_mat = T_mirror_cam @ self.scaling_matrix
            mirror.affine_transform_obj.set_affine_matrix(T_compose_mat)
            logger.debug(f"Updated {mirror.name} affine matrix: {T_compose_mat}")

    def update_transparency(self, value):
        self.transparency_label.setText(f"Transparency: {value}%")
        self.photom_window.setWindowOpacity(1.0 - (value / 100.0))

    def play_drawing(self):
        for trace in self.photom_window.drawingTraces:
            logger.debug(f"Playing trace: {trace}")
            self.photom_assembly.set_position(self._current_mirror_idx, trace)

    def display_rectangle(self):
        self.photom_window.switch_to_calibration_scene()

    def closeEvent(self, event):
        self.closeAllWindows()
        super().closeEvent(event)

    def closeAllWindows(self):
        self.photom_window.close()
        for window in self.imageWindows:
            window.close()
        self.close()
        QApplication.quit()

    def updateRoiDropdown(self) -> None:
        """Refresh the ROI dropdown from the drawn shapes."""
        self.roi_dropdown.clear()
        self.roi_dropdown.addItem("Select ROI")
        for roi in self.photom_window.shapes.keys():
            self.roi_dropdown.addItem(f"Shape {roi + 1}")

    def onRoiSelected(self) -> None:
        selected_roi = self.roi_dropdown.currentText()
        if selected_roi:
            if selected_roi == "Select ROI":
                self.photom_window.selected_shape_id = None
            else:
                self.photom_window.selected_shape_id = int(selected_roi.split(" ")[1]) - 1
        self.photom_window.update()

    def addPatternDropdownItems(self) -> None:
        for pattern in self.patterns:
            self.pattern_dropdown.addItem(pattern)

    def onPatternSelected(self) -> None:
        selected_pattern = self.pattern_dropdown.currentText()
        self.bidirectional_params_widget.setVisible(selected_pattern == "Bidirectional")
        self.spiral_params_widget.setVisible(selected_pattern == "Spiral")
        self.delay_time_widget.setVisible(selected_pattern in self.patterns)

    def onApplyPatternClick(self) -> None:
        """Fill the selected ROI with the selected pattern."""
        selected_roi = self.roi_dropdown.currentText()
        if selected_roi == "Select ROI":
            return
        roi_number = int(selected_roi.split(" ")[1]) - 1
        selected_pattern = self.pattern_dropdown.currentText()
        shape = self.photom_window.shapes[roi_number]

        try:
            if selected_pattern == "Bidirectional":
                horizontal_spacing = int(self.horizontal_spacing_input.text())
                vertical_spacing = int(self.vertical_spacing_input.text())
                num_points = self.num_points_input.text()
                shape.pattern_points = []
                shape.ablation_points = []
                shape._pattern_bidirectional(
                    vertical_spacing=vertical_spacing,
                    horizontal_spacing=horizontal_spacing,
                    num_points=int(num_points) if num_points else None,
                )
            elif selected_pattern == "Spiral":
                num_points = self.spiral_nums_input.text()
                shape.ablation_points.clear()
                shape._pattern_spiral(num_points=int(num_points) if num_points else None)
        except ValueError:
            logger.warning("Invalid pattern parameter")

        self.photom_window.update()

    def onDeleteClick(self) -> None:
        selected_roi = self.roi_dropdown.currentText()
        if selected_roi and selected_roi != "Select ROI":
            roi_id = int(selected_roi.split(" ")[1]) - 1
            del self.photom_window.shapes[roi_id]
            self.photom_window.update()
            self.updateRoiDropdown()


class LaserMarkerWindow(QMainWindow):
    """Semi-transparent overlay: drag the marker to steer the mirror, right-click to fire."""

    windowClosed = Signal()
    shapesUpdated = Signal()

    def __init__(
        self,
        photom_controls: PhotomApp | None = None,
        name="Laser Marker",
        sensor_size_yx: tuple = (2048, 2048),
        window_pos: tuple = (100, 100),
        fixed_width: int = 800,
    ):
        super().__init__()
        self.photom_controls = photom_controls
        self.window_name = name
        self.aspect_ratio = sensor_size_yx[1] / sensor_size_yx[0]
        self.fixed_width = fixed_width
        calculated_height = int(self.fixed_width / self.aspect_ratio)
        self.window_pos = tuple(window_pos)
        self.window_geometry = self.window_pos + (self.fixed_width, calculated_height)
        self.setMouseTracking(True)
        self.setWindowOpacity(self.photom_controls._laser_window_transparency)
        self.game_mode = False
        self.stacked_widget = QStackedWidget()

        # FLAGS (set before the window shows and starts filtering events)
        self._right_click_hold = False
        self._left_click_hold = False
        self._follow_pending = False  # a deferred marker -> mirror update is queued
        self._out_of_range_warned = False

        self.lastPoint = None
        self.drawing = False
        self.drawingTraces = []
        self.currentTrace = []
        self.shapes = {}
        self.curr_shape_points = []
        self.curr_shape_id = 0
        self.selected_shape_id = None

        self.initialize_UI()
        self.initMarker()
        self.initDrawingScene()

        tetragon_coords = calculate_rectangle_corners(
            [self.window_geometry[-2] / 5, self.window_geometry[-1] / 5],
            center=[self.window_geometry[-2] / 2, self.window_geometry[-1] / 2],
        )
        self.init_tetragon(tetragon_coords=tetragon_coords)

        self.setCentralWidget(self.stacked_widget)
        self.switch_to_shooting_scene()
        self.show()

    def initialize_UI(self):
        logger.debug(f"window geometry: {self.window_geometry}")
        self.setWindowTitle(self.window_name)
        self.setFixedSize(self.window_geometry[2], self.window_geometry[3])

    def update_window_geometry(self, new_width, new_height):
        self.window_geometry = self.window_pos + (new_width, new_height)
        self.shooting_view.setFixedSize(new_width, new_height)
        self.shooting_scene.setSceneRect(0, 0, new_width, new_height)
        self.setFixedSize(new_width, new_height)

        self.drawing_view.setFixedSize(new_width, new_height)
        self.drawing_scene.setSceneRect(0, 0, new_width, new_height)
        self.drawablePixmap = QPixmap(int(new_width), int(new_height))
        self.drawablePixmap.fill(Qt.GlobalColor.transparent)
        self.drawablePixmapItem.setPixmap(self.drawablePixmap)

    def recenter_marker(self):
        self.display_marker_center(
            self.marker,
            (self.window_geometry[-2] / 2, self.window_geometry[-1] / 2),
        )

    def initMarker(self):
        self.shooting_scene = QGraphicsScene(self)
        self.shooting_scene.setSceneRect(
            0, 0, self.window_geometry[-2], self.window_geometry[-1]
        )

        self.shooting_view = QGraphicsView(self.shooting_scene)
        self.shooting_view.setMouseTracking(True)
        self.setCentralWidget(self.shooting_view)
        self.shooting_view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.shooting_view.setFixedSize(self.window_geometry[-2], self.window_geometry[-1])

        self.shooting_view.installEventFilter(self)
        self.shooting_view.viewport().installEventFilter(self)
        self.setMouseTracking(True)

        pixmap = QPixmap(str(MARKER_IMAGE))
        if pixmap.isNull():
            raise FileNotFoundError(f"Laser marker image not found: {MARKER_IMAGE}")
        pixmap = pixmap.scaled(
            40,
            40,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.marker = ClickablePixmapItem(pixmap)
        self.marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.display_marker_center(
            self.marker, (self.window_geometry[-2] / 2, self.window_geometry[-1] / 2)
        )
        self.shooting_scene.addItem(self.marker)

        # Dashed guide rectangle, 2/3 of the window
        rect_width = 2 * self.window_geometry[-2] / 3
        rect_height = 2 * self.window_geometry[-1] / 3
        rect_x = (self.window_geometry[-2] - rect_width) / 2
        rect_y = (self.window_geometry[-1] - rect_height) / 2
        pen = QPen(QColor(0, 0, 0))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)

        self.dashed_rectangle = QGraphicsRectItem(rect_x, rect_y, rect_width, rect_height)
        self.dashed_rectangle.setPen(pen)
        self.dashed_rectangle.setBrush(QBrush(Qt.GlobalColor.transparent))
        self.shooting_scene.addItem(self.dashed_rectangle)

        self.stacked_widget.addWidget(self.shooting_view)

        self.shooting_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.shooting_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def init_tetragon(
        self, tetragon_coords: list = ((100, 100), (200, 100), (200, 200), (100, 200))
    ):
        self.calibration_scene = QGraphicsScene(self)
        self.calibration_scene.setSceneRect(
            0, 0, self.window_geometry[-2], self.window_geometry[-1]
        )

        self.calibration_view = QGraphicsView(self.calibration_scene)
        self.calibration_view.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.calibration_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.calibration_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.vertices = []
        for x, y in tetragon_coords:
            vertex = QGraphicsEllipseItem(0, 0, 10, 10)
            vertex.setBrush(Qt.GlobalColor.red)
            vertex.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            vertex.setPos(x, y)
            self.vertices.append(vertex)
            self.calibration_scene.addItem(vertex)

        self.calibration_view.installEventFilter(self)
        self.setMouseTracking(True)
        self.stacked_widget.addWidget(self.calibration_view)

    def initDrawingScene(self):
        self.drawing_scene = QGraphicsScene()
        self.drawing_scene.setSceneRect(
            0, 0, self.window_geometry[-2], self.window_geometry[-1]
        )

        self.drawing_view = QGraphicsView(self.drawing_scene)
        self.drawing_view.setFixedSize(self.window_geometry[-2], self.window_geometry[-1])
        self.drawing_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.drawing_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.drawing_view.setMouseTracking(True)
        self.drawing_view.installEventFilter(self)
        self.drawing_view.viewport().installEventFilter(self)

        # A transparent pixmap the size of the scene, drawn into by paintEvent
        self.drawablePixmapItem = QGraphicsPixmapItem()
        self.drawing_scene.addItem(self.drawablePixmapItem)
        self.drawablePixmap = QPixmap(self.window_geometry[-2], self.window_geometry[-1])
        self.drawablePixmap.fill(Qt.GlobalColor.transparent)
        self.drawablePixmapItem.setPixmap(self.drawablePixmap)

        self.stacked_widget.addWidget(self.drawing_view)

    def switchToDrawingScene(self):
        self.stacked_widget.setCurrentWidget(self.drawing_view)

    def switch_to_shooting_scene(self):
        self.stacked_widget.setCurrentWidget(self.shooting_view)

    def switch_to_calibration_scene(self):
        self.stacked_widget.setCurrentWidget(self.calibration_view)

    def toggleDrawingScene(self):
        if self.stacked_widget.currentWidget() == self.drawing_view:
            self.switch_to_shooting_scene()
        else:
            self.switchToDrawingScene()

    def clearDrawing(self):
        self.drawablePixmap.fill(Qt.GlobalColor.transparent)
        self.drawablePixmapItem.setPixmap(self.drawablePixmap)
        self.drawingTraces = []
        self.currentTrace = []

    def get_coordinates(self):
        return [vertex.pos() for vertex in self.vertices]

    def create_tetragon(self, tetragon_coords):
        self.vertices = []
        for x, y in tetragon_coords:
            vertex = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
            vertex.setBrush(Qt.GlobalColor.red)
            vertex.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.vertices.append(vertex)
            vertex.setVisible(True)
            self.calibration_scene.addItem(vertex)

    def update_vertices(self, new_coordinates):
        if len(self.vertices) != len(new_coordinates):
            logger.error("Mismatch in the number of vertices and new coordinates")
            return
        for vertex, (x, y) in zip(self.vertices, new_coordinates, strict=True):
            vertex.setPos(x, y)

    def eventFilter(self, source, event):
        """Mouse handling for the shooting and drawing views."""
        etype = event.type()
        if etype in _TRACED_MOUSE_EVENTS and logger.isEnabledFor(logging.DEBUG):
            self._trace_mouse_event(source, event)
        if etype == QEvent.Type.MouseButtonPress:
            if event.buttons() == Qt.MouseButton.LeftButton:
                self._left_click_hold = True
                if self.stacked_widget.currentWidget() == self.drawing_view:
                    self.drawing = True
                    self.lastPoint = _event_pos(event)
                    self.curr_shape_points.append(_event_pos(event))
                    self.update()
                elif self.stacked_widget.currentWidget() == self.shooting_view:
                    self._out_of_range_warned = False  # warn once per drag
                    if source is self.shooting_view.viewport():
                        self._grab_marker_at(_event_pos(event))
                    self._follow_marker()

            elif event.buttons() == Qt.MouseButton.RightButton:
                self._right_click_hold = True
                self.photom_controls.photom_assembly.laser[0].toggle_emission = True

        elif etype == QEvent.Type.MouseMove:
            if self._left_click_hold:
                if source == self.shooting_view.viewport():
                    self._follow_marker()
                elif self.drawing and source == self.drawing_view.viewport():
                    self.curr_shape_points.append(_event_pos(event))
                    self.update()

        elif etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self._left_click_hold = False

                if self.drawing:
                    self.curr_shape_points.append(_event_pos(event))
                    # close the shape
                    self.curr_shape_points.append(self.curr_shape_points[0])
                    self.shapes[self.curr_shape_id] = ShapeTrace(self.curr_shape_points)
                    self.shapesUpdated.emit()

                    self.curr_shape_points = []
                    self.drawing = False
                    self.curr_shape_id += 1
                    self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                self._right_click_hold = False
                self.photom_controls.photom_assembly.laser[0].toggle_emission = False
                if self.game_mode:
                    self._game_mode_marker(event)
                time.sleep(0.3)

        return super().eventFilter(source, event)

    def resizeEvent(self, a0) -> None:
        # Triggered after manual resizing or setGeometry()
        super().resizeEvent(a0)
        rect = self.shooting_view.sceneRect()
        self.shooting_scene.setSceneRect(0, 0, rect.width(), rect.height())
        self.drawing_scene.setSceneRect(0, 0, rect.width(), rect.height())
        self.drawing_view.setSceneRect(0, 0, rect.width(), rect.height())
        self._update_scene_items(rect.width(), rect.height())

    def remove_score_text(self, text_item):
        self.shooting_scene.removeItem(text_item)

    def _game_mode_marker(self, event):
        # Show "+100" at the click position for one second
        score_text = QGraphicsSimpleTextItem("+100")
        score_text.setBrush(QColor(255, 255, 0))
        font = QFont()
        font.setPointSize(30)
        score_text.setFont(font)
        pos = _event_pos(event)
        score_text.setPos(pos.x() + 15, pos.y() - 70)
        self.shooting_scene.addItem(score_text)
        QTimer.singleShot(1000, lambda: self.remove_score_text(score_text))

    def _update_scene_items(self, new_width, new_height):
        rect_width = new_width * 2 / 3
        rect_height = new_height * 2 / 3
        rect_x = (new_width - rect_width) / 2
        rect_y = (new_height - rect_height) / 2

        # Keep the marker on the laser spot (upstream recentred it, leaving the laser
        # elsewhere). Before the control panel is built there is no mirror to follow.
        controls = self.photom_controls
        if controls is not None and getattr(controls, "mirror_widgets", None):
            # quiet: mid-resize the window affine may not be rescaled yet
            controls.sync_marker_to_mirror(warn=False)
        else:
            self.recenter_marker()

        self.dashed_rectangle.setRect(rect_x, rect_y, rect_width, rect_height)
        self.dashed_rectangle.setPen(QPen(QColor(0, 0, 0), 2, Qt.PenStyle.DashLine))
        self.shooting_view.update()

    def _trace_mouse_event(self, source, event) -> None:
        """Debug log of a mouse event: what received it, buttons, position, marker."""
        names = {
            id(self.shooting_view): "shooting_view",
            id(self.shooting_view.viewport()): "shooting_viewport",
            id(self.drawing_view): "drawing_view",
            id(self.drawing_view.viewport()): "drawing_viewport",
            id(self.calibration_view): "calibration_view",
        }
        pos = _event_pos(event)
        marker = self.get_marker_center(self.marker)
        # every item under the cursor, topmost first; the marker must be among them
        # for a press to grab it (the dashed guide rectangle above it ignores clicks)
        under = [
            "marker" if item is self.marker else type(item).__name__
            for item in self.shooting_view.items(pos)
        ]
        logger.debug(
            f"mouse {event.type().name} on {names.get(id(source), type(source).__name__)}: "
            f"pos=({pos.x()}, {pos.y()}) button={event.button().name} "
            f"buttons={event.buttons()} hold={self._left_click_hold} "
            f"scene={type(self.stacked_widget.currentWidget()).__name__}"
            f"{'(shooting)' if self.stacked_widget.currentWidget() is self.shooting_view else ''} "
            f"marker=({marker[0]:.0f}, {marker[1]:.0f}) "
            f"items_under_cursor={under}"
        )

    def _grab_marker_at(self, viewport_pos: QPoint) -> None:
        """Bring the marker under a press that missed it, so the press drags it.

        Only a press *on* the 40 px marker used to drag it; a press anywhere else
        steered the mirror to wherever the marker already was and then ignored the
        drag. The event filter runs before the view handles the press, so centring
        the marker on the cursor here makes the view grab it like a direct hit.
        """
        if self.marker in self.shooting_view.items(viewport_pos):
            return
        scene_pos = self.shooting_view.mapToScene(viewport_pos)
        self.display_marker_center(self.marker, (scene_pos.x(), scene_pos.y()))

    def _follow_marker(self):
        """Steer the mirror to the marker once the view has finished moving it.

        The event filter sees a mouse event *before* the view hands it to the marker
        item, so steering immediately aimed at the marker's previous position: the
        laser lagged one event behind the drag (in coPylot too). Defer to the next
        event-loop turn, and coalesce: if mirror writes are slower than mouse moves,
        only the latest marker position is sent.
        """
        if not self._follow_pending:
            self._follow_pending = True
            QTimer.singleShot(0, self._move_marker_and_update_sliders)

    def _move_marker_and_update_sliders(self):
        """Steer the current mirror to the marker, via the mirror widget's sliders."""
        self._follow_pending = False
        if self.photom_controls is not None:
            marker_position = self.get_marker_center(
                self.marker, coords=(self.marker.pos().x(), self.marker.pos().y())
            )
            mirror_widget = self.photom_controls.mirror_widgets[
                self.photom_controls._current_mirror_idx
            ]
            new_coords = mirror_widget.mirror.affine_transform_obj.apply_affine(
                marker_position
            )
            target = (new_coords[0][0], new_coords[1][0])
            logger.debug(
                f"steer mirror {mirror_widget.mirror.name}: marker=({marker_position[0]:.0f}, "
                f"{marker_position[1]:.0f}) -> target=({target[0]:.5f}, {target[1]:.5f})"
            )
            self._warn_if_out_of_range(mirror_widget, target)
            mirror_widget.mirror_x_slider.setValue(target[0])
            mirror_widget.mirror_y_slider.setValue(target[1])

    def _warn_if_out_of_range(self, mirror_widget, target) -> None:
        """Say so when the target is clamped to the mirror's range (once per drag).

        The sliders silently clamp to the mirror's movement limits, so an
        out-of-range target pins the laser at the edge however far the marker is
        dragged. Without a calibration this is every target: the window maps to
        sensor pixels (thousands), while the Optotune mirror accepts +/-1.
        """
        (x_lo, x_hi), (y_lo, y_hi) = (
            mirror_widget.movement_limits_x,
            mirror_widget.movement_limits_y,
        )
        in_range = x_lo <= target[0] <= x_hi and y_lo <= target[1] <= y_hi
        if in_range or self._out_of_range_warned:
            return
        self._out_of_range_warned = True
        controls = self.photom_controls
        uncalibrated = controls.calibration_sources[controls._current_mirror_idx] is None and (
            np.allclose(controls.T_mirror_cam_matrix, np.eye(3))
        )
        hint = (
            "the mirror is uncalibrated; load a calibration (Load Calibration, or "
            "--calibration)"
            if uncalibrated
            else "aim inside the calibrated area"
        )
        message = (
            f"Marker maps to mirror ({target[0]:.4g}, {target[1]:.4g}), outside the "
            f"mirror's range x[{x_lo}, {x_hi}] y[{y_lo}, {y_hi}]; the laser is pinned "
            f"at the edge: {hint}."
        )
        logger.warning(message)
        controls.statusBar().showMessage(message, 10000)

    def _roi_tracing(self, pattern_delay: float = 1.0):
        """Step the mirror through the selected shape's ablation points."""
        delay_secs = self.photom_controls.delay_input.text()
        pattern_delay = float(delay_secs) if delay_secs else 1.0
        logger.info(f"ROI tracing with {pattern_delay} s between points")
        if self.selected_shape_id is not None:
            shape = self.shapes[self.selected_shape_id]
            mirror_widget = self.photom_controls.mirror_widgets[
                self.photom_controls._current_mirror_idx
            ]
            new_ablation_points = []
            for position in shape.ablation_points:
                new_coords = mirror_widget.mirror.affine_transform_obj.apply_affine(position)
                mirror_widget.mirror_x_slider.setValue(new_coords[0][0])
                mirror_widget.mirror_y_slider.setValue(new_coords[1][0])
                time.sleep(pattern_delay)
                logger.debug(f"time: {datetime.now()}")
                new_ablation_points.append(new_coords)
            return new_ablation_points

    def get_marker_center(self, marker, coords=None):
        if coords is None:
            coords = (marker.x(), marker.y())
        center_x = coords[0] + marker.pixmap().width() / 2
        center_y = coords[1] + marker.pixmap().height() / 2
        return [center_x, center_y]

    def display_marker_center(self, marker, coords=None):
        if coords is None:
            coords = (marker.x(), marker.y())
        center_x = coords[0] - marker.pixmap().width() / 2
        center_y = coords[1] - marker.pixmap().height() / 2
        marker.setPos(center_x, center_y)

    def closeEvent(self, event):
        self.windowClosed.emit()
        super().closeEvent(event)

    def draw_shapes(self) -> None:
        """Draw the shapes (the selected one in red) into the drawing pixmap."""
        self.drawablePixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.drawablePixmap)
        for shape_id, shape in self.shapes.items():
            color = (
                Qt.GlobalColor.red
                if shape_id == self.selected_shape_id
                else Qt.GlobalColor.black
            )
            painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
            border_points = shape.border_points
            for i in range(len(border_points) - 1):
                painter.drawLine(border_points[i], border_points[i + 1])

        if self.drawing and len(self.curr_shape_points) > 1:
            painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
            for i in range(len(self.curr_shape_points) - 1):
                painter.drawLine(self.curr_shape_points[i], self.curr_shape_points[i + 1])
        painter.end()
        self.drawablePixmapItem.setPixmap(self.drawablePixmap)

    def draw_patterns(self) -> None:
        """Draw ablation points (green) and pattern paths (gray)."""
        painter = QPainter(self.drawablePixmap)
        brush = QBrush(Qt.GlobalColor.green, Qt.BrushStyle.SolidPattern)
        point_size = 5

        for shape in self.shapes.values():
            for point in shape.ablation_points:
                painter.setBrush(brush)
                painter.drawEllipse(
                    QPoint(int(point[0]), int(point[1])), point_size, point_size
                )
            if shape.pattern_points:
                painter.setPen(QPen(Qt.GlobalColor.gray, 2, Qt.PenStyle.SolidLine))
                for point in shape.pattern_points:
                    painter.drawPoint(QPoint(int(point[0]), int(point[1])))
        painter.end()
        self.drawablePixmapItem.setPixmap(self.drawablePixmap)

    def paintEvent(self, event):
        self.draw_shapes()
        self.draw_patterns()


class ImageWindow(QMainWindow):
    """Shows the calibration plot."""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Overlay of laser grid and MIP points")

        self.label = QLabel(self)
        self.pixmap = QPixmap(str(image_path))
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
        self.resize(self.pixmap.width() + 20, self.pixmap.height() + 20)
        self.center()

    def center(self):
        frame = self.frameGeometry()
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())
