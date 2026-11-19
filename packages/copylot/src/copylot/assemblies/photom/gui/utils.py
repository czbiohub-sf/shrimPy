"""Worker threads and small Qt helpers for the photom GUI.

Vendored from coPylot ``copylot/assemblies/photom/gui/utils.py`` (main @ 0bcc3a3),
BSD-3-Clause; see the package LICENSE. Ported from PyQt5 to qtpy.

Changes from upstream:

- ``CalibrationWithCameraThread`` takes the mirror ROI, grid size and output
  directory instead of hardcoding them (upstream wrote to
  ``C:\\Users\\ZebraPhysics\\Documents\\tmp\\test_calib``), reads the mirror index
  when it starts (upstream fixed it at construction, so calibrating after
  switching mirrors calibrated mirror 0), and emits ``failed`` instead of dying
  silently when calibration raises
- ``DoubleSlider.setSingleStep`` passes an ``int`` (PyQt6 rejects floats)
"""

from __future__ import annotations

import logging
import time

from pathlib import Path

import numpy as np

from qtpy.QtCore import QRectF, QThread, Signal
from qtpy.QtGui import QPainterPath
from qtpy.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QSlider

logger = logging.getLogger(__name__)


class PWMWorker(QThread):
    finished = Signal()
    progress = Signal(int)  # percent complete
    _stop_requested = False

    def __init__(self, arduino_pwm, command, repetitions, time_interval_s, duration):
        super().__init__()
        self.arduino_pwm = arduino_pwm
        self.command = command
        self.repetitions = repetitions
        self.time_interval_s = time_interval_s
        self.duration = duration

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        for i in range(self.repetitions):
            if self._stop_requested:
                break
            self.arduino_pwm.send_command(self.command)
            # The Arduino does not report completion; wait out the pulse train.
            time.sleep(self.duration / 1000)
            self.progress.emit(int((i + 1) / self.repetitions * 100))
            time.sleep(self.time_interval_s)

        self.finished.emit()


class CalibrationWithCameraThread(QThread):
    """Runs ``PhotomAssembly.calibrate_w_camera`` off the GUI thread.

    Emits ``finished(T_affine, plot_path)`` on success and ``failed(message)``
    if calibration raises.
    """

    finished = Signal(np.ndarray, str)
    failed = Signal(str)

    def __init__(
        self,
        photom_assembly,
        current_mirror_idx: int = 0,
        mirror_roi=((-0.01, -0.03), (0.015, 0.0)),
        grid_n_points: int = 5,
        save_dir: Path | None = None,
    ):
        super().__init__()
        self.photom_assembly = photom_assembly
        self.current_mirror_idx = current_mirror_idx
        self.mirror_roi = [list(p) for p in mirror_roi]  # [[x0, y0], [x1, y1]]
        self.grid_n_points = grid_n_points
        self.save_dir = Path(save_dir) if save_dir is not None else Path.cwd()

    def run(self):
        try:
            T_mirror_cam_matrix, plot_save_path = self.photom_assembly.calibrate_w_camera(
                mirror_index=self.current_mirror_idx,
                camera_index=0,
                rectangle_boundaries=self.mirror_roi,
                grid_n_points=self.grid_n_points,
                save_calib_stack_path=self.save_dir,
                verbose=True,
            )
        except Exception as e:
            logger.exception("Calibration with camera failed")
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished.emit(T_mirror_cam_matrix, str(plot_save_path))


class DoubleSlider(QSlider):
    """A QSlider with float values, at ``decimals`` decimal places."""

    doubleValueChanged = Signal(float)

    def __init__(self, decimals=5, *args, **kargs):
        super().__init__(*args, **kargs)
        self._multi = 10**decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        self.doubleValueChanged.emit(float(super().value()) / self._multi)

    def value(self):
        return float(super().value()) / self._multi

    def setMinimum(self, value):
        return super().setMinimum(int(value * self._multi))

    def setMaximum(self, value):
        return super().setMaximum(int(value * self._multi))

    def setSingleStep(self, value):
        return super().setSingleStep(int(value * self._multi))

    def singleStep(self):
        return float(super().singleStep()) / self._multi

    def setValue(self, value):
        super().setValue(int(value * self._multi))


class ClickablePixmapItem(QGraphicsPixmapItem):
    """A movable pixmap whose whole bounding box is clickable/draggable."""

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)

    def boundingRect(self):
        return QRectF(self.pixmap().rect())

    def shape(self):
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path
