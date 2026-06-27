"""Reusable Qt controls for deskew widgets.

A small, behavior-free panel -- geometry spin boxes (angle / pixel size / scan step),
"Display deskewed" / "Display raw" buttons, and a status line -- that emits signals. The
embedding widget connects the signals to its own apply logic, so the UI and the geometry
inputs are defined once and shared by the standalone offline widget and shrimpy's live
acquisition viewer.

Imports qtpy, so this module (unlike the deskew core) requires a Qt environment.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_deskew_preview.deskew import LS_ANGLE_DEG, PIXEL_SIZE_UM

# Default scan step (um); the user sets this for their acquisition.
DEFAULT_SCAN_STEP_UM = 0.15


def _make_spin(
    value: float, lo: float, hi: float, step: float, decimals: int
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setKeyboardTracking(False)  # emit only on enter / focus-out / arrows
    spin.setValue(value)
    return spin


class DeskewControls(QWidget):
    """Geometry fields + Display deskewed/raw buttons + status, with signals.

    Signals
    -------
    displayDeskewedRequested : emitted when "Display deskewed" is clicked.
    displayRawRequested : emitted when "Display raw" is clicked.
    geometryChanged : emitted when angle / pixel size / scan step changes.
    """

    displayDeskewedRequested = Signal()
    displayRawRequested = Signal()
    geometryChanged = Signal()

    def __init__(
        self, scan_step_um: float = DEFAULT_SCAN_STEP_UM, parent: object = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._angle = _make_spin(LS_ANGLE_DEG, 0.0, 89.9, 1.0, 2)
        self._pixel = _make_spin(PIXEL_SIZE_UM, 1e-4, 100.0, 0.001, 4)
        self._scan = _make_spin(scan_step_um, 1e-4, 1000.0, 0.01, 4)
        form = QFormLayout()
        form.addRow("Angle (°)", self._angle)
        form.addRow("Pixel size (µm)", self._pixel)
        form.addRow("Scan step (µm)", self._scan)
        layout.addLayout(form)

        deskew_button = QPushButton("Display deskewed")
        deskew_button.clicked.connect(lambda: self.displayDeskewedRequested.emit())
        layout.addWidget(deskew_button)

        raw_button = QPushButton("Display raw")
        raw_button.clicked.connect(lambda: self.displayRawRequested.emit())
        layout.addWidget(raw_button)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

        for spin in (self._angle, self._pixel, self._scan):
            spin.valueChanged.connect(lambda *_: self.geometryChanged.emit())

    @property
    def angle(self) -> float:
        return float(self._angle.value())

    @property
    def pixel_size(self) -> float:
        return float(self._pixel.value())

    @property
    def scan_step(self) -> float:
        return float(self._scan.value())

    def set_status(self, text: str) -> None:
        self._status.setText(text)
