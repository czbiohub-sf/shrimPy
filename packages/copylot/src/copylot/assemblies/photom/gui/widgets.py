"""Laser, mirror and Arduino-PWM control widgets for the photom GUI.

Vendored from coPylot ``copylot/assemblies/photom/gui/widgets.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE. Ported from PyQt5 to
qtpy.

Changes from upstream: logging instead of ``print``; ``ArduinoPWMWidget``'s
pulse-duration field no longer raises ``AttributeError`` (``self.time_period``)
when given invalid text.
"""

from __future__ import annotations

import logging
import time

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from copylot.assemblies.photom.gui.utils import DoubleSlider, PWMWorker

logger = logging.getLogger(__name__)

_OFF_STYLE = "background-color: magenta"
_ON_STYLE = "background-color: green"


class LaserWidget(QWidget):
    def __init__(self, laser):
        super().__init__()
        self.laser = laser

        self.emission_state = 0  # 0 = off, 1 = on
        self.emission_delay = 0  # 0 = off, 1 = 5 s delay

        self._curr_power = 0
        self._slider_decimal = 1
        self._curr_laser_pulse_mode = False

        self.initializer_laser()
        self.initialize_UI()

    def initializer_laser(self):
        # Start with emission off and zero power
        self.laser.toggle_emission = 0
        self.laser.emission_delay = self.emission_delay
        self.laser.pulse_power = self._curr_power
        self.laser.power = self._curr_power

        # Make sure the laser is in continuous mode
        if self.laser.pulse_mode == 1:
            self.laser.toggle_emission = 1
            time.sleep(0.2)
            self.laser.pulse_mode = self._curr_laser_pulse_mode ^ 1
            time.sleep(0.2)
            self.laser.toggle_emission = 0

    def initialize_UI(self):
        layout = QVBoxLayout()

        self.laser_label = QLabel(self.laser.name)
        layout.addWidget(self.laser_label)

        self.laser_power_slider = DoubleSlider(
            orientation=Qt.Orientation.Horizontal, decimals=self._slider_decimal
        )
        self.laser_power_slider.setMinimum(0)
        self.laser_power_slider.setMaximum(100)
        self.laser_power_slider.setValue(self.laser.power)
        self.laser_power_slider.sliderReleased.connect(self.on_slider_released)
        self.laser_power_slider.valueChanged.connect(self.update_displayed_power)
        layout.addWidget(self.laser_power_slider)

        self.power_edit = QLineEdit(f"{self._curr_power:.2f}")
        self.power_edit.returnPressed.connect(self.edit_power)
        layout.addWidget(self.power_edit)

        self.pulse_mode_button = QPushButton("Pulse Mode")
        self.pulse_mode_button.clicked.connect(self.laser_pulse_mode)
        layout.addWidget(self.pulse_mode_button)
        self.pulse_mode_button.setStyleSheet(_OFF_STYLE)

        self.laser_toggle_button = QPushButton("Toggle")
        self.laser_toggle_button.clicked.connect(self.toggle_laser)
        if self.emission_state == 0:
            self.laser_toggle_button.setStyleSheet(_OFF_STYLE)
        layout.addWidget(self.laser_toggle_button)

        self.setLayout(layout)

    def toggle_laser(self):
        self.emission_state = self.emission_state ^ 1
        self.laser.toggle_emission = self.emission_state
        self.laser_toggle_button.setStyleSheet(
            _OFF_STYLE if self.emission_state == 0 else _ON_STYLE
        )

    def update_power(self, value):
        if self._curr_laser_pulse_mode:
            self.laser.pulse_power = self._curr_power
        else:
            self.laser.power = self._curr_power
        self.power_edit.setText(f"{self._curr_power:.2f}")

    def on_slider_released(self):
        self._curr_power = self.laser_power_slider.value()
        self.update_power(self._curr_power)

    def update_displayed_power(self, value):
        # Show the value while dragging; the laser is updated on release.
        self._curr_power = value / (10**self._slider_decimal)
        self.power_edit.setText(f"{self._curr_power:.2f}")

    def edit_power(self):
        try:
            power_value = float(self.power_edit.text())
            if 0 <= power_value <= 100:  # percent
                self._curr_power = power_value
                self.update_power(self._curr_power)
            else:
                self.power_edit.setText(f"{self._curr_power:.2f}")
            logger.info(f"{self.laser.name} power: {self._curr_power}")
        except ValueError:
            self.power_edit.setText(f"{self._curr_power:.2f}")

    def laser_pulse_mode(self):
        self._curr_laser_pulse_mode = not self._curr_laser_pulse_mode
        self.laser.toggle_emission = 1
        if self._curr_laser_pulse_mode:
            self.pulse_mode_button.setStyleSheet(_ON_STYLE)
            self.laser.pulse_power = self._curr_power
            self.laser.pulse_mode = 1
            time.sleep(0.2)
        else:
            self.laser.power = self._curr_power
            self.laser.pulse_mode = 0
            self.pulse_mode_button.setStyleSheet(_OFF_STYLE)
            time.sleep(0.2)
            self.laser_toggle_button.setStyleSheet(_OFF_STYLE)
            self.laser.toggle_emission = 0
            self.emission_state = 0

        logger.info(
            f"{self.laser.name} pulse mode: {self._curr_laser_pulse_mode}, "
            f"digital modulation: {self.laser.pulse_mode}"
        )


class MirrorWidget(QWidget):
    def __init__(self, mirror):
        super().__init__()
        self.mirror = mirror

        self.check_mirror_limits()
        self.initialize_UI()

    def initialize_UI(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Mirror X Position"))
        self.mirror_x_slider = DoubleSlider(orientation=Qt.Orientation.Horizontal)
        self.mirror_x_slider.setMinimum(self.movement_limits_x[0])
        self.mirror_x_slider.setMaximum(self.movement_limits_x[1])
        self.mirror_x_slider.doubleValueChanged.connect(self.update_mirror_x)
        layout.addWidget(self.mirror_x_slider)

        self.mirror_x_label = QLabel(f"X: {self.mirror.position_x}")
        layout.addWidget(self.mirror_x_label)

        layout.addWidget(QLabel("Mirror Y Position"))
        self.mirror_y_slider = DoubleSlider(orientation=Qt.Orientation.Horizontal)
        self.mirror_y_slider.setMinimum(self.movement_limits_y[0])
        self.mirror_y_slider.setMaximum(self.movement_limits_y[1])
        self.mirror_y_slider.doubleValueChanged.connect(self.update_mirror_y)
        layout.addWidget(self.mirror_y_slider)

        self.mirror_y_label = QLabel(f"Y: {self.mirror.position_y}")
        layout.addWidget(self.mirror_y_label)

        self.setLayout(layout)

    def update_mirror_x(self, value):
        self.mirror.position_x = value
        self.mirror_x_label.setText(f"X: {value}")

    def update_mirror_y(self, value):
        self.mirror.position_y = value
        self.mirror_y_label.setText(f"Y: {value}")

    def check_mirror_limits(self):
        movement_limits = self.mirror.movement_limits
        self.movement_limits_x = movement_limits[0:2]
        self.movement_limits_y = movement_limits[2:4]


class ArduinoPWMWidget(QWidget):
    def __init__(self, photom_assembly, arduino_pwm, parent):
        super().__init__()
        self.parent = parent
        self.arduino_pwm = arduino_pwm
        self.photom_assembly = photom_assembly
        # defaults
        self.duty_cycle = 50  # [%] (0-100)
        self.time_period_ms = 100  # [ms]
        self.frequency = 1000.0 / self.time_period_ms  # [Hz]
        self.duration = 1  # number of reps
        self.repetitions = 1  # runs once by default
        self.time_interval_s = 0  # [s]

        self.command = f"U,{self.duty_cycle},{self.frequency},{self.duration}"

        self._curr_laser_idx = 0

        self.initialize_UI()

    def initialize_UI(self):
        layout = QGridLayout()

        self.laser_dropdown = QComboBox()
        for laser in self.photom_assembly.laser:
            self.laser_dropdown.addItem(laser.name)
        layout.addWidget(QLabel("Select Laser:"), 0, 0)
        layout.addWidget(self.laser_dropdown, 0, 1)
        self.laser_dropdown.setCurrentIndex(self._curr_laser_idx)
        self.laser_dropdown.currentIndexChanged.connect(self.current_laser_changed)

        layout.addWidget(QLabel("Duty Cycle [%]:"), 1, 0)
        self.duty_cycle_edit = QLineEdit(f"{self.duty_cycle}")
        self.duty_cycle_edit.returnPressed.connect(self.edit_duty_cycle)
        layout.addWidget(self.duty_cycle_edit, 1, 1)

        layout.addWidget(QLabel("Pulse Duration [ms]:"), 2, 0)
        self.time_period_edit = QLineEdit(f"{self.time_period_ms}")
        self.time_period_edit.returnPressed.connect(self.edit_time_period)
        layout.addWidget(self.time_period_edit, 2, 1)

        layout.addWidget(QLabel("Pulse repetitions:"), 3, 0)
        self.duration_edit = QLineEdit(f"{self.duration}")
        self.duration_edit.returnPressed.connect(self.edit_duration)
        layout.addWidget(self.duration_edit, 3, 1)

        layout.addWidget(QLabel("Timelapse - number of timepoints:"), 4, 0)
        self.repetitions_edit = QLineEdit(f"{self.repetitions}")
        self.repetitions_edit.textChanged.connect(self.edit_repetitions)
        layout.addWidget(self.repetitions_edit, 4, 1)

        layout.addWidget(QLabel("Timelapse - Time interval [s]:"), 5, 0)
        self.time_interval_edit = QLineEdit(f"{self.time_interval_s}")
        self.time_interval_edit.textChanged.connect(self.edit_time_interval)
        layout.addWidget(self.time_interval_edit, 5, 1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button, 6, 0, 1, 2)

        self.start_button = QPushButton("Start PWM")
        self.start_button.clicked.connect(self.start_pwm)
        layout.addWidget(self.start_button, 7, 0, 1, 2)

        self.stop_button = QPushButton("Stop PWM")
        self.stop_button.clicked.connect(self.stop_pwm)
        layout.addWidget(self.stop_button, 8, 0, 1, 2)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)
        layout.addWidget(self.progressBar, 9, 0, 1, 2)

        self.setLayout(layout)

    def edit_duty_cycle(self):
        try:
            self.duty_cycle = float(self.duty_cycle_edit.text())
            self.update_command()
        except ValueError:
            self.duty_cycle_edit.setText(f"{self.duty_cycle}")

    def edit_time_period(self):
        try:
            self.time_period_ms = float(self.time_period_edit.text())
            self.frequency = 1000.0 / self.time_period_ms
            self.update_command()
        except (ValueError, ZeroDivisionError):
            self.time_period_edit.setText(f"{self.time_period_ms}")

    def edit_duration(self):
        try:
            self.duration = self.time_period_ms * float(self.duration_edit.text())
            self.update_command()
        except ValueError:
            self.duration_edit.setText(f"{self.duration}")

    def edit_repetitions(self):
        try:
            self.repetitions = int(self.repetitions_edit.text())
        except ValueError:
            self.repetitions_edit.setText(f"{self.repetitions}")

    def edit_time_interval(self):
        try:
            self.time_interval_s = float(self.time_interval_edit.text())
        except ValueError:
            self.time_interval_edit.setText(f"{self.time_interval_s}")

    def get_current_parameters_from_gui(self):
        self.duty_cycle = float(self.duty_cycle_edit.text())
        self.time_period_ms = float(self.time_period_edit.text())
        self.frequency = 1000.0 / self.time_period_ms
        self.duration = float(self.duration_edit.text()) * self.time_period_ms
        self.repetitions = int(self.repetitions_edit.text())
        self.time_interval_s = float(self.time_interval_edit.text())

    def update_command(self):
        self.command = f"U,{self.duty_cycle},{self.frequency},{self.duration}"
        self.arduino_pwm.send_command(self.command)

    def start_pwm(self):
        laser_widget = self.parent.laser_widgets[self._curr_laser_idx]
        if not laser_widget._curr_laser_pulse_mode:
            logger.info("Setting up laser in pulse mode...")
            laser_widget.laser_pulse_mode()

        logger.info("Starting PWM...")
        self.pwm_worker = PWMWorker(
            self.arduino_pwm,
            "S",
            self.repetitions,
            self.time_interval_s,
            self.duration,
        )
        self.progressBar.setValue(0)
        self.pwm_worker.finished.connect(self.on_pwm_finished)
        self.pwm_worker.progress.connect(self.update_progress_bar)
        self.pwm_worker.start()

    def update_progress_bar(self, value):
        self.progressBar.setValue(value)

    def stop_pwm(self):
        if hasattr(self, "pwm_worker") and self.pwm_worker.isRunning():
            self.pwm_worker.request_stop()

    def on_pwm_finished(self):
        logger.info("PWM operation completed.")

    def current_laser_changed(self, index):
        self._curr_laser_idx = index
        self.apply_settings()

    def apply_settings(self):
        self._curr_laser_idx = self.laser_dropdown.currentIndex()
        self.get_current_parameters_from_gui()
        # update_command() already sends the command to the Arduino. Upstream then
        # sent it a second time here; kept so the Arduino sees the same traffic.
        self.update_command()
        self.arduino_pwm.send_command(self.command)
