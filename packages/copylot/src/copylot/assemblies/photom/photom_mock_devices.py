"""Simulated lasers, mirror, Arduino and camera for running photom without hardware.

Vendored from coPylot ``copylot/assemblies/photom/photom_mock_devices.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE.

Changes from upstream:

- logging instead of ``print`` (mirror moves log at debug level)
- ``MockLaser`` gains ``emission_delay`` and ``pulse_power``, which the GUI's
  laser widget sets; ``MockLaser.power`` clamps out-of-range values instead of
  ignoring them
- ``MockFlirCamera`` (random noise, a 2-tuple ``image_size`` that crashed
  ``calibrate_w_camera``, coupled to coPylot's FLIR camera base) is replaced by
  :class:`MockCamera`, which renders the laser spot where a mirror points, so
  camera calibration runs end to end in demo mode
"""

from __future__ import annotations

import logging
import time

from collections.abc import Sequence

import numpy as np

from copylot.hardware.lasers.abstract_laser import AbstractLaser
from copylot.hardware.mirrors.abstract_mirror import AbstractMirror

logger = logging.getLogger(__name__)


class MockLaser(AbstractLaser):
    def __init__(self, name, serial_number=None, port=None, baudrate=19200, timeout=1):
        self.name = name
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._device_id = serial_number
        self._drive_control_mode = "automatic"
        self._toggle_emission = False
        self._power = 0.0
        self._pulse_power = 0.0
        self._pulse_mode = False
        self._emission_delay = 0
        self._maximum_power = 100.0
        self._current_control_mode = "automatic"
        self._external_power_control = False
        self._status = "disconnected"

    @property
    def device_id(self):
        return self._device_id

    def connect(self):
        self._status = "connected"
        logger.info(f"Mock laser {self.name} connected")

    def disconnect(self):
        self._status = "disconnected"
        logger.info(f"Mock laser {self.name} disconnected")

    @property
    def drive_control_mode(self) -> str:
        return self._drive_control_mode

    @drive_control_mode.setter
    def drive_control_mode(self, value: str):
        self._drive_control_mode = value

    @property
    def toggle_emission(self) -> bool:
        return self._toggle_emission

    @toggle_emission.setter
    def toggle_emission(self, value: bool):
        self._toggle_emission = value
        logger.info(f"Mock laser {self.name} emission {'on' if value else 'off'}")

    @property
    def emission_delay(self) -> int:
        return self._emission_delay

    @emission_delay.setter
    def emission_delay(self, value: int):
        self._emission_delay = value

    def _clamp(self, value: float) -> float:
        return min(max(float(value), 0.0), self._maximum_power)

    @property
    def power(self) -> float:
        return self._power

    @power.setter
    def power(self, value: float):
        self._power = self._clamp(value)
        logger.info(f"Mock laser {self.name} power set to {self._power}")

    @property
    def pulse_power(self) -> float:
        return self._pulse_power

    @pulse_power.setter
    def pulse_power(self, value: float):
        self._pulse_power = self._clamp(value)

    @property
    def pulse_mode(self) -> bool:
        return self._pulse_mode

    @pulse_mode.setter
    def pulse_mode(self, value: bool):
        self._pulse_mode = value
        logger.info(f"Mock laser {self.name} pulse mode {'on' if value else 'off'}")

    @property
    def maximum_power(self) -> float:
        return self._maximum_power

    @property
    def current_control_mode(self) -> str:
        return self._current_control_mode

    @current_control_mode.setter
    def current_control_mode(self, value: str):
        self._current_control_mode = value

    @property
    def external_power_control(self) -> bool:
        return self._external_power_control

    @external_power_control.setter
    def external_power_control(self, value: bool):
        self._external_power_control = value

    @property
    def status(self) -> str:
        return self._status


class MockMirror(AbstractMirror):
    def __init__(
        self,
        name: str = "OPTOTUNE_MIRROR",
        com_port: str | None = None,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.mirror = "mirror_instance"
        self._movement_limits = [-2000.0, 2000.0, -2000.0, 2000.0]
        self._position = [0.0, 0.0]
        self._relative_position = [0.0, 0.0]
        self._step_resolution = 0.01
        self._external_drive_control = "manual"
        self.position_x = pos_x
        self.position_y = pos_y

    @property
    def device_id(self):
        return self.name

    @device_id.setter
    def device_id(self, value: str):
        self.name = value

    def _in_limits(self, value: float, axis: int) -> bool:
        lo, hi = self._movement_limits[2 * axis : 2 * axis + 2]
        return lo <= value <= hi

    @property
    def position(self) -> list[float]:
        return list(self._position)

    @position.setter
    def position(self, value: Sequence[float]):
        if self._in_limits(value[0], 0) and self._in_limits(value[1], 1):
            self._position = [float(value[0]), float(value[1])]
            logger.debug(f"Mock mirror position set to {self._position}")
        else:
            logger.warning(f"Mock mirror position {list(value)} out of bounds")

    @property
    def position_x(self) -> float:
        return self._position[0]

    @position_x.setter
    def position_x(self, value: float):
        if self._in_limits(value, 0):
            self._position[0] = float(value)
        else:
            logger.warning(f"Mock mirror position X {value} out of bounds")

    @property
    def position_y(self) -> float:
        return self._position[1]

    @position_y.setter
    def position_y(self, value: float):
        if self._in_limits(value, 1):
            self._position[1] = float(value)
        else:
            logger.warning(f"Mock mirror position Y {value} out of bounds")

    @property
    def relative_position(self) -> list[float]:
        return self._relative_position

    @relative_position.setter
    def relative_position(self, value: list[float]):
        self._relative_position = value

    @property
    def movement_limits(self) -> list[float]:
        return self._movement_limits

    @movement_limits.setter
    def movement_limits(self, value: list[float]):
        self._movement_limits = value

    @property
    def step_resolution(self) -> float:
        return self._step_resolution

    @step_resolution.setter
    def step_resolution(self, value: float):
        self._step_resolution = value

    def set_home(self):
        self._position = [0.0, 0.0]

    def set_origin(self, axis: str):
        if axis == "x":
            self._position[0] = 0.0
        elif axis == "y":
            self._position[1] = 0.0

    @property
    def external_drive_control(self) -> str:
        return self._external_drive_control

    @external_drive_control.setter
    def external_drive_control(self, value: bool):
        self._external_drive_control = value

    def voltage_to_position(self, voltage: list[float]) -> list[float]:
        return [v * 10 for v in voltage]


class MockArduinoPWM:
    def __init__(self, serial_port, baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        logger.info(f"MockArduinoPWM on {serial_port} @ {baud_rate} baud")

    def connect(self):
        pass

    def close(self):
        pass

    def send_command(self, command):
        logger.info(f"MockArduinoPWM: {command}")
        time.sleep(0.5)  # simulate the Arduino's processing time


class MockCamera:
    """Images a Gaussian laser spot wherever ``mirror`` points.

    The spot is at ``center + (mirror.position - mirror_center) * pixels_per_unit``
    in (x, y); positions off the sensor give a blank (background-only) frame.

    Parameters
    ----------
    mirror : mirror
        The mirror steering the laser.
    sensor_size_yx : (int, int)
        Frame shape.
    mirror_center : (float, float)
        Mirror (x, y) imaged at the sensor center.
    pixels_per_unit : float
        Camera pixels per unit of mirror position.
    spot_sigma : float
        Spot size, in pixels.
    """

    def __init__(
        self,
        mirror,
        sensor_size_yx: tuple[int, int] = (480, 640),
        mirror_center: Sequence[float] = (0.0, 0.0),
        pixels_per_unit: float = 1.0,
        spot_sigma: float = 4.0,
    ):
        self.mirror = mirror
        self.height, self.width = sensor_size_yx
        self.mirror_center = np.asarray(mirror_center, dtype=float)
        self.pixels_per_unit = pixels_per_unit
        self.spot_sigma = spot_sigma
        self._is_open = False

    def open(self, index=0):
        self._is_open = True
        logger.info("Mock camera opened")

    def close(self):
        self._is_open = False
        logger.info("Mock camera closed")

    @property
    def image_size(self) -> tuple[int, int, int, int]:
        """``(width, height, offset_x, offset_y)``."""
        return (self.width, self.height, 0, 0)

    def spot_xy(self) -> np.ndarray:
        offset = np.asarray(self.mirror.position, dtype=float) - self.mirror_center
        return np.array([self.width / 2, self.height / 2]) + offset * self.pixels_per_unit

    def snap(self) -> np.ndarray:
        x, y = self.spot_xy()
        # Render only a small window around the spot: full-frame exp() on a
        # 2448x2048 sensor is slow enough to make demo calibration drag.
        frame = np.full((self.height, self.width), 100, dtype=np.uint16)
        r = int(6 * self.spot_sigma)
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, self.width)
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, self.height)
        if x0 < x1 and y0 < y1:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            spot = 3000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.spot_sigma**2))
            frame[y0:y1, x0:x1] += spot.astype(np.uint16)
        return frame
