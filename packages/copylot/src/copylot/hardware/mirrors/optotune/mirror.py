"""Optotune MR-E-2 tip/tilt mirror controller.

Original author: Yang Liu. Vendored from coPylot
``copylot/hardware/mirrors/optotune/mirror.py`` (main @ 0bcc3a3), BSD-3-Clause;
see the package LICENSE. Manuals:
https://www.optotune.com/fast-steering-mirrors

Requires Optotune's MR-E Python SDK (``optoMDC`` / ``optoKummenberg``), from the
``optotune`` extra; see ``third_party/optotune/README.md``. Upstream coPylot
bundled an older copy of the SDK; this uses Optotune's released 1.3.5434 wheels.

Changes from upstream:

- dropped unused imports (``os.device_encoding``, ``zmq.device``; pyzmq was not
  otherwise a dependency)
- ``__del__`` no longer raises ``AttributeError`` when ``__init__`` failed
  before connecting, and ``disconnect()`` can be called explicitly; position
  changes log at debug level (they fire on every slider move)
- ``ValueError`` when no MR-E-2 board is found and no ``com_port`` was given
  (upstream passed ``None`` to the SDK)
- connects with ``optoMDC.connectmre2``, SDK 1.3's name for the MR-E-2 board
  (``optoMDC.connect`` in the SDK copy coPylot bundled)
- position getters accept both register-read forms: SDK 1.3 returns the value,
  coPylot's bundled SDK returned ``[value]`` (indexing a float raised ``TypeError``)
"""

from __future__ import annotations

import logging

import optoMDC

from optoMDC.tools.list_comports import get_mre2_port

from copylot.hardware.mirrors.abstract_mirror import AbstractMirror

logger = logging.getLogger(__name__)


def _register_value(response) -> float:
    """A register read as a float, across Optotune SDK versions.

    The SDK copy coPylot bundled returned the reply as a list (``[value]``); SDK
    1.3 returns the value itself (``get_register`` returns ``response[0]``).
    """
    if isinstance(response, (list, tuple)):
        response = response[0]
    return float(response)


class OptoMirror(AbstractMirror):
    """Optotune MR-E-2 fast steering mirror over a COM port.

    Positions are normalized angles, ``theta / tan(50 deg)`` (50 degrees is the
    maximum optical deflection each way), each in ``[-1, 1]``. The combination
    must stay inside the unit circle (``x**2 + y**2 < 1``): any combination works
    while ``|x|, |y| < 0.7``; beyond that the controller pulls the point back to
    the nearest edge of the unit circle.

    Parameters
    ----------
    name : str
        Device label.
    com_port : str, optional
        COM port of the MR-E-2 board. Found by USB hardware id if omitted.
    pos_x, pos_y : float
        Initial position.
    """

    def __init__(
        self,
        name: str = "OPTOTUNE_MIRROR",
        com_port: str | None = None,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.mirror = None
        if com_port is None:
            com_port = get_mre2_port()
            if com_port is None:
                raise ValueError("No Optotune MR-E-2 board found; pass com_port")
        self.mirror = optoMDC.connectmre2(com_port)

        self.channel_x = self.mirror.Mirror.Channel_0
        self.channel_x.SetControlMode(optoMDC.Units.XY)
        self.channel_x.StaticInput.SetAsInput()

        self.channel_y = self.mirror.Mirror.Channel_1
        self.channel_y.SetControlMode(optoMDC.Units.XY)
        self.channel_y.StaticInput.SetAsInput()
        logger.info(f"Optotune mirror {name} connected on {com_port}")
        self.position_x = pos_x
        self.position_y = pos_y

        self._movement_limits = [-1.0, 1.0, -1.0, 1.0]

    def disconnect(self):
        """Center the mirror and close the connection (idempotent)."""
        if self.mirror is None:
            return
        self.position_x = 0
        self.position_y = 0
        self.mirror.disconnect()
        self.mirror = None
        logger.info(f"Optotune mirror {self.name} disconnected")

    def __del__(self):
        try:
            self.disconnect()
        except Exception:  # interpreter shutdown, lost connection
            pass

    @property
    def device_id(self):
        return self.name

    @device_id.setter
    def device_id(self, value: str):
        self.name = value
        logger.info(f"device_id set to: {value}")

    @property
    def position(self):
        """``(x, y)`` normalized angular position."""
        return self.position_x, self.position_y

    @position.setter
    def position(self, value: list[float]):
        self.position_x = value[0]
        self.position_y = value[1]

    @property
    def position_x(self):
        """Normalized angular x position, in ``[-1, 1]``."""
        return _register_value(self.channel_x.StaticInput.GetXY())

    @position_x.setter
    def position_x(self, value):
        self.channel_x.StaticInput.SetXY(value)
        logger.debug(f"position_x set to: {value}")

    @property
    def position_y(self):
        """Normalized angular y position, in ``[-1, 1]``."""
        return _register_value(self.channel_y.StaticInput.GetXY())

    @position_y.setter
    def position_y(self, value):
        self.channel_y.StaticInput.SetXY(value)
        logger.debug(f"position_y set to: {value}")

    @property
    def relative_position(self) -> list[float]:
        """Not implemented upstream."""

    @relative_position.setter
    def relative_position(self, value: list[float]):
        pass

    @property
    def movement_limits(self) -> list[float]:
        """``[x_min, x_max, y_min, y_max]``."""
        return self._movement_limits

    @movement_limits.setter
    def movement_limits(self, value: list[float]):
        self._movement_limits = value

    @property
    def step_resolution(self) -> float:
        """Not implemented upstream."""

    @step_resolution.setter
    def step_resolution(self, value: float):
        pass

    def set_home(self):
        """Not implemented upstream."""

    def set_origin(self, axis: str):
        """Not implemented upstream."""

    @property
    def external_drive_control(self) -> str:
        """Not implemented upstream."""

    @external_drive_control.setter
    def external_drive_control(self, value: bool):
        pass

    def voltage_to_position(self, voltage: list[float]) -> list[float]:
        """Not implemented upstream."""
