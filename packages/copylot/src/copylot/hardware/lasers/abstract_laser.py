"""Interface shared by the laser drivers.

Vendored from coPylot ``copylot/hardware/lasers/abstract_laser.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod


class AbstractLaser(metaclass=ABCMeta):
    """Members known to be common across laser adapters.

    Not a final interface; members are added as needs arise.
    """

    _device_id = None

    @property
    def device_id(self):
        """Device id (serial number) of the laser."""
        return self._device_id

    @abstractmethod
    def connect(self):
        """Connect to the laser."""

    @abstractmethod
    def disconnect(self):
        """Disconnect from the laser."""

    @property
    @abstractmethod
    def drive_control_mode(self) -> str:
        """The laser drive control mode."""

    @drive_control_mode.setter
    @abstractmethod
    def drive_control_mode(self, value: str):
        pass

    @property
    @abstractmethod
    def toggle_emission(self) -> bool:
        """Laser emission on/off."""

    @toggle_emission.setter
    @abstractmethod
    def toggle_emission(self, value: bool):
        pass

    @property
    @abstractmethod
    def power(self) -> float:
        """Laser power."""

    @power.setter
    @abstractmethod
    def power(self, value: float):
        pass

    @property
    @abstractmethod
    def pulse_mode(self) -> bool:
        """Laser pulse (digital modulation) mode."""

    @pulse_mode.setter
    @abstractmethod
    def pulse_mode(self, value: bool):
        pass

    @property
    @abstractmethod
    def maximum_power(self) -> float:
        """Maximum laser power."""

    @property
    @abstractmethod
    def current_control_mode(self) -> str:
        """Laser current control."""

    @current_control_mode.setter
    @abstractmethod
    def current_control_mode(self, value: str):
        pass

    @property
    @abstractmethod
    def external_power_control(self) -> bool:
        """External (analog) power control."""

    @external_power_control.setter
    @abstractmethod
    def external_power_control(self, value: bool):
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        """Laser status."""
