"""Interface shared by the mirror drivers.

Vendored from coPylot ``copylot/hardware/mirrors/abstract_mirror.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE. Unchanged apart from
formatting.
"""

from abc import ABCMeta, abstractmethod


class AbstractMirror(metaclass=ABCMeta):
    """AbstractMirror

    This class includes only the members that known to be common
    across different mirror adapters. By no
    means this class in a final state. We will be making additions
    as needs rise.

    """

    def __init__(self):
        self.name: str = "AbstractMirror"
        self.affine_transform_obj = None
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0

    @property
    def device_id(self):
        "Returns the device unique id(name or serial number)of the current mirror"
        return self.name

    @device_id.setter
    @abstractmethod
    def device_id(self, value: str):
        "Sets the device unique id(name or serial number)of the current mirror"
        pass

    @property
    @abstractmethod
    def position(self) -> list[float, float]:
        """Get the current mirror position XY"""
        pass

    @position.setter
    @abstractmethod
    def position(self, value: list[float, float]):
        """Set the mirror position XY"""
        pass

    @property
    @abstractmethod
    def position_x(self) -> float:
        """Get the current mirror position X"""
        pass

    @position_x.setter
    @abstractmethod
    def position_x(self, value: float):
        """Set the mirror position X"""
        pass

    @property
    @abstractmethod
    def position_y(self) -> float:
        """Get the current mirror position Y"""
        pass

    @position_y.setter
    @abstractmethod
    def position_y(self, value: float):
        """Set the mirror position Y"""
        pass

    @property
    @abstractmethod
    def relative_position(self) -> list[float, float]:
        """Get the current relative mirror position"""
        pass

    @relative_position.setter
    @abstractmethod
    def relative_position(self, value: list[float, float]):
        """Set the relative mirror position"""
        pass

    @property
    @abstractmethod
    def movement_limits(self) -> list[float, float, float, float]:
        """Get the current mirror movement limits"""
        pass

    @movement_limits.setter
    @abstractmethod
    def movement_limits(self, value: list[float, float, float, float]):
        """Set the mirror movement limits"""
        pass

    @property
    @abstractmethod
    def step_resolution(self) -> float:
        """Get the current mirror step resolution"""
        pass

    @step_resolution.setter
    @abstractmethod
    def step_resolution(self, value: float):
        """Set the mirror step resolution"""
        pass

    @abstractmethod
    def set_home(self):
        """Set the mirror home position"""
        pass

    @abstractmethod
    def set_origin(self, axis: str):
        """Set the mirror origin for a specific axis"""
        pass

    @property
    @abstractmethod
    def external_drive_control(self) -> str:
        """Get the current mirror drive mode"""
        pass

    @external_drive_control.setter
    @abstractmethod
    def external_drive_control(self, value: bool):
        """Set the mirror drive mode"""
        pass

    @abstractmethod
    def voltage_to_position(self, voltage: list[float, float]) -> list[float, float]:
        """Convert voltage to position"""
        pass
