"""Vortran Stradus laser RS-232 / COM-port wrapper.

Vendored from copylot (copylot.hardware.lasers.vortran.vortran) — see
https://www.vortranlaser.com/ for the command reference. The original
``AbstractLaser`` base class was empty so it has been dropped, and
``copylot.logger`` is replaced with a standard module logger.
"""

from __future__ import annotations

import logging
import time

import serial

from serial.tools import list_ports

logger = logging.getLogger(__name__)


class VortranLaser:
    """Serial wrapper for a Vortran Stradus laser on a COM port."""

    GLOBAL_CMD = ["ECHO", "PROMPT"]
    GLOBAL_QUERY = ["?BPT", "?H", "?IL", "?SFV", "?SPV"]
    LASER_CMD = ["C", "DELAY", "EPC", "LC", "LE", "LP", "PP", "PUL"]
    LASER_QUERY = [
        "?C",
        "?CC",
        "?CT",
        "?DELAY",
        "?EPC",
        "?FC",
        "?FD",
        "?FP",
        "?FV",
        "?LC",
        "?LCS",
        "?LE",
        "?LH",
        "?LI",
        "?LP",
        "?LPS",
        "?LW",
        "?MAXP",
        "?OBT",
        "?OBTS",
        "?PP",
        "?PUL",
        "?RP",
    ]
    VOLTRAN_CMDS = GLOBAL_CMD + GLOBAL_QUERY + LASER_CMD + LASER_QUERY

    def __init__(self, serial_number=None, port=None, baudrate=19200, timeout=1):
        """Connect to a Vortran Stradus laser over a serial COM port.

        Parameters
        ----------
        serial_number : str, optional
            Serial number for the device. If given, every candidate port is
            opened and only a laser matching this serial number is accepted.
        port : str, optional
            Explicit COM port (e.g. ``"COM6"``). If omitted, ports are
            auto-discovered via :meth:`get_lasers`.
        baudrate : int
            Serial baud rate.
        timeout : int
            Read/write timeout in seconds.
        """
        self.port: str | None = port
        self.baudrate: int = baudrate
        self.address = None
        self.timeout: int = timeout

        self.serial_number: str | None = None
        self.part_number: int | None = None
        self.wavelength: int | None = None
        self.laser_shape: str | None = None
        self._in_serial_num = serial_number

        self._curr_power = None
        self._ctrl_mode = None
        self._delay = None
        self._ext_power_ctrl = None
        self._current_ctrl = None
        self._toggle_emission = None
        self._pulse_power = None
        self._pulse_mode = None
        self._max_power = None
        self._is_connected = False
        self._status = None

        self.connect()

    def connect(self):
        """Open the serial port and read back the laser's identity."""
        if self.port is None:
            ports = (
                list_ports.comports()
                if self._in_serial_num is not None
                else [laser[0] for laser in VortranLaser.get_lasers()]
            )
        else:
            ports = [self.port]

        laser_found = False
        for port in ports:
            self.port = port
            try:
                self.address = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                )
                self._identify_laser()
            except RuntimeError:
                logger.debug(f"Runtime error while connecting to port {self.port}")
                self.disconnect()
                continue

            if self._in_serial_num is None or self._in_serial_num == self.serial_number:
                logger.debug(
                    f"Connected to Vortran laser {self.serial_number} on serial port {self.port}"
                )
                laser_found = True
                break
            else:
                self.disconnect()

        if not laser_found:
            if self._in_serial_num is not None:
                message = f"No laser found for serial number {self._in_serial_num}"
            else:
                message = f"No laser found on ports {ports}"
            logger.warning(message)

    def disconnect(self):
        """Close the serial port if open."""
        if self.is_connected:
            self.address.close()
        self.address = None

    @property
    def is_connected(self):
        self._is_connected = self.address is not None and self.address.is_open
        return self._is_connected

    def _write_cmd(self, cmd, value=None):
        """Send ``cmd`` (optionally ``cmd=value``) to the laser and parse the reply."""
        try:
            if cmd in VortranLaser.VOLTRAN_CMDS:
                if value is not None:
                    cmd_LF = cmd + "=" + str(value) + "\r"  # Noqa: N806
                else:
                    cmd_LF = cmd + "\r"  # Noqa: N806
                logger.debug(
                    f"Write to laser <{self.serial_number}> -> cmd:<{cmd_LF.encode('utf-8')}>"
                )
                self.address.write(cmd_LF.encode("utf-8"))
                return self._read_cmd(cmd)
            raise ValueError(f"Command '{cmd}' not found")
        except Exception as e:
            raise RuntimeError(f"Error sending command: {e}") from None

    def _read_cmd(self, cmd):
        """Read until we see ``cmd=<values>`` on the line, parse and return values."""
        try:
            elapsed_time = 0
            values = None
            start_time = time.perf_counter()
            while elapsed_time < self.timeout:
                msg = self.address.readline().decode()
                elapsed_time = time.perf_counter() - start_time
                if len(msg) > 1 and (msg.endswith("\n") or msg.endswith("\r")):
                    if msg[: len(cmd) + 1] == (cmd + "="):
                        values = msg[len(cmd) + 1 : -2].split(", ")
                        logger.debug(f"msg_out > {msg}")
                        break
            if values is None:
                values = ["0"]
            return values
        except RuntimeWarning:
            logger.warning("Error: No response from laser")

    def _identify_laser(self):
        """Populate serial number, part number, wavelength, max power, shape."""
        laser_param = self._write_cmd("?LI")
        self.serial_number = laser_param[0]
        self.part_number = int(laser_param[1][:-2])
        self.wavelength = int(laser_param[2][:-2])
        self._max_power = float(laser_param[3][:-2])
        self.laser_shape = laser_param[4]
        logger.debug(f"Laser param: {laser_param}")

    @property
    def drive_control_mode(self):
        self._ctrl_mode = self._write_cmd("?C")[0]
        return self._ctrl_mode

    @drive_control_mode.setter
    def control_mode(self, mode):
        self._ctrl_mode = self._write_cmd("C", str(mode))[0]

    @property
    def emission_delay(self):
        self._delay = self._write_cmd("?DELAY")[0]
        return self._delay

    @emission_delay.setter
    def emission_delay(self, mode):
        self._delay = self._write_cmd("DELAY", str(mode))[0]

    @property
    def external_power_control(self):
        self._ext_power_ctrl = self._write_cmd("?EPC")[0]
        return self._ext_power_ctrl

    @external_power_control.setter
    def external_power_control(self, control):
        self._ext_power_ctrl = self._write_cmd("EPC", str(control))[0]

    @property
    def current_control(self):
        self._current_ctrl = self._write_cmd("?LC")[0]
        return self._current_ctrl

    @current_control.setter
    def current_control(self, value):
        self._current_ctrl = self._write_cmd("LC", value)[0]

    @property
    def toggle_emission(self):
        self._toggle_emission = self._write_cmd("?LE")[0]
        return self._toggle_emission

    @toggle_emission.setter
    def toggle_emission(self, value):
        self._toggle_emission = self._write_cmd("LE", value)[0]

    def turn_on(self):
        logger.debug("Turning laser: ON")
        self.toggle_emission = 1
        return self._toggle_emission

    def turn_off(self):
        logger.debug("Turning laser: OFF")
        self.toggle_emission = 0
        return self._toggle_emission

    @property
    def laser_power(self):
        self._curr_power = float(self._write_cmd("?LP")[0])
        return self._curr_power

    @laser_power.setter
    def laser_power(self, power: float):
        if self._max_power is not None and power > self._max_power:
            power = self._max_power
            logger.debug(f"Maximum power is: {self._max_power}")
        logger.debug(f"Setting power: {power}")
        self._curr_power = float(self._write_cmd("LP", power)[0])

    @property
    def pulse_power(self):
        self._pulse_power = float(self._write_cmd("?PP")[0])
        return self._pulse_power

    @pulse_power.setter
    def pulse_power(self, power):
        logger.debug(f"Setting pulse power: {power}")
        self._pulse_power = float(self._write_cmd("PP", str(power))[0])

    @property
    def pulse_mode(self):
        self._pulse_mode = self._write_cmd("?PUL")[0]
        return self._pulse_mode

    @pulse_mode.setter
    def pulse_mode(self, mode=0):
        self._pulse_mode = self._write_cmd("PUL", str(mode))[0]

    @property
    def maximum_power(self) -> float:
        self._max_power = float(self._write_cmd("?MAXP")[0])
        return self._max_power

    def _echo_off(self):
        self._write_cmd("ECHO", 0)
        logger.debug("Echo Off")

    @property
    def status(self) -> tuple:
        fault_code = self._write_cmd("?FC")
        fault_description = self._write_cmd("?FD")
        self._status = (int(fault_code[0]), str(fault_description[0]))
        return self._status

    @staticmethod
    def get_lasers():
        """Probe all COM ports and return ``[(port, serial_number), ...]``."""
        com_ports = list_ports.comports()
        lasers = []
        for port in com_ports:
            try:
                laser = VortranLaser(port=port.device)
                if laser.serial_number is not None:
                    lasers.append((laser.port, laser.serial_number))
                    logger.debug(f"Found: {laser.port}:{laser.serial_number}")
                    laser.disconnect()
            except (RuntimeWarning, RuntimeError):
                logger.warning(f"No laser found in {port}")

        if len(lasers) == 0:
            logger.critical("No lasers found...")
            raise RuntimeError("No Vortran lasers detected on any COM port")

        return lasers


def setup_vortran_laser(com_port: str) -> VortranLaser:
    """Open a Vortran laser on ``com_port`` and enable digital pulse modulation.

    Matches the helper from the archived pycromanager engine
    (``microscope_operations.setup_vortran_laser``).
    """
    logger.debug(f"Setting up Vortran laser on COM port {com_port}")
    laser = VortranLaser(port=com_port)
    laser.pulse_mode = 1
    return laser
