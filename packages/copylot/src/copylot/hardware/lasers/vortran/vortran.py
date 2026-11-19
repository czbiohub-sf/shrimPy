"""Vortran Stradus laser driver over RS-232 (COM port).

Original author: edyoshikun. Vendored from coPylot
``copylot/hardware/lasers/vortran/vortran.py`` (main @ 0bcc3a3), BSD-3-Clause;
see the package LICENSE. For the command set, see the
manuals at https://www.vortranlaser.com/.

Changes from upstream:

- logs through the module loggers under ``copylot`` (no handlers attached at import)
- ``name`` is optional, so :meth:`VortranLaser.get_lasers` (which constructs
  lasers by port only) no longer raises ``TypeError``
- the drive-control-mode setter is bound to ``drive_control_mode`` (upstream
  bound it to a separate ``control_mode`` property, so ``drive_control_mode``
  was read-only)
- connecting by serial number opens each port's device name rather than the
  ``ListPortInfo`` object, and the "not found" warning reports the requested
  serial number
- the ``power`` / ``pulse_power`` setters only clamp to the maximum power once
  it is known, instead of failing with ``TypeError`` before identification
- a port that cannot be opened (``serial.SerialException``: in use, no device)
  is skipped like a non-responding one, instead of aborting the port scan
- :meth:`VortranLaser.get_lasers` raises ``RuntimeError`` (was bare ``Exception``)
"""

from __future__ import annotations

import logging
import time

import serial

from serial.tools import list_ports

from copylot.hardware.lasers.abstract_laser import AbstractLaser

logger = logging.getLogger(__name__)


class VortranLaser(AbstractLaser):
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

    def __init__(
        self,
        name: str | None = None,
        serial_number: str | None = None,
        port: str | None = None,
        baudrate: int = 19200,
        timeout: float = 1,
    ):
        """Connect to a Vortran Stradus laser over a COM port.

        Default parameters are taken from the Vortran documentation.

        Parameters
        ----------
        name : str, optional
            Label for this laser.
        serial_number : str, optional
            Serial number of the laser. If given (and ``port`` is not), every
            COM port is tried until a laser with this serial number answers.
        port : str, optional
            COM port the laser is connected to, e.g. ``"COM5"``.
        baudrate : int
            Baud rate for serial communication.
        timeout : float
            Timeout for serial write/read, in seconds.
        """
        # Serial communication
        self.name = name
        self.port: str | None = port
        self.baudrate: int = baudrate
        self.address = None
        self.timeout: float = timeout

        # Laser specs
        self.serial_number: str | None = None
        self.part_number: int | None = None
        self.wavelength: int | None = None
        self.laser_shape: str | None = None
        self._in_serial_num = serial_number

        # Properties
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
        """Open the serial connection and identify the laser.

        Reads the serial number, part number, wavelength, maximum power, and
        beam shape. Logs a warning, rather than raising, if no laser is found.
        """
        if self.port is None:
            ports = (
                [port.device for port in list_ports.comports()]
                if self._in_serial_num is not None
                else [laser[0] for laser in VortranLaser.get_lasers()]
            )
        else:
            ports = [self.port]

        # try to connect to each candidate port
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
            except (RuntimeError, serial.SerialException) as e:
                logger.debug(f"Could not connect to a laser on port {self.port}: {e}")
                self.disconnect()
                continue

            # Accept the connection unless the user specified a serial number,
            # in which case the laser must match it.
            if self._in_serial_num is None or self._in_serial_num == self.serial_number:
                logger.info(
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
        """Close the serial connection."""
        if self.is_connected:
            self.address.close()
        self.address = None

    @property
    def is_connected(self) -> bool:
        """Whether the serial connection is open."""
        self._is_connected = self.address is not None and self.address.is_open
        logger.debug(f"{self.port} connected: {self._is_connected}")
        return self._is_connected

    def _write_cmd(self, cmd, value=None):
        """Write a command to the laser and return its parsed reply.

        All commands must terminate with a carriage return (``\\r``) to be
        processed.

        Parameters
        ----------
        cmd : str
            Vortran Stradus command.
        value : int, float or str, optional
            Value to set with the command.

        Returns
        -------
        list of str
            The parsed reply to the command.
        """
        try:
            if cmd in VortranLaser.VOLTRAN_CMDS:
                if value is not None:
                    cmd_LF = cmd + "=" + str(value) + "\r"
                else:
                    cmd_LF = cmd + "\r"
                logger.debug(
                    f"Write to laser <{self.serial_number}> -> cmd:<{cmd_LF.encode('utf-8')}>"
                )
                self.address.write(cmd_LF.encode("utf-8"))
                return self._read_cmd(cmd)
            else:
                raise ValueError(f"Command '{cmd}' not found")
        except Exception as e:
            raise RuntimeError(f"Error sending command: {str(e)}") from e

    def _read_cmd(self, cmd):
        """Read and parse the laser's reply to ``cmd``.

        Returns ``['0']`` if no reply arrives within the timeout.
        """
        # All replies to commands are in upper case.
        try:
            elapsed_time = 0
            values = None
            start_time = time.perf_counter()
            while elapsed_time < self.timeout:
                msg = self.address.readline().decode()
                elapsed_time = time.perf_counter() - start_time
                if len(msg) > 1:
                    if msg.endswith("\n") or msg.endswith("\r"):
                        if msg[: len(cmd) + 1] == (cmd + "="):
                            values = msg[len(cmd) + 1 : -2].split(", ")
                            logger.debug(f"msg_out > {msg}")
                            break
            logger.debug(f"Parsed values: {values}")
            if values is None:
                values = ["0"]
            return values
        except RuntimeWarning:
            logger.info("Error: No response from laser")

    def _identify_laser(self):
        """Read the connected laser's serial number, part number, wavelength,
        maximum power, and beam shape."""
        laser_param = self._write_cmd("?LI")
        self.serial_number = laser_param[0]
        self.part_number = int(laser_param[1][:-2])
        self.wavelength = int(laser_param[2][:-2])
        self._max_power = float(laser_param[3][:-2])
        self.laser_shape = laser_param[4]
        logger.debug(f"Laser param: {laser_param}")

    @property
    def drive_control_mode(self):
        """Laser drive control mode (1 = current control, 0 = power control)."""
        self._ctrl_mode = self._write_cmd("?C")[0]
        return self._ctrl_mode

    @drive_control_mode.setter
    def drive_control_mode(self, mode):
        self._ctrl_mode = self._write_cmd("C", str(mode))[0]

    @property
    def emission_delay(self):
        """5 second laser emission delay (1 = on, 0 = off)."""
        self._delay = self._write_cmd("?DELAY")[0]
        return self._delay

    @emission_delay.setter
    def emission_delay(self, mode):
        self._delay = self._write_cmd("DELAY", str(mode))[0]

    @property
    def external_power_control(self):
        """External power control (1 = external control, 0 = off)."""
        self._ext_power_ctrl = self._write_cmd("?EPC")[0]
        return self._ext_power_ctrl

    @external_power_control.setter
    def external_power_control(self, control):
        self._ext_power_ctrl = self._write_cmd("EPC", str(control))[0]

    @property
    def current_control_mode(self):
        """Laser current control (0 - max)."""
        self._current_ctrl = self._write_cmd("?LC")[0]
        return self._current_ctrl

    @current_control_mode.setter
    def current_control_mode(self, value):
        self._current_ctrl = self._write_cmd("LC", value)[0]

    @property
    def toggle_emission(self):
        """Laser emission (1 = on, 0 = off)."""
        self._toggle_emission = self._write_cmd("?LE")[0]
        return self._toggle_emission

    @toggle_emission.setter
    def toggle_emission(self, value):
        if isinstance(value, bool):
            value = str(int(value))
        elif isinstance(value, int):
            value = str(value)

        self._toggle_emission = self._write_cmd("LE", value)[0]

    def turn_on(self):
        """Turn laser emission on."""
        logger.info("Turning laser: ON")
        self.toggle_emission = 1
        return self._toggle_emission

    def turn_off(self):
        """Turn laser emission off."""
        logger.info("Turning laser: OFF")
        self.toggle_emission = 0
        return self._toggle_emission

    def _clamp_power(self, power: float) -> float:
        if self._max_power is not None and power > self._max_power:
            logger.info(f"Maximum power is: {self._max_power}")
            return self._max_power
        return power

    @property
    def power(self):
        """Laser power, clamped to the maximum power. Requires pulse mode OFF."""
        self._curr_power = float(self._write_cmd("?LP")[0])
        return self._curr_power

    @power.setter
    def power(self, power: float):
        power = self._clamp_power(power)
        logger.info(f"Setting power: {power}")
        self._curr_power = float(self._write_cmd("LP", power)[0])

    @property
    def pulse_power(self):
        """Pulse power, clamped to the maximum power. Requires pulse mode ON."""
        self._pulse_power = float(self._write_cmd("?PP")[0])
        return self._pulse_power

    @pulse_power.setter
    def pulse_power(self, power):
        power = self._clamp_power(power)
        logger.info(f"Setting pulse power: {power}")
        self._pulse_power = float(self._write_cmd("PP", str(power))[0])

    @property
    def pulse_mode(self):
        """Pulse (digital modulation) mode (1 = on, 0 = off)."""
        self._pulse_mode = self._write_cmd("?PUL")[0]
        return self._pulse_mode

    @pulse_mode.setter
    def pulse_mode(self, mode=0):
        logger.debug(f"Digital Modulation: {mode}")
        self._pulse_mode = self._write_cmd("PUL", str(mode))[0]

    @property
    def maximum_power(self) -> float:
        """Maximum laser power."""
        self._max_power = float(self._write_cmd("?MAXP")[0])
        return self._max_power

    def _echo_off(self):
        self._write_cmd("ECHO", 0)
        logger.debug("Echo Off")

    @property
    def status(self) -> tuple[int, str]:
        """``(fault code, fault description)`` reported by the laser."""
        fault_code = self._write_cmd("?FC")
        fault_description = self._write_cmd("?FD")
        self._status = (int(fault_code[0]), str(fault_description[0]))
        return self._status

    @staticmethod
    def get_lasers() -> list[tuple[str, str]]:
        """Probe every COM port and return ``(port, serial_number)`` per laser found.

        Raises
        ------
        RuntimeError
            If no laser is found on any port.
        """
        lasers = []
        for port in list_ports.comports():
            try:
                laser = VortranLaser(port=port.device)
                if laser.serial_number is not None:
                    lasers.append((laser.port, laser.serial_number))
                    logger.info(f"Found: {laser.port}:{laser.serial_number}")
                    laser.disconnect()
            except (RuntimeWarning, RuntimeError):
                logger.debug(f"No laser found in {port}")

        if len(lasers) == 0:
            logger.info("No lasers found...")
            raise RuntimeError("No Vortran lasers found on any COM port")

        return lasers
