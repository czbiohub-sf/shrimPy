"""VortranLaser against a mocked serial port (ported from coPylot's test_vortran.py)."""

from __future__ import annotations

from unittest import mock

import pytest
import serial

from copylot.hardware.lasers.vortran.vortran import VortranLaser


@pytest.fixture
def serial_number():
    return "some-serial-number"


@pytest.fixture
def mock_serial(serial_number):
    """Mock serial.Serial; readline answers the ``?LI`` query from ``_identify_laser``."""
    with mock.patch("serial.Serial", autospec=True) as mock_serial:
        instance = mock_serial.return_value
        instance.readline.return_value = (
            f"?LI={serial_number}, 600nm, 400mW, 500mW, laser-shape\r\n".encode()
        )
        # autospec does not see the `is_open` attribute
        instance.is_open = True
        yield mock_serial


@pytest.fixture
def mock_ports():
    ports = [mock.MagicMock(device=port) for port in ["some-port", "another-port"]]
    with mock.patch("serial.tools.list_ports.comports", autospec=True) as comports:
        comports.return_value = ports
        yield ports


@pytest.fixture
def create_laser(mock_serial):
    """Build a laser whose post-identification readline calls return the given lines."""

    def _create_laser(readline_return_values):
        mock_serial.return_value.readline.side_effect = [
            mock_serial.return_value.readline.return_value,
            *[value.encode() for value in readline_return_values],
        ]
        return VortranLaser(port="some-port")

    return _create_laser


def _expected_serial_call(port, laser):
    return mock.call(
        port=port,
        baudrate=laser.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=laser.timeout,
    )


def test_connect_with_serial_number(mock_serial, mock_ports, serial_number):
    """Ports are tried in order until one answers with the requested serial number,
    and each is opened by its device name."""
    laser = VortranLaser(serial_number=serial_number)

    assert mock_serial.call_args_list == [_expected_serial_call(mock_ports[0].device, laser)]
    assert laser.is_connected
    assert laser.serial_number == serial_number
    assert laser.address == mock_serial.return_value


@pytest.mark.usefixtures("mock_serial", "mock_ports")
def test_connect_with_wrong_serial_number(caplog):
    laser = VortranLaser(serial_number="obviously wrong serial number")
    assert not laser.is_connected
    assert "obviously wrong serial number" in caplog.text


def test_connect_with_port(mock_serial, serial_number):
    """With a port, the serial number is parsed but not checked."""
    laser = VortranLaser(port="some-port")
    assert mock_serial.call_args_list == [_expected_serial_call("some-port", laser)]
    assert laser.is_connected
    assert laser.serial_number == serial_number
    assert laser._max_power == 500.0


@pytest.mark.usefixtures("mock_serial", "mock_ports")
def test_connect_with_no_args(serial_number):
    """Without port or serial number, the first laser from get_lasers() is used."""
    laser = VortranLaser()
    assert laser.is_connected
    assert laser.serial_number == serial_number


def test_disconnect(mock_serial):
    laser = VortranLaser(port="some-port")
    # a second disconnect is a no-op
    laser.disconnect()
    laser.disconnect()
    mock_serial.return_value.close.assert_called_once()


@pytest.mark.usefixtures("mock_serial")
def test_get_lasers(mock_ports, serial_number):
    laser_info = VortranLaser.get_lasers()
    assert laser_info == [(port.device, serial_number) for port in mock_ports]


def test_get_lasers_none_found(mock_serial, mock_ports):
    mock_serial.side_effect = serial.SerialException("no device")
    with pytest.raises(RuntimeError, match="No Vortran lasers"):
        VortranLaser.get_lasers()


def test_turn_laser_on_off(create_laser):
    laser = create_laser(["LE=1\r\n", "LE=0\r\n"])
    assert laser.turn_on() == "1"
    assert laser.turn_off() == "0"


def test_set_get_power(create_laser):
    # the getter's reply is prefixed by the *query* '?LP', the setter's by 'LP'
    laser = create_laser(["LP=50\r\n", "?LP=50\r\n"])
    laser.power = 50
    assert laser.power == 50


def test_power_is_clamped_to_maximum(create_laser, mock_serial):
    laser = create_laser(["LP=500\r\n"])
    laser.power = 9000
    mock_serial.return_value.write.assert_called_with(b"LP=500.0\r")


def test_drive_control_mode_is_settable(create_laser, mock_serial):
    laser = create_laser(["C=1\r\n", "?C=1\r\n"])
    laser.drive_control_mode = 1
    mock_serial.return_value.write.assert_called_with(b"C=1\r")
    assert laser.drive_control_mode == "1"
