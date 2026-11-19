"""Minimal serial control of the Arduino that pulses the photom lasers (PWM).

Vendored from coPylot ``copylot/assemblies/photom/utils/arduino.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE. The matching Arduino
sketches are in ``arduino/``.

The Arduino parses:

- ``U,duty_cycle,frequency,duration``: update the PWM settings
- ``S``: start the PWM

Changes from upstream: logging instead of ``print``, no tqdm, and ``close()`` /
``__del__`` are safe when the connection never opened.
"""

from __future__ import annotations

import logging
import time

import serial

logger = logging.getLogger(__name__)


class ArduinoPWM:
    def __init__(self, serial_port: str, baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.connect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def connect(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate)
            time.sleep(2)  # wait for the Arduino to reset after the port opens
        except serial.SerialException as e:
            logger.error(f"Could not connect to the Arduino on {self.serial_port}: {e}")

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None
            logger.info("Arduino serial connection closed.")

    def send_command(self, command: str):
        logger.info(f"Sending Arduino command: {command}")
        self.ser.write((command + "\n").encode())
        time.sleep(1)  # wait for the Arduino to process the command
        while self.ser.in_waiting:
            logger.info(f"Arduino: {self.ser.readline().decode().strip()}")

    def set_pwm(self, duty_cycle, frequency, duration):
        self.send_command(f"U,{duty_cycle},{frequency},{duration}")

    def start(self):
        self.send_command("S")

    def start_timelapse(self, repetitions=1, time_interval_s=5):
        for i in range(repetitions):
            logger.info(f"Timelapse repetition {i + 1}/{repetitions}")
            self.start()
            time.sleep(time_interval_s)
