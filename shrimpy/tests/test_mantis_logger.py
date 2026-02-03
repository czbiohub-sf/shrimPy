"""Tests for the mantis_logger module."""

import logging
import tempfile

from pathlib import Path

import pytest

from pymmcore_plus._logger import logger as pymmcore_logger

from shrimpy.mantis.mantis_logger import configure_mantis_logger, get_mantis_logger

pytest.skip("Temporarily skipping all logger tests", allow_module_level=True)


def test_get_mantis_logger():
    """Test that get_mantis_logger returns the mantis logger."""
    logger = get_mantis_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "mantis"


def test_configure_mantis_logger():
    """Test that configure_mantis_logger sets up handlers correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = configure_mantis_logger(tmpdir, "test_acquisition")

        # Check that logger is configured
        assert isinstance(logger, logging.Logger)
        assert logger.name == "mantis"
        assert logger.level == logging.DEBUG

        # Check that mantis logger has 2 handlers (console + file)
        assert len(logger.handlers) == 2

        # Check handler levels
        handler_levels = [h.level for h in logger.handlers]
        assert logging.INFO in handler_levels  # console handler
        assert logging.DEBUG in handler_levels  # file handler

        # Check that log directory was created
        log_dir = Path(tmpdir) / "logs"
        assert log_dir.exists()

        # Check that log file was created
        log_files = list(log_dir.glob("test_acquisition_log_*.txt"))
        assert len(log_files) == 1


def test_pymmcore_logger_not_modified():
    """Test that pymmcore-plus logger handlers are not cleared."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Record initial pymmcore logger handlers count
        initial_handlers = len(pymmcore_logger.handlers)

        # Configure mantis logger
        configure_mantis_logger(tmpdir, "test_acquisition")

        # pymmcore logger should have gained one handler (file handler)
        # but original handlers should not be cleared
        assert len(pymmcore_logger.handlers) == initial_handlers + 1


def test_logger_console_output():
    """Test that console handler is configured at INFO level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = configure_mantis_logger(tmpdir, "test_acquisition")

        # Find console handler (StreamHandler)
        console_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                console_handler = handler
                break

        assert console_handler is not None
        assert console_handler.level == logging.INFO


def test_logger_file_output():
    """Test that DEBUG messages are written to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = configure_mantis_logger(tmpdir, "test_acquisition")

        # Write some log messages
        logger.debug("Test DEBUG message from mantis")
        logger.info("Test INFO message from mantis")
        logger.warning("Test WARNING message from mantis")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Find log file
        log_dir = Path(tmpdir) / "logs"
        log_files = list(log_dir.glob("test_acquisition_log_*.txt"))
        assert len(log_files) == 1

        # Read log file
        log_content = log_files[0].read_text()

        # Check that all messages are in the file
        assert "Test DEBUG message from mantis" in log_content
        assert "Test INFO message from mantis" in log_content
        assert "Test WARNING message from mantis" in log_content
        # Check that logger name is in the file
        assert "mantis" in log_content


def test_pymmcore_events_captured():
    """Test that pymmcore-plus logger events are captured to mantis log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        configure_mantis_logger(tmpdir, "test_acquisition")

        # Write a message from pymmcore logger
        pymmcore_logger.info("Test message from pymmcore-plus")

        # Flush all handlers
        for handler in pymmcore_logger.handlers:
            handler.flush()

        # Find log file
        log_dir = Path(tmpdir) / "logs"
        log_files = list(log_dir.glob("test_acquisition_log_*.txt"))
        assert len(log_files) == 1

        # Read log file
        log_content = log_files[0].read_text()

        # Check that pymmcore message is in the file
        assert "Test message from pymmcore-plus" in log_content
        # Check that logger name shows it's from pymmcore-plus
        assert "pymmcore-plus" in log_content


def test_multiple_logger_configurations():
    """Test that reconfiguring the logger replaces mantis handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First configuration
        logger1 = configure_mantis_logger(tmpdir, "acquisition1")
        handler_count_1 = len(logger1.handlers)

        # Second configuration should replace handlers
        logger2 = configure_mantis_logger(tmpdir, "acquisition2")
        handler_count_2 = len(logger2.handlers)

        # Should have same number of handlers
        assert handler_count_1 == handler_count_2 == 2

        # Should be the same logger instance
        assert logger1 is logger2
