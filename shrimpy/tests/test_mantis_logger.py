"""Tests for mantis logging configuration."""

import logging
import tempfile

from pathlib import Path

import pytest


@pytest.fixture
def config_file() -> Path:
    """Fixture that provides path to logging config file."""
    return Path(__file__).parent.parent.parent / "config" / "logging.ini"


@pytest.fixture
def temp_log_dir():
    """Fixture that provides a temporary directory and ensures logging cleanup.

    This prevents file locking issues on Windows when temp directories are cleaned up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
        # Close all handlers before temp directory cleanup
        logging.shutdown()
        # Extra cleanup to ensure all handlers are closed
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except Exception:
                    pass


def test_logging_config_file_exists(config_file):
    """Test that logging configuration file exists."""
    assert config_file.exists(), f"Logging config not found at {config_file}"


def test_configure_logging_from_cli(config_file, temp_log_dir):
    """Test the configure_logging function from cli.acquire."""
    from shrimpy._logging import configure_logging

    output_dir = temp_log_dir
    log_file = configure_logging(config_file, output_dir, "test_acquisition")

    # Check that log file path is returned
    assert isinstance(log_file, Path)
    assert log_file.exists()

    # Check that log directory was created
    log_dir = output_dir / "logs"
    assert log_dir.exists()

    # Check that log file was created with timestamp naming
    log_files = list(log_dir.glob("test_acquisition_log_*.log"))
    assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"


def test_logger_hierarchy(config_file, temp_log_dir):
    """Test that logger hierarchy is properly set up."""
    from shrimpy._logging import configure_logging

    configure_logging(config_file, temp_log_dir, "test_acquisition")

    # Get loggers from the shrimpy hierarchy
    cli_logger = logging.getLogger("shrimpy.cli.acquire")
    mantis_logger = logging.getLogger("shrimpy.mantis.mantis_engine")

    # Both should be configured through the hierarchy
    assert cli_logger.isEnabledFor(logging.INFO)
    assert mantis_logger.isEnabledFor(logging.DEBUG)


def test_logger_file_handler(config_file, temp_log_dir):
    """Test that log file uses FileHandler as in the original setup."""
    from shrimpy._logging import configure_logging

    configure_logging(config_file, temp_log_dir, "test_acquisition")

    # Get the root shrimpy logger
    shrimpy_logger = logging.getLogger("shrimpy")

    # Find file handler
    file_handler = None
    for handler in shrimpy_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    # FileHandler should be configured
    assert file_handler is not None, "FileHandler not found"
    assert file_handler.level == logging.DEBUG


def test_logger_writes_to_file(config_file, temp_log_dir):
    """Test that log messages are written to file."""
    from shrimpy._logging import configure_logging

    output_dir = temp_log_dir
    configure_logging(config_file, output_dir, "test_acquisition")

    # Get a logger and write messages
    logger = logging.getLogger("shrimpy.mantis.mantis_engine")
    logger.debug("Test DEBUG message")
    logger.info("Test INFO message")
    logger.warning("Test WARNING message")

    # Flush all handlers
    for handler in logging.getLogger("shrimpy").handlers:
        handler.flush()

    # Check log file content
    log_dir = output_dir / "logs"
    log_files = list(log_dir.glob("test_acquisition_log_*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()

    # All messages should be in the file (DEBUG level)
    assert "Test DEBUG message" in log_content
    assert "Test INFO message" in log_content
    assert "Test WARNING message" in log_content


def test_multiple_acquisitions_separate_logs(config_file, temp_log_dir):
    """Test that multiple acquisitions create separate log files."""
    from shrimpy._logging import configure_logging

    output_dir = temp_log_dir

    # First acquisition
    configure_logging(config_file, output_dir, "acquisition_1")
    logger1 = logging.getLogger("shrimpy.test1")
    logger1.info("Message from acquisition 1")

    # Close handlers to release file handles
    for handler in logging.getLogger("shrimpy").handlers[:]:
        handler.close()
        logging.getLogger("shrimpy").removeHandler(handler)

    # Second acquisition
    configure_logging(config_file, output_dir, "acquisition_2")
    logger2 = logging.getLogger("shrimpy.test2")
    logger2.info("Message from acquisition 2")

    # Close handlers
    for handler in logging.getLogger("shrimpy").handlers[:]:
        handler.close()
        logging.getLogger("shrimpy").removeHandler(handler)

    # Check that both log files exist with timestamp naming
    log_dir = output_dir / "logs"
    log_files_1 = list(log_dir.glob("acquisition_1_log_*.log"))
    log_files_2 = list(log_dir.glob("acquisition_2_log_*.log"))

    assert len(log_files_1) == 1
    assert len(log_files_2) == 1

    # Check content
    assert "Message from acquisition 1" in log_files_1[0].read_text()
    assert "Message from acquisition 2" in log_files_2[0].read_text()


def test_fallback_logging_when_config_missing(config_file, temp_log_dir):
    """Test that logging falls back to basic config when config file is missing."""
    from shrimpy._logging import configure_logging

    output_dir = temp_log_dir

    # Temporarily move config file (simulate missing)
    backup_name = config_file.with_suffix(".ini.bak")

    try:
        if config_file.exists():
            config_file.rename(backup_name)

        # Should not raise exception, should use fallback
        log_file = configure_logging(config_file, output_dir, "test_acquisition")

        # Log file path should be returned
        assert isinstance(log_file, Path)

        # Log file should still be created with timestamp naming
        log_dir = output_dir / "logs"
        log_files = list(log_dir.glob("test_acquisition_log_*.log"))
        assert len(log_files) == 1
        assert log_file.exists()

    finally:
        # Restore config file
        if backup_name.exists():
            backup_name.rename(config_file)


def test_pymmcore_logger_captured(config_file, temp_log_dir):
    """Test that pymmcore-plus logger events are captured to the log file."""
    from shrimpy._logging import configure_logging

    configure_logging(config_file, temp_log_dir, "test_acquisition")

    # Get pymmcore-plus logger and write a message
    pymmcore_logger = logging.getLogger("pymmcore-plus")
    pymmcore_logger.info("Test message from pymmcore-plus")

    # Flush all handlers
    for handler in pymmcore_logger.handlers:
        handler.flush()

    # Check that message is in the log file
    log_dir = temp_log_dir / "logs"
    log_files = list(log_dir.glob("test_acquisition_log_*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()

    assert "Test message from pymmcore-plus" in log_content
    assert "pymmcore-plus" in log_content


def test_pymmcore_logger_captured_after_import(config_file, temp_log_dir):
    """Test that pymmcore-plus events reach the shrimpy log file when pymmcore-plus's
    configure_logging() runs before shrimpy's — the real import-time scenario where
    MantisEngine is imported first to force the correct ordering.
    """
    from pymmcore_plus._logger import configure_logging as pymmcore_configure_logging
    from pymmcore_plus._logger import logger as pymmcore_logger

    from shrimpy._logging import configure_logging

    # Replicate import-time: pymmcore-plus sets up its own handlers first
    pymmcore_configure_logging(file=None, log_to_stderr=False)

    # Then shrimpy configures logging, which should add a file handler without
    # disturbing the handlers already on the pymmcore-plus logger
    configure_logging(config_file, temp_log_dir, "test_acquisition")

    pymmcore_logger.info("Test message from pymmcore-plus after import")

    for handler in pymmcore_logger.handlers:
        handler.flush()

    log_dir = temp_log_dir / "logs"
    log_files = list(log_dir.glob("test_acquisition_log_*.log"))
    assert len(log_files) == 1
    assert "Test message from pymmcore-plus after import" in log_files[0].read_text()


def test_pymmcore_plus_own_handlers_preserved(config_file, temp_log_dir):
    """Test that shrimpy's configure_logging() adds to the pymmcore-plus logger
    without removing its own handlers (stderr, rotating log file).
    """
    from pymmcore_plus._logger import configure_logging as pymmcore_configure_logging
    from pymmcore_plus._logger import logger as pymmcore_logger

    from shrimpy._logging import configure_logging

    # Set up pymmcore-plus's own handlers (simulating import-time with stderr handler)
    pymmcore_configure_logging(file=None, log_to_stderr=True)
    handlers_before = list(pymmcore_logger.handlers)
    assert len(handlers_before) > 0

    configure_logging(config_file, temp_log_dir, "test_acquisition")

    # All original pymmcore-plus handlers must still be present
    for handler in handlers_before:
        assert handler in pymmcore_logger.handlers, (
            f"pymmcore-plus handler {handler!r} was removed by shrimpy configure_logging"
        )

    # The shrimpy file handler must be added on top, not in place of existing ones
    added_handlers = [h for h in pymmcore_logger.handlers if h not in handlers_before]
    assert len(added_handlers) == 1
    assert isinstance(added_handlers[0], logging.FileHandler)


def test_detailed_formatter(config_file, temp_log_dir):
    """Test that the detailed formatter includes module and function names."""
    from shrimpy._logging import configure_logging

    configure_logging(config_file, temp_log_dir, "test_acquisition")

    # Get a logger and write a message
    logger = logging.getLogger("shrimpy.mantis.mantis_engine")
    logger.info("Test message with detailed format")

    # Flush handlers
    for handler in logging.getLogger("shrimpy").handlers:
        handler.flush()

    # Check log file content
    log_dir = temp_log_dir / "logs"
    log_files = list(log_dir.glob("test_acquisition_log_*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()

    # Should include logger name, module, and function
    assert "shrimpy.mantis.mantis_engine" in log_content
    assert "test_mantis_logger" in log_content  # module name
    assert "test_detailed_formatter" in log_content  # function name
