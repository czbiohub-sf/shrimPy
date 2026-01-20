"""Mantis logging configuration for V2 acquisition engine.

This module configures logging for the mantis V2 acquisition engine:
- Creates a separate 'mantis' logger with console (INFO) and file (DEBUG) handlers
- Captures pymmcore-plus logger events to the same log file
- Does not modify pymmcore-plus logger handlers
"""

import logging
import os

from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Union

from pymmcore_plus._logger import logger as pymmcore_logger


def configure_mantis_logger(
    save_dir: Union[str, os.PathLike],
    acquisition_name: str = 'mantis_acquisition',
) -> logging.Logger:
    """Configure the mantis logger with console and file handlers.

    Creates a separate 'mantis' logger that logs:
    - INFO level messages to console
    - DEBUG level messages to a log file in the save directory

    Also adds a file handler to pymmcore-plus logger to capture its events.

    Parameters
    ----------
    save_dir : str or PathLike
        Directory where the log file will be saved
    acquisition_name : str, optional
        Name of the acquisition, used in the log filename

    Returns
    -------
    logging.Logger
        Configured mantis logger instance
    """
    # Create logs directory
    log_dir = Path(save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    log_filename = f'{acquisition_name}_log_{timestamp}.txt'
    log_path = log_dir / log_filename

    # Get or create the mantis logger
    mantis_logger = logging.getLogger('mantis')
    mantis_logger.setLevel(logging.DEBUG)
    mantis_logger.propagate = False

    # Clear any existing handlers to avoid duplicates
    mantis_logger.handlers.clear()

    # Configure console handler (INFO level) for mantis logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(module)s.%(funcName)s - %(message)s')
    console_handler.setFormatter(console_format)
    mantis_logger.addHandler(console_handler)

    # Configure file handler (DEBUG level) for mantis logger
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s.%(module)s.%(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    mantis_logger.addHandler(file_handler)

    # Add file handler to pymmcore-plus logger to capture its events
    # Create a separate handler instance for pymmcore logger
    pymmcore_file_handler = logging.FileHandler(log_path)
    pymmcore_file_handler.setLevel(logging.DEBUG)
    pymmcore_file_handler.setFormatter(file_format)
    pymmcore_logger.addHandler(pymmcore_file_handler)

    mantis_logger.info(f'Mantis acquisition log initialized at: {log_path}')

    return mantis_logger


def log_conda_environment(
    log_dir: Union[str, os.PathLike], logger: logging.Logger = None
) -> tuple[bytes, bytes]:
    """Save a log of the current conda environment.

    This function saves the conda environment information to a text file
    in the logs directory, matching the V1 behavior.

    Parameters
    ----------
    log_dir : str or PathLike
        Directory where the log file will be saved (typically the logs subdirectory)
    logger : logging.Logger, optional
        Logger to use for logging messages. If None, uses mantis logger.

    Returns
    -------
    output : bytes
        Standard output from the logging process
    errors : bytes
        Standard error from the logging process
    """
    if logger is None:
        logger = get_mantis_logger()

    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    log_filename = f'conda_environment_log_{timestamp}.txt'
    log_path = log_dir / log_filename

    # Get current conda environment
    try:
        conda_environment = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    except KeyError:
        logger.warning('CONDA_DEFAULT_ENV not found in environment variables')
        return b'', b'No conda environment found'

    # Define absolute path to log_environment.ps1 script
    log_script_path = Path.home() / "log_environment.ps1"

    if not log_script_path.exists():
        logger.warning(f'log_environment.ps1 script not found at {log_script_path}')
        return b'', b'log_environment.ps1 script not found'

    # `pwsh` command launches PowerShell 7
    cmd = f"pwsh -Command conda activate {conda_environment}; {log_script_path} {log_path}"

    try:
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
        output, errors = process.communicate()
        logger.debug(f'Conda environment log saved at: {log_path}')
        return output, errors
    except Exception as e:
        logger.error(f'Failed to log conda environment: {e}')
        return b'', str(e).encode()


def get_mantis_logger() -> logging.Logger:
    """Get the mantis logger instance.

    Returns the 'mantis' logger. If not yet configured, returns a basic
    logger with console output only.

    Returns
    -------
    logging.Logger
        The mantis logger
    """
    return logging.getLogger('mantis')
