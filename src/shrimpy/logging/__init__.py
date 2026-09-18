"""Logging configuration for shrimPy acquisitions.

The INI file that drives it (``logging.ini``) sits beside this module as package data,
so it resolves the same way from a source checkout and from an installed wheel.

Absolute imports are the default in Python 3, so the ``logging`` imported below is the
standard library module, not this package.
"""

from __future__ import annotations

import logging
import logging.config
import os

from datetime import datetime
from importlib.resources import files
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

# ``files()`` returns a Traversable; shrimPy is never installed zipped (it drives local
# hardware), so this is always a real filesystem path and can be handed straight to
# ``fileConfig``.
DEFAULT_LOGGING_CONFIG = Path(str(files("shrimpy.logging") / "logging.ini"))


class _IgnorePropertyChangedWarnings(logging.Filter):
    """Suppress noisy warnings from pymmcore-plus when a device property cannot
    be read before emitting a propertyChanged signal.

    These are benign for TriggerScope TTL/DAC channels that do not expose
    'State' or 'Label' properties, and clutter the log with WARNING-level
    noise on every hardware-sequenced event.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.funcName != "_property_change_emission_ensured"


def configure_logging(
    output_dir: Path,
    name: str,
    config_file: Path | None = None,
) -> Path:
    """Configure logging from config file.

    Parameters
    ----------
    output_dir : Path
        Output directory where logs will be saved.
    name : str
        Acquisition name used for log file naming.
    config_file : Path, optional
        Path to a logging configuration INI file. Defaults to the packaged
        :data:`DEFAULT_LOGGING_CONFIG`. If the file does not exist, falls back to a
        basic INFO-level console + file configuration rather than failing: losing
        the preferred log format must never stop an acquisition from starting.

    Returns
    -------
    Path
        Path to log file.
    """

    if config_file is None:
        config_file = DEFAULT_LOGGING_CONFIG

    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path with timestamp (matching original convention)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    log_file = log_dir / f"{name}_log_{timestamp}.log"

    if config_file.exists():
        # Load logging configuration from file
        # Use forward slashes for cross-platform compatibility (avoids Windows backslash issues)
        logging.config.fileConfig(
            config_file,
            defaults={"log_file": log_file.as_posix()},
            disable_existing_loggers=False,
        )
        file_handler = next(
            (
                h
                for h in logging.getLogger("shrimpy").handlers
                if isinstance(h, logging.FileHandler)
            ),
            None,
        )
    else:
        # Fallback to basic config if config file not found
        file_handler = logging.FileHandler(log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                file_handler,
            ],
        )

    # Add the shrimpy file handler to the pymmcore-plus logger without touching its
    # own handlers (stderr output, rotating log file to pymmcore-plus's data dir).
    if file_handler is not None:
        pymmcore_logger = logging.getLogger("pymmcore-plus")
        pymmcore_logger.setLevel(logging.DEBUG)
        pymmcore_logger.addHandler(file_handler)
        pymmcore_logger.addFilter(_IgnorePropertyChangedWarnings())

    return log_file


def find_log_file() -> Path | None:
    """Return the path of the FileHandler attached to the shrimpy logger.

    Used to point subprocesses (e.g. the DynaTrack worker) at the acquisition's
    log file so their messages land alongside the engine's.
    """
    for handler in logging.getLogger("shrimpy").handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def log_conda_environment(log_dir: Path) -> tuple[bytes | None, bytes | None]:
    """Log current conda environment information to a file.

    Creates a log file with conda environment details including installed packages
    and their versions. Uses `conda list` to capture the environment state.

    Parameters
    ----------
    log_dir : Path
        Directory where the environment log file will be written.

    Returns
    -------
    tuple[bytes | None, bytes | None]
        A tuple of (stdout, stderr) from the conda list command.
        stdout contains the environment information if successful.
        stderr contains any error messages if the command failed.
    """
    # Get current conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    # define absolute path to log_environment.ps1 script
    log_script_path = Path.home() / "log_environment.ps1"

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    log_file = log_dir / f"conda_env_{conda_env}_{timestamp}.txt"

    # `pwsh` command launches PowerShell 7. Do not use `powershell` as it
    # launches PowerShell 6 which is not configured with conda
    # need to call `conda activate` to activate the correct conda environment,
    # otherwise a log of the `base` environment is written
    if conda_env and log_script_path.exists():
        cmd = f"pwsh -Command conda activate {conda_env}; {log_script_path} {log_file}"
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
        output, errors = process.communicate()

        if output:
            output = output.decode("ascii").strip()
        if errors:
            errors = errors.decode("ascii").strip()

        return output, errors

    return None, None
