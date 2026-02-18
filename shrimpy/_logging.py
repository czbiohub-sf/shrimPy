import logging
import logging.config
import os

from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen


def configure_logging(
    config_file: Path,
    output_dir: Path,
    name: str,
) -> Path:
    """Configure logging from config file.

    Parameters
    ----------
    config_file : Path
        Path to logging configuration INI file.
    output_dir : Path
        Output directory where logs will be saved.
    name : str
        Acquisition name used for log file naming.

    Returns
    -------
    Path
        Path to log file.
    """

    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path with timestamp (matching original convention)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    log_file = log_dir / f"{name}_log_{timestamp}.txt"

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

    return log_file


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

        return output.decode("ascii").strip(), errors.decode("ascii").strip()

    return None, None
