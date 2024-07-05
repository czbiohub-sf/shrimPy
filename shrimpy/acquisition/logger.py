import logging
import os

from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Union


def configure_debug_logger(filepath: str, logger_name: str = 'shrimpy'):
    """Add a file handler at DEBUG level for a given logger

    Parameters
    ----------
    filepath : str
        Debug log file path
    logger_name : str, optional
        Logger name, by default 'shrimpy'
    """
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logging.getLogger(logger_name).addHandler(file_handler)


def log_conda_environment(log_path: Union[str, os.PathLike]):
    """Save a log of the current conda environment as defined in the
    `compmicro-docs/hpc/log_environment.ps1` PowerShell script. A copy of the
    script is kept in a separate location, given in `log_script_path` to avoid
    unintended changes to the log format or changes to the PowerShell script
    path.

    Parameters
    ----------
    log_path : str or PathLike
        Path where the environment log file will be written

    Returns
    -------
    output : str
    errors : str
    """

    # Validate log_path input
    log_path = str(Path(log_path).absolute())
    assert log_path.endswith('.txt'), 'Log path must point to a .txt file'

    # get current conda environment
    conda_environment = os.environ['CONDA_DEFAULT_ENV']
    # define absolute path to log_environment.ps1 script
    log_script_path = str(Path.home() / "log_environment.ps1")

    # `pwsh` command launches PowerShell 7. Do not use `powershell` as it
    # launches PowerShell 6 which is not configured with conda
    # need to call `conda activate` to activate the correct conda environment,
    # otherwise a log of the `base` environment is written
    cmd = f"pwsh -Command conda activate {conda_environment}; {log_script_path} {log_path}"
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    output, errors = process.communicate()

    return output, errors
