import logging
import os

from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Union


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


# Setup console handler
def get_console_handler():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(module)s.%(funcName)s - %(message)s')
    console_handler.setFormatter(console_format)
    return console_handler


# Setup file handler
def get_file_handler(filename):
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    return file_handler


# Setup root logger
def configure_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(filename))
    logger.propagate = False
