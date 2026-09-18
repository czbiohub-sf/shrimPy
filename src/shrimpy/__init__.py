"""shrimpy - Custom acquisition engines for optical microscopes."""

import os

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shrimpy")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

# Set pymmcore-plus circular buffer memory to 2048 MB
if not os.environ.get("PYMM_BUFFER_SIZE_MB"):
    os.environ["PYMM_BUFFER_SIZE_MB"] = "2048"
