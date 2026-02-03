"""shrimpy - Custom acquisition engines for optical microscopes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shrimpy")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
