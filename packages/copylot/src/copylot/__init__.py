"""The photom subset of coPylot (https://github.com/czbiohub-sf/coPylot).

Vendored from coPylot main @ 0bcc3a3 (BSD-3-Clause; see the package LICENSE) and
maintained here now that coPylot is being archived. coPylot's module paths are
kept, so ``from copylot...`` imports resolve unchanged:

- ``copylot.hardware.lasers``: ``AbstractLaser``, ``vortran.vortran.VortranLaser``
- ``copylot.hardware.mirrors``: ``AbstractMirror``, ``optotune.mirror.OptoMirror``
  (Optotune's SDK comes from the ``optotune`` extra, not this package)
- ``copylot.assemblies.photom``: ``PhotomAssembly``, its ``utils``, simulated
  devices, and the Qt ``gui``

Everything else in coPylot (other devices, its other GUIs) is not vendored. Each
module's docstring lists its changes from coPylot.

Unlike coPylot, importing ``copylot`` attaches no console handler: modules log
to ``logging.getLogger(__name__)`` under the ``copylot`` logger, and the
application decides where that goes (shrimPy's ``configure_logging`` adds its
file handler). :func:`enable_logging` is kept for standalone use.
"""

import logging

__version__ = "0.1.0"

logger = logging.getLogger("copylot")
logger.addHandler(logging.NullHandler())


def enable_logging(log_filepath: str = "copylot_debug_log.txt", level=logging.DEBUG):
    """Log ``copylot`` messages to a text file.

    Parameters
    ----------
    log_filepath : str
        Path to the log file.
    level : int
        Logging level of the file handler.
    """
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
        )
    )
    logger.setLevel(min(logger.level or level, level))
    logger.addHandler(fh)
