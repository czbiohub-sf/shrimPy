"""Placeholder acquisition engine for the iSIM microscope.

Not implemented yet — this is a skeleton that inherits the shared behavior
from :class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing
defaults, hardware logging, autofocus dispatch and event skipping, hardware
reset on teardown, and ``acquire()``), and marks the hooks an iSIM
implementation is expected to fill in. See
:mod:`shrimpy.engines.mantis_engine` for a worked example.

As it stands, an acquisition runs only with autofocus disabled or with the
simulated ``demo-PFS`` method; any other method raises ``NotImplementedError``
from ``BaseEngine.engage_autofocus``.
"""

from __future__ import annotations

import logging

from shrimpy.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class ISIMEngine(BaseEngine):
    """MDA engine for the iSIM microscope (placeholder).

    Hooks to implement:

    - ``__init__``: microscope-specific defaults (e.g. acquisition timeouts)
      via ``kwargs.setdefault(...)`` before ``super().__init__()``
    - ``engage_autofocus(event) -> bool``: the hardware autofocus routine
    - ``setup_sequence`` / ``setup_event`` / ``teardown_sequence``: iSIM
      hardware setup around the corresponding ``super()`` call
    """
