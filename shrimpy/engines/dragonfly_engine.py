"""Acquisition engine for the Dragonfly microscope.

Inherits the shared behavior from
:class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing defaults,
hardware logging, autofocus dispatch and event skipping, hardware reset on
teardown, and ``acquire()``) and adds the Leica Adaptive Focus Control (AFC)
autofocus routine. See :mod:`shrimpy.engines.mantis_engine` for a more
elaborate subclass.

Unlike mantis, Dragonfly acquisitions are usually not hardware-sequenced, so
``setup_event`` mostly receives single events, one per Z slice; AFC therefore
engages at the bottom of each Z-stack rather than once per burst. Both paths
are handled by :meth:`BaseEngine._should_engage_autofocus`.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from shrimpy.engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from useq import MDAEvent

logger = logging.getLogger(__name__)


class DragonflyEngine(BaseEngine):
    """MDA engine for the Dragonfly microscope.

    Hooks left to implement:

    - ``setup_sequence`` / ``setup_event`` / ``teardown_sequence``: Dragonfly
      hardware setup around the corresponding ``super()`` call
    """

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage Leica AFC for ``event``.

        Called by :meth:`~shrimpy.engines.base_engine.BaseEngine._engage_autofocus`
        once per Z-stack (or once per burst, if the event was sequenced) when
        autofocus is enabled with a method other than ``demo-PFS``. The
        acquisition of events for which this returns False is skipped.
        """
        z_position = self._get_autofocus_z_position(event)
        return self._engage_leica_afc(self._autofocus_stage, z_position)

    def _engage_leica_afc(self, z_stage_name: str, z_position: float) -> bool:
        """Move the Z stage to ``z_position`` and run a full AFC focus.

        Parameters
        ----------
        z_stage_name : str
            The name of the z stage device which is moved before focusing.
        z_position : float
            The target position at which autofocus will be engaged.

        Returns
        -------
        bool
            True if the AFC call succeeded.
        """
        core = self.mmcore
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

        for z_offset in z_offsets:
            core.setPosition(z_stage_name, z_position + z_offset)
            core.waitForDevice(z_stage_name)

            try:
                core.fullFocus()
            except Exception:
                logger.debug(f"Autofocus failed to engage with Z offset of {z_offset} um")
                continue
            logger.debug(f"Autofocus engaged with Z offset of {z_offset} um")
            return True

        # return z stage to original position if autofocus attempts failed
        core.setPosition(z_stage_name, z_position)
        core.waitForDevice(z_stage_name)

        logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")
        return False
