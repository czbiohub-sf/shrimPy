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
    from pymmcore_plus.metadata import SummaryMetaV1
    from useq import MDAEvent, MDASequence

logger = logging.getLogger(__name__)


class DragonflyEngine(BaseEngine):
    """MDA engine for the Dragonfly microscope.

    Runs the whole sequence with the shutter held open (autoshutter off); see
    :meth:`setup_sequence`.

    Hooks left to implement:

    - ``setup_event``: Dragonfly hardware setup around the corresponding
      ``super()`` call
    """

    def __init__(self, mmc, *args, **kwargs):
        super().__init__(mmc, *args, **kwargs)
        # Autoshutter state to restore in teardown_sequence; None while no
        # sequence is running.
        self._autoshutter_to_restore: bool | None = None

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Open the shutter for the whole run, then run the shared setup.

        The shutter is controlled through hardware blanking both all channels.
        Setting the autoshutter shate must happen before
        ``super().setup_sequence()``.
        """
        core = self.mmcore
        self._autoshutter_to_restore = core.getAutoShutter()
        logger.info("Disabling autoshutter and opening the shutter for the sequence")
        core.setAutoShutter(False)
        core.setShutterOpen(True)

        return super().setup_sequence(sequence)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Close the shutter and restore autoshutter, then the shared teardown.
        """
        super().teardown_sequence(sequence)

        core = self.mmcore
        logger.debug("Closing the shutter and restoring autoshutter")
        core.setShutterOpen(False)
        if self._autoshutter_to_restore is not None:
            core.setAutoShutter(self._autoshutter_to_restore)
            self._autoshutter_to_restore = None

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

        # Check if autofocus is already engaged
        if core.isContinuousFocusLocked():
            logger.debug("Continuous autofocus is already engaged")
            return True

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
