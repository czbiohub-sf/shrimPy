"""Acquisition engine for the Dragonfly microscope.

Inherits the shared behavior from
:class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing defaults,
hardware logging, autofocus dispatch and event skipping, hardware reset on
teardown, and ``acquire()``) and adds the Leica Adaptive Focus Control (AFC)
autofocus routine. See :mod:`shrimpy.engines.mantis_engine` for a more
elaborate subclass.

Unlike mantis, Dragonfly acquisitions are usually not hardware-sequenced, so
``setup_event`` mostly receives single events, one per Z slice; AFC therefore
engages on arrival at each position rather than once per burst, and the lock
is reused for every channel and slice acquired there. Both paths are handled by
:meth:`BaseEngine._should_engage_autofocus`.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from shrimpy.engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from useq import MDAEvent

logger = logging.getLogger(__name__)

# Smallest focus-stage move worth commanding, in um. Below this the stage is
# treated as already at the target and no command is sent at all; see
# _move_focus_stage() for why a zero-distance move is not merely wasteful but
# hangs. 0.1 um is far below the depth of field of any objective on this scope,
# so skipping such a move cannot change where the sample is imaged.
MIN_FOCUS_MOVE_UM = 0.1


class DragonflyEngine(BaseEngine):
    """MDA engine for the Dragonfly microscope.

    Hooks left to implement:

    - ``setup_sequence`` / ``setup_event`` / ``teardown_sequence``: Dragonfly
      hardware setup around the corresponding ``super()`` call
    """

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage Leica AFC for ``event``.

        Called by :meth:`~shrimpy.engines.base_engine.BaseEngine._engage_autofocus`
        once per position (or once per burst, if the event was sequenced) when
        autofocus is enabled with a method other than ``demo-PFS``. The
        acquisition of events for which this returns False is skipped.
        """
        z_position = self._get_autofocus_z_position(event)
        return self._engage_leica_afc(self._autofocus_stage, z_position)

    def _move_focus_stage(self, stage: str, target: float) -> bool:
        """Move ``stage`` to ``target``, skipping moves below the step threshold.

        Returns True when the stage is at ``target`` — including when it was
        already there and no command was sent — and False when the move failed,
        in which case the caller should try the next Z offset.

        Moves shorter than ``MIN_FOCUS_MOVE_UM`` are skipped rather than
        commanded, because commanding a Leica FocusDrive to the position it
        already holds *hangs*. The LeicaDMI adapter derives ``Busy()`` from the
        asynchronous ``$71004`` z-drive motion flag: it raises the flag when it
        issues the move and clears it when the motor reports stopped. A move the
        firmware resolves to zero encoder steps never moves the motor, so
        ``$71004`` never arrives, ``Busy()`` stays True, and ``waitForDevice``
        throws after the 5 s core timeout.

        The drive works in integer encoder steps, so whether a nominally-zero
        move crosses a step boundary comes down to a few nm of encoder dither.
        That made it fail on roughly 2% of engagements while the other 98%
        cleared in 0.000 s — exactly the intermittent ``waitForDevice`` timeout
        this guard removes.
        """
        core = self.mmcore

        try:
            position_before = core.getPosition(stage)
        except Exception:
            logger.exception(f"Could not read {stage} position; commanding the move anyway")
            position_before = None

        if position_before is not None:
            # Rounded before comparing so that a distance of exactly the
            # threshold still moves: abs(100.1 - 100.0) is 0.09999999999999432
            # in binary floating point. 6 decimals of a um is far finer than any
            # stage resolution, so this only cancels representation error.
            distance = round(abs(target - position_before), 6)
            if distance < MIN_FOCUS_MOVE_UM:
                logger.debug(
                    f"{stage} is already on target: {distance:.4f} um is below the "
                    f"{MIN_FOCUS_MOVE_UM} um move threshold"
                )
                return True

        try:
            core.setPosition(stage, target)
            core.waitForDevice(stage)
        except Exception:
            logger.exception(f"Failed to move {stage} to {target}")
            return False
        return True

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
            if not self._move_focus_stage(z_stage_name, z_position + z_offset):
                continue

            try:
                core.fullFocus()
            except Exception:
                logger.debug(f"Autofocus failed to engage with Z offset of {z_offset} um")
                continue
            logger.debug(f"Autofocus engaged with Z offset of {z_offset} um")
            return True

        # return z stage to original position if autofocus attempts failed
        self._move_focus_stage(z_stage_name, z_position)

        logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")
        return False
