from __future__ import annotations

import logging
import time

import numpy as np

from pymmcore_plus.core import CMMCorePlus
from useq import MDAEvent

from shrimpy.engines.base_engine import BaseEngine

# Get the logger instance (will be configured by the CLI entry point)
logger = logging.getLogger(__name__)

MANTIS_XY_STAGE_NAME = "XYStage:XY:31"
SLOW_XY_STAGE_SPEED = 2.0  # in mm/s, used for short moves to maintain autofocus lock
FAST_XY_STAGE_SPEED = 5.75  # in mm/s, used for long moves
NEGLIGIBLE_XY_DISTANCE = 1  # in um, moves below this are ignored
SHORT_XY_DISTANCE = 2000  # in um, threshold between slow and fast speed


class MantisEngine(BaseEngine):
    """Custom MDA engine for the Mantis microscope.

    This engine extends :class:`~shrimpy.engines.base_engine.BaseEngine` with
    mantis-specific hardware setup and configuration, including:
    - Acquisition timeouts tuned for hardware-sequenced acquisition
    - Nikon PFS continuous autofocus
    - XY stage speed modulation

    DynaTrack position tracking is shared by all engines and lives in
    :class:`~shrimpy.engines.base_engine.BaseEngine`; mantis enables it from
    ``metadata.dynatrack`` like any other microscope.
    """

    def __init__(self, mmc: CMMCorePlus, *args, **kwargs):
        """Initialize and register the MantisEngine with the core.

        Parameters
        ----------
        mmc : CMMCorePlus
            The Micro-Manager core instance
        """
        # Set acquisition timeout to guard against stalling due to dropped frames
        # or missed trigger pulses
        kwargs.setdefault("timeout_base", 10.0)
        kwargs.setdefault("timeout_multiplier", 1.0)
        kwargs.setdefault("timeout_first_frame", None)
        kwargs.setdefault("timeout_action", "warn")
        super().__init__(mmc, *args, **kwargs)
        self._xy_stage_speed = None

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare mantis hardware for each event.

        Mantis acquires with hardware sequencing, so ``event`` is normally a
        ``SequencedEvent`` covering a whole Z-stack and PFS is engaged once per
        burst by :meth:`BaseEngine.setup_event`.
        """
        # TODO: debug resetting xy stage speed
        # self._adjust_xy_stage_speed(event)

        # Move the XY stage, engage autofocus, and prepare the shared hardware
        super().setup_event(event)

    def _adjust_xy_stage_speed(self, event: MDAEvent) -> None:
        """Modulate XY stage speed based on distance to target position.

        This method adjusts the XY stage speed before moving to a new position,
        using a slower speed for short moves to help maintain autofocus lock.

        Parameters
        ----------
        event : MDAEvent
            The MDA event containing the target XY position.
        """
        if not self._use_autofocus or not self._xy_stage_device:
            return

        # Only adjust speed for Mantis XY stage, not demo XY stage
        if self._xy_stage_device != MANTIS_XY_STAGE_NAME:
            return

        last_x, last_y = self.mmcore._last_xy_position.get(None) or (None, None)
        target_x, target_y = event.x_pos, event.y_pos

        if not all(v is not None for v in [last_x, last_y, target_x, target_y]):
            return

        distance = np.linalg.norm([target_x - last_x, target_y - last_y])
        # If the move is negligible, skip speed adjustment
        if distance < NEGLIGIBLE_XY_DISTANCE:
            return

        speed = SLOW_XY_STAGE_SPEED if distance < SHORT_XY_DISTANCE else FAST_XY_STAGE_SPEED

        # If the speed is already set appropriately, no need to update
        if self._xy_stage_speed == speed:
            return

        self.mmcore.setProperty(self._xy_stage_device, "MotorSpeedX-S(mm/s)", speed)
        self.mmcore.setProperty(self._xy_stage_device, "MotorSpeedY-S(mm/s)", speed)

        self._xy_stage_speed = speed
        logger.debug(f"Set stage speed to {speed} mm/s")

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage Nikon PFS continuous autofocus for ``event``.

        Called by :meth:`~shrimpy.engines.base_engine.BaseEngine._engage_autofocus`
        when autofocus is enabled with a method other than ``demo-PFS``. The
        acquisition of events for which this returns False is skipped.
        """
        z_position = self._get_autofocus_z_position(event)
        return self._engage_nikon_pfs(self._autofocus_stage, z_position)

    def _engage_nikon_pfs(self, z_stage_name: str, z_position: float) -> bool:
        """
        Attempt to engage Nikon PFS continuous autofocus. This function will log a
        message and continue if continuous autofocus is already engaged. Otherwise,
        it will attempt to engage autofocus, moving the z stage by amounts given in
        `z_offsets`, if necessary.

        Parameters
        ----------
        z_stage_name : str
            The name of the z stage device which will be moved to help engage autofocus.
        z_position : float
            The target position at which autofocus will be engaged.

        Returns
        -------
        bool
            True if continuous autofocus is engaged.
        """
        core = self.mmcore
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

        # Turn on autofocus if it has been turned off. This call has no effect is
        # continuous autofocus is already engaged
        try:
            core.fullFocus()
            time.sleep(0.2)  # needed before we can call isContinuousFocusLocked()
            logger.debug("Call to fullFocus() succeeded")
        except Exception:
            logger.debug("Call to fullFocus() failed")
            # Wait for viscous immersion oil to catch up, usually needed when switching wells
            time.sleep(5)

        # Check if autofocus is already engaged
        if core.isContinuousFocusLocked():
            logger.debug("Continuous autofocus is already engaged")
            return True

        for z_offset in z_offsets:
            core.setPosition(z_stage_name, z_position + z_offset)
            core.waitForDevice(z_stage_name)

            # This call engages autofocus
            core.enableContinuousFocus(True)
            time.sleep(1)  # Wait for autofocus to engage

            if core.isContinuousFocusLocked():
                logger.debug(f"Continuous autofocus engaged with Z offset of {z_offset} um")
                return True
            else:
                logger.debug(f"Autofocus call failed with Z offset of {z_offset} um")

        # return z stage to original position if autofocus attempts failed
        core.setPosition(z_stage_name, z_position)
        core.waitForDevice(z_stage_name)

        logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")
        return False
