from __future__ import annotations

import logging
import time

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.metadata import SummaryMetaV1
from useq import MDAEvent, MDASequence

from shrimpy.config import ShrimpyMetadata
from shrimpy.dynatrack import DynaTrack
from shrimpy.engines.base_engine import BaseEngine

# Get the logger instance (will be configured by the CLI entry point)
logger = logging.getLogger(__name__)

MANTIS_XY_STAGE_NAME = "XYStage:XY:31"
SLOW_XY_STAGE_SPEED = 2.0  # in mm/s, used for short moves to maintain autofocus lock
FAST_XY_STAGE_SPEED = 5.75  # in mm/s, used for long moves
NEGLIGIBLE_XY_DISTANCE = 1  # in um, moves below this are ignored
SHORT_XY_DISTANCE = 2000  # in um, threshold between slow and fast speed


def _find_shrimpy_log_file() -> Path | None:
    """Return the path of the FileHandler attached to the shrimpy logger."""
    for handler in logging.getLogger("shrimpy").handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


class MantisEngine(BaseEngine):
    """Custom MDA engine for the Mantis microscope.

    This engine extends :class:`~shrimpy.engines.base_engine.BaseEngine` with
    mantis-specific hardware setup and configuration, including:
    - Acquisition timeouts tuned for hardware-sequenced acquisition
    - Nikon PFS continuous autofocus
    - XY stage speed modulation
    - DynaTrack position tracking
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
        self._dynatrack: DynaTrack | None = None

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup mantis-specific hardware before the sequence starts.

        The microscope settings are read from ``sequence.metadata`` and
        validated by :class:`~shrimpy.config.ShrimpyMetadata`; missing sections
        fall back to their defaults (autofocus and DynaTrack disabled).
        """
        # Configure the shared hardware settings (autofocus, XY stage) and call
        # the parent setup so SummaryMetaV1 captures the fully configured
        # hardware state and the setup event applies the ROI.
        result = super().setup_sequence(sequence)

        core = self.mmcore
        meta = ShrimpyMetadata.from_sequence(sequence)

        # Setup DynaTrack position tracking. The XY pixel size (from the core)
        # and the sequence z_plan step are the single source of truth for all
        # scale parameters; DynaTrack derives and injects them. Built after the
        # parent setup so the pixel size and any grid-plan FOV sizes reflect the
        # state the setup event leaves the hardware in.
        self._dynatrack = DynaTrack.from_config(
            meta.dynatrack,
            sequence,
            data_path=self._data_path,
            pixel_size_um=core.getPixelSizeUm(),
        )
        if self._dynatrack is not None:
            core.mda.events.frameReady.connect(self._dynatrack.on_frame_ready)
            cfg = self._dynatrack.config
            preprocessing = cfg.preprocessing or ["none"]
            logger.info(
                "DynaTrack enabled: "
                f"input_channel={cfg.input_channel} -> tracking_channel={cfg.tracking_channel}, "
                f"preprocessing=[{', '.join(preprocessing)}], "
                f"tracking_method={cfg.tracking_method}, "
                f"tracking_interval={cfg.tracking_interval}, "
                f"reference_update_interval={cfg.reference_update_interval}"
            )

            # DynaTrack runs in a worker subprocess for GPU/torch isolation:
            # torch's OpenMP runtime segfaults when it coexists with the sequenced
            # camera readout in the acquisition process. The worker is started after
            # the setup event has applied the ROI, so getImageHeight/Width reflects
            # the actual acquired frame size (also used to build the preprocessor,
            # when configured, inside the worker).
            zyx_shape = (
                max(sequence.sizes.get("z", 1), 1),
                core.getImageHeight(),
                core.getImageWidth(),
            )
            self._dynatrack.start(zyx_shape=zyx_shape, log_file_path=_find_shrimpy_log_file())

        return result

    def event_iterator(self, events: Iterable[MDAEvent]):
        """Wrap event iteration to apply position updates before logging.

        By applying position updates here (before the MDA runner emits
        ``eventStarted``), the logged event reflects the corrected
        coordinates rather than the original sequence values.

        At timepoint boundaries the iterator drains any pending DynaTrack
        update so that (a) position corrections are applied before the new
        timepoint starts and (b) frame data does not accumulate unboundedly
        in the executor queue.
        """
        last_t: int | None = None
        for event in super().event_iterator(events):
            if self._dynatrack is not None:
                idx = (
                    event.events[0].index if isinstance(event, SequencedEvent) else event.index
                )
                t_idx = idx.get("t", 0)
                if last_t is not None and t_idx != last_t:
                    self._dynatrack.drain_pending()
                last_t = t_idx
                event = self._dynatrack.apply_position_update(event)
            yield event

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # TODO: debug resetting xy stage speed
        # self._adjust_xy_stage_speed(event)

        # Move the XY stage, engage autofocus, and prepare the shared hardware
        super().setup_event(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        # DynaTrack: disconnect callback and shutdown
        if self._dynatrack is not None:
            self.mmcore.mda.events.frameReady.disconnect(self._dynatrack.on_frame_ready)
            self._dynatrack.shutdown()
            self._dynatrack = None

        super().teardown_sequence(sequence)

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
