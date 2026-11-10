"""Acquisition engine for the iSIM microscope.

Inherits the shared behavior from
:class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing
defaults, hardware logging, autofocus dispatch and event skipping, hardware
reset on teardown, and ``acquire()``). The only iSIM-specific step so far is
selecting the Hamamatsu ORCA-Quest camera and cropping it to the iSIM field of
view when the engine is created; see :meth:`ISIMEngine.__init__`.

Autofocus is not implemented yet: an acquisition runs only with autofocus
disabled or with the simulated ``demo-PFS`` method; any other method raises
``NotImplementedError`` from ``BaseEngine.engage_autofocus``.
"""

from __future__ import annotations

import logging

from shrimpy.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)

#: Device label of the Hamamatsu ORCA-Quest camera in the iSIM MM config.
ORCA_QUEST_CAMERA = "HamamatsuHam_DCAM"

#: ``(x, y, width, height)`` of the ORCA-Quest ROI covering the iSIM field of view.
ORCA_QUEST_ROI = (976, 684, 2048, 1500)


class ISIMEngine(BaseEngine):
    """MDA engine for the iSIM microscope.

    Hooks left to implement:

    - ``engage_autofocus(event) -> bool``: the hardware autofocus routine
    - ``setup_sequence`` / ``setup_event`` / ``teardown_sequence``: iSIM
      hardware setup around the corresponding ``super()`` call
    """

    def __init__(self, mmc, *args, **kwargs):
        super().__init__(mmc, *args, **kwargs)
        self._configure_orca_quest(mmc)

    @staticmethod
    def _configure_orca_quest(core) -> None:
        """Make the ORCA-Quest the active camera and apply the iSIM ROI.

        Done only if the camera is loaded, so the engine still works with other
        configs (e.g. the Micro-Manager demo config). ``setROI`` acts on the
        current camera, hence ``setCameraDevice`` first.
        """
        loaded_devices = core.getLoadedDevices()
        logger.debug(f"Loaded devices: {', '.join(loaded_devices)}")

        if ORCA_QUEST_CAMERA not in loaded_devices:
            logger.info(
                f"{ORCA_QUEST_CAMERA} is not loaded; leaving the camera and ROI unchanged"
            )
            return

        logger.info(f"Setting camera to {ORCA_QUEST_CAMERA} with ROI {ORCA_QUEST_ROI}")
        core.setCameraDevice(ORCA_QUEST_CAMERA)
        core.setROI(*ORCA_QUEST_ROI)
