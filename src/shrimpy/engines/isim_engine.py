"""Acquisition engine for the iSIM microscope.

Inherits the shared behavior from
:class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing
defaults, hardware logging, autofocus dispatch and event skipping, hardware
reset on teardown, and ``acquire()``). The iSIM-specific steps so far are
selecting the Hamamatsu ORCA-Quest camera and cropping it to the iSIM field of
view when the engine is created (see :meth:`ISIMEngine.__init__`), and
:class:`ISIMCore`, which works around Micro-Manager reporting a pixel size of 0
for the iSIM's Multi Camera.

Autofocus is not implemented yet: an acquisition runs only with autofocus
disabled or with the simulated ``demo-PFS`` method; any other method raises
``NotImplementedError`` from ``BaseEngine.engage_autofocus``.
"""

from __future__ import annotations

import logging

from shrimpy.engines.base_engine import BaseEngine
from shrimpy.robust_cmmcore import RobustCMMCore

logger = logging.getLogger(__name__)

#: Device label of the Hamamatsu ORCA-Quest camera in the iSIM MM config.
ORCA_QUEST_CAMERA = "HamamatsuHam_DCAM"

#: ``(x, y, width, height)`` of the ORCA-Quest ROI covering the iSIM field of view.
ORCA_QUEST_ROI = (976, 684, 2048, 1500)


class ISIMCore(RobustCMMCore):
    """Core for the iSIM: :class:`RobustCMMCore` plus a Multi Camera pixel-size fix.

    The iSIM's Multi Camera pairs the ORCA-Quest (binning 1) with the Oryx, which
    is binned 2x to match its image size. MMCore scales the matching pixel-size
    config by the current camera's binning, and the Multi Camera adapter reports a
    binning of 0 when its physical cameras' binnings differ
    (``MultiCamera::GetBinning``), so ``getPixelSizeUm()`` returns 0 and
    ``getPixelSizeAffine()`` all zeros, although the config defines a size.

    In that case this returns the pixel size and affine of the matching pixel-size
    config as defined (e.g. ``Core,Camera=Multi Camera`` with the objective), with
    no scaling. Everything else is unchanged.
    """

    _mixed_binning_logged = False

    def getPixelSizeUm(self, cached: bool = False) -> float:
        pixel_size = super().getPixelSizeUm(cached)
        if pixel_size == 0.0 and (config := self._mixed_binning_config(cached)):
            return self.getPixelSizeUmByID(config)
        return pixel_size

    def getPixelSizeAffine(self, cached: bool = False) -> tuple[float, ...]:
        affine = tuple(super().getPixelSizeAffine(cached))
        if not any(affine) and (config := self._mixed_binning_config(cached)):
            return tuple(self.getPixelSizeAffineByID(config))
        return affine

    def _mixed_binning_config(self, cached: bool) -> str | None:
        """Return the matching pixel-size config, if the current camera is a Multi
        Camera whose physical cameras mix binning (its reported binning is 0).

        Calls that may raise go through ``super()``, bypassing RobustCMMCore's
        retries: this runs for every frame's metadata.
        """
        camera = self.getCameraDevice()
        if not camera or self.getNumberOfCameraChannels() < 2:
            return None
        try:
            if super().getProperty(camera, "Binning") != "0":
                return None
            config = super().getCurrentPixelSizeConfig(cached)
        except RuntimeError:
            return None
        if not config:
            return None

        if not self._mixed_binning_logged:
            self._mixed_binning_logged = True
            logger.warning(
                f"{camera} mixes binning across its cameras, so Micro-Manager reports "
                f"a pixel size of 0; using pixel-size config {config!r} as defined "
                f"({self.getPixelSizeUmByID(config)} um/px)"
            )
        return config


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
