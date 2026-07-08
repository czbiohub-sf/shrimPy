"""Engine-facing coordinator for online FOV selection.

Mirrors ``shrimpy.dynatrack.manager.DynaTrack``: built from the acquisition
metadata via :meth:`FovSelection.from_metadata`, it turns one pre-scan FOV's
brightfield z-stack into a good/bad verdict:

    BF z-stack -> reconstruct (deskew -> phase -> virtual stain)   [preprocessing.py]
               -> project -> segment -> features -> tree predict   [pipeline.py]

The heavy objects (reconstruction preprocessor, Cellpose model, trained tree)
are built lazily on the first FOV (``_ensure_built``), since the reconstruction
transfer function needs the acquired ZYX shape.

Config lives under ``metadata.mantis.fov_selection`` and maps onto
:class:`FovSelectionConfig`. Scale parameters (XY pixel size, Z step) are the
single source of truth injected into the deskew/phase sub-configs -- as
DynaTrack does -- so they are not duplicated in the config.
"""

from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING

import numpy as np

from shrimpy.fov_selection import pipeline as P
from shrimpy.fov_selection.preprocessing import build_preprocessor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Channels the decision needs (segmented + fed to the model).
DEFAULT_TARGET_CHANNELS = ["nuclei", "membrane"]


class FovSelection:
    """Coordinates the online FOV-selection decision for one acquisition.

    Parameters
    ----------
    config : dict
        The ``fov_selection`` metadata block.
    pixel_size_um : float
        XY pixel size (microns) -- injected into deskew/phase and used for
        physical feature units.
    z_step_um : float
        Z step (microns) of the pre-scan z_plan -- injected into deskew
        (``scan_step_um``) and phase (``z_pixel_size``).
    """

    def __init__(self, config: dict, pixel_size_um: float, z_step_um: float) -> None:
        self.config = config
        self._pixel_size_um = pixel_size_um
        self._z_step_um = z_step_um
        self._input_channel = config.get("input_channel", "BF - Oblique")
        self._projection = config.get("projection", "sum")
        self._threshold = float(config.get("threshold", 0.5))
        self._target_channels = list(
            (config.get("reconstruction", {}).get("virtual_staining", {}) or {}).get(
                "target_channels", DEFAULT_TARGET_CHANNELS
            )
        )
        # Lazily built (need the acquired ZYX shape for the transfer function).
        self._preprocessor = None
        self._cellpose = None
        self._model = None

    @classmethod
    def from_metadata(
        cls,
        meta: dict | None,
        pixel_size_um: float,
        z_step_um: float,
    ) -> FovSelection | None:
        """Build the coordinator from the ``fov_selection`` metadata block.

        Returns ``None`` when FOV selection is disabled.
        """
        if not meta or not meta.get("enabled", False):
            return None
        if not meta.get("model_path"):
            raise ValueError("fov_selection.model_path is required when enabled")
        return cls(meta, pixel_size_um=pixel_size_um, z_step_um=z_step_um)

    def _inject_scales(self, recon: dict) -> dict:
        """Inject XY pixel size / Z step into the deskew and phase sub-configs.

        Single source of truth (as DynaTrack does), so the pixel/step values are
        never duplicated in the config and cannot drift.
        """
        recon = copy.deepcopy(recon)
        deskew = recon.get("deskew")
        if deskew is not None:
            deskew["pixel_size_um"] = self._pixel_size_um
            deskew["scan_step_um"] = self._z_step_um
        phase = recon.get("phase")
        if phase is not None:
            tf = phase.setdefault("transfer_function", {})
            tf["yx_pixel_size"] = self._pixel_size_um
            tf["z_pixel_size"] = self._z_step_um
        return recon

    def _ensure_built(self, zyx_shape: tuple[int, int, int]) -> None:
        """Build the preprocessor, Cellpose model, and tree on the first FOV."""
        if self._model is not None:
            return
        recon = self._inject_scales(self.config.get("reconstruction", {}) or {})
        self._preprocessor = build_preprocessor(
            zyx_shape=zyx_shape,
            preprocessing=recon.get("preprocessing"),
            deskew=recon.get("deskew"),
            phase=recon.get("phase"),
            virtual_staining=recon.get("virtual_staining"),
        )
        if self._preprocessor is None:
            raise ValueError(
                "fov_selection.reconstruction produced no preprocessor; a "
                "'deskew'/'phase'/'vs' pipeline is required to make nuclei/membrane."
            )
        self._cellpose = P.load_cellpose_model(
            model_name=self.config.get("cellpose_model"), gpu=True
        )
        self._model = P.load_fov_model(self.config["model_path"])
        logger.info("FOV selection: reconstruction + Cellpose + tree ready")

    def _to_numpy(self, x) -> np.ndarray:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def decide(self, bf_zyx: np.ndarray) -> tuple[float, bool]:
        """Reconstruct one BF z-stack and return ``(proba_good, is_good)``."""
        bf_zyx = np.asarray(bf_zyx)
        self._ensure_built(tuple(bf_zyx.shape))

        channels = self._preprocessor(bf_zyx)  # {'nuclei', 'membrane', 'phase'}
        projections, masks = {}, {}
        for organelle in self._target_channels:
            vol = self._to_numpy(channels[organelle])
            proj = P.project_zyx(vol, self._projection)
            projections[organelle] = proj
            masks[organelle] = P.segment_2d(proj, self._cellpose, organelle)

        matrix = P.fov_feature_matrix(
            projections, masks, self._pixel_size_um, self._projection, source="vs"
        )
        proba, good = P.predict_good(self._model, matrix, self._threshold)
        return float(proba[0]), bool(good[0])

    def select(self, prescan_store_path, position_names) -> list[str]:
        """Decide each named position in the pre-scan store; return good names.

        Positions are opened **by explicit path** (``<store>/<name>``) rather than
        via the plate's ``positions()`` enumeration: the acquisition write path
        can leave the HCS well ``images`` metadata incomplete, so ``positions()``
        finds nothing even though the arrays are present. The caller (controller)
        already knows the acquired position names. See docs/replay_camera_todo.md.
        """
        from iohub.ngff import open_ome_zarr

        store = str(prescan_store_path).rstrip("/")
        good: list[str] = []
        for name in position_names:
            pos = open_ome_zarr(f"{store}/{name}", mode="r")
            chans = list(pos.channel_names)
            if self._input_channel not in chans:
                raise ValueError(
                    f"input_channel {self._input_channel!r} not in {chans} at {name}"
                )
            ci = chans.index(self._input_channel)
            bf_zyx = np.asarray(pos.data[0, ci])  # (Z, Y, X)
            proba, is_good = self.decide(bf_zyx)
            logger.info(
                "FOV selection: %s -> proba=%.3f %s",
                name,
                proba,
                "GOOD" if is_good else "bad",
            )
            if is_good:
                good.append(name)
        logger.info("FOV selection: %d/%d positions kept", len(good), len(position_names))
        return good
