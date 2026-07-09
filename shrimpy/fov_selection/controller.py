"""Two-phase adaptive-acquisition controller for FOV selection.

Orchestrates the online FOV-selection experiment as ONE user experiment made of
two internal acquisition phases:

  1. pre-scan  : acquire every candidate position once (single timepoint), using
                 the pre-scan parameters (default: same as the timelapse, but
                 typically the brightfield channel only).
  2. decision  : choose the "good" positions. If the config supplies a
                 ``model_path`` (+ reconstruction), the real ``FovSelection``
                 manager runs reconstruct -> project -> segment -> features ->
                 tree per FOV; otherwise a dummy selector is used (M1).
  3. timelapse : the full acquisition, restricted to the selected positions.

The user sets this up as a single experiment (one config, one launch); the two
acquisition phases happen automatically and are invisible to the user.

Dependency direction (Option C): this controller depends on the engine and the
fov_selection package; the engine depends on NOTHING here and is NOT modified
for FOV selection. Each phase runs through the engine's existing public
``acquire(output_dir, name, mda_config)`` method.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

from shrimpy.fov_selection.manager import FovSelection
from shrimpy.fov_selection.online_selection import select_good_fovs

if TYPE_CHECKING:
    from useq import MDASequence

    from shrimpy.mantis.mantis_engine import MantisEngine

logger = logging.getLogger(__name__)


class FovSelectionAcquisition:
    """Drives a pre-scan -> select -> timelapse acquisition on one engine.

    Parameters
    ----------
    engine : MantisEngine
        An acquisition engine exposing ``acquire(output_dir, name, mda_config)``.
    """

    def __init__(self, engine: MantisEngine) -> None:
        self._engine = engine

    def run(self, sequence: MDASequence, output_dir: str | Path, name: str) -> None:
        """Run the two-phase FOV-selection acquisition from one config."""
        output_dir = Path(output_dir)
        fov_cfg = (sequence.metadata.get("mantis", {}) or {}).get("fov_selection", {})

        # --- Phase 1: pre-scan (single timepoint over all candidate positions)
        prescan = self._build_prescan(sequence, fov_cfg)
        logger.info(
            "FOV selection: pre-scan of %d positions, channels=%s",
            len(prescan.stage_positions),
            [c.config for c in prescan.channels],
        )
        prescan_name = f"{name}_prescan"
        self._engine.acquire(output_dir=output_dir, name=prescan_name, mda_config=prescan)

        # --- Decision: choose the good FOVs ---------------------------------
        good = self._select(sequence, fov_cfg, output_dir, prescan_name)
        if not good:
            logger.warning("FOV selection kept no positions; skipping timelapse.")
            return

        # --- Phase 2: timelapse over the selected positions (full config) ---
        timelapse = sequence.replace(stage_positions=good)
        logger.info("FOV selection: timelapse of %d selected positions", len(good))
        self._engine.acquire(output_dir=output_dir, name=name, mda_config=timelapse)

    def _select(self, sequence, fov_cfg, output_dir, prescan_name):
        """Return the good subset of ``sequence.stage_positions``.

        Uses the real ``FovSelection`` manager when a ``model_path`` is
        configured (reads the just-written pre-scan store); otherwise falls back
        to the dummy selector.
        """
        if not fov_cfg.get("model_path"):
            logger.info("FOV selection: no model_path configured; using dummy selector")
            return select_good_fovs(sequence.stage_positions)

        z_step = getattr(sequence.z_plan, "step", None) if sequence.z_plan else None
        px_um = fov_cfg.get("pixel_size_um") or self._engine.mmcore.getPixelSizeUm()
        fs = FovSelection.from_metadata(fov_cfg, pixel_size_um=px_um, z_step_um=z_step)

        prescan_store = self._latest_store(output_dir, prescan_name)
        logger.info("FOV selection: analyzing pre-scan store %s", prescan_store.name)
        names = [p.name for p in sequence.stage_positions]
        good_names = set(fs.select(prescan_store, names))
        return [p for p in sequence.stage_positions if p.name in good_names]

    @staticmethod
    def _latest_store(output_dir: Path, base_name: str) -> Path:
        """Locate the just-written ``<base_name>_N.ome.zarr`` (engine appends N)."""
        candidates = sorted(
            output_dir.glob(f"{base_name}_*.ome.zarr"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError(
                f"No pre-scan store {base_name}_*.ome.zarr found in {output_dir}"
            )
        return candidates[-1]

    @staticmethod
    def _build_prescan(sequence: MDASequence, fov_cfg: dict) -> MDASequence:
        """Build the pre-scan sequence from the explicit ``prescan`` block.

        Fields omitted from the ``prescan`` block default to the timelapse
        sequence's values; the time plan is always forced to a single timepoint.
        """
        prescan_cfg = fov_cfg.get("prescan", {}) or {}
        channels = prescan_cfg.get("channels", sequence.channels)
        z_plan = prescan_cfg.get("z_plan", sequence.z_plan)
        return sequence.replace(
            channels=channels,
            z_plan=z_plan,
            time_plan={"loops": 1, "interval": 0},
        )
