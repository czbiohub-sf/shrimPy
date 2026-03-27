"""DynaTrack preprocessing — phase reconstruction and virtual staining.

Builds a preprocessing callable from ``DynaTrackConfig`` that transforms
raw brightfield z-stacks before phase cross-correlation. The callable is
passed as the ``preprocessor`` argument to ``DynaTrackUpdater``.

Requires optional dependencies: ``waveorder`` (phase) and ``viscy`` (VS).
Install via::

    uv sync --group dynatrack
"""

from __future__ import annotations

import gc
import logging

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from shrimpy.mantis.dynatrack import DynaTrackConfig

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_preprocessor(
    config: DynaTrackConfig,
    zyx_shape: tuple[int, int, int],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a preprocessing callable from DynaTrack config.

    Parameters
    ----------
    config : DynaTrackConfig
        Must have ``preprocessing`` (e.g. ``['phase']`` or ``['phase', 'vs']``)
        and ``shift_estimation_channel`` (e.g. ``'phase'``, ``'vs_nuclei'``).
    zyx_shape : tuple[int, int, int]
        Shape of the z-stack ``(Z, Y, X)`` — needed for transfer function
        calculation.

    Returns
    -------
    Callable or None
        A function ``(np.ndarray) -> np.ndarray`` that preprocesses a ZYX
        stack, or ``None`` if no preprocessing is configured.
    """
    pipeline = config.preprocessing
    channel = config.shift_estimation_channel

    if not pipeline or channel == "raw":
        return None

    if "phase" not in pipeline:
        logger.warning("DynaTrack preprocessing requires 'phase' step; got %s", pipeline)
        return None

    preprocessor = _LabelfreePreprocessor(
        zyx_shape=zyx_shape,
        phase_config=config.phase_config or {},
        vs_config=config.vs_config if "vs" in pipeline else None,
        output_channel=channel,
    )
    return preprocessor


class _LabelfreePreprocessor:
    """Stateful preprocessor that caches the transfer function and VS model.

    Callable as ``preprocessor(volume_bf: np.ndarray) -> np.ndarray``.
    """

    def __init__(
        self,
        zyx_shape: tuple[int, int, int],
        phase_config: dict[str, Any],
        vs_config: dict[str, Any] | None,
        output_channel: str,
    ) -> None:
        self._zyx_shape = zyx_shape
        self._phase_config = phase_config
        self._vs_config = vs_config
        self._output_channel = output_channel

        # Cached state (computed on first call)
        self._transfer_function: tuple[torch.Tensor, ...] | None = None
        self._vs_model = None

    def __call__(self, volume_bf: np.ndarray) -> np.ndarray:
        """Preprocess a brightfield z-stack.

        Parameters
        ----------
        volume_bf : np.ndarray
            Raw brightfield volume, shape ``(Z, Y, X)``.

        Returns
        -------
        np.ndarray
            Preprocessed volume for shift estimation.
        """
        # Phase reconstruction
        volume_phase = self._reconstruct_phase(volume_bf)

        if self._output_channel == "phase":
            return volume_phase

        # Virtual staining (requires phase)
        if self._vs_config is not None and self._output_channel in (
            "vs_nuclei",
            "vs_membrane",
        ):
            return self._predict_vs(volume_phase)

        logger.warning(
            "DynaTrack: unknown output channel '%s', returning phase",
            self._output_channel,
        )
        return volume_phase

    def _reconstruct_phase(self, volume_bf: np.ndarray) -> np.ndarray:
        """Apply phase reconstruction via waveorder."""
        from waveorder.models.phase_thick_3d import (
            apply_inverse_transfer_function,
            calculate_transfer_function,
        )

        # Compute transfer function once and cache
        if self._transfer_function is None:
            logger.info("DynaTrack: computing transfer function...")
            tf_config = dict(self._phase_config.get("transfer_function", {}))
            tf_config["zyx_shape"] = self._zyx_shape
            self._transfer_function = calculate_transfer_function(**tf_config)

        # Apply inverse transfer function
        logger.debug("DynaTrack: reconstructing phase...")
        tf_tensors = tuple(tf.to(DEVICE) for tf in self._transfer_function)
        t_volume = torch.as_tensor(volume_bf, device=DEVICE, dtype=torch.float32)

        inverse_config = dict(self._phase_config.get("apply_inverse", {}))
        z_padding = self._phase_config.get("transfer_function", {}).get("z_padding", 0)

        t_phase = apply_inverse_transfer_function(
            t_volume, *tf_tensors, z_padding=z_padding, **inverse_config
        )

        phase = t_phase.detach().cpu().numpy()
        del t_volume, tf_tensors, t_phase
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return phase

    def _predict_vs(self, volume_phase: np.ndarray) -> np.ndarray:
        """Apply virtual staining via viscy."""

        # Load model once and cache
        if self._vs_model is None:
            logger.info("DynaTrack: loading VS model...")
            self._vs_model = self._load_vs_model()

        logger.debug("DynaTrack: predicting virtual staining...")
        # viscy expects (B, C, Z, Y, X) input
        t_input = torch.as_tensor(
            volume_phase[np.newaxis, np.newaxis], device=DEVICE, dtype=torch.float32
        )

        with torch.no_grad():
            t_output = self._vs_model.predict_sliding_windows(t_input)

        # Output shape: (B, C_out, Z, Y, X) where C_out = [nuclei, membrane]
        if self._output_channel == "vs_nuclei":
            result = t_output[0, 0].detach().cpu().numpy()
        elif self._output_channel == "vs_membrane":
            result = t_output[0, 1].detach().cpu().numpy()
        else:
            result = t_output[0, 0].detach().cpu().numpy()

        del t_input, t_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _load_vs_model(self):
        """Load and wrap the VS model for inference."""
        import importlib

        from viscy.translation.engine import AugmentedPredictionVSUNet

        cfg = self._vs_config
        model_cfg = cfg["model"].copy()
        init_args = model_cfg["init_args"]
        class_path = model_cfg["class_path"]

        if "ckpt_path" in cfg:
            init_args["ckpt_path"] = cfg["ckpt_path"]

        # Dynamic import of model class
        module_path, class_name = class_path.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_path), class_name)

        model = model_class(**init_args).to(DEVICE).eval()

        wrapper = (
            AugmentedPredictionVSUNet(
                model=model.model,
                forward_transforms=[lambda t: t],
                inverse_transforms=[lambda t: t],
            )
            .to(DEVICE)
            .eval()
        )

        wrapper.on_predict_start()
        return wrapper
