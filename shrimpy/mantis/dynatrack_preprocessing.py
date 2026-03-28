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
import time as _time

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import torch

    from shrimpy.mantis.dynatrack import DynaTrackConfig

logger = logging.getLogger(__name__)


def _resolve_deskew_params(config: dict[str, Any]) -> dict[str, Any]:
    """Convert user-facing deskew config to biahub API parameters.

    Accepts ``pixel_size_um`` and ``scan_step_um`` and computes
    ``px_to_scan_ratio = pixel_size_um / scan_step_um``.
    """
    params = dict(config)
    pixel_size_um = params.pop("pixel_size_um", None)
    scan_step_um = params.pop("scan_step_um", None)
    if pixel_size_um is not None and scan_step_um is not None:
        params["px_to_scan_ratio"] = pixel_size_um / scan_step_um
    return params


def _resolve_device() -> torch.device:
    """Return the best available torch device (lazy import)."""
    from waveorder.device import resolve_device

    device = resolve_device("auto")
    logger.info("DynaTrack compute device: %s", device)
    return device


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

    if "phase" not in pipeline and "deskew" not in pipeline:
        logger.warning(
            "DynaTrack preprocessing requires 'deskew' and/or 'phase' step; got %s",
            pipeline,
        )
        return None

    preprocessor = _LabelfreePreprocessor(
        zyx_shape=zyx_shape,
        deskew_config=(
            _resolve_deskew_params(config.deskew_config)
            if "deskew" in pipeline and config.deskew_config
            else None
        ),
        phase_config=config.phase_config if "phase" in pipeline else None,
        vs_config=config.vs_config if "vs" in pipeline else None,
        output_channel=channel,
    )
    preprocessor.warm_up()
    return preprocessor


class _LabelfreePreprocessor:
    """Stateful preprocessor that caches the transfer function and VS model.

    Callable as ``preprocessor(volume_bf: np.ndarray) -> np.ndarray``.

    Uses ``waveorder.models.phase_thick_3d`` directly (not the xarray-based
    ``waveorder.api.phase``) for lower overhead and explicit GPU control.
    """

    def __init__(
        self,
        zyx_shape: tuple[int, int, int],
        deskew_config: dict[str, Any] | None,
        phase_config: dict[str, Any] | None,
        vs_config: dict[str, Any] | None,
        output_channel: str,
    ) -> None:
        self._zyx_shape = zyx_shape
        self._deskew_config = deskew_config
        self._phase_config = phase_config
        self._vs_config = vs_config
        self._output_channel = output_channel
        self._device = None

        # Cached state (computed by warm_up or lazily on first call)
        self._transfer_function: tuple[torch.Tensor, ...] | None = None
        self._vs_model = None

    def warm_up(self) -> None:
        """Pre-compute the transfer function and load the VS model.

        Called before the acquisition starts so the first DynaTrack update
        doesn't pay the initialization cost.
        """
        self._device = _resolve_device()

        # If deskewing, the phase TF must use the deskewed shape
        if self._deskew_config is not None and self._phase_config is not None:
            from biahub.deskew import get_deskewed_data_shape

            deskewed_shape, _ = get_deskewed_data_shape(
                raw_data_shape=self._zyx_shape,
                **self._deskew_config,
            )
            logger.info(
                "DynaTrack: deskew will reshape %s -> %s",
                self._zyx_shape,
                deskewed_shape,
            )
            self._zyx_shape = deskewed_shape

        if self._phase_config is not None:
            self._compute_transfer_function()

        if self._vs_config is not None:
            logger.info("DynaTrack: pre-loading VS model...")
            self._vs_model = self._load_vs_model()
            logger.info("DynaTrack: VS model ready")

    def _compute_transfer_function(self) -> None:
        """Compute the transfer function and move to the target device."""

        from waveorder.models.phase_thick_3d import calculate_transfer_function

        if self._device is None:
            self._device = _resolve_device()

        logger.info("DynaTrack: computing transfer function...")
        t0 = _time.monotonic()

        tf_params = dict(self._phase_config.get("transfer_function", {}))
        tf_params.pop("zyx_shape", None)
        tf_params["zyx_shape"] = self._zyx_shape

        # calculate_transfer_function runs on CPU internally
        real_tf, imag_tf = calculate_transfer_function(**tf_params)

        # Move to target device for fast apply_inverse
        self._transfer_function = (
            real_tf.to(self._device),
            imag_tf.to(self._device),
        )

        elapsed = _time.monotonic() - t0
        logger.info(
            "DynaTrack: transfer function ready on %s (%.1fs, computed on CPU)",
            self._device,
            elapsed,
        )

    def __call__(self, volume_bf: np.ndarray) -> dict[str, np.ndarray]:
        """Preprocess a brightfield z-stack.

        Parameters
        ----------
        volume_bf : np.ndarray
            Raw brightfield volume, shape ``(Z, Y, X)``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of channel name to ZYX array. Always includes
            ``'phase'``; may also include ``'vs_nuclei'`` and
            ``'vs_membrane'`` when VS is enabled.
        """
        channels: dict[str, np.ndarray] = {}

        # 1. Deskew
        if self._deskew_config is not None:
            volume_bf = self._deskew(volume_bf)

        # 2. Phase reconstruction
        if self._phase_config is not None:
            volume_phase = self._reconstruct_phase(volume_bf)
            channels["phase"] = volume_phase
        else:
            volume_phase = volume_bf

        # 3. Virtual staining
        if self._vs_config is not None:
            vs_result = self._predict_vs(volume_phase)
            channels.update(vs_result)

        # If no phase or VS, return the (possibly deskewed) raw volume
        if not channels:
            channels["raw"] = volume_bf

        self._log_gpu_memory()
        return channels

    def _deskew(self, volume: np.ndarray) -> np.ndarray:
        """Apply deskewing via biahub."""
        from biahub.deskew import deskew as biahub_deskew

        logger.info("DynaTrack: deskewing volume %s...", volume.shape)
        t0 = _time.monotonic()

        device = str(self._device) if self._device is not None else "cpu"
        result = biahub_deskew(
            raw_data=volume,
            device=device,
            **self._deskew_config,
        )

        logger.info(
            "DynaTrack: deskew took %.1fs (%s -> %s)",
            _time.monotonic() - t0,
            volume.shape,
            result.shape,
        )
        return result

    def _reconstruct_phase(self, volume_bf: np.ndarray) -> np.ndarray:
        """Apply phase reconstruction via waveorder on the target device."""
        import torch

        from waveorder.models.phase_thick_3d import apply_inverse_transfer_function

        # Compute transfer function once and cache
        if self._transfer_function is None:
            self._compute_transfer_function()

        logger.info("DynaTrack: reconstructing phase on %s...", self._device)
        t0 = _time.monotonic()

        # Move volume to device and reconstruct
        t_volume = torch.as_tensor(volume_bf, device=self._device, dtype=torch.float32)

        inverse_config = dict(self._phase_config.get("apply_inverse", {}))
        z_padding = self._phase_config.get("transfer_function", {}).get("z_padding", 0)

        t_phase = apply_inverse_transfer_function(
            t_volume, *self._transfer_function, z_padding=z_padding, **inverse_config
        )

        phase = t_phase.detach().cpu().numpy()
        del t_volume, t_phase
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DynaTrack: phase reconstruction took %.1fs", _time.monotonic() - t0)
        return phase

    def _predict_vs(self, volume_phase: np.ndarray) -> dict[str, np.ndarray]:
        """Apply virtual staining via viscy.

        Returns
        -------
        dict[str, np.ndarray]
            ``{'vs_nuclei': ..., 'vs_membrane': ...}`` ZYX arrays.
        """
        import torch

        if self._vs_model is None:
            logger.info("DynaTrack: loading VS model...")
            self._vs_model = self._load_vs_model()

        logger.info("DynaTrack: predicting virtual staining...")
        t0 = _time.monotonic()

        device = self._device
        # viscy expects (B, C, Z, Y, X) input
        t_input = torch.as_tensor(
            volume_phase[np.newaxis, np.newaxis], device=device, dtype=torch.float32
        )

        with torch.no_grad():
            t_output = self._vs_model.predict_sliding_windows(t_input)

        # Output shape: (B, C_out, Z, Y, X) where C_out = [nuclei, membrane]
        result = {
            "vs_nuclei": t_output[0, 0].detach().cpu().numpy(),
            "vs_membrane": t_output[0, 1].detach().cpu().numpy(),
        }

        del t_input, t_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DynaTrack: VS prediction took %.1fs", _time.monotonic() - t0)
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

        module_path, class_name = class_path.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_path), class_name)

        device = self._device
        model = model_class(**init_args).to(device).eval()

        # Extract the bare nn.Module and discard the Lightning wrapper
        bare_model = model.model
        del model
        gc.collect()

        wrapper = (
            AugmentedPredictionVSUNet(
                model=bare_model,
                forward_transforms=[lambda t: t],
                inverse_transforms=[lambda t: t],
            )
            .to(device)
            .eval()
        )

        wrapper.on_predict_start()
        return wrapper

    @staticmethod
    def _log_gpu_memory() -> None:
        """Log GPU memory usage if CUDA is available."""
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                logger.debug(
                    "DynaTrack GPU memory: %.0f MB allocated, %.0f MB reserved",
                    alloc,
                    reserved,
                )
        except ImportError:
            pass
