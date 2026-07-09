"""Shared label-free preprocessing -- deskew, phase reconstruction, virtual staining.

Turns a raw brightfield ``(Z, Y, X)`` stack into one or more processed channels.
Used by both DynaTrack (``shrimpy.dynatrack``) and FOV selection
(``shrimpy.fov_selection``); it is deliberately decoupled from either package's
config object -- :func:`build_preprocessor` takes the reconstruction settings as
explicit arguments so each caller extracts them from its own config.

For a VS pipeline the preprocessor turns a raw brightfield ``(Z, Y, X)`` stack
into a dict of virtual-stained channels ``{'nuclei': ..., 'membrane': ...}``
(plus ``'phase'`` as a debug intermediate). For a non-VS pipeline it returns a
single entry keyed by ``output_channel`` (raw/deskewed/phase volume).

Requires optional dependencies: ``waveorder`` (phase) and ``cytoland`` (VS),
which live in the ``dynatrack`` / ``fov`` dependency groups::

    uv sync --group dynatrack   # or: --group fov
"""

from __future__ import annotations

import logging
import time as _time

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import torch

logger = logging.getLogger(__name__)


def _settings_kwargs(func: Callable, settings: Any) -> dict[str, Any]:
    """Return the fields of a pydantic *settings* model that *func* accepts.

    The upstream settings models (biahub ``DeskewSettings``, waveorder transfer
    function / apply-inverse settings) carry more fields than the low-level
    ``biahub``/``waveorder`` functions take. Dumping the model and filtering to
    the callable's signature keeps us from passing unexpected keyword arguments
    when a settings schema gains a field the function does not consume.
    """
    import inspect

    accepted = set(inspect.signature(func).parameters)
    return {k: v for k, v in settings.model_dump().items() if k in accepted}


def _resolve_device() -> torch.device:
    """Return the best available torch device (lazy import)."""
    from waveorder.device import resolve_device

    device = resolve_device("auto")
    logger.info("Preprocessing compute device: %s", device)
    return device


def build_preprocessor(
    zyx_shape: tuple[int, int, int],
    preprocessing: list[str] | None,
    deskew: dict | None = None,
    phase: dict | None = None,
    virtual_staining: dict | None = None,
    output_channel: str = "phase",
) -> Callable[[np.ndarray], dict] | None:
    """Build a preprocessing callable from explicit reconstruction settings.

    The pixel size / z step must already be injected into ``deskew`` and
    ``phase`` by the caller (DynaTrack and FOV selection both do this from a
    single source of truth: ``core.getPixelSizeUm()`` and the z_plan step).

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Shape of the raw z-stack ``(Z, Y, X)`` -- needed for the transfer
        function.
    preprocessing : list[str] | None
        Ordered steps, e.g. ``['deskew', 'phase', 'vs']``. ``None``/``[]`` ->
        no preprocessing (returns ``None``). Only the reconstruction steps
        (``'deskew'``, ``'phase'``, ``'vs'``) are consumed here; downstream
        steps such as projection/segmentation are handled by the caller.
    deskew, phase, virtual_staining : dict | None
        Sub-configs for the biahub deskew, waveorder phase, and cytoland VS
        steps (only used when the corresponding step is in ``preprocessing``).
    output_channel : str
        Name for the single output channel in a NON-VS pipeline. Ignored for a
        VS pipeline (which emits one channel per ``target_channels`` name).

    Returns
    -------
    Callable or None
        ``(np.ndarray ZYX) -> dict[str, torch.Tensor]`` or ``None``.
    """
    pipeline = preprocessing

    if not pipeline:
        return None

    if "phase" not in pipeline and "deskew" not in pipeline:
        logger.warning(
            "Preprocessing requires a 'deskew' and/or 'phase' step; got %s",
            pipeline,
        )
        return None

    deskew_settings = None
    if "deskew" in pipeline and deskew:
        from biahub.settings import DeskewSettings

        deskew_settings = DeskewSettings(**deskew)

    phase_settings = None
    if "phase" in pipeline and phase:
        from waveorder.api.phase import Settings as PhaseSettings

        phase_settings = PhaseSettings(**phase)

    preprocessor = _LabelfreePreprocessor(
        zyx_shape=zyx_shape,
        deskew_settings=deskew_settings,
        phase_settings=phase_settings,
        vs_config=virtual_staining if "vs" in pipeline else None,
        output_channel=output_channel,
    )
    preprocessor.warm_up()
    return preprocessor


class _LabelfreePreprocessor:
    """Stateful preprocessor that caches the transfer function and VS model.

    Callable as ``preprocessor(volume_bf: np.ndarray) -> dict[str, torch.Tensor]``.

    Uses ``waveorder.models.phase_thick_3d`` directly (not the xarray-based
    ``waveorder.api.phase``) for lower overhead and explicit GPU control.
    """

    def __init__(
        self,
        zyx_shape: tuple[int, int, int],
        deskew_settings: Any | None,
        phase_settings: Any | None,
        vs_config: dict[str, Any] | None,
        output_channel: str,
    ) -> None:
        self._zyx_shape = zyx_shape
        # Validated upstream settings models: biahub DeskewSettings and the
        # waveorder phase settings.
        self._deskew_settings = deskew_settings
        self._phase_settings = phase_settings
        self._vs_config = vs_config
        self._output_channel = output_channel
        self._device = None

        # Cached state (computed by warm_up or lazily on first call)
        self._transfer_function: tuple[torch.Tensor, ...] | None = None
        self._vs_model = None
        # Derived from vs_config when the VS model is loaded (see _load_vs_model)
        self._vs_target_channels: list[str] = ["nuclei", "membrane"]
        self._vs_step: int = 1

    def warm_up(self) -> None:
        """Pre-compute the transfer function and load the VS model.

        Called before acquisition starts so the first reconstruction doesn't
        pay the initialization cost.
        """
        self._device = _resolve_device()

        # If deskewing, downstream shape is deskewed + rotated (np.rot90 in _deskew).
        if self._deskew_settings is not None:
            from biahub.deskew import get_deskewed_data_shape

            deskewed_shape, _ = get_deskewed_data_shape(
                raw_data_shape=self._zyx_shape,
                **_settings_kwargs(get_deskewed_data_shape, self._deskew_settings),
            )
            logger.info(
                "Preprocessing: deskew will reshape %s -> %s",
                self._zyx_shape,
                deskewed_shape,
            )
            self._zyx_shape = deskewed_shape

        if self._phase_settings is not None:
            self._compute_transfer_function()

        if self._vs_config is not None:
            logger.info("Preprocessing: pre-loading VS model...")
            self._vs_model = self._load_vs_model()
            logger.info("Preprocessing: VS model ready")

    def _compute_transfer_function(self) -> None:
        """Compute the transfer function and move to the target device."""
        from waveorder.models.phase_thick_3d import calculate_transfer_function

        if self._device is None:
            self._device = _resolve_device()

        logger.info("Preprocessing: computing transfer function...")
        t0 = _time.monotonic()

        tf_params = _settings_kwargs(
            calculate_transfer_function, self._phase_settings.transfer_function
        )
        tf_params["zyx_shape"] = self._zyx_shape

        # calculate_transfer_function runs on CPU internally
        real_tf, imag_tf = calculate_transfer_function(**tf_params)

        # Move to target device for fast apply_inverse
        self._transfer_function = (
            real_tf.to(self._device),
            imag_tf.to(self._device),
        )

        logger.info(
            "Preprocessing: transfer function ready on %s (%.1fs, computed on CPU)",
            self._device,
            _time.monotonic() - t0,
        )

    def __call__(self, volume_bf: np.ndarray) -> dict[str, torch.Tensor]:
        """Preprocess a brightfield z-stack -> dict of channel ZYX tensors.

        The input numpy volume is moved to the target device once; all
        subsequent steps (deskew, phase, VS) operate on torch tensors and
        return tensors on-device. Callers convert to numpy only when needed.

        For a VS pipeline the keys are ``virtual_staining.target_channels``
        (e.g. ``'nuclei'``, ``'membrane'``) plus ``'phase'`` (debug). For a
        non-VS pipeline there is a single entry keyed by ``output_channel``.
        """
        import torch

        channels: dict[str, torch.Tensor] = {}

        # Move to device once; downstream steps stay on-device.
        volume = torch.as_tensor(volume_bf, device=self._device, dtype=torch.float32)

        # 1. Deskew
        if self._deskew_settings is not None:
            volume = self._deskew(volume)

        # 2. Phase reconstruction
        if self._phase_settings is not None:
            volume_phase = self._reconstruct_phase(volume)
        else:
            volume_phase = volume

        # 3. Virtual staining
        if self._vs_config is not None:
            # VS emits one channel per vs target (e.g. 'nuclei', 'membrane').
            # Keep 'phase' as a debug-only intermediate.
            if self._phase_settings is not None:
                channels["phase"] = volume_phase
            channels.update(self._predict_vs(volume_phase))
        else:
            # Non-VS: a single processed representation of the input channel
            # (raw/deskewed/phase), keyed by output_channel.
            channels[self._output_channel] = volume_phase

        self._log_gpu_memory()
        return channels

    def _deskew(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply deskewing via biahub's ``fast_deskew_zyx`` on the target device."""
        from biahub.deskew import fast_deskew_zyx

        logger.info("Preprocessing: deskewing volume %s...", tuple(volume.shape))
        t0 = _time.monotonic()

        result = fast_deskew_zyx(
            raw_data=volume, **_settings_kwargs(fast_deskew_zyx, self._deskew_settings)
        )

        logger.info(
            "Preprocessing: deskew took %.1fs (%s -> %s)",
            _time.monotonic() - t0,
            tuple(volume.shape),
            tuple(result.shape),
        )
        return result

    def _reconstruct_phase(self, volume_bf: torch.Tensor) -> torch.Tensor:
        """Apply phase reconstruction via waveorder on the target device."""
        from waveorder.models.phase_thick_3d import apply_inverse_transfer_function

        # Compute transfer function once and cache
        if self._transfer_function is None:
            self._compute_transfer_function()

        logger.info("Preprocessing: reconstructing phase on %s...", self._device)
        t0 = _time.monotonic()

        inverse_config = _settings_kwargs(
            apply_inverse_transfer_function, self._phase_settings.apply_inverse
        )
        z_padding = self._phase_settings.transfer_function.z_padding

        t_phase = apply_inverse_transfer_function(
            volume_bf, *self._transfer_function, z_padding=z_padding, **inverse_config
        )

        logger.info("Preprocessing: phase reconstruction took %.1fs", _time.monotonic() - t0)
        return t_phase

    def _predict_vs(self, volume_phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply virtual staining via cytoland; one channel per target channel.

        The target channels and the sliding-window step are read from the
        config in :meth:`_load_vs_model` (as biahub does).

        Returns
        -------
        dict[str, torch.Tensor]
            e.g. ``{'nuclei': ..., 'membrane': ...}`` ZYX tensors on device.
        """
        import torch

        if self._vs_model is None:
            logger.info("Preprocessing: loading VS model...")
            self._vs_model = self._load_vs_model()

        logger.info("Preprocessing: predicting virtual staining...")
        t0 = _time.monotonic()

        # cytoland expects (B, C, Z, Y, X) input
        t_input = volume_phase.to(dtype=torch.float32)[None, None]

        with torch.no_grad():
            t_output = self._vs_model.predict_sliding_windows(
                t_input, out_channel=len(self._vs_target_channels), step=self._vs_step
            )

        # Output shape: (B, C_out, Z, Y, X), one channel per target channel,
        # keyed by its bare target-channel name.
        result = {
            name: t_output[0, i].detach() for i, name in enumerate(self._vs_target_channels)
        }

        logger.info("Preprocessing: VS prediction took %.1fs", _time.monotonic() - t0)
        return result

    def _load_vs_model(self):
        """Validate the VS config and load the cytoland model for inference.

        Mirrors biahub's ``virtual_stain``: the ``model`` block is validated and
        instantiated against cytoland's own ``VSUNet`` class via jsonargparse
        (the same machinery ``viscy predict`` uses), so the config stays in sync
        with cytoland and a bad key errors early. ``out_channel`` and the
        sliding-window ``step`` are derived from the config rather than
        hard-coded.

        Reconstruction runs on the raw reconstructed volume, so input
        normalization and test-time augmentation are intentionally not applied;
        if the config requests either, a warning is logged rather than silently
        ignoring it.
        """
        import jsonargparse

        from cytoland.engine import AugmentedPredictionVSUNet, VSUNet

        cfg = self._vs_config

        # Warn (don't silently ignore) about features that are not applied.
        model_init_args = cfg.get("model", {}).get("init_args", {})
        if model_init_args.get("test_time_augmentations"):
            logger.warning(
                "VS: 'test_time_augmentations' is set in virtual_staining, but it is "
                "not applied here; ignoring it."
            )
        if "data" in cfg or "normalizations" in cfg:
            logger.warning(
                "VS: input normalization configured in virtual_staining is not "
                "applied; reconstruction runs on the raw reconstructed volume."
            )

        # Validate/instantiate the model against cytoland's VSUNet signature.
        parser = jsonargparse.ArgumentParser()
        parser.add_subclass_arguments(VSUNet, "model")
        parser.add_argument("--ckpt_path", type=str, required=True)
        parser.add_argument("--sliding_window_step", type=int, default=1)
        parser.add_argument(
            "--target_channels", type=list[str], default=["nuclei", "membrane"]
        )
        parsed = parser.parse_object(cfg)

        # Route ckpt_path into the model init args so VSUNet loads the
        # checkpoint's state_dict itself (as `viscy predict` does), keeping
        # shrimpy free of any assumption about the checkpoint layout.
        parsed.model.init_args.ckpt_path = str(parsed.ckpt_path)
        instances = parser.instantiate_classes(parsed)
        vsunet = instances.model

        self._vs_target_channels = list(parsed.target_channels)
        self._vs_step = int(parsed.sliding_window_step)

        device = self._device
        vsunet.eval().to(device)
        # Wrap the bare nn.Module without TTA transforms (identity defaults).
        return AugmentedPredictionVSUNet(model=vsunet.model).to(device).eval()

    @staticmethod
    def _log_gpu_memory() -> None:
        """Log GPU memory usage if CUDA is available."""
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                logger.debug(
                    "GPU memory: %.0f MB allocated, %.0f MB reserved",
                    alloc,
                    reserved,
                )
        except ImportError:
            pass
