"""FOV-selection preprocessing -- deskew, phase reconstruction, virtual staining.

Copied from ``shrimpy/dynatrack/preprocessing.py`` and DECOUPLED from
``DynaTrackConfig`` so FOV-selection reconstruction can evolve independently of
DynaTrack's tracking. The only real change from the DynaTrack version is
``build_preprocessor``: it takes explicit arguments (``preprocessing``,
``deskew``, ``phase``, ``virtual_staining``) instead of a config object. The
``_LabelfreePreprocessor`` class is unchanged -- it already took explicit
settings.

For a VS pipeline the preprocessor turns a raw brightfield ``(Z, Y, X)`` stack
into a dict of virtual-stained channels ``{'nuclei': ..., 'membrane': ...}``
(plus ``'phase'`` as a debug intermediate) -- exactly the inputs the FOV
pipeline segments.

Requires optional dependencies: ``waveorder`` (phase) and ``cytoland`` (VS),
which live in the ``dynatrack`` dependency group::

    uv sync --group dynatrack
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
    logger.info("FOV selection compute device: %s", device)
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

    Decoupled from DynaTrack: instead of a ``DynaTrackConfig`` this takes the
    reconstruction fields directly. The pixel size / z step must already be
    injected into ``deskew`` and ``phase`` by the caller (see
    ``FovSelectionConfig``/the manager), as DynaTrack does at runtime.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Shape of the raw z-stack ``(Z, Y, X)`` -- needed for the transfer
        function.
    preprocessing : list[str] | None
        Ordered steps, e.g. ``['deskew', 'phase', 'vs']``. ``None``/``[]`` ->
        no preprocessing (returns ``None``).
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
            "FOV-selection preprocessing requires 'deskew' and/or 'phase' step; got %s",
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
    Copied verbatim from DynaTrack (it already took explicit settings).
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
        self._deskew_settings = deskew_settings
        self._phase_settings = phase_settings
        self._vs_config = vs_config
        self._output_channel = output_channel
        self._device = None

        self._transfer_function: tuple[torch.Tensor, ...] | None = None
        self._vs_model = None
        self._vs_target_channels: list[str] = ["nuclei", "membrane"]
        self._vs_step: int = 1

    def warm_up(self) -> None:
        """Pre-compute the transfer function and load the VS model."""
        self._device = _resolve_device()

        if self._deskew_settings is not None:
            from biahub.deskew import get_deskewed_data_shape

            deskewed_shape, _ = get_deskewed_data_shape(
                raw_data_shape=self._zyx_shape,
                **_settings_kwargs(get_deskewed_data_shape, self._deskew_settings),
            )
            logger.info(
                "FOV selection: deskew will reshape %s -> %s",
                self._zyx_shape,
                deskewed_shape,
            )
            self._zyx_shape = deskewed_shape

        if self._phase_settings is not None:
            self._compute_transfer_function()

        if self._vs_config is not None:
            logger.info("FOV selection: pre-loading VS model...")
            self._vs_model = self._load_vs_model()
            logger.info("FOV selection: VS model ready")

    def _compute_transfer_function(self) -> None:
        """Compute the transfer function and move to the target device."""
        from waveorder.models.phase_thick_3d import calculate_transfer_function

        if self._device is None:
            self._device = _resolve_device()

        logger.info("FOV selection: computing transfer function...")
        t0 = _time.monotonic()

        tf_params = _settings_kwargs(
            calculate_transfer_function, self._phase_settings.transfer_function
        )
        tf_params["zyx_shape"] = self._zyx_shape

        real_tf, imag_tf = calculate_transfer_function(**tf_params)

        self._transfer_function = (
            real_tf.to(self._device),
            imag_tf.to(self._device),
        )

        logger.info(
            "FOV selection: transfer function ready on %s (%.1fs, computed on CPU)",
            self._device,
            _time.monotonic() - t0,
        )

    def __call__(self, volume_bf: np.ndarray) -> dict[str, torch.Tensor]:
        """Preprocess a brightfield z-stack -> dict of channel ZYX tensors.

        For a VS pipeline the keys are ``virtual_staining.target_channels``
        (e.g. ``'nuclei'``, ``'membrane'``) plus ``'phase'`` (debug). For a
        non-VS pipeline there is a single entry keyed by ``output_channel``.
        """
        import torch

        channels: dict[str, torch.Tensor] = {}

        volume = torch.as_tensor(volume_bf, device=self._device, dtype=torch.float32)

        if self._deskew_settings is not None:
            volume = self._deskew(volume)

        if self._phase_settings is not None:
            volume_phase = self._reconstruct_phase(volume)
        else:
            volume_phase = volume

        if self._vs_config is not None:
            if self._phase_settings is not None:
                channels["phase"] = volume_phase
            channels.update(self._predict_vs(volume_phase))
        else:
            channels[self._output_channel] = volume_phase

        self._log_gpu_memory()
        return channels

    def _deskew(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply deskewing via biahub's ``fast_deskew_zyx`` on the target device."""
        from biahub.deskew import fast_deskew_zyx

        logger.info("FOV selection: deskewing volume %s...", tuple(volume.shape))
        t0 = _time.monotonic()

        result = fast_deskew_zyx(
            raw_data=volume, **_settings_kwargs(fast_deskew_zyx, self._deskew_settings)
        )

        logger.info(
            "FOV selection: deskew took %.1fs (%s -> %s)",
            _time.monotonic() - t0,
            tuple(volume.shape),
            tuple(result.shape),
        )
        return result

    def _reconstruct_phase(self, volume_bf: torch.Tensor) -> torch.Tensor:
        """Apply phase reconstruction via waveorder on the target device."""
        from waveorder.models.phase_thick_3d import apply_inverse_transfer_function

        if self._transfer_function is None:
            self._compute_transfer_function()

        logger.info("FOV selection: reconstructing phase on %s...", self._device)
        t0 = _time.monotonic()

        inverse_config = _settings_kwargs(
            apply_inverse_transfer_function, self._phase_settings.apply_inverse
        )
        z_padding = self._phase_settings.transfer_function.z_padding

        t_phase = apply_inverse_transfer_function(
            volume_bf, *self._transfer_function, z_padding=z_padding, **inverse_config
        )

        logger.info("FOV selection: phase reconstruction took %.1fs", _time.monotonic() - t0)
        return t_phase

    def _predict_vs(self, volume_phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply virtual staining via cytoland; one channel per target channel."""
        import torch

        if self._vs_model is None:
            logger.info("FOV selection: loading VS model...")
            self._vs_model = self._load_vs_model()

        logger.info("FOV selection: predicting virtual staining...")
        t0 = _time.monotonic()

        t_input = volume_phase.to(dtype=torch.float32)[None, None]

        with torch.no_grad():
            t_output = self._vs_model.predict_sliding_windows(
                t_input, out_channel=len(self._vs_target_channels), step=self._vs_step
            )

        result = {
            name: t_output[0, i].detach() for i, name in enumerate(self._vs_target_channels)
        }

        logger.info("FOV selection: VS prediction took %.1fs", _time.monotonic() - t0)
        return result

    def _load_vs_model(self):
        """Validate the VS config and load the cytoland model for inference."""
        import jsonargparse

        from cytoland.engine import AugmentedPredictionVSUNet, VSUNet

        cfg = self._vs_config

        model_init_args = cfg.get("model", {}).get("init_args", {})
        if model_init_args.get("test_time_augmentations"):
            logger.warning(
                "FOV selection VS: 'test_time_augmentations' is set but not applied; "
                "ignoring it."
            )
        if "data" in cfg or "normalizations" in cfg:
            logger.warning(
                "FOV selection VS: input normalization configured but not applied; "
                "reconstruction runs on the raw phase volume."
            )

        parser = jsonargparse.ArgumentParser()
        parser.add_subclass_arguments(VSUNet, "model")
        parser.add_argument("--ckpt_path", type=str, required=True)
        parser.add_argument("--sliding_window_step", type=int, default=1)
        parser.add_argument(
            "--target_channels", type=list[str], default=["nuclei", "membrane"]
        )
        parsed = parser.parse_object(cfg)

        parsed.model.init_args.ckpt_path = str(parsed.ckpt_path)
        instances = parser.instantiate_classes(parsed)
        vsunet = instances.model

        self._vs_target_channels = list(parsed.target_channels)
        self._vs_step = int(parsed.sliding_window_step)

        device = self._device
        vsunet.eval().to(device)
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
                    "FOV selection GPU memory: %.0f MB allocated, %.0f MB reserved",
                    alloc,
                    reserved,
                )
        except ImportError:
            pass
