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

# Reconstruction steps this module knows how to apply. A pipeline that contains any of
# these builds a preprocessor; one that contains none needs no reconstruction and the
# caller uses the raw stack directly (build_preprocessor returns None). Downstream steps
# (projection, segmentation) are consumed by the caller, not here.
RECON_STEPS = ("flatfield", "deskew", "phase", "vs")


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
    require_gpu: bool = False,
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
        Ordered steps, e.g. ``['flatfield', 'deskew', 'phase', 'vs']``.
        ``None``/``[]`` -> no preprocessing (returns ``None``). Only the
        reconstruction steps (``'flatfield'``, ``'deskew'``, ``'phase'``,
        ``'vs'``) are consumed here; downstream steps such as
        projection/segmentation are handled by the caller.
    deskew, phase, virtual_staining : dict | None
        Sub-configs for the biahub deskew, waveorder phase, and cytoland VS
        steps (only used when the corresponding step is in ``preprocessing``).
    output_channel : str
        Name for the single output channel in a NON-VS pipeline. Ignored for a
        VS pipeline (which emits one channel per ``target_channels`` name).
    require_gpu : bool
        When True, ``warm_up`` raises if the resolved compute device is the CPU
        (no GPU detected) and each call verifies the reconstruction ran on the
        GPU. Used by FOV selection, whose per-FOV decision is impractical on CPU.

    Returns
    -------
    Callable or None
        ``(np.ndarray ZYX) -> dict[str, torch.Tensor]`` or ``None``.
    """
    pipeline = preprocessing or []

    # Build a preprocessor for whatever reconstruction steps are present (in any
    # combination -- flatfield only, deskew only, deskew+phase, vs, ...). No
    # reconstruction step means the raw stack is used as-is: return None and let the
    # caller pass the raw input through. No single step is mandatory.
    if not any(step in pipeline for step in RECON_STEPS):
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
        apply_flatfield="flatfield" in pipeline,
        deskew_settings=deskew_settings,
        phase_settings=phase_settings,
        vs_config=virtual_staining if "vs" in pipeline else None,
        output_channel=output_channel,
        require_gpu=require_gpu,
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
        apply_flatfield: bool = False,
        require_gpu: bool = False,
    ) -> None:
        self._zyx_shape = zyx_shape
        self._apply_flatfield = apply_flatfield
        # Validated upstream settings models: biahub DeskewSettings and the
        # waveorder phase settings.
        self._deskew_settings = deskew_settings
        self._phase_settings = phase_settings
        self._vs_config = vs_config
        self._output_channel = output_channel
        self._require_gpu = require_gpu
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
        if self._require_gpu and self._device.type == "cpu":
            raise RuntimeError(
                "GPU required but none detected: the preprocessing compute device "
                "resolved to CPU. Deskew/phase/virtual-staining for FOV selection "
                "must run on a GPU -- run on a GPU node (check CUDA / "
                "CUDA_VISIBLE_DEVICES), or disable fov_selection."
            )

        # If deskewing, downstream shape is deskewed + rotated (np.rot90 in _deskew).
        if self._deskew_settings is not None:
            from biahub.deskew import get_deskewed_data_shape

            deskewed_shape, _ = get_deskewed_data_shape(
                raw_data_shape=self._zyx_shape,
                **_settings_kwargs(get_deskewed_data_shape, self._deskew_settings),
            )
            # px_to_scan_ratio is what sets the deskewed X (scan-axis) extent, so log it
            # with the shape: an unexpected pixel size shows up here as a stretched or
            # squashed X and is otherwise invisible until someone eyeballs a projection.
            logger.info(
                "Preprocessing: deskew will reshape %s -> %s "
                "(px_to_scan_ratio=%s, pixel_size_um=%s, scan_step_um=%s)",
                self._zyx_shape,
                deskewed_shape,
                getattr(self._deskew_settings, "px_to_scan_ratio", None),
                getattr(self._deskew_settings, "pixel_size_um", None),
                getattr(self._deskew_settings, "scan_step_um", None),
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

    def __call__(
        self,
        volume_bf: np.ndarray,
        label: str = "",
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Preprocess a brightfield z-stack -> dict of channel ZYX tensors.

        The input numpy volume is moved to the target device once; all
        subsequent steps (deskew, phase, VS) operate on torch tensors and
        return tensors on-device. Callers convert to numpy only when needed.

        ``label`` (e.g. the FOV/position name) is prefixed to each step's INFO
        log so success/failure is attributable to a specific FOV:
        ``[<label>] deskew ok (0.4s)`` / ``[<label>] virtual staining FAILED: ...``.

        For a VS pipeline the keys are ``virtual_staining.target_channels``
        (e.g. ``'nuclei'``, ``'membrane'``) plus ``'phase'`` (debug). For a
        non-VS pipeline there is a single entry keyed by ``output_channel``.

        ``return_intermediates`` additionally exposes the per-step debug
        intermediates -- the post-``'deskew'`` volume and the ``'phase'`` volume
        (whenever those steps ran) -- so a caller can persist every stage for
        step-by-step debugging. Off by default so the normal decision path is
        unchanged.
        """
        import torch

        pfx = f"[{label}] " if label else ""
        channels: dict[str, torch.Tensor] = {}

        # Move to device once; all steps (flat-field, deskew, phase, VS) run on-device.
        volume = torch.as_tensor(volume_bf, device=self._device, dtype=torch.float32)

        # 0. Flat-field correction (BRIGHT-FIELD only), matching the production
        # reconstruction (biahub mantis-v2.nf 0-flatfield step).
        if self._apply_flatfield:
            volume = self._step(pfx, "flatfield", self._flat_field_BF, volume)

        # 1. Deskew
        volume_deskewed = None
        if self._deskew_settings is not None:
            volume = self._step(pfx, "deskew", self._deskew, volume)
            volume_deskewed = volume

        # 2. Phase reconstruction
        if self._phase_settings is not None:
            volume_phase = self._step(pfx, "phase", self._reconstruct_phase, volume)
        else:
            volume_phase = volume

        # 3. Virtual staining
        if self._vs_config is not None:
            # VS emits one channel per vs target (e.g. 'nuclei', 'membrane').
            # Keep 'phase' as a debug-only intermediate.
            if self._phase_settings is not None:
                channels["phase"] = volume_phase
            channels.update(
                self._step(pfx, "virtual staining", self._predict_vs, volume_phase)
            )
        else:
            # Non-VS: a single processed representation of the input channel
            # (raw/deskewed/phase), keyed by output_channel.
            channels[self._output_channel] = volume_phase

        # Expose per-step intermediates for debug persistence (does not change the
        # decision, which only reads the VS target channels).
        if return_intermediates:
            if volume_deskewed is not None:
                channels.setdefault("deskew", volume_deskewed)
            if self._phase_settings is not None:
                channels.setdefault("phase", volume_phase)

        if self._require_gpu:
            offenders = [n for n, t in channels.items() if t.device.type == "cpu"]
            if offenders:
                raise RuntimeError(
                    f"{pfx}GPU required but preprocessing output is on CPU "
                    f"(channels {offenders}); a reconstruction step fell back to CPU."
                )

        self._log_gpu_memory()
        return channels

    @staticmethod
    def _step(pfx: str, name: str, fn, arg):
        """Run one preprocessing step, logging explicit per-FOV success/failure.

        Logs ``[<label>] <name> ok (<t>s)`` on success; on failure logs
        ``[<label>] <name> FAILED: <error>`` at ERROR level and re-raises so the
        caller (worker) records the FOV as bad.
        """
        t0 = _time.monotonic()
        try:
            result = fn(arg)
        except Exception as exc:
            logger.error("%s%s FAILED: %s", pfx, name, exc)
            raise
        logger.info("%s%s ok (%.1fs)", pfx, name, _time.monotonic() - t0)
        return result

    def _flat_field_BF(self, volume: torch.Tensor) -> torch.Tensor:
        """Flat-field correct a **bright-field** z-stack on the compute device.

        NOTE: bright-field only -- do NOT use on fluorescence images. It divides
        out the per-pixel median-over-Z illumination pattern (preserving its
        mean), the same correction biahub's production reconstruction
        (``mantis-v2.nf`` 0-flatfield step) applies. That assumes a spatially
        structured, roughly static transmitted-light background, which holds for
        bright field but not for fluorescence (sparse signal on a dark
        background) -- there the per-pixel median is ~background and dividing by
        it would corrupt the signal.

        Implemented in torch so it runs on the GPU (the previous numpy version
        was a large per-FOV CPU bottleneck). ``Tensor.quantile(0.5)`` matches
        ``numpy.median`` (linear interpolation, averaging the two middle values
        for an even Z count), unlike ``torch.median`` which returns the lower
        middle -- so the result matches biahub's offline flat-fielded input.
        """
        static_pattern = volume.quantile(0.5, dim=0)  # (Y, X) per-pixel median over Z
        return volume / static_pattern * static_pattern.mean()

    def _deskew(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply deskewing via biahub's ``fast_deskew_zyx`` on the target device."""
        from biahub.deskew import fast_deskew_zyx

        logger.debug("Preprocessing: deskewing volume %s...", tuple(volume.shape))
        result = fast_deskew_zyx(
            raw_data=volume, **_settings_kwargs(fast_deskew_zyx, self._deskew_settings)
        )
        logger.debug(
            "Preprocessing: deskew %s -> %s", tuple(volume.shape), tuple(result.shape)
        )
        return result

    def _reconstruct_phase(self, volume_bf: torch.Tensor) -> torch.Tensor:
        """Apply phase reconstruction via waveorder on the target device."""
        from waveorder.models.phase_thick_3d import apply_inverse_transfer_function

        # Compute transfer function once and cache
        if self._transfer_function is None:
            self._compute_transfer_function()

        logger.debug("Preprocessing: reconstructing phase on %s...", self._device)
        inverse_config = _settings_kwargs(
            apply_inverse_transfer_function, self._phase_settings.apply_inverse
        )
        z_padding = self._phase_settings.transfer_function.z_padding

        t_phase = apply_inverse_transfer_function(
            volume_bf, *self._transfer_function, z_padding=z_padding, **inverse_config
        )
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

        logger.debug("Preprocessing: predicting virtual staining...")
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
