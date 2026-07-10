from __future__ import annotations

import copy
import json
import logging
import os
import time

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psutil

from ome_writers import (
    AcquisitionSettings,
)
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import SummaryMetaV1
from pymmcore_plus.metadata.serialize import to_builtins
from useq import MDAEvent, MDASequence

from shrimpy.dynatrack import DynaTrack
from shrimpy.fov_selection import FovSelection

# Get the logger instance (will be configured by the CLI entry point)
logger = logging.getLogger(__name__)

MANTIS_XY_STAGE_NAME = "XYStage:XY:31"
DEMO_PFS_METHOD = "demo-PFS"
SLOW_XY_STAGE_SPEED = 2.0  # in mm/s, used for short moves to maintain autofocus lock
FAST_XY_STAGE_SPEED = 5.75  # in mm/s, used for long moves
NEGLIGIBLE_XY_DISTANCE = 1  # in um, moves below this are ignored
SHORT_XY_DISTANCE = 2000  # in um, threshold between slow and fast speed

_PROC = psutil.Process(os.getpid())


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024**3)


def _find_shrimpy_log_file() -> Path | None:
    """Return the path of the FileHandler attached to the shrimpy logger."""
    for handler in logging.getLogger("shrimpy").handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


class MantisEngine(MDAEngine):
    """Custom MDA engine for the Mantis microscope.

    This engine extends the default MDAEngine to handle mantis-specific
    hardware setup and configuration, including:
    - TriggerScope sequencing configuration
    - ROI setup
    - Axial Piezo (AP Galvo) focus control
    - TTL blanking
    - Autofocus after XY stage movements
    """

    def __init__(self, mmc: CMMCorePlus, *args, **kwargs):
        """Initialize and register the MantisEngine with the core.

        Parameters
        ----------
        mmc : CMMCorePlus
            The Micro-Manager core instance
        """
        kwargs.setdefault("use_hardware_sequencing", True)
        kwargs.setdefault("force_set_xy_position", False)
        # Set acquisition timeout to guard against stalling due to dropped frames
        # or missed trigger pulses
        kwargs.setdefault("timeout_base", 10.0)
        kwargs.setdefault("timeout_multiplier", 1.0)
        kwargs.setdefault("timeout_first_frame", None)
        kwargs.setdefault("timeout_action", "warn")
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._autofocus_fail_at_index = None
        self._xy_stage_device = None
        self._xy_stage_speed = None
        self._dynatrack: DynaTrack | None = None
        self._fov: FovSelection | None = None
        # Good FOV names from the pre-scan run, captured in teardown_sequence so
        # acquire() can build the timelapse run after the pre-scan run returns.
        self._fov_good_names: list[str] = []
        self._data_path: Path | None = None

        # Register event callbacks for logging
        mmc.mda.set_engine(self)
        mmc.events.propertyChanged.connect(self._on_property_changed)
        mmc.events.roiSet.connect(self._on_roi_set)
        mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)

    def _on_property_changed(self, device: str, property_name: str, value: str) -> None:
        """Log property changes at debug level."""
        # Ignore select property changes
        if property_name in ("PFS Status", "PFS in Range", "FocusMaintenance"):
            return
        logger.debug(f"Property changed: {device}.{property_name} = {value}")

    def _on_roi_set(self, camera: str, x: int, y: int, width: int, height: int) -> None:
        """Log ROI changes at debug level."""
        logger.debug(
            f"Setting ROI on {camera} to x={x}, y={y}, width={width}, height={height}"
        )

    def _on_xy_stage_position_changed(self, device: str, x: float, y: float) -> None:
        """Log stage position changes at debug level."""
        logger.debug(f"XY stage position changed: device={device}, x={x:.2f}, y={y:.2f}")

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup mantis-specific hardware before the sequence starts.

        Reads mantis-specific settings from sequence.metadata['mantis'] if present,
        otherwise uses default values.
        """
        logger.info("Setting up Mantis-specific hardware for acquisition sequence")

        core = self.mmcore

        # Extract mantis settings from metadata
        microscope_meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}

        # Set autofocus settings
        if autofocus := microscope_meta.get("autofocus"):
            if autofocus.get("enabled"):
                self._use_autofocus = True
                self._autofocus_stage = autofocus.get("stage")
                self._autofocus_method = autofocus.get("method")
                logger.info(f"Enabling autofocus with method: {self._autofocus_method}")
                if not self._autofocus_method == DEMO_PFS_METHOD:
                    core.setAutoFocusDevice(self._autofocus_method)
            else:
                logger.info("Autofocus is disabled for this acquisition")

        # Store XY stage device name
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        # Setup DynaTrack position tracking. The XY pixel size (from the core)
        # and the sequence z_plan step are the single source of truth for all
        # scale parameters; DynaTrack derives and injects them.
        self._dynatrack = DynaTrack.from_metadata(
            microscope_meta.get("dynatrack"),
            sequence,
            data_path=self._data_path,
            pixel_size_um=core.getPixelSizeUm(),
        )
        if self._dynatrack is not None:
            self.mmcore.mda.events.frameReady.connect(self._dynatrack.on_frame_ready)
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

        # Setup streaming FOV selection. This is the PRE-SCAN run: acquire()
        # builds a pre-scan sequence (input_channel only, one timepoint, all
        # candidate positions) and runs it first; the decision streams in via
        # ``frameReady``. The subsequent timelapse run disables ``fov_selection``
        # in its metadata, so ``from_metadata`` returns ``None`` there.
        self._fov = FovSelection.from_metadata(
            microscope_meta.get("fov_selection"),
            sequence,
            pixel_size_um=core.getPixelSizeUm(),
            data_path=self._data_path,
        )
        if self._fov is not None:
            self.mmcore.mda.events.frameReady.connect(self._fov.on_frame_ready)
            logger.info(
                "FOV selection pre-scan: on '%s', %d candidate positions",
                self._fov.prescan_channel,
                len(sequence.stage_positions),
            )

        logger.info("Mantis hardware setup completed successfully")

        # Call parent setup so SummaryMetaV1 captures the fully configured
        # hardware state and the setup event applies the ROI.
        result = super().setup_sequence(sequence)

        # DynaTrack runs in a worker subprocess for GPU/torch isolation:
        # torch's OpenMP runtime segfaults when it coexists with the sequenced
        # camera readout in the acquisition process. The worker is started after
        # the setup event has applied the ROI, so getImageHeight/Width reflects
        # the actual acquired frame size (also used to build the preprocessor,
        # when configured, inside the worker).
        if self._dynatrack is not None:
            zyx_shape = (
                max(sequence.sizes.get("z", 1), 1),
                self.mmcore.getImageHeight(),
                self.mmcore.getImageWidth(),
            )
            self._dynatrack.start(zyx_shape=zyx_shape, log_file_path=_find_shrimpy_log_file())

        # FOV selection runs its reconstruction in a worker subprocess for the
        # same torch/GPU isolation reason; start it after the ROI is applied so
        # the acquired frame shape (used to build the transfer function) is known.
        if self._fov is not None:
            zyx_shape = (
                max(sequence.sizes.get("z", 1), 1),
                self.mmcore.getImageHeight(),
                self.mmcore.getImageWidth(),
            )
            self._fov.start(zyx_shape=zyx_shape, log_file_path=_find_shrimpy_log_file())

        return result

    def event_iterator(self, events: Iterable[MDAEvent]):
        """Wrap event iteration to apply DynaTrack position updates.

        FOV selection needs no per-event handling here: the pre-scan run is
        input-channel-only and the timelapse run contains only good FOVs, so
        there is no barrier or gating (see ``acquire``).

        The per-event timepoint is read from the first sub-event of a hardware
        ``SequencedEvent`` (assumes a sequenced group does not span timepoints,
        the same assumption DynaTrack makes).

        DynaTrack (when enabled) drains pending position updates at timepoint
        boundaries and applies the corrected coordinates before the runner emits
        ``eventStarted``.
        """
        last_t: int | None = None
        for event in super().event_iterator(events):
            ev0 = event.events[0] if isinstance(event, SequencedEvent) else event
            t_idx = ev0.index.get("t", 0)

            # --- DynaTrack position updates -------------------------------
            if self._dynatrack is not None:
                if last_t is not None and t_idx != last_t:
                    self._dynatrack.drain_pending()
                last_t = t_idx
                event = self._dynatrack.apply_position_update(event)

            yield event

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare mantis hardware for each event."""
        # Set XY stage position and engage autofocus
        # Note: this command will not move the stage if the target position is the same
        # as the last commanded position and force_set_xy_position is False.
        # TODO: debug resetting xy stage speed
        # self._adjust_xy_stage_speed(event)
        self._set_event_xy_position(event)
        # _set_event_xy_position does not wait for the stage to reach the target position
        if self._xy_stage_device:
            self.mmcore.waitForDevice(self._xy_stage_device)

        # Engage autofocus
        self._engage_autofocus(event)

        # Skip acquisition if autofocus failed
        if self._use_autofocus and not self._autofocus_success:
            num_frames = len(event.events) if isinstance(event, SequencedEvent) else 1
            raise SkipEvent(num_frames=num_frames, reason="autofocus failed")

        # DEBUG:
        free_capacity = self.mmcore.getBufferFreeCapacity()
        total_capacity = self.mmcore.getBufferTotalCapacity()
        logger.debug(f"Circular buffer capacity: {free_capacity} / {total_capacity} frames")
        logger.debug(
            f"MantisEngine[mem]: setup_event rss={_rss_gb():.2f} GB "
            f"mm_buf_used={total_capacity - free_capacity}/{total_capacity}"
        )

        # Call parent setup_event
        super().setup_event(event)

    def teardown_sequence(self, sequence):
        # DynaTrack: disconnect callback and shutdown
        if self._dynatrack is not None:
            self.mmcore.mda.events.frameReady.disconnect(self._dynatrack.on_frame_ready)
            self._dynatrack.shutdown()
            self._dynatrack = None

        # FOV selection (pre-scan run): capture the good FOV names before shutting
        # down so acquire() can build the timelapse run, then disconnect + shut
        # down the worker. good_position_names() survives shutdown() (only the
        # frame buffers are cleared, not the verdicts).
        if self._fov is not None:
            self._fov.drain()
            self._fov_good_names = self._fov.good_position_names()
            self.mmcore.mda.events.frameReady.disconnect(self._fov.on_frame_ready)
            self._fov.shutdown()
            self._fov = None

        super().teardown_sequence(sequence)

        core = self.mmcore
        microscope_meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}

        if reset_hardware_sequencing_settings := microscope_meta.get(
            "reset_hardware_sequencing_settings"
        ):
            logger.info(
                f"Resetting {len(reset_hardware_sequencing_settings)} hardware sequencing settings"
            )
            for setting in reset_hardware_sequencing_settings:
                logger.debug(f"  Setting {setting[0]}.{setting[1]} = {setting[2]}")
                core.setProperty(setting[0], setting[1], setting[2])
        else:
            logger.debug("No reset hardware sequencing settings specified")

    def _set_event_properties(self, properties: Iterable[tuple]) -> None:
        """Set properties for the current event."""
        for device, prop, value in properties:
            if (
                prop == Keyword.Position
                and device == self._autofocus_stage
                and self._use_autofocus
            ):
                # Skip setting Z position if autofocus is enabled to avoid
                # disengaging autofocus lock; autofocus algorithm will set Z
                # position independently
                logger.debug(
                    "Skipping Z set on autofocus stage: %s.%s = %s", device, prop, value
                )
                continue
            super()._set_event_properties([(device, prop, value)])

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

    def _engage_autofocus(self, event: MDAEvent) -> None:
        if not self._use_autofocus:
            logger.debug("Autofocus is disabled.")
            return

        if self._autofocus_method == DEMO_PFS_METHOD:
            self._engage_demo_pfs(
                event=event,
                success_rate=0.5,
                fail_at_index=self._autofocus_fail_at_index,
            )
        else:
            z_position = None
            if event.properties:
                for dev, prop, value in event.properties:
                    if dev == self._autofocus_stage and prop == "Position":
                        z_position = value
                        break
            if z_position is None:
                z_position = self.mmcore.getPosition(self._autofocus_stage)
            self._engage_nikon_pfs(self._autofocus_stage, z_position)

    def _engage_demo_pfs(
        self,
        event: MDAEvent | None = None,
        success_rate: float = 0.9,
        fail_at_index: list[dict] | None = None,
    ):
        """Engage demo PFS continuous autofocus.

        If ``fail_at_index`` is provided, autofocus deterministically fails
        when the event index matches any entry in the list. Otherwise, success
        is random based on ``success_rate``.

        Parameters
        ----------
        event : MDAEvent | None
            The current MDA event (used for deterministic failure matching).
        success_rate : float
            The probability of success for the demo PFS call. Only used when
            ``fail_at_index`` is not provided.
        fail_at_index : list[dict] | None
            List of index dicts to fail at, e.g. ``[{"p": 0}, {"t": 1, "p": 2}]``.
            Each dict is matched against the event index — if all keys in the
            dict match the event index, autofocus fails at that event.
        """
        if fail_at_index is not None and event is not None:
            # For SequencedEvents, use the first sub-event's index
            event_index = (
                event.events[0].index if isinstance(event, SequencedEvent) else event.index
            )
            self._autofocus_success = not any(
                all(event_index.get(k) == v for k, v in idx.items()) for idx in fail_at_index
            )
        else:
            self._autofocus_success = np.random.random() < success_rate

        if self._autofocus_success:
            logger.debug(f"{DEMO_PFS_METHOD} call succeeded")
        else:
            logger.debug(f"{DEMO_PFS_METHOD} call failed")

    def _engage_nikon_pfs(self, z_stage_name: str, z_position: float):
        """
                Attempt to engage Nikon PFS continuous autofocus. This function will log a
                message and continue if continuous autofocus is already engaged. Otherwise,
                it will attempt to engage autofocus, moving the z stage by amounts given in
                `z_offsets`, if necessary.
        `
                Parameters`
                ----------
                z_stage_name : str
                    The name of the z stage device which will be moved to help engage autofocus.
                z_position : float
                    The target position at which autofocus will be engaged.
        """
        core = self.mmcore
        self._autofocus_success = False
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
            self._autofocus_success = True
            logger.debug("Continuous autofocus is already engaged")
            return

        for z_offset in z_offsets:
            core.setPosition(z_stage_name, z_position + z_offset)
            core.waitForDevice(z_stage_name)

            # This call engages autofocus
            core.enableContinuousFocus(True)
            time.sleep(1)  # Wait for autofocus to engage

            if core.isContinuousFocusLocked():
                self._autofocus_success = True
                logger.debug(f"Continuous autofocus engaged with Z offset of {z_offset} um")
                break
            else:
                logger.debug(f"Autofocus call failed with Z offset of {z_offset} um")

        if not self._autofocus_success:
            # return z stage to original position if autofocus attempts failed
            core.setPosition(z_stage_name, z_position)
            core.waitForDevice(z_stage_name)

            logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")

    def acquire(
        self,
        output_dir: str | Path,
        name: str,
        mda_config: MDASequence | str | Path,
    ) -> None:
        """Run a Mantis microscope acquisition.

        When ``metadata.mantis.fov_selection`` is enabled this runs the two-run
        adaptive acquisition (pre-scan then timelapse on good FOVs only);
        otherwise it is a single ordinary run.

        Parameters
        ----------
        output_dir : str | Path
            Directory where acquisition data will be saved.
        name : str
            Base acquisition name; an index suffix will be appended automatically.
        mda_config : MDASequence | str | Path
            An MDASequence object or path to an MDA sequence configuration YAML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _get_next_acquisition_name(output_dir, name)

        if isinstance(mda_config, MDASequence):
            sequence = mda_config
        else:
            logger.info(f"Loading MDA sequence from {mda_config}")
            sequence = MDASequence.from_file(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"
        self._data_path = data_path

        fov_cfg = _enabled_fov_config(sequence)
        if fov_cfg is None:
            logger.info(f"Starting acquisition: {name}")
            self._run_mda(sequence, data_path, write_summary=True)
            logger.info("Acquisition completed successfully")
            return

        self._acquire_with_fov_selection(sequence, output_dir, name, data_path, fov_cfg)

    def _acquire_with_fov_selection(
        self,
        sequence: MDASequence,
        output_dir: Path,
        name: str,
        data_path: Path,
        fov_cfg: dict,
    ) -> None:
        """Two sequential runs: pre-scan (decide) then timelapse on good FOVs.

        Run 1 pre-scans all candidate FOVs on ``prescan_channel`` only (one
        timepoint, full z); the decision streams in via ``frameReady`` and the
        good FOV names are captured in ``teardown_sequence``. Run 2 images only
        the good FOVs. See ``docs/fov_selection_integration_plan.md``.
        """
        prescan_channel = fov_cfg.get("prescan_channel", "BF - Oblique")
        prescan_seq = _build_prescan_sequence(sequence, prescan_channel)

        prescan_output = None
        if fov_cfg.get("save_prescan"):
            prescan_output = output_dir / f"{name}_prescan.ome.zarr"

        logger.info(
            "FOV selection: pre-scan run on '%s' (%d candidate FOVs); output=%s",
            prescan_channel,
            len(prescan_seq.stage_positions),
            prescan_output or "discarded",
        )
        self._run_mda(prescan_seq, prescan_output, write_summary=prescan_output is not None)

        good = list(self._fov_good_names)
        logger.info(
            "FOV selection: %d/%d FOVs good: %s",
            len(good),
            len(prescan_seq.stage_positions),
            good,
        )
        if not good:
            logger.warning("FOV selection: no good FOVs; skipping the timelapse run.")
            return

        timelapse_seq = _build_timelapse_sequence(sequence, good)
        logger.info("FOV selection: timelapse run on %d good FOVs", len(good))
        self._run_mda(timelapse_seq, data_path, write_summary=True)
        logger.info("Acquisition completed successfully")

    def _run_mda(
        self,
        sequence: MDASequence,
        output: Path | None,
        *,
        write_summary: bool,
    ) -> None:
        """Run one ``core.mda.run``. ``output=None`` writes nothing to disk.

        ``frameReady`` is still emitted when ``output`` is ``None`` (it is
        independent of the sink), so the pre-scan run drives the decision without
        producing a store.
        """
        out_settings = None
        if output is not None:
            out_settings = AcquisitionSettings(
                root_path=output, compression="blosc-zstd", format="acquire-zarr"
            )

        # Write summary metadata after the zarr store is created.
        # TODO: remove once ome-writers supports root-level metadata natively.
        if write_summary and output is not None:

            def _write_summary_metadata(_seq: MDASequence, meta: object) -> None:
                self.mmcore.mda.events.sequenceStarted.disconnect(_write_summary_metadata)
                if meta and isinstance(meta, dict):
                    (output / "summary_metadata.json").write_text(
                        json.dumps(to_builtins(meta))
                    )

            self.mmcore.mda.events.sequenceStarted.connect(_write_summary_metadata)

        self.mmcore.mda.run(
            sequence,
            output=out_settings,
            dimension_overrides={"z": {"chunk_size": min(512, sequence.sizes["z"])}},
            overwrite=False,
        )


def _enabled_fov_config(sequence: MDASequence) -> dict | None:
    """Return the ``fov_selection`` metadata block when enabled, else ``None``."""
    meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}
    fov_cfg = meta.get("fov_selection")
    if fov_cfg and fov_cfg.get("enabled"):
        return fov_cfg
    return None


def _build_prescan_sequence(sequence: MDASequence, prescan_channel: str) -> MDASequence:
    """Pre-scan sequence: ``prescan_channel`` only, one timepoint, all candidates.

    Keeps the candidate ``stage_positions`` and ``z_plan`` (full z is needed for
    virtual-staining quality) and ``fov_selection`` metadata (so
    ``setup_sequence`` builds the ``FovSelection`` coordinator).
    """
    channels = [c for c in sequence.channels if c.config == prescan_channel]
    if not channels:
        raise ValueError(
            f"FOV selection prescan_channel {prescan_channel!r} is not one of the "
            f"acquisition channels {[c.config for c in sequence.channels]}."
        )
    return sequence.replace(channels=channels, time_plan={"loops": 1, "interval": 0})


def _build_timelapse_sequence(sequence: MDASequence, good_names: list[str]) -> MDASequence:
    """Timelapse sequence: good FOVs only, ``fov_selection`` disabled.

    Uses the original ``time_plan`` as-is (``loops`` is the timelapse point count;
    the pre-scan is its own run, so there is no ``+1``). Disabling
    ``fov_selection`` in the metadata makes ``setup_sequence`` build no
    coordinator for this run.
    """
    good_positions = _filter_good_positions(sequence, good_names)
    meta = copy.deepcopy(sequence.metadata) if sequence.metadata else {}
    fov_cfg = meta.get("mantis", {}).get("fov_selection")
    if fov_cfg is not None:
        fov_cfg["enabled"] = False
    return sequence.replace(stage_positions=good_positions, metadata=meta)


def _row_index_to_letter(index: int) -> str:
    """Zero-based row index -> name (A, B, ..., Z, AA, ...), matching useq."""
    name = ""
    while index >= 0:
        name = chr(index % 26 + 65) + name
        index = index // 26 - 1
    return name


def _filter_good_positions(sequence: MDASequence, good_names: list[str]) -> list:
    """Candidate positions whose name is in ``good_names`` (order preserved).

    Iterating ``sequence.stage_positions`` yields expanded ``AbsolutePosition``
    objects that carry ``plate_row``/``plate_col``, so the filtered list still
    produces a proper HCS OME-Zarr for the good FOVs. Two adjustments are made so
    the rebuilt explicit list matches what the ``WellPlatePlan`` path would emit:

    * integer plate coordinates are converted to strings (``1 -> "B"``,
      ``3 -> "4"``) -- useq forbids a field-suffixed name on a standalone position
      with *integer* plate coords, but accepts an explicit name with *string* ones;
    * the useq FOV name (e.g. ``"B4_0000"``) is reduced to the per-well field name
      (``"0000"``) -- the well's image path must be alphanumeric (iohub rejects the
      underscore), and this is exactly what ome-writers' WellPlatePlan builder uses.

    The ReplayCamera maps positions by ``plate_row``/``plate_col`` and per-well
    order, not by name, so the rename does not affect offline replay.
    """
    good = set(good_names)
    out = []
    for idx, pos in enumerate(sequence.stage_positions):
        if (pos.name or f"p{idx}") not in good:
            continue
        if isinstance(pos.plate_row, int) and isinstance(pos.plate_col, int):
            row_letter = _row_index_to_letter(pos.plate_row)
            col_label = str(pos.plate_col + 1)
            well_name = f"{row_letter}{col_label}"
            field_name = pos.name or ""
            if field_name.startswith(f"{well_name}_"):
                field_name = field_name[len(well_name) + 1 :]
            field_name = "".join(c for c in field_name if c.isalnum()) or f"{idx:04d}"
            pos = pos.model_copy(
                update={
                    "plate_row": row_letter,
                    "plate_col": col_label,
                    "name": field_name,
                }
            )
        out.append(pos)
    return out


def _get_next_acquisition_name(output_dir: Path, name: str) -> str:
    """Get next available acquisition name with incremented index.

    Parameters
    ----------
    output_dir : Path
        Output directory where acquisitions are saved.
    name : str
        Base acquisition name.

    Returns
    -------
    str
        Acquisition name with index (e.g., "acq_1", "acq_2", etc.).
    """
    idx = 1
    while True:
        indexed_name = f"{name}_{idx}"
        data_path = output_dir / f"{indexed_name}.ome.zarr"
        if not data_path.exists():
            return indexed_name
        idx += 1
