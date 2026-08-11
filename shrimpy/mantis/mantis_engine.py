from __future__ import annotations

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
from shrimpy.fov_selection.sequences import (
    build_prescan_sequence,
    build_timelapse_sequence,
    fov_selection_config,
)

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
        self._fov_passed_names: list[str] = []
        # Feature-viewer CSV written by a calibration pre-scan, captured in
        # teardown_sequence so acquire() can open the viewer on it after the run returns.
        self._fov_calibration_csv: Path | None = None
        self._data_path: Path | None = None
        # Dedup index appended to the acquisition name (None when the bare name was
        # free); see acquire(). Sibling artifacts append it after their own suffix.
        self._run_index: int | None = None

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

        logger.info("Mantis hardware setup completed successfully")

        # Call parent setup so SummaryMetaV1 captures the fully configured
        # hardware state and the setup event applies the ROI.
        result = super().setup_sequence(sequence)

        # Read the pixel size only AFTER the setup event has been applied. MM resolves
        # getPixelSizeUm() from whichever pixel-size config group currently matches the
        # device property values, so before super().setup_sequence() it reflects leftover
        # hardware state (whatever the GUI was last left in) rather than the state this
        # acquisition actually runs with. Reading it early silently produced a different
        # px_to_scan_ratio between otherwise-identical runs, which changed the deskewed
        # X extent (the scan axis) and stretched every downstream projection / mask /
        # physical feature. Same reason _zyx_shape() is deferred to after this point.
        pixel_size_um = core.getPixelSizeUm()
        logger.info(
            "Pixel size: %.5f um/px (config %r)",
            pixel_size_um,
            core.getCurrentPixelSizeConfig(),
        )

        # Setup DynaTrack position tracking. The XY pixel size (from the core)
        # and the sequence z_plan step are the single source of truth for all
        # scale parameters; DynaTrack derives and injects them.
        self._dynatrack = DynaTrack.from_metadata(
            microscope_meta.get("dynatrack"),
            sequence,
            data_path=self._data_path,
            pixel_size_um=pixel_size_um,
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
            pixel_size_um=pixel_size_um,
            data_path=self._data_path,
            run_index=self._run_index,
        )
        if self._fov is not None:
            self.mmcore.mda.events.frameReady.connect(self._fov.on_frame_ready)
            logger.info(
                "FOV selection pre-scan: on '%s', %d candidate positions",
                self._fov.fov_selection_channel,
                len(sequence.stage_positions),
            )

        # DynaTrack runs in a worker subprocess for GPU/torch isolation:
        # torch's OpenMP runtime segfaults when it coexists with the sequenced
        # camera readout in the acquisition process. The worker is started after
        # the setup event has applied the ROI, so getImageHeight/Width reflects
        # the actual acquired frame size (also used to build the preprocessor,
        # when configured, inside the worker).
        if self._dynatrack is not None:
            self._dynatrack.start(
                zyx_shape=self._zyx_shape(sequence),
                log_file_path=_find_shrimpy_log_file(),
            )

        # FOV selection runs its reconstruction in a worker subprocess for the
        # same torch/GPU isolation reason; start it after the ROI is applied so
        # the acquired frame shape (used to build the transfer function) is known.
        if self._fov is not None:
            self._fov.start(
                zyx_shape=self._zyx_shape(sequence),
                log_file_path=_find_shrimpy_log_file(),
            )

        return result

    def _zyx_shape(self, sequence: MDASequence) -> tuple[int, int, int]:
        """Acquired (Z, Y, X) frame shape for the current ROI.

        Shared by DynaTrack and FOV selection when starting their worker
        subprocesses. Called after the setup event has applied the ROI, so
        ``getImageHeight``/``getImageWidth`` reflect the actual frame size.
        """
        return (
            max(sequence.sizes.get("z", 1), 1),
            self.mmcore.getImageHeight(),
            self.mmcore.getImageWidth(),
        )

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

        # FOV selection (pre-scan run): capture the passing FOV names before shutting
        # down so acquire() can build the timelapse run, then disconnect + shut
        # down the worker. passed_position_names() survives shutdown() (only the
        # frame buffers are cleared, not the verdicts).
        if self._fov is not None:
            self._fov.drain()
            # Read calibration mode from the (pre-scan) sequence metadata rather than the
            # coordinator, so the branch is decided by config alone.
            calibration_mode = bool(
                fov_selection_config(sequence).get("calibration_mode", False)
            )
            if calibration_mode:
                # Calibration pre-scan: no selection / no timelapse. Capture the
                # feature-viewer CSV so acquire() can open the viewer on it.
                self._fov_calibration_csv = self._fov.calibration_matrix_csv
                self._fov_passed_names = []
                logger.info(
                    "FOV selection calibration: pre-scan complete, %d/%d FOVs scored "
                    "(all features extracted); feature matrix at %s",
                    self._fov.num_decided,
                    len(sequence.stage_positions),
                    self._fov_calibration_csv,
                )
            else:
                # Capture the selection FIRST, before anything that touches the filesystem.
                # Ordering matters: this used to run after finalize_debug_summary(), so a
                # PermissionError writing the debug CSV (a spreadsheet app holding it open)
                # aborted teardown before _fov_passed_names was ever assigned -- the timelapse
                # was then skipped for "no FOVs passed" despite a perfectly good selection.
                # The debug writers are individually guarded now; this ordering makes the
                # science independent of them regardless.
                self._fov_passed_names = self._fov.passed_position_names()
                logger.info(
                    "FOV selection: %d/%d FOVs passed: %s",
                    len(self._fov_passed_names),
                    len(sequence.stage_positions),
                    self._fov_passed_names,
                )
                self._fov.log_selection_summary()
                self._fov.finalize_debug_summary()
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
        # Index the acquisition name. The index is kept separately so sibling artifacts
        # (<name>_fov_debug/, <name>_prescan.ome.zarr) can append it at the END of their
        # own name rather than inheriting it mid-name -- "acq_fov_debug_1", not
        # "acq_1_fov_debug".
        base_name = name
        name = _get_next_acquisition_name(output_dir, base_name)
        self._run_index = int(name[len(base_name) + 1 :])

        if isinstance(mda_config, MDASequence):
            sequence = mda_config
        else:
            logger.info(f"Loading MDA sequence from {mda_config}")
            sequence = MDASequence.from_file(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"
        self._data_path = data_path

        fov_cfg = fov_selection_config(sequence)
        if not fov_cfg.get("enabled", False):
            # FOV selection is off -> ordinary single-run acquisition.
            logger.info(f"Starting acquisition: {name}")
            self._run_mda(sequence, data_path, write_summary=True)
        else:
            # FOV selection is on -> adaptive two-run acquisition: a pre-scan run
            # decides which candidate FOVs pass selection (self._fov_passed_names,
            # captured in teardown_sequence), then the timelapse images only
            # those. Sequence building lives in shrimpy/fov_selection/sequences.py.
            prescan_seq = build_prescan_sequence(sequence, fov_cfg)
            n_candidates = len(prescan_seq.stage_positions)
            logger.info("Starting FOV-selection pre-scan: %d candidate FOVs", n_candidates)
            # The pre-scan run writes nothing to disk itself: the decision streams
            # via frameReady, and (when save_pre_scan_omezarr is set) the worker
            # writes the per-step reconstruction to <name>_prescan.ome.zarr.
            #
            # Time the whole call: teardown_sequence drains the outstanding decisions
            # before mda.run() returns, so this span covers imaging AND every FOV's
            # reconstruction/segmentation/scoring -- i.e. the real cost of the pre-scan,
            # not just the stage-and-camera time.
            prescan_started = time.monotonic()
            self._run_mda(prescan_seq, None, write_summary=False)
            prescan_elapsed = time.monotonic() - prescan_started
            logger.info(
                "FOV-selection pre-scan finished in %s (%d FOVs, %.1f s/FOV)",
                _format_duration(prescan_elapsed),
                n_candidates,
                prescan_elapsed / n_candidates if n_candidates else float("nan"),
            )

            if fov_cfg.get("calibration_mode", False):
                # Calibration mode stops after the pre-scan: no timelapse is run. Instead
                # the feature viewer opens on the pre-scan's feature matrix so the user can
                # pick features, tune the score function, and save a ranking profile to
                # drive a later standard (pre-scan + timelapse) acquisition.
                logger.info(
                    "FOV-selection calibration mode: skipping the timelapse run and "
                    "opening the feature viewer."
                )
                self._launch_feature_viewer(self._fov_calibration_csv)
                logger.info("Calibration pre-scan completed successfully")
                return

            passed = list(self._fov_passed_names)
            if not passed:
                logger.warning("FOV selection: no FOVs passed; skipping the timelapse run.")
                return
            timelapse_seq = build_timelapse_sequence(sequence, prescan_seq, passed)
            self._save_selected_fov_config(timelapse_seq, output_dir)
            self._run_mda(timelapse_seq, data_path, write_summary=True)

        logger.info("Acquisition completed successfully")

    def _save_selected_fov_config(self, timelapse_seq: MDASequence, output_dir: Path) -> None:
        """Record the acquisition config with the SELECTED FOVs filled into ``stage_positions``.

        The config an FOV-selection experiment starts from leaves ``stage_positions`` empty --
        the candidates live under ``fov_selection.prescan_mda`` and the real positions are only
        known after the pre-scan. This writes the same sequence with that gap filled: one entry
        per selected FOV carrying its absolute ``x``/``y``, the well's ``ZDrive`` coarse focus,
        and its ``plate_row``/``plate_col``.

        Saved as ``config_<experiment folder>.yaml`` in the experiment folder, beside the
        hand-written ``config.yaml`` it mirrors. A purely descriptive record of what the run
        chose -- nothing reads it back. Because the name follows the FOLDER rather than the
        acquisition, a second acquisition in the same folder would land on it; the
        deduplication index is appended (``config_<folder>_1.yaml``) when the engine had to
        bump the acquisition name, so an existing record is not silently replaced.

        ``exclude_defaults`` keeps the file close to the hand-written config rather than
        expanding every useq default. The ``setup.action`` type discriminator is restored by
        hand -- pydantic drops it as a default, and without it the emitted YAML would not be
        valid against the ``Action`` union even for inspection.

        Never raises -- this is a record written next to the data, and a failure to write it
        must not take the acquisition down between the pre-scan and the timelapse.
        """
        import yaml

        stem = f"config_{output_dir.name}"
        if self._run_index is not None:
            stem = f"{stem}_{self._run_index}"
        path = output_dir / f"{stem}.yaml"
        try:
            data = timelapse_seq.model_dump(mode="json", exclude_defaults=True)
            setup = data.get("setup")
            if isinstance(setup, dict) and isinstance(setup.get("action"), dict):
                setup["action"].setdefault("type", timelapse_seq.setup.action.type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception(
                "FOV selection: could not write the selected-FOV config to %s; the "
                "acquisition is unaffected",
                path,
            )
            return
        logger.info(
            "FOV selection: wrote the acquisition config with %d selected FOVs to %s",
            len(timelapse_seq.stage_positions),
            path,
        )

    def _launch_feature_viewer(self, csv_path: Path | None) -> None:
        """Open the FOV feature viewer on a calibration pre-scan's feature matrix.

        Launched as a detached subprocess (``python -m
        shrimpy.fov_selection.feature_viewer <csv>``) so its Qt event loop stays clear of
        the acquisition process. Never raises: the calibration data is already on disk, so a
        failure to launch is logged with the manual command rather than taking the run down.
        """
        if csv_path is None or not Path(csv_path).exists():
            logger.warning(
                "FOV-selection calibration: feature matrix %s was not written; open the "
                "viewer manually once the CSV exists: "
                "`python -m shrimpy.fov_selection.feature_viewer <csv>`.",
                csv_path,
            )
            return
        import subprocess
        import sys

        csv_path = Path(csv_path)
        logger.info("FOV-selection calibration: launching the feature viewer on %s", csv_path)
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "shrimpy.fov_selection.feature_viewer",
                    "--start-tab",
                    "rank",
                    str(csv_path),
                ]
            )
        except Exception:
            logger.exception(
                "FOV-selection calibration: could not launch the feature viewer; open it "
                "manually: `python -m shrimpy.fov_selection.feature_viewer %s`.",
                csv_path,
            )

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
                        json.dumps(to_builtins(meta), indent=2, default=str) + "\n"
                    )

            self.mmcore.mda.events.sequenceStarted.connect(_write_summary_metadata)

        self.mmcore.mda.run(
            sequence,
            output=out_settings,
            dimension_overrides={"z": {"chunk_size": min(512, sequence.sizes["z"])}},
            overwrite=False,
        )


def _format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``'42.3s'`` / ``'7m 12s'`` / ``'1h 03m 20s'``.

    A pre-scan over a plate runs from seconds to hours, and a bare float of seconds is
    hard to read at the top of that range -- so the unit scales with the magnitude.
    """
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


# Upper bound on the dedup search. Reaching it means something is generating names in a
# loop rather than a human running experiments; better to fail loudly than spin forever.
MAX_ACQUISITION_INDEX = 10_000


def acquisition_artifact_paths(output_dir: Path, name: str, run_index: int) -> list[Path]:
    """Every path an acquisition called ``name`` would write in ``output_dir``.

    The output store plus the FOV-selection siblings (``<base>_fov_debug/``,
    ``<base>_prescan.ome.zarr``). A name is only free when ALL of these are free.
    """
    from shrimpy.fov_selection.manager import sibling_artifact_paths

    data_path = output_dir / f"{name}.ome.zarr"
    return [data_path, *sibling_artifact_paths(data_path, run_index)]


def _get_next_acquisition_name(output_dir: Path, name: str) -> str:
    """Return ``name`` with the next free ``_<idx>`` suffix (``acq_1``, ``acq_2``, ...).

    The index is ALWAYS appended -- the bare ``name`` is never used as a store name. This
    keeps every acquisition in a folder consistently numbered, so runs sort and read as a
    series rather than "the first one" plus numbered stragglers.

    Guards an acquisition from crashing (the zarr writer refuses to overwrite) or
    silently clobbering a previous experiment: the index is bumped until a fully unused
    name is found.

    "Free" deliberately means *no artifact of that name exists*, not *no complete
    acquisition of that name exists*. Completeness is not knowable and not the point: a
    run that dies mid-pre-scan writes ``<name>_fov_debug/`` and possibly
    ``<name>_prescan.ome.zarr`` but never creates ``<name>.ome.zarr`` (the pre-scan run
    passes ``output=None``). Testing only the store would hand the next run the same
    name, and its worker would append rows to the dead run's ``fov_summary.csv`` and
    reuse its debug directory. Leftovers are never reused or cleaned up -- a new name is
    always allocated, and the incomplete folder is left untouched for inspection.

    Parameters
    ----------
    output_dir : Path
        Output directory where acquisitions are saved.
    name : str
        Base acquisition name.

    Returns
    -------
    str
        A name none of whose artifacts exist (e.g. ``acq_1``, ``acq_2``, ...).
    """
    conflicts: list[Path] = []
    for run_index in range(1, MAX_ACQUISITION_INDEX + 1):
        candidate = f"{name}_{run_index}"
        taken = [
            p
            for p in acquisition_artifact_paths(output_dir, candidate, run_index)
            if p.exists()
        ]
        if not taken:
            if conflicts:
                logger.info(
                    "Acquisition name %r is already in use (found %s); using %r instead",
                    name,
                    ", ".join(sorted(p.name for p in conflicts)),
                    candidate,
                )
            return candidate
        conflicts.extend(taken)
    raise RuntimeError(
        f"Could not find a free acquisition name for {name!r} in {output_dir} after "
        f"{MAX_ACQUISITION_INDEX} attempts."
    )
