"""Shared MDA engine for shrimPy microscopes.

:class:`BaseEngine` collects the acquisition behavior that is common to every
microscope shrimPy drives (mantis, iSIM, Dragonfly, ...):

- hardware-sequenced acquisition defaults (``use_hardware_sequencing``,
  ``force_set_xy_position``),
- verbose hardware logging (property changes, ROI changes, XY stage moves),
- continuous-autofocus handling, including the simulated ``demo-PFS`` method
  used with the Micro-Manager demo config, deciding when to engage for
  sequenced vs. single events, and skipping events whose autofocus did not
  engage,
- resetting hardware properties in ``teardown_sequence``,
- the smart-microscopy features shared by all microscopes — currently DynaTrack
  position tracking, switched on from ``metadata.dynatrack``,
- the :meth:`BaseEngine.acquire` entry point that runs an ``MDASequence`` and
  writes OME-Zarr.

Microscope-specific engines subclass it and override the pieces that differ:

- :meth:`BaseEngine.engage_autofocus` — the hardware autofocus routine. The
  base implementation raises ``NotImplementedError``; it is only reached when
  autofocus is enabled with a method other than ``demo-PFS``.
- ``setup_sequence`` / ``setup_event`` / ``teardown_sequence`` — call
  ``super()`` and add microscope-specific hardware setup around it.

See :mod:`shrimpy.engines.mantis_engine` for the reference implementation.
"""

from __future__ import annotations

import logging
import os

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psutil

from ome_writers import AcquisitionSettings
from pymmcore_plus.core import CMMCorePlus
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import SummaryMetaV1
from useq import Axis, MDAEvent, MDASequence

from shrimpy._logging import find_log_file
from shrimpy.config import ShrimpyMetadata, load_config
from shrimpy.dynatrack import DynaTrack

logger = logging.getLogger(__name__)

DEMO_PFS_METHOD = "demo-PFS"
DEMO_PFS_SUCCESS_RATE = 0.5  # probability that a demo-PFS call succeeds

# The axes that identify one autofocus position. A new stage position or grid
# site moves the sample in XY, where the focal plane can differ; a new
# timepoint revisits a position the sample may have drifted away from since.
# Stepping the channel or the Z slice does neither, so the focus obtained on
# arrival is reused for every channel and slice acquired there.
AUTOFOCUS_POSITION_AXES = frozenset({Axis.TIME, Axis.POSITION, Axis.GRID})

_PROC = psutil.Process(os.getpid())


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024**3)


def first_event(event: MDAEvent) -> MDAEvent:
    """Return the first sub-event of a ``SequencedEvent``, or ``event`` itself.

    Engine hooks receive either a single :class:`~useq.MDAEvent` or a
    :class:`~pymmcore_plus.core._sequencing.SequencedEvent` bundling the frames
    of one hardware-sequenced burst; the burst's index, position and properties
    are those of its first sub-event.
    """
    return event.events[0] if isinstance(event, SequencedEvent) else event


def num_frames(event: MDAEvent) -> int:
    """Return the number of frames ``event`` acquires."""
    return len(event.events) if isinstance(event, SequencedEvent) else 1


class BaseEngine(MDAEngine):
    """Base MDA engine shared by all shrimPy microscopes.

    Parameters
    ----------
    mmc : CMMCorePlus
        The Micro-Manager core instance. The engine registers itself with
        ``mmc.mda`` and connects to the core's property / ROI / stage signals
        for logging.
    *args, **kwargs
        Forwarded to :class:`~pymmcore_plus.mda.MDAEngine`. shrimPy defaults
        ``use_hardware_sequencing`` to True and ``force_set_xy_position`` to
        False; subclasses may set microscope-specific defaults (e.g. acquisition
        timeouts) before calling ``super().__init__()``.
    """

    #: Device properties excluded from debug logging. ``BaseEngine`` logs every
    #: property change; subclasses override this with the microscope-specific
    #: properties that change too often to be useful in the log.
    NOISY_PROPERTIES: tuple[str, ...] = ()

    def __init__(self, mmc: CMMCorePlus, *args, **kwargs):
        kwargs.setdefault("use_hardware_sequencing", True)
        kwargs.setdefault("force_set_xy_position", False)
        super().__init__(mmc, *args, **kwargs)
        self._use_autofocus = False
        self._autofocus_success = False
        self._autofocus_stage = None
        self._autofocus_method = None
        self._autofocus_fail_at_index = None
        self._home_focus_device = False
        # Position key of the last event autofocus was attempted for; see
        # _autofocus_position_key(). None means "not attempted yet this run".
        self._last_autofocus_position: tuple | None = None
        # Core-Focus device and the position it held before any z_plan move; see
        # _capture_focus_home(). Both None when focus homing does not apply.
        self._focus_device: str | None = None
        self._focus_home: float | None = None
        self._xy_stage_device = None
        self._data_path: Path | None = None
        self._dynatrack: DynaTrack | None = None

        # Register event callbacks for logging
        mmc.mda.set_engine(self)
        mmc.events.propertyChanged.connect(self._on_property_changed)
        mmc.events.roiSet.connect(self._on_roi_set)
        mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)

    # ------------------------------------------------------------------
    # Logging callbacks
    # ------------------------------------------------------------------

    def _on_property_changed(self, device: str, property_name: str, value: str) -> None:
        """Log property changes at debug level.

        Properties named in :attr:`NOISY_PROPERTIES` are skipped, which
        lets a microscope filter out its own high-frequency device properties.
        """
        if property_name in self.NOISY_PROPERTIES:
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

    # ------------------------------------------------------------------
    # MDAEngine protocol
    # ------------------------------------------------------------------

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Configure shared hardware settings before the sequence starts.

        The microscope settings are read from ``sequence.metadata`` and
        validated by :class:`~shrimpy.config.ShrimpyMetadata`; missing sections
        fall back to their defaults (autofocus and DynaTrack disabled).
        """
        logger.info("Setting up hardware for acquisition sequence")

        core = self.mmcore
        meta = ShrimpyMetadata.from_sequence(sequence)

        # Set autofocus settings. The position key is per-run state: FOV
        # selection runs two sequences through one engine, and the timelapse
        # must not inherit the last position of the pre-scan.
        self._last_autofocus_position = None
        autofocus = meta.autofocus
        if autofocus.enabled:
            self._use_autofocus = True
            self._autofocus_stage = autofocus.stage
            self._autofocus_method = autofocus.method
            self._home_focus_device = autofocus.home_focus_device
            logger.info(f"Enabling autofocus with method: {self._autofocus_method}")
            if not self._autofocus_method == DEMO_PFS_METHOD:
                core.setAutoFocusDevice(self._autofocus_method)
        else:
            self._home_focus_device = False
            logger.info("Autofocus is disabled for this acquisition")

        # Store XY stage device name
        self._xy_stage_device = core.getXYStageDevice()
        logger.debug(f"XY stage device: {self._xy_stage_device}")

        # Call parent setup so SummaryMetaV1 captures the fully configured
        # hardware state and the setup event applies the ROI.
        result = super().setup_sequence(sequence)

        # Deferred to after the setup event: the Core-Focus device it selects is
        # the one whose home position we track.
        self._capture_focus_home()

        self._setup_dynatrack(meta, sequence)

        return result

    def setup_event(self, event: MDAEvent) -> None:
        """Move to the event position, engage autofocus, and prepare hardware.

        ``event`` is either a single :class:`~useq.MDAEvent` or a
        :class:`~pymmcore_plus.core._sequencing.SequencedEvent` bundling the
        frames of one hardware-sequenced burst. Autofocus engages once per
        burst, or once per position for single events — see
        :meth:`_should_engage_autofocus`. When it fails, every frame the event
        would have acquired is skipped.
        """
        # Move the XY stage, then wait for it: _set_event_xy_position does not
        # block. The single events of one Z-stack all carry the same XY position,
        # so only the first of them moves.
        if self._should_move_xy(event):
            self._set_event_xy_position(event)
            self.mmcore.waitForDevice(self._xy_stage_device)

        # Return the Core-Focus device to its home position before autofocus, so
        # the plane autofocus locks onto is the same for every position instead
        # of wherever the previous Z-stack left the device. This is also what
        # returns the device home at the *end* of a Z-stack: the stack being
        # unwound here is the one that just finished.
        if self._use_autofocus and self._should_engage_autofocus(event):
            self._return_focus_device_home("before autofocus at a new position")

        # Engage autofocus
        self._engage_autofocus(event)

        # Skip acquisition if autofocus failed. For single events, the outcome of
        # the last engagement stands, so the whole position is skipped, not only
        # the frame at which autofocus was attempted.
        if self._use_autofocus and not self._autofocus_success:
            raise SkipEvent(num_frames=num_frames(event), reason="autofocus failed")

        self._log_memory_usage()

        # Call parent setup_event
        super().setup_event(event)

    def _should_move_xy(self, event: MDAEvent) -> bool:
        """Return whether the XY stage should be moved for ``event``.

        Mirrors the guard inside ``MDAEngine._set_event_xy_position``, which is
        a no-op when there is no XY stage, when the event carries no XY
        position, or when that position equals the last commanded one and
        ``force_set_xy_position`` is False. Deciding here rather than letting
        the move no-op keeps the blocking ``waitForDevice`` out of the events
        that do not move — the Z slices of a stack all repeat one position.

        Must be called *before* the move, while the last commanded position is
        still the previous one.
        """
        if not self._xy_stage_device:
            return False

        event_x, event_y = event.x_pos, event.y_pos
        if event_x is None and event_y is None:
            return False
        if self.force_set_xy_position:
            return True

        last_x, last_y = self.mmcore._last_xy_position.get(None) or (None, None)
        return not (
            (event_x is None or event_x == last_x) and (event_y is None or event_y == last_y)
        )

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Return the hardware to a safe idle state after the sequence."""
        # The last Z-stack of the run has no successor to unwind it in
        # setup_event, so close it out here. Before super(), whose
        # _restore_initial_state() may also move Z.
        self._return_focus_device_home("end of sequence")

        self._teardown_dynatrack()

        super().teardown_sequence(sequence)

        core = self.mmcore
        meta = ShrimpyMetadata.from_sequence(sequence)

        if reset_hardware_sequencing_settings := meta.reset_hardware_sequencing_settings:
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

    def _log_memory_usage(self) -> None:
        """Log process memory and circular buffer occupancy at debug level."""
        free_capacity = self.mmcore.getBufferFreeCapacity()
        total_capacity = self.mmcore.getBufferTotalCapacity()
        logger.debug(f"Circular buffer capacity: {free_capacity} / {total_capacity} frames")
        logger.debug(
            f"{type(self).__name__}[mem]: setup_event rss={_rss_gb():.2f} GB "
            f"mm_buf_used={total_capacity - free_capacity}/{total_capacity}"
        )

    # ------------------------------------------------------------------
    # Autofocus
    # ------------------------------------------------------------------

    def _capture_focus_home(self) -> None:
        """Record the Core-Focus device's position before any z_plan move.

        This is the plane autofocus is engaged at, and the plane the device is
        returned to at the end of every Z-stack (see
        :meth:`_return_focus_device_home`). Without it the focal reference is
        wherever the previous Z-stack happened to stop — on the Dragonfly the
        piezo sat at the top of the previous stack, so AFC locked onto the
        *top* slice and the whole stack was acquired below focus.

        Must be called after ``super().setup_sequence()``: the sequence's
        ``setup`` event is what applies ``Core-Focus``, so reading it earlier
        returns whichever stage the Micro-Manager config defaults to.

        Opt-in, via ``metadata.autofocus.home_focus_device``: the Core-Focus
        device differing from the autofocus stage does not by itself mean the
        two interact. On mantis they do not — Core-Focus is the light-sheet
        scan galvo, which moves neither sample nor objective — so homing there
        would only add a redundant galvo move before every sequenced burst.
        Homing is also skipped when Core-Focus *is* the autofocus stage, where
        there is no residual z_plan offset to neutralize and moving the device
        would fight the z_plan.
        """
        core = self.mmcore
        self._focus_device = None
        self._focus_home = None

        if not self._use_autofocus or not self._home_focus_device:
            return

        focus_device = core.getFocusDevice()
        if not focus_device:
            logger.debug("No Core-Focus device; not tracking a focus home position")
            return
        if focus_device == self._autofocus_stage:
            logger.debug(
                f"Core-Focus device {focus_device!r} is the autofocus stage; "
                "the z_plan and autofocus drive the same device, so no homing"
            )
            return

        try:
            home = core.getPosition(focus_device)
        except Exception:
            logger.exception(
                f"Could not read the position of Core-Focus device {focus_device!r}; "
                "autofocus will engage wherever the z_plan leaves it"
            )
            return

        self._focus_device = focus_device
        self._focus_home = home
        logger.info(
            f"Core-Focus device {focus_device!r} home position: {home} um. Autofocus "
            f"engages here (autofocus stage: {self._autofocus_stage!r}) and the device "
            "returns here at the end of each Z-stack."
        )

    def _return_focus_device_home(self, reason: str) -> None:
        """Move the Core-Focus device back to its captured home position.

        A no-op unless :meth:`_capture_focus_home` found a home to track, or
        when the device is already there — which is the common case for the
        first Z-stack of a run.
        """
        if self._focus_device is None or self._focus_home is None:
            return

        core = self.mmcore
        try:
            current = core.getPosition(self._focus_device)
        except Exception:
            logger.exception(f"Could not read {self._focus_device} position ({reason})")
            current = None

        if current == self._focus_home:
            logger.debug(
                f"{self._focus_device} already at its home position "
                f"{self._focus_home} um ({reason})"
            )
            return

        logger.debug(
            f"Returning {self._focus_device} from {current} to its home position "
            f"{self._focus_home} um ({reason})"
        )
        try:
            core.setPosition(self._focus_device, self._focus_home)
            core.waitForDevice(self._focus_device)
        except Exception:
            logger.exception(
                f"Failed to return {self._focus_device} to {self._focus_home} um ({reason})"
            )

    @staticmethod
    def _autofocus_position_key(event: MDAEvent) -> tuple:
        """Return the axes of ``event.index`` that identify one focus position.

        Only the time, position and grid indices are kept (see
        ``AUTOFOCUS_POSITION_AXES``), so every event acquired at one XY location
        within one timepoint maps to the same key regardless of its channel or
        Z slice. Keys are sorted so that they compare equal independently of
        index insertion order.
        """
        return tuple(
            sorted(
                (str(axis), index)
                for axis, index in event.index.items()
                if axis in AUTOFOCUS_POSITION_AXES
            )
        )

    def _should_engage_autofocus(self, event: MDAEvent) -> bool:
        """Return whether autofocus should be engaged for ``event``.

        A ``SequencedEvent`` is acquired as one hardware-triggered burst, so
        autofocus engages once, before the burst starts.

        Single events are delivered one frame at a time, so autofocus engages
        once per *position* — the first event whose
        :meth:`_autofocus_position_key` differs from the last one attempted.
        Only the time, position and grid axes form that key, so stepping the
        channel or the Z slice reuses the focus obtained on arrival, while a new
        timepoint re-engages.

        Keying on Z alone would re-engage once per channel, because the channel
        axis is outer to Z: at one XY location a 4-channel Z-stack presents
        ``c=0,z=0``, ``c=1,z=0``, ``c=2,z=0``, ``c=3,z=0``, all of which are
        ``z == 0``.

        A sequence with none of those axes has one focus position, so it
        engages on the first event and not again.

        Pure: the position is recorded by :meth:`_engage_autofocus`, which also
        logs the reuse, so :meth:`setup_event` can ask this question first —
        to decide whether to home the focus device — without emitting a
        duplicate log line.
        """
        if isinstance(event, SequencedEvent):
            return True

        return self._autofocus_position_key(event) != self._last_autofocus_position

    def _engage_autofocus(self, event: MDAEvent) -> None:
        """Engage autofocus for ``event``, recording the outcome.

        Does nothing when autofocus is disabled or when
        :meth:`_should_engage_autofocus` rejects the event, in which case the
        outcome of the previous engagement stands. Otherwise dispatches to the
        simulated :meth:`_engage_demo_pfs` when the configured method is
        ``demo-PFS``, and to the microscope-specific :meth:`engage_autofocus`
        otherwise. The outcome is stored in ``self._autofocus_success``;
        :meth:`setup_event` skips the event when autofocus is enabled but did
        not engage.
        """
        if not self._use_autofocus:
            logger.debug("Autofocus is disabled.")
            return

        if not self._should_engage_autofocus(event):
            logger.debug(
                f"Autofocus already engaged for this position; reusing it for "
                f"index={dict(event.index)}"
            )
            return

        # Record the position before dispatching, not after: a failed attempt
        # must not be retried for every remaining channel and slice of this
        # position. _autofocus_success carries the outcome, and setup_event
        # skips the whole position on failure.
        if not isinstance(event, SequencedEvent):
            self._last_autofocus_position = self._autofocus_position_key(event)

        if self._autofocus_method == DEMO_PFS_METHOD:
            self._autofocus_success = self._engage_demo_pfs(
                event=event,
                fail_at_index=self._autofocus_fail_at_index,
            )
            return

        self._autofocus_success = bool(self.engage_autofocus(event))

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage the microscope's hardware autofocus for ``event``.

        Subclasses must implement this method; the acquisition of any event for
        which it returns False is skipped (see :meth:`setup_event`). It is only
        called when autofocus is enabled with a method other than ``demo-PFS``.

        Parameters
        ----------
        event : MDAEvent
            The event that is about to be acquired.

        Returns
        -------
        bool
            True if autofocus engaged successfully.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement engage_autofocus(); "
            f"autofocus method {self._autofocus_method!r} is not supported. "
            "Override engage_autofocus() in the microscope engine, disable "
            f"autofocus, or use the {DEMO_PFS_METHOD!r} method."
        )

    def _engage_demo_pfs(
        self,
        event: MDAEvent | None = None,
        success_rate: float = DEMO_PFS_SUCCESS_RATE,
        fail_at_index: list[dict] | None = None,
    ) -> bool:
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

        Returns
        -------
        bool
            True if the simulated autofocus call succeeded.
        """
        if fail_at_index is not None and event is not None:
            # For SequencedEvents, use the first sub-event's index
            event_index = first_event(event).index
            success = not any(
                all(event_index.get(k) == v for k, v in idx.items()) for idx in fail_at_index
            )
        else:
            success = np.random.random() < success_rate

        if success:
            logger.debug(f"{DEMO_PFS_METHOD} call succeeded")
        else:
            logger.debug(f"{DEMO_PFS_METHOD} call failed")

        return success

    def _get_autofocus_z_position(self, event: MDAEvent) -> float:
        """Return the target Z position of the autofocus stage for ``event``.

        Z positions are not written to the autofocus stage while autofocus is
        enabled (see :meth:`_set_event_properties`), so the target position is
        read from the event's properties when present, and from the stage's
        current position otherwise. For a ``SequencedEvent``, the position of
        the first frame of the burst is used.
        """
        event = first_event(event)
        if event.properties:
            for dev, prop, value in event.properties:
                if dev == self._autofocus_stage and prop == "Position":
                    return float(value)
        return self.mmcore.getPosition(self._autofocus_stage)

    # ------------------------------------------------------------------
    # DynaTrack position tracking
    # ------------------------------------------------------------------

    def _setup_dynatrack(self, meta: ShrimpyMetadata, sequence: MDASequence) -> None:
        """Build and start DynaTrack, if ``metadata.dynatrack`` enables it.

        The XY pixel size (from the core) and the sequence z_plan step are the
        single source of truth for all scale parameters; DynaTrack derives and
        injects them.

        Called after the parent ``setup_sequence``, and reading the pixel size
        only then, because MM resolves getPixelSizeUm() from whichever
        pixel-size config group currently matches the device property values.
        Read earlier it reflects leftover hardware state (whatever the GUI was
        last left in) rather than the state this acquisition runs with, which
        silently produced a different px_to_scan_ratio between
        otherwise-identical runs — changing the deskewed X extent (the scan
        axis) and stretching every downstream preprocessing result. Any
        grid-plan FOV sizes likewise reflect the setup event's state.
        """
        core = self.mmcore
        self._dynatrack = DynaTrack.from_config(
            meta.dynatrack,
            sequence,
            data_path=self._data_path,
            pixel_size_um=core.getPixelSizeUm(),
        )
        if self._dynatrack is None:
            return

        core.mda.events.frameReady.connect(self._dynatrack.on_frame_ready)
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

        # DynaTrack runs in a worker subprocess for GPU/torch isolation:
        # torch's OpenMP runtime segfaults when it coexists with the sequenced
        # camera readout in the acquisition process. The worker is started after
        # the setup event has applied the ROI, so getImageHeight/Width reflects
        # the actual acquired frame size (also used to build the preprocessor,
        # when configured, inside the worker).
        zyx_shape = (
            max(sequence.sizes.get("z", 1), 1),
            core.getImageHeight(),
            core.getImageWidth(),
        )
        self._dynatrack.start(zyx_shape=zyx_shape, log_file_path=find_log_file())

    def _teardown_dynatrack(self) -> None:
        """Disconnect and shut down DynaTrack, if it is running."""
        if self._dynatrack is None:
            return
        self.mmcore.mda.events.frameReady.disconnect(self._dynatrack.on_frame_ready)
        self._dynatrack.shutdown()
        self._dynatrack = None

    def event_iterator(self, events: Iterable[MDAEvent]):
        """Wrap event iteration to apply position updates before logging.

        By applying position updates here (before the MDA runner emits
        ``eventStarted``), the logged event reflects the corrected
        coordinates rather than the original sequence values.

        At timepoint boundaries the iterator drains any pending DynaTrack
        update so that (a) position corrections are applied before the new
        timepoint starts and (b) frame data does not accumulate unboundedly
        in the executor queue.
        """
        last_t: int | None = None
        for event in super().event_iterator(events):
            if self._dynatrack is not None:
                t_idx = first_event(event).index.get("t", 0)
                if last_t is not None and t_idx != last_t:
                    self._dynatrack.drain_pending()
                last_t = t_idx
                event = self._dynatrack.apply_position_update(event)
            yield event

    # ------------------------------------------------------------------
    # Acquisition entry point
    # ------------------------------------------------------------------

    def acquire(
        self,
        output_dir: str | Path,
        name: str,
        mda_config: MDASequence | str | Path,
    ) -> None:
        """Run an acquisition and write the data as OME-Zarr.

        Parameters
        ----------
        output_dir : str | Path
            Directory where acquisition data will be saved.
        name : str
            Base acquisition name; an index suffix will be appended automatically.
        mda_config : MDASequence | str | Path
            An MDASequence object or path to an acquisition configuration YAML
            file (an MDASequence with the microscope settings under
            ``metadata``; see :mod:`shrimpy.config`).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = _get_next_acquisition_name(output_dir, name)

        if isinstance(mda_config, MDASequence):
            sequence = mda_config
        else:
            logger.info(f"Loading acquisition config from {mda_config}")
            # Validates the shrimPy metadata sections before any hardware setup
            sequence = load_config(mda_config)

        data_path = output_dir / f"{name}.ome.zarr"
        self._data_path = data_path

        # Summary metadata is written by pymmcore-plus into the zarr root group's
        # attributes, under `attributes.pymmcore_plus.summary_metadata`.

        logger.info(f"Starting acquisition: {name}")
        self.mmcore.mda.run(
            sequence,
            output=AcquisitionSettings(
                root_path=data_path, compression="blosc-zstd", format="acquire-zarr"
            ),
            dimension_overrides={"z": {"chunk_size": min(512, sequence.sizes.get("z", 1))}},
            overwrite=False,
        )
        logger.info("Acquisition completed successfully")


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
