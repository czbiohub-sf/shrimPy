"""AcquisitionController for autofocus-aware, reactive microscopy.

This controller owns the autofocus decision loop and uses stream.skip()
from ome-writers to handle autofocus failures.
It also provides a foundation for smart microscopy where acquisition
events can be modified on-the-fly based on acquired data.
"""

from __future__ import annotations

import logging
import threading

from queue import Queue

import numpy as np

from ome_writers import OMEStream
from pymmcore_plus.core._sequencing import SequencedEvent
from useq import MDAEvent, MDASequence

from shrimpy.mantis.mantis_engine import MantisEngine

logger = logging.getLogger(__name__)


def _event_index(event: MDAEvent) -> dict:
    """Extract the event index, using the first sub-event for SequencedEvents."""
    if isinstance(event, SequencedEvent):
        return dict(event.events[0].index)
    return dict(event.index)


def _n_frames(event: MDAEvent) -> int:
    """Return the number of frames an event will produce."""
    return len(event.events) if isinstance(event, SequencedEvent) else 1


class AcquisitionController:
    """Controls acquisition flow: autofocus, event queuing, frame skipping.

    For each event from the sequence (after hardware sequencing is resolved):
    1. Detect new XY positions
    2. Engage autofocus
    3. If autofocus succeeds: queue the event for normal execution by the MDA runner
    4. If autofocus fails: call stream.skip() for the corresponding frames

    The controller feeds events to mda.run() via a Queue, enabling
    smart microscopy where events are modified based on acquired data.

    Parameters
    ----------
    engine : MantisEngine
        The engine instance (provides hardware control and autofocus methods).
    stream : OMEStream
        The ome-writers stream for appending frames and skipping on failure.
    sequence : MDASequence
        The acquisition sequence to execute.
    """

    STOP_EVENT = object()

    def __init__(self, engine: MantisEngine, stream: OMEStream, sequence: MDASequence):
        self.engine = engine
        self.stream = stream
        self.sequence = sequence
        self._queue: Queue = Queue()
        self._event_finished = threading.Event()
        self._event_finished.set()  # initially no pending event

    def run(self) -> None:
        """Run the acquisition with autofocus-aware event control."""
        core = self.engine.mmcore

        # Setup hardware before starting the MDA runner. This applies
        # mantis-specific settings (ROI, initialization, autofocus config).
        # The MDA runner will call setup_sequence again with a placeholder
        # GeneratorMDASequence, but that has no metadata so it's a no-op.
        self.engine.setup_sequence(self.sequence)

        # Connect stream.append to frameReady
        @core.mda.events.frameReady.connect
        def _on_frame_ready(frame: np.ndarray, _event: MDAEvent, _meta: dict) -> None:
            self.on_frame_ready(frame, _event, _meta)

        # Signal when an event finishes so the controller can safely skip
        @core.mda.events.eventFinished.connect
        def _on_event_finished(_event: MDAEvent) -> None:
            self._event_finished.set()

        # Start MDA runner in background with queue iterator
        queue_iter = iter(self._queue.get, self.STOP_EVENT)
        mda_thread = core.run_mda(queue_iter)
        logger.info("Starting acquisition sequence")

        try:
            # Iterate over events from event_iterator (handles hardware sequencing).
            # SequencedEvents bundle z-slices; plain MDAEvents are individual frames.
            for event in self.engine.event_iterator(self.sequence):
                idx = _event_index(event)
                n = _n_frames(event)

                self.engine._adjust_xy_stage_speed(event)
                self.engine._set_event_xy_position(event)
                self.engine._engage_autofocus(event)

                if not self.engine._use_autofocus or self.engine._autofocus_success:
                    # Wait for previous event to finish before queuing the next
                    self._event_finished.wait()
                    self._event_finished.clear()
                    self._queue.put(event)
                    logger.debug(f"Queued event {idx} ({n} frame(s))")
                else:
                    # Wait for MDA thread to finish the previous event
                    # before calling skip, to keep the stream in order
                    self._event_finished.wait()
                    self.stream.skip(frames=n)
                    logger.info(f"Autofocus failed at {idx}, skipped {n} frame(s)")
        finally:
            self._queue.put(self.STOP_EVENT)
            mda_thread.join()
            logger.info("Acquisition completed successfully")

    def on_frame_ready(self, frame: np.ndarray, event: MDAEvent, meta: dict) -> None:
        """Called when a frame is acquired. Override for smart microscopy.

        Parameters
        ----------
        frame : np.ndarray
            The acquired image.
        event : MDAEvent
            The event that produced this frame.
        meta : dict
            Frame metadata.
        """
        self.stream.append(frame)
