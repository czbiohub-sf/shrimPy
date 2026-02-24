# AcquisitionController Plan

## Context

MantisEngine currently overrides `exec_event()` to pad zarr with zeros when autofocus fails. This is fragile (modifies superclass behavior) and doesn't support smart microscopy where events are modified on-the-fly. The goal is to introduce an `AcquisitionController` that owns the autofocus/skip decision loop, uses `stream.skip()` from ome-writers instead of zero-padding, and provides hooks for reactive/smart microscopy.

**Constraint**: `test_autofocus_failure_pads_with_zeros` must still pass — the zarr output should contain zeros at failed (t, p) positions.

## Architecture

```
AcquisitionController (new — owns the reactive loop)
  ├── MantisEngine (hardware setup, no autofocus in setup_event)
  ├── OMEStream (from create_stream, passed in or created via engine.create_stream)
  └── Queue[MDAEvent] (feeds events to mda.run)

MantisEngine (simplified — delegates more to superclass)
  ├── create_stream(sequence, data_path) → OMEStream context manager (refactored from acquire)
  ├── acquire(output_dir, name, mda_config) → convenience wrapper
  ├── setup_sequence() — revert to super() (hardware init moves to controller)
  ├── setup_event() — revert to super() (no autofocus, no XY speed, no stage move)
  ├── exec_event() — revert to super() (remove zero-padding override)
  ├── teardown_sequence() — hardware reset (unchanged)
  ├── _adjust_xy_stage_speed(event) — kept, called by controller
  ├── _engage_autofocus(event) — kept, called by controller
  └── _engage_nikon_pfs / _engage_demo_pfs — kept, called by _engage_autofocus
```

## Files to modify

### 1. `shrimpy/mantis/mantis_engine.py`

**a) Add `create_stream(sequence)` method** — extract stream creation from `acquire()`:
```python
def create_stream(self, sequence: MDASequence, output_dir: Path, name: str):
    """Create an OMEStream for the given sequence. Returns context manager."""
    # Move zarr settings logic here (ROI, chunk shapes, AcquisitionSettings)
    # Returns create_stream(settings) — caller uses `with engine.create_stream(...) as stream:`
```

**b) Remove `setup_event()` override entirely** — revert to `super().setup_event()`. The controller now handles XY stage speed, XY positioning, and autofocus before queuing events. The engine's superclass `setup_event` handles channel config, exposure, z positioning, etc.

**c) Remove `exec_event()` override entirely** — no more zero-padding. The controller calls `stream.skip()` instead.

**d) Simplify `setup_sequence()`** — keep only mantis-specific hardware init (initialization_settings, ROI, hardware sequencing settings, z_stage, autofocus config). Move XY stage device storage and speed adjustment to the controller. The autofocus settings (`_use_autofocus`, `_autofocus_method`, etc.) are still parsed here since the controller reads them from the engine.

**e) Simplify `acquire()`** — simple acquisition without autofocus concerns:
```python
def acquire(self, output_dir, name, mda_config):
    sequence = ...  # parse mda_config as before
    name = _get_next_acquisition_name(output_dir, name)
    data_path = output_dir / f"{name}.ome.zarr"
    with self.create_stream(sequence, data_path) as stream:

        @self.mmcore.mda.events.frameReady.connect
        def _on_frame_ready(frame, _event, _meta):
            stream.append(frame)

        self.mmcore.mda.run(sequence)
```
Note: `acquire()` does NOT use the controller. It runs `mda.run(sequence)` directly — no autofocus, no skip logic. For autofocus-aware acquisition, use `AcquisitionController` directly.

### 2. `shrimpy/mantis/acquisition_controller.py` (new file)

```python
class AcquisitionController:
    """Controls acquisition flow: autofocus, event queuing, frame skipping.

    For each (t, p) group in the sequence:
    1. Move to XY position
    2. Engage autofocus
    3. If autofocus succeeds: queue the events (z-stack, channels) for normal execution
    4. If autofocus fails: call stream.skip(frames=n) for the skipped frames

    The controller feeds events to mda.run() via a Queue, enabling
    smart microscopy where events are modified based on acquired data.
    """

    def __init__(self, engine, stream, sequence):
        self.engine = engine
        self.stream = stream  # OMEStream with .append() and .skip()
        self.sequence = sequence
        self._queue = Queue()

    def run(self):
        core = self.engine.mmcore

        # Connect stream.append to frameReady
        @core.mda.events.frameReady.connect
        def _on_frame_ready(frame, _event, _meta):
            self.stream.append(frame)

        # Start MDA runner in background with queue iterator
        queue_iter = iter(self._queue.get, self.STOP_EVENT)
        core.run_mda(queue_iter)

        # Iterate over events from event_iterator (handles hardware sequencing)
        # SequencedEvents bundle z-slices; plain MDAEvents are individual frames
        for event in self.engine.event_iterator(self.sequence):
            if self._is_new_position(event):
                self.engine._adjust_xy_stage_speed(event)
                self.engine._set_event_xy_position(event)
                self.engine._engage_autofocus(event)

            if self.engine._autofocus_success or not self.engine._use_autofocus:
                self._queue.put(event)  # MDA runner executes normally
            else:
                # Skip all frames in this event
                n = len(event.events) if isinstance(event, SequencedEvent) else 1
                self.stream.skip(frames=n)

        self._queue.put(self.STOP_EVENT)
```

**Key design decisions:**
- Controller iterates the MDASequence directly and decides per-event whether to queue or skip
- XY movement + autofocus happens on the controller thread (main thread), before queuing
- The MDA runner thread only sees events that passed autofocus — no zero-padding needed
- `stream.skip()` advances the zarr write pointer, leaving zeros (zarr fill_value)
- Smart microscopy: subclass and override `on_frame_ready()` / `on_position_complete()` to modify upcoming events

### 3. `shrimpy/tests/test_mantis_engine.py`

- Remove `test_exec_event_autofocus_failure_yields_zero_padded_image` (exec_event no longer does this)
- Update `test_autofocus_demo_pfs_dispatched` if the call signature changed
- Update any tests that relied on autofocus in `setup_event`

### 4. `shrimpy/tests/test_mantis_integration.py`

- `test_autofocus_failure_pads_with_zeros` — should still pass since zarr fill_value is 0 and `stream.skip()` leaves zeros. May need minor adjustments to how `_autofocus_fail_at_index` is set.

## Event flow: before vs after

**Before (current):**
```
mda.run(sequence)
  → engine.setup_event(event)     # moves stage + autofocus
  → engine.exec_event(event)      # if AF failed: yield zeros
  → frameReady → stream.append()
```

**After (controller):**
```
controller.run()
  → for event in sequence:
      → controller: move stage, engage autofocus
      → if AF success: queue.put(event)
        → mda thread: engine.setup_event(event)  # hardware only
        → mda thread: engine.exec_event(event)    # always real acquisition
        → frameReady → stream.append()
      → if AF failure: stream.skip(frames=n_frames_in_group)
```

## Open questions to resolve during implementation

1. **Thread safety**: `stream.skip()` is called from the controller thread while `stream.append()` is called from the MDA runner thread (via frameReady). Need to verify ome-writers thread safety or add locking. Since skip and append are never concurrent for the same (t, p) — the controller either skips OR queues, never both — this may be safe without locking.

## Verification

```bash
# The key test must still pass
uv run pytest shrimpy/tests/test_mantis_integration.py::test_autofocus_failure_pads_with_zeros -v

# Full suite
uv run pytest -v
```
