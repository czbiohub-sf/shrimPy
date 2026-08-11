# DynaTrack

DynaTrack keeps a moving biological structure centered in the field of view
across a long timelapse. After each position's z-stack is acquired, it measures
how far the structure has drifted from where it should be and nudges that
position's stage coordinates so the next visit re-centers it. It was built to
follow zebrafish neuromasts as the embryo grows, but nothing in the algorithm
is neuromast-specific.

DynaTrack is microscope-agnostic. An acquisition engine builds a
[`DynaTrack`](manager.py) coordinator from its metadata and drives it through a
handful of calls; the engine never touches the tracking internals.

## Working principle

The acquisition proceeds as a normal multi-dimensional acquisition (timepoints
× positions × channels × z). DynaTrack layers a feedback loop on top:

1. **Buffer.** As frames stream in, DynaTrack collects the z-slices for the
   configured `input_channel` (an acquisition channel name, e.g. `BF`) of each
   `(timepoint, position)`. When a stack is complete it is handed off for
   analysis.
2. **Preprocess (optional).** The raw stack can be transformed before shift
   estimation — deskewed (light-sheet), phase-reconstructed, and/or
   virtually stained — so tracking runs on the most informative representation.
   The channel tracked is set by `tracking_channel`: without VS it names an
   input channel (its raw / deskewed / deskew+phase volume, per
   `preprocessing`); with VS it names one of `virtual_staining.target_channels`
   (e.g. `nuclei`, `membrane`).
3. **Estimate shift.** The chosen [tracking method](#tracking-methods) computes
   a translational offset (Z, Y, X, in pixels) between the current stack and a
   target — either a stored reference stack or the ROI centre.
4. **Convert to stage microns.** Pixel shifts are scaled to microns using the
   XY pixel size (`core.getPixelSizeUm()`) and Z step (the sequence's
   `z_plan.step`) — a single source of truth, not config fields — then mapped
   from image axes to stage axes via `image_to_stage_matrix_xyz` (a
   per-microscope rotation/flip). Optional `shift.limits` (deadband + clip) and
   `shift.dampening` tame noise and overshoot.
5. **Correct the position.** The stage coordinates the stack was acquired at are
   the baseline; the measured drift is *subtracted* from them (the stage moves
   opposite to the drift to re-centre) and written back to the position store.
   The next event for that position is issued at the corrected coordinates.

Anchoring corrections to the acquisition baseline — the coordinates actually
commanded for that stack — rather than to the live store value is what keeps
late-arriving asynchronous corrections from accumulating against a target that
has already moved.

### Isolation and backpressure

The heavy computation (deskew, phase, virtual staining, FFTs) runs in a
**separate worker subprocess**. This keeps torch's OpenMP runtime away from the
sequenced camera readout — running both in one process segfaults — and gives
the GPU work its own context. Frame data is shuttled to the worker one stack at
a time.

Updates are asynchronous, but at each **timepoint boundary** the coordinator
*drains* any in-flight update before the next timepoint starts. This applies
backpressure: a slow tracker briefly pauses the acquisition between timepoints
instead of letting frame buffers grow without bound.

## Tracking methods

Set `tracking_method` in the config. Methods differ in what they compare
against and whether they need a reference stack.

| Method | Target | Reference stack? | Notes |
| --- | --- | --- | --- |
| `pcc` | previous reference stack | yes | Phase cross-correlation on the raw/preprocessed volume. General-purpose default. |
| `multiotsu_pcc` | previous reference stack | yes | Multi-Otsu threshold → phase cross-correlation on the binary masks. Robust when intensity varies. |
| `multiotsu_center_of_mass` | reference mask centroid | yes | Multi-Otsu threshold → area-weighted centroid difference. The reference is only a fixed target centroid, so re-anchoring is unnecessary. |
| `intensity_center_of_mass` | ROI centre | **no** | Intensity-weighted centroid vs. the volume's geometric centre in deskew space. Referenceless; corrects from the first timepoint. |
| `roi_center_pcc` | ROI centre | **no** | Cross-correlate against a synthetic Gaussian blob centred on the ROI centre. Referenceless; corrects from the first timepoint. |

**Reference-based methods** (`pcc`, `multiotsu_pcc`, `multiotsu_center_of_mass`)
store the first stack per position as the reference and measure drift relative
to it. For long timelapses where the sample changes enough that matching a
stale reference degrades, `reference_update_interval: N` re-anchors the
reference every N timepoints (0 = never). On a re-anchor timepoint the current
stack becomes the new reference and **no** correction is applied.

**Referenceless methods** (`intensity_center_of_mass`, `roi_center_pcc`) have no
reference to re-anchor — their target is always the ROI centre — so
`reference_update_interval` is ignored, and they apply a correction on every
timepoint.

### Method-specific parameters

- `segmentation.otsu_sigma`, `segmentation.otsu_component` — Gaussian blur sigma
  and which multi-Otsu threshold (0 = lower, 1 = upper/brightest) for the
  `multiotsu_*` methods.
- `roi_center.blob_sigma` — Gaussian blob radius (px) for the `roi_center_pcc`
  template; set roughly to the structure radius.
- `roi_center.background_percentile`, `roi_center.blur_sigma` — for
  `intensity_center_of_mass`: subtract a background floor and/or blur before
  weighting so a uniform pedestal or speckle doesn't pull the centroid toward
  the geometric centre.

## Configuration

DynaTrack is configured in the `metadata.dynatrack` section of an acquisition
config, mapping directly onto [`DynaTrackConfig`](tracking.py) fields. `enabled`, `input_channel`, and
`z_device` sit alongside the tracking parameters:

The XY pixel size and Z step are **not** config fields — they are derived at
runtime from `core.getPixelSizeUm()` and `z_plan.step` (single source of truth)
and injected into `deskew` / `phase` and the px→µm conversion.

```yaml
metadata:
  dynatrack:
    enabled: true
    input_channel: BF        # required; acquisition channel name fed to the tracker
    z_device: ObjectiveZ     # Z written to this device's Position property
    tracking_channel: BF     # required; channel the shift is estimated on
    tracking_method: pcc
    tracking_interval: 1
    # ... preprocessing, deskew, phase, virtual_staining, etc.
```

The section is validated even when `enabled: false`, so `input_channel` and
`tracking_channel` must be present; omit the whole section to disable tracking.

See [`config/mda/mantis/dynatrack_demo.yaml`](../../config/mda/mantis/dynatrack_demo.yaml)
for a fully commented example.

## Package layout

| Module | Responsibility |
| --- | --- |
| [`manager.py`](manager.py) | `DynaTrack` — the engine-facing coordinator (frame buffering, worker lifecycle, position corrections). |
| [`tracking.py`](tracking.py) | `DynaTrackConfig` and `DynaTrackUpdater` — the tracking algorithms and shift estimation. |
| [`worker.py`](worker.py) | `DynaTrackWorker` — the subprocess that runs preprocessing and shift estimation. |
| [`position_update.py`](position_update.py) | Internal position-update infrastructure: `PositionStore`, `PositionUpdater` (extension point for custom trackers), `PositionUpdateManager`. |

The deskew → phase → virtual-staining callable is built by
[`shrimpy/preprocessing.py`](../preprocessing.py), which is shared with FOV
selection rather than owned by this package.

## Integrating with a new engine

```python
from shrimpy.config import ShrimpyMetadata
from shrimpy.dynatrack import DynaTrack

# in setup_sequence, before hardware setup:
meta = ShrimpyMetadata.from_sequence(sequence)
self._dynatrack = DynaTrack.from_config(
    meta.dynatrack, sequence, data_path=self._data_path, pixel_size_um=core.getPixelSizeUm()
)
if self._dynatrack is not None:
    core.mda.events.frameReady.connect(self._dynatrack.on_frame_ready)

# after the ROI has been applied (so frame shape is known):
if self._dynatrack is not None:
    zyx_shape = (n_z, core.getImageHeight(), core.getImageWidth())
    self._dynatrack.start(zyx_shape=zyx_shape, log_file_path=log_file)

# in the event iterator:
if last_t is not None and t_idx != last_t:
    self._dynatrack.drain_pending()          # backpressure at timepoint boundary
event = self._dynatrack.apply_position_update(event)

# in teardown_sequence:
core.mda.events.frameReady.disconnect(self._dynatrack.on_frame_ready)
self._dynatrack.shutdown()
```

`from_config` returns `None` when tracking is disabled or the sequence has no
stage positions, so the engine only wires up the callbacks when tracking is
active. To run a **custom tracker** (or drive tracking in-process for tests),
construct the coordinator directly with your own `PositionUpdater`:

```python
DynaTrack(config, sequence, updater=MyUpdater())  # runs in-process, no subprocess
```
