# FOV-selection integration

The offline FOV-selection pipeline (`shrimpy/fov_selection/`) is integrated into
the Mantis acquisition engine as an **online, adaptive acquisition**: in a single
run the microscope pre-scans every candidate FOV once, decides which are "good"
with the trained model, and images the timelapse only on those.

It is folded into the engine exactly like DynaTrack — the engine reads
`metadata.mantis.fov_selection` and drives everything through `setup_sequence` /
`event_iterator` / `setup_event` / `teardown_sequence`. There is no separate
controller and no CLI routing; `shrimpy acquire mantis` runs it like any other
acquisition.

Developed/validated offline with the ReplayCamera replaying a real A549 store
(raw BF + GFP), which matches what the microscope produces online.

## Experiment flow — ONE streaming run

```
core.mda.run(sequence)   # sequence.stage_positions <- fov_selection.search_mda candidates
│
├─ t = 0  PRE-SCAN      input_channel only (full z), all candidate FOVs
│         └─ frameReady ─► worker subprocess (per FOV, as its z-stack completes):
│                             deskew -> phase -> virtual stain (nuclei/membrane)
│                             -> project -> segment -> features -> tree -> good/bad
│
├─ BARRIER at t0 -> t1  event_iterator drains all streamed decisions
│
└─ t >= 1 TIMELAPSE     full channels + z, ONLY on FOVs decided "good"
                        (bad FOVs skipped via SkipEvent)
```

- **Pre-scan** = timepoint 0. Only `input_channel` frames are acquired (full z,
  needed for virtual-staining quality), across all candidates.
- **Decision** streams in during the pre-scan via `frameReady`, computed in a
  worker subprocess (torch/GPU isolation — the same reason DynaTrack uses one).
- **Timelapse** = timepoints `>= 1`, full channels/z, only on good FOVs.

Because the first timepoint is the pre-scan, set `time_plan.loops =
desired_timelapse_points + 1`.

## Why a single run (and its one constraint)

`core.mda` is a single, non-reentrant runner: calling `core.mda.run()` inside
`setup_sequence` corrupts the outer run's state (`_sink`/`_state` are instance
attributes). So the pre-scan cannot be a nested run — it is the **first timepoint
of the one run**, with the decision streamed via `frameReady`.

**Constraint:** one run => one output store => one array shape per position, so
the **pre-scan z-range must equal the timelapse z-range**. A single-z-slice
pre-scan (e.g. fluorescence-based selection) would need two stores / two runs and
is out of scope here.

## Engine integration (`shrimpy/mantis/mantis_engine.py`)

- `acquire()` — `_inject_fov_candidates()` substitutes
  `fov_selection.search_mda.stage_positions` into the run sequence, so the output
  store is shaped for all candidates with correct channel/position indices. The
  main `stage_positions` is empty.
- `setup_sequence()` — builds `FovSelection.from_metadata(sequence,
  pixel_size_um=core.getPixelSizeUm())`, connects `frameReady -> on_frame_ready`,
  and starts the worker after the ROI is applied.
- `event_iterator()` — yields **every** event (never drops) and drains the
  streamed decision once at the t0->t1 boundary (safe: `frameReady` fires
  synchronously during execution, so all t0 frames are processed before the first
  t>=1 event is pulled).
- `setup_event()` — raises `SkipEvent(num_frames=…)` for events not to acquire
  (non-`input_channel` frames at t=0; not-"good" FOVs at t>=1). **Skipping here,
  not dropping in `event_iterator`, is required**: the output sink advances a
  per-event cursor and must see an append or an explicit skip for every declared
  event, or frames land in the wrong slots.
- `teardown_sequence()` — disconnects `frameReady` and shuts down the worker.

## FOV-selection package (`shrimpy/fov_selection/`)

- `manager.py` — `FovSelection`, the streaming coordinator (mirrors `DynaTrack`):
  `from_metadata` -> `start` (spawns worker) -> `on_frame_ready` (buffers t0
  `input_channel` stacks, submits per FOV as each completes) -> `drain` (barrier)
  -> `is_good` / `good_position_names` (gating) -> `shutdown`. Bounded to one
  in-flight decision (backpressure — never holds a whole plate in RAM).
- `worker.py` — `FovSelectionWorker` subprocess: builds preprocessor + Cellpose +
  tree once, decides one FOV per message.
- `pipeline.py` — per-FOV `decide_fov` (project -> segment -> features -> predict),
  reusing the offline feature code (`object_feature_rows` + `group_features`) so
  online features match training exactly.

## Reconstruction = UNIFIED, shared with DynaTrack

The deskew -> phase -> virtual-staining chain lives in **`shrimpy/preprocessing.py`**
(`build_preprocessor` / `_LabelfreePreprocessor`), shared by both DynaTrack and
FOV selection. It is decoupled from either package's config object —
`build_preprocessor` takes the reconstruction settings as explicit arguments, and
each caller extracts them from its own config. (This replaced an earlier plan to
copy the file into `fov_selection/`.) Tests: `shrimpy/tests/test_preprocessing.py`.

## Configuration (`config/mda/mantis/fov_online_demo.yaml`)

DynaTrack-style: `deskew` / `phase` / `virtual_staining` blocks directly under
`fov_selection`; scale (XY pixel size, Z step) is fetched from the Core and the
z_plan step and injected (single source of truth, never in the config).

- `input_channel` — acquired channel fed to reconstruction (pre-scan channel).
- `fov_selection_channels` — channels the decision is computed on (raw or
  preprocessed; here VS `nuclei`/`membrane`).
- `preprocessing` — ordered step list, e.g.
  `['deskew','phase','vs','sum_projection','segmentation']`. Reconstruction steps
  feed `build_preprocessor`; projection + segmentation are consumed by the
  pipeline.
- `segmentation` — switchable model + params (only `cellpose` today; defaults
  fall back to the batch script's training values).
- `model` — trained FOV-goodness `.joblib` `path` + `threshold`.
- `search_mda.stage_positions` — a useq **`WellPlatePlan`** (select wells + a
  per-well FOV grid) so it scales and produces a proper HCS OME-Zarr
  (`plate_row`/`plate_col` + slash-free field names). The main `stage_positions`
  is empty.

## Offline ReplayCamera mapping

A `WellPlatePlan` names positions `"B3_0000"` (unique — good for gating) with
`plate_row`/`plate_col`, while the replay source is keyed `"B/3/000000"`. The
ReplayCamera builds an **index -> source-key map** at `sequenceStarted` by
reconstructing each position's well from `plate_row`/`plate_col` and taking the
k-th source FOV in that well, so replayed data lines up with the output HCS
layout. This shim exists only offline; on the microscope the `WellPlatePlan`
drives real stage coordinates and there is one naming system.

## Milestones

- **M1** — two-phase skeleton + dummy selector (learning the mechanics). Removed.
- **M2** — real decision (reconstruction + pipeline + trained model), as a
  two-run batch flow behind a controller. Superseded by M3.
- **M3 — DONE.** Single streaming run folded into the engine; `frameReady` +
  worker subprocess; `event_iterator` barrier + `setup_event` gating; unified
  `shrimpy/preprocessing.py`; `WellPlatePlan` config + HCS output; standalone
  scripts moved to `shrimpy/scripts/`. Validated end-to-end on replay (B3 bad,
  B4/B5 good -> gated timelapse; uniform HCS store).

## Tests

- `test_preprocessing.py` — shared preprocessing wiring.
- `test_fov_selection_manager.py` — streaming buffering / verdicts / drain barrier
  (injected in-process decider; no GPU).
- `test_fov_selection_engine.py` — `event_iterator` drain-once + yields-all, and
  `_fov_skip_frames` skip decisions.
