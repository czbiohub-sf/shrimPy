# FOV-selection integration

The offline FOV-selection pipeline (`shrimpy/fov_selection/`) is integrated into
the Mantis acquisition engine as an **online, adaptive acquisition**: the
microscope pre-scans every candidate FOV once, decides which are "good" with the
trained model, and images the timelapse only on those.

`acquire()` orchestrates **two sequential `core.mda.run()` calls** around one
`FovSelection` object (read from `metadata.mantis.fov_selection`). No separate
controller, no CLI routing; `shrimpy acquire mantis` runs it like any other
acquisition. Validated offline with the ReplayCamera replaying a real A549 store.

## Experiment flow — TWO sequential runs

```
RUN 1  PRE-SCAN     core.mda.run(prescan_seq, output=None)
│                   all candidates · input_channel only · loops=1 · full z
│      └─ frameReady ─► worker subprocess (per FOV as its z-stack completes):
│                          deskew -> phase -> vs -> project -> segment
│                          -> features -> tree -> good/bad
├─ run 1 returns; drain; good = good_position_names()
│
RUN 2  TIMELAPSE    core.mda.run(timelapse_seq, output=<name>.ome.zarr)
                    GOOD FOVs only · full channels + z · loops=N
```

- Pre-scan run: `input_channel` only (full z, for VS quality), one timepoint,
  all candidates. `output=None` writes nothing (unless `save_prescan`); the
  decision still streams via `frameReady` (emitted independent of the sink).
- Timelapse run: only good FOVs are in `stage_positions`, so there is no
  per-event gating. `time_plan.loops` = timelapse points (**no `+1`**).

## Why two sequential runs

`core.mda` forbids *nested* runs, but `run()` rebuilds its sink + per-run state
each call, so calling it **twice in sequence is safe**. This removes the old
single-run constraints: the pre-scan is not a timepoint of the main store, no
shared-shape requirement, no `event_iterator` barrier, no `SkipEvent` gating —
at the cost of a little orchestration in `acquire()` (two runs + a position
filter).

## Engine integration (`shrimpy/mantis/mantis_engine.py`)

- `acquire()` — orchestrator: build `prescan_seq` (`_build_prescan_sequence`),
  run it, read the good names (captured in teardown), build `timelapse_seq`
  (`_build_timelapse_sequence` + `_filter_good_positions`, `fov_selection`
  disabled), run it into the main store. No good FOV → warn and skip run 2.
- `setup_sequence()` — builds `FovSelection.from_metadata(...)` + starts the
  worker after ROI; non-`None` only for the pre-scan run (timelapse disables it).
- `teardown_sequence()` — captures `good_position_names()` onto the engine, then
  disconnects `frameReady` and shuts down the worker (verdicts survive shutdown).
- `event_iterator()` / `setup_event()` — no FOV logic (DynaTrack logic unchanged).
- Candidates live in top-level `stage_positions` (a `WellPlatePlan`); filtering
  to the good subset preserves `plate_row`/`plate_col` → HCS store of good FOVs.

## FOV-selection package (`shrimpy/fov_selection/`)

- `manager.py` — `FovSelection`, the streaming coordinator (mirrors `DynaTrack`):
  `from_metadata` -> `start` (spawns worker) -> `on_frame_ready` (buffers the
  pre-scan stacks, submits per FOV as each completes) -> `drain` (after run 1)
  -> `good_position_names` -> `shutdown`. Bounded to one in-flight decision
  (backpressure — never holds a whole plate in RAM).
- `worker.py` — `FovSelectionWorker` subprocess: builds preprocessor + Cellpose +
  tree once, decides one FOV per message; writes per-FOV debug artifacts when
  `save_decision` is set.
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
- `save_prescan` (bool) — write the pre-scan run to `<name>_prescan.ome.zarr`
  instead of discarding it (`output=None`).
- `save_decision` (bool) — write per-FOV debug artifacts (preprocessed FOV,
  features CSV, goodness) to a debug dir next to the output.
- Top-level `stage_positions` — a useq **`WellPlatePlan`** (select wells + a
  per-well FOV grid), the candidate FOVs; produces a proper HCS OME-Zarr
  (`plate_row`/`plate_col` + slash-free field names).

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
- **M3.** Single streaming run folded into the engine (`event_iterator` barrier
  + `setup_event` gating); unified `shrimpy/preprocessing.py`; `WellPlatePlan`
  config + HCS output. Superseded by M4.
- **M4 — IN PROGRESS.** Two sequential runs (pre-scan `output=None` + timelapse
  on good FOVs only); candidates in top-level `stage_positions` (drop
  `_inject_fov_candidates`); `save_prescan` + `save_decision` flags; engine hooks
  simplified (no barrier/`SkipEvent`).

## Tests

- `test_preprocessing.py` — shared preprocessing wiring.
- `test_fov_selection_manager.py` — streaming buffering / verdicts / drain.
- `test_fov_selection_acquire.py` — sequence builders + good-FOV position filter
  (HCS metadata preserved); no GPU.

## TODO (M5)

- **Test on a full dataset.** Run over the whole plate (all 147 positions) rather
  than the 3-position demo — validate memory (one-in-flight backpressure),
  wall-clock, and the HCS output at scale.

## Known issue (not blocking)

`make format` reports ~25 pre-existing `ruff` naming complaints (e.g. `X`/`Xi`
in `pipeline.py:predict_good`, `import pipeline as P` in `worker.py`) in code
that predates the M4 work — the same 25 appear with our changes stashed, so they
are not introduced here. The auto-formatting step still runs; only the style
*check* fails. Left as-is for now (small mechanical renames when we get to it).
