# FOV-selection integration plan

Integrate the offline FOV-selection pipeline (`shrimpy/fov_selection/`) into the
acquisition engine as an **online, adaptive acquisition**: pre-scan every
position once, decide which are "good", then run the timelapse only on those.

Developed/tested offline with the ReplayCamera replaying
`.../0-flatfield/2026_06_24_A549_H2BC21_FOV_selection_1.zarr` (raw BF+GFP), which
matches what the microscope produces online.

## Experiment flow

```
PRE-SCAN                     DECISION (Option A: batch, one FOV at a time)      TIMELAPSE
MDASequence:                 per FOV, read pre-scan zarr from disk:              MDASequence:
  all positions                reconstruct: deskew -> phase -> virtual stain       good positions only
  t = 1                        -> nuclei / membrane                                 full z-stack
  full z                       pipeline: project -> segment -> features -> predict  all channels
  BF-Oblique only              -> good / bad                                        time_plan (loops > 1)
core.mda.run()  ─────────▶   good position list  ────────────────────────────▶   core.mda.run()
```

- **Pre-scan** (phase 1): all positions, `t=1`, **full z**, **BF-Oblique only**.
  Full z is required for virtual-staining quality.
- **Timelapse** (phase 2): good positions only, full z, all channels, real
  `time_plan`.

## Architecture

- **Two `core.mda.run()` calls** with a decision step between — NOT one static
  `MDASequence`, and NOT DynaTrack's per-event position updates. FOV selection is
  a *subset choice between two runs*.
- The frozen `MDASequence` problem is avoided by **building a fresh phase-2
  sequence** from the selected positions (`sequence.replace(stage_positions=...)`),
  not mutating the pre-scan one.
- **Memory:** process one FOV at a time, read lazily from the pre-scan zarr on
  disk (~one FOV, a few GB, in RAM). The plate never sits in RAM.

## Reconstruction = COPY DynaTrack's preprocessing (decoupled), not import it

The deskew -> phase -> virtual-staining chain already exists in
`shrimpy/dynatrack/preprocessing.py` (`build_preprocessor` /
`_LabelfreePreprocessor`, using biahub `DeskewSettings`, waveorder phase, and
cytoland VS).

**Decision:** copy it into `shrimpy/fov_selection/` (e.g.
`fov_selection/preprocessing.py`) rather than importing from `dynatrack/`, so
FOV-selection reconstruction can be tuned independently without affecting
DynaTrack's tracking behavior. The two are separate.

This is not a pure file copy — `preprocessing.py` is coupled to
`DynaTrackConfig`, reading: `preprocessing` (`['deskew','phase','vs']`),
`tracking_channel`, `deskew`, `phase`, `virtual_staining`. The copy must also get
its own small reconstruction config (e.g. `FovSelectionReconConfig`) exposing
those fields (with `tracking_channel` -> `input_channel = 'BF - Oblique'` and
`virtual_staining.target_channels = ['nuclei','membrane']`). Done in M2.

## pymmcore / shrimpy structure (reference)

```
MDASequence (frozen plan) --iterate--> MDAEvent(s) --core.mda.run()--> MDAEngine --emit--> frameReady(img, event)
```
- Engine hooks (in `shrimpy/mantis/mantis_engine.py`): `setup_sequence` (once),
  `event_iterator` (transform event stream), `setup_event` (per-event HW),
  `teardown_sequence` (once). Current acquisition entry point: `acquire()` (~L460)
  calls `core.mda.run(sequence, ...)` once.
- `core.mda.run()` accepts an `MDASequence` OR any `Iterable[MDAEvent]`, and can
  be called multiple times — this is what enables the two-phase adaptive flow.

## Milestones

- **M1 — two-phase skeleton + dummy selector.** Pre-scan (BF/full-z/all-pos) +
  trivial good/bad rule + timelapse over "good" positions. Purpose: learn the
  pymmcore/shrimpy acquisition mechanics. No reconstruction/segmentation/model.
- **M2 — real decision.** Copy DynaTrack's `preprocessing.py` into
  `fov_selection/` + its own recon config (decoupled) for reconstruction, then
  the fov_selection pipeline (project -> segment -> features -> predict) +
  trained model. Wrap as a `FovSelection` manager class mirroring `DynaTrack`.
- **M3 — streaming + GPU worker.** Move per-FOV processing to `frameReady`, in a
  subprocess (torch/OpenMP isolation), for wall-clock overlap. Later.

## Conventions / decisions

- Term: **pre-scan** (phase 1), **timelapse** (phase 2).
- Processing timing: **batch after pre-scan** (Option A) first; streaming is M3.
- Output layout: pre-scan -> `<name>_prescan.ome.zarr`, timelapse -> `<name>.ome.zarr`.

## Open items

- Confirm a runnable reconstruction config (BF -> nuclei/membrane) — start from
  DynaTrack's, but copied/decoupled into `fov_selection/` (see reconstruction
  section).
- Known ReplayCamera/write-path bugs to fix before trusting written output as a
  real-data stand-in: see `docs/replay_camera_todo.md`.
