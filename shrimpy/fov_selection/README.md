# FOV selection

Online, streaming selection of "good" fields of view during an acquisition, plus a
feature viewer for tuning the selection model and offline scripts for training one.

The package is microscope agnostic: an acquisition engine (`shrimpy/engines/`) builds a
`FovSelection` coordinator from its `metadata.fov_selection` config and interacts with that
object only. Everything else (reconstruction, segmentation, feature extraction, the worker
subprocess) is an implementation detail behind it.

## Two-run adaptive acquisition

FOV selection turns one acquisition into two runs:

```
1. PRE-SCAN   image every candidate FOV once on fov_selection_channel,
              score each one, decide which are worth a timelapse
              │
              ▼
2. TIMELAPSE  run the real (multi-timepoint) acquisition on the good FOVs only
```

The engine builds both sequences from a single user config (`sequences.py`), runs the
pre-scan, drains the per-FOV verdicts, then runs the timelapse over the FOVs that passed.

In **calibration mode** the engine stops after the pre-scan, extracts every available
feature (not just the model's), and opens the feature viewer on the result so you can
explore features and design a ranking model before committing to a full run.

## Per-FOV decision pipeline

Each pre-scan FOV's z-stack runs through the same in-memory pipeline (`pipeline.decide_fov`),
in a worker subprocess for torch/GPU isolation:

```
raw z-stack (fov_selection_channel)
  # preprocessor (preprocessing.py): reconstruction only
  → reconstruct        deskew / phase / virtual staining  → {channel: (Z, Y, X)}
  # FOV-selection decision (this package), driven by decide_fov:
  → project            sum | max | middle | logstd | best_focus_z   (pipeline.project_zyx)
  → reduce to ONE 2D image by `target`                         (pipeline._resolve_seg_input)
  → segment ONE mask   cellpose | instanseg | otsu             (segmentation.py)
  → features           per-object then per-FOV aggregation     (feature_extraction.py)
  → verdict            model.predict(features)                 (fov_model.py)
```

Because selection always produces exactly **one** mask (the `target`), feature columns are
plain single-mask keys (for example `coverage_frac`), with no channel prefix. The model
reads features by name only and never sees which channel produced them, so any model type
pairs with any preprocessing.

### The preprocessor is reconstruction only

**Projection and segmentation are deliberately NOT part of the preprocessor.** The
preprocessor (`shrimpy/preprocessing.py`, `build_preprocessor`) does only reconstruction
(flatfield / deskew / phase / VS) and returns 3D channel volumes `{channel: (Z, Y, X)}`;
projection, segmentation, feature extraction, and scoring live in this package and are driven
by `decide_fov`. The seam sits there for three reasons:

- **Reuse.** `preprocessing.py` is shared with DynaTrack, which needs the same reconstruction
  but has a completely different downstream (position tracking, not projection + segmentation).
  Only the reconstruction is common, so only the reconstruction belongs in the shared module.
- **Natural 3D-to-2D boundary.** Reconstruction is the optical step that produces 3D volumes;
  projection (which collapses Z, and for `best_focus_z` needs its own optics) and segmentation
  are the analysis step that turns a volume into a mask. The clean handoff is "reconstructed
  3D volumes out, 2D analysis in".
- **Keeps the shared module free of FOV-selection concerns.** Reducing the reconstruction
  outputs to a single mask by `target`, the one-mask feature schema, and the choice of
  segmenter are all selection-specific, so they stay here rather than in `preprocessing.py`.

In the config's `preprocessing:` list this split is explicit: the reconstruction steps
(`flatfield` / `deskew` / `phase` / `vs`) are consumed by `build_preprocessor`, while the
projection step and `segmentation` are consumed by `decide_fov`. With no reconstruction step
in the list, `build_preprocessor` returns `None` and `decide_fov` segments the raw stack
directly.

## Configuration

Everything lives under `metadata.fov_selection` in the acquisition YAML. See
`config/mda/fov_selection_demo.yaml` for a fully commented example. Key fields:

| Field | Meaning |
|-------|---------|
| `enabled` | turn FOV selection on for this run |
| `calibration_mode` | pre-scan only, extract all features, open the viewer |
| `prescan_mda` | a complete nested MDASequence: the candidate FOVs, pre-scan z_plan, and channels (single timepoint) |
| `fov_selection_channel` | the acquired channel imaged in the pre-scan and fed to reconstruction |
| `target` | `cells` or `nuclei`: the object to segment and score (drives the InstanSeg head, the VS channels predicted, and how outputs reduce to one segmentation input) |
| `preprocessing` | ordered steps, e.g. `['deskew', 'phase', 'vs', 'sum_projection', 'segmentation']` |
| `deskew` / `phase` / `virtual_staining` | reconstruction sub-configs (scale values are injected, never duplicated) |
| `best_focus_z` | optics for the `best_focus_z` projection (detection NA, illumination wavelength) |
| `segmentation` | backend `cellpose` / `instanseg` / `otsu`, plus `path`, `diameters`, thresholds |
| `model` | the selection model (see below) |
| `save_decision` | write per-FOV projection/mask PNGs (`prescan_fov/`, `prescan_mask/`) and `fov_summary.csv` under `<name>_fov_debug/`; after the drain, the selected FOVs' projections are also gathered into `selected_fov/`. Same folder structure in normal and calibration mode |
| `save_pre_scan_omezarr` | write the full per-step reconstruction to `<name>_prescan.ome.zarr` |
| `require_gpu` | fail fast if reconstruction cannot run on a GPU (default true) |

The block itself is validated by `FovSelection.from_metadata` and the coordinator when they
are built (a bad model type, a missing/unknown `fov_selection_channel`, a `target` outside
`{cells, nuclei}`, or a model asking for an unproducible feature all raise before any
hardware is touched). `shrimpy/config.py` keeps `fov_selection` as a single opaque section
under its strict `extra="forbid"` metadata schema, so there is no second source of truth.

## Selection models

Set `model.type` to one of:

| `type` | Class | How it selects |
|--------|-------|----------------|
| `ranking_by_defined_range` | `DesirabilityModel` | per-feature desirability curves (`gaussian` / `lognormal` / `sigmoid`) combined into one score (`aggregation`: `gaussian` / `sum` / `product`); keeps the `top_fov` highest-scoring FOVs **per position** (per well / grid center). `top_fov` is **required** (pure ranking, no per-FOV pass/fail); `threshold` does not apply |
| `classification_by_thresholding` | `ThresholdingModel` | hard `[lo, hi]` box: a FOV passes only if every feature is in range. Per-FOV pass/fail, so no `top_fov` is needed |
| `classification_tree` | `TrainedTreeModel` | a trained `.joblib` (median imputer + decision tree) loaded from `model.path` |

Curve parameters are interpretable (`center`/`fwhm`, `center`/`fold`, `midpoint`/`width`),
the same knobs the feature viewer's Rank tab edits.

## Feature schema

FOV selection segments one mask, so every feature is a plain key. Features are split into
three categories by **what input each one needs** (`feature_extraction.py`). That is a real
distinction, not cosmetic: it is what decides how much work a per-FOV decision costs.

1. **Object-level** (`FeatureExtractor.object_feature_rows`): one row per segmented object
   (area, shape, intensity, nearest-neighbor spacing). These are not model features on their
   own; they are the intermediate per-object table that the FOV-level aggregates are computed
   from. Building it runs `regionprops` + a KD-tree, which is the expensive part of a decision.
2. **FOV-level aggregates** (`FeatureExtractor.group_features`, from that per-object table):
   `coverage_frac`, `object_counts`, `average_object_intensity`, `nn_um_mean`, `nn_cv`,
   `com_offset_norm`, `mean_distance_to_center_fov`, `empty_grid_frac`, `occupancy_entropy`,
   `angular_uniformity`. Each is a summary of the object table, so this level needs level 1
   first.
3. **Mask-derived** (`FeatureExtractor.mask_gap_features`, from the label-mask pixels):
   `max_empty_radius`, `mask_occupancy_entropy`, `edge_frac`, `central_cov_ratio`. These
   describe the spatial layout of the foreground and cannot be recovered from object centroids,
   so they read the mask directly rather than the object table.

**Why the split matters.** A few keys are computable from the mask alone, with no object table
at all: `coverage_frac`, `object_counts`, and `mask_occupancy_entropy` (the
`MASK_ONLY_FEATURE_KEYS` set in `pipeline.py`). When the model only asks for those,
`extract_features` takes a mask-only fast path and skips the `regionprops` / KD-tree extraction
entirely; otherwise it builds the per-object table once, and both the aggregates and the
mask-derived features draw from it. Categorizing by input dependency is what lets the pipeline
do the minimum work for the features a given model actually requests, and it keeps the
mask-derived features (which centroids cannot express) cleanly separate from the per-object
aggregates.

## Feature viewer

```bash
python -m shrimpy.fov_selection.feature_viewer [CSV ...]
```

A Qt GUI to explore FOV-level features, label FOVs, and tune the ranking model, with each
FOV shown as an image thumbnail. Tabs:

- **Analysis**: interactive 2D/3D scatter (PCA / t-SNE / UMAP) with per-feature threshold
  filters; selected FOVs shown as a thumbnail grid, grouped by well when the CSV carries
  `well_row` / `well_col` (the pre-scan writes these for plate-based candidates).
- **Label**: FOVs grouped into per-goodness-class panels; drag a thumbnail to relabel, save
  writes the `goodness` column back to the CSV.
- **Rank**: per-feature value histograms with the desirability curve overlaid, a knob table,
  and a Re-rank button; FOVs listed best-first by score. Save/load ranking profiles as YAML
  (paste one into `model.features`). A **Write proba/rank to CSV** button writes the current
  ranking's score (as `proba`) and best-first `rank` back to each FOV's source CSV, so a
  calibration `fov_summary.csv` ends up carrying the selection tuned here.
- **Score map**: 2D desirability heatmaps over pairs of features.

A calibration pre-scan launches this automatically on `--start-tab rank`, seeding the Rank
tab from the config's `model` block (passed inline, so no profile file is written) and
falling back to data-seeded defaults when the config defines no model. Use the Rank tab's
Save button to write a ranking profile once tuned.

## Package layout

```
fov_selection/
├── manager.py            FovSelection coordinator (engine-facing): buffering,
│                         verdict store, drain, per-position top-K selection
├── sequences.py          build the pre-scan and timelapse MDASequences
├── pipeline.py           per-FOV decision: project → segment → features → verdict
├── worker.py             subprocess isolation (WorkerConfig + FovSelectionWorker)
├── fov_model.py          pluggable models + interpretable curve conversions
├── segmentation.py       Cellpose / InstanSeg / Otsu backends
├── feature_extraction.py FeatureExtractor (object-level and FOV-level features)
├── prescan_artifacts.py     per-FOV pre-scan PNG / CSV / OME-Zarr writers + finalize
├── acquisition_artifacts.py once-per-run records (recovery config, viewer launch)
├── plate_naming.py       plate labels and path-name sanitizers
└── feature_viewer/       Qt GUI
    ├── app.py            main window, Analysis tab, and the `main()` entry point
    ├── _common.py        shared constants, theme, helper widgets
    ├── label_tab.py      LabelTabMixin
    ├── rank_tab.py       RankTabMixin
    ├── score_map_tab.py  ScoreMapTabMixin
    └── data.py           Qt-free data layer (load CSVs, wire PNGs, run reduction)
```

The GUI is split into one mixin per tab; `FeatureViewer` inherits them, so every method
still shares one window instance.

## Offline model training

Two scripts in `shrimpy/scripts/` support building a trained `classification_tree` model
from annotated data (they are separate from the online engine above):

- `make_projection_store.py`: project channels of an OME-Zarr into a compact 2D store.
- `predict_fov_goodness.py`: train a decision tree on labeled FOV feature matrices and
  write `predicted_good_proba` / `predicted_goodness` columns onto a new matrix.

## Dependencies

Installed via the `fov` dependency group (`uv sync --group fov`): cellpose + dinov3 (cpdino
segmentation), instanseg (alternative segmenter), the analysis stack (scikit-learn,
scikit-image, scipy, pandas, matplotlib, joblib), and Qt (via qtpy) for the viewer. iohub
provides OME-Zarr I/O; waveorder provides the `best_focus_z` focus metric.
