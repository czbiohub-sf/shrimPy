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
  → reconstruct        deskew / phase / virtual staining      (preprocessing.py)
  → project            sum | max | middle | logstd | best_focus_z
  → reduce to ONE 2D image by `target`                         (pipeline._resolve_seg_input)
  → segment ONE mask   cellpose | instanseg | otsu             (segmentation.py)
  → features           per-object then per-FOV aggregation     (feature_extraction.py)
  → verdict            model.predict(features)                 (fov_model.py)
```

Because selection always produces exactly **one** mask (the `target`), feature columns are
plain single-mask keys (for example `coverage_frac`), with no channel prefix. The model
reads features by name only and never sees which channel produced them, so any model type
pairs with any preprocessing.

## Configuration

Everything lives under `metadata.fov_selection` in the acquisition YAML. See
`config/mda/mantis/fov_selection_demo.yaml` for a fully commented example. Key fields:

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
| `ranking_by_defined_range` | `DesirabilityModel` | per-feature desirability curves (`gaussian` / `lognormal` / `sigmoid`) combined into one score (`aggregation`: `gaussian` / `sum` / `product`); keeps the `top_fov` highest-scoring FOVs **per position** (per well / grid center) |
| `classification_by_thresholding` | `ThresholdingModel` | hard `[lo, hi]` box: a FOV passes only if every feature is in range |
| `classification_tree` | `TrainedTreeModel` | a trained `.joblib` (median imputer + decision tree) loaded from `model.path` |

Curve parameters are interpretable (`center`/`fwhm`, `center`/`fold`, `midpoint`/`width`),
the same knobs the feature viewer's Rank tab edits.

## Feature schema

FOV selection segments one mask, so every feature is a plain key. Two groups
(`feature_extraction.py`):

- **FOV-level** (`FeatureExtractor.group_features`, from the per-object table):
  `coverage_frac`, `nn_um_mean`, `nn_cv`, `com_offset_norm`,
  `mean_distance_to_center_fov`, `empty_grid_frac`, `occupancy_entropy`,
  `angular_uniformity`.
- **Mask-derived** (`FeatureExtractor.mask_gap_features`, need the label mask itself):
  `max_radius_corner_to_edge`, `mask_occupancy_entropy`, `edge_frac`,
  `central_cov_ratio`.

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
