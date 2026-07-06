# FOV selection pipeline

From a raw OME-Zarr z-stack to a viewer-ready, FOV-level feature table, then a
good-vs-bad FOV prediction. Each stage is a single script that reads the previous
stage's output and writes the next. Config lives in a `CONFIG` block (paths, channels,
dataset list) at the top of each script — edit there, then run.

```
z-stack .zarr
   │  1. make_projection_store.py        (CPU, heavy I/O)
   ▼
<dataset>_proj.zarr                       projected channels, Z=1
   │  2. segment_cellpose.py             (GPU)
   ▼
<dataset>_proj_cpdino.zarr               projections + <channel>_cpdino label channels
   │  3. extract_fov_features.py         (CPU) — per-object table
   ▼
fov_features/<tag>_object_features.csv    one row per segmented object
   │  4. build_fov_feature_matrix.py     (CPU, fast) — per-FOV aggregation
   ▼
fov_features/<tag>_fov_feature_matrix.csv one row per FOV (viewer-ready)
   │  5. feature_viewer                  (GUI)   explore / label / cluster FOVs
   │  6. predict_fov_goodness.py         train tree on labeled FOVs, predict new ones
   ▼
predicted_goodness column + final_model/ artifacts
```

## Stages

| # | Script | In → Out |
|---|--------|----------|
| 1 | `make_projection_store.py` | z-stack `.zarr` → `<dataset>_proj.zarr` |
| 2 | `segment_cellpose.py` | `_proj.zarr` → `_proj_cpdino.zarr` |
| 3 | `extract_fov_features.py` | `_proj_cpdino.zarr` → `<tag>_object_features.csv` |
| 4 | `build_fov_feature_matrix.py` | `<tag>_object_features.csv` → `<tag>_fov_feature_matrix.csv` |
| 5 | `feature_viewer/` | `*_fov_feature_matrix.csv` (+ composites) → explore / label |
| 6 | `predict_fov_goodness.py` | labeled matrices → trained tree + predictions on a new matrix |

Steps 2–4 accept a single-dataset selector so datasets can be fanned out one job each;
run with no argument to process every configured dataset.

## FOV feature matrix (the viewer contract)

`build_fov_feature_matrix.py` emits one row per FOV — identified by
`(dataset, well_row, well_col, fov, timepoint)` — with every channel/organelle/
segmentation-source/projection variant laid out side by side as prefixed columns:

```
<organelle>_<segmentation_source>_<projection_type>__<feature>
e.g. nuclei_vs_sum__coverage_frac, membrane_vs_sum__objects_per_10um2
```

10 features per variant (density: `object_count`, `objects_per_10um2`, `coverage_frac`;
edge: `edge_frac`; spatial: `com_offset_norm`, `nn_um_mean`, `nn_cv`,
`densest_grid_frac`; shape: `eccentricity_mean`, `solidity_mean`). Every row carries a
numeric `goodness` label (good=1, neutral=0, bad=-1, NaN=unlabeled), which the feature
viewer reads and writes back in place.

## Feature viewer

```bash
python -m shrimpy.fov_selection.feature_viewer
```

Loads the FOV feature matrices, resolves one composite image per FOV from
`fov_features/fov_composites/<dataset>/`, lets you filter, label goodness, and run 3D
dimensionality reduction (PCA / t-SNE / UMAP). Qt-free core in `feature_viewer/data.py`.

## FOV goodness prediction

```bash
python -m shrimpy.fov_selection.predict_fov_goodness
```

Trains one depth-3 decision tree on the labeled datasets (features settled by analysis:
`nuclei_vs_sum__coverage_frac` + `membrane_vs_sum__objects_per_10um2`; label
good+neutral vs bad), saves the model + diagnostics under
`fov_features/split_analysis/final_model/`, and writes `predicted_good_proba` /
`predicted_goodness` columns onto an unlabeled target matrix.

## Dependencies

Installed via the `fov` dependency group (`uv sync --group fov`): cellpose + dinov3
(cpdino segmentation), stardist / instanseg (alternative segmenters), napari +
napari-iohub (viewing), and the analysis stack (scikit-learn, scikit-image, scipy,
pandas, matplotlib, joblib). iohub provides OME-Zarr I/O.
