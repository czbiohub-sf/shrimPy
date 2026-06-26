# napari-deskew-preview

A napari widget for **quick, on-demand deskew preview** of oblique-plane (light-sheet /
OPM) data — for browsing saved acquisitions offline, without materializing the full
deskewed volume.

> Internal tool of the [czbiohub-sf](https://github.com/czbiohub-sf) organization.

## Why

Deskew is a pure affine in which only the scan axis needs fractional interpolation, so a
single deskewed axial plane maps to **one raw tilt row resampled in 1-D**. This widget
computes only the plane on screen (~few MB, sub-ms) instead of the whole ~1 GB deskewed
volume — ideal when you glance at a few planes of a large dataset. For producing whole
corrected volumes (saving / analysis), use a GPU deskew such as biahub's `fast_deskew_zyx`
or [napari-lattice](https://github.com/BioimageAnalysisCoreWEHI/napari_lattice).

## Install (internal)

```bash
pip install "napari-deskew-preview @ git+https://github.com/czbiohub-sf/napari-deskew-preview"
```

## Use

1. Open napari and load your raw oblique data (e.g. an OME-Zarr position via
   `napari-ome-zarr`). The widget treats the layer's **last three axes as
   `(Z_scan, Y_tilt, X_cover)`** and any leading axes (T, C, position) as a batch.
2. `Plugins → Deskew Preview`.
3. Set **Angle**, **Pixel size**, and **Scan step** (defaults: 30°, 0.1133 µm) and click
   **Deskew selected layer**. The layer's data is replaced **in place** with a lazy
   deskewed view; editing the fields rebuilds it, and **Restore raw** puts the original
   data back.

   Replacing in place (rather than adding a second layer) is deliberate: a raw and a
   deskewed layer have different axis sizes (e.g. scan `1068` vs deskewed depth `256`), so
   keeping both would make napari union their extents into oversized sliders.

## Library API (source-agnostic)

The deskew core is numpy-only and works on any array-like via a `gather` callable:

```python
from napari_deskew_preview import deskewed_layer, array_gather

# raw: array-like indexed as (*batch, Z_scan, Y_tilt, X_cover)  (zarr / dask / numpy)
data, projector = deskewed_layer(
    array_gather(raw),
    raw_zyx_shape=raw.shape[-3:],
    scan_step_um=0.313,
    batch_sizes=raw.shape[:-3],
    ls_angle_deg=30.0,
    pixel_size_um=0.1133,
)
viewer.add_image(data)  # lazy: planes computed on demand
```

## Notes

- The deskew geometry matches biahub's `fast_deskew_zyx` to float32 precision.
- The same source-agnostic core is used by shrimpy's live acquisition viewer; this package
  is the standalone/offline home and the basis for future unification.
