"""
Extract per-object features from the segmented _cpdino stores into ONE CSV per
dataset (every FOV concatenated) + an optional per-FOV summary table for ranking.

For every labeled object (each nucleus and each cell/membrane) in every FOV and
timepoint, writes one row. All rows for a dataset go into a single
<tag>_object_features.csv -- a FOV with 50 nuclei + 40 cells at a timepoint
contributes 90 rows, and every FOV/timepoint in the store is appended.

Per-object columns:
    dataset, well_row, well_col, fov, timepoint   identifiers / metadata
    channel                "nuclei" or "membrane"
    segmentation_source    "vs" (virtual staining) or "fluor"
    label_id               instance id within that channel/timepoint
    centroid_x_norm/_y_norm  mask-center / image size (0..1)
    area_px, area_um2        mask size (pixels and physical)
    equivalent_diameter_um   diameter of a circle with the same area
    eccentricity, solidity, extent   shape descriptors (QC / debris)
    intensity_mean/_max      prediction-channel signal under the mask
    bbox_min_row/_col,       raw mask bounding box in px (skimage convention: min
    bbox_max_row/_col          inclusive, max exclusive). Edge-touching is derived
                               downstream at any margin from these + image size, so
                               the margin can be re-tuned without re-extracting.
    dist_to_edge_norm        centroid distance to nearest border / image size
    nearest_neighbor_dist_um  to nearest object of the same channel (crowding)
    image_width_px, image_height_px   total image size

This is a single-purpose step: it writes ONLY the per-object table. FOV-level
aggregation (one row per FOV, viewer-ready) is the next stage,
build_fov_feature_matrix.py, which consumes this CSV.

    python extract_fov_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from iohub import open_ome_zarr
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

# =============================================================================
# CONFIG -- edit these
# =============================================================================
OUT_ROOT = "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output"

DATASETS = [
    {
        "tag": "2026_03_25_A549_organelles_DENV_ZIKV",
        "store": f"{OUT_ROOT}/2026_03_25_A549_strong_organelles_DENV_ZIKV_time_course_H2BC21_proj_cpdino.zarr",
    },
    {
        "tag": "2026_03_26_A549_CAAX_H2B_DENV_ZIKV",
        "store": f"{OUT_ROOT}/2026_03_26_A549_CAAX_H2B_DENV_ZIKV_proj_cpdino.zarr",
    },
    {
        "tag": "2026_06_24_A549_H2BC21_HB_20x",
        "store": f"{OUT_ROOT}/2026_06_24_A549_H2BC21_FOV_selection_HB_20x_proj_cpdino.zarr",
    },
    {
        "tag": "2026_06_24_A549_H2BC21",
        "store": f"{OUT_ROOT}/2026_06_24_A549_H2BC21_FOV_selection_proj_cpdino.zarr",
    },
    {
        "tag": "2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV",
        "store": f"{OUT_ROOT}/2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV_proj_cpdino.zarr",
    },
]

OUTPUT_ROOT = Path(f"{OUT_ROOT}/fov_features")

MASK_HINT = "cpdino"  # channels containing this are instance-label masks

# segmentation_source: projection channel name keyword -> source. Masks segmented
# from a "*_prediction_*" channel are virtual staining ("vs").
SOURCE_BY_KEYWORD = {"prediction": "vs"}
SOURCE_DEFAULT = "fluor"
SOURCE_OVERRIDE: dict[tuple[str, str], str] = {
    # ("2026_06_24_A549_H2BC21", "nuclei"): "fluor",
}

RESUME = True  # skip a dataset whose per-object CSV already exists

# skimage regionprops properties (new-style names).
PROPS = (
    "label",
    "centroid",
    "area",
    "equivalent_diameter_area",
    "eccentricity",
    "solidity",
    "extent",
    "bbox",
    "intensity_mean",
    "intensity_max",
)
PER_OBJECT_COLUMNS = [
    "dataset",
    "well_row",
    "well_col",
    "fov",
    "timepoint",
    "channel",
    "segmentation_source",
    "projection_type",
    "label_id",
    "centroid_x_norm",
    "centroid_y_norm",
    "area_px",
    "area_um2",
    "equivalent_diameter_um",
    "eccentricity",
    "solidity",
    "extent",
    "intensity_mean",
    "intensity_max",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "dist_to_edge_norm",
    "nearest_neighbor_dist_um",
    "image_width_px",
    "image_height_px",
    "pixel_size_um",
]
# =============================================================================


def channel_label(mask_channel: str) -> str:
    """'nuclei_cellpose' -> 'nuclei', 'membrane_cellpose' -> 'membrane'."""
    return mask_channel.split(f"_{MASK_HINT}")[0]


def organelle_label(channel: str) -> str:
    """Organelle the channel labels: 'nuclei_GFP_maxproj' -> 'nuclei',
    'membrane_prediction_sumproj' -> 'membrane' (first underscore-delimited token)."""
    return (channel or "").split("_")[0]


def projection_type(channel: str) -> str:
    """'sum' / 'max' from a source channel name (e.g. '*_sumproj' / '*_maxproj')."""
    n = (channel or "").lower()
    if "maxproj" in n or n.endswith("_max"):
        return "max"
    if "sumproj" in n or n.endswith("_sum"):
        return "sum"
    return "unknown"


def infer_source(dataset_tag: str, label: str, proj_channels: list[str]) -> str:
    if (dataset_tag, label) in SOURCE_OVERRIDE:
        return SOURCE_OVERRIDE[(dataset_tag, label)]
    proj = next((c for c in proj_channels if c.startswith(label)), "")
    for kw, src in SOURCE_BY_KEYWORD.items():
        if kw in proj:
            return src
    return SOURCE_DEFAULT


def object_feature_rows(
    lbl,
    intensity,
    px_um,
    *,
    dataset_tag,
    well_row,
    well_col,
    fov,
    timepoint,
    channel,
    source,
    projection_type,
):
    """Per-object feature rows from a single (label mask, intensity) pair.

    Array-based core shared by the batch pipeline (``rows_for_timepoint``) and
    the online FOV-selection decision, so both compute identical features.

    Parameters
    ----------
    lbl : np.ndarray
        2D instance-label mask (Y, X), integer ids (0 = background).
    intensity : np.ndarray | None
        2D intensity image (Y, X) the mask was segmented from, or None.
    px_um : float
        XY pixel size in microns (isotropic).
    channel : str
        Organelle/channel label, e.g. ``'nuclei'`` / ``'membrane'``.
    """
    out: list[dict] = []
    lbl = np.asarray(lbl).astype(np.uint32)
    if lbl.max() == 0:
        return out
    Y, X = lbl.shape
    intensity = np.asarray(intensity, np.float32) if intensity is not None else None
    p = regionprops_table(lbl, intensity_image=intensity, properties=PROPS)
    cy, cx = p["centroid-0"], p["centroid-1"]
    # nearest-neighbor distance among same-channel centroids (px -> um)
    if len(cy) >= 2:
        d, _ = cKDTree(np.column_stack([cy, cx])).query(np.column_stack([cy, cx]), k=2)
        nn_px = d[:, 1]
    else:
        nn_px = np.full(len(cy), np.nan)
    for k in range(len(p["label"])):
        cyk, cxk = float(cy[k]), float(cx[k])
        out.append(
            {
                "dataset": dataset_tag,
                "well_row": well_row,
                "well_col": well_col,
                "fov": fov,
                "timepoint": timepoint,
                "channel": channel,
                "organelle": organelle_label(channel),
                "segmentation_source": source,
                "projection_type": projection_type,
                "label_id": int(p["label"][k]),
                "centroid_x_norm": cxk / X,
                "centroid_y_norm": cyk / Y,
                "area_px": int(p["area"][k]),
                "area_um2": float(p["area"][k]) * px_um * px_um,
                "equivalent_diameter_um": float(p["equivalent_diameter_area"][k]) * px_um,
                "eccentricity": float(p["eccentricity"][k]),
                "solidity": float(p["solidity"][k]),
                "extent": float(p["extent"][k]),
                "intensity_mean": float(p["intensity_mean"][k]),
                "intensity_max": float(p["intensity_max"][k]),
                # Raw mask bounding box in pixels (skimage convention: min inclusive,
                # max exclusive). Kept raw so any edge margin can be applied later
                # from the FOV matrix WITHOUT re-extracting -- edge_frac_<k> is derived
                # in build_fov_feature_matrix.py by comparing these to a margin.
                "bbox_min_row": int(p["bbox-0"][k]),
                "bbox_min_col": int(p["bbox-1"][k]),
                "bbox_max_row": int(p["bbox-2"][k]),
                "bbox_max_col": int(p["bbox-3"][k]),
                "dist_to_edge_norm": min(cxk / X, cyk / Y, (X - cxk) / X, (Y - cyk) / Y),
                "nearest_neighbor_dist_um": float(nn_px[k]) * px_um,
                "image_width_px": X,
                "image_height_px": Y,
                "pixel_size_um": px_um,
            }
        )
    return out


def rows_for_timepoint(dataset_tag, name, pos, t, mask_channels, proj_channels):
    """Per-object row dicts for ONE (position, timepoint), both channels.

    Thin store-bound wrapper over ``object_feature_rows``: pulls the label mask
    + intensity image for each channel out of the zarr Position node and
    delegates the per-object feature computation.
    """
    row, col, fov = name.split("/")
    arr = pos["0"]  # (T, C, 1, Y, X)
    chans = list(pos.channel_names)
    px_um = float(list(pos.scale)[-1])  # X pixel size (um); isotropic XY here
    out = []
    for c in mask_channels:
        label = channel_label(c)
        source = infer_source(dataset_tag, label, proj_channels)
        lbl = np.asarray(arr[t, chans.index(c), 0]).astype(np.uint32)
        # intensity image = the prediction channel this mask was segmented from
        proj = next((p for p in proj_channels if p.startswith(label)), None)
        intensity = np.asarray(arr[t, chans.index(proj), 0], np.float32) if proj else None
        out.extend(
            object_feature_rows(
                lbl,
                intensity,
                px_um,
                dataset_tag=dataset_tag,
                well_row=row,
                well_col=col,
                fov=fov,
                timepoint=t,
                channel=label,
                source=source,
                projection_type=projection_type(proj or c),
            )
        )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default=None,
        help="Extract features for a single dataset by tag (default: all in DATASETS). "
        "Used to fan out one job per dataset (see submit_features.sh).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print configured dataset tags (one per line) and exit.",
    )
    cli = ap.parse_args()

    if cli.list:
        for ds in DATASETS:
            print(ds["tag"])
        return

    datasets = DATASETS
    if cli.dataset is not None:
        datasets = [d for d in DATASETS if d["tag"] == cli.dataset]
        if not datasets:
            raise SystemExit(
                f"unknown dataset tag {cli.dataset!r}; known: {[d['tag'] for d in DATASETS]}"
            )

    for ds in datasets:
        tag, store = ds["tag"], ds["store"]
        plate = open_ome_zarr(store, mode="r")
        all_ch = list(plate.channel_names)
        mask_channels = [c for c in all_ch if MASK_HINT in c]
        proj_channels = [c for c in all_ch if MASK_HINT not in c]
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        obj_csv = OUTPUT_ROOT / f"{tag}_object_features.csv"
        if RESUME and obj_csv.exists():
            print(
                f"\n[{tag}] {obj_csv.name} exists -- skipping (RESUME). Delete it to rebuild."
            )
            plate.close()
            continue

        positions = list(plate.positions())
        n_t = positions[0][1]["0"].shape[0]
        total = len(positions) * n_t
        print(
            f"\n[{tag}] {len(positions)} positions x {n_t} t = {total} FOV-timepoints | masks={mask_channels}"
        )
        all_rows = []
        n = 0
        for name, pos in positions:
            for t in range(pos["0"].shape[0]):
                rows = rows_for_timepoint(tag, name, pos, t, mask_channels, proj_channels)
                all_rows.extend(rows)
                n += 1
                if n % 50 == 0 or n == total:
                    print(f"  [{n}/{total}] {name} t{t}: {len(rows)} objects")

        # One CSV per dataset: every FOV/timepoint/object concatenated.
        pd.DataFrame(all_rows, columns=PER_OBJECT_COLUMNS).to_csv(obj_csv, index=False)
        print(f"[{tag}] per-object features -> {obj_csv}  ({len(all_rows)} rows)")
        plate.close()

    print(f"\nDone. Features under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
