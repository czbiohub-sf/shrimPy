"""
Build reduced FOV-level feature tables from per-object CSVs, for ALL datasets, into
one shared folder (fov_features/). Aggregates to ONE row per FOV -- where a FOV is a
unique physical acquisition, identified by (dataset, well_row, well_col, fov, timepoint)
for zarr and (dataset, filename, ...) for OpenCell.

All feature variants for that same FOV -- different channels/organelles (nuclei vs
membrane), segmentation sources (virtual staining "vs" vs fluorescence "fluor"), and
projection methods (max vs sum) -- are laid out side by side as PREFIXED COLUMNS on
that single row, rather than as separate rows. The prefix is:

    <organelle>_<segmentation_source>_<projection_type>

e.g. for the CAAX/H2B dataset each FOV row carries the 10 features below for each of:
    nuclei_vs_max, nuclei_vs_sum, membrane_vs_max, membrane_vs_sum,
    nuclei_fluor_max, nuclei_fluor_sum
as e.g. `nuclei_vs_max__object_count`, `membrane_vs_sum__coverage_frac`, ... Variants
absent for a given FOV (e.g. an empty mask) are left NaN.

Each row is one FOV, so FOV visualization is one composite image per row; the feature
viewer resolves those from fov_composites/ by FOV identity (see feature_viewer/data.py)
rather than a path stored here.

Every matrix also carries a `goodness` label column: good=1, neutral=0, bad=-1 (NaN
when unlabeled). OpenCell FOVs are labeled from manual scoring; all other datasets start
unlabeled (NaN) and can be labeled interactively in the feature viewer.

The 10 per-variant features (same set as the OpenCell analysis matrix,
build_fov_features.py):
  Density            object_count, objects_per_10um2, coverage_frac
  Edge cut-off       edge_frac
  Spatial distrib.   com_offset_norm, nn_um_mean, nn_cv, densest_grid_frac
  Shape              eccentricity_mean, solidity_mean

Two input kinds:
  - zarr datasets   : <tag>_object_features.csv in FEATURES_DIR (HCS positions).
  - OpenCell (tif)  : nuclei_object_features.csv from segment_opencell.py; carries
                      goodness/score labels.

"object" = nucleus or cell depending on the row's `channel`. NN is taken in pixels
(from nearest_neighbor_dist_px if present, else nearest_neighbor_dist_um / pixel size).

    python build_fov_feature_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output/fov_features"
)
OPENCELL_CSV = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output/"
    "opencell/nuclei_segmentation/nuclei_object_features.csv"
)
OPENCELL_OUT = FEATURES_DIR / "2019_opencell_fov_scoring_fov_feature_matrix.csv"
GRID = 8

# Columns that identify ONE FOV (-> one output row). Everything else about a FOV is
# spread across prefixed variant columns.
ZARR_ID_KEYS = ["dataset", "well_row", "well_col", "fov", "timepoint"]
OPENCELL_ID_KEYS = ["dataset", "filename", "goodness", "score"]

# Per-FOV metadata that is constant across variants (carried through verbatim).
FOV_META = ["image_width_px", "image_height_px", "pixel_size_um"]

# Columns whose combination distinguishes one variant from another within a FOV. The
# clean column prefix is built from these (see variant_tags()).
VARIANT_KEYS = ["organelle", "segmentation_source", "projection_type"]

# Border margin for edge_frac, as a fraction of image height/width. The VS prediction
# has a ~0.3-0.6% zero-pad band, so masks of edge-clipped cells stop a few px short of
# the true border; this margin absorbs it. Derived from the raw per-object bbox (see
# group_features), so it can be re-tuned and the matrix rebuilt WITHOUT re-extracting
# object features.
EDGE_MARGIN_FRAC = 0.01

# The per-variant features produced by group_features(), in output order.
FEATURE_NAMES = [
    "object_count",
    "objects_per_10um2",
    "coverage_frac",
    "edge_frac",
    "com_offset_norm",
    "nn_um_mean",
    "nn_cv",
    "densest_grid_frac",
    "eccentricity_mean",
    "solidity_mean",
]


def group_features(g: pd.DataFrame) -> dict:
    n = len(g)
    W = float(g["image_width_px"].iloc[0])
    H = float(g["image_height_px"].iloc[0])
    px = float(g["pixel_size_um"].iloc[0])
    cx = g["centroid_x_norm"].to_numpy(float) * W
    cy = g["centroid_y_norm"].to_numpy(float) * H

    rec = {
        "object_count": n,
        # nuclei per 10 um^2 (physical density)
        "objects_per_10um2": 10.0 * n / (W * H * px * px) if px else np.nan,
        "coverage_frac": float(g["area_px"].sum() / (W * H)),
    }
    # Edge-touching is derived here from the raw per-object bounding box, so the
    # margin can be re-tuned WITHOUT re-extracting object features. A mask counts as
    # touching the border if its bbox comes within EDGE_MARGIN_FRAC of any edge (the
    # margin absorbs the VS zero-pad band).
    if {"bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"} <= set(g.columns):
        b_top = g["bbox_min_row"].to_numpy(float)
        b_left = g["bbox_min_col"].to_numpy(float)
        b_bot = g["bbox_max_row"].to_numpy(float)  # skimage max is exclusive
        b_right = g["bbox_max_col"].to_numpy(float)
        my, mx = EDGE_MARGIN_FRAC * H, EDGE_MARGIN_FRAC * W
        touch = (b_top <= my) | (b_left <= mx) | (b_bot >= H - my) | (b_right >= W - mx)
        rec["edge_frac"] = float(touch.mean())
    else:  # legacy per-object CSV without bbox columns
        src = "touches_edge" if "touches_edge" in g.columns else None
        rec["edge_frac"] = float(g[src].astype(bool).mean()) if src else np.nan
    com = np.array([cx.mean(), cy.mean()])
    rec["com_offset_norm"] = float(np.hypot(*(com - [W / 2, H / 2])) / (0.5 * np.hypot(W, H)))
    # Nearest-neighbor spacing in PHYSICAL units (um) so it is invariant to
    # magnification (pixel size); NN distance is a local density measure, so it is
    # independent of FOV size too. nn_cv (std/mean) is a unitless ratio -- the
    # per-FOV pixel-size factor cancels, so it is invariant either way.
    if "nearest_neighbor_dist_um" in g.columns:
        nn_um = g["nearest_neighbor_dist_um"].to_numpy(float)
    elif "nearest_neighbor_dist_px" in g.columns:
        nn_um = g["nearest_neighbor_dist_px"].to_numpy(float) * px
    else:
        nn_um = np.full(n, np.nan)
    nn_um = nn_um[~np.isnan(nn_um)]
    rec["nn_um_mean"] = float(nn_um.mean()) if nn_um.size else np.nan
    rec["nn_cv"] = float(nn_um.std() / nn_um.mean()) if nn_um.size and nn_um.mean() else np.nan
    gx = np.clip((cx / W * GRID).astype(int), 0, GRID - 1)
    gy = np.clip((cy / H * GRID).astype(int), 0, GRID - 1)
    counts = np.zeros(GRID * GRID, int)
    np.add.at(counts, gy * GRID + gx, 1)
    rec["densest_grid_frac"] = float(counts.max() / n)
    rec["eccentricity_mean"] = float(g["eccentricity"].mean())
    rec["solidity_mean"] = float(g["solidity"].mean())
    return rec


def variant_tags(obj: pd.DataFrame) -> dict[str, str]:
    """Map each `channel` -> clean column prefix `<organelle>_<source>_<projection>`.

    If two channels would collapse to the same prefix (they should not, given the
    naming), fall back to the raw channel name for those so no variant is silently
    merged.
    """
    meta = obj[["channel"] + VARIANT_KEYS].drop_duplicates()
    tag_by_channel, channels_by_tag = {}, {}
    for r in meta.itertuples(index=False):
        prefix = f"{r.organelle}_{r.segmentation_source}_{r.projection_type}"
        tag_by_channel[r.channel] = prefix
        channels_by_tag.setdefault(prefix, set()).add(r.channel)
    for prefix, chans in channels_by_tag.items():
        if len(chans) > 1:
            print(
                f"  ! variant-prefix collision {prefix!r} <- {sorted(chans)}; "
                f"using raw channel names for these"
            )
            for c in chans:
                tag_by_channel[c] = c
    return tag_by_channel


def build(obj: pd.DataFrame, id_keys: list[str], out: Path, tag: str) -> None:
    # organelle (nuclei/membrane) tracks alongside the full channel name; derive if the
    # object CSV predates the column.
    if "organelle" not in obj.columns:
        obj = obj.assign(organelle=obj["channel"].astype(str).str.split("_").str[0])

    tag_by_channel = variant_tags(obj)
    # deterministic, readable variant order (groups by organelle, then source, then proj)
    variant_order = sorted(set(tag_by_channel.values()))

    rows = []
    for fov_key, fov_g in obj.groupby(id_keys, sort=False):
        ids = dict(zip(id_keys, fov_key if isinstance(fov_key, tuple) else (fov_key,)))
        rec = dict(ids)
        for m in FOV_META:
            rec[m] = fov_g[m].iloc[0]
        for ch, var_g in fov_g.groupby("channel", sort=False):
            prefix = tag_by_channel[ch]
            for k, v in group_features(var_g).items():
                rec[f"{prefix}__{k}"] = v
        rows.append(rec)

    df = pd.DataFrame(rows)

    # FOV quality label: good=1, neutral=0, bad=-1 (NaN when unlabeled). OpenCell rows
    # carry a string `goodness` from manual scoring (the -1/0/1 `score` is also kept);
    # zarr datasets are unlabeled for now and can be labeled in the feature viewer.
    if "goodness" in df.columns:
        df["goodness"] = df["goodness"].map({"good": 1.0, "neutral": 0.0, "bad": -1.0})
    else:
        df["goodness"] = np.nan

    ordered = list(id_keys)
    if "goodness" not in ordered:  # place the label right after the FOV identifiers
        ordered.append("goodness")
    ordered += FOV_META
    for prefix in variant_order:
        ordered += [f"{prefix}__{k}" for k in FEATURE_NAMES]
    ordered = [c for c in ordered if c in df.columns]
    df = df.reindex(columns=ordered + [c for c in df.columns if c not in ordered])
    df.to_csv(out, index=False)
    print(
        f"[{tag}] {len(obj)} objects -> {len(df)} FOV rows "
        f"({len(variant_order)} variants: {', '.join(variant_order)}) -> {out.name}"
    )


def zarr_tags() -> list[str]:
    """Dataset tags with a per-object CSV in FEATURES_DIR (one FOV matrix each)."""
    return sorted(
        c.name.replace("_object_features.csv", "")
        for c in FEATURES_DIR.glob("*_object_features.csv")
    )


def build_zarr(tag: str) -> None:
    csv = FEATURES_DIR / f"{tag}_object_features.csv"
    if not csv.exists():
        raise SystemExit(f"no per-object CSV for tag {tag!r}: {csv}")
    build(pd.read_csv(csv), ZARR_ID_KEYS, FEATURES_DIR / f"{tag}_fov_feature_matrix.csv", tag)


def build_opencell() -> None:
    if OPENCELL_CSV.exists():
        oc_tag = OPENCELL_OUT.name.replace("_fov_feature_matrix.csv", "")
        build(pd.read_csv(OPENCELL_CSV), OPENCELL_ID_KEYS, OPENCELL_OUT, oc_tag)
    else:
        print(f"OpenCell CSV not found: {OPENCELL_CSV}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        default=None,
        help="Build the FOV matrix for a single zarr dataset by tag (default: all zarr "
        "datasets in FEATURES_DIR + OpenCell). Use 'opencell' for the OpenCell matrix.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the buildable zarr dataset tags (one per line) and exit.",
    )
    cli = ap.parse_args()

    if cli.list:
        for t in zarr_tags():
            print(t)
        return

    if cli.tag is not None:
        if cli.tag == "opencell":
            build_opencell()
        else:
            build_zarr(cli.tag)
        return

    for tag in zarr_tags():
        build_zarr(tag)
    build_opencell()


if __name__ == "__main__":
    main()
