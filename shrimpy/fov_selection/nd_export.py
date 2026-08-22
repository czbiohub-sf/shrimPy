"""Export a FOV feature matrix (``fov_summary.csv``) to an AnnData zarr for Embedding Atlas.

Embedding Atlas (and ndea) render an embedding scatter over the per-FOV feature table: they
need ``X`` (the feature matrix), ``obs`` (identity / metadata), and an ``obsm`` embedding. This
converts a shrimPy ``fov_summary.csv`` into an AnnData zarr v3 store with exactly that.

Used two ways:

- During acquisition, when ``fov_selection.save_pre_scan_nd`` is set: the manager calls
  :func:`write_feature_anndata` post-drain on the pre-scan's ``fov_summary.csv``, writing
  ``fov_summary.zarr`` beside it (best-effort).
- Offline as a CLI: ``python -m shrimpy.fov_selection.nd_export <matrix.csv> [-o out.zarr]``.

Layout written:
    ``X``            (n_fov, n_feature)  raw feature values, NaN preserved
    ``var_names``                        feature column names
    ``obs``                              identity / metadata columns present in the CSV, a copy
                                         of every feature (so it is queryable in one SELECT),
                                         and -- when the source columns exist -- the ndea crop
                                         keys ``fov_name`` / ``t`` / ``x`` / ``y``
    ``obsm["X_pca"]``                    median-imputed + standardized PCA (the scatter needs an
                                         embedding; nothing plots without one)

Feature columns are DISCOVERED, not hardcoded: any numeric column that is not a known identity /
metadata / output column (:data:`NON_FEATURE_COLUMNS`) is treated as a feature, so new or
renamed features flow through without editing this file.
"""

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are identity, grouping, model output, or acquisition metadata -- never
# features. Everything else numeric is a feature. This is a superset across the live
# fov_summary.csv (name / filename / proba / selected / position / rank / well_row / well_col)
# and older offline matrices (fov / timepoint / goodness / image_* / pixel_size_um).
NON_FEATURE_COLUMNS = frozenset(
    {
        # identity / grouping
        "name", "filename", "fov_name", "well", "well_row", "well_col", "fov", "timepoint", "t",
        # model outputs / decisions / labels
        "proba", "good", "score", "selected", "position", "rank",
        "goodness", "goodness_label", "goodness_probability",
        # acquisition metadata
        "image_width_px", "image_height_px", "pixel_size_um",
        # ndea spatial crop keys
        "x", "y",
    }
)

GOODNESS_LABELS = {-1.0: "bad", 0.0: "neutral", 1.0: "good"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric feature columns: not identity/metadata/output, and not entirely NaN.

    An all-NaN column is dropped (it carries no signal and would break the PCA imputation),
    with a warning naming it.
    """
    out: list[str] = []
    for col in frame.columns:
        if col in NON_FEATURE_COLUMNS or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if frame[col].isna().all():
            logger.warning("FOV ND export: feature %r is entirely NaN; dropping it", col)
            continue
        out.append(col)
    return out


def _pca(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Median-impute, standardize, then PCA. Imputation is for the embedding only -- ``X`` keeps
    its NaNs so the feature histograms show real coverage."""
    from sklearn.decomposition import PCA

    filled = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    std = filled.std(axis=0)
    std[std == 0] = 1.0
    scaled = (filled - filled.mean(axis=0)) / std
    n = min(n_components, *scaled.shape)
    return PCA(n_components=n).fit_transform(scaled).astype(np.float32)


def build_anndata(frame: pd.DataFrame, field_width: int = 6):
    """Build the AnnData object from a feature-matrix DataFrame (see the module docstring)."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "save_pre_scan_nd / the ND export needs anndata; install the FOV group "
            "(`uv sync --group fov`)."
        ) from exc

    names = feature_columns(frame)
    if not names:
        raise ValueError("no usable (numeric, non-metadata, non-empty) feature columns found")

    obs = frame[[c for c in frame.columns if c in NON_FEATURE_COLUMNS]].copy()

    # ndea crop addressing, added only when the source columns exist. The pre-scan
    # fov_summary.csv has no `fov` / `timepoint` / image-size columns, so these are simply
    # skipped and Embedding Atlas shows the embedding + features without per-FOV crops.
    if {"well_row", "well_col", "fov"} <= set(frame.columns):
        obs["fov_name"] = [
            f"{r}/{c}/{int(f):0{field_width}d}"
            for r, c, f in zip(frame["well_row"], frame["well_col"], frame["fov"], strict=True)
        ]
    if "timepoint" in frame.columns:
        obs["t"] = frame["timepoint"].astype("int32")
    if {"image_width_px", "image_height_px"} <= set(frame.columns):
        # Whole-FOV "centroid" = the field centre, so a crop with a large half-window renders
        # the entire FOV as a thumbnail.
        obs["x"] = (frame["image_width_px"] / 2.0).astype("float32")
        obs["y"] = (frame["image_height_px"] / 2.0).astype("float32")

    if "goodness" in obs:
        obs["goodness_label"] = (
            obs["goodness"].map(GOODNESS_LABELS).fillna("unlabeled").astype("category")
        )
    if {"well_row", "well_col"} <= set(frame.columns):
        obs["well"] = (
            frame["well_row"].astype(str) + frame["well_col"].astype(str)
        ).astype("category")

    # Row index: the CSV join key (the PNG stem), falling back to `name`.
    for key in ("filename", "name"):
        if key in frame.columns:
            obs.index = pd.Index(frame[key].astype(str), name=None)
            break

    # Features live in both X (what PCA runs on and what var expects) and obs (queryable in one
    # SELECT). Use .to_numpy() so the obs copy is positional, not index-aligned -- obs.index has
    # been relabelled while `frame` still has a RangeIndex, so a Series assign would fill NaN.
    for name in names:
        obs[name] = frame[name].to_numpy(dtype=np.float32)

    X = frame[names].to_numpy(dtype=np.float32)  # NaN preserved
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(names)))
    adata.obsm["X_pca"] = _pca(X)
    return adata


def write_feature_anndata(
    csv_path, out_path=None, *, field_width: int = 6, plate=None
) -> Path:
    """Read a feature-matrix CSV and write it as an AnnData zarr v3 store; return the path.

    ``out_path`` defaults to ``<csv stem>.zarr`` beside the CSV. ``plate`` is an optional
    OME-Zarr HCS store to read the ``fov_name`` field-path width from (offline use).
    """
    import zarr

    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)

    width = field_width
    if plate is not None:
        detected = plate_field_width(Path(plate))
        if detected is not None:
            width = detected

    adata = build_anndata(frame, width)
    out = Path(out_path) if out_path else csv_path.with_suffix(".zarr")
    # zarr 3 already defaults to format 3; set it explicitly so the store shape does not
    # silently depend on the caller's zarr config.
    with zarr.config.set({"default_zarr_format": 3}):
        adata.write_zarr(out)
    _assert_readable_index(out)
    return out


def plate_field_width(plate: Path) -> int | None:
    """Read one well's field paths from an OME-Zarr HCS store to learn the path width."""
    for name in ("zarr.json", ".zattrs"):
        root = plate / name
        if not root.exists():
            continue
        attrs = json.loads(root.read_text())
        attrs = attrs.get("attributes", attrs)
        spec = attrs.get("plate") or attrs.get("ome", {}).get("plate")
        if not spec or not spec.get("wells"):
            return None
        well_dir = plate / spec["wells"][0]["path"]
        for wname in ("zarr.json", ".zattrs"):
            wf = well_dir / wname
            if not wf.exists():
                continue
            wattrs = json.loads(wf.read_text())
            wattrs = wattrs.get("attributes", wattrs)
            well = wattrs.get("well") or wattrs.get("ome", {}).get("well", {})
            images = well.get("images") or []
            if images:
                return len(images[0]["path"])
    return None


def _assert_readable_index(out: Path) -> None:
    """Fail loudly if the obs index landed as a nullable-string GROUP rather than a plain array.

    pandas >= 3 hands string columns over as ``StringArray``, which anndata encodes as
    ``nullable-string-array``; Embedding Atlas / ndea size the frame with a plain-array open and
    die with "Not found: v2 array". This check is what tells you if that regresses (pin
    ``pandas<3`` if it fires).
    """
    meta = json.loads((out / "obs" / "zarr.json").read_text())
    index_name = meta["attributes"]["_index"]
    node = json.loads((out / "obs" / index_name / "zarr.json").read_text())
    if node.get("node_type") != "array":
        raise ValueError(
            f"{out}: obs index '{index_name}' was written as a "
            f"{node.get('attributes', {}).get('encoding-type')} group, which Embedding Atlas "
            "cannot ingest. Pin pandas<3."
        )


def main() -> None:
    """CLI: convert a feature-matrix CSV to an AnnData zarr for Embedding Atlas."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output .zarr (default: <csv stem>.zarr)")
    ap.add_argument("--plate", type=Path, help="OME-Zarr HCS store to read the field width from")
    ap.add_argument("--field-width", type=int, default=6, help="fov_name zero-padding (default 6)")
    args = ap.parse_args()

    out = write_feature_anndata(
        args.csv, args.out, field_width=args.field_width, plate=args.plate
    )
    import anndata as ad

    adata = ad.read_zarr(out)
    print(f"wrote {out}")
    print(f"  {adata.n_obs} obs x {adata.n_vars} vars, obsm: {list(adata.obsm)}")
    print(f"  features: {list(adata.var_names)}")


if __name__ == "__main__":
    main()
