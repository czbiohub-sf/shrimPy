"""
Qt-free core for the FOV feature viewer: load feature CSVs, wire each FOV row to its
per-channel PNG(s), filter, and run 3D dimensionality reduction. Kept separate from the
GUI so it can be tested headless.

Standard layout (one CSV per dataset, sibling PNG folders next to it):

    fov_summary.csv    # one row per FOV; MUST have a `filename` column
    prescan_fov/       # brightfield image per FOV, named <filename>.png
    prescan_mask/      # optional mask channel
    prescan_fluor/     # optional fluorescence channel

The live pre-scan writes ``fov_summary.csv`` in both normal and calibration mode; any CSV
with a ``filename`` column loads (older datasets used ``<name>_fov_feature_matrix.csv``).

The image folders are the fixed ``prescan_*`` names written by both the live save_decision
and calibration pre-scans (see prescan_artifacts.py). Legacy datasets that used the older
stem-prefixed folders (``<stem>_png`` / ``<stem>_<channel>_png``) still open: each channel
falls back to that name when its fixed folder is absent.

Wiring is a strict 1:1 join: the CSV `filename` column equals the PNG stem.
"""

from __future__ import annotations

import re

from pathlib import Path

import numpy as np
import pandas as pd

_IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

# Reduced-embedding columns (PCA1, TSNE2, UMAP3, ...) are outputs, not input features.
REDUCED_RE = re.compile(r"^(PCA|TSNE|UMAP)\d+$")

# NUMERIC columns that are identifiers, labels, or model/acquisition outputs, never
# reduction features. Only numeric non-features need listing here: feature_columns already
# drops string columns (filename, well_row, dataset) and internal "__"-prefixed columns
# (__dataset, __png, __src, __png_<channel>). The feature columns themselves are plain
# single-mask keys (e.g. coverage_frac, nn_um_mean) and are NOT listed. These columns
# stay in the frame as filterable metadata, just kept off the plot axes.
META_BLACKLIST = {
    "well_col",  # FOV identity (integer-valued, but an ID)
    "fov",
    "timepoint",
    "goodness",  # ground-truth label (from CSV; also editable in the Label tab)
    "goodness_probability",  # classifier output P(good); a label/output, not a reduction input
    "score",  # ranking-model output (produced in the Rank tab, not read from CSV)
    # Decision outputs written by the normal-mode fov_summary.csv (prescan_artifacts.py): the
    # model score and the whole-run selection, kept as filterable metadata but never a
    # reduction/plot axis, so that CSV loads in the viewer just like the calibration matrix.
    "proba",
    "selected",
    "rank",
    "image_width_px",  # acquisition metadata
    "image_height_px",
    "pixel_size_um",
}

try:
    import umap  # noqa: F401

    HAS_UMAP = True
except Exception:  # noqa: BLE001
    HAS_UMAP = False

METHODS = ["PCA", "t-SNE"] + (["UMAP"] if HAS_UMAP else [])


# ============================================================= image wiring
CHANNELS = ("brightfield", "mask", "fluor")  # FOV thumbnail channels (viewer toggle)

# Fixed image-folder name per channel, written by the pre-scan (prescan_artifacts.py). The
# brightfield slot holds the projection; `selected_fov` is deliberately absent -- the viewer
# shows the full candidate set, never the chosen subset.
_CHANNEL_DIRNAME = {
    "brightfield": "prescan_fov",
    "mask": "prescan_mask",
    "fluor": "prescan_fluor",
}


def _index_by_stem(folder: str | Path) -> dict[str, str]:
    """Map filename stem -> absolute image path for one PNG folder (sorted -> stable)."""
    folder = Path(folder)
    stem_to_path: dict[str, str] = {}
    if not folder.is_dir():
        return stem_to_path
    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() in _IMG_EXT:
            stem_to_path.setdefault(image_path.stem, str(image_path))
    return stem_to_path


def wire_folder(df: pd.DataFrame, png_folder: str | Path) -> list[str]:
    """Resolve each row's image from one folder by the `filename` column (== PNG stem).

    Rows with no `filename` value or no matching file get "" (the viewer shows a
    placeholder)."""
    stem_to_path = _index_by_stem(png_folder)
    if not stem_to_path or "filename" not in df.columns:
        return [""] * len(df)
    return [stem_to_path.get(str(filename), "") for filename in df["filename"]]


def _channel_png_folder(csv_path, channel):
    """Sibling PNG folder for a channel, next to the CSV.

    Prefers the fixed ``prescan_*`` folder the live pre-scan writes (``prescan_fov`` /
    ``prescan_mask`` / ``prescan_fluor``); falls back to the legacy stem-prefixed name
    (``<stem>_png`` / ``<stem>_<channel>_png``) when that fixed folder is absent, so existing
    offline datasets still open."""
    csv_path = Path(csv_path)
    fixed = csv_path.with_name(_CHANNEL_DIRNAME[channel])
    if fixed.is_dir():
        return fixed
    suffix = "_png" if channel == "brightfield" else f"_{channel}_png"
    return csv_path.with_name(csv_path.stem + suffix)


def wire_channels(df: pd.DataFrame, csv_path, brightfield_folder=None) -> list[str]:
    """Set ``__png_<channel>`` for each channel whose sibling folder exists, and ``__png`` to
    the default (the mask-overlay channel if present, else brightfield, else the first present).
    Returns the channels found. An explicit ``brightfield_folder`` (CLI --png-folder) overrides
    the brightfield sibling."""
    found_channels = []
    for channel in CHANNELS:
        if channel == "brightfield" and brightfield_folder:
            folder = brightfield_folder
        else:
            folder = _channel_png_folder(csv_path, channel)
        if folder and Path(folder).is_dir():
            df[f"__png_{channel}"] = wire_folder(df, folder)
            found_channels.append(channel)
    if found_channels:
        # Prefer the segmentation-overlay ("mask") channel as the default view so the mask is
        # visible on load; fall back to brightfield, then whatever is present.
        if "mask" in found_channels:
            default_channel = "mask"
        elif "brightfield" in found_channels:
            default_channel = "brightfield"
        else:
            default_channel = found_channels[0]
        df["__png"] = df[f"__png_{default_channel}"]
    return found_channels


# ============================================================= loaders
def _read_matrix(csv_path) -> pd.DataFrame:
    """Read one feature CSV, stamping the internal __dataset (tag) and __src columns."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    dataset_tag = csv_path.name.replace("_fov_feature_matrix.csv", "").replace(".csv", "")
    df["__dataset"] = dataset_tag
    df["__src"] = str(csv_path)
    return df


def _concat_datasets(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset frames and normalize the image-path columns.

    Datasets can differ in which channels exist (e.g. one has no fluor folder), so
    concatenation leaves NaN in a ``__png*`` column for datasets that lack that channel.
    Replace those NaNs with "" so the viewer's ``if png and Path(png).exists()`` guard
    reads them as 'no image' -- a float NaN would otherwise pass the truthiness check and
    crash ``Path()``."""
    df = pd.concat(dataframes, ignore_index=True, sort=False)
    for col in [c for c in df.columns if c == "__png" or c.startswith("__png_")]:
        df[col] = df[col].fillna("")
    return df


def load_matrices(csv_paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate feature CSVs, wiring each row to its per-channel sibling PNG folders.

    Each CSV must carry a `filename` column; images live in sibling folders next to the
    CSV (see the module docstring / wire_channels). Rows with no matching image get "".
    """
    dataframes = []
    for csv_path in csv_paths:
        df = _read_matrix(csv_path)
        if not wire_channels(df, csv_path):
            df["__png"] = [""] * len(df)
        dataframes.append(df)
    return _concat_datasets(dataframes)


def load_paired(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Load (csv, png_folder) pairs; each row's brightfield __png comes from the given
    folder, while mask/fluor are still resolved from sibling folders if present."""
    dataframes = []
    for csv_path, brightfield_folder in pairs:
        df = _read_matrix(csv_path)
        if not wire_channels(df, csv_path, brightfield_folder=brightfield_folder):
            df["__png"] = [""] * len(df)
        dataframes.append(df)
    return _concat_datasets(dataframes)


# ============================================================= features / filters
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, non-metadata columns usable as reduction features / plot axes.

    A candidate feature is any numeric column that is not an identifier/label/output
    (META_BLACKLIST) and not a reduced-embedding output (PCA1, TSNE2, ...). With a single
    dataset loaded, every candidate is returned -- whatever features that dataset computed.
    With several datasets loaded together, only features COMPUTED FOR ALL of them are kept:
    concatenating datasets with different feature sets leaves a column all-NaN for any
    dataset that never computed it, so a column all-NaN within any one dataset is treated
    as absent there and dropped from the shared set.
    """
    candidates = [
        col
        for col in df.columns
        if not col.startswith("__")  # internal columns (__dataset, __png, ...)
        and col not in META_BLACKLIST
        and not REDUCED_RE.match(col)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if "__dataset" not in df.columns or df["__dataset"].nunique() <= 1:
        return candidates
    computed_in = {
        dataset: group[candidates].notna().any() for dataset, group in df.groupby("__dataset")
    }
    return [col for col in candidates if all(has[col] for has in computed_in.values())]


# ============================================================= dimensionality reduction
def run_reduction(
    X: np.ndarray,
    method: str,
    *,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """Standardize + impute features, return an (n, 3) embedding. Always 3 comps."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    n = X.shape[0]
    if method == "PCA":
        from sklearn.decomposition import PCA

        return PCA(n_components=3, random_state=seed).fit_transform(X)
    if method == "t-SNE":
        from sklearn.manifold import TSNE

        perp = min(perplexity, max(5, (n - 1) // 3))
        return TSNE(
            n_components=3,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
    if method == "UMAP":
        import umap

        return umap.UMAP(
            n_components=3, n_neighbors=min(n_neighbors, max(2, n - 1)), random_state=seed
        ).fit_transform(X)
    raise ValueError(f"unknown method {method!r}")
