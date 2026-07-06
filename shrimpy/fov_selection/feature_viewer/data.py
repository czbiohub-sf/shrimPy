"""
Qt-free core for the FOV feature viewer: load feature CSVs, wire each FOV row to its
composite image under fov_composites/, filter, and run 3D dimensionality reduction.
Kept separate from the GUI so it can be tested headless.

Wiring: the feature matrix is now ONE row per FOV (dataset, well_row, well_col, fov,
timepoint), so each row maps to a single composite PNG. Composite filenames vary by
historical naming scheme across datasets, so instead of a hardcoded path we index each
subfolder of COMPOSITES_ROOT and match rows by FOV identity (well_col, fov, timepoint),
with a filename-stem fallback for OpenCell tiles. Each dataset is auto-assigned the
subfolder that covers the most of its rows.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Where the per-FOV composite images live (one subfolder per dataset).
COMPOSITES_ROOT = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/"
    "fov_selection_output/fov_features/fov_composites"
)
_IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

# Reduced-embedding columns (PCA1, TSNE2, UMAP3, ...) are outputs, not input features.
REDUCED_RE = re.compile(r"^(PCA|TSNE|UMAP)\d+$")

# Columns that are identifiers/metadata, never used as reduction features. Per-variant
# feature columns (e.g. nuclei_vs_max__object_count) are NOT here -- they ARE the
# features. Any leftover "*__png" pointer columns are dropped via feature_columns'
# numeric test and the app's column filters.
META_BLACKLIST = {
    "dataset", "well_row", "well_col", "fov", "timepoint",
    "filename", "goodness", "score",
    "image_width_px", "image_height_px", "pixel_size_um",
    "__dataset", "__png", "__src",
}

try:
    import umap  # noqa: F401
    HAS_UMAP = True
except Exception:  # noqa: BLE001
    HAS_UMAP = False

METHODS = ["PCA", "t-SNE"] + (["UMAP"] if HAS_UMAP else [])


def _file_identity_keys(stem: str) -> list:
    """Best-effort FOV-identity keys parsed from a composite filename stem, spanning
    the several historical naming schemes:

        B-2-000000_t00   -> ("stem", ...), ("wct", "2", 0, 0)
        0-1-000000       -> ("stem", ...), ("wct", "1", 0, 0)
        DENV_0_t0_N44    -> ("stem", ...), ("wct", "DENV", 0, 0)
        3_000000_t0_N52  -> ("stem", ...), ("wct", "3", 0, 0)
        <opencell name>  -> ("stem", ...)   (matched by filename)

    "wct" key = (well_col, fov, timepoint); well_row is dropped because it is constant
    within a dataset and absent from most schemes. A trailing _N<count> annotation is
    ignored, and timepoint defaults to 0 when absent.
    """
    keys = [("stem", stem)]
    m_t = re.search(r"_t(\d+)", stem)
    t = int(m_t.group(1)) if m_t else 0
    core = stem[: m_t.start()] if m_t else stem
    core = re.sub(r"_N\d+$", "", core)      # drop trailing object-count annotation
    parts = [p for p in re.split(r"[-_]", core) if p]
    int_idx = [i for i, p in enumerate(parts) if p.isdigit()]
    if int_idx:
        fi = int_idx[-1]
        well_col = parts[fi - 1] if fi >= 1 else ""
        keys.append(("wct", str(well_col), int(parts[fi]), t))
    return keys


def index_composites(folder: Path) -> dict:
    """Map identity keys -> absolute composite path for one dataset subfolder."""
    idx: dict = {}
    if not folder.is_dir():
        return idx
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in _IMG_EXT:
            for k in _file_identity_keys(p.stem):
                idx.setdefault(k, str(p))    # sorted() -> stable first-match
    return idx


def _row_identity_keys(row: pd.Series) -> list:
    """Identity keys for a FOV row, in match-priority order (stem, then well/fov/t)."""
    keys = []
    fn = row.get("filename")
    if isinstance(fn, str) and fn:
        keys.append(("stem", fn))
    fov = row.get("fov")
    if fov is not None and not (isinstance(fov, float) and np.isnan(fov)):
        t = row.get("timepoint", 0)
        t = 0 if (t is None or (isinstance(t, float) and np.isnan(t))) else int(t)
        keys.append(("wct", str(row.get("well_col", "")), int(fov), t))
    return keys


def wire_composites(df: pd.DataFrame, composites_root: Path, cache: dict) -> list[str]:
    """Resolve each row's composite image. Every subfolder of `composites_root` is
    indexed once (memoized in `cache`), then this dataset is assigned the subfolder
    covering the most of its rows; each row picks its first matching key."""
    root = Path(composites_root)
    if not root.is_dir():
        return [""] * len(df)
    folders = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            if sub not in cache:
                cache[sub] = index_composites(sub)
            folders[sub] = cache[sub]
    if not folders:
        return [""] * len(df)
    row_keys = [_row_identity_keys(df.iloc[i]) for i in range(len(df))]
    best = max(folders.values(),
               key=lambda idx: sum(any(k in idx for k in ks) for ks in row_keys))
    return [next((best[k] for k in ks if k in best), "") for ks in row_keys]


def load_matrices(csv_paths: list[str | Path],
                  composites_root: Path = COMPOSITES_ROOT) -> pd.DataFrame:
    """Concatenate feature CSVs; add __dataset (tag), __src, and __png (composite path).

    Each *_fov_feature_matrix.csv is now one row per FOV, so each row maps to a single
    composite image. __png is resolved from `composites_root` by FOV identity (see
    wire_composites); rows with no matching composite get "" and the viewer shows a
    placeholder.
    """
    frames = []
    cache: dict = {}
    for p in csv_paths:
        p = Path(p)
        df = pd.read_csv(p)
        tag = p.name.replace("_fov_feature_matrix.csv", "").replace(".csv", "")
        df["__dataset"] = tag
        df["__src"] = str(p)
        df["__png"] = wire_composites(df, composites_root, cache)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, non-metadata columns usable as reduction features / plot axes."""
    cols = []
    for c in df.columns:
        if c in META_BLACKLIST or REDUCED_RE.match(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


@dataclass
class Filter:
    column: str
    kind: str            # "range" (numeric) or "isin" (categorical)
    lo: float = 0.0
    hi: float = 0.0
    values: tuple = ()

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "range":
            v = df[self.column].to_numpy(float)
            return (v >= self.lo) & (v <= self.hi)
        return df[self.column].astype(str).isin([str(x) for x in self.values]).to_numpy()

    def label(self) -> str:
        if self.kind == "range":
            return f"{self.column} in [{self.lo:g}, {self.hi:g}]"
        vals = ", ".join(map(str, self.values))
        return f"{self.column} = {{{vals}}}"


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> np.ndarray:
    """Positional indices (into df) passing ALL filters."""
    mask = np.ones(len(df), bool)
    for f in filters:
        mask &= f.mask(df)
    return np.where(mask)[0]


def run_reduction(X: np.ndarray, method: str, *, perplexity: float = 30.0,
                  n_neighbors: int = 15, seed: int = 0) -> np.ndarray:
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
        return TSNE(n_components=3, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=seed).fit_transform(X)
    if method == "UMAP":
        import umap
        return umap.UMAP(n_components=3, n_neighbors=min(n_neighbors, max(2, n - 1)),
                         random_state=seed).fit_transform(X)
    raise ValueError(f"unknown method {method!r}")
